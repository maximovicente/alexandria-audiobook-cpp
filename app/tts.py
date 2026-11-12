import os
import re
import json
import threading
import shutil
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import subprocess
import urllib.request
import urllib.error
import base64
import hashlib
import atexit

DEFAULT_PAUSE_MS = 500  # Pause between different speakers
SAME_SPEAKER_PAUSE_MS = 250  # Shorter pause for same speaker continuing


def sanitize_filename(name):
    """Make a string safe for use in filenames"""
    name = re.sub(r'[^\w\-]', '_', name)
    return name.lower()


def combine_audio_with_pauses(audio_segments, speakers, pause_ms=DEFAULT_PAUSE_MS,
                              same_speaker_pause_ms=SAME_SPEAKER_PAUSE_MS,
                              pause_overrides=None):
    """Combine audio segments with pauses between them.

    Args:
        pause_overrides: Optional list aligned with audio_segments. Each entry is
            the pause (ms) to insert *after* that segment, or None to use the
            default speaker-change logic. The last entry is ignored.
    """
    if not audio_segments:
        return None

    combined = audio_segments[0]
    prev_speaker = speakers[0]

    for i, (segment, speaker) in enumerate(zip(audio_segments[1:], speakers[1:])):
        override = pause_overrides[i] if pause_overrides else None
        if override is not None:
            gap = AudioSegment.silent(duration=override)
        elif speaker == prev_speaker:
            gap = AudioSegment.silent(duration=same_speaker_pause_ms)
        else:
            gap = AudioSegment.silent(duration=pause_ms)
        combined += gap + segment
        prev_speaker = speaker

    return combined


def compute_timeline(chunks_with_audio, pause_ms=DEFAULT_PAUSE_MS,
                     same_speaker_pause_ms=SAME_SPEAKER_PAUSE_MS):
    """Compute a timeline of (chunk, segment, abs_start_ms) tuples.

    Args:
        chunks_with_audio: list of (chunk_dict, AudioSegment) tuples.
            Each chunk_dict may have an optional 'pause_after' key (int ms)
            that overrides the default pause inserted after that chunk.
        pause_ms: Default pause between different speakers.
        same_speaker_pause_ms: Default pause when same speaker continues.

    Returns:
        list of (chunk_dict, AudioSegment, abs_start_ms) tuples.
    """
    timeline = []
    cursor_ms = 0
    prev_speaker = None
    prev_chunk = None

    for chunk, segment in chunks_with_audio:
        if prev_speaker is not None:
            override = prev_chunk.get("pause_after")
            if override is not None:
                gap = int(override)
            elif chunk["speaker"] == prev_speaker:
                gap = same_speaker_pause_ms
            else:
                gap = pause_ms
            cursor_ms += gap

        timeline.append((chunk, segment, cursor_ms))
        cursor_ms += len(segment)
        prev_speaker = chunk["speaker"]
        prev_chunk = chunk

    return timeline


class TTSEngine:
    """TTS engine supporting local (qwen-tts) and external (Gradio) backends.

    Mode is determined by config["tts"]["mode"]:
      - "local": Loads Qwen3TTSModel directly. No external server needed.
      - "external": Connects via Gradio client to a running TTS server.

    Models and clients are lazily initialized on first use.
    """

    def __init__(self, config):
        tts_config = config.get("tts", {})
        self._mode = tts_config.get("mode", "external")
        self._backend = tts_config.get("backend", "cpp")
        self._cpp_executable = tts_config.get(
            "cpp_executable",
            r"native\qwentts-cpp\build\Release\qwen-tts.exe",
        )
        self._cpp_model_dir = tts_config.get(
            "cpp_model_dir",
            r"native\qwentts-cpp\models",
        )
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._cpp_executable = self._resolve_cpp_path(self._cpp_executable, project_root)
        self._cpp_model_dir = self._resolve_cpp_path(self._cpp_model_dir, project_root)
        self._cpp_auto_download = bool(tts_config.get("cpp_auto_download", True))
        self._cpp_server_port = int(tts_config.get("cpp_server_port", 4211))
        self._python_fallback = bool(tts_config.get("python_fallback", False))
        self._url = tts_config.get("url", "http://127.0.0.1:7860")
        self._device = tts_config.get("device", "auto")
        self._compile_codec_enabled = tts_config.get("compile_codec", False)

        # Language setting (passed to Qwen3-TTS)
        self._language = tts_config.get("language", "English")

        # Sub-batching config
        self._sub_batch_enabled = tts_config.get("sub_batch_enabled", True)
        self._sub_batch_min_size = max(1, tts_config.get("sub_batch_min_size", 4))
        self._sub_batch_ratio = max(1.0, float(tts_config.get("sub_batch_ratio", 5)))
        self._sub_batch_max_items = int(tts_config.get("sub_batch_max_items", 0))  # 0 = auto

        # Lazy-loaded backends (guarded by _model_lock to prevent concurrent loads)
        self._model_lock = threading.Lock()
        self._local_custom_model = None
        self._local_clone_model = None
        self._local_design_model = None
        self._local_lora_model = None
        self._warmup_needed = True  # cleared after first batch warmup
        self._lora_adapter_path = None  # track which adapter is currently loaded
        self._gradio_client = None
        self._cpp_clone_server = None
        self._cpp_clone_voice = None
        self._cpp_server_model = None
        self._cpp_clone_server_lock = threading.Lock()
        self._cpp_error_local = threading.local()
        # Prompt caches for clone and LoRA voice generation.
        self._clone_prompt_cache = {}
        self._lora_prompt_cache = {}
        atexit.register(self.close)

    @staticmethod
    def _resolve_cpp_path(value, project_root):
        if os.path.isabs(value):
            return value
        return os.path.abspath(os.path.join(project_root, value))

    @property
    def mode(self):
        return self._mode

    @staticmethod
    def _concat_audio(wav):
        """Concatenate audio array(s) into a single numpy array."""
        if isinstance(wav, list):
            return np.concatenate(wav) if len(wav) > 1 else wav[0]
        return wav

    @staticmethod
    def _clear_gpu_cache():
        """Free GPU memory: garbage-collect Python objects, then clear CUDA cache."""
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    @staticmethod
    def _reset_compile_cache():
        """Reset torch.compile dynamo state to prevent guard accumulation.

        torch.compile(dynamic=True) accumulates shape guards across calls.
        With varying batch sizes and sequence lengths, the guard list grows
        and CPU-side guard evaluation becomes a bottleneck, causing
        progressive throughput degradation.  Resetting clears all in-memory
        guards; the next call pays a one-time recompilation cost (fast due
        to inductor disk cache) but prevents the slowdown from compounding.

        Only applied on ROCm (AMD GPUs). On NVIDIA, max-autotune mode
        re-benchmarks all kernel variants after each reset, and the
        benchmarking cost scales with tensor size — causing worse slowdown
        than the guard accumulation it prevents.
        """
        import torch
        if not (hasattr(torch.version, "hip") and torch.version.hip):
            return  # skip on NVIDIA/CPU — recompilation cost outweighs benefit
        torch._dynamo.reset()

    def _estimate_max_batch_size(self, model, clone_prompt_tokens=0,
                                ref_text_chars=0, max_text_chars=0,
                                max_new_tokens=2048):
        """Estimate how many sequences fit in free VRAM based on KV cache math.

        Uses the talker's architecture (num_layers, num_kv_heads, head_dim) to
        calculate KV cache bytes per token, then estimates total tokens per
        sequence from clone prompt size + text length + max generation length.

        Returns max batch size (>= 1).  Falls back to a large default on CPU
        or if the model config is inaccessible.
        """
        import torch
        if not torch.cuda.is_available():
            return 9999

        try:
            config = model.model.talker.config
            num_layers = config.num_hidden_layers
            num_kv_heads = config.num_key_value_heads
            head_dim = config.hidden_size // config.num_attention_heads
        except AttributeError:
            return 9999  # can't read config, skip estimation

        dtype_bytes = 2  # bf16
        kv_per_token = num_layers * 2 * num_kv_heads * head_dim * dtype_bytes

        # Total tokens per sequence (worst case: padded to longest + full generation)
        overhead = 10  # role tokens + prefix + special tokens
        ref_text_tokens = ref_text_chars // 3 if ref_text_chars else 0
        text_tokens = max_text_chars // 3 if max_text_chars else 0
        total_tokens = overhead + clone_prompt_tokens + ref_text_tokens + text_tokens + max_new_tokens

        # Overhead factor covers prefill activations, codec, allocator fragmentation
        OVERHEAD_FACTOR = 2.0
        mem_per_seq = total_tokens * kv_per_token * OVERHEAD_FACTOR

        # Available = driver-level free + PyTorch reserved-but-unallocated
        free_driver, _ = torch.cuda.mem_get_info()
        reserved_unused = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        free_total = free_driver + reserved_unused

        budget = int(free_total * 0.8)
        max_batch = max(1, budget // mem_per_seq)

        print(f"VRAM estimate: {free_total / 1e9:.1f}GB free, "
              f"{total_tokens} tok/seq ({clone_prompt_tokens} prompt + "
              f"{ref_text_tokens + text_tokens} text + {max_new_tokens} gen), "
              f"{mem_per_seq / 1e6:.0f}MB/seq -> max_batch={max_batch}")

        return max_batch

    def _build_sub_batches(self, texts, max_items=None):
        """Split sorted-by-length texts into sub-batches.

        Splits on three criteria (checked in order):
        1. VRAM item limit: when max_items is set (from _estimate_max_batch_size)
        2. Length ratio: when longest/shortest > sub_batch_ratio
        3. Minimum size: ratio splits only happen after sub_batch_min_size items

        Returns list of (start, end) index tuples.
        """
        if not self._sub_batch_enabled or len(texts) <= 1:
            return [(0, len(texts))]

        # Manual cap overrides VRAM estimate when set (take the stricter of the two)
        if self._sub_batch_max_items > 0:
            max_items = min(max_items, self._sub_batch_max_items) if max_items else self._sub_batch_max_items

        sub_batches = []
        batch_start = 0

        for i in range(1, len(texts)):
            shortest = max(len(texts[batch_start]), 1)
            should_split = False

            # VRAM-estimated item limit (highest priority — based on actual
            # free GPU memory and per-sequence KV cache cost)
            if max_items is not None and (i - batch_start) >= max_items:
                should_split = True
            # Ratio split: large length disparity wastes padding —
            # only split after min_size items to preserve parallelism
            elif (i - batch_start) >= self._sub_batch_min_size:
                if len(texts[i]) > self._sub_batch_ratio * shortest:
                    should_split = True

            if should_split:
                sub_batches.append((batch_start, i))
                batch_start = i

        sub_batches.append((batch_start, len(texts)))
        return sub_batches

    # ── Lazy initialization ──────────────────────────────────────

    def _warmup_model(self, model):
        """Run a short warmup generation to pre-tune MIOpen/GPU solvers.

        First generation after model load is ~2x slower due to MIOpen autotuning.
        This warmup pays that cost upfront so real generations run at full speed.
        """
        import time
        t0 = time.time()
        try:
            model.generate_custom_voice(
                text="The ancient library stood at the crossroads of two forgotten paths, its weathered stone walls covered in ivy that had been growing for centuries.",
                language=self._language,
                speaker="serena",
                instruct="neutral",
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
            print(f"Warmup done in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"Warmup failed (non-fatal): {e}")

    def _resolve_device(self):
        """Resolve 'auto' device to the best available."""
        if self._device != "auto":
            return self._device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"


    def _enable_rocm_optimizations(self):
        """Apply ROCm-specific optimizations. No-op on NVIDIA/CPU.

        1. FLASH_ATTENTION_TRITON_AMD_ENABLE: Lets qwen_tts whisper encoder
           use native flash attention via Triton AMD backend.
        2. MIOPEN_FIND_MODE=2: Forces MIOpen to use fast-find instead of
           exhaustive search, avoiding workspace allocation failures that
           cause fallback to slow GEMM algorithms.
        3. MIOPEN_LOG_LEVEL=4: Suppress noisy MIOpen workspace warnings.
        4. triton_key shim: Bridges pytorch-triton-rocm's get_cache_key()
           to the triton_key() that PyTorch's inductor expects.
        """
        try:
            import torch
            if not (hasattr(torch.version, "hip") and torch.version.hip):
                return  # not ROCm
        except ImportError:
            return

        # MIOpen: use fast-find to avoid workspace allocation failures
        os.environ.setdefault("MIOPEN_FIND_MODE", "2")
        # Suppress MIOpen workspace warnings
        os.environ.setdefault("MIOPEN_LOG_LEVEL", "4")

        # Flash attention via Triton AMD backend
        os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")

        # Fix triton_key compatibility for torch.compile on ROCm
        try:
            from triton.compiler import compiler as triton_compiler
            if not hasattr(triton_compiler, "triton_key"):
                import triton
                triton_compiler.triton_key = lambda: f"pytorch-triton-rocm-{triton.__version__}"
        except ImportError:
            pass

        # Correct under-reported GPU properties on consumer RDNA2/3.
        # ROCm reports half the CU count and warp size 32 instead of 64,
        # causing PyTorch to under-schedule work on RX 6000/7000 GPUs.
        self._patch_rdna_device_properties(torch)


    @staticmethod
    def _patch_rdna_device_properties(torch):
        """Monkey-patch torch.cuda.get_device_properties to report correct
        CU count and wavefront size for consumer RDNA2/3 GPUs.

        ROCm exposes these GPUs with half CU count and warp_size=32
        (matching the CDNA/MI convention). The actual hardware has the
        full CU count and native wavefront64. Under-reporting causes
        PyTorch to generate smaller kernel launches.

        Based on AMD-GPU-BOOST (github.com/Painter3000/AMD-GPU-BOOST).
        """
        if hasattr(torch.cuda, '_rdna_props_patched'):
            return

        # Known RDNA GPU corrections: {name_substring: (true_CUs, true_warp)}
        _rdna_corrections = {
            "7900 XTX": (96, 64),
            "7900 XT":  (84, 64),
            "7900 GRE": (80, 64),
            "7800 XT":  (60, 64),
            "7700 XT":  (54, 64),
            "7600":     (32, 64),
            "6950 XT":  (80, 64),
            "6900 XT":  (80, 64),
            "6800 XT":  (72, 64),
            "6800":     (60, 64),
            "6750 XT":  (40, 64),
            "6700 XT":  (40, 64),
            "6700":     (36, 64),
            "6650 XT":  (32, 64),
            "6600 XT":  (32, 64),
            "6600":     (28, 64),
        }

        original_fn = torch.cuda.get_device_properties
        _cache = {}

        def _patched_get_device_properties(device=None):
            if device is None:
                device = torch.cuda.current_device()
            key = int(device) if not isinstance(device, int) else device

            if key in _cache:
                return _cache[key]

            props = original_fn(device)

            # Find matching correction
            correction = None
            for substr, vals in _rdna_corrections.items():
                if substr in props.name:
                    correction = vals
                    break

            if correction:
                from types import SimpleNamespace
                true_cus, true_warp = correction
                patched = SimpleNamespace()
                for attr in dir(props):
                    if not attr.startswith('_'):
                        try:
                            setattr(patched, attr, getattr(props, attr))
                        except (AttributeError, RuntimeError):
                            pass
                patched.multi_processor_count = true_cus
                patched.warp_size = true_warp
                old_threads = props.multi_processor_count * props.warp_size
                new_threads = true_cus * true_warp
                print(f"  [RDNA fix] {props.name}: CUs {props.multi_processor_count}->{true_cus}, "
                      f"warp {props.warp_size}->{true_warp}, "
                      f"threads {old_threads}->{new_threads}")
                _cache[key] = patched
                return patched

            _cache[key] = props
            return props

        torch.cuda.get_device_properties = _patched_get_device_properties
        torch.cuda._rdna_props_patched = True

    def _compile_codec(self, model):
        """Apply torch.compile to the audio codec for faster decoding.

        The codec decoder has 136 attention modules and many small ops that
        benefit enormously from compilation.  Profiling shows the codec is
        47% of single-gen time and 85% of batch time uncompiled.  With
        torch.compile (dynamic=True, max-autotune), batch throughput
        improves from ~1.3x to ~4.3x real-time and single generation
        drops from ~14s to ~9s.

        max-autotune mode benchmarks GPU kernels to pick the fastest and
        handles varying batch sizes gracefully (unlike reduce-overhead
        which uses CUDA graphs that break on shape changes).
        """
        import torch
        try:
            codec = model.model.speech_tokenizer.model
            model.model.speech_tokenizer.model = torch.compile(
                codec, mode="max-autotune", dynamic=True,
            )
            print("Codec compiled with torch.compile (dynamic=True).")
        except Exception as e:
            print(f"Codec compilation skipped (non-fatal): {e}")

    @staticmethod
    def _resolve_local_model_path(model_id):
        """Check if a HuggingFace model is cached locally and return its snapshot path.

        Uses try_to_load_from_cache to find the local snapshot directory.
        Returns the local path string if cached, or None if not cached.
        """
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(model_id, "config.json")
        if isinstance(result, str):
            # result is the full path to config.json inside the snapshot dir
            return os.path.dirname(result)
        return None

    @staticmethod
    def _load_model(model_cls, model_id, load_kwargs):
        """Load a model, preferring local cache to avoid network issues.

        Checks if the model snapshot exists in the HF cache and loads from
        the local directory path directly, bypassing all HF Hub network calls.
        Falls back to normal download on first install when cache is empty.
        If loading from local cache fails (e.g. incomplete snapshot), retries
        with the model ID so HF Hub can download any missing files.
        """
        local_path = TTSEngine._resolve_local_model_path(model_id)
        if local_path:
            print(f"  Loading from local cache: {local_path}")
            try:
                return model_cls.from_pretrained(local_path, **load_kwargs)
            except Exception as e:
                import traceback
                print(f"  Warning: Failed to load from local cache: {e}")
                traceback.print_exc()
                print(f"  Retrying with model ID (may download missing files)...")
                return model_cls.from_pretrained(model_id, **load_kwargs)
        else:
            print(f"  Model not cached locally, downloading {model_id}...")
            return model_cls.from_pretrained(model_id, **load_kwargs)

    def _init_local_custom(self):
        """Load Qwen3-TTS CustomVoice model on demand."""
        if self._local_custom_model is not None:
            return self._local_custom_model

        with self._model_lock:
            if self._local_custom_model is not None:
                return self._local_custom_model

            self._enable_rocm_optimizations()

            import torch
            from qwen_tts import Qwen3TTSModel

            device = self._resolve_device()
            dtype = torch.bfloat16 if "cuda" in device else torch.float32

            print(f"Loading Qwen3-TTS CustomVoice model on {device} ({dtype})...")
            load_kwargs = {"dtype": dtype}
            if device != "cpu":
                load_kwargs["device_map"] = device
            self._local_custom_model = self._load_model(
                Qwen3TTSModel, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", load_kwargs,
            )
            if self._compile_codec_enabled:
                self._compile_codec(self._local_custom_model)
            print("CustomVoice model loaded.")
            return self._local_custom_model

    def _init_local_clone(self):
        """Load Qwen3-TTS Base model (for voice cloning) on demand."""
        if self._local_clone_model is not None:
            return self._local_clone_model

        with self._model_lock:
            if self._local_clone_model is not None:
                return self._local_clone_model

            self._enable_rocm_optimizations()

            import torch
            from qwen_tts import Qwen3TTSModel

            device = self._resolve_device()
            dtype = torch.bfloat16 if "cuda" in device else torch.float32

            print(f"Loading Qwen3-TTS Base model (voice cloning) on {device} ({dtype})...")
            load_kwargs = {"dtype": dtype}
            if device != "cpu":
                load_kwargs["device_map"] = device
            self._local_clone_model = self._load_model(
                Qwen3TTSModel, "Qwen/Qwen3-TTS-12Hz-1.7B-Base", load_kwargs,
            )
            if self._compile_codec_enabled:
                self._compile_codec(self._local_clone_model)
            print("Base model (voice cloning) loaded.")
            return self._local_clone_model

    def _init_local_design(self):
        """Load Qwen3-TTS VoiceDesign model on demand."""
        if self._local_design_model is not None:
            return self._local_design_model

        with self._model_lock:
            if self._local_design_model is not None:
                return self._local_design_model

            self._enable_rocm_optimizations()

            import torch
            from qwen_tts import Qwen3TTSModel

            device = self._resolve_device()
            dtype = torch.bfloat16 if "cuda" in device else torch.float32

            print(f"Loading Qwen3-TTS VoiceDesign model on {device} ({dtype})...")
            load_kwargs = {"dtype": dtype}
            if device != "cpu":
                load_kwargs["device_map"] = device
            self._local_design_model = self._load_model(
                Qwen3TTSModel, "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", load_kwargs,
            )
            if self._compile_codec_enabled:
                self._compile_codec(self._local_design_model)
            print("VoiceDesign model loaded.")
            return self._local_design_model

    def _init_local_lora(self, adapter_path):
        """Load Qwen3-TTS Base model with a LoRA adapter on demand.

        Caches the model; if a different adapter is requested the old one
        is unloaded first to free VRAM.
        """
        if self._local_lora_model is not None and self._lora_adapter_path == adapter_path:
            return self._local_lora_model

        with self._model_lock:
            if self._local_lora_model is not None and self._lora_adapter_path == adapter_path:
                return self._local_lora_model

            # Unload previous adapter if switching
            if self._local_lora_model is not None:
                print(f"Unloading previous LoRA adapter ({self._lora_adapter_path})...")
                del self._local_lora_model
                self._local_lora_model = None
                self._lora_adapter_path = None
                self._lora_prompt_cache.clear()
                self._clear_gpu_cache()

            self._enable_rocm_optimizations()

            import torch
            from qwen_tts import Qwen3TTSModel
            from peft import PeftModel

            device = self._resolve_device()
            dtype = torch.bfloat16 if "cuda" in device else torch.float32

            print(f"Loading Qwen3-TTS Base model + LoRA adapter on {device} ({dtype})...")
            load_kwargs = {"dtype": dtype}
            if device != "cpu":
                load_kwargs["device_map"] = device

            model = self._load_model(
                Qwen3TTSModel, "Qwen/Qwen3-TTS-12Hz-1.7B-Base", load_kwargs,
            )

            # Wrap the talker with the LoRA adapter
            model.model.talker = PeftModel.from_pretrained(
                model.model.talker,
                adapter_path,
            )
            model.model.talker.eval()

            if self._compile_codec_enabled:
                self._compile_codec(model)

            self._local_lora_model = model
            self._lora_adapter_path = adapter_path
            print(f"LoRA adapter loaded from {adapter_path}")
            return model

    def unload_models(self):
        """Free all cached local TTS models and clear GPU cache.

        Called after a full conversion finishes (or via /api/unload) so
        Qwen3-TTS base/clone/design/LoRA weights don't sit in VRAM
        indefinitely. Next TTS call re-loads on demand.
        """
        with self._model_lock:
            self._stop_cpp_server()
            unloaded = []
            for attr in ("_local_custom_model", "_local_clone_model",
                         "_local_design_model", "_local_lora_model"):
                if getattr(self, attr) is not None:
                    setattr(self, attr, None)
                    unloaded.append(attr)
            self._lora_adapter_path = None
            self._lora_prompt_cache.clear()
            self._clone_prompt_cache.clear()
            self._warmup_needed = True
            if unloaded:
                print(f"Unloaded TTS models: {', '.join(unloaded)}")
                self._clear_gpu_cache()
            return unloaded

    def _stop_cpp_server(self):
        with self._cpp_clone_server_lock:
            server = self._cpp_clone_server
            self._cpp_clone_server = None
            self._cpp_server_model = None
            self._cpp_clone_voice = None
            if server is None or server.poll() is not None:
                return
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()

    def close(self):
        """Stop owned native processes during app shutdown."""
        try:
            self._stop_cpp_server()
        except Exception as exc:
            print(f"[TTS] Native server cleanup warning: {exc}")

    def _init_external(self):
        """Create Gradio client on demand."""
        if self._gradio_client is not None:
            return self._gradio_client

        from gradio_client import Client

        print(f"Connecting to TTS server at {self._url}...")
        self._gradio_client = Client(self._url)
        print("Connected to external TTS server.")
        return self._gradio_client

    # ── Clone prompt cache (local mode) ──────────────────────────

    def _get_clone_prompt(self, speaker, voice_config):
        """Get or create a cached voice clone prompt for a speaker."""
        voice_data = self._voice_data(speaker, voice_config)
        ref_audio_path = voice_data.get("ref_audio")
        ref_text = voice_data.get("ref_text")

        if not ref_audio_path or not ref_text:
            raise ValueError(f"Clone voice for '{speaker}' missing ref_audio or ref_text")
        # Resolve relative paths against project root (parent of app/)
        if not os.path.isabs(ref_audio_path):
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ref_audio_path = os.path.join(root_dir, ref_audio_path)
        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"Reference audio not found for '{speaker}': {ref_audio_path}")

        # Check cache — invalidate if ref_audio changed
        if speaker in self._clone_prompt_cache:
            cached_path, cached_prompt = self._clone_prompt_cache[speaker]
            if cached_path == ref_audio_path:
                return cached_prompt
            print(f"Voice changed for '{speaker}', rebuilding clone prompt...")

        model = self._init_local_clone()

        # Load reference audio as numpy array
        audio_array, sample_rate = sf.read(ref_audio_path)
        # Ensure mono
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        print(f"Creating clone prompt for '{speaker}'...")
        prompt = model.create_voice_clone_prompt(
            ref_audio=(audio_array, sample_rate),
            ref_text=ref_text,
        )
        self._clone_prompt_cache[speaker] = (ref_audio_path, prompt)
        print(f"Clone prompt cached for '{speaker}'.")
        return prompt

    # ── Core generation methods ──────────────────────────────────

    def generate_custom_voice(self, text, instruct_text, speaker, voice_config, output_path):
        """Generate audio using CustomVoice model. Returns True on success."""
        if self._mode == "local":
            return self._local_generate_custom(text, instruct_text, speaker, voice_config, output_path)
        else:
            return self._external_generate_custom(text, instruct_text, speaker, voice_config, output_path)

    def generate_clone_voice(self, text, speaker, voice_config, output_path):
        """Generate audio using voice cloning. Returns True on success."""
        if self._mode == "local":
            return self._local_generate_clone(text, speaker, voice_config, output_path)
        else:
            return self._external_generate_clone(text, speaker, voice_config, output_path)

    def _generate_cpp_voice(self, text, instruct_text, speaker, voice_config, output_path):
        """Generate one line with qwen-tts.cpp when the matching GGUF exists.

        Returns None when this voice type is not available in the native backend,
        allowing the existing Python implementation to handle it.
        """
        import time
        import wave

        self._cpp_error_local.message = None
        exe_path = self._cpp_executable
        binary_dir = os.path.dirname(exe_path)
        codec_path = os.path.join(self._cpp_model_dir, "qwen-tokenizer-12hz-Q8_0.gguf")
        models_dir = self._cpp_model_dir

        if self._backend != "cpp" or not os.path.isfile(exe_path):
            return None

        voice_data = self._voice_data(speaker, voice_config)
        voice_type = voice_data.get("type", "custom")
        if voice_type == "custom":
            model_path = self._ensure_cpp_model("custom")
            codec_path = self._ensure_cpp_codec()
            if not model_path or not codec_path:
                self._cpp_error_local.message = (
                    "Native CustomVoice model or tokenizer is unavailable"
                )
                return None
            speaker_value = (voice_data.get("voice") or speaker).lower()
            instruct_value = (instruct_text or voice_data.get("default_style") or "neutral").strip()
            print(f"[Blackwell Core] Generating single {voice_type} segment for '{speaker}'...")
            generation_start = time.time()
            try:
                self._ensure_cpp_synthesis_server(model_path, codec_path)
                self._request_cpp_speech(
                    text,
                    speaker_value if speaker_value != "narrator" else "serena",
                    instruct_value,
                    output_path,
                )
            except Exception as exc:
                message = f"Native CustomVoice server failed: {exc}"
                self._cpp_error_local.message = message
                print(f"[Blackwell Core] {message}")
                return False
            elapsed = time.time() - generation_start
            audio_duration = self._wav_duration(output_path)
            print(
                f" -> [Metrics] Single Runtime: {elapsed:.2f}s | "
                f"Audio Generated: {audio_duration:.2f}s | "
                f"RTF: {elapsed / max(audio_duration, 0.001):.3f}"
            )
            print(f"[Blackwell Core] Single-segment WAV written to {output_path}")
            return True
        elif voice_type == "clone":
            model_path = self._ensure_cpp_model("base")
            codec_path = self._ensure_cpp_codec()
            ref_audio_path = voice_data.get("ref_audio")
            ref_text = voice_data.get("ref_text")
            if not model_path or not ref_audio_path or not ref_text:
                if not model_path:
                    self._cpp_error_local.message = "Native clone Base model is unavailable"
                    print(
                        "[Blackwell Core] Native clone model unavailable; "
                        "using the configured Python clone backend."
                    )
                return None
            if not os.path.isabs(ref_audio_path):
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ref_audio_path = os.path.join(root_dir, ref_audio_path)
            if not os.path.isfile(ref_audio_path):
                self._cpp_error_local.message = f"Clone reference audio not found: {ref_audio_path}"
                return None
            print(f"[Blackwell Core] Generating single {voice_type} segment for '{speaker}'...")
            generation_start = time.time()
            try:
                voice_key = self._ensure_cpp_clone_server(
                    model_path, codec_path, ref_audio_path, ref_text
                )
                self._request_cpp_clone(text, voice_key, output_path)
            except Exception as exc:
                message = f"Native clone server failed: {exc}"
                self._cpp_error_local.message = message
                print(f"[Blackwell Core] {message}")
                return False
            elapsed = time.time() - generation_start
            audio_duration = self._wav_duration(output_path)
            print(
                f" -> [Metrics] Single Runtime: {elapsed:.2f}s | "
                f"Audio Generated: {audio_duration:.2f}s | "
                f"RTF: {elapsed / max(audio_duration, 0.001):.3f}"
            )
            print(f"[Blackwell Core] Single-segment WAV written to {output_path}")
            return True
        else:
            return None

        print(f"[Blackwell Core] Generating single {voice_type} segment for '{speaker}'...")
        generation_start = time.time()
        try:
            completed = subprocess.run(
                args,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=binary_dir,
                check=False,
            )
        except OSError as exc:
            print(f"[Blackwell Core] Native generation unavailable: {exc}")
            return None
        finally:
            pass

        if completed.returncode != 0 or not os.path.isfile(output_path):
            output = completed.stdout.decode("utf-8", errors="replace").strip()
            print(f"[Blackwell Core] Single-segment generation failed ({completed.returncode}): {output}")
            return False

        elapsed = time.time() - generation_start
        try:
            with wave.open(output_path, "rb") as wav_file:
                audio_duration = wav_file.getnframes() / max(wav_file.getframerate(), 1)
        except (wave.Error, OSError):
            audio_duration = max((os.path.getsize(output_path) - 44) / 24000.0, 0.0)
        rtf = elapsed / max(audio_duration, 0.001)
        print(
            f" -> [Metrics] Single Runtime: {elapsed:.2f}s | "
            f"Audio Generated: {audio_duration:.2f}s | RTF: {rtf:.3f}"
        )
        print(f"[Blackwell Core] Single-segment WAV written to {output_path}")
        return True

    @staticmethod
    def _wav_duration(path):
        import wave
        try:
            with wave.open(path, "rb") as wav_file:
                return wav_file.getnframes() / max(wav_file.getframerate(), 1)
        except (wave.Error, OSError):
            return max((os.path.getsize(path) - 44) / 24000.0, 0.0)

    def _cpp_server_url(self, suffix):
        return f"http://127.0.0.1:{self._cpp_server_port}{suffix}"

    def _cpp_server_get(self, suffix):
        with urllib.request.urlopen(self._cpp_server_url(suffix), timeout=2) as response:
            return response.read()

    def _ensure_cpp_clone_server(self, model_path, codec_path, ref_audio_path, ref_text):
        import time

        self._ensure_cpp_synthesis_server(model_path, codec_path)

        voice_key = hashlib.sha256(
            (os.path.abspath(ref_audio_path) + "\0" + ref_text).encode("utf-8")
        ).hexdigest()[:16]
        with self._cpp_clone_server_lock:
            if self._cpp_clone_voice != voice_key:
                with open(ref_audio_path, "rb") as ref_file:
                    wav_b64 = base64.b64encode(ref_file.read()).decode("ascii")
                self._post_cpp_json("/v1/audio/voices", {
                    "name": voice_key,
                    "ref_text": ref_text,
                    "wav_b64": wav_b64,
                })
                self._cpp_clone_voice = voice_key
            return voice_key

    def _ensure_cpp_synthesis_server(self, model_path, codec_path):
        import time

        with self._cpp_clone_server_lock:
            if self._cpp_server_model == model_path:
                try:
                    self._cpp_server_get("/health")
                    return
                except (OSError, urllib.error.URLError):
                    self._cpp_server_model = None

            if self._cpp_clone_server is not None and self._cpp_clone_server.poll() is None:
                self._cpp_clone_server.terminate()
                try:
                    self._cpp_clone_server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._cpp_clone_server.kill()
                self._cpp_clone_server = None
                self._cpp_server_model = None

            server_path = os.path.join(os.path.dirname(self._cpp_executable), "tts-server.exe")
            if not os.path.isfile(server_path):
                raise FileNotFoundError(f"Native server not found: {server_path}")
            self._cpp_clone_server = subprocess.Popen(
                [
                    server_path, "--model", model_path, "--codec", codec_path,
                    "--host", "127.0.0.1", "--port", str(self._cpp_server_port),
                    "--lang", self._language, "--max-batch", "4",
                ],
                cwd=os.path.dirname(server_path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            for _ in range(120):
                try:
                    self._cpp_server_get("/health")
                    self._cpp_server_model = model_path
                    self._cpp_clone_voice = None
                    return
                except (OSError, urllib.error.URLError):
                    time.sleep(0.25)
            raise RuntimeError("Native TTS server did not become ready")

    def _post_cpp_json(self, suffix, payload):
        request = urllib.request.Request(
            self._cpp_server_url(suffix),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Native server HTTP {exc.code}: {detail}") from exc

    def _request_cpp_clone(self, text, voice, output_path):
        self._request_cpp_speech(text, voice, "", output_path)

    def _request_cpp_speech(self, text, voice, instruct, output_path):
        payload = {
            "input": text,
            "voice": voice,
            "language": self._language,
            "response_format": "wav",
            "max_new_tokens": 512,
        }
        if instruct:
            payload["instructions"] = instruct
        body = self._post_cpp_json("/v1/audio/speech", payload)
        with open(output_path, "wb") as output_file:
            output_file.write(body)

    def generate_voice(self, text, instruct_text, speaker, voice_config, output_path):
        """Generate audio using the appropriate method based on voice type config."""
        voice_data = self._voice_data(speaker, voice_config)
        if not voice_data:
            print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
            return False

        voice_type = voice_data.get("type", "custom")

        if self._mode == "local" and self._backend == "cpp":
            if voice_type not in ("custom", "clone"):
                message = (
                    f"Native C++ backend does not support voice type '{voice_type}' "
                    "for this segment; select Python backend or enable fallback."
                )
                print(f"[TTS] {message}")
                if not self._python_fallback:
                    return False
            cpp_result = self._generate_cpp_voice(
                text, instruct_text, speaker, voice_config, output_path
            )
            if cpp_result is not None:
                return cpp_result
            if not self._python_fallback:
                print(f"[TTS] Native C++ backend cannot handle voice '{speaker}'. Python fallback is disabled.")
                return False

        if voice_type == "clone":
            return self.generate_clone_voice(text, speaker, voice_config, output_path)
        elif voice_type in ("lora", "builtin_lora"):
            return self.generate_lora_voice(text, instruct_text, voice_data, output_path)
        elif voice_type == "design":
            return self.generate_design_voice(text, instruct_text, voice_data, output_path)
        else:
            return self.generate_custom_voice(text, instruct_text, speaker, voice_config, output_path)

    # ── Voice design generation ──────────────────────────────────

    def generate_voice_design(self, description, sample_text, language=None, seed=-1):
        """Generate a voice from a text description using the VoiceDesign model.

        Args:
            description: Natural language description of the desired voice
            sample_text: Text to synthesize with the designed voice
            language: Language code (defaults to engine's configured language)
            seed: Random seed (-1 for random, >= 0 for reproducible)

        Returns:
            (wav_path, sample_rate) on success

        Raises:
            RuntimeError: If generation fails
        """
        import time
        import tempfile
        import torch

        lang = language or self._language
        print(f"VoiceDesign: generating preview for description='{description[:80]}...'"
              f"{f', seed={seed}' if seed >= 0 else ''}")

        model = self._init_local_design()

        if seed >= 0:
            torch.manual_seed(seed)

        t_start = time.time()
        wavs, sr = model.generate_voice_design(
            text=sample_text,
            instruct=description,
            language=lang,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        gen_time = time.time() - t_start

        if wavs is None or len(wavs) == 0:
            raise RuntimeError("VoiceDesign model returned no audio")

        audio = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
        duration = len(audio) / sr
        print(f"VoiceDesign: done in {gen_time:.1f}s -> {duration:.1f}s audio")

        # Save to previews directory
        previews_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "designed_voices", "previews")
        os.makedirs(previews_dir, exist_ok=True)

        filename = f"preview_{int(time.time() * 1000)}.wav"
        wav_path = os.path.join(previews_dir, filename)
        self._save_wav(audio, sr, wav_path)

        return wav_path, sr

    def generate_design_voice(self, text, instruct_text, voice_data, output_path):
        """Generate audio using VoiceDesign model with combined description + instruct.

        The voice_data 'description' field provides the base voice identity,
        and the per-line instruct_text is appended for delivery/emotion direction.
        """
        import shutil

        base_desc = (voice_data.get("description") or "").strip()
        instruct = (instruct_text or "").strip()

        if base_desc and instruct:
            description = f"{base_desc}, {instruct}"
        elif base_desc:
            description = base_desc
        elif instruct:
            description = instruct
        else:
            print("Warning: Design voice has no description or instruct. Using generic.")
            description = "A clear, natural speaking voice"

        wav_path, sr = self.generate_voice_design(description=description, sample_text=text)
        shutil.copy2(wav_path, output_path)
        return True

    # ── LoRA voice generation ────────────────────────────────────

    def generate_lora_voice(self, text, instruct_text, voice_data, output_path):
        """Generate audio using a LoRA-finetuned Base model.

        The adapter directory must contain:
          - PEFT adapter weights (adapter_model.safetensors / adapter_config.json)
          - ref_sample.wav (reference audio for voice cloning prompt)
          - training_meta.json (with ref_sample_text)

        The LoRA weights refine voice identity beyond what the reference alone provides.
        """
        try:
            import torch
            import time

            adapter_path = voice_data.get("adapter_path")
            if not adapter_path:
                print(f"Error: No adapter_path in voice_data")
                return False

            # Resolve relative paths against project root
            if not os.path.isabs(adapter_path):
                root_dir = os.path.dirname(os.path.dirname(__file__))
                adapter_path = os.path.join(root_dir, adapter_path)

            if not os.path.isdir(adapter_path):
                # Auto-download built-in adapters from HF
                adapter_id = os.path.basename(adapter_path)
                if adapter_id.startswith("builtin_"):
                    print(f"Adapter {adapter_id} not downloaded, attempting auto-download...")
                    try:
                        from hf_utils import download_builtin_adapter
                        builtin_dir = os.path.dirname(adapter_path)
                        download_builtin_adapter(adapter_id, builtin_dir)
                    except Exception as e:
                        print(f"Error: Auto-download failed for {adapter_id}: {e}")
                        return False
                else:
                    print(f"Error: LoRA adapter path not found: {adapter_path}")
                    return False

            # Load reference audio and text from adapter directory
            ref_wav_path = os.path.join(adapter_path, "ref_sample.wav")
            meta_path = os.path.join(adapter_path, "training_meta.json")

            if not os.path.exists(ref_wav_path):
                print(f"Error: ref_sample.wav not found in {adapter_path}")
                return False
            if not os.path.exists(meta_path):
                print(f"Error: training_meta.json not found in {adapter_path}")
                return False

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            ref_text = meta.get("ref_sample_text", "")
            if not ref_text:
                print(f"Error: ref_sample_text missing from training_meta.json")
                return False

            print(f"TTS [local lora] generating for adapter={os.path.basename(adapter_path)}, "
                  f"text='{text[:50]}...'")

            model = self._init_local_lora(adapter_path)

            # Build or reuse voice clone prompt for this adapter
            if adapter_path not in self._lora_prompt_cache:
                audio_array, sample_rate = sf.read(ref_wav_path)
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=1)
                print(f"Creating clone prompt for LoRA adapter...")
                prompt = model.create_voice_clone_prompt(
                    ref_audio=(audio_array, sample_rate),
                    ref_text=ref_text,
                    x_vector_only_mode=True,
                )
                self._lora_prompt_cache[adapter_path] = prompt
                print(f"Clone prompt cached for LoRA adapter.")

            prompt = self._lora_prompt_cache[adapter_path]

            # Build instruct_ids so the Base model can follow style prompts
            gen_extra = {}
            instruct = instruct_text or ""
            character_style = voice_data.get("character_style", "") or voice_data.get("default_style", "")
            if character_style:
                instruct = f"{instruct} {character_style}".strip()
            if instruct:
                instruct_formatted = f"<|im_start|>user\n{instruct}<|im_end|>\n"
                gen_extra["instruct_ids"] = model._tokenize_texts([instruct_formatted])

            t_start = time.time()
            wavs, sr = model.generate_voice_clone(
                text=text,
                voice_clone_prompt=prompt,
                non_streaming_mode=True,
                max_new_tokens=2048,
                **gen_extra,
            )
            gen_time = time.time() - t_start

            if wavs is None or len(wavs) == 0:
                print(f"Error: No audio generated for: '{text[:50]}...'")
                return False

            audio = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
            duration = len(audio) / sr
            rtf = duration / gen_time if gen_time > 0 else 0
            print(f"TTS [local lora] done: {gen_time:.1f}s -> {duration:.1f}s audio ({rtf:.2f}x real-time)")
            self._save_wav(audio, sr, output_path)
            return True

        except Exception as e:
            import traceback
            print(f"Error generating LoRA voice: {e}")
            traceback.print_exc()
            return False

    # ── Batch generation ─────────────────────────────────────────

    @staticmethod
    def _voice_data(speaker, voice_config):
        voice_data = voice_config.get(speaker)
        if voice_data is not None:
            return voice_data
        return next(
            (value for key, value in voice_config.items()
             if key.casefold() == speaker.casefold()),
            {},
        )

    def _cpp_base_model_path(self):
        return self._ensure_cpp_model("base", download=False)

    def _ensure_cpp_codec(self):
        """Return the native tokenizer GGUF, downloading it on first use."""
        return self._ensure_cpp_asset("qwen-tokenizer-12hz-Q8_0.gguf")

    def _ensure_cpp_model(self, variant, download=None):
        filenames = {
            "custom": "qwen-talker-1.7b-customvoice-Q8_0.gguf",
            "base": "qwen-talker-1.7b-base-Q8_0.gguf",
        }
        filename = filenames.get(variant)
        if not filename:
            return None
        return self._ensure_cpp_asset(filename, download=download)

    def _ensure_cpp_asset(self, filename, download=None):
        path = os.path.join(self._cpp_model_dir, filename)
        if os.path.isfile(path):
            return path
        if download is False or not self._cpp_auto_download:
            return None

        try:
            from huggingface_hub import hf_hub_download

            os.makedirs(self._cpp_model_dir, exist_ok=True)
            print(f"[Blackwell Core] Downloading native GGUF asset: {filename}")
            downloaded = hf_hub_download(
                repo_id="Serveurperso/Qwen3-TTS-GGUF",
                filename=filename,
                local_dir=self._cpp_model_dir,
            )
            return downloaded if os.path.isfile(downloaded) else path
        except Exception as exc:
            print(f"[Blackwell Core] Native GGUF download failed for {filename}: {exc}")
            return None

    def generate_batch(self, chunks, voice_config, output_dir, batch_seed=-1, cancel_check=None):
        """Root master batch supervisor overridden natively to process chunks via parallel Blackwell C++ binaries."""
        import time
        import json

        if not hasattr(self, "_global_start_time") or self._global_start_time == 0:
            self._global_start_time = time.time()
            self._global_audio_duration = 0.0
            self._processed_chunks_counter = 0

        batch_start_time = time.time()
        results = {"completed": [], "failed": []}
        if cancel_check and cancel_check():
            return results
        groups = {"custom": [], "clone": [], "lora": [], "design": []}
        for chunk in chunks:
            voice_data = self._voice_data(chunk.get("speaker", ""), voice_config)
            voice_type = voice_data.get("type", "custom")
            if voice_type in ("lora", "builtin_lora"):
                group_name = "lora"
            elif voice_type in groups:
                group_name = voice_type
            else:
                group_name = "custom"
            groups[group_name].append(chunk)

        if self._mode == "external":
            for chunk in chunks:
                idx = chunk["index"]
                output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                if self.generate_voice(
                    chunk.get("text", ""), chunk.get("instruct", ""),
                    chunk.get("speaker", ""), voice_config, output_path,
                ):
                    results["completed"].append(idx)
                else:
                    results["failed"].append((idx, "External TTS generation failed"))
            batch_elapsed = time.time() - batch_start_time
            batch_audio = sum(
                self._wav_duration(os.path.join(output_dir, f"temp_batch_{idx}.wav"))
                for idx in results["completed"]
                if os.path.exists(os.path.join(output_dir, f"temp_batch_{idx}.wav"))
            )
            self._global_audio_duration += batch_audio
            print(
                f" -> [Metrics] Batch Runtime: {batch_elapsed:.2f}s | "
                f"Audio Generated: {batch_audio:.2f}s | "
                f"BATCH RTF: {batch_elapsed / max(batch_audio, 0.001):.3f}"
            )
            return results

        if groups["custom"]:
            if self._backend == "cpp":
                custom_results = self._local_batch_custom(
                    groups["custom"], voice_config, output_dir, batch_seed,
                    cancel_check=cancel_check,
                )
            else:
                custom_results = self._local_batch_python(
                    groups["custom"], voice_config, output_dir
                )
            results["completed"].extend(custom_results["completed"])
            results["failed"].extend(custom_results["failed"])
        if groups["clone"]:
            if self._backend != "cpp":
                clone_results = self._local_batch_clone(
                    groups["clone"], voice_config, output_dir
                )
            elif self._ensure_cpp_model("base") and self._ensure_cpp_codec():
                clone_results = self._local_batch_cpp_clone(
                    groups["clone"], voice_config, output_dir
                )
            elif self._python_fallback:
                print("[TTS] Clone batch is using the opted-in Python Qwen3-TTS Base backend.")
                clone_results = self._local_batch_clone(
                    groups["clone"], voice_config, output_dir
                )
            else:
                error = "Native C++ clone model not found; Python fallback is disabled."
                print(f"[TTS] {error}")
                clone_results = {
                    "completed": [],
                    "failed": [(chunk["index"], error) for chunk in groups["clone"]],
                }
            results["completed"].extend(clone_results["completed"])
            results["failed"].extend(clone_results["failed"])
        if groups["lora"]:
            lora_results = self._local_batch_lora(
                groups["lora"], voice_config, output_dir
            )
            results["completed"].extend(lora_results["completed"])
            results["failed"].extend(lora_results["failed"])
        for chunk in groups["design"]:
            idx = chunk["index"]
            output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
            if self.generate_voice(
                chunk.get("text", ""), chunk.get("instruct", ""),
                chunk.get("speaker", ""), voice_config, output_path,
            ):
                results["completed"].append(idx)
            else:
                results["failed"].append((idx, "VoiceDesign generation failed"))
        batch_elapsed = time.time() - batch_start_time

        batch_audio = 0.0
        if "completed" in results:
            for idx in results["completed"]:
                target_wav = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                if os.path.exists(target_wav):
                    dur = self._wav_duration(target_wav)
                    batch_audio += dur
                    self._global_audio_duration += dur

        batch_rtf = batch_elapsed / max(0.001, batch_audio)
        print(f" -> [Metrics] Batch Runtime: {batch_elapsed:.2f}s | Audio Generated: {batch_audio:.2f}s | BATCH RTF: {batch_rtf:.3f}")

        self._processed_chunks_counter += len(results["completed"]) + len(results["failed"])

        project_manager_path = os.path.join(output_dir, "chunks.json")
        db_len = 0
        if os.path.exists(project_manager_path):
            with open(project_manager_path, "r", encoding="utf-8") as jf:
                db_len = len(json.load(jf))

        if db_len > 0 and self._processed_chunks_counter >= db_len:
            global_elapsed = time.time() - self._global_start_time
            total_audio = self._global_audio_duration
            if total_audio > 0:
                print(f"\n=======================================================")
                print(f"      GLOBAL PIPELINE ENGINE PERFORMANCE METRICS")
                print(f"=======================================================")
                print(f" -> Total Generation Compute Time : {global_elapsed:.2f} seconds")
                print(f" -> Total Audio Duration Compiled  : {total_audio:.2f} seconds")
                print(f" -> DEFINITIVE GLOBAL ENGINE RTF   : {global_elapsed/max(0.001, total_audio):.3f}")
                print(f"=======================================================\n")
            self._global_start_time = 0
            self._global_audio_duration = 0.0
            self._processed_chunks_counter = 0

        return results

    def _local_batch_python(self, chunks, voice_config, output_dir):
        """Generate a local Python batch while preserving the stock engine path."""
        results = {"completed": [], "failed": []}
        for chunk in chunks:
            idx = chunk["index"]
            output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
            try:
                success = self.generate_voice(
                    chunk.get("text", ""),
                    chunk.get("instruct", ""),
                    chunk.get("speaker", ""),
                    voice_config,
                    output_path,
                )
            except Exception as exc:
                success = False
                error = str(exc)
            else:
                error = "Python TTS generation failed"
            if success:
                results["completed"].append(idx)
            else:
                results["failed"].append((idx, error))
        return results

    def _local_batch_cpp_clone(self, chunks, voice_config, output_dir):
        """Generate clone chunks through qwen-tts.cpp without loading Python TTS."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {"completed": [], "failed": []}
        print(f"\n[Blackwell Core] Native clone batch: {len(chunks)} chunks")

        def process_chunk(chunk):
            idx = chunk["index"]
            output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
            success = self._generate_cpp_voice(
                chunk.get("text", ""),
                chunk.get("instruct", ""),
                chunk.get("speaker", ""),
                voice_config,
                output_path,
            )
            return idx, success, getattr(self._cpp_error_local, "message", None)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                idx, success, error = future.result()
                if success:
                    results["completed"].append(idx)
                else:
                    results["failed"].append((
                        idx,
                        error or "Native C++ clone generation failed",
                    ))

        return results

    def _local_batch_custom(self, chunks, voice_config, output_dir, batch_seed=-1, cancel_check=None):
        """Batch generate custom voice utilizing concurrent execution of optimized C++ binaries with cloning support."""
        import struct
        import json
        import re
        import time
        import subprocess
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {"completed": [], "failed": []}
        if cancel_check and cancel_check():
            return results

        exe_path = self._cpp_executable
        binary_dir = os.path.dirname(exe_path)
        model_path = self._ensure_cpp_model("custom")
        codec_path = self._ensure_cpp_codec()
        if cancel_check and cancel_check():
            return results
        if not model_path or not codec_path:
            error = "Native C++ CustomVoice model or tokenizer is unavailable."
            print(f"[TTS] {error}")
            results["failed"] = [(chunk["index"], error) for chunk in chunks]
            return results

        max_workers = 4
        print(f"\n[Blackwell Core] Hijacking local batch queue for {len(chunks)} fragments across {max_workers} parallel C++ threads...")

        def process_single_chunk(chunk):
            idx = chunk["index"]
            text_val = chunk.get("text", "")
            instruct_text = chunk.get("instruct", "")
            speaker_name = chunk.get("speaker", "serena")

            voice_data = self._voice_data(speaker_name, voice_config)
            speaker_val = voice_data.get("voice", speaker_name).lower()
            character_style = voice_data.get("character_style", "") or voice_data.get("default_style", "")

            instruct_val = instruct_text if instruct_text else "neutral"
            if character_style:
                instruct_val = f"{instruct_val} {character_style}"

            if not instruct_val.strip():
                instruct_val = "Clear textbook narration; formal and steady tone."

            target_wav_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
            if not text_val.strip():
                return idx, True, 0, None

            if self._backend == "cpp":
                success = self._generate_cpp_voice(
                    text_val, instruct_val, speaker_name, voice_config, target_wav_path
                )
                duration = self._wav_duration(target_wav_path) if success else 0
                return (
                    idx,
                    success,
                    duration,
                    None if success else (
                        getattr(self._cpp_error_local, "message", None)
                        or "Native CustomVoice generation failed"
                    ),
                )

            try:
                sentences = re.split(r'(?<=[\.!?])\s+', text_val)
                chunk_files = []
                sub_idx = 1

                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    temp_sub_file = os.path.join(binary_dir, f"temp_fork_batch_chunk_{idx}_{sub_idx}.wav")

                    args = [
                        f'"{exe_path}"',
                        "--model", f'"{model_path}"',
                        "--codec", f'"{codec_path}"',
                        "--speaker", speaker_val if speaker_val != "narrator" else "serena",
                        "--lang", "English",
                        "--max-new", "512",
                        "--instruct", f'"{instruct_val}"',
                        "-o", f'"{temp_sub_file}"'
                    ]

                    cmd_string = " ".join(args)
                    p = subprocess.Popen(cmd_string, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=binary_dir)
                    p.communicate(input=sentence.encode('utf-8'))

                    if os.path.exists(temp_sub_file):
                        chunk_files.append(temp_sub_file)
                    sub_idx += 1

                if chunk_files:
                    with open(target_wav_path, "wb") as master:
                        with open(chunk_files[0], "rb") as first:
                            master.write(first.read())
                        for i in range(1, len(chunk_files)):
                            with open(chunk_files[i], "rb") as cf:
                                data = cf.read()
                                if len(data) > 44:
                                    master.write(data[44:])

                    if os.path.exists(target_wav_path):
                        with open(target_wav_path, "r+b") as master_patch:
                            master_patch.seek(0, 2)
                            file_size = master_patch.tell()
                            master_patch.seek(4)
                            master_patch.write(struct.pack("<I", file_size - 8))
                            master_patch.seek(40)
                            master_patch.write(struct.pack("<I", file_size - 44))
                        audio_seconds = self._wav_duration(target_wav_path)
                    else:
                        audio_seconds = 0

                    for f in chunk_files:
                        try: os.remove(f)
                        except: pass
                    return idx, True, audio_seconds, None
                else:
                    return idx, False, 0, "Binary voice generation failed."
            except Exception as e:
                return idx, False, 0, str(e)

        total_audio_time = 0
        executor = ThreadPoolExecutor(max_workers=max_workers)
        future_to_chunk = {executor.submit(process_single_chunk, chunk): chunk for chunk in chunks}
        cancelled = False
        try:
            for future in as_completed(future_to_chunk):
                if cancel_check and cancel_check():
                    cancelled = True
                    print("[CANCEL] Cancellation requested - stopping native batch generation")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                p_idx, success, duration, error_msg = future.result()
                if success:
                    results["completed"].append(p_idx)
                else:
                    results["failed"].append((p_idx, error_msg))
                    print(f" -> [Blackwell Core] Chunk {p_idx} FAILED: {error_msg}")
        finally:
            if not cancelled:
                executor.shutdown(wait=True)

        return results

    def _local_batch_clone(self, chunks, voice_config, output_dir):
        """Batch generate clone voices, grouped by speaker.

        Chunks sharing the same speaker (same reference audio) are batched
        together through generate_voice_clone(text=[list], ...).
        Sub-batching by text length is applied within each speaker group.
        """
        import torch
        import time

        results = {"completed": [], "failed": []}

        # Group chunks by speaker
        speaker_groups = {}
        for chunk in chunks:
            speaker = chunk.get("speaker", "")
            speaker_groups.setdefault(speaker, []).append(chunk)

        model = self._init_local_clone()

        self._clear_gpu_cache()


        t_total_start = time.time()
        total_audio_duration = 0.0

        for speaker, group in speaker_groups.items():
            try:
                prompt = self._get_clone_prompt(speaker, voice_config)
            except Exception as e:
                print(f"  Error building clone prompt for '{speaker}': {e}")
                for chunk in group:
                    results["failed"].append((chunk["index"], str(e)))
                continue

            texts = [c["text"] for c in group]
            indices = [c["index"] for c in group]

            # Sort by text length for sub-batching efficiency
            sort_order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            texts = [texts[i] for i in sort_order]
            indices = [indices[i] for i in sort_order]

            # Estimate max batch size from VRAM + clone prompt overhead
            clone_tokens = prompt[0].ref_code.shape[0] if prompt[0].ref_code is not None else 0
            ref_text_chars = len(prompt[0].ref_text) if prompt[0].ref_text else 0
            max_items = self._estimate_max_batch_size(
                model, clone_tokens, ref_text_chars, len(texts[-1]),
            )
            sub_batches = self._build_sub_batches(texts, max_items=max_items)

            print(f"Batch [clone] speaker='{speaker}': {len(texts)} chunks "
                  f"in {len(sub_batches)} sub-batch(es)")

            for sb_idx, (start, end) in enumerate(sub_batches):
                sb_texts = texts[start:end]
                sb_indices = indices[start:end]

                print(f"  Sub-batch {sb_idx+1}/{len(sub_batches)}: {len(sb_texts)} chunks "
                      f"({len(sb_texts[0])}-{len(sb_texts[-1])} chars/chunk)")

                try:
                    t_start = time.time()
                    wavs_list, sr = model.generate_voice_clone(
                        text=sb_texts,
                        voice_clone_prompt=prompt,
                        non_streaming_mode=True,
                        max_new_tokens=2048,
                    )
                    gen_time = time.time() - t_start

                    if wavs_list is None:
                        for idx in sb_indices:
                            results["failed"].append((idx, "Batch returned None"))
                        continue

                    sb_audio_duration = 0.0
                    for wav, idx in zip(wavs_list, sb_indices):
                        try:
                            output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                            audio = self._concat_audio(wav)
                            self._save_wav(audio, sr, output_path)
                            results["completed"].append(idx)
                            duration = len(audio) / sr
                            sb_audio_duration += duration
                        except Exception as e:
                            print(f"    Error saving chunk {idx}: {e}")
                            results["failed"].append((idx, str(e)))

                    total_audio_duration += sb_audio_duration
                    sb_rtf = sb_audio_duration / gen_time if gen_time > 0 else 0
                    print(f"  Sub-batch {sb_idx+1} done: {gen_time:.1f}s -> {sb_audio_duration:.1f}s audio ({sb_rtf:.2f}x RT)")

                except Exception as e:
                    print(f"  Sub-batch {sb_idx+1} failed: {e}")
                    for idx in sb_indices:
                        results["failed"].append((idx, f"Batch error: {e}"))

                self._clear_gpu_cache()

        total_time = time.time() - t_total_start
        rtf = total_audio_duration / total_time if total_time > 0 else 0
        print(f"Batch [clone] total: {total_time:.1f}s -> {total_audio_duration:.1f}s audio ({rtf:.2f}x real-time)")



        return results

    def _local_batch_lora(self, chunks, voice_config, output_dir):
        """Batch generate LoRA voices, grouped by adapter.

        Chunks sharing the same adapter are batched together through
        generate_voice_clone(text=[list], instruct_ids=[list], ...).
        Sub-batching by text length is applied within each adapter group.
        """
        import torch
        import time

        results = {"completed": [], "failed": []}
        root_dir = os.path.dirname(os.path.dirname(__file__))

        # Group chunks by adapter_path (resolved to absolute)
        adapter_groups = {}  # adapter_path -> (voice_data, [chunks])
        for chunk in chunks:
            speaker = chunk.get("speaker", "")
            voice_data = voice_config.get(speaker, {})
            adapter_path = voice_data.get("adapter_path", "")

            if not adapter_path:
                results["failed"].append((chunk["index"], "No adapter_path"))
                continue

            if not os.path.isabs(adapter_path):
                adapter_path = os.path.join(root_dir, adapter_path)

            if adapter_path not in adapter_groups:
                adapter_groups[adapter_path] = (voice_data, [])
            adapter_groups[adapter_path][1].append(chunk)

        self._clear_gpu_cache()


        t_total_start = time.time()
        total_audio_duration = 0.0

        for adapter_path, (voice_data, group) in adapter_groups.items():
            if not os.path.isdir(adapter_path):
                print(f"  Error: adapter path not found: {adapter_path}")
                for chunk in group:
                    results["failed"].append((chunk["index"], f"Adapter not found: {adapter_path}"))
                continue

            # Load adapter and build/get clone prompt
            try:
                ref_wav_path = os.path.join(adapter_path, "ref_sample.wav")
                meta_path = os.path.join(adapter_path, "training_meta.json")
                if not os.path.exists(ref_wav_path) or not os.path.exists(meta_path):
                    raise FileNotFoundError(f"Missing ref_sample.wav or training_meta.json in {adapter_path}")

                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                ref_text = meta.get("ref_sample_text", "")
                if not ref_text:
                    raise ValueError("ref_sample_text missing from training_meta.json")

                model = self._init_local_lora(adapter_path)

                if adapter_path not in self._lora_prompt_cache:
                    audio_array, sample_rate = sf.read(ref_wav_path)
                    if audio_array.ndim > 1:
                        audio_array = audio_array.mean(axis=1)
                    print(f"Creating clone prompt for LoRA adapter...")
                    prompt = model.create_voice_clone_prompt(
                        ref_audio=(audio_array, sample_rate),
                        ref_text=ref_text,
                        x_vector_only_mode=True,
                    )
                    self._lora_prompt_cache[adapter_path] = prompt
                    print(f"Clone prompt cached for LoRA adapter.")

                prompt = self._lora_prompt_cache[adapter_path]
            except Exception as e:
                print(f"  Error loading LoRA adapter {os.path.basename(adapter_path)}: {e}")
                for chunk in group:
                    results["failed"].append((chunk["index"], str(e)))
                continue

            character_style = voice_data.get("character_style", "") or voice_data.get("default_style", "")

            texts = [c["text"] for c in group]
            instructs_raw = [c.get("instruct", "") for c in group]
            indices = [c["index"] for c in group]

            # Sort by text length
            sort_order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            texts = [texts[i] for i in sort_order]
            instructs_raw = [instructs_raw[i] for i in sort_order]
            indices = [indices[i] for i in sort_order]

            # Estimate max batch size from VRAM + clone prompt overhead
            clone_tokens = prompt[0].ref_code.shape[0] if prompt[0].ref_code is not None else 0
            ref_text_chars = len(prompt[0].ref_text) if prompt[0].ref_text else 0
            max_items = self._estimate_max_batch_size(
                model, clone_tokens, ref_text_chars, len(texts[-1]),
            )
            sub_batches = self._build_sub_batches(texts, max_items=max_items)

            print(f"Batch [lora] adapter='{os.path.basename(adapter_path)}': {len(texts)} chunks "
                  f"in {len(sub_batches)} sub-batch(es)")

            for sb_idx, (start, end) in enumerate(sub_batches):
                sb_texts = texts[start:end]
                sb_instructs = instructs_raw[start:end]
                sb_indices = indices[start:end]

                print(f"  Sub-batch {sb_idx+1}/{len(sub_batches)}: {len(sb_texts)} chunks "
                      f"({len(sb_texts[0])}-{len(sb_texts[-1])} chars/chunk)")

                try:
                    # Build instruct_ids list for this sub-batch
                    instruct_ids = []
                    for inst in sb_instructs:
                        instruct = inst or ""
                        if character_style:
                            instruct = f"{instruct} {character_style}".strip()
                        if instruct:
                            instruct_formatted = f"<|im_start|>user\n{instruct}<|im_end|>\n"
                            instruct_ids.append(model._tokenize_texts([instruct_formatted])[0])
                        else:
                            instruct_ids.append(None)

                    gen_extra = {}
                    if any(iid is not None for iid in instruct_ids):
                        gen_extra["instruct_ids"] = instruct_ids

                    t_start = time.time()
                    wavs_list, sr = model.generate_voice_clone(
                        text=sb_texts,
                        voice_clone_prompt=prompt,
                        non_streaming_mode=True,
                        max_new_tokens=2048,
                        **gen_extra,
                    )
                    gen_time = time.time() - t_start

                    if wavs_list is None:
                        for idx in sb_indices:
                            results["failed"].append((idx, "Batch returned None"))
                        continue

                    sb_audio_duration = 0.0
                    for wav, idx in zip(wavs_list, sb_indices):
                        try:
                            output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                            audio = self._concat_audio(wav)
                            self._save_wav(audio, sr, output_path)
                            results["completed"].append(idx)
                            duration = len(audio) / sr
                            sb_audio_duration += duration
                        except Exception as e:
                            print(f"    Error saving chunk {idx}: {e}")
                            results["failed"].append((idx, str(e)))

                    total_audio_duration += sb_audio_duration
                    sb_rtf = sb_audio_duration / gen_time if gen_time > 0 else 0
                    print(f"  Sub-batch {sb_idx+1} done: {gen_time:.1f}s -> {sb_audio_duration:.1f}s audio ({sb_rtf:.2f}x RT)")

                except Exception as e:
                    print(f"  Sub-batch {sb_idx+1} failed: {e}")
                    for idx in sb_indices:
                        results["failed"].append((idx, f"Batch error: {e}"))

                self._clear_gpu_cache()

        total_time = time.time() - t_total_start
        rtf = total_audio_duration / total_time if total_time > 0 else 0
        print(f"Batch [lora] total: {total_time:.1f}s -> {total_audio_duration:.1f}s audio ({rtf:.2f}x real-time)")



        return results

    # ── External backend methods ─────────────────────────────────

    def _external_generate_custom(self, text, instruct_text, speaker, voice_config, output_path):
        """Generate custom voice audio via external Gradio server."""
        try:
            voice_data = self._voice_data(speaker, voice_config)
            if not voice_data:
                print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
                return False

            voice = voice_data.get("voice", "Ryan")
            default_style = voice_data.get("default_style", "")
            seed = int(voice_data.get("seed", -1))

            instruct = instruct_text if instruct_text else (default_style if default_style else "neutral")

            print(f"TTS [external] generating with instruct='{instruct}' for text='{text[:50]}...'")

            client = self._init_external()

            result = client.predict(
                text=text,
                language=self._language,
                speaker=voice,
                instruct=instruct,
                model_size="1.7B",
                seed=seed,
                api_name="/generate_custom_voice"
            )

            generated_audio_filepath = result[0]
            if not generated_audio_filepath or not os.path.exists(generated_audio_filepath):
                print(f"Error: No audio file generated for: '{text[:50]}...'")
                return False

            if os.path.getsize(generated_audio_filepath) == 0:
                print(f"Error: Generated audio file is empty for: '{text[:50]}...'")
                return False

            shutil.copy(generated_audio_filepath, output_path)
            return True

        except Exception as e:
            import traceback
            print(f"Error generating custom voice for '{speaker}': {e}")
            traceback.print_exc()
            return False

    def _external_generate_clone(self, text, speaker, voice_config, output_path):
        """Generate voice-cloned audio via external Gradio server."""
        try:
            from gradio_client import handle_file

            voice_data = self._voice_data(speaker, voice_config)
            if not voice_data:
                print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
                return False

            ref_audio = voice_data.get("ref_audio")
            ref_text = voice_data.get("ref_text")
            seed = int(voice_data.get("seed", -1))

            if not ref_audio or not ref_text:
                print(f"Warning: Clone voice for '{speaker}' missing ref_audio or ref_text. Skipping.")
                return False

            # Resolve relative paths against project root
            if not os.path.isabs(ref_audio):
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ref_audio = os.path.join(root_dir, ref_audio)

            if not os.path.exists(ref_audio):
                print(f"Warning: Reference audio not found for '{speaker}': {ref_audio}")
                return False

            client = self._init_external()

            result = client.predict(
                handle_file(ref_audio),
                ref_text,
                text,
                self._language,
                False,       # use_xvector_only
                "1.7B",
                200,         # max_chunk_chars
                0,           # chunk_gap
                seed,
                api_name="/generate_voice_clone"
            )

            generated_audio_filepath = result[0]
            if not generated_audio_filepath or not os.path.exists(generated_audio_filepath):
                print(f"Error: No audio file generated for: '{text[:50]}...'")
                return False

            if os.path.getsize(generated_audio_filepath) == 0:
                print(f"Error: Generated audio file is empty for: '{text[:50]}...'")
                return False

            shutil.copy(generated_audio_filepath, output_path)
            return True

        except Exception as e:
            import traceback
            print(f"Error generating clone voice for '{speaker}': {e}")
            traceback.print_exc()
            return False

    def _sequential_custom(self, chunks, voice_config, output_dir, batch_seed=-1):
        """Sequential custom voice generation for external mode (no native batch)."""
        results = {"completed": [], "failed": []}

        for chunk in chunks:
            idx = chunk["index"]
            output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
            try:
                success = self.generate_custom_voice(
                    chunk.get("text", ""),
                    chunk.get("instruct", ""),
                    chunk.get("speaker", ""),
                    voice_config,
                    output_path,
                )
                if success:
                    results["completed"].append(idx)
                    print(f"Batch chunk {idx} saved: {os.path.getsize(output_path)} bytes")
                else:
                    results["failed"].append((idx, "Custom voice generation failed"))
            except Exception as e:
                results["failed"].append((idx, str(e)))

        return results

    # ── Utility ──────────────────────────────────────────────────

    @staticmethod
    def _save_wav(audio_array, sample_rate, output_path):
        """Save a numpy audio array as a WAV file."""
        # Ensure numpy array
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array)
        # Flatten if needed
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()
        sf.write(output_path, audio_array, sample_rate)
