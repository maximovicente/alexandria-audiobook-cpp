module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: [
        "if not exist native\\qwentts-cpp\\CMakeLists.txt git clone --recurse-submodules https://github.com/ServeurpersoCom/qwentts.cpp.git native\\qwentts-cpp",
        "git -C native\\qwentts-cpp fetch --depth 1 origin de2ad83",
        "git -C native\\qwentts-cpp checkout --force de2ad83",
        "git -C native\\qwentts-cpp submodule update --init --recursive"
      ]
    }
  }, {
    method: "shell.run",
    params: {
      message: "uv cache clean"
    }
  }, {
    method: "shell.run",
    params: {
      path: "app",
      message: "python -m venv env"
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "uv pip uninstall google-genai",
        "uv pip install -r requirements.txt",
        "uv pip install qwen-tts==0.1.1"
      ]
    }
  }, {
    method: "script.start",
    params: {
      uri: "torch.js",
      params: {
        path: "app",
        venv: "env",
        flashattention: true
      }
    }
  }, {
    method: "shell.run",
    params: {
      path: ".",
      message: "if exist native\\qwentts-cpp\\build\\Release\\qwen-tts.exe (echo [Native] Using existing precompiled qwen-tts.cpp binaries) else (cmd /c build_native.cmd)"
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "python -c \"from huggingface_hub import hf_hub_download; import os; d='../native/qwentts-cpp/models'; os.makedirs(d, exist_ok=True); hf_hub_download(repo_id='Serveurperso/Qwen3-TTS-GGUF', filename='qwen-tokenizer-12hz-Q8_0.gguf', local_dir=d); hf_hub_download(repo_id='Serveurperso/Qwen3-TTS-GGUF', filename='qwen-talker-1.7b-customvoice-Q8_0.gguf', local_dir=d); hf_hub_download(repo_id='Serveurperso/Qwen3-TTS-GGUF', filename='qwen-talker-1.7b-base-Q8_0.gguf', local_dir=d)\""
      ]
    }
  }, {
    method: "notify",
    params: {
      html: "Installation Complete! Click 'Start' to launch the application."
    }
  }]
}
