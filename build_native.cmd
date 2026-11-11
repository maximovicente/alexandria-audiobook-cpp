@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "SOURCE=%ROOT%native\qwentts-cpp"
set "BUILD=%SOURCE%\build"
set "CC="
set "CXX="

if not exist "%SOURCE%\CMakeLists.txt" (
  echo [Native] qwentts.cpp source is missing: %SOURCE%
  exit /b 1
)

where vswhere >nul 2>nul
if errorlevel 1 set "PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\Installer"
where vswhere >nul 2>nul
if errorlevel 1 (
  echo [Native] vswhere.exe is required to locate the MSVC toolchain.
  exit /b 1
)

for /f "usebackq tokens=*" %%I in (`vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%I"
if not defined VSROOT (
  echo [Native] Visual Studio C++ Build Tools were not found.
  exit /b 1
)

call "%VSROOT%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b 1

where cmake >nul 2>nul
if errorlevel 1 (
  echo [Native] cmake is required.
  exit /b 1
)
where ninja >nul 2>nul
if errorlevel 1 (
  echo [Native] ninja is required.
  exit /b 1
)
where nvcc >nul 2>nul
if errorlevel 1 (
  echo [Native] CUDA nvcc is required for the native CUDA build.
  exit /b 1
)

cmake -S "%SOURCE%" -B "%BUILD%" -G "Ninja Multi-Config" -DGGML_CUDA=ON
if errorlevel 1 exit /b 1
cmake --build "%BUILD%" --config Release --parallel
if errorlevel 1 exit /b 1

echo [Native] qwen-tts.cpp build completed.
exit /b 0
