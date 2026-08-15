@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul || (echo Python launcher not found. Install Python 3.11 x64.& pause & exit /b 1)
where ffmpeg >nul 2>nul || echo WARNING: FFmpeg is not in PATH. Install FFmpeg before running the app.
if not exist .venv (
  py -3.11 -m venv .venv || (echo Python 3.11 is required/recommended.& pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip wheel setuptools

echo.
echo [1/2] Installing CUDA-enabled PyTorch 2.11.0 (CUDA 12.6)...
python -m pip install --upgrade --force-reinstall torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
  echo PyTorch CUDA installation failed.
  pause
  exit /b 1
)

echo.
echo [2/2] Installing application dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. Check README.md.
  pause
  exit /b 1
)

echo.
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
echo.
echo Installation complete.
pause
