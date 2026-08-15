@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo .venv not found. Run install_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade --force-reinstall torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
pause
