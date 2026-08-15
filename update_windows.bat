@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo .venv not found. Run install_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency update failed.
  pause
  exit /b 1
)
echo.
python doctor.py
echo.
echo Voice Dataset Forge dependencies updated.
pause
