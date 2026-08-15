@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Virtual environment not found. Run install_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python main.py
