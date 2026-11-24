@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup_and_run.bat once first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
streamlit run app.py
pause
