@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo   EcoSort - Smart Waste Classification Assistant
echo ================================================
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or 3.11 and try again.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 goto :error
)
call .venv\Scripts\activate.bat
echo Installing required packages...
python -m pip install --upgrade pip
if errorlevel 1 goto :error
pip install -r requirements.txt
if errorlevel 1 goto :error
echo.
echo Starting EcoSort in your browser...
echo The first classification downloads the pretrained model and needs internet.
streamlit run app.py
if errorlevel 1 goto :error
goto :eof
:error
echo.
echo Setup did not finish. Read README.md for troubleshooting.
pause
