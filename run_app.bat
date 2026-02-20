@echo off
echo ========================================================
echo   Email Spam Detector - Single Launch
echo ========================================================
echo.

echo [1/2] Checking dependencies...
pip install -r requirements.txt >nul
echo Dependencies ready.

echo [2/2] Starting application suite...
python start_project.py
pause
