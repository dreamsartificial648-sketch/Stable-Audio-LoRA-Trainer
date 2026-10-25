@echo off
setlocal
cd /d "%~dp0"
python scripts\install_runtime.py --ui-only
if errorlevel 1 goto failed
echo.
echo Desktop ready. Open Launch.bat, then install the training runtime from Setup.
pause
exit /b 0
:failed
echo.
echo Setup failed. Install Python 3.10, 3.11 or 3.12 with Add Python to PATH enabled.
pause
exit /b 1

