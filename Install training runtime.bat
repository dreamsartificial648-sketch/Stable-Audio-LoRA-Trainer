@echo off
setlocal
cd /d "%~dp0"
python scripts\install_runtime.py
if errorlevel 1 goto failed
pause
exit /b 0
:failed
echo Runtime installation failed. Review the error above.
pause
exit /b 1

