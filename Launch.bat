@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" main.py %*
if errorlevel 1 pause
exit /b
:missing
echo Run Setup.bat first to install the desktop app.
pause
exit /b 1

