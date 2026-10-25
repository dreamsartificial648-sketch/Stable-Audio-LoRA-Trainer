@echo off
setlocal
cd /d "%~dp0"
if not exist "runtime\.venv\Scripts\hf.exe" goto missing
"runtime\.venv\Scripts\hf.exe" auth login
pause
exit /b
:missing
echo Install the training runtime first using the app's Setup page.
echo For a custom runtime, run its hf executable with: auth login
pause
exit /b 1

