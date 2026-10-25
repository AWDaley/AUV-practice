@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -c "import torch" >nul 2>&1
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 goto failed
echo.
echo Open http://127.0.0.1:8000 in your browser.
echo Keep this terminal open. Ctrl+C stops the simulator.
".venv\Scripts\python.exe" -m auv_sim
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo Startup failed. See the error above. Install Python 3.11 or 3.12 and try again.
pause
exit /b 1
