@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Run install_backend.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" run_server.py
pause
