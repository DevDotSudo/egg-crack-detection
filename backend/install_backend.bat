@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
py -3.11 -m venv .venv
if errorlevel 1 py -3.13 -m venv .venv
if errorlevel 1 py -3 -m venv .venv
:install
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
echo Backend dependencies installed.
pause
