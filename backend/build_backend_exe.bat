@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe call .venv\Scripts\activate.bat
pip install pyinstaller
pyinstaller --noconfirm --clean --onefile --name backend --paths src --collect-all cv2 --collect-all uvicorn --hidden-import app.api.main --hidden-import multipart run_server.py
mkdir dist\backend 2>nul
copy /Y dist\backend.exe dist\backend\backend.exe
