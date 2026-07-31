@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe call .venv\Scripts\activate.bat
pip install pyinstaller
pyinstaller --noconfirm --clean --onefile --name backend --collect-all cv2 --collect-all uvicorn --hidden-import multipart app/main.py
mkdir dist\backend 2>nul
copy /Y dist\backend.exe dist\backend\backend.exe
