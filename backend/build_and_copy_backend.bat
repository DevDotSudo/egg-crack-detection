@echo off
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  py -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller

python -m PyInstaller --noconfirm --clean --onefile --name backend --paths src --collect-all cv2 --collect-all uvicorn --hidden-import app.api.main --hidden-import multipart run_server.py

set "RELEASE_DIR=%~dp0..\frontend\build\windows\x64\runner\Release"
if not exist "%RELEASE_DIR%" (
  echo.
  echo Flutter release folder was not found.
  echo Run this first inside the frontend folder:
  echo flutter build windows
  echo.
  pause
  exit /b 1
)

if not exist "%RELEASE_DIR%\backend" mkdir "%RELEASE_DIR%\backend"
copy /Y "%~dp0dist\backend.exe" "%RELEASE_DIR%\backend\backend.exe"

echo.
echo Backend created and copied successfully.
echo Location: %RELEASE_DIR%\backend\backend.exe
pause
