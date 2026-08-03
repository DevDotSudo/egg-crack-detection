Egg Crack Detection System — How to Run the Application

1. Open the Project Folder

Open PowerShell inside the main project folder:

cd "C:\Users\Davie\Programming\Flutter Development\egg-crack-detection"

The project should contain at least:

egg-crack-detection
├── backend
├── lib
├── windows
├── pubspec.yaml
├── install_backend.bat
└── run_backend.bat

2. First-Time Backend Installation

Run this only during the first setup, after replacing the backend, or after deleting the virtual environment:

.\install_backend.bat

Wait until the required Python packages are installed successfully.

If the script is inside the backend folder instead, use:

cd backend
.\install_backend.bat
cd ..

3. Install Flutter Dependencies

From the main project folder, run:

flutter pub get

Confirm that Flutter detects Windows desktop support:

flutter devices

The output should include a Windows device.

4. Start the Backend

From the main project folder, run:

.\run_backend.bat

Keep this PowerShell window open while using the application.

The backend should run at:

http://127.0.0.1:8756

To confirm that it is working, open this address in a browser:

http://127.0.0.1:8756/health

A successful response should show that the service is ready.

The Flutter application may also start the backend automatically through BackendLauncherService. Starting it manually first is recommended during development because backend errors remain visible in PowerShell.

5. Run the Flutter Windows Application

Open a second PowerShell window in the same project folder:

cd "C:\Users\Davie\Programming\Flutter Development\egg-crack-detection"

Run the application:

flutter run -d windows

Wait for the Windows desktop application to open.

6. Use the Application

After the application opens:

Confirm that the backend status is connected or ready.

Open Detect to inspect one image.

Open Camera to use live camera inspection.

Open Batch to process several images.

Open History to review saved results.

Open Reports to generate and export the CSV report.

For camera inspection:

Place the egg in the holder.

Keep the camera, egg, light, and enclosure positions consistent.

Check the focus status.

Capture the image.

Review the verdict and overlay.

Save the result to history when needed.

7. Filtering Images

After processing an image, the latest filtering outputs are saved in:

assets\filtering

The folder contains:

original.png
egg_mask.png
dark_response.png
bright_response.png
threshold_mask.png
candidate_components.png
accepted_components.png
final_crack_mask.png
final_overlay.png

Each new detection replaces the previous files.

8. Stop the Application

To stop Flutter, return to the Flutter PowerShell window and press:

Ctrl + C

To stop the backend, return to the backend PowerShell window and press:

Ctrl + C

9. Daily Running Process

After the first-time setup, the normal process is:

PowerShell Window 1 — Backend

cd "C:\Users\Davie\Programming\Flutter Development\egg-crack-detection"
.\run_backend.bat

PowerShell Window 2 — Flutter Application

cd "C:\Users\Davie\Programming\Flutter Development\egg-crack-detection"
flutter run -d windows

10. Common Startup Problems

Port 8756 is already in use

Find the process using the backend port:

$backendPid = (Get-NetTCPConnection -LocalPort 8756 -ErrorAction SilentlyContinue).OwningProcess

Stop it:

Stop-Process -Id $backendPid -Force

Then start the backend again:

.\run_backend.bat

Flutter cannot connect to the backend

Check:

http://127.0.0.1:8756/health

If it does not open, restart the backend.

Flutter packages are missing

Run:

flutter clean
flutter pub get
flutter run -d windows

Python virtual environment is missing

Run:

.\install_backend.bat

Then:

.\run_backend.bat

Quick Start

# Window 1
cd "C:\Users\Davie\Programming\Flutter Development\egg-crack-detection"
.\run_backend.bat

# Window 2
cd "C:\Users\Davie\Programming\Flutter Development\egg-crack-detection"
flutter run -d windows
