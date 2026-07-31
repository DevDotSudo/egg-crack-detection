$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
$runnerDirectory = Join-Path $projectRoot 'windows\runner'
$flutterWindow = Join-Path $runnerDirectory 'flutter_window.cpp'
$runnerCMake = Join-Path $runnerDirectory 'CMakeLists.txt'
$bridgeHeader = Join-Path $runnerDirectory 'camera_focus_bridge.h'
$bridgeSource = Join-Path $runnerDirectory 'camera_focus_bridge.cpp'

$pubspec = Join-Path $projectRoot 'pubspec.yaml'
if (Test-Path $pubspec) {
    $pubspecLines = Get-Content $pubspec
    $cleanedPubspecLines = $pubspecLines | Where-Object {
        $_ -notmatch '^\s*(opencv_dart|opencv_flutter|opencv)\s*:'
    }
    Set-Content -Path $pubspec -Value $cleanedPubspecLines -Encoding UTF8
}

if (-not (Test-Path $flutterWindow) -or -not (Test-Path $runnerCMake)) {
    Write-Host 'Windows runner files were not found.' -ForegroundColor Red
    Write-Host 'Run this from the Flutter project root first:' -ForegroundColor Yellow
    Write-Host 'flutter create --platforms=windows .' -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $bridgeHeader) -or -not (Test-Path $bridgeSource)) {
    Write-Host 'The camera autofocus bridge files are missing.' -ForegroundColor Red
    exit 1
}

$flutterWindowText = Get-Content $flutterWindow -Raw

if ($flutterWindowText -notmatch '#include\s+"camera_focus_bridge\.h"') {
    $includeReplacement = '#include "flutter_window.h"' + [Environment]::NewLine + '#include "camera_focus_bridge.h"'
    $flutterWindowText = $flutterWindowText -replace '#include\s+"flutter_window\.h"', $includeReplacement
}

if ($flutterWindowText -notmatch 'RegisterCameraFocusBridge\s*\(') {
    $registrationPattern = 'RegisterPlugins\(flutter_controller_->engine\(\)\);'
    $registrationReplacement = "RegisterPlugins(flutter_controller_->engine());`r`n  RegisterCameraFocusBridge(flutter_controller_->engine()->messenger());"

    if ($flutterWindowText -notmatch $registrationPattern) {
        Write-Host 'Could not find RegisterPlugins in flutter_window.cpp.' -ForegroundColor Red
        exit 1
    }

    $flutterWindowText = [regex]::Replace(
        $flutterWindowText,
        $registrationPattern,
        $registrationReplacement,
        1
    )
}

Set-Content -Path $flutterWindow -Value $flutterWindowText -Encoding UTF8

$cmakeText = Get-Content $runnerCMake -Raw

if ($cmakeText -notmatch 'camera_focus_bridge\.cpp') {
    $sourcePattern = 'add_executable\(\$\{BINARY_NAME\}\s+WIN32\s*\r?\n'
    $sourceReplacement = 'add_executable(${BINARY_NAME} WIN32' + [Environment]::NewLine + '  "camera_focus_bridge.cpp"' + [Environment]::NewLine + '  "camera_focus_bridge.h"' + [Environment]::NewLine

    if ($cmakeText -notmatch $sourcePattern) {
        Write-Host 'Could not find the runner source list in CMakeLists.txt.' -ForegroundColor Red
        exit 1
    }

    $cmakeText = [regex]::Replace(
        $cmakeText,
        $sourcePattern,
        $sourceReplacement,
        1
    )
}

Set-Content -Path $runnerCMake -Value $cmakeText -Encoding UTF8

Write-Host 'Logitech C525 autofocus bridge installed.' -ForegroundColor Green
Write-Host 'Frontend OpenCV dependencies removed from pubspec when present.' -ForegroundColor Green
Write-Host 'Run: flutter clean' -ForegroundColor Cyan
Write-Host 'Run: flutter pub get' -ForegroundColor Cyan
Write-Host 'Run: flutter run -d windows' -ForegroundColor Cyan
