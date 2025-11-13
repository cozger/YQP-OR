# PowerShell script to attach all USB cameras to WSL2
# Run this in an elevated PowerShell window on Windows

Write-Host "=== USB Camera Attachment Script for WSL2 ===" -ForegroundColor Cyan
Write-Host ""

# Check if running as administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if usbipd is installed
try {
    $null = Get-Command usbipd -ErrorAction Stop
} catch {
    Write-Host "ERROR: usbipd is not installed!" -ForegroundColor Red
    Write-Host "Install from: https://github.com/dorssel/usbipd-win/releases" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Listing USB devices..." -ForegroundColor Yellow
usbipd list

Write-Host ""
Write-Host "Finding camera devices..." -ForegroundColor Yellow

# Get all camera devices
$cameras = usbipd list | Select-String "Camera|Webcam|Video|Imaging"

if ($cameras) {
    foreach ($camera in $cameras) {
        $line = $camera.ToString()
        if ($line -match "^(\d+-\d+)\s+") {
            $busid = $matches[1]

            if ($line -match "Not shared") {
                Write-Host "Binding and attaching camera: $busid" -ForegroundColor Green
                usbipd bind --busid $busid
                Start-Sleep -Seconds 1
                usbipd attach --wsl --busid $busid
                Write-Host "  Camera $busid attached successfully" -ForegroundColor Green
            } elseif ($line -match "Attached") {
                Write-Host "  Camera $busid is already attached" -ForegroundColor Cyan
            }
        }
    }
} else {
    Write-Host "No camera devices found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Camera attachment complete ===" -ForegroundColor Green
Write-Host "You can now run ./run_gui.sh in WSL2" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"