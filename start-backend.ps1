# Start Backend Server for Online Proctoring System
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Online Proctoring - Backend Server" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Resolve project paths from script location so this works from any cwd.
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvActivate = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"
$venvPip = Join-Path $projectRoot ".venv\Scripts\pip.exe"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$serverDir = Join-Path $projectRoot "server"
$requirementsFile = Join-Path $serverDir "requirements.txt"
$serverFile = Join-Path $serverDir "server.py"

Set-Location $projectRoot

# Suppress TensorFlow oneDNN/absl startup warnings (harmless log noise).
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:TF_ENABLE_ONEDNN_OPTS = "0"

# Check if virtual environment exists
if (-Not (Test-Path $venvActivate)) {
    # Fix: use ASCII-only string text to avoid quote terminator issues from encoding glitches.
    Write-Host "[WARN] Virtual environment not found. Creating..." -ForegroundColor Yellow
    py -3.10 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Virtual environment created`n" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Yellow
& $venvActivate

# Check if requirements are installed in .venv
Write-Host "[INFO] Checking dependencies..." -ForegroundColor Yellow
$requiredPackages = @(
    "flask-cors",
    "aiortc",
    "mediapipe",
    "keras-facenet",
    "ultralytics",
    "tensorflow",
    "tensorflow-hub",
    "resemblyzer",
    "silero-vad",
    "reportlab",
    "python-docx"
)

$missingPackages = @()
foreach ($package in $requiredPackages) {
    & $venvPip show $package *> $null
    if ($LASTEXITCODE -ne 0) {
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host "[WARN] Dependencies not installed. Installing with correct order..." -ForegroundColor Yellow
    Write-Host "[INFO] Missing packages: $($missingPackages -join ', ')" -ForegroundColor Yellow

    # Install protobuf FIRST to avoid version conflicts
    & $venvPip install protobuf==3.20.3
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install protobuf. Aborting startup." -ForegroundColor Red
        exit 1
    }

    # Then install everything else (pinned list in server\requirements.txt)
    & $venvPip install -r $requirementsFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Dependency installation did not complete. Please rerun .\start-backend.ps1" -ForegroundColor Red
        exit 1
    }

    # Speaker encoder + VAD without their dependency lists, so they cannot move the TF/numpy pins
    & $venvPip install --no-deps Resemblyzer==0.1.4 silero-vad==6.2.3
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install resemblyzer / silero-vad." -ForegroundColor Red
        exit 1
    }

    Write-Host "[OK] Dependencies installed`n" -ForegroundColor Green
} else {
    Write-Host "[OK] Dependencies verified`n" -ForegroundColor Green
}

# Start server banner
Write-Host "[INFO] Starting backend server..." -ForegroundColor Green
Write-Host "   Server will run on http://localhost:5000" -ForegroundColor Cyan
Write-Host "   Press Ctrl+C to stop`n" -ForegroundColor Yellow

# Run from server\ so the backend modules (config, video, audio, aggregation) import directly.
Set-Location $serverDir
& $venvPython $serverFile
