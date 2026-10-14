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
    "python-docx",
    "safetensors"
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
    & $venvPip install protobuf==4.25.9
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
    & $venvPip install --no-deps Resemblyzer==0.1.4 silero-vad==6.2.3 mediapipe==0.10.21
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install resemblyzer / silero-vad." -ForegroundColor Red
        exit 1
    }

    Write-Host "[OK] Dependencies installed`n" -ForegroundColor Green
} else {
    Write-Host "[OK] Dependencies verified`n" -ForegroundColor Green
}

# GZ3: L2CS-Net gaze weights (server\models is not in git); downloads once and checks the SHA-256
$l2csWeights = Join-Path $serverDir "models\l2cs\l2cs_gaze360_resnet50.safetensors"
if (-not (Test-Path $l2csWeights)) {
    Write-Host "[INFO] Downloading L2CS-Net gaze weights (~96 MB)..." -ForegroundColor Yellow
    Push-Location $serverDir
    & $venvPython -m video.l2cs --download
    Pop-Location
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] L2CS weights not downloaded; gaze will use the landmark (v2) method." -ForegroundColor Yellow
    }
}

# FAS1: MiniFASNet anti-spoofing weights (downloaded, SHA-1 checked, converted to .safetensors)
$fasDir = Join-Path $serverDir "modelsntispoof"
if (-not (Test-Path (Join-Path $fasDir "2.7_80x80_MiniFASNetV2.safetensors"))) {
    Write-Host "[INFO] Downloading MiniFASNet anti-spoofing weights (~4 MB)..." -ForegroundColor Yellow
    Push-Location $serverDir
    & $venvPython -m video.antispoof --download
    Pop-Location
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] Anti-spoofing weights not downloaded; spoofing will not be checked." -ForegroundColor Yellow
    }
}

# Start server banner
Write-Host "[INFO] Starting backend server..." -ForegroundColor Green
Write-Host "   Server will run on http://localhost:5000" -ForegroundColor Cyan
Write-Host "   Press Ctrl+C to stop`n" -ForegroundColor Yellow

# Run from server\ so the backend modules (config, video, audio, aggregation) import directly.
Set-Location $serverDir
& $venvPython $serverFile
