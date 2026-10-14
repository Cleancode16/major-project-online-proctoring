# Start Frontend for Online Proctoring System
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Online Proctoring - Frontend" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Resolve project directories from the script location so this works from any cwd.
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendDir = Join-Path $projectRoot "client\proctoring"

if (-Not (Test-Path $frontendDir)) {
    Write-Host "[ERROR] Frontend directory not found: $frontendDir" -ForegroundColor Red
    exit 1
}

Set-Location $frontendDir

# Check if node_modules exists
if (-Not (Test-Path ".\node_modules")) {
    Write-Host "[WARN] Node modules not found. Installing..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] npm install failed." -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Node modules installed`n" -ForegroundColor Green
} else {
    Write-Host "[OK] Node modules already installed`n" -ForegroundColor Green
}

# Ensure key Vite plugin exists even if node_modules is partially broken.
npm ls @vitejs/plugin-react --depth=0 *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] Missing @vitejs/plugin-react. Installing..." -ForegroundColor Yellow
    npm install -D @vitejs/plugin-react
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install @vitejs/plugin-react." -ForegroundColor Red
        exit 1
    }
}

# Start development server
Write-Host "[INFO] Starting frontend development server..." -ForegroundColor Green
Write-Host "   Frontend will run on http://localhost:5173" -ForegroundColor Cyan
Write-Host "   Press Ctrl+C to stop`n" -ForegroundColor Yellow
npm run dev
