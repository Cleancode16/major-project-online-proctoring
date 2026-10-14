# Start the whole Online Proctoring System: backend (server\) and frontend (client\proctoring)
# Each runs in its own PowerShell window (via start-backend.ps1 / start-frontend.ps1) so their logs stay separate.
# Usage (from anywhere):   powershell -ExecutionPolicy Bypass -File .\start-all.ps1
# Stop: close the two windows (or press Ctrl+C in each).

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendScript = Join-Path $projectRoot "start-backend.ps1"
$frontendScript = Join-Path $projectRoot "start-frontend.ps1"
$backendUrl = "http://localhost:5000/api/health"
$frontendUrl = "http://localhost:5173"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Online Proctoring - start all" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

foreach ($script in @($backendScript, $frontendScript)) {
    if (-Not (Test-Path $script)) {
        Write-Host "[ERROR] Missing $script" -ForegroundColor Red
        exit 1
    }
}

# 1) Backend: installs missing Python packages on first run, then serves http://localhost:5000
Write-Host "[INFO] Starting backend in a new window..." -ForegroundColor Yellow
Start-Process powershell -WorkingDirectory $projectRoot -ArgumentList @(
    "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$backendScript`""
)

# 2) Frontend: runs npm install if node_modules is missing, then the Vite dev server on http://localhost:5173
Write-Host "[INFO] Starting frontend in a new window..." -ForegroundColor Yellow
Start-Process powershell -WorkingDirectory $projectRoot -ArgumentList @(
    "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$frontendScript`""
)

# 3) Wait for the backend (model loading / first-run installs can take a few minutes), then open the app
Write-Host "[INFO] Waiting for the backend at $backendUrl ..." -ForegroundColor Yellow
$deadline = (Get-Date).AddMinutes(15)
$ready = $false
while ((Get-Date) -lt $deadline) {
    try {
        $response = Invoke-WebRequest -Uri $backendUrl -UseBasicParsing -TimeoutSec 3
        if ($response.StatusCode -eq 200) { $ready = $true; break }
    } catch {
        Start-Sleep -Seconds 3
    }
}

if ($ready) {
    Write-Host "[OK] Backend is up. Opening $frontendUrl" -ForegroundColor Green
    Start-Process $frontendUrl
} else {
    Write-Host "[WARN] Backend did not answer within 15 minutes. Check the backend window for errors," -ForegroundColor Yellow
    Write-Host "       then open $frontendUrl yourself." -ForegroundColor Yellow
}
