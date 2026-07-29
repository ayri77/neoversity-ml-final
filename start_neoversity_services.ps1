param(
    [switch]$NoBrowser,
    [switch]$SyncDependencies
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-ListeningPort {
    param([int]$Port)
    try {
        return [bool](Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop)
    }
    catch {
        return $false
    }
}

function ConvertTo-PowerShellLiteral {
    param([string]$Value)
    return "'" + $Value.Replace("'", "''") + "'"
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$requiredFiles = @(
    "pyproject.toml",
    "apps/experiment_control_panel.py",
    "configs/mlflow/local.yaml"
)

foreach ($relativePath in $requiredFiles) {
    if (-not (Test-Path (Join-Path $repoRoot $relativePath))) {
        throw "Run this script from the repository root. Missing: $relativePath"
    }
}

$authorityKey = Join-Path $env:USERPROFILE "churn-ml-secrets/optuna-lifecycle-authority.key"
if (-not (Test-Path $authorityKey -PathType Leaf)) {
    throw "Optuna authority key was not found under the current user profile."
}

if ($SyncDependencies -or -not (Test-Path (Join-Path $repoRoot ".venv/Scripts/python.exe"))) {
    Write-Step "Synchronizing the virtual environment with the UI dependency group"
    uv sync --extra ui
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed with exit code $LASTEXITCODE."
    }
}

$env:CHURN_ML_OPTUNA_LIFECYCLE_AUTHORITY_KEY_FILE = $authorityKey

$shellCommand = Get-Command pwsh -ErrorAction SilentlyContinue
if ($null -eq $shellCommand) {
    $shellCommand = Get-Command powershell -ErrorAction Stop
}

$repoLiteral = ConvertTo-PowerShellLiteral $repoRoot
$keyLiteral = ConvertTo-PowerShellLiteral $authorityKey

if (Test-ListeningPort 8501) {
    Write-Host "Streamlit is already listening on http://localhost:8501" -ForegroundColor Yellow
}
else {
    Write-Step "Starting Streamlit control panel"
    $streamlitCommand = @"
`$Host.UI.RawUI.WindowTitle = 'Neoversity - Streamlit Control Panel'
Set-Location $repoLiteral
`$env:CHURN_ML_OPTUNA_LIFECYCLE_AUTHORITY_KEY_FILE = $keyLiteral
uv run --extra ui streamlit run apps/experiment_control_panel.py
"@
    Start-Process -FilePath $shellCommand.Source -WorkingDirectory $repoRoot -ArgumentList @(
        "-NoExit",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-Command", $streamlitCommand
    ) | Out-Null
}

if (Test-ListeningPort 5000) {
    Write-Host "MLflow is already listening on http://127.0.0.1:5000" -ForegroundColor Yellow
}
else {
    Write-Step "Starting MLflow tracking server"
    $mlflowCommand = @"
`$Host.UI.RawUI.WindowTitle = 'Neoversity - MLflow'
Set-Location $repoLiteral
uv run mlflow server --backend-store-uri "sqlite:///artifacts/mlflow/mlflow.db" --default-artifact-root "artifacts/mlflow/mlartifacts" --host 127.0.0.1 --port 5000
"@
    Start-Process -FilePath $shellCommand.Source -WorkingDirectory $repoRoot -ArgumentList @(
        "-NoExit",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-Command", $mlflowCommand
    ) | Out-Null
}

Write-Step "Waiting briefly for local services"
Start-Sleep -Seconds 4

Write-Host ""
Write-Host "Streamlit: http://localhost:8501" -ForegroundColor Green
Write-Host "MLflow:    http://127.0.0.1:5000" -ForegroundColor Green
Write-Host "Authority key is configured for child processes and is not printed." -ForegroundColor DarkGray

if (-not $NoBrowser) {
    Start-Process "http://localhost:8501"
    Start-Process "http://127.0.0.1:5000"
}

Write-Host ""
Write-Step "Recent UI jobs"
$jobsRoot = Join-Path $repoRoot "artifacts/ui_jobs"
if (Test-Path $jobsRoot) {
    Get-ChildItem $jobsRoot -Directory |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 5 Name, LastWriteTime |
        Format-Table -AutoSize
}
else {
    Write-Host "No UI job directory exists yet."
}

Write-Step "Recent Optuna search artifacts"
$searchRoot = Join-Path $repoRoot "artifacts/optuna_searches"
if (Test-Path $searchRoot) {
    Get-ChildItem $searchRoot -Directory |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 5 Name, LastWriteTime |
        Format-Table -AutoSize
}
else {
    Write-Host "No Optuna search artifact directory exists yet."
}
