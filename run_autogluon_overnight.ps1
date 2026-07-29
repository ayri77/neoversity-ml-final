$ErrorActionPreference = "Stop"

$python = ".venv-autogluon\Scripts\python.exe"
$runner = "scripts\run_autogluon.py"
$gpuConfig = "configs\autogluon\realtabpfn_overnight_gpu_5h.yaml"
$cpuConfig = "configs\autogluon\catboost_overnight_cpu_3h.yaml"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"

Write-Host "Validating AutoGluon overnight configs..."
& $python -u $runner validate --config $gpuConfig
if ($LASTEXITCODE -ne 0) {
    throw "RealTabPFN config validation failed with exit code $LASTEXITCODE."
}

& $python -u $runner validate --config $cpuConfig
if ($LASTEXITCODE -ne 0) {
    throw "CatBoost config validation failed with exit code $LASTEXITCODE."
}

$gpuRunId = "ag-v3-realtabpfn-gpu-s42-$stamp"
Write-Host "Starting RealTabPFN GPU run: $gpuRunId"
& $python -u $runner train `
    --config $gpuConfig `
    --run-id $gpuRunId

$gpuExit = $LASTEXITCODE
Write-Host "RealTabPFN run finished with exit code $gpuExit."

$cpuRunId = "ag-v3-catboost-cpu-s42-$stamp"
Write-Host "Starting CatBoost CPU fallback/follow-up run: $cpuRunId"
& $python -u $runner train `
    --config $cpuConfig `
    --run-id $cpuRunId

$cpuExit = $LASTEXITCODE
Write-Host "CatBoost run finished with exit code $cpuExit."

Write-Host ""
Write-Host "Run directories:"
Write-Host "  artifacts\autogluon_runs\$gpuRunId"
Write-Host "  artifacts\autogluon_runs\$cpuRunId"

if ($gpuExit -ne 0 -and $cpuExit -ne 0) {
    exit 1
}
exit 0
