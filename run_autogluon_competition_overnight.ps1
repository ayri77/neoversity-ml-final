#requires -Version 5.1
<#
.SYNOPSIS
  Overnight competition sequence: RealTabPFN (5h GPU) then TabM (3h GPU).

.DESCRIPTION
  Validates configs, records diagnostics, then trains two independent AutoGluon
  runs with separate run IDs and OOF artifacts. Does not start Neoversity
  services. A RealTabPFN failure does not skip TabM.

.PARAMETER ValidateOnly
  Run preflight checks and config validation only. Does not allocate runs or
  start training.
#>
[CmdletBinding()]
param(
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
# PowerShell 7.3+ can treat native non-zero exits as terminating errors.
# Keep RealTabPFN failures from aborting before TabM starts.
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$RepoRoot = $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv-autogluon\Scripts\python.exe"
$Runner = Join-Path $RepoRoot "scripts\run_autogluon.py"
$RealTabPfnConfig = Join-Path $RepoRoot "configs\autogluon\realtabpfn_overnight_gpu_5h.yaml"
$TabmConfig = Join-Path $RepoRoot "configs\autogluon\tabm_overnight_gpu_3h.yaml"
$TrainFeatures = Join-Path $RepoRoot "data\processed\v3_targeted_missingness\X_train.parquet"
$TrainTarget = Join-Path $RepoRoot "data\processed\v3_targeted_missingness\y_train.parquet"
$ArtifactsRoot = Join-Path $RepoRoot "artifacts\autogluon_runs"

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$OvernightRoot = Join-Path $RepoRoot ("artifacts\autogluon_overnight\" + $Stamp)
$TranscriptPath = Join-Path $OvernightRoot "overnight.log"
$SummaryPath = Join-Path $OvernightRoot "summary.json"

$RealTabPfnRunId = "ag-v3-realtabpfn-gpu-s42-$Stamp"
$TabmRunId = "ag-v3-tabm-gpu-s42-$Stamp"
$RealTabPfnRunDir = Join-Path $ArtifactsRoot $RealTabPfnRunId
$TabmRunDir = Join-Path $ArtifactsRoot $TabmRunId

$TranscriptStarted = $false
$SleepPreventionEnabled = $false
$ExitCode = 2
$StartTime = (Get-Date).ToUniversalTime().ToString("o")
$RealTabPfnExit = $null
$TabmExit = $null
$OverallStatus = "preflight_failed"
$Diagnostics = [ordered]@{
    system_ram_gb = $null
    gpu_memory = $null
    diagnostic_errors = @()
}

function Write-Info {
    param([string]$Message)
    Write-Host $Message
}

function Get-SystemRamDiagnostics {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $totalBytes = [double]$os.TotalVisibleMemorySize * 1024.0
    $freeBytes = [double]$os.FreePhysicalMemory * 1024.0
    return [ordered]@{
        total_gb = [math]::Round($totalBytes / 1GB, 2)
        free_gb = [math]::Round($freeBytes / 1GB, 2)
        available_gb = [math]::Round($freeBytes / 1GB, 2)
    }
}

function Get-GpuMemoryDiagnostics {
    $nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        return [ordered]@{ available = $false; reason = "nvidia-smi not found" }
    }
    $raw = & nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv,noheader
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi exited with code $LASTEXITCODE"
    }
    $gpus = @()
    foreach ($line in @($raw)) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        $parts = @($line -split ",\s*")
        if ($parts.Count -lt 5) {
            continue
        }
        $gpus += [ordered]@{
            index = $parts[0].Trim()
            name = $parts[1].Trim()
            memory_total = $parts[2].Trim()
            memory_free = $parts[3].Trim()
            memory_used = $parts[4].Trim()
        }
    }
    return [ordered]@{ available = $true; gpus = $gpus }
}

function Enable-SleepPrevention {
    if (-not ("NeoversitySleepPreventer" -as [type])) {
        Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class NeoversitySleepPreventer {
    public const uint ES_SYSTEM_REQUIRED = 0x00000001;
    public const uint ES_CONTINUOUS = 0x80000000;
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
    }
    $flags = [NeoversitySleepPreventer]::ES_CONTINUOUS -bor [NeoversitySleepPreventer]::ES_SYSTEM_REQUIRED
    [void][NeoversitySleepPreventer]::SetThreadExecutionState($flags)
}

function Disable-SleepPrevention {
    if ("NeoversitySleepPreventer" -as [type]) {
        [void][NeoversitySleepPreventer]::SetThreadExecutionState(
            [NeoversitySleepPreventer]::ES_CONTINUOUS
        )
    }
}

function Write-SummaryJson {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        $Payload
    )
    $json = $Payload | ConvertTo-Json -Depth 8
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($Path, $json, $utf8NoBom)
}

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
    }
}

try {
    New-Item -ItemType Directory -Path $OvernightRoot -Force | Out-Null
    Start-Transcript -Path $TranscriptPath -Force | Out-Null
    $TranscriptStarted = $true

    Write-Info "Competition overnight orchestration stamp: $Stamp"
    Write-Info "Overnight directory: $OvernightRoot"

    Assert-PathExists -Path $Python -Label "AutoGluon Python executable"
    Assert-PathExists -Path $Runner -Label "AutoGluon runner"
    Assert-PathExists -Path $RealTabPfnConfig -Label "RealTabPFN config"
    Assert-PathExists -Path $TabmConfig -Label "TabM config"
    Assert-PathExists -Path $TrainFeatures -Label "Processed train features"
    Assert-PathExists -Path $TrainTarget -Label "Processed train target"

    try {
        $Diagnostics.system_ram_gb = Get-SystemRamDiagnostics
        Write-Info (
            "System RAM free/total GB: " +
            "$($Diagnostics.system_ram_gb.free_gb)/$($Diagnostics.system_ram_gb.total_gb)"
        )
    }
    catch {
        $Diagnostics.diagnostic_errors = @(
            $Diagnostics.diagnostic_errors + @("system_ram: $($_.Exception.Message)")
        )
        Write-Info "WARNING: could not collect system RAM diagnostics: $($_.Exception.Message)"
    }

    try {
        $Diagnostics.gpu_memory = Get-GpuMemoryDiagnostics
        Write-Info ("GPU diagnostics: " + ($Diagnostics.gpu_memory | ConvertTo-Json -Compress -Depth 6))
    }
    catch {
        $Diagnostics.diagnostic_errors = @(
            $Diagnostics.diagnostic_errors + @("gpu_memory: $($_.Exception.Message)")
        )
        Write-Info "WARNING: could not collect GPU diagnostics: $($_.Exception.Message)"
    }

    Write-Info "Validating RealTabPFN config..."
    & $Python -u $Runner validate --config $RealTabPfnConfig
    if ($LASTEXITCODE -ne 0) {
        throw "RealTabPFN config validation failed with exit code $LASTEXITCODE."
    }

    Write-Info "Validating TabM config..."
    & $Python -u $Runner validate --config $TabmConfig
    if ($LASTEXITCODE -ne 0) {
        throw "TabM config validation failed with exit code $LASTEXITCODE."
    }

    if ($ValidateOnly) {
        $OverallStatus = "validate_only_ok"
        $ExitCode = 0
        Write-Info "ValidateOnly succeeded; no runs allocated."
    }
    else {
        if (Test-Path -LiteralPath $RealTabPfnRunDir) {
            throw "RealTabPFN run directory already exists: $RealTabPfnRunDir"
        }
        if (Test-Path -LiteralPath $TabmRunDir) {
            throw "TabM run directory already exists: $TabmRunDir"
        }

        Enable-SleepPrevention
        $SleepPreventionEnabled = $true
        Write-Info "Sleep prevention enabled for training."

        Write-Info "Starting RealTabPFN GPU run: $RealTabPfnRunId"
        & $Python -u $Runner train --config $RealTabPfnConfig --run-id $RealTabPfnRunId
        $RealTabPfnExit = $LASTEXITCODE
        Write-Info "RealTabPFN run finished with exit code $RealTabPfnExit."

        Write-Info "Starting TabM GPU run: $TabmRunId"
        & $Python -u $Runner train --config $TabmConfig --run-id $TabmRunId
        $TabmExit = $LASTEXITCODE
        Write-Info "TabM run finished with exit code $TabmExit."

        $realOk = ($null -ne $RealTabPfnExit) -and ($RealTabPfnExit -eq 0)
        $tabmOk = ($null -ne $TabmExit) -and ($TabmExit -eq 0)
        if ($realOk -and $tabmOk) {
            $OverallStatus = "both_succeeded"
            $ExitCode = 0
        }
        elseif ($realOk -or $tabmOk) {
            $OverallStatus = "partial_success"
            $ExitCode = 0
        }
        else {
            $OverallStatus = "both_failed"
            $ExitCode = 1
        }
    }
}
catch {
    Write-Info "ERROR: $($_.Exception.Message)"
    if ($OverallStatus -ne "validate_only_ok") {
        $OverallStatus = "preflight_or_validation_failed"
    }
    $ExitCode = 2
}
finally {
    if ($SleepPreventionEnabled) {
        try {
            Disable-SleepPrevention
            Write-Info "Sleep prevention restored."
        }
        catch {
            Write-Info "WARNING: failed to restore sleep prevention: $($_.Exception.Message)"
        }
    }

    $endTime = (Get-Date).ToUniversalTime().ToString("o")
    $summary = [ordered]@{
        schema_version = 1
        mode = $(if ($ValidateOnly) { "validate_only" } else { "train" })
        start_time_utc = $StartTime
        end_time_utc = $endTime
        stamp = $Stamp
        overnight_directory = $OvernightRoot
        configs = [ordered]@{
            realtabpfn = $RealTabPfnConfig
            tabm = $TabmConfig
        }
        run_ids = [ordered]@{
            realtabpfn = $RealTabPfnRunId
            tabm = $TabmRunId
        }
        run_directories = [ordered]@{
            realtabpfn = $RealTabPfnRunDir
            tabm = $TabmRunDir
        }
        exit_codes = [ordered]@{
            realtabpfn = $RealTabPfnExit
            tabm = $TabmExit
            orchestrator = $ExitCode
        }
        overall_status = $OverallStatus
        diagnostics = $Diagnostics
        transcript = $TranscriptPath
    }

    try {
        if (-not (Test-Path -LiteralPath $OvernightRoot)) {
            New-Item -ItemType Directory -Path $OvernightRoot -Force | Out-Null
        }
        Write-SummaryJson -Path $SummaryPath -Payload $summary
    }
    catch {
        Write-Info "WARNING: failed to write summary JSON: $($_.Exception.Message)"
    }

    Write-Host ""
    Write-Host "Summary path: $SummaryPath"
    Write-Host "RealTabPFN run directory: $RealTabPfnRunDir"
    Write-Host "TabM run directory: $TabmRunDir"
    Write-Host "Overall status: $OverallStatus"
    Write-Host "Orchestrator exit code: $ExitCode"

    if ($TranscriptStarted) {
        try {
            Stop-Transcript | Out-Null
        }
        catch {
            # Transcript may already be stopped after Ctrl+C; ignore.
        }
    }
}

exit $ExitCode
