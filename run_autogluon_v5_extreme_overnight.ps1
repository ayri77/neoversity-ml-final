#requires -Version 5.1

<#
.SYNOPSIS
  Run one AutoGluon extreme-quality experiment on
  v5_joint_missingness_pattern.

.DESCRIPTION
  Performs:
  1. Preflight path and resource checks.
  2. AutoGluon config validation.
  3. One supervised AutoGluon training run.
  4. Read-only inspection with predictor loading.
  5. Durable transcript and summary JSON creation.

.PARAMETER ValidateOnly
  Run preflight and config validation without allocating a training run.

.PARAMETER MinimumFreeRamGb
  Minimum free system RAM required before starting training.
#>

[CmdletBinding()]
param(
    [switch]$ValidateOnly,

    [ValidateRange(1, 128)]
    [double]$MinimumFreeRamGb = 20.0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# PowerShell 7.3+ may treat native non-zero exit codes as terminating errors.
# We handle native exit codes explicitly.
if (
    Get-Variable `
        -Name PSNativeCommandUseErrorActionPreference `
        -ErrorAction SilentlyContinue
) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$RepoRoot = $PSScriptRoot

$Python = Join-Path `
    $RepoRoot `
    ".venv-autogluon\Scripts\python.exe"

$Runner = Join-Path `
    $RepoRoot `
    "scripts\run_autogluon.py"

$Config = Join-Path `
    $RepoRoot `
    "configs\autogluon\extreme_seqmem_v5.yaml"

$TrainFeatures = Join-Path `
    $RepoRoot `
    "data\processed\v5_joint_missingness_pattern\X_train.parquet"

$TrainTarget = Join-Path `
    $RepoRoot `
    "data\processed\v5_joint_missingness_pattern\y_train.parquet"

$ArtifactsRoot = Join-Path `
    $RepoRoot `
    "artifacts\autogluon_runs"

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"

$RunId = "ag-v5-extreme-seqmem-s42-$Stamp"
$RunDir = Join-Path $ArtifactsRoot $RunId

$OrchestrationRoot = Join-Path `
    $RepoRoot `
    "artifacts\autogluon_overnight\v5-extreme-$Stamp"

$TranscriptPath = Join-Path `
    $OrchestrationRoot `
    "overnight.log"

$SummaryPath = Join-Path `
    $OrchestrationRoot `
    "summary.json"

$InspectionPath = Join-Path `
    $OrchestrationRoot `
    "inspection.json"

$TranscriptStarted = $false
$SleepPreventionEnabled = $false

$StartTimeUtc = (Get-Date).ToUniversalTime().ToString("o")
$EndTimeUtc = $null

$ValidateExitCode = $null
$TrainExitCode = $null
$InspectExitCode = $null

$OverallStatus = "not_started"
$ExitCode = 2

$Diagnostics = [ordered]@{
    system_ram = $null
    gpu = $null
    diagnostic_errors = @()
}

function Write-Info {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Message
    )

    Write-Host $Message
}

function Assert-PathExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
    }
}

function Get-SystemRamDiagnostics {
    $os = Get-CimInstance `
        -ClassName Win32_OperatingSystem

    $totalBytes = (
        [double]$os.TotalVisibleMemorySize * 1024.0
    )

    $freeBytes = (
        [double]$os.FreePhysicalMemory * 1024.0
    )

    return [ordered]@{
        total_gb = [math]::Round(
            $totalBytes / 1GB,
            2
        )
        free_gb = [math]::Round(
            $freeBytes / 1GB,
            2
        )
        used_gb = [math]::Round(
            ($totalBytes - $freeBytes) / 1GB,
            2
        )
        free_percent = [math]::Round(
            ($freeBytes / $totalBytes) * 100.0,
            2
        )
    }
}

function Get-GpuDiagnostics {
    $nvidiaSmi = Get-Command `
        "nvidia-smi" `
        -ErrorAction SilentlyContinue

    if ($null -eq $nvidiaSmi) {
        return [ordered]@{
            available = $false
            reason = "nvidia-smi not found"
            gpus = @()
        }
    }

    $raw = & nvidia-smi `
        --query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu `
        --format=csv,noheader

    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi exited with code $LASTEXITCODE"
    }

    $gpus = @()

    foreach ($line in @($raw)) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }

        $parts = @($line -split ",\s*")

        if ($parts.Count -lt 6) {
            continue
        }

        $gpus += [ordered]@{
            index = $parts[0].Trim()
            name = $parts[1].Trim()
            memory_total = $parts[2].Trim()
            memory_free = $parts[3].Trim()
            memory_used = $parts[4].Trim()
            utilization = $parts[5].Trim()
        }
    }

    return [ordered]@{
        available = $true
        gpus = $gpus
    }
}

function Enable-SleepPrevention {
    if (-not ("NeoversitySleepPreventer" -as [type])) {
        Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class NeoversitySleepPreventer
{
    public const uint ES_SYSTEM_REQUIRED = 0x00000001;
    public const uint ES_CONTINUOUS = 0x80000000;

    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(
        uint esFlags
    );
}
"@
    }

    $flags = (
        [NeoversitySleepPreventer]::ES_CONTINUOUS `
        -bor `
        [NeoversitySleepPreventer]::ES_SYSTEM_REQUIRED
    )

    [void][NeoversitySleepPreventer]::SetThreadExecutionState(
        $flags
    )
}

function Disable-SleepPrevention {
    if ("NeoversitySleepPreventer" -as [type]) {
        [void][NeoversitySleepPreventer]::SetThreadExecutionState(
            [NeoversitySleepPreventer]::ES_CONTINUOUS
        )
    }
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        $Payload
    )

    $json = $Payload |
        ConvertTo-Json -Depth 12

    $utf8NoBom = New-Object `
        System.Text.UTF8Encoding `
        $false

    [System.IO.File]::WriteAllText(
        $Path,
        $json,
        $utf8NoBom
    )
}

try {
    New-Item `
        -ItemType Directory `
        -Path $OrchestrationRoot `
        -Force |
        Out-Null

    Start-Transcript `
        -Path $TranscriptPath `
        -Force |
        Out-Null

    $TranscriptStarted = $true

    Write-Info ""
    Write-Info "AutoGluon v5 extreme overnight run"
    Write-Info "=================================="
    Write-Info "Repository: $RepoRoot"
    Write-Info "Config: $Config"
    Write-Info "Run ID: $RunId"
    Write-Info "Run directory: $RunDir"
    Write-Info "Orchestration directory: $OrchestrationRoot"
    Write-Info ""

    $OverallStatus = "preflight"

    Assert-PathExists `
        -Path $Python `
        -Label "AutoGluon Python executable"

    Assert-PathExists `
        -Path $Runner `
        -Label "AutoGluon runner"

    Assert-PathExists `
        -Path $Config `
        -Label "AutoGluon v5 config"

    Assert-PathExists `
        -Path $TrainFeatures `
        -Label "v5 training features"

    Assert-PathExists `
        -Path $TrainTarget `
        -Label "v5 training target"

    if (
        -not $ValidateOnly `
        -and `
        (Test-Path -LiteralPath $RunDir)
    ) {
        throw "Run directory already exists: $RunDir"
    }

    try {
        $Diagnostics.system_ram = Get-SystemRamDiagnostics

        Write-Info (
            "System RAM free/total: " +
            "$($Diagnostics.system_ram.free_gb) / " +
            "$($Diagnostics.system_ram.total_gb) GB"
        )

        if (
            -not $ValidateOnly `
            -and `
            [double]$Diagnostics.system_ram.free_gb `
                -lt $MinimumFreeRamGb
        ) {
            throw (
                "Insufficient free RAM for overnight training. " +
                "Required: $MinimumFreeRamGb GB; " +
                "available: $($Diagnostics.system_ram.free_gb) GB."
            )
        }

        if (
            [double]$Diagnostics.system_ram.free_gb `
                -lt 20.0
        ) {
            Write-Info (
                "WARNING: less than 20 GB RAM is free. " +
                "Some AutoGluon models may be skipped."
            )
        }
    }
    catch {
        $Diagnostics.diagnostic_errors = @(
            $Diagnostics.diagnostic_errors +
            @("system_ram: $($_.Exception.Message)")
        )

        throw
    }

    try {
        $Diagnostics.gpu = Get-GpuDiagnostics

        Write-Info (
            "GPU diagnostics: " +
            (
                $Diagnostics.gpu |
                    ConvertTo-Json `
                        -Compress `
                        -Depth 8
            )
        )

        if (-not $Diagnostics.gpu.available) {
            throw "NVIDIA GPU diagnostics are unavailable."
        }
    }
    catch {
        $Diagnostics.diagnostic_errors = @(
            $Diagnostics.diagnostic_errors +
            @("gpu: $($_.Exception.Message)")
        )

        throw
    }

    Write-Info ""
    Write-Info "Validating AutoGluon configuration..."

    & $Python `
        -u `
        $Runner `
        validate `
        --config `
        $Config

    $ValidateExitCode = $LASTEXITCODE

    if ($ValidateExitCode -ne 0) {
        throw (
            "Configuration validation failed with " +
            "exit code $ValidateExitCode."
        )
    }

    Write-Info "Configuration validation succeeded."

    if ($ValidateOnly) {
        $OverallStatus = "validate_only_ok"
        $ExitCode = 0

        Write-Info ""
        Write-Info "ValidateOnly completed successfully."
    }
    else {
        $OverallStatus = "training"

        Enable-SleepPrevention
        $SleepPreventionEnabled = $true

        Write-Info ""
        Write-Info "Sleep prevention enabled."
        Write-Info "Starting AutoGluon training..."
        Write-Info ""

        & $Python `
            -u `
            $Runner `
            train `
            --config `
            $Config `
            --run-id `
            $RunId

        $TrainExitCode = $LASTEXITCODE

        if ($TrainExitCode -ne 0) {
            $OverallStatus = "training_failed"

            throw (
                "AutoGluon training failed with " +
                "exit code $TrainExitCode."
            )
        }

        Write-Info ""
        Write-Info "Training completed successfully."
        Write-Info "Inspecting completed predictor..."

        $OverallStatus = "inspection"

        $InspectionLines = @(
            & $Python `
                -u `
                $Runner `
                inspect `
                --run-dir `
                $RunDir `
                --attempt-load
        )

        $InspectExitCode = $LASTEXITCODE

        $InspectionText = (
            $InspectionLines `
                -join `
                [Environment]::NewLine
        )

        [System.IO.File]::WriteAllText(
            $InspectionPath,
            $InspectionText,
            (
                New-Object `
                    System.Text.UTF8Encoding `
                    $false
            )
        )

        foreach ($line in $InspectionLines) {
            Write-Host $line
        }

        if ($InspectExitCode -ne 0) {
            $OverallStatus = "inspection_failed"

            throw (
                "AutoGluon inspection failed with " +
                "exit code $InspectExitCode."
            )
        }

        try {
            $InspectionReport = (
                $InspectionText |
                    ConvertFrom-Json
            )
        }
        catch {
            $OverallStatus = "inspection_invalid_json"

            throw (
                "Inspection output could not be parsed as JSON: " +
                $_.Exception.Message
            )
        }

        if (
            $InspectionReport.classification `
                -ne "complete"
        ) {
            $OverallStatus = "inspection_not_complete"

            throw (
                "Inspection classification is " +
                "'$($InspectionReport.classification)', " +
                "expected 'complete'."
            )
        }

        if (
            -not `
            $InspectionReport.predictor_loading_succeeded
        ) {
            $OverallStatus = "predictor_load_failed"

            throw (
                "Inspection completed, but predictor loading " +
                "did not succeed."
            )
        }

        $OverallStatus = "completed"
        $ExitCode = 0

        Write-Info ""
        Write-Info "AutoGluon run completed and predictor loaded successfully."
    }
}
catch {
    Write-Info ""
    Write-Info "ERROR: $($_.Exception.Message)"

    if ($OverallStatus -eq "not_started") {
        $OverallStatus = "preflight_failed"
    }
    elseif ($OverallStatus -eq "preflight") {
        $OverallStatus = "preflight_failed"
    }

    if ($ExitCode -eq 0) {
        $ExitCode = 1
    }
}
finally {
    if ($SleepPreventionEnabled) {
        try {
            Disable-SleepPrevention
            Write-Info "Sleep prevention restored."
        }
        catch {
            Write-Info (
                "WARNING: failed to restore sleep settings: " +
                $_.Exception.Message
            )
        }
    }

    $EndTimeUtc = (
        Get-Date
    ).ToUniversalTime().ToString("o")

    $Summary = [ordered]@{
        schema_version = 1
        mode = $(if ($ValidateOnly) {
            "validate_only"
        }
        else {
            "train"
        })
        overall_status = $OverallStatus
        start_time_utc = $StartTimeUtc
        end_time_utc = $EndTimeUtc
        stamp = $Stamp
        config = $Config
        run_id = $RunId
        run_directory = $RunDir
        orchestration_directory = $OrchestrationRoot
        transcript = $TranscriptPath
        inspection = $InspectionPath
        minimum_free_ram_gb = $MinimumFreeRamGb
        exit_codes = [ordered]@{
            validation = $ValidateExitCode
            training = $TrainExitCode
            inspection = $InspectExitCode
            orchestrator = $ExitCode
        }
        diagnostics = $Diagnostics
    }

    try {
        Write-JsonFile `
            -Path $SummaryPath `
            -Payload $Summary
    }
    catch {
        Write-Info (
            "WARNING: failed to write summary JSON: " +
            $_.Exception.Message
        )
    }

    Write-Host ""
    Write-Host "=================================="
    Write-Host "Overall status: $OverallStatus"
    Write-Host "Run ID: $RunId"
    Write-Host "Run directory: $RunDir"
    Write-Host "Summary: $SummaryPath"
    Write-Host "Transcript: $TranscriptPath"
    Write-Host "Inspection: $InspectionPath"
    Write-Host "Exit code: $ExitCode"
    Write-Host "=================================="

    if ($TranscriptStarted) {
        try {
            Stop-Transcript | Out-Null
        }
        catch {
            # Transcript may already be stopped after interruption.
        }
    }
}

exit $ExitCode