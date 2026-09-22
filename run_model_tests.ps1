param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$TestName,

    [string]$ModelPath = "models/ppo_metadrive.zip",
    [int]$Episodes = 5,
    [int]$MaxSteps = 1000,
    [int]$RecordSteps = 1000,
    [int]$Seed = 0,
    [int]$Fps = 30,
    [int]$ScreenSize = 672,
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackupRoot = Join-Path $ProjectRoot "model_backup"

if ($TestName -in @(".", "..") -or
    $TestName.IndexOfAny([System.IO.Path]::GetInvalidFileNameChars()) -ge 0 -or
    $TestName.EndsWith(".") -or
    $TestName.EndsWith(" ")) {
    throw "TestName contains characters that cannot be used in a folder name."
}

if ($Episodes -lt 1 -or $MaxSteps -lt 1 -or $RecordSteps -lt 1) {
    throw "Episodes and step counts must be at least 1."
}

if ($Fps -lt 1 -or $ScreenSize -lt 1) {
    throw "Fps and ScreenSize must be at least 1."
}

$OutputDirectory = Join-Path $BackupRoot $TestName
$EvaluationReport = Join-Path $OutputDirectory "evaluation.txt"
$RecordingReport = Join-Path $OutputDirectory "recording.txt"
$VideoOutput = Join-Path $OutputDirectory "first_person.mp4"
$RunSummary = Join-Path $OutputDirectory "run_summary.txt"

New-Item -ItemType Directory -Path $BackupRoot -Force | Out-Null

if (Test-Path -LiteralPath $OutputDirectory) {
    $ExistingFiles = Get-ChildItem -LiteralPath $OutputDirectory -Force
    if ($ExistingFiles.Count -gt 0) {
        throw "Test output already exists: $OutputDirectory. Use a new TestName."
    }
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

Push-Location $ProjectRoot
try {
    Write-Host "[1/2] Running analyze/model_evaluate.py..."
    & $PythonExecutable -m analyze.model_evaluate `
        --model-path $ModelPath `
        --episodes $Episodes `
        --max-steps $MaxSteps `
        --output $EvaluationReport

    if ($LASTEXITCODE -ne 0) {
        throw "analyze/model_evaluate.py failed with exit code $LASTEXITCODE."
    }

    Write-Host "[2/2] Running record/1st_person.py..."
    & $PythonExecutable -m record.1st_person `
        --model-path $ModelPath `
        --steps $RecordSteps `
        --seed $Seed `
        --fps $Fps `
        --screen-size $ScreenSize `
        --output $VideoOutput `
        --report-output $RecordingReport

    if ($LASTEXITCODE -ne 0) {
        throw "record/1st_person.py failed with exit code $LASTEXITCODE."
    }

    @(
        "MetaDrive Test Run"
        "=================="
        "Test name         : $TestName"
        "Model             : $ModelPath"
        "Evaluation report : $EvaluationReport"
        "Recording report  : $RecordingReport"
        "First-person video: $VideoOutput"
        "Status            : completed"
    ) | Set-Content -LiteralPath $RunSummary -Encoding utf8

    Write-Host "All tests completed."
    Write-Host "Results saved to: $OutputDirectory"
}
finally {
    Pop-Location
}
