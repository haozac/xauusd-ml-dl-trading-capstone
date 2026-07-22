[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\capstone\repo',

    [Parameter(Mandatory = $true)]
    [string]$ConfigPath,

    [string]$TaskName = 'Capstone-Dual-Rehearsal-Shadow',

    [ValidateSet('shadow', 'live')]
    [string]$ExecutionMode = 'shadow',

    [double]$DurationHours = 24,

    [switch]$OrdersEnabled,

    [string]$ConfirmLive = ''
)

$ErrorActionPreference = 'Stop'

if ($ExecutionMode -eq 'shadow' -and $OrdersEnabled) {
    throw 'OrdersEnabled cannot be used with shadow mode.'
}

$requiredLiveToken = 'I_UNDERSTAND_DUAL_REHEARSAL_SENDS_DEMO_ORDERS'
if ($OrdersEnabled -and $ConfirmLive -cne $requiredLiveToken) {
    throw 'The exact live confirmation token was not supplied.'
}

$repo = (Resolve-Path -LiteralPath $RepoRoot).Path
$config = if ([System.IO.Path]::IsPathRooted($ConfigPath)) {
    (Resolve-Path -LiteralPath $ConfigPath).Path
}
else {
    (Resolve-Path -LiteralPath (Join-Path $repo $ConfigPath)).Path
}

$python = Join-Path $repo '.venv-deployment\Scripts\python.exe'
$launcher = Join-Path $repo 'scripts\deployment\run_dual_strategy_rehearsal.py'
foreach ($required in @($python, $launcher, $config)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required file not found: $required"
    }
}

$localDirectory = Join-Path $repo 'runtime\local'
$taskLogDirectory = Join-Path $repo 'runtime\dual_live_task_logs'
New-Item -ItemType Directory -Path $localDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $taskLogDirectory -Force | Out-Null

$wrapper = Join-Path $localDirectory ("run_" + ($TaskName -replace '[^A-Za-z0-9_-]', '_') + '.cmd')
$taskLog = Join-Path $taskLogDirectory (($TaskName -replace '[^A-Za-z0-9_-]', '_') + '.log')

$orderArguments = if ($OrdersEnabled) {
    "--orders-enabled --confirm-live $requiredLiveToken"
}
else {
    '--orders-disabled'
}

$wrapperContent = @"
@echo off
setlocal
cd /d "$repo"
set PYTHONUNBUFFERED=1
set CUDA_VISIBLE_DEVICES=-1
set TF_DETERMINISTIC_OPS=1
set TF_CUDNN_DETERMINISTIC=1
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=1
"$python" -u "$launcher" --repo-root "$repo" --config "$config" --execution-mode $ExecutionMode $orderArguments --duration-hours $DurationHours >> "$taskLog" 2>&1
exit /b %ERRORLEVEL%
"@

Set-Content -LiteralPath $wrapper -Value $wrapperContent -Encoding ascii

$action = New-ScheduledTaskAction `
    -Execute 'cmd.exe' `
    -Argument "/d /c `"$wrapper`"" `
    -WorkingDirectory $repo

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Highest

$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Hours ([Math]::Ceiling($DurationHours + 3))) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

$task = New-ScheduledTask `
    -Action $action `
    -Principal $principal `
    -Settings $settings `
    -Description 'Supervised dual-model XAUUSD rehearsal. Run only in the logged-on capstoneadmin desktop session.'

Register-ScheduledTask `
    -TaskName $TaskName `
    -InputObject $task `
    -Force | Out-Null

Write-Host ''
Write-Host 'Scheduled task installed. It has no automatic trigger.'
Write-Host 'Start it only after both MT5 terminals are logged in and connected.'
[PSCustomObject]@{
    TaskName = $TaskName
    ExecutionMode = $ExecutionMode
    OrdersEnabled = $OrdersEnabled.IsPresent
    DurationHours = $DurationHours
    ConfigPath = $config
    WrapperPath = $wrapper
    TaskLog = $taskLog
    LogonType = 'Interactive - run only while user session is logged on'
    AutomaticTrigger = $false
} | Format-List
