[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d{4}$')]
    [string]$ModelALast4,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d{4}$')]
    [string]$ModelBLast4,

    [string]$RepoRoot = 'C:\capstone\repo',

    [ValidateSet('shadow', 'live')]
    [string]$ExecutionMode = 'shadow',

    [switch]$OrdersEnabled,

    [double]$DurationHours = 24,

    [string]$RuntimeRoot = ''
)

$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path -LiteralPath $RepoRoot).Path
$template = Join-Path $repo 'config\dual_live_rehearsal_template.yaml'
if (-not (Test-Path -LiteralPath $template)) {
    throw "Template not found: $template"
}

if ($ExecutionMode -eq 'shadow' -and $OrdersEnabled) {
    throw 'OrdersEnabled cannot be used with shadow mode.'
}

if ([string]::IsNullOrWhiteSpace($RuntimeRoot)) {
    $RuntimeRoot = if ($ExecutionMode -eq 'live') {
        'runtime/dual_live_rehearsal_live'
    }
    else {
        'runtime/dual_live_rehearsal_shadow'
    }
}

$localDirectory = Join-Path $repo 'runtime\local'
New-Item -ItemType Directory -Path $localDirectory -Force | Out-Null
$destination = Join-Path $localDirectory "dual_live_rehearsal_$ExecutionMode.yaml"

$content = Get-Content -LiteralPath $template -Raw
$content = $content.Replace('REPLACE_MODEL_A_LAST4', $ModelALast4)
$content = $content.Replace('REPLACE_MODEL_B_LAST4', $ModelBLast4)
$content = $content -replace '(?m)^  execution_mode: .+$', "  execution_mode: $ExecutionMode"
$content = $content -replace '(?m)^  orders_enabled: .+$', ('  orders_enabled: ' + $OrdersEnabled.IsPresent.ToString().ToLowerInvariant())
$content = $content -replace '(?m)^  duration_hours: .+$', "  duration_hours: $DurationHours"
$content = $content -replace '(?m)^  runtime_root: .+$', "  runtime_root: $RuntimeRoot"

Set-Content -LiteralPath $destination -Value $content -Encoding utf8

Write-Host ''
Write-Host 'Local dual-live config created.'
[PSCustomObject]@{
    ConfigPath = $destination
    ExecutionMode = $ExecutionMode
    OrdersEnabled = $OrdersEnabled.IsPresent
    DurationHours = $DurationHours
    RuntimeRoot = $RuntimeRoot
    ModelAAccount = "*****$ModelALast4"
    ModelBAccount = "*****$ModelBLast4"
    StoredUnderIgnoredRuntimeFolder = $true
} | Format-List
