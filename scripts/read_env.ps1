param(
    [Parameter(Mandatory = $true)]
    [string]$Key
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$envFile = Join-Path $root ".env"

if (-not (Test-Path $envFile)) {
    Write-Error ".env not found at $envFile"
    exit 1
}

foreach ($line in Get-Content $envFile -Encoding UTF8) {
    $trimmed = $line.Trim()
    if ($trimmed -eq "" -or $trimmed.StartsWith("#")) { continue }
    if ($trimmed -match '^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
        if ($Matches[1] -eq $Key) {
            $value = $Matches[2].Trim()
            if (
                ($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))
            ) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            Write-Output $value
            exit 0
        }
    }
}

exit 1
