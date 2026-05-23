# Create Clip Assist shortcut on the user's Desktop.
param(
    [string]$ShortcutName = "Clip Assist"
)

$ErrorActionPreference = "Stop"
$Scripts = $PSScriptRoot
$Root = (Resolve-Path (Join-Path $Scripts "..")).Path
$VbsLauncher = Join-Path $Scripts "start_clip_assist.vbs"

if (-not (Test-Path $VbsLauncher)) {
    throw "Missing launcher: $VbsLauncher"
}

$Desktop = [Environment]::GetFolderPath("Desktop")
$LinkPath = Join-Path $Desktop "$ShortcutName.lnk"

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($LinkPath)
$Shortcut.TargetPath = "wscript.exe"
$Shortcut.Arguments = "`"$VbsLauncher`""
$Shortcut.WorkingDirectory = $Root
$Shortcut.WindowStyle = 7
$Shortcut.Description = "Start Clip Assist (see .env HOTKEY= for your shortcut)"
$Shortcut.Save()

Write-Host "Created desktop shortcut:" -ForegroundColor Green
Write-Host "  $LinkPath"
Write-Host ""
Write-Host "Double-click to start. Use scripts\restart_service.bat after code changes."
