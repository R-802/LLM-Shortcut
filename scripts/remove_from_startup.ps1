# Remove Clip Assist from Windows startup (scheduled task + Startup folder shortcuts).
param(
    [string]$TaskName = "ClipAssist"
)

$ErrorActionPreference = "SilentlyContinue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$envFile = Join-Path $Root ".env"

if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^\s*TASK_NAME\s*=\s*(.+)$') {
            $TaskName = $Matches[1].Trim().Trim('"').Trim("'")
            break
        }
    }
}

$removed = $false

# 1) Logon scheduled task (from setup.bat "run at Windows logon")
schtasks /delete /tn $TaskName /f 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Removed scheduled task: $TaskName" -ForegroundColor Green
    $removed = $true
} else {
    Write-Host "No scheduled task named: $TaskName" -ForegroundColor DarkGray
}

# Legacy task name from older installs
if ($TaskName -ne "DataHotkeyLLM") {
    schtasks /delete /tn "DataHotkeyLLM" /f 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Removed legacy scheduled task: DataHotkeyLLM" -ForegroundColor Green
        $removed = $true
    }
}

# 2) Shortcuts in the user Startup folder (if any were placed there manually)
$startupFolder = [Environment]::GetFolderPath("Startup")
$markers = @(
    "start_clip_assist",
    "start_exam_helper",
    "run_service.bat",
    "Clip Assist",
    "Exam Helper",
    "DataHotkey",
    "ClipAssist",
    $Root
)

if (Test-Path $startupFolder) {
    $shell = New-Object -ComObject WScript.Shell
    foreach ($lnk in Get-ChildItem $startupFolder -Filter "*.lnk" -ErrorAction SilentlyContinue) {
        try {
            $sc = $shell.CreateShortcut($lnk.FullName)
            $haystack = "$($sc.TargetPath) $($sc.Arguments) $($sc.Description) $($lnk.Name)"
            $match = $false
            foreach ($m in $markers) {
                if ($haystack -like "*$m*") { $match = $true; break }
            }
            if ($match) {
                Remove-Item $lnk.FullName -Force
                Write-Host "Removed Startup shortcut: $($lnk.Name)" -ForegroundColor Green
                $removed = $true
            }
        } catch {
            # skip broken shortcuts
        }
    }
}

if (-not $removed) {
    Write-Host ""
    Write-Host "Nothing to remove. Clip Assist is not set to run at logon." -ForegroundColor Yellow
    Write-Host "You can still start it from the Desktop shortcut or scripts\start_clip_assist.vbs." -ForegroundColor DarkGray
} else {
    Write-Host ""
    Write-Host "Done. Clip Assist will not start automatically at logon." -ForegroundColor Green
    Write-Host "To stop a running copy now, end pythonw.exe in Task Manager or run scripts\restart_service.bat." -ForegroundColor DarkGray
}
