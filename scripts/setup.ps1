# Interactive setup: .env, pip install, optional Windows logon task, start service.
param(
    [switch]$SkipStartup,
    [switch]$SkipStart
)

$ErrorActionPreference = "Stop"
$Scripts = $PSScriptRoot
$Root = (Resolve-Path (Join-Path $Scripts "..")).Path
$EnvFile = Join-Path $Root ".env"
$Requirements = Join-Path $Scripts "requirements.txt"

function Test-IsAdministrator {
    $principal = New-Object Security.Principal.WindowsPrincipal(
        [Security.Principal.WindowsIdentity]::GetCurrent())
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-SchTasks {
    param([Parameter(ValueFromRemainingArguments)][string[]]$TaskArgs)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    & schtasks.exe @TaskArgs 2>&1 | Out-Null
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldPreference
    return $exitCode
}

function Read-DotEnv {
    param([string]$Path)
    $map = @{}
    if (-not (Test-Path $Path)) { return $map }
    foreach ($line in Get-Content $Path -Encoding UTF8) {
        $t = $line.Trim()
        if ($t -eq "" -or $t.StartsWith("#")) { continue }
        if ($t -match '^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
            $v = $Matches[2].Trim().Trim('"').Trim("'")
            $map[$Matches[1]] = $v
        }
    }
    return $map
}

function Prompt-Value {
    param(
        [string]$Label,
        [string]$Default = "",
        [switch]$Required,
        [switch]$Secret
    )
    $suffix = if ($Default) { " [$Default]" } else { "" }
    while ($true) {
        if ($Secret) {
            $secure = Read-Host "$Label$suffix (leave blank to keep)" -AsSecureString
            $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
                [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
        } else {
            $plain = Read-Host "$Label$suffix"
        }
        if ([string]::IsNullOrWhiteSpace($plain)) {
            if ($Default) { return $Default }
            if (-not $Required) { return "" }
            Write-Host "  This field is required." -ForegroundColor Yellow
            continue
        }
        return $plain.Trim()
    }
}

function Find-Python {
    param([hashtable]$Existing)
    if ($Existing.PYTHON_EXE -and (Test-Path $Existing.PYTHON_EXE)) {
        $py = $Existing.PYTHON_EXE
        $pyw = $Existing.PYTHONW_EXE
        if (-not $pyw) {
            $pyw = Join-Path (Split-Path $py -Parent) "pythonw.exe"
        }
        if (Test-Path $pyw) { return @{ PY = $py; PYW = $pyw } }
    }

    $candidates = @(
        (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:USERPROFILE\miniconda3\python.exe",
        "$env:USERPROFILE\anaconda3\python.exe",
        "C:\Python312\python.exe"
    ) | Where-Object { $_ -and (Test-Path $_) }

    foreach ($py in $candidates) {
        $pyw = Join-Path (Split-Path $py -Parent) "pythonw.exe"
        if (Test-Path $pyw) { return @{ PY = $py; PYW = $pyw } }
    }

    Write-Host ""
    Write-Host "Python was not found automatically." -ForegroundColor Yellow
    $py = Prompt-Value -Label "Full path to python.exe" -Required
    $pyw = Join-Path (Split-Path $py -Parent) "pythonw.exe"
    if (-not (Test-Path $pyw)) {
        $pyw = Prompt-Value -Label "Full path to pythonw.exe" -Required
    }
    return @{ PY = $py; PYW = $pyw }
}

Write-Host ""
Write-Host "=== Exam helper setup ===" -ForegroundColor Cyan
Write-Host "Project: $Root"
Write-Host ""

$existing = Read-DotEnv -Path $EnvFile
$pyPaths = Find-Python -Existing $existing

Write-Host "Python:" -ForegroundColor Cyan
$pythonExe = Prompt-Value -Label "PYTHON_EXE" -Default $pyPaths.PY -Required
$pythonwExe = Prompt-Value -Label "PYTHONW_EXE" -Default $pyPaths.PYW -Required
if (-not (Test-Path $pythonExe)) { throw "PYTHON_EXE not found: $pythonExe" }
if (-not (Test-Path $pythonwExe)) { throw "PYTHONW_EXE not found: $pythonwExe" }

Write-Host ""
Write-Host "Notifications:" -ForegroundColor Cyan
$ntfyAnswer = Read-Host "Enable ntfy.sh push notifications? (Y/n)"
if ($ntfyAnswer -match '^[Nn]') {
    $ntfyEnabledStr = "false"
    $ntfy = ""
    Write-Host "  Notifications off. Answers still copy to clipboard; see app.log." -ForegroundColor DarkGray
} else {
    $ntfyEnabledStr = "true"
    $ntfy = Prompt-Value -Label "NTFY_TOPIC (ntfy.sh topic name)" -Default $existing.NTFY_TOPIC -Required
}

Write-Host ""
Write-Host "LLM API keys (press Enter to skip optional keys):" -ForegroundColor Cyan
Write-Host "  At least one key is recommended for failover."
$gemini = Prompt-Value -Label "GEMINI_API_KEY" -Default $existing.GEMINI_API_KEY -Secret
$groq = Prompt-Value -Label "GROQ_API_KEY" -Default $existing.GROQ_API_KEY -Secret
$hf = Prompt-Value -Label "HF_TOKEN" -Default $existing.HF_TOKEN -Secret
$openrouter = Prompt-Value -Label "OPENROUTER_API_KEY" -Default $existing.OPENROUTER_API_KEY -Secret

if (-not ($gemini -or $groq -or $hf -or $openrouter)) {
    Write-Host ""
    Write-Host "Warning: No API keys set. The service will start but Ctrl+B will fail until you add keys to .env" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Hotkey:" -ForegroundColor Cyan
Write-Host "  Format: modifier+key (e.g. ctrl+b, ctrl+shift+f5). Default: ctrl+b"
$hotkeyDefault = if ($existing.HOTKEY) { $existing.HOTKEY } else { "ctrl+b" }
$hotkey = $hotkeyDefault
while ($true) {
    $hotkeyInput = Read-Host "Shortcut [$hotkeyDefault]"
    if ([string]::IsNullOrWhiteSpace($hotkeyInput)) {
        $hotkeyInput = $hotkeyDefault
    }
    $validateScript = Join-Path $Scripts "validate_hotkey.py"
    $json = & $pythonExe $validateScript $hotkeyInput 2>&1 | Out-String
    try {
        $result = $json | ConvertFrom-Json
    } catch {
        Write-Host "  Could not validate shortcut. Try again." -ForegroundColor Yellow
        continue
    }
    if ($result.ok) {
        $hotkey = $result.normalized
        Write-Host "  Using: $($result.display)" -ForegroundColor Green
        break
    }
    Write-Host "  $($result.error)" -ForegroundColor Yellow
    Write-Host "  Try again (Enter alone = ctrl+b)." -ForegroundColor DarkGray
    $hotkeyDefault = "ctrl+b"
}

Write-Host ""
Write-Host "RAG (semantic search over context/):" -ForegroundColor Cyan
$ragDefault = if ($existing.RAG_ENABLED -eq "false") { "n" } else { "Y" }
$ragAnswer = Read-Host "Enable RAG for context/ folder? (Y/n) [$ragDefault]"
$ragEnabledStr = if ($ragAnswer -match '^[Nn]' -or ($ragDefault -eq 'n' -and [string]::IsNullOrWhiteSpace($ragAnswer))) { "false" } else { "true" }

Write-Host ""
$taskName = Prompt-Value -Label "TASK_NAME (Windows scheduled task)" -Default $(if ($existing.TASK_NAME) { $existing.TASK_NAME } else { "DataHotkeyLLM" })

$envContent = @(
    "# Generated by scripts/setup.ps1 - do not commit",
    "PYTHON_EXE=$pythonExe",
    "PYTHONW_EXE=$pythonwExe",
    "",
    "NTFY_ENABLED=$ntfyEnabledStr",
    "NTFY_TOPIC=$ntfy",
    "",
    "GEMINI_API_KEY=$gemini",
    "GROQ_API_KEY=$groq",
    "HF_TOKEN=$hf",
    "OPENROUTER_API_KEY=$openrouter",
    "",
    "HOTKEY=$hotkey",
    "",
    "RAG_ENABLED=$ragEnabledStr",
    "RAG_TOP_K=5",
    "RAG_CHUNK_CHARS=700",
    "RAG_CHUNK_OVERLAP=120",
    "",
    "TASK_NAME=$taskName"
) -join "`r`n"

Set-Content -Path $EnvFile -Value $envContent -Encoding UTF8
Write-Host ""
Write-Host "Wrote $EnvFile" -ForegroundColor Green

Write-Host ""
Write-Host "Installing Python packages..." -ForegroundColor Cyan
& $pythonExe -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
Write-Host "Dependencies installed." -ForegroundColor Green

if ($ragEnabledStr -eq "true") {
    $contextDir = Join-Path $Root "context"
    if (Test-Path $contextDir) {
        Write-Host "Building RAG index from context/ ..." -ForegroundColor Cyan
        & $pythonExe (Join-Path $Scripts "index_rag.py")
    } else {
        Write-Host "  No context/ folder yet. Add PDFs or notes there, then run index_rag.bat" -ForegroundColor DarkGray
    }
}

$registerStartup = -not $SkipStartup
if (-not $SkipStartup) {
    $answer = Read-Host "Register to run at Windows logon? (Y/n)"
    if ($answer -match '^[Nn]') { $registerStartup = $false }
}

if ($registerStartup) {
    if (-not (Test-IsAdministrator)) {
        Write-Host ""
        Write-Host "Logon task requires administrator rights. Skipping scheduled task." -ForegroundColor Yellow
        Write-Host "  To enable auto-start: right-click setup.bat -> Run as administrator." -ForegroundColor Yellow
        $registerStartup = $false
    }
}

if ($registerStartup) {
    $launcher = Join-Path $Scripts "run_service.bat"
    Invoke-SchTasks /delete /tn $taskName /f | Out-Null
    $tr = "cmd /c `"`"$launcher`"`""
    $taskExit = Invoke-SchTasks /create /tn $taskName /tr $tr /sc onlogon /rl highest /f
    if ($taskExit -ne 0) {
        Write-Host ""
        Write-Host "Could not create scheduled task (exit $taskExit)." -ForegroundColor Yellow
        Write-Host "  Try: right-click setup.bat -> Run as administrator" -ForegroundColor Yellow
    } else {
        Write-Host "Registered scheduled task: $taskName" -ForegroundColor Green
    }
}

if (-not $SkipStart) {
    Write-Host ""
    Write-Host "Starting service..." -ForegroundColor Cyan
    & (Join-Path $Scripts "run_service.bat")
}

Write-Host ""
Write-Host "Done. Ctrl+B runs the exam helper. Logs: $Root\app.log" -ForegroundColor Green
Write-Host "For development restarts, use: restart_service.bat" -ForegroundColor Green
Write-Host ""
exit 0
