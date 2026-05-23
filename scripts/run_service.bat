@echo off
setlocal EnableDelayedExpansion
set "SCRIPTS=%~dp0"
set "ROOT=%SCRIPTS%.."
cd /d "%ROOT%"

if not exist "%ROOT%\.env" (
    echo .env not found. Run setup.bat first.
    exit /b 1
)

for /f "delims=" %%p in ('powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPTS%read_env.ps1" -Key PYTHON_EXE') do set "PY=%%p"
for /f "delims=" %%p in ('powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPTS%read_env.ps1" -Key PYTHONW_EXE') do set "PYW=%%p"

if not defined PY (
    echo PYTHON_EXE not set in .env
    exit /b 1
)
if not defined PYW (
    echo PYTHONW_EXE not set in .env
    exit /b 1
)

if not exist "%PY%" (
    echo python not found: %PY%
    exit /b 1
)
if not exist "%PYW%" (
    echo pythonw not found: %PYW%
    exit /b 1
)
if not exist "%ROOT%\app.py" (
    echo app.py not found at %ROOT%\app.py
    exit /b 1
)

"%PY%" -m pip install -r "%SCRIPTS%requirements.txt" -q

REM Stop old headless instances before restart
taskkill /F /IM pythonw.exe >nul 2>&1
timeout /t 2 /nobreak >nul

start "" "%PYW%" "%ROOT%\app.py"
