@echo off
setlocal
cd /d "%~dp0.."
echo Stopping Clip Assist so the RAG index is not locked...
taskkill /F /IM pythonw.exe >nul 2>&1
timeout /t 2 /nobreak >nul
for /f "delims=" %%p in ('powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0read_env.ps1" -Key PYTHON_EXE') do set "PY=%%p"
if not defined PY set "PY=python"
"%PY%" "%~dp0index_rag.py"
if errorlevel 1 (
    echo Index build failed.
    pause
    exit /b 1
)
echo.
echo Done. Restart Clip Assist with scripts\restart_service.bat
pause
