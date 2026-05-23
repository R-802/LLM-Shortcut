@echo off
setlocal
cd /d "%~dp0.."
for /f "delims=" %%p in ('powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0read_env.ps1" -Key PYTHON_EXE') do set "PY=%%p"
if not defined PY set "PY=python"
"%PY%" "%~dp0index_rag.py"
pause
