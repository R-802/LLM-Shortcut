@echo off
cd /d "%~dp0"
for /f "delims=" %%p in ('powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\read_env.ps1" -Key PYTHON_EXE') do set "PY=%%p"
if not defined PY set "PY=python"
"%PY%" "%~dp0scripts\index_rag.py"
pause
