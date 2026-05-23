@echo off
echo Removing Clip Assist from Windows startup...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0remove_from_startup.ps1"
echo.
pause
