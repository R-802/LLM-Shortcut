@echo off
setlocal
cd /d "%~dp0"

echo Restarting exam helper...
call "%~dp0scripts\run_service.bat"
if errorlevel 1 exit /b 1

echo.
echo Check app.log for: Hotkey service started
echo.

if /i not "%~1"=="--nopause" pause
