@echo off
setlocal
cd /d "%~dp0.."

echo Restarting Clip Assist...
call "%~dp0run_service.bat"
if errorlevel 1 exit /b 1

echo.
echo Check app.log for: Hotkey service started
echo.

if /i not "%~1"=="--nopause" pause
