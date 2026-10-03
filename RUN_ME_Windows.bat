@echo off
REM Double-click this file to run the whole pipeline on Windows.
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 RUN_ME.py %*
    goto :eof
)
where python >nul 2>nul
if %errorlevel%==0 (
    python RUN_ME.py %*
    goto :eof
)
echo.
echo Python was not found on this computer.
echo 1. Download Python 3.11 or 3.12 from https://www.python.org/downloads/
echo 2. During installation tick "Add python.exe to PATH".
echo 3. Double-click this file again.
pause
