@echo off
rem Open Maiman Studio as a desktop application, straight from this checkout.
rem
rem Double-click it. The first run installs Electron into desktop\node_modules,
rem which takes a minute; every run after that opens the window directly.
rem Needs Node.js and a Python that has NumPy. Set MAIMAN_PYTHON to use a
rem Python other than the one on PATH.

setlocal
cd /d "%~dp0"

where npm >nul 2>nul
if errorlevel 1 (
  echo Maiman Studio needs Node.js to open as a desktop app.
  echo Install it from https://nodejs.org ^(the LTS version^), then run this again.
  pause
  exit /b 1
)

if "%MAIMAN_PYTHON%"=="" set "MAIMAN_PYTHON=python"
"%MAIMAN_PYTHON%" -c "import numpy" >nul 2>nul
if errorlevel 1 (
  echo The engine needs Python 3.11+ with NumPy, and "%MAIMAN_PYTHON%" does not have it.
  echo Install Python from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^),
  echo then run:  %MAIMAN_PYTHON% -m pip install -e .
  pause
  exit /b 1
)

cd desktop
if not exist node_modules\electron (
  echo First run: installing Electron ^(once^)...
  call npm install --no-audit --no-fund
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
call npm start --silent
if errorlevel 1 pause
