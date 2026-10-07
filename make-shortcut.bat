@echo off
rem Double-click once: puts a "Maiman Studio" shortcut with the Maiman icon in
rem this folder and opens the studio. Use that shortcut from then on.
rem Set MAIMAN_PYTHON to use a Python other than the one on PATH.
setlocal
cd /d "%~dp0"
if "%MAIMAN_PYTHON%"=="" set "MAIMAN_PYTHON=python"
"%MAIMAN_PYTHON%" -c "import numpy" >nul 2>nul
if errorlevel 1 (
  echo Maiman needs Python 3.11+ with NumPy, and "%MAIMAN_PYTHON%" does not have it.
  echo Install Python from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^),
  echo then run:  %MAIMAN_PYTHON% -m pip install -e .
  pause
  exit /b 1
)
"%MAIMAN_PYTHON%" tools\make_shortcut.py || (pause & exit /b 1)
start "" "Maiman Studio.lnk"
