@echo off
cd /d "%~dp0"
set PYTHONPATH=%CD%\src
python batch_run.py %*
pause
