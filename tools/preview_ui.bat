@echo off
cd /d "%~dp0.."
if not defined PORT set PORT=7872
set GRADIO_SERVER_PORT=%PORT%
set PYTHONUTF8=1
set CRISPZ_LOG_LEVEL=debug
.venv\Scripts\python.exe app.py
