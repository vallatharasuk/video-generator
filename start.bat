@echo off
cd /d D:\video-generator
if not exist .venv\Scripts\python.exe (
  echo Python environment is missing.
  pause
  exit /b 1
)
.venv\Scripts\python.exe app.py
pause
