@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create .venv and install requirements.txt first. See docs/ANNOTATION.md.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8001 in your browser. Keep this window open.
".venv\Scripts\python.exe" -m scripts.annotate
pause
