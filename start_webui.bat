@echo off
setlocal

cd /d "%~dp0"
if not exist "chatbot-env\Scripts\python.exe" (
    echo Virtual environment missing. Follow the setup instructions in Readme.md.
    pause
    exit /b 1
)
"chatbot-env\Scripts\python.exe" webui.py
pause
