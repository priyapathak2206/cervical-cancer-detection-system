@echo off

echo ==========================================
echo Cervical Cell Detection System
echo ==========================================
echo.

echo Starting Flask AI backend...
start "Cervical AI Backend" cmd /k "cd /d D:\cervical_canceer && py app.py"

timeout /t 3 /nobreak >nul

echo Starting frontend...
start "Cervical Frontend" cmd /k "cd /d D:\cervical_canceer\frontend && py -m http.server 5500"

timeout /t 2 /nobreak >nul

echo.
echo ==========================================
echo Project started successfully
echo ==========================================
echo.
echo Open:
echo http://127.0.0.1:5500
echo.

start http://127.0.0.1:5500

pause