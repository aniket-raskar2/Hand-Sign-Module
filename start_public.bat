@echo off
echo ============================================
echo   Hand Sign Detection - Public Server
echo ============================================
echo.
echo Starting Django server on 0.0.0.0:8000...
start "Django Server" cmd /k "cd /d c:\Users\Aniket\Downloads\Final_code && venv\Scripts\activate && python manage.py runserver 0.0.0.0:8000"
echo.
echo Starting MiroTalk video calling on port 8080...
start "MiroTalk Server" cmd /k "cd /d C:\Users\Aniket\OneDrive\Desktop\video-calling\mirotalkc2c && npm start"
echo.
echo Waiting for servers to start...
timeout /t 5 /nobreak >nul
echo.
echo Starting ngrok tunnels...
start "ngrok Tunnels" cmd /k "ngrok http 8000"
echo.
echo ============================================
echo   All 3 servers are starting!
echo   1. Django        - localhost:8000
echo   2. MiroTalk      - localhost:8080
echo   3. ngrok tunnel  - check ngrok window
echo ============================================
echo.
pause
