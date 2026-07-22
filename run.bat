@echo off
echo ============================================
echo   PromptLab - Prompt Engineering Workbench
echo ============================================
echo.

:: Check for .env
if not exist .env (
    echo [!] .env not found - copying .env.example to .env
    copy .env.example .env > nul
    echo [!] Please edit .env with your API keys and database credentials!
    echo.
)

:: Install backend dependencies
echo [1/2] Installing backend dependencies...
cd backend
pip install -r requirements.txt -q
cd ..

:: Install frontend dependencies
echo [2/2] Installing frontend dependencies...
cd frontend
pip install -r requirements.txt -q
cd ..

echo.
echo ============================================
echo   Starting PromptLab...
echo ============================================
echo.
echo   Backend API: http://127.0.0.1:8000
echo   API Docs:    http://127.0.0.1:8000/docs
echo   Frontend:    http://localhost:8501
echo.

:: Start backend in a new window
start "PromptLab API" cmd /k "cd backend && python main.py"

:: Wait for backend
echo Waiting for API to start...
timeout /t 3 /nobreak > nul

:: Start frontend in a new window
start "PromptLab UI" cmd /k "cd frontend && streamlit run app.py --server.port 8501"

echo.
echo Both servers started! Open http://localhost:8501 in your browser.
echo.
pause
