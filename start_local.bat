@echo off
setlocal enabledelayedexpansion

echo 🚀 Starting NotchNet Local...

:: 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install it from https://python.org/
    exit /b 1
)

:: 2. Create virtual environment if it doesn't exist
if not exist venv (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

:: 3. Activate venv
call venv\Scripts\activate

:: 4. Install/Update dependencies
echo ⬇️ Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

:: 5. Configure Environment
set FLASK_APP=server.py

:: Load .env if it exists to respect user config
if exist .env (
    for /f "usebackq tokens=1,2 delims==" %%A in (".env") do (
        set "key=%%A"
        if not "!key:~0,1!"=="#" (
            set "%%A=%%B"
        )
    )
)

:: Set defaults only if not already set by .env
if "%LOCAL_MODE%"=="" set LOCAL_MODE=true
if "%CLOUD_MODE%"=="" set CLOUD_MODE=true
if "%LLM_MODEL%"=="" set LLM_MODEL=moonshotai/kimi-k2:free

echo.
echo ---------------------------------------------------
echo ✅ Configuration
echo ---------------------------------------------------
echo 🤖 LLM Model: %LLM_MODEL%
echo 🧠 Embeddings: Local (HuggingFace)
echo ---------------------------------------------------

:: 6. Check for Index
if not exist faiss_index (
    echo 🧠 No knowledge base found. Building initial index...
    python config/build_index.py
)

:: 7. Start Server
echo ✅ Setup complete. Starting Server...
echo ---------------------------------------------------
echo 🌐 Server running at http://localhost:8000
echo 📄 API Documentation in README.md
echo ---------------------------------------------------

python server.py
pause
