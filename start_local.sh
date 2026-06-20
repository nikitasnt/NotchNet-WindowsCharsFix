#!/bin/bash

# NotchNet Local Startup Script (Mac/Linux)

echo "🚀 Starting NotchNet Local..."

# 1. Check for Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python is not installed. Please install it first."
    exit 1
fi

# 2. Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# 3. Activate venv
source venv/bin/activate

# 4. Install/Update dependencies
echo "⬇️ Installing dependencies..."
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install -r requirements.txt

# 5. Configure Environment
export FLASK_APP=server.py

# Load .env if it exists to respect user config
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Set defaults only if not already set by .env
export LOCAL_MODE=${LOCAL_MODE:-"true"}
export CLOUD_MODE=${CLOUD_MODE:-"true"}
export LLM_MODEL=${LLM_MODEL:-"xiaomi/mimo-v2-flash:free"}

echo ""
echo "---------------------------------------------------"
echo "✅ Configuration"
echo "---------------------------------------------------"
echo "🤖 LLM Model: $LLM_MODEL"
echo "🧠 Embeddings: Local (HuggingFace)"
echo "---------------------------------------------------"

# 6. Check for Index (and Sentence Transformers)
if [ ! -d "faiss_index" ]; then
    echo "🧠 No knowledge base found. Building initial index..."
    $PYTHON_CMD config/build_index.py
fi

# 7. Start Server
echo "✅ Setup complete. Starting Server..."
echo "---------------------------------------------------"
echo "🌐 Server running at http://localhost:8000"
echo "📄 API Documentation in README.md"
echo "---------------------------------------------------"

$PYTHON_CMD server.py
