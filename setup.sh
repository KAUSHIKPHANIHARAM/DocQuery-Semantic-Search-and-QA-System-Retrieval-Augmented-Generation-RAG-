#!/bin/bash
# Quick Setup and Run Script for PDF QA System

echo "🚀 PDF Question Answering System - Quick Setup"
echo "=============================================="
echo ""

# Check Python version
echo "✓ Checking Python version..."
python --version || { echo "❌ Python not found. Please install Python 3.8+"; exit 1; }

# Create virtual environment
echo "✓ Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "✓ Activating virtual environment..."
source venv/bin/activate || { echo "❌ Failed to activate venv"; exit 1; }

# Install dependencies
echo "✓ Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1 && echo "✅ Dependencies installed" || { echo "❌ Failed to install dependencies"; exit 1; }

# Download NLTK data
echo "✓ Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True)" 2>/dev/null && echo "✅ NLTK data downloaded" || echo "⚠️  NLTK download skipped"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "✓ Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created (edit with your API keys if needed)"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the app, run:"
echo "  streamlit run app.py"
echo ""
