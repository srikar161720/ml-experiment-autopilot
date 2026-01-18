#!/bin/bash
# Setup script for ML Experiment Autopilot

set -e

echo "Setting up ML Experiment Autopilot..."

# Check if we're in the project root
if [ ! -f "requirements.txt" ]; then
    echo "Error: Run this script from the project root directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env and add your GEMINI_API_KEY"
fi

# Create output directories
echo "Creating output directories..."
mkdir -p data/sample
mkdir -p outputs/experiments
mkdir -p outputs/reports
mkdir -p outputs/models
mkdir -p outputs/mlruns

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your GEMINI_API_KEY"
echo "2. Run ./scripts/download_data.sh to get sample datasets"
echo "3. Run: python -m src.main --help"
