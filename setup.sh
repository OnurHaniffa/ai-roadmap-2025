#!/bin/bash

# Exit immediately if a command fails
set -e

echo "🔧 Setting up AI Roadmap Environment..."

# Remove old venv if it exists
rm -rf .venv

# Make sure Python 3.11 is available
brew install python@3.11 || true
brew link python@3.11 --force --overwrite

# Create a fresh virtual environment (with pip upgraded)
python3.11 -m venv .venv --upgrade-deps

# Activate the environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install pandas numpy matplotlib seaborn scikit-learn tabulate jupyter notebook ipykernel xgboost lightgbm tensorflow torch torchvision torchaudio openpyxl pillow requests

echo "✅ Environment setup complete! Activate it anytime with: source .venv/bin/activate"
