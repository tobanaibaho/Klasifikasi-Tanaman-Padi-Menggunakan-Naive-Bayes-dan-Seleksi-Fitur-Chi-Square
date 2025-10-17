#!/bin/bash
# ======================================================
# Setup script for Streamlit Cloud deployment
# ======================================================

echo "🔧 Setting up environment for Streamlit app..."

# Upgrade pip to the latest version
pip install --upgrade pip

# Ensure essential build tools are available
pip install --upgrade setuptools wheel

# Install all required Python dependencies
pip install -r requirements.txt

# Optional: Show installed packages (for debugging)
pip list

echo "✅ Environment setup complete. Starting Streamlit..."
