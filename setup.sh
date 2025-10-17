#!/bin/bash
# ===========================================================
# Setup Script for Streamlit Deployment
# Author: Toba Jordi Naibaho
# ===========================================================

# Pastikan pip terbaru
pip install --upgrade pip

# Install dependencies dari requirements.txt
pip install -r requirements.txt

# Pastikan Streamlit bisa mengenali folder .streamlit
mkdir -p ~/.streamlit/

# Konfigurasi default Streamlit agar tidak crash saat deploy
echo "\
[general]\n\
email = \"toba.jordi@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = \$PORT\n\
" > ~/.streamlit/config.toml
