#!/bin/bash
# Colab Setup Script for CPB v5.0.5

echo "Installing dependencies..."
pip install torch --quiet
pip install pandas numpy --quiet

echo "Dependencies installed. Ready to run CPB v5.0.5"
echo ""
echo "Next step: Execute in a new Colab cell:"
echo ""
echo "import requests"
echo "import time"
echo "url = 'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_complete_fixed.py?t=' + str(int(time.time()))"
echo "exec(requests.get(url).text)"
