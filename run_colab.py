#!/usr/bin/env python3
"""
Colab Runner - Clears cache and runs CPB v5.0.5
Use in Colab with: exec(open('run_colab.py').read())
"""

import requests
import time
import sys

print("Clearing cache...")
sys.path.clear()

print("Downloading CPB v5.0.5...")
url = 'https://raw.githubusercontent.com/caizongxun/cpbv2/main/v5_complete_fixed.py?t=' + str(int(time.time()))

try:
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        print("Downloaded successfully. Running...")
        exec(response.text)
    else:
        print(f"Error: HTTP {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
