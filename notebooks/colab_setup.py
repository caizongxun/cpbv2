# ============================================================================
# Colab Setup - Cell 1 (Execute this first)
# ============================================================================
# Version: 1.0
# Purpose: Fix NumPy incompatibility and restart runtime
# ============================================================================

print("\n" + "="*80)
print("Colab NumPy Fix - Cell 1")
print("="*80 + "\n")

import os
import sys

print("[*] Installing NumPy 1.26.4 with --ignore-installed...")
print("[!] This will downgrade Colab's default NumPy 2.0.2\n")

os.system(
    f"{sys.executable} -m pip install --ignore-installed --no-cache-dir numpy==1.26.4 -q"
)

print("\n[✓] NumPy 1.26.4 installed successfully")
print("[!] Restarting runtime...\n")
print("[*] After restart, execute Cell 2 to start training\n")

os._exit(0)
