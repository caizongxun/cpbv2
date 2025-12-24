#!/bin/bash

# 卸載舊版本
pip uninstall -y torch torchvision torchaudio

# 清除 pip 快取
pip cache purge

# 安裝最新 PyTorch with CUDA 12.4 支持
# 使用官方推薦的安裝方式
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "Installation complete"
