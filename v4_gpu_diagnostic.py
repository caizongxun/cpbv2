#!/usr/bin/env python3
"""
V4 GPU使用診斷工具
直接在Colab/Kaggle執行以檢查GPU是否真的被使用
"""

import torch
import subprocess
import sys
from pathlib import Path

class GPUDiagnostic:
    """完整的GPU診斷工具"""
    
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def print_header(self, title):
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)
    
    def check_cuda_availability(self):
        """檢查CUDA基本環境"""
        self.print_header("1. CUDA環境檢查")
        
        checks = {
            "CUDA Available": torch.cuda.is_available(),
            "Device Count": torch.cuda.device_count(),
            "Current Device": torch.cuda.current_device(),
            "Device Name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "PyTorch Version": torch.__version__,
            "CUDA Version": torch.version.cuda,
            "cuDNN Version": torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A",
        }
        
        for key, value in checks.items():
            status = "✓" if value else "✗"
            print(f"{status} {key}: {value}")
        
        return torch.cuda.is_available()
    
    def check_cudnn_settings(self):
        """檢查cuDNN設定"""
        self.print_header("2. cuDNN設定")
        
        if not torch.cuda.is_available():
            print("✗ CUDA不可用，跳過")
            return
        
        settings = {
            "cuDNN Enabled": torch.backends.cudnn.enabled,
            "cuDNN Benchmark": torch.backends.cudnn.benchmark,
            "cuDNN Deterministic": torch.backends.cudnn.deterministic,
        }
        
        for key, value in settings.items():
            print(f"  {key}: {value}")
        
        # 測試是否能使用LSTM
        try:
            lstm = torch.nn.LSTM(4, 256, 2, batch_first=True).to(self.device)
            x = torch.randn(8, 30, 4).to(self.device)
            output, _ = lstm(x)
            print("\n✓ LSTM可以在GPU上執行")
            return True
        except Exception as e:
            print(f"\n✗ LSTM執行失敗: {str(e)[:100]}")
            return False
    
    def check_memory_allocation(self):
        """檢查GPU記憶體分配"""
        self.print_header("3. GPU記憶體狀態")
        
        if not torch.cuda.is_available():
            print("✗ CUDA不可用，跳過")
            return
        
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / 1e9
        
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        
        print(f"總記憶體: {total_memory:.2f} GB")
        print(f"已分配: {allocated:.2f} GB")
        print(f"已保留: {reserved:.2f} GB")
        print(f"可用: {total_memory - reserved:.2f} GB")
    
    def test_gpu_computation(self):
        """測試GPU計算"""
        self.print_header("4. GPU計算測試")
        
        if not torch.cuda.is_available():
            print("✗ CUDA不可用，跳過")
            return
        
        try:
            # 強制GPU計算
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            # 矩陣乘法
            print("\n測試1: 矩陣乘法 (1000x1000)")
            x = torch.randn(1000, 1000, device=self.device, dtype=torch.float32)
            y = torch.randn(1000, 1000, device=self.device, dtype=torch.float32)
            
            torch.cuda.synchronize()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"✓ 完成 (峰值記憶體: {peak_mem:.2f} GB)")
            
            # LSTM前向傳播
            print("\nTest 2: LSTM前向傳播 (batch=32, seq=30, hidden=256)")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            lstm = torch.nn.LSTM(4, 256, 2, batch_first=True).to(self.device)
            x = torch.randn(32, 30, 4, device=self.device)
            
            torch.cuda.synchronize()
            output, _ = lstm(x)
            torch.cuda.synchronize()
            
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"✓ 完成 (峰值記憶體: {peak_mem:.2f} GB)")
            
            if peak_mem < 0.1:
                print("\n⚠️  警告: 記憶體使用 < 100MB")
                print("   可能原因:")
                print("   1. 計算在CPU執行")
                print("   2. cuDNN被禁用")
                return False
            else:
                print("\n✓ GPU記憶體使用正常")
                return True
                
        except Exception as e:
            print(f"✗ 測試失敗: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_nvidia_smi(self):
        """直接檢查nvidia-smi輸出"""
        self.print_header("5. nvidia-smi 即時監控")
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
                 '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                print(result.stdout)
                return True
            else:
                print("✗ nvidia-smi 命令失敗")
                return False
        except FileNotFoundError:
            print("✗ nvidia-smi 未找到 (可能在某些環境中不可用)")
            return False
        except Exception as e:
            print(f"✗ 錯誤: {str(e)}")
            return False
    
    def run_full_diagnostic(self):
        """執行完整診斷"""
        print("\n" + "#" * 70)
        print("# V4 GPU 診斷工具 - 完整測試")
        print("#" * 70)
        
        results = {
            "CUDA環境": self.check_cuda_availability(),
            "cuDNN設定": self.check_cudnn_settings(),
            "記憶體狀態": self.check_memory_allocation() or True,  # 總是成功
            "計算測試": self.test_gpu_computation(),
            "nvidia-smi": self.check_nvidia_smi(),
        }
        
        self.print_header("診斷結果摘要")
        
        for test_name, result in results.items():
            status = "✓ 通過" if result else "✗ 失敗"
            print(f"{status}: {test_name}")
        
        if all(results.values()):
            print("\n✓✓✓ 所有測試通過！GPU配置正常")
            return True
        else:
            print("\n✗✗✗ 某些測試失敗，請檢查上述詳情")
            return False


if __name__ == "__main__":
    diagnostic = GPUDiagnostic()
    success = diagnostic.run_full_diagnostic()
    sys.exit(0 if success else 1)
