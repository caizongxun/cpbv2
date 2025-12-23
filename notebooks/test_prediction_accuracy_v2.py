"""
CPB V2 模型預測準確度測試框架
改進方案：模型學習動態波動率而不是硬編碼百分比

主要改進：
1. 模型輸出 [predicted_price, volatility] 而非單一價格
2. 使用 ATR 計算動態波動率基準
3. 聯合優化價格和波動率預測
4. 支持多時間框架 (3H, 5H, 12H)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

class PredictionAccuracyTesterV2:
    """改進版預測準確度測試器"""
    
    def __init__(self, model_type='v2_dynamic_volatility'):
        self.model_type = model_type
        self.results = []
        
    def generate_historical_klines(self, num_candles=20, start_price=87800):
        """生成模擬歷史 K 線"""
        klines = []
        current_price = start_price
        
        for i in range(num_candles):
            # 隨機波動
            daily_return = np.random.normal(0, 0.015)  # 1.5% std dev
            open_price = current_price
            close_price = current_price * (1 + daily_return)
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.01))
            
            klines.append({
                'time': i,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.uniform(1000, 5000)
            })
            
            current_price = close_price
        
        return klines
    
    def calculate_atr(self, klines, period=14):
        """計算平均真實波幅 (ATR)"""
        if len(klines) < period:
            return 0
        
        trs = []
        for i in range(1, len(klines)):
            high_low = klines[i]['high'] - klines[i]['low']
            high_close = abs(klines[i]['high'] - klines[i-1]['close'])
            low_close = abs(klines[i]['low'] - klines[i-1]['close'])
            tr = max(high_low, high_close, low_close)
            trs.append(tr)
        
        atr = np.mean(trs[-period:]) if trs else 0
        return atr
    
    def calculate_volatility(self, klines, period=14):
        """計算歷史波動率 (百分比)"""
        if len(klines) < period:
            return 0
        
        close_prices = [k['close'] for k in klines[-period:]]
        returns = np.diff(close_prices) / close_prices[:-1]
        volatility = np.std(returns) * 100  # 轉換為百分比
        
        return volatility
    
    def predict_v1_hardcoded(self, current_price, direction):
        """V1 版本：硬編碼百分比"""
        if direction == 'up':
            price_3h = current_price * 1.02
            price_5h = current_price * 1.03
        else:
            price_3h = current_price * 0.98
            price_5h = current_price * 0.97
        
        return price_3h, price_5h
    
    def predict_v2_dynamic(self, current_price, direction, volatility, atr):
        """V2 版本：動態波動率"""
        # 基於 ATR 的動態幅度
        atr_percent = (atr / current_price) * 100
        
        # 波動率調整係數
        vol_factor = max(0.5, min(2.0, volatility / 2.0))
        
        # 動態幅度 = ATR% * 波動率係數
        dynamic_move_3h = atr_percent * vol_factor * 1.5
        dynamic_move_5h = atr_percent * vol_factor * 2.5
        
        if direction == 'up':
            price_3h = current_price * (1 + dynamic_move_3h / 100)
            price_5h = current_price * (1 + dynamic_move_5h / 100)
        else:
            price_3h = current_price * (1 - dynamic_move_3h / 100)
            price_5h = current_price * (1 - dynamic_move_5h / 100)
        
        return price_3h, price_5h, dynamic_move_3h, dynamic_move_5h
    
    def generate_future_klines(self, start_price, num_candles=5, true_direction=None):
        """生成模擬未來實際走勢"""
        klines = []
        current_price = start_price
        
        for i in range(num_candles):
            if true_direction == 'down':
                daily_return = np.random.normal(-0.003, 0.008)  # 略微下跌
            elif true_direction == 'up':
                daily_return = np.random.normal(0.003, 0.008)   # 略微上漲
            else:
                daily_return = np.random.normal(0, 0.01)
            
            open_price = current_price
            close_price = current_price * (1 + daily_return)
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.008))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.008))
            
            klines.append({
                'time': i,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.uniform(1000, 5000)
            })
            
            current_price = close_price
        
        return klines
    
    def compare_models(self, num_tests=5):
        """比較 V1 和 V2 模型的準確度"""
        print("\n" + "="*70)
        print("CPB 模型 V1 vs V2 對比測試")
        print("="*70)
        
        v1_errors = []
        v2_errors = []
        
        for test_num in range(num_tests):
            print(f"\n[測試 {test_num+1}/{num_tests}]")
            
            # 生成歷史數據
            hist_klines = self.generate_historical_klines(num_candles=20)
            current_price = hist_klines[-1]['close']
            
            # 計算指標
            atr = self.calculate_atr(hist_klines)
            volatility = self.calculate_volatility(hist_klines)
            direction = 'down' if np.random.random() > 0.5 else 'up'
            
            print(f"  當前價格: ${current_price:.2f}")
            print(f"  ATR(14): ${atr:.2f}")
            print(f"  波動率: {volatility:.2f}%")
            print(f"  預測方向: {'看跌' if direction == 'down' else '看漲'}")
            
            # V1 預測
            v1_price_3h, v1_price_5h = self.predict_v1_hardcoded(current_price, direction)
            
            # V2 預測
            v2_price_3h, v2_price_5h, move_3h, move_5h = self.predict_v2_dynamic(
                current_price, direction, volatility, atr
            )
            
            print(f"\n  V1 預測 (硬編碼):")
            print(f"    3H: ${v1_price_3h:.2f} (±2%)")
            print(f"    5H: ${v1_price_5h:.2f} (±3%)")
            
            print(f"\n  V2 預測 (動態波動率):")
            print(f"    3H: ${v2_price_3h:.2f} ({move_3h:+.2f}%)")
            print(f"    5H: ${v2_price_5h:.2f} ({move_5h:+.2f}%)")
            
            # 生成實際未來走勢
            future_klines = self.generate_future_klines(
                current_price, 
                num_candles=5,
                true_direction=direction
            )
            actual_price_5h = future_klines[-1]['close']
            
            print(f"\n  實際 5H 價格: ${actual_price_5h:.2f}")
            
            # 計算誤差
            v1_error = abs(v1_price_5h - actual_price_5h) / actual_price_5h * 100
            v2_error = abs(v2_price_5h - actual_price_5h) / actual_price_5h * 100
            
            v1_errors.append(v1_error)
            v2_errors.append(v2_error)
            
            print(f"\n  V1 誤差: {v1_error:.2f}%")
            print(f"  V2 誤差: {v2_error:.2f}%")
            print(f"  改進: {v1_error - v2_error:+.2f}% (負數表示V2更好)")
            
            self.results.append({
                'test': test_num + 1,
                'current_price': current_price,
                'volatility': volatility,
                'atr': atr,
                'direction': direction,
                'actual_5h': actual_price_5h,
                'v1_pred': v1_price_5h,
                'v2_pred': v2_price_5h,
                'v1_error': v1_error,
                'v2_error': v2_error
            })
        
        # 整體統計
        print("\n" + "="*70)
        print("整體統計")
        print("="*70)
        print(f"\nV1 平均誤差: {np.mean(v1_errors):.2f}%")
        print(f"V2 平均誤差: {np.mean(v2_errors):.2f}%")
        print(f"平均改進: {np.mean(v1_errors) - np.mean(v2_errors):+.2f}%")
        print(f"改進率: {(1 - np.mean(v2_errors)/np.mean(v1_errors)) * 100:.1f}%")
        
        return v1_errors, v2_errors
    
    def plot_comparison(self, v1_errors, v2_errors):
        """繪製對比圖表"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 誤差箱線圖
        ax1 = axes[0, 0]
        ax1.boxplot([v1_errors, v2_errors], labels=['V1 (硬編碼)', 'V2 (動態波動率)'])
        ax1.set_ylabel('預測誤差 (%)')
        ax1.set_title('模型誤差分佈對比')
        ax1.grid(True, alpha=0.3)
        
        # 誤差趨勢
        ax2 = axes[0, 1]
        ax2.plot(range(1, len(v1_errors)+1), v1_errors, 'o-', label='V1', linewidth=2)
        ax2.plot(range(1, len(v2_errors)+1), v2_errors, 's-', label='V2', linewidth=2)
        ax2.set_xlabel('測試次數')
        ax2.set_ylabel('預測誤差 (%)')
        ax2.set_title('誤差趨勢')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 誤差直方圖
        ax3 = axes[1, 0]
        ax3.hist(v1_errors, alpha=0.6, label='V1', bins=5)
        ax3.hist(v2_errors, alpha=0.6, label='V2', bins=5)
        ax3.set_xlabel('預測誤差 (%)')
        ax3.set_ylabel('頻次')
        ax3.set_title('誤差分佈直方圖')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 改進情況
        ax4 = axes[1, 1]
        improvements = [v1_errors[i] - v2_errors[i] for i in range(len(v1_errors))]
        colors = ['green' if x > 0 else 'red' for x in improvements]
        ax4.bar(range(1, len(improvements)+1), improvements, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax4.set_xlabel('測試次數')
        ax4.set_ylabel('改進幅度 (%)')
        ax4.set_title('V2 相對於 V1 的改進 (正數=更好)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('model_comparison_v1_vs_v2.png', dpi=300, bbox_inches='tight')
        print("\n圖表已保存: model_comparison_v1_vs_v2.png")
        
        return fig

def main():
    tester = PredictionAccuracyTesterV2()
    v1_errors, v2_errors = tester.compare_models(num_tests=10)
    tester.plot_comparison(v1_errors, v2_errors)
    
    # 保存結果為 JSON
    with open('model_comparison_results.json', 'w') as f:
        json.dump(tester.results, f, indent=2)
    
    print("\n" + "="*70)
    print("測試完成！結果已保存")
    print("="*70)

if __name__ == '__main__':
    main()
