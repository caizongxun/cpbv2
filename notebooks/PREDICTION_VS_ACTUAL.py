#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB v2: 預測符號 vs 實際符號比較
預測線段和實際線段的差別可視化
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*90)
print("VISUALIZATION: Predicted vs Actual Signals")
print("="*90)

plt.style.use('seaborn-v0_8-darkgrid')

class SignalComparison:
    def __init__(self, y_actual, y_predicted, prices=None, timestamps=None):
        """
        y_actual: 實際標籤 (0=Down, 1=Up)
        y_predicted: 模型預測 (0=Down, 1=Up)
        prices: 價格數據（可選，用於背景參考）
        timestamps: 時間戳（可選）
        """
        self.y_actual = np.array(y_actual)
        self.y_predicted = np.array(y_predicted)
        self.prices = prices
        self.timestamps = timestamps
        self.n_samples = len(y_actual)
        
        # 計算指標
        self.accuracy = accuracy_score(y_actual, y_predicted)
        self.precision = precision_score(y_actual, y_predicted, zero_division=0)
        self.recall = recall_score(y_actual, y_predicted, zero_division=0)
        self.f1 = f1_score(y_actual, y_predicted, zero_division=0)
        
        # 計算對/錯
        self.correct = (y_actual == y_predicted)
        self.n_correct = np.sum(self.correct)
        self.n_incorrect = np.sum(~self.correct)
    
    def plot_signals_timeline(self, figsize=(16, 8)):
        """時間線上的預測 vs 實際"""
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(self.n_samples)
        
        # 繪製實際信號
        ax.scatter(x, self.y_actual * 1.05, s=100, alpha=0.6, color='#2ecc71', 
                  label='Actual', edgecolors='black', linewidth=1, marker='^', zorder=3)
        
        # 繪製預測信號
        ax.scatter(x, self.y_predicted * 0.95, s=100, alpha=0.6, color='#e74c3c',
                  label='Predicted', edgecolors='black', linewidth=1, marker='v', zorder=3)
        
        # 標示對/錯
        correct_idx = np.where(self.correct)[0]
        incorrect_idx = np.where(~self.correct)[0]
        
        for idx in correct_idx:
            ax.plot([idx, idx], [self.y_actual[idx] * 1.05, self.y_predicted[idx] * 0.95],
                   color='green', alpha=0.3, linewidth=2, linestyle='--')
        
        for idx in incorrect_idx:
            ax.plot([idx, idx], [self.y_actual[idx] * 1.05, self.y_predicted[idx] * 0.95],
                   color='red', alpha=0.5, linewidth=2, linestyle='--')
        
        # 背景著色
        for i, is_correct in enumerate(self.correct):
            if is_correct:
                ax.axvspan(i-0.4, i+0.4, alpha=0.1, color='green')
            else:
                ax.axvspan(i-0.4, i+0.4, alpha=0.1, color='red')
        
        ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Signal (0=Down, 1=Up)', fontsize=12, fontweight='bold')
        ax.set_title('Predicted vs Actual Signals Over Time', fontsize=14, fontweight='bold')
        ax.set_ylim([-0.2, 1.2])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Down', 'Up'])
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 添加統計信息
        stats_text = f"Accuracy: {self.accuracy:.2%} | Precision: {self.precision:.2%} | Recall: {self.recall:.2%}"
        ax.text(0.5, 1.08, stats_text, transform=ax.transAxes, ha='center',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_signal_comparison_bars(self, figsize=(14, 8)):
        """信號比較柱狀圖"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 準確率柱
        ax = axes[0, 0]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [self.accuracy, self.precision, self.recall, self.f1]
        colors = ['#2ecc71' if v > 0.7 else '#f39c12' if v > 0.5 else '#e74c3c' for v in values]
        bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=2)
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Performance Metrics', fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.axhline(y=0.70, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axhline(y=0.50, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 對/錯分布
        ax = axes[0, 1]
        labels = ['Correct', 'Incorrect']
        sizes = [self.n_correct, self.n_incorrect]
        colors_pie = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0.1)
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=colors_pie, explode=explode,
                                           textprops={'fontweight': 'bold', 'fontsize': 11})
        ax.set_title('Prediction Correctness', fontweight='bold')
        
        # 3. 混淆矩陣熱力圖
        ax = axes[1, 0]
        cm = np.array([[np.sum((self.y_actual == 0) & (self.y_predicted == 0)),
                       np.sum((self.y_actual == 0) & (self.y_predicted == 1))],
                      [np.sum((self.y_actual == 1) & (self.y_predicted == 0)),
                       np.sum((self.y_actual == 1) & (self.y_predicted == 1))]])
        
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predict Down', 'Predict Up'])
        ax.set_yticklabels(['Actual Down', 'Actual Up'])
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('Actual', fontweight='bold')
        ax.set_title('Confusion Matrix', fontweight='bold')
        
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm[i, j], ha="center", va="center",
                             color="white" if cm[i, j] > cm.max() / 2 else "black",
                             fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        
        # 4. 信號分布
        ax = axes[1, 1]
        signal_categories = ['Actual Down', 'Actual Up', 'Pred Down', 'Pred Up']
        signal_counts = [
            np.sum(self.y_actual == 0),
            np.sum(self.y_actual == 1),
            np.sum(self.y_predicted == 0),
            np.sum(self.y_predicted == 1)
        ]
        colors_bars = ['#e74c3c', '#2ecc71', '#e67e22', '#3498db']
        
        bars = ax.bar(signal_categories, signal_counts, color=colors_bars, 
                     edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Signal Distribution', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, signal_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_prediction_errors(self, figsize=(14, 7)):
        """預測錯誤分析"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 左：錯誤類型
        ax = axes[0]
        false_positives = np.sum((self.y_actual == 0) & (self.y_predicted == 1))
        false_negatives = np.sum((self.y_actual == 1) & (self.y_predicted == 0))
        true_positives = np.sum((self.y_actual == 1) & (self.y_predicted == 1))
        true_negatives = np.sum((self.y_actual == 0) & (self.y_predicted == 0))
        
        error_types = ['True\nNegatives', 'False\nPositives', 'False\nNegatives', 'True\nPositives']
        error_counts = [true_negatives, false_positives, false_negatives, true_positives]
        colors_errors = ['#2ecc71', '#e74c3c', '#e74c3c', '#2ecc71']
        
        bars = ax.bar(error_types, error_counts, color=colors_errors, 
                     edgecolor='black', linewidth=2, alpha=0.7)
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Error Type Distribution', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, error_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 右：性能指標
        ax = axes[1]
        ax.axis('off')
        
        report_text = f"""
        PREDICTION ERROR ANALYSIS
        {'='*50}
        
        Performance Metrics:
          • Accuracy:   {self.accuracy:.4f} ({self.accuracy*100:.2f}%)
          • Precision:  {self.precision:.4f} ({self.precision*100:.2f}%)
          • Recall:     {self.recall:.4f} ({self.recall*100:.2f}%)
          • F1 Score:   {self.f1:.4f}
        
        Prediction Results:
          ✓ Correct:    {self.n_correct} ({self.n_correct/self.n_samples*100:.2f}%)
          ✗ Incorrect:  {self.n_incorrect} ({self.n_incorrect/self.n_samples*100:.2f}%)
        
        Error Breakdown:
          • True Negatives (TN):   {true_negatives}  (Correct Down)
          • False Positives (FP):  {false_positives}  (Wrong Up)
          • False Negatives (FN):  {false_negatives}  (Wrong Down)
          • True Positives (TP):   {true_positives}  (Correct Up)
        
        Signal Distribution:
          • Actual Down:     {np.sum(self.y_actual == 0)} ({np.sum(self.y_actual == 0)/len(self.y_actual)*100:.1f}%)
          • Actual Up:       {np.sum(self.y_actual == 1)} ({np.sum(self.y_actual == 1)/len(self.y_actual)*100:.1f}%)
          • Predicted Down:  {np.sum(self.y_predicted == 0)} ({np.sum(self.y_predicted == 0)/len(self.y_predicted)*100:.1f}%)
          • Predicted Up:    {np.sum(self.y_predicted == 1)} ({np.sum(self.y_predicted == 1)/len(self.y_predicted)*100:.1f}%)
        
        Insights:
          • Model Bias: {'Bias to UP' if np.sum(self.y_predicted == 1) > np.sum(self.y_actual == 1) else 'Bias to DOWN' if np.sum(self.y_predicted == 0) > np.sum(self.y_actual == 0) else 'Balanced'}
          • Sensitivity: {self.recall:.2%} (能抓住多少實際UP)
          • Specificity: {true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0:.2%} (能抓住多少實際DOWN)
        """
        
        ax.text(0.05, 0.95, report_text, transform=ax.transAxes,
               fontfamily='monospace', fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_price_with_signals(self, figsize=(16, 9)):
        """價格圖表與信號疊加"""
        if self.prices is None:
            print("需要價格數據才能繪製此圖表")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(self.prices))
        
        # 繪製價格線
        ax.plot(x, self.prices, 'k-', linewidth=2, label='Price', zorder=2)
        
        # 繪製實際信號
        actual_up_idx = np.where(self.y_actual == 1)[0]
        actual_down_idx = np.where(self.y_actual == 0)[0]
        
        ax.scatter(actual_up_idx, self.prices[actual_up_idx], s=150, color='#2ecc71',
                  marker='^', label='Actual Up', edgecolors='black', linewidth=1.5, zorder=3)
        ax.scatter(actual_down_idx, self.prices[actual_down_idx], s=150, color='#e67e22',
                  marker='v', label='Actual Down', edgecolors='black', linewidth=1.5, zorder=3)
        
        # 繪製預測信號
        pred_up_idx = np.where(self.y_predicted == 1)[0]
        pred_down_idx = np.where(self.y_predicted == 0)[0]
        
        ax.scatter(pred_up_idx, self.prices[pred_up_idx] * 1.002, s=100, color='#3498db',
                  marker='^', label='Predicted Up', edgecolors='blue', linewidth=1, 
                  alpha=0.6, zorder=2)
        ax.scatter(pred_down_idx, self.prices[pred_down_idx] * 0.998, s=100, color='#e74c3c',
                  marker='v', label='Predicted Down', edgecolors='red', linewidth=1,
                  alpha=0.6, zorder=2)
        
        # 標示對/錯區域
        for i, is_correct in enumerate(self.correct):
            if not is_correct and i < len(self.prices):
                ax.axvspan(i-0.3, i+0.3, alpha=0.1, color='red', zorder=1)
        
        ax.set_xlabel('Time Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax.set_title('Price Chart with Predicted vs Actual Signals', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_signal_alignment(self, window_size=50, figsize=(16, 10)):
        """信號對齊窗口（顯示滑動窗口中的對齊情況）"""
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        n_windows = min(3, max(1, len(self.y_actual) // window_size))
        
        for w in range(n_windows):
            ax = axes[w]
            start_idx = w * window_size
            end_idx = min(start_idx + window_size, len(self.y_actual))
            
            window_actual = self.y_actual[start_idx:end_idx]
            window_pred = self.y_predicted[start_idx:end_idx]
            window_correct = self.correct[start_idx:end_idx]
            
            x_window = np.arange(len(window_actual))
            
            # 背景著色
            for i, is_correct in enumerate(window_correct):
                if is_correct:
                    ax.axvspan(i-0.4, i+0.4, alpha=0.1, color='green')
                else:
                    ax.axvspan(i-0.4, i+0.4, alpha=0.1, color='red')
            
            # 繪製線條
            ax.plot(x_window, window_actual, 'go-', linewidth=2, markersize=8, 
                   label='Actual', zorder=2)
            ax.plot(x_window, window_pred, 'r^--', linewidth=2, markersize=8,
                   label='Predicted', zorder=2)
            
            # 連接線表示對/錯
            for i in range(len(window_actual)):
                if window_actual[i] != window_pred[i]:
                    ax.plot([i, i], [window_actual[i], window_pred[i]],
                           'r--', linewidth=1, alpha=0.5, zorder=1)
            
            ax.set_ylabel('Signal', fontweight='bold')
            ax.set_ylim([-0.2, 1.2])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Down', 'Up'])
            ax.set_xlim([-0.5, len(window_actual) - 0.5])
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            window_acc = np.sum(window_correct) / len(window_actual)
            ax.set_title(f'Window {w+1} (Index {start_idx}-{end_idx-1}): Accuracy = {window_acc:.2%}',
                        fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def generate_detailed_report(self):
        """生成詳細報告"""
        print("\n" + "="*90)
        print("DETAILED SIGNAL COMPARISON REPORT")
        print("="*90)
        
        print(f"\n[OVERALL PERFORMANCE]")
        print(f"Total Samples: {self.n_samples}")
        print(f"Correct Predictions: {self.n_correct} ({self.n_correct/self.n_samples*100:.2f}%)")
        print(f"Incorrect Predictions: {self.n_incorrect} ({self.n_incorrect/self.n_samples*100:.2f}%)")
        
        print(f"\n[METRICS]")
        print(f"Accuracy:  {self.accuracy:.4f} ({self.accuracy*100:.2f}%)")
        print(f"Precision: {self.precision:.4f} ({self.precision*100:.2f}%)")
        print(f"Recall:    {self.recall:.4f} ({self.recall*100:.2f}%)")
        print(f"F1 Score:  {self.f1:.4f}")
        
        # 混淆矩陣
        tn = np.sum((self.y_actual == 0) & (self.y_predicted == 0))
        fp = np.sum((self.y_actual == 0) & (self.y_predicted == 1))
        fn = np.sum((self.y_actual == 1) & (self.y_predicted == 0))
        tp = np.sum((self.y_actual == 1) & (self.y_predicted == 1))
        
        print(f"\n[CONFUSION MATRIX]")
        print(f"True Negatives (TN):   {tn}")
        print(f"False Positives (FP):  {fp}")
        print(f"False Negatives (FN):  {fn}")
        print(f"True Positives (TP):   {tp}")
        
        print(f"\n[SIGNAL DISTRIBUTION]")
        print(f"Actual Down:     {np.sum(self.y_actual == 0)} ({np.sum(self.y_actual == 0)/len(self.y_actual)*100:.1f}%)")
        print(f"Actual Up:       {np.sum(self.y_actual == 1)} ({np.sum(self.y_actual == 1)/len(self.y_actual)*100:.1f}%)")
        print(f"Predicted Down:  {np.sum(self.y_predicted == 0)} ({np.sum(self.y_predicted == 0)/len(self.y_predicted)*100:.1f}%)")
        print(f"Predicted Up:    {np.sum(self.y_predicted == 1)} ({np.sum(self.y_predicted == 1)/len(self.y_predicted)*100:.1f}%)")
        
        print(f"\n[MODEL BIAS]")
        actual_up_ratio = np.sum(self.y_actual == 1) / len(self.y_actual)
        pred_up_ratio = np.sum(self.y_predicted == 1) / len(self.y_predicted)
        print(f"Actual Up Ratio:    {actual_up_ratio:.2%}")
        print(f"Predicted Up Ratio: {pred_up_ratio:.2%}")
        
        if abs(pred_up_ratio - actual_up_ratio) > 0.1:
            if pred_up_ratio > actual_up_ratio:
                print(f"→ Model has BULLISH BIAS (Overpredicts UP by {(pred_up_ratio - actual_up_ratio):.2%})")
            else:
                print(f"→ Model has BEARISH BIAS (Overpredicts DOWN by {(actual_up_ratio - pred_up_ratio):.2%})")
        else:
            print(f"→ Model is BALANCED")
        
        print(f"\n[ANALYSIS]")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"Sensitivity (Recall): {sensitivity:.2%} - 能捕捉到多少實際的UP信號")
        print(f"Specificity: {specificity:.2%} - 能捕捉到多少實際的DOWN信號")
        
        print("\n" + "="*90 + "\n")


# 使用示例
if __name__ == "__main__":
    # 生成示例數據
    np.random.seed(42)
    n_samples = 150
    
    # 實際信號（50% Up, 50% Down）
    y_actual = np.random.randint(0, 2, n_samples)
    
    # 預測信號（有70% 準確率）
    accuracy_rate = 0.75
    y_predicted = y_actual.copy()
    n_errors = int(n_samples * (1 - accuracy_rate))
    error_indices = np.random.choice(n_samples, n_errors, replace=False)
    y_predicted[error_indices] = 1 - y_predicted[error_indices]
    
    # 可選：生成價格數據
    prices = np.cumsum(np.random.randn(n_samples) * 0.02) + 100
    
    # 建立比較器
    print("\n[Creating Signal Comparison...]")
    comparator = SignalComparison(y_actual, y_predicted, prices=prices)
    
    # 生成報告
    comparator.generate_detailed_report()
    
    # 繪製圖表
    print("[Generating visualizations...]\n")
    comparator.plot_signals_timeline()
    comparator.plot_signal_comparison_bars()
    comparator.plot_prediction_errors()
    comparator.plot_price_with_signals()
    comparator.plot_signal_alignment(window_size=50)
    
    print("[Complete!]\n")
