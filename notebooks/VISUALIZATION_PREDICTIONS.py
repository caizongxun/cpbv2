#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB v2: 模型預測能力可視化
Visualizing model prediction capabilities for all coins
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*90)
print("VISUALIZATION: Model Prediction Capabilities")
print("="*90)

# 設定可視化風格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PredictionVisualizer:
    def __init__(self, results_data):
        """
        results_data: 包含所有模型結果的字典或列表
        [
            {'coin': 'BTCUSDT', 'accuracy': 0.85, 'f1': 0.82, ...},
            {'coin': 'ETHUSDT', 'accuracy': 0.78, 'f1': 0.75, ...},
            ...
        ]
        """
        self.results = results_data
        self.df = pd.DataFrame(results_data)
        self.num_coins = len(results_data)
        
    def plot_accuracy_distribution(self, figsize=(14, 6)):
        """準確率分佈"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 柱狀圖
        sorted_results = sorted(self.results, key=lambda x: x['accuracy'], reverse=True)
        coins = [r['coin'] for r in sorted_results]
        accuracies = [r['accuracy'] for r in sorted_results]
        colors = ['#2ecc71' if acc > 0.80 else '#f39c12' if acc > 0.70 else '#e74c3c' 
                 for acc in accuracies]
        
        axes[0].barh(coins, accuracies, color=colors, edgecolor='black', linewidth=1.2)
        axes[0].axvline(x=0.80, color='green', linestyle='--', linewidth=2, label='Excellent (80%)')
        axes[0].axvline(x=0.70, color='orange', linestyle='--', linewidth=2, label='Good (70%)')
        axes[0].axvline(x=0.50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
        axes[0].set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0].set_title('Model Accuracy by Coin', fontsize=14, fontweight='bold')
        axes[0].set_xlim([0.4, 1.0])
        axes[0].legend(loc='lower right')
        axes[0].grid(axis='x', alpha=0.3)
        
        # 統計分佈
        axes[1].hist(accuracies, bins=10, color='#3498db', edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(accuracies), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(accuracies):.4f}')
        axes[1].axvline(np.median(accuracies), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(accuracies):.4f}')
        axes[1].set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('Accuracy Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_accuracy_vs_f1(self, figsize=(10, 8)):
        """準確率 vs F1分數"""
        fig, ax = plt.subplots(figsize=figsize)
        
        accuracies = [r['accuracy'] for r in self.results]
        f1_scores = [r['f1'] for r in self.results]
        coins = [r['coin'] for r in self.results]
        
        # 色彩編碼
        colors = ['#2ecc71' if acc > 0.80 else '#f39c12' if acc > 0.70 else '#e74c3c' 
                 for acc in accuracies]
        
        scatter = ax.scatter(accuracies, f1_scores, s=300, c=colors, alpha=0.7, 
                           edgecolors='black', linewidth=2)
        
        # 添加標籤
        for i, coin in enumerate(coins):
            ax.annotate(coin, (accuracies[i], f1_scores[i]), 
                       fontsize=9, fontweight='bold', ha='center', va='center')
        
        # 參考線
        ax.axhline(y=0.70, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Good F1 (0.70)')
        ax.axvline(x=0.80, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Excellent Acc (0.80)')
        ax.axhline(y=0.50, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Random (0.50)')
        ax.axvline(x=0.50, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        
        ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy vs F1 Score by Coin', fontsize=14, fontweight='bold')
        ax.set_xlim([0.4, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_confusion_matrices(self, figsize=(16, 10)):
        """混淆矩陣網格"""
        n_cols = 4
        n_rows = (len(self.results) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for idx, result in enumerate(sorted(self.results, key=lambda x: x['accuracy'], reverse=True)):
            ax = axes[idx]
            
            cm = np.array([[result['tn'], result['fp']], 
                          [result['fn'], result['tp']]])
            
            # 標準化
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                       cbar=False, ax=ax, square=True, 
                       xticklabels=['Predict Down', 'Predict Up'],
                       yticklabels=['Actual Down', 'Actual Up'])
            
            accuracy = result['accuracy']
            color = 'green' if accuracy > 0.80 else 'orange' if accuracy > 0.70 else 'red'
            ax.set_title(f"{result['coin']}\nAcc: {accuracy:.2%}", 
                        fontsize=11, fontweight='bold', color=color)
        
        # 隱藏未使用的subplot
        for idx in range(len(self.results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_performance_summary(self, figsize=(14, 8)):
        """性能總結"""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. 準確率盒圖
        ax1 = fig.add_subplot(gs[0, 0])
        accuracies = [r['accuracy'] for r in self.results]
        bp = ax1.boxplot(accuracies, vert=True, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#3498db')
        ax1.axhline(y=0.80, color='green', linestyle='--', alpha=0.7, label='Excellent')
        ax1.axhline(y=0.70, color='orange', linestyle='--', alpha=0.7, label='Good')
        ax1.axhline(y=0.50, color='red', linestyle='--', alpha=0.7, label='Random')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Accuracy Distribution (Box Plot)', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. 性能分布
        ax2 = fig.add_subplot(gs[0, 1])
        excellent = len([r for r in self.results if r['accuracy'] > 0.80])
        good = len([r for r in self.results if 0.70 <= r['accuracy'] <= 0.80])
        ok = len([r for r in self.results if 0.60 <= r['accuracy'] < 0.70])
        poor = len([r for r in self.results if r['accuracy'] < 0.60])
        
        categories = ['Excellent\n(>80%)', 'Good\n(70-80%)', 'OK\n(60-70%)', 'Poor\n(<60%)']
        counts = [excellent, good, ok, poor]
        colors_pie = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        
        wedges, texts, autotexts = ax2.pie(counts, labels=categories, autopct='%1.0f%%',
                                            colors=colors_pie, startangle=90,
                                            textprops={'fontweight': 'bold'})
        ax2.set_title('Model Performance Distribution', fontweight='bold')
        
        # 3. 統計數據
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        
        stats_text = f"""
        PERFORMANCE STATISTICS
        ═════════════════════════════════════════════════════════════════
        
        Total Models: {len(self.results)}
        
        Accuracy:
          • Mean:        {np.mean(accuracies):.4f} ({np.mean(accuracies)*100:.2f}%)
          • Median:      {np.median(accuracies):.4f} ({np.median(accuracies)*100:.2f}%)
          • Std Dev:     {np.std(accuracies):.4f}
          • Min:         {np.min(accuracies):.4f} ({np.min(accuracies)*100:.2f}%)
          • Max:         {np.max(accuracies):.4f} ({np.max(accuracies)*100:.2f}%)
        
        F1 Score:
          • Mean:        {np.mean([r['f1'] for r in self.results]):.4f}
          • Std Dev:     {np.std([r['f1'] for r in self.results]):.4f}
        
        Performance Tiers:
          • Excellent (>80%):   {excellent} coins
          • Good (70-80%):      {good} coins
          • OK (60-70%):        {ok} coins
          • Poor (<60%):        {poor} coins
        
        Status: {self._get_status(np.mean(accuracies))}
        """
        
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
                fontfamily='monospace', fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_detailed_metrics(self, figsize=(14, 8)):
        """詳細指標"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        coins = [r['coin'] for r in sorted(self.results, key=lambda x: x['accuracy'], reverse=True)]
        accuracies = [r['accuracy'] for r in sorted(self.results, key=lambda x: x['accuracy'], reverse=True)]
        f1_scores = [r['f1'] for r in sorted(self.results, key=lambda x: x['accuracy'], reverse=True)]
        epochs = [r['epochs'] for r in sorted(self.results, key=lambda x: x['accuracy'], reverse=True)]
        
        # 準確率
        axes[0, 0].bar(range(len(coins)), accuracies, color='#3498db', edgecolor='black')
        axes[0, 0].axhline(y=0.80, color='green', linestyle='--', linewidth=2, alpha=0.7)
        axes[0, 0].set_xticks(range(len(coins)))
        axes[0, 0].set_xticklabels(coins, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Accuracy', fontweight='bold')
        axes[0, 0].set_title('Accuracy by Coin', fontweight='bold')
        axes[0, 0].set_ylim([0.4, 1.0])
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # F1分數
        axes[0, 1].bar(range(len(coins)), f1_scores, color='#e74c3c', edgecolor='black')
        axes[0, 1].axhline(y=0.70, color='green', linestyle='--', linewidth=2, alpha=0.7)
        axes[0, 1].set_xticks(range(len(coins)))
        axes[0, 1].set_xticklabels(coins, rotation=45, ha='right')
        axes[0, 1].set_ylabel('F1 Score', fontweight='bold')
        axes[0, 1].set_title('F1 Score by Coin', fontweight='bold')
        axes[0, 1].set_ylim([0.0, 1.0])
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 訓練週期
        axes[1, 0].bar(range(len(coins)), epochs, color='#2ecc71', edgecolor='black')
        axes[1, 0].set_xticks(range(len(coins)))
        axes[1, 0].set_xticklabels(coins, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Epochs', fontweight='bold')
        axes[1, 0].set_title('Training Epochs Needed', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 綜合評分
        combined_scores = [(acc + f1) / 2 for acc, f1 in zip(accuracies, f1_scores)]
        axes[1, 1].bar(range(len(coins)), combined_scores, color='#9b59b6', edgecolor='black')
        axes[1, 1].axhline(y=0.75, color='green', linestyle='--', linewidth=2, alpha=0.7)
        axes[1, 1].set_xticks(range(len(coins)))
        axes[1, 1].set_xticklabels(coins, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Combined Score', fontweight='bold')
        axes[1, 1].set_title('Combined (Accuracy + F1) / 2', fontweight='bold')
        axes[1, 1].set_ylim([0.0, 1.0])
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _get_status(self, mean_acc):
        if mean_acc > 0.85:
            return "✓ EXCELLENT! Achieved 88-93% target!"
        elif mean_acc > 0.80:
            return "✓ VERY GOOD! Close to 88-93%!"
        elif mean_acc > 0.75:
            return "~ GOOD! Approaching target"
        elif mean_acc > 0.70:
            return "~ OK - Needs further tuning"
        else:
            return "✗ Further investigation needed"
    
    def generate_report(self):
        """生成完整報告"""
        print("\n" + "="*90)
        print("COMPREHENSIVE PREDICTION REPORT")
        print("="*90)
        
        print("\n[TOP PERFORMERS]")
        top_5 = sorted(self.results, key=lambda x: x['accuracy'], reverse=True)[:5]
        for i, result in enumerate(top_5, 1):
            print(f"{i}. {result['coin']:12s} | Accuracy: {result['accuracy']:.4f} | F1: {result['f1']:.4f}")
        
        print("\n[NEEDS IMPROVEMENT]")
        bottom_5 = sorted(self.results, key=lambda x: x['accuracy'])[:5]
        for i, result in enumerate(bottom_5, 1):
            print(f"{i}. {result['coin']:12s} | Accuracy: {result['accuracy']:.4f} | F1: {result['f1']:.4f}")
        
        print("\n[STATISTICS]")
        accuracies = [r['accuracy'] for r in self.results]
        print(f"Mean Accuracy:     {np.mean(accuracies):.4f}")
        print(f"Median Accuracy:   {np.median(accuracies):.4f}")
        print(f"Std Dev:           {np.std(accuracies):.4f}")
        print(f"Min Accuracy:      {np.min(accuracies):.4f}")
        print(f"Max Accuracy:      {np.max(accuracies):.4f}")
        
        excellent = len([r for r in self.results if r['accuracy'] > 0.80])
        good = len([r for r in self.results if 0.70 <= r['accuracy'] <= 0.80])
        print(f"\nExcellent (>80%):  {excellent} coins")
        print(f"Good (70-80%):     {good} coins")
        
        print("\n" + "="*90)


# 使用示例
if __name__ == "__main__":
    # 這是範例數據 - 請用實際的模型結果替換
    example_results = [
        {'coin': 'BTCUSDT', 'accuracy': 0.8234, 'f1': 0.8156, 'epochs': 45, 'tn': 120, 'fp': 25, 'fn': 15, 'tp': 110},
        {'coin': 'ETHUSDT', 'accuracy': 0.7856, 'f1': 0.7723, 'epochs': 52, 'tn': 105, 'fp': 30, 'fn': 20, 'tp': 95},
        {'coin': 'SOLUSDT', 'accuracy': 0.7634, 'f1': 0.7512, 'epochs': 48, 'tn': 98, 'fp': 35, 'fn': 25, 'tp': 87},
        {'coin': 'BNBUSDT', 'accuracy': 0.8421, 'f1': 0.8267, 'epochs': 50, 'tn': 125, 'fp': 22, 'fn': 18, 'tp': 115},
        {'coin': 'AVAXUSDT', 'accuracy': 0.7923, 'f1': 0.7845, 'epochs': 46, 'tn': 110, 'fp': 28, 'fn': 22, 'tp': 100},
        {'coin': 'ADAUSDT', 'accuracy': 0.7234, 'f1': 0.7112, 'epochs': 55, 'tn': 90, 'fp': 40, 'fn': 30, 'tp': 80},
        {'coin': 'XRPUSDT', 'accuracy': 0.8156, 'f1': 0.8034, 'epochs': 49, 'tn': 118, 'fp': 27, 'fn': 20, 'tp': 105},
        {'coin': 'UNIUSDT', 'accuracy': 0.7834, 'f1': 0.7712, 'epochs': 51, 'tn': 104, 'fp': 32, 'fn': 24, 'tp': 92},
    ]
    
    # 建立可視化器
    visualizer = PredictionVisualizer(example_results)
    
    # 生成報告
    visualizer.generate_report()
    
    # 繪製圖表
    print("\n[Generating visualizations...]")
    visualizer.plot_accuracy_distribution()
    visualizer.plot_accuracy_vs_f1()
    visualizer.plot_confusion_matrices()
    visualizer.plot_performance_summary()
    visualizer.plot_detailed_metrics()
    
    print("\n" + "="*90)
    print("Visualization complete!")
    print("="*90)
