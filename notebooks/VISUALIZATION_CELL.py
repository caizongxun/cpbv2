#!/usr/bin/env python3
# ============================================================================
# VISUALIZATION CELL - 預測線圖 vs 實際價格對比
# 直接套製到 Colab 上面的代碼之後執行
# ============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
from datetime import datetime

print('\n' + '='*90)
print('VISUALIZATION - PREDICTIONS vs ACTUAL PRICES')
print('='*90)

# ============================================================================
# STEP 1: 重新構備所有必要數據
# ============================================================================

print('\n[STEP 1] Reconstructing prediction data...')

all_predictions = {}

for coin in all_data:
    try:
        print(f'  {coin}...', end=' ', flush=True)
        
        df = all_data[coin]
        
        # 重新計算特徵
        fe = EnhancedFeatureEngineer(df)
        df_features = fe.calculate_all()
        feature_cols = fe.get_features()
        
        # 重新生成目標
        target_gen = ImprovedTargetGenerator(df_features['close'].values)
        y, valid_indices = target_gen.get_labels()
        
        # 預處理
        prep = EnhancedPreprocessor(df_features.iloc[valid_indices], lookback=CONFIG['lookback'])
        features, _ = prep.prepare(feature_cols)
        X, y_seq = prep.create_sequences(y)
        data = prep.split_data(X, y_seq)
        
        if len(X) < 50:
            print('X (insufficient data)')
            continue
        
        # 獲取模型（需要重新加載或使用已訓練的模型）
        model = ImprovedLSTMModel(
            input_size=features.shape[-1],
            hidden_size=128,
            num_layers=3,
            dropout=0.4
        )
        
        # 訓練模型
        trainer = EnhancedTrainer(model, device=device)
        history = trainer.train(data['X_train'], data['y_train'], data['X_val'], data['y_val'], CONFIG)
        
        # 獲取完整預測
        model.eval()
        X_all_t = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            logits = model(X_all_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = logits.argmax(dim=1).cpu().numpy()
            y_pred_probs = probs[:, 1]  # 向上概率
        
        # 映射回原始時間序列
        timestamps = df.iloc[valid_indices + CONFIG['lookback']]['timestamp'].values
        prices = df.iloc[valid_indices + CONFIG['lookback']]['close'].values
        
        accuracy = accuracy_score(y_seq, y_pred)
        
        all_predictions[coin] = {
            'timestamps': timestamps,
            'prices': prices,
            'predictions': y_pred,
            'probabilities': y_pred_probs,
            'actual_labels': y_seq,
            'accuracy': accuracy
        }
        
        print(f'OK ({len(y_pred)} predictions)')
    except Exception as e:
        print(f'X {str(e)[:40]}')

print(f'\nRecovered predictions for {len(all_predictions)} coins')

# ============================================================================
# STEP 2: 創建綜合視覺化
# ============================================================================

print('\n[STEP 2] Creating visualizations...')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(20, 16))
fig.suptitle('CPB v2: Prediction Performance Analysis - Predicted Direction vs Actual Price',
             fontsize=18, fontweight='bold', y=0.995)

colors = {
    'price': '#1f77b4',      # Blue - Actual price
    'up': '#2ca02c',         # Green - Predicted Up
    'down': '#d62728',       # Red - Predicted Down
    'correct': '#17becf',    # Cyan - Correct prediction
    'wrong': '#ff7f0e',      # Orange - Wrong prediction
}

for idx, (coin, data) in enumerate(sorted(all_predictions.items()), 1):
    # ========== Subplot 1: Price + Prediction Direction ==========
    ax1 = plt.subplot(3, 3, idx*3-2)
    
    timestamps = pd.to_datetime(data['timestamps'])
    prices = data['prices']
    predictions = data['predictions']
    actual = data['actual_labels']
    
    # Calculate correct/wrong
    correct = predictions == actual
    
    # Plot price line
    ax1.plot(timestamps, prices, color=colors['price'], linewidth=2, label='Actual Price', zorder=3)
    
    # Plot prediction points
    up_mask = predictions == 1
    down_mask = predictions == 0
    
    ax1.scatter(timestamps[up_mask], prices[up_mask],
               color=colors['up'], s=20, alpha=0.6, label='Predicted Up', zorder=4)
    ax1.scatter(timestamps[down_mask], prices[down_mask],
               color=colors['down'], s=20, alpha=0.6, label='Predicted Down', zorder=4)
    
    ax1.set_title(f'{coin} - Price & Prediction Direction', fontweight='bold')
    ax1.set_ylabel('Price (USDT)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # ========== Subplot 2: Accuracy Display ==========
    ax2 = plt.subplot(3, 3, idx*3-1)
    
    correct_count = np.sum(correct)
    wrong_count = np.sum(~correct)
    accuracy = data['accuracy']
    
    bars = ax2.bar(['Correct', 'Wrong'],
                 [correct_count, wrong_count],
                 color=[colors['correct'], colors['wrong']],
                 alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, [correct_count, wrong_count]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}\n({val/len(correct)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    ax2.set_title(f'{coin} - Accuracy: {accuracy*100:.2f}%',
                  fontweight='bold', color='darkgreen' if accuracy > 0.85 else 'darkorange')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ========== Subplot 3: Probability Distribution ==========
    ax3 = plt.subplot(3, 3, idx*3)
    
    probabilities = data['probabilities']
    
    ax3.hist(probabilities, bins=30, color=colors['price'],
             alpha=0.7, edgecolor='black', linewidth=1)
    
    ax3.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Boundary (0.5)')
    ax3.axvline(np.mean(probabilities), color='green', linestyle='--',
               linewidth=2, label=f'Mean ({np.mean(probabilities):.3f})')
    
    ax3.set_title(f'{coin} - Prediction Probability Distribution', fontweight='bold')
    ax3.set_xlabel('Upward Probability')
    ax3.set_ylabel('Frequency')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('predictions_vs_prices.png', dpi=150, bbox_inches='tight')
print('  Saved: predictions_vs_prices.png')
plt.show()

# ============================================================================
# STEP 3: Detailed Comparison Table
# ============================================================================

print('\n[STEP 3] Detailed comparison table')

comparison_data = []
for coin, data in sorted(all_predictions.items()):
    correct = np.sum(data['predictions'] == data['actual_labels'])
    total = len(data['predictions'])
    accuracy = data['accuracy']
    avg_prob = np.mean(data['probabilities'])
    
    comparison_data.append({
        'Coin': coin,
        'Total Signals': total,
        'Correct': correct,
        'Wrong': total - correct,
        'Accuracy': f'{accuracy*100:.2f}%',
        'Avg Probability': f'{avg_prob:.4f}',
        'Confidence': 'HIGH' if avg_prob > 0.6 else 'MEDIUM' if avg_prob > 0.5 else 'LOW'
    })

comparison_df = pd.DataFrame(comparison_data)
print('\n' + comparison_df.to_string(index=False))

# ============================================================================
# STEP 4: Time-Series Detailed Analysis
# ============================================================================

print('\n[STEP 4] Creating time-series detail analysis...')

fig, axes = plt.subplots(len(all_predictions), 2, figsize=(18, 5*len(all_predictions)))
if len(all_predictions) == 1:
    axes = axes.reshape(1, -1)

for idx, (coin, data) in enumerate(sorted(all_predictions.items())):
    
    timestamps = pd.to_datetime(data['timestamps'])
    prices = data['prices']
    predictions = data['predictions']
    actual = data['actual_labels']
    probabilities = data['probabilities']
    
    # ========== Left: Price + Prediction Signal ==========
    ax_left = axes[idx, 0]
    
    # Plot price line
    ax_left.plot(timestamps, prices, color=colors['price'], linewidth=2.5, label='Actual Price')
    
    # Background color - based on actual direction
    for i in range(len(actual)-1):
        if actual[i] == 1:
            ax_left.axvspan(timestamps[i], timestamps[i+1], alpha=0.1, color='green')
        else:
            ax_left.axvspan(timestamps[i], timestamps[i+1], alpha=0.1, color='red')
    
    # Plot prediction points
    correct = predictions == actual
    ax_left.scatter(timestamps[correct], prices[correct],
                 color=colors['correct'], s=50, alpha=0.8,
                 marker='^', label='Correct Prediction', zorder=5, edgecolors='black')
    ax_left.scatter(timestamps[~correct], prices[~correct],
                 color=colors['wrong'], s=50, alpha=0.8,
                 marker='v', label='Wrong Prediction', zorder=5, edgecolors='black')
    
    ax_left.set_title(f'{coin} - Price Trend & Prediction Accuracy (Accuracy: {data["accuracy"]*100:.2f}%)',
                    fontweight='bold', fontsize=11)
    ax_left.set_ylabel('Price (USDT)', fontweight='bold')
    ax_left.legend(loc='best', fontsize=9)
    ax_left.grid(True, alpha=0.3)
    ax_left.tick_params(axis='x', rotation=45)
    
    # ========== Right: Prediction Probability Time Series ==========
    ax_right = axes[idx, 1]
    
    # Plot probability line
    ax_right.plot(timestamps, probabilities, color='purple', linewidth=2, label='Upward Probability')
    ax_right.fill_between(timestamps, 0.5, probabilities,
                        where=(probabilities >= 0.5),
                        color='green', alpha=0.3, label='Predicted Up')
    ax_right.fill_between(timestamps, 0.5, probabilities,
                        where=(probabilities < 0.5),
                        color='red', alpha=0.3, label='Predicted Down')
    
    # Add boundary lines
    ax_right.axhline(0.5, color='black', linestyle='--', linewidth=1.5, label='Decision Boundary')
    ax_right.axhline(0.6, color='green', linestyle=':', alpha=0.5)
    ax_right.axhline(0.4, color='red', linestyle=':', alpha=0.5)
    
    # Background - based on actual direction
    for i in range(len(actual)-1):
        if actual[i] == 1:
            ax_right.axvspan(timestamps[i], timestamps[i+1], alpha=0.05, color='green')
        else:
            ax_right.axvspan(timestamps[i], timestamps[i+1], alpha=0.05, color='red')
    
    ax_right.set_title(f'{coin} - Prediction Probability Timeline', fontweight='bold', fontsize=11)
    ax_right.set_ylabel('Upward Probability', fontweight='bold')
    ax_right.set_ylim(0, 1)
    ax_right.legend(loc='best', fontsize=9)
    ax_right.grid(True, alpha=0.3)
    ax_right.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('detailed_predictions_analysis.png', dpi=150, bbox_inches='tight')
print('  Saved: detailed_predictions_analysis.png')
plt.show()

# ============================================================================
# STEP 5: Statistical Summary
# ============================================================================

print('\n' + '='*90)
print('VISUALIZATION SUMMARY')
print('='*90)

print(f'\nGenerated Charts:')
print(f'  1. predictions_vs_prices.png (3x3 grid - 3 subplots per coin)')
print(f'  2. detailed_predictions_analysis.png (Detailed time-series analysis)')

print(f'\nVisualization Content per Coin:')
print(f'  • Price trend + Prediction direction (Up/Down signals)')
print(f'  • Accuracy statistics (Correct/Wrong count and percentage)')
print(f'  • Prediction probability distribution (Histogram)')
print(f'  • Detailed time-series comparison')
print(f'  • Prediction signal accuracy markers (^ for Correct, v for Wrong)')

print(f'\nChart Legend:')
print(f'  Green background: Actual price moved UP')
print(f'  Red background: Actual price moved DOWN')
print(f'  ^ Marker: Correct prediction')
print(f'  v Marker: Wrong prediction')
print(f'  Purple line: Model\'s upward probability')
print(f'  0.5 line: Decision boundary')

print('\n' + '='*90)
print('VISUALIZATION COMPLETE')
print('='*90 + '\n')
