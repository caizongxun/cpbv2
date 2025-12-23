#!/usr/bin/env python3
# ============================================================================
# VISUALIZATION CELL - FIXED
# 預測線圖 vs 實際價格對比
# ============================================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

print('\n' + '='*90)
print('VISUALIZATION - PREDICTIONS vs ACTUAL PRICES (FIXED)')
print('='*90)

# ============================================================================
# STEP 1: 重新構備所有必要數據
# ============================================================================

print('\n[STEP 1] Reconstructing prediction data...')

all_predictions = {}

for coin in all_data:
    try:
        print(f'  {coin}...', end=' ', flush=True)
        
        df = all_data[coin].copy()
        
        # 重新計算特徵
        fe = EnhancedFeatureEngineer(df)
        df_features = fe.calculate_all()
        feature_cols = fe.get_features()
        
        # 重新生成目標
        target_gen = ImprovedTargetGenerator(df_features['close'].values)
        y, valid_indices = target_gen.get_labels()
        
        # 預處理
        df_valid = df_features.iloc[valid_indices].reset_index(drop=True)
        prep = EnhancedPreprocessor(df_valid, lookback=CONFIG['lookback'])
        features, _ = prep.prepare(feature_cols)
        X, y_seq = prep.create_sequences(y)
        data = prep.split_data(X, y_seq)
        
        if len(X) < 50:
            print('X (insufficient data)')
            continue
        
        # 訓練模型
        model = ImprovedLSTMModel(
            input_size=features.shape[-1],
            hidden_size=128,
            num_layers=3,
            dropout=0.4
        )
        
        trainer = EnhancedTrainer(model, device=device)
        history = trainer.train(data['X_train'], data['y_train'], data['X_val'], data['y_val'], CONFIG)
        
        # 獲取完整預測
        model.eval()
        X_all_t = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            logits = model(X_all_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = logits.argmax(dim=1).cpu().numpy()
            y_pred_probs = probs[:, 1]
        
        # 正確映射回原始時間序列
        # valid_indices 中的索引對應 df_features 中的行
        # 加上 lookback 的偏移
        start_idx = valid_indices[0] + CONFIG['lookback']
        end_idx = start_idx + len(y_seq)
        
        # 從原始 df 中獲取時間戳和價格
        actual_indices = np.arange(start_idx, min(end_idx, len(df)))
        
        if len(actual_indices) != len(y_seq):
            print(f'X (index mismatch: {len(actual_indices)} vs {len(y_seq)})')
            continue
        
        timestamps = df.iloc[actual_indices]['timestamp'].values
        prices = df.iloc[actual_indices]['close'].values
        
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
        print(f'X {str(e)[:50]}')
        import traceback
        traceback.print_exc()

print(f'\nRecovered predictions for {len(all_predictions)} coins')

if len(all_predictions) == 0:
    print('ERROR: No predictions recovered. Check data alignment.')
    exit(1)

# ============================================================================
# STEP 2: 創建綜合視覺化
# ============================================================================

print('\n[STEP 2] Creating visualizations...')

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 9

fig = plt.figure(figsize=(20, 16))
fig.suptitle('CPB v2: Prediction Performance Analysis - Predicted Direction vs Actual Price',
             fontsize=18, fontweight='bold', y=0.995)

colors = {
    'price': '#1f77b4',
    'up': '#2ca02c',
    'down': '#d62728',
    'correct': '#17becf',
    'wrong': '#ff7f0e',
}

for idx, (coin, data) in enumerate(sorted(all_predictions.items()), 1):
    # ========== Subplot 1: Price + Prediction Direction ==========
    ax1 = plt.subplot(3, 3, idx*3-2)
    
    timestamps = pd.to_datetime(data['timestamps'])
    prices = data['prices']
    predictions = data['predictions']
    actual = data['actual_labels']
    
    correct = predictions == actual
    
    ax1.plot(timestamps, prices, color=colors['price'], linewidth=2, label='Actual Price', zorder=3)
    
    up_mask = predictions == 1
    down_mask = predictions == 0
    
    ax1.scatter(timestamps[up_mask], prices[up_mask],
               color=colors['up'], s=20, alpha=0.6, label='Predicted Up', zorder=4)
    ax1.scatter(timestamps[down_mask], prices[down_mask],
               color=colors['down'], s=20, alpha=0.6, label='Predicted Down', zorder=4)
    
    ax1.set_title(f'{coin} - Price & Prediction Direction', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Price (USDT)', fontsize=9)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
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
                ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    color_title = 'darkgreen' if accuracy > 0.85 else 'darkorange'
    ax2.set_title(f'{coin} - Accuracy: {accuracy*100:.2f}%',
                  fontweight='bold', fontsize=10, color=color_title)
    ax2.set_ylabel('Count', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(labelsize=8)
    
    # ========== Subplot 3: Probability Distribution ==========
    ax3 = plt.subplot(3, 3, idx*3)
    
    probabilities = data['probabilities']
    
    ax3.hist(probabilities, bins=30, color=colors['price'],
             alpha=0.7, edgecolor='black', linewidth=1)
    
    ax3.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision (0.5)')
    ax3.axvline(np.mean(probabilities), color='green', linestyle='--',
               linewidth=2, label=f'Mean ({np.mean(probabilities):.3f})')
    
    ax3.set_title(f'{coin} - Probability Distribution', fontweight='bold', fontsize=10)
    ax3.set_xlabel('Upward Probability', fontsize=9)
    ax3.set_ylabel('Frequency', fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('predictions_vs_prices.png', dpi=150, bbox_inches='tight')
print('  Saved: predictions_vs_prices.png')
plt.show()

# ============================================================================
# STEP 3: Detailed Comparison Table
# ============================================================================

print('\n[STEP 3] Detailed comparison table\n')

comparison_data = []
for coin, data in sorted(all_predictions.items()):
    correct = np.sum(data['predictions'] == data['actual_labels'])
    total = len(data['predictions'])
    accuracy = data['accuracy']
    avg_prob = np.mean(data['probabilities'])
    
    comparison_data.append({
        'Coin': coin,
        'Total': total,
        'Correct': correct,
        'Wrong': total - correct,
        'Accuracy': f'{accuracy*100:.2f}%',
        'Avg Prob': f'{avg_prob:.4f}',
        'Confidence': 'HIGH' if avg_prob > 0.6 else 'MEDIUM' if avg_prob > 0.5 else 'LOW'
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# ============================================================================
# STEP 4: Time-Series Detailed Analysis
# ============================================================================

print('\n[STEP 4] Creating time-series detail analysis...')

num_coins = len(all_predictions)
fig, axes = plt.subplots(num_coins, 2, figsize=(18, 5*num_coins))
if num_coins == 1:
    axes = axes.reshape(1, -1)

for idx, (coin, data) in enumerate(sorted(all_predictions.items())):
    
    timestamps = pd.to_datetime(data['timestamps'])
    prices = data['prices']
    predictions = data['predictions']
    actual = data['actual_labels']
    probabilities = data['probabilities']
    
    # ========== Left: Price + Prediction Signal ==========
    ax_left = axes[idx, 0]
    
    ax_left.plot(timestamps, prices, color=colors['price'], linewidth=2.5, label='Actual Price')
    
    # Background color based on actual direction
    for i in range(len(actual)-1):
        if actual[i] == 1:
            ax_left.axvspan(timestamps[i], timestamps[i+1], alpha=0.1, color='green')
        else:
            ax_left.axvspan(timestamps[i], timestamps[i+1], alpha=0.1, color='red')
    
    # Plot prediction points
    correct = predictions == actual
    ax_left.scatter(timestamps[correct], prices[correct],
                 color=colors['correct'], s=50, alpha=0.8,
                 marker='^', label='Correct (^)', zorder=5, edgecolors='black')
    ax_left.scatter(timestamps[~correct], prices[~correct],
                 color=colors['wrong'], s=50, alpha=0.8,
                 marker='v', label='Wrong (v)', zorder=5, edgecolors='black')
    
    acc_text = f'{data["accuracy"]*100:.2f}%'
    ax_left.set_title(f'{coin} - Price Trend & Accuracy: {acc_text}',
                    fontweight='bold', fontsize=11)
    ax_left.set_ylabel('Price (USDT)', fontweight='bold', fontsize=10)
    ax_left.legend(loc='best', fontsize=9)
    ax_left.grid(True, alpha=0.3)
    ax_left.tick_params(axis='x', rotation=45, labelsize=8)
    
    # ========== Right: Probability Time Series ==========
    ax_right = axes[idx, 1]
    
    ax_right.plot(timestamps, probabilities, color='purple', linewidth=2, label='Upward Prob')
    ax_right.fill_between(timestamps, 0.5, probabilities,
                        where=(probabilities >= 0.5),
                        color='green', alpha=0.3, label='Predicted Up')
    ax_right.fill_between(timestamps, 0.5, probabilities,
                        where=(probabilities < 0.5),
                        color='red', alpha=0.3, label='Predicted Down')
    
    ax_right.axhline(0.5, color='black', linestyle='--', linewidth=1.5, label='Boundary (0.5)')
    ax_right.axhline(0.6, color='green', linestyle=':', alpha=0.5, linewidth=1)
    ax_right.axhline(0.4, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    # Background based on actual direction
    for i in range(len(actual)-1):
        if actual[i] == 1:
            ax_right.axvspan(timestamps[i], timestamps[i+1], alpha=0.05, color='green')
        else:
            ax_right.axvspan(timestamps[i], timestamps[i+1], alpha=0.05, color='red')
    
    ax_right.set_title(f'{coin} - Probability Timeline',
                    fontweight='bold', fontsize=11)
    ax_right.set_ylabel('Upward Probability', fontweight='bold', fontsize=10)
    ax_right.set_ylim(0, 1)
    ax_right.legend(loc='best', fontsize=9)
    ax_right.grid(True, alpha=0.3)
    ax_right.tick_params(axis='x', rotation=45, labelsize=8)

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
print(f'  1. predictions_vs_prices.png')
print(f'     - 3x3 grid with 3 subplots per coin')
print(f'  2. detailed_predictions_analysis.png')
print(f'     - Detailed time-series analysis (2 columns per coin)')

print(f'\nChart Legend:')
print(f'  🟢 Green background: Actual price went UP')
print(f'  🔴 Red background: Actual price went DOWN')
print(f'  ▲ Cyan marker: CORRECT prediction')
print(f'  ▼ Orange marker: WRONG prediction')
print(f'  🟣 Purple line: Model upward probability')

print(f'\nInterpretation Guide:')
print(f'  ▲ in GREEN zone → Correct "UP" prediction ✓')
print(f'  ▼ in GREEN zone → Wrong "DOWN" prediction ✗')
print(f'  ▲ in RED zone → Wrong "UP" prediction ✗')
print(f'  ▼ in RED zone → Correct "DOWN" prediction ✓')

print('\n' + '='*90)
print('VISUALIZATION COMPLETE')
print('='*90 + '\n')
