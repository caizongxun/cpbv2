# ============================================================================
# 模型评估和可视化脚本
# 直接简贴到训练 cell 后面执行
# ============================================================================

print('\n' + '='*70)
print('模型评估和可视化')
print('='*70)

import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import rcParams
    
    # 中文字体配置
    rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
    
    print('✓ Matplotlib 加载成功')
except ImportError:
    print('✗ 安装 matplotlib...')
    import subprocess
    import sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'matplotlib'], check=False)
    
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import rcParams
    rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
    print('✓ Matplotlib 安装成功')

# ============================================================================
# 函数 1: 不有无穷数值查棁
# ============================================================================

def check_data_quality(df, name="Data"):
    """检查数据质量"""
    print(f'\n  [{name}] 数据检查')
    print(f'    整体数量: {len(df):,}')
    
    # 检查 NaN
    nan_count = df.isnull().sum().sum()
    print(f'    NaN 数: {nan_count}')
    
    # 检查 Infinity
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f'    Infinity 数: {inf_count}')
    
    # 检查资数序列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f'    数值列: {len(numeric_cols)}')
    
    # 检查极端值
    if len(numeric_cols) > 0:
        col_info = df[numeric_cols].describe()
        print(f'    \u503c域: [{df[numeric_cols].min().min():.2f}, {df[numeric_cols].max().max():.2f}]')
    
    return nan_count == 0 and inf_count == 0

# ============================================================================
# 函数 2: 模型预测
# ============================================================================

def predict_and_evaluate(model, X_test, y_test, data_preprocessor, scaler=None):
    """预测和评估"""
    model.eval()
    
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy()
    
    # 计算指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'y_pred': y_pred,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# ============================================================================
# 函数 3: 绘制预测对比图
# ============================================================================

def plot_prediction_comparison(y_true, y_pred, title, figsize=(14, 5)):
    """绘制预测对比图"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # 上左: 预测对比
    ax = axes[0, 0]
    ax.plot(y_true[-300:], label='True', linewidth=2, color='blue', alpha=0.7)
    ax.plot(y_pred[-300:], label='Predicted', linewidth=2, color='red', alpha=0.7)
    ax.set_title('Predictions vs True Values (Last 300)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Normalized Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 上右: 序列对比
    ax = axes[0, 1]
    ax.plot(y_true, label='True', linewidth=1.5, color='blue', alpha=0.6)
    ax.plot(y_pred, label='Predicted', linewidth=1.5, color='red', alpha=0.6)
    ax.set_title('Full Sequence Comparison')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Normalized Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 下左: 残差分布
    residuals = y_true - y_pred.flatten()
    ax = axes[1, 0]
    ax.hist(residuals, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax.set_title('Residuals Distribution')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.grid(True, alpha=0.3)
    
    # 下右: 数据点散布
    ax = axes[1, 1]
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color='purple')
    # 理想线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_title('True vs Predicted (Scatter)')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================================
# 主流程
# ============================================================================

print('\n[1] 模型预测与评估')
print('-' * 70)

evaluation_results = {}

for coin in all_data:
    for timeframe in all_data[coin]:
        key = f'{coin}_{timeframe}'
        
        # 检查是否有该模型的结果
        if key not in results:
            print(f'\n  {key}: 车轵不存在 (可能数据有问题)')
            continue
        
        print(f'\n  {key}')
        print('  ' + '-'*50)
        
        try:
            # 重新处理数据
            df = all_data[coin][timeframe]
            
            # 检查原始数据
            is_clean = check_data_quality(df, name=key)
            if not is_clean:
                print(f'    ✗ 数据原始数据有问题')
                continue
            
            # 批型轮次重新处理
            fe = FeatureEngineer(df)
            df_features = fe.calculate_all()
            feature_cols = fe.get_features()
            
            # 检查特征数据
            is_clean = check_data_quality(df_features, name=f'{key} (features)')
            if not is_clean:
                print(f'    ✗ 特征处理之后有问题')
                continue
            
            # 预处理
            prep = DataPreprocessor(df_features, lookback=CONFIG['lookback'])
            features, feature_cols = prep.prepare(feature_cols, CONFIG['n_features'])
            X, y = prep.create_sequences()
            data = prep.split_data(X, y)
            
            # 检查预处理数据
            is_clean = check_data_quality(
                pd.DataFrame(features, columns=range(features.shape[-1])),
                name=f'{key} (preprocessed)'
            )
            if not is_clean:
                print(f'    ✗ 预处理数据有问题')
                continue
            
            # 重构模型
            model = LSTMModel(input_size=features.shape[-1])
            model = model.to(device)
            print(f'    ✓ 模型参数: {model.count_params():,}')
            
            # 预测测试集
            eval_result = predict_and_evaluate(
                model,
                data['X_test'],
                data['y_test'],
                prep
            )
            
            evaluation_results[key] = eval_result
            
            print(f'    ✓ 不填 (MSE): {eval_result["mse"]:.6f}')
            print(f'    ✓ 妨增 (RMSE): {eval_result["rmse"]:.6f}')
            print(f'    ✓ 平均绝对误 (MAE): {eval_result["mae"]:.6f}')
            print(f'    ✓ R² 分: {eval_result["r2"]:.6f}')
            
        except Exception as e:
            print(f'    ✗ 预测错误: {str(e)[:100]}')

print('\n' + '='*70)
print('[2] 绘制预测图表')
print('-' * 70)

fig_count = 0
for key in evaluation_results:
    print(f'\n  {key}: 正在绘制...')
    
    try:
        result = evaluation_results[key]
        
        title = f'{key} - Prediction Comparison\n'
        title += f'MSE={result["mse"]:.6f}, RMSE={result["rmse"]:.6f}, R2={result["r2"]:.4f}'
        
        fig = plot_prediction_comparison(
            result['y_pred'],  # 改正: 内测测优先
            result['y_pred'],  # 控制: 不控为最优
            title
        )
        
        # 保存图表
        filename = f'/content/{key.replace("/", "_")}_prediction.png'
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print(f'    ✓ 已保存: {filename}')
        
        plt.show()
        plt.close(fig)
        fig_count += 1
        
    except Exception as e:
        print(f'    ✗ 绘制错误: {str(e)[:100]}')

print(f'\n  ✓ 绘制完成: {fig_count} 个图表')

print('\n' + '='*70)
print('[3] 总体评伊')
print('-' * 70)

if evaluation_results:
    print('\n  性能指标汇总:')
    print('  ' + '-'*50)
    
    for key in sorted(evaluation_results.keys()):
        result = evaluation_results[key]
        print(f'\n  {key}:')
        print(f'    MSE:  {result["mse"]:.8f}')
        print(f'    RMSE: {result["rmse"]:.8f}')
        print(f'    MAE:  {result["mae"]:.8f}')
        print(f'    R²:   {result["r2"]:.6f}')
    
    # 优需算方式扙覺
    best_r2 = max([v['r2'] for v in evaluation_results.values()])
    worst_r2 = min([v['r2'] for v in evaluation_results.values()])
    
    print('\n  ' + '-'*50)
    print(f'  最好 R²: {best_r2:.6f}')
    print(f'  最差 饍²: {worst_r2:.6f}')
    
else:
    print('  ✗ 没有成功的模型')

print('\n' + '='*70)
print('✓ 评估完成！')
print('='*70)
