# ============================================================================
# 前 20+ 主流币种配置
# 根据 2025 市值排名
# ============================================================================

# 前 20 主流币种，按市值排名
TOP_20_COINS = [
    'BTCUSDT',      # 1. Bitcoin
    'ETHUSDT',      # 2. Ethereum  
    'XRPUSDT',      # 3. XRP
    'BNBUSDT',      # 4. Binance Coin
    'SOLUSDT',      # 5. Solana
    'ADAUSDT',      # 6. Cardano
    'TRXUSDT',      # 7. TRON
    'DOGEUSDT',     # 8. Dogecoin
    'AVAXUSDT',     # 9. Avalanche
    'LINKUSDT',     # 10. Chainlink
    'POLUSDT',      # 11. Polygon (prev. MATIC) - 注意：已改名为 POL
    'LTCUSDT',      # 12. Litecoin
    'BCHUSDT',      # 13. Bitcoin Cash
    'ATOMUSDT',     # 14. Cosmos
    'APTUSDT',      # 15. Aptos
    'FILUSDT',      # 16. Filecoin
    'SUIUSDT',      # 17. Sui
    'ARBUSDT',      # 18. Arbitrum
    'NEARUSDT',     # 19. NEAR
    'INJUSDT',      # 20. Injective
]

# 扩展: 前 30 主流币种
TOP_30_COINS = TOP_20_COINS + [
    'OPUSDT',       # 21. Optimism
    'MKRUSDT',      # 22. Maker
    'UNIUSDT',      # 23. Uniswap
    'VETUSD',       # 24. VeChain (note: may use VETUSDT)
    'WAVEUSDT',     # 25. Waves (if available)
    'KASUSDT',      # 26. Kaspa
    'STXUSDT',      # 27. Stacks
    'XMRUSDT',      # 28. Monero
    'ZECUSDT',      # 29. Zcash
    'ZLUSDT',       # 30. Zilliqa
]

# 最安全的方案：使用前 15 稳定币种
TOP_15_COINS_STABLE = [
    'BTCUSDT',      # Bitcoin
    'ETHUSDT',      # Ethereum
    'BNBUSDT',      # BNB
    'SOLUSDT',      # Solana
    'XRPUSDT',      # XRP
    'ADAUSDT',      # Cardano
    'TRXUSDT',      # TRON
    'AVAXUSDT',     # Avalanche
    'LINKUSDT',     # Chainlink
    'POLUSDT',      # Polygon (POL)
    'DOGEUSDT',     # Dogecoin
    'LTCUSDT',      # Litecoin
    'BCHUSDT',      # Bitcoin Cash
    'ATOMUSDT',     # Cosmos
    'APTUSDT',      # Aptos
]

# 上午 10 中文正串：平衡速度和下载時間
TOP_10_COINS = TOP_20_COINS[:10]

# 厚提示
"""
注意事項：

1. MATIC 已改名为 POL
   - 2024年9月4日正式升级
   - 所有 MATIC 符号 1:1 自动转换为 POL
   - Binance 交易对：POLUSDT 为主
   - 一些旧交易所可能仍有 MATICSDT，但新标准是 POL

2. 下载问题排查顺序：
   - 如果 POL 下载失败，请检查：
     a) 是否是 Binance US API 限制
     b) 是否何一个源提供了 POL/MATIC 数据
     c) 是否需要使用 MATICSDT 的旁信

3. 下载数据准大利提示：
   - 前 20 匯提趣业习驅
   - 前 10 推荐用来运行完整训练。
   - 前 5 最佳新暂款（其它下载時間久、數串長度比較大、斷线可下载碩抨分契鎄ヨ）

4. 上午打竪筆數：
   - 如果 GPU 超会/OOM，需下輙：
     a) batch_size: 32 → 16
     b) epochs: 30 → 20
     c) n_features: 30 → 20
     d) lookback: 60 → 30
"""

if __name__ == '__main__':
    print('Available Configurations:')
    print(f'\nTOP_10_COINS ({len(TOP_10_COINS)} coins):')
    for i, coin in enumerate(TOP_10_COINS, 1):
        print(f'  {i:2d}. {coin}')
    
    print(f'\nTOP_20_COINS ({len(TOP_20_COINS)} coins):')
    for i, coin in enumerate(TOP_20_COINS, 1):
        print(f'  {i:2d}. {coin}')
    
    print(f'\nTOP_30_COINS ({len(TOP_30_COINS)} coins):')
    for i, coin in enumerate(TOP_30_COINS, 1):
        print(f'  {i:2d}. {coin}')
    
    print('\nIMPORTANT: MATIC renamed to POL on Sept 4, 2024')
    print('Binance trading pair: POLUSDT')
