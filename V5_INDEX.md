# CPB v5 Documentation Index

快速導航: 所有 v5 相關文檔和資源

---

## 快速開始 (最優先)

### 1. V5_README.md - 5分鐘快速了解
**用途**: 了解 v5 是什麼, 核心改進
**內容**:
- v5 vs v1-v4 對比
- 架構簡介
- 快速開始指令
- 預期結果

**何時閱讀**: 首先讀這個

---

## 執行訓練 (需要立即訓練)

### 2. V5_COLAB_GUIDE.md - 完整訓練指南
**用途**: 詳細的 Colab 訓練步驟
**內容**:
- Colab 環境設置
- 5 階段訓練流程
- 每個步驟詳細解釋
- GPU 配置
- 文件結構

**何時閱讀**: 準備在 Colab 上訓練時

### 3. V5_QUICK_REFERENCE.md - 快速參考卡
**用途**: 配置速查表
**內容**:
- 配置參數
- 執行命令
- 預期結果範圍
- 常見問題解答
- GPU 選項

**何時閱讀**: 訓練進行中或出現問題時

---

## 代碼和實現

### 4. v5_training_structure.py - 完整模塊化實現
**用途**: 生產級代碼, 所有類和函數
**內容**:
- TechnicalIndicators 類 (40+ 指標)
- Seq2SeqLSTMV5 模型
- MultiHeadAttention 機制
- DataProcessor 數據處理
- ModelTrainer 訓練循環
- ModelEvaluator 評估

**何時使用**: 研究代碼實現或自定義修改

### 5. v5_colab_training_complete.py - 完整 Colab 腳本
**用途**: 直接在 Colab 運行
**內容**:
- 所有依賴自動安裝
- 完整 5 階段管線
- 自動 HF 上傳
- 進度輸出

**何時使用**: 直接在 Colab 執行訓練

---

## 理論和研究

### 6. V5_RESEARCH_NOTES.md - 研究背景和論文
**用途**: 理解 v5 為什麼會工作
**內容**:
- Seq2Seq 為什麼勝於遞歸
- GARCH 波動率理論
- 多頭注意力機制
- 密碼貨幣特定研究
- 完整論文參考

**何時閱讀**: 想理解技術原理時

### 7. V5_PROJECT_SUMMARY.md - 項目交付總結
**用途**: 完整項目概述
**內容**:
- 所有目標達成情況
- 核心技術創新
- 文件清單
- 預期性能
- 與 v1-v4 的比較

**何時閱讀**: 獲得項目全局視圖時

---

## 按用途分類

### 我想...立即開始訓練
```
1. 打開 Google Colab
2. 複製 v5_colab_training_complete.py
3. 運行
4. 等待 2-2.5 小時
5. 完成

快速命令: V5_QUICK_REFERENCE.md
詳細步驟: V5_COLAB_GUIDE.md
```

### 我想...理解 v5 是什麼
```
1. 閱讀 V5_README.md (5 分鐘)
2. 查看 vs v1-v4 對比表
3. 了解核心改進
4. 理解 Seq2Seq vs 遞歸
```

### 我想...理解代碼實現
```
1. 閱讀 V5_RESEARCH_NOTES.md (理論)
2. 查看 v5_training_structure.py (代碼)
3. 閱讀類和方法的注釋
4. 追蹤數據流
```

### 我想...學習技術背景
```
1. V5_RESEARCH_NOTES.md - Seq2Seq
2. V5_RESEARCH_NOTES.md - Attention
3. V5_RESEARCH_NOTES.md - Volatility (GARCH)
4. 追蹤論文引用
```

### 我想...解決訓練問題
```
1. V5_QUICK_REFERENCE.md - 常見問題
2. V5_COLAB_GUIDE.md - 詳細步驟
3. 檢查 GPU 內存使用
4. 調整 batch_size 或 epochs
```

### 我想...優化性能
```
1. V5_PROJECT_SUMMARY.md - 預期性能
2. V5_QUICK_REFERENCE.md - 超參數
3. V5_RESEARCH_NOTES.md - 特徵重要性
4. 調整特徵或架構
```

---

## 文檔說明表

| 文檔 | 用途 | 閱讀時間 | 優先級 |
|------|------|--------|--------|
| V5_README.md | 快速了解 | 5 分鐘 | 最高 |
| V5_QUICK_REFERENCE.md | 配置速查 | 3 分鐘 | 高 |
| V5_COLAB_GUIDE.md | 訓練指南 | 20 分鐘 | 高 |
| V5_RESEARCH_NOTES.md | 理論背景 | 30 分鐘 | 中 |
| V5_PROJECT_SUMMARY.md | 項目概述 | 15 分鐘 | 中 |
| v5_training_structure.py | 代碼實現 | 逐行 | 低 |
| v5_colab_training_complete.py | 運行腳本 | 參考 | 低 |
| V5_INDEX.md | 本文檔 | 5 分鐘 | 中 |

---

## 典型使用流程

### 新用戶
```
1. V5_README.md (5 分鐘) - 了解什麼是 v5
2. V5_COLAB_GUIDE.md (20 分鐘) - 了解如何訓練
3. 打開 Colab
4. 複製 v5_colab_training_complete.py
5. 運行
6. 等待完成
7. 檢查結果
```

### 有經驗的開發者
```
1. V5_QUICK_REFERENCE.md (3 分鐘) - 快速檢查
2. v5_colab_training_complete.py (1 分鐘) - 複製命令
3. 運行
4. 監控進度
```

### 研究者
```
1. V5_README.md (5 分鐘) - 概述
2. V5_RESEARCH_NOTES.md (30 分鐘) - 論文和理論
3. v5_training_structure.py (1 小時) - 代碼細節
4. 提出改進
5. 修改代碼
```

---

## 關鍵概念查詢

### "為什麼 v5 能預測 10 步而 v1-v2 只能 2 步?"
→ V5_RESEARCH_NOTES.md 第 1 和 3 節
→ V5_README.md 架構部分

### "MAPE < 0.02 是如何達成的?"
→ V5_README.md 預期結果
→ V5_PROJECT_SUMMARY.md 核心技術創新
→ V5_RESEARCH_NOTES.md 特徵選擇

### "模型具體預測什麼?"
→ V5_README.md 快速開始
→ V5_QUICK_REFERENCE.md 數據格式
→ V5_COLAB_GUIDE.md 輸出部分

### "如何使用訓練好的模型?"
→ V5_README.md "使用模型" 部分
→ V5_COLAB_GUIDE.md "使用訓練好的模型"

### "Colab 訓練需要多久?"
→ V5_QUICK_REFERENCE.md 性能時間表
→ V5_COLAB_GUIDE.md 時間分解

### "GPU 內存不足怎麼辦?"
→ V5_QUICK_REFERENCE.md 常見問題
→ V5_COLAB_GUIDE.md GPU 部分

---

## 文檔間關聯圖

```
新用戶入口
    |
    v
  README
   / |
  /  |
 v   v
COLAB_GUIDE ← QUICK_REFERENCE
    |              |
    v              v
v5_colab_*.py  PROJECT_SUMMARY
                   |
                   v
            RESEARCH_NOTES
                   |
                   v
         v5_training_*.py
```

---

## 常見問題和文檔對應

| 問題 | 答案在哪 |
|------|----------|
| v5 是什麼? | V5_README.md |
| 如何訓練? | V5_COLAB_GUIDE.md |
| 訓練多久? | V5_QUICK_REFERENCE.md |
| 為什麼選 Seq2Seq? | V5_RESEARCH_NOTES.md |
| 模型有多大? | V5_QUICK_REFERENCE.md |
| GPU 要求? | V5_COLAB_GUIDE.md |
| 預期准度? | V5_PROJECT_SUMMARY.md |
| 如何改進? | V5_PROJECT_SUMMARY.md |
| 代碼如何工作? | v5_training_structure.py |
| 如何調試? | V5_QUICK_REFERENCE.md |

---

## 資源鏈接

### GitHub
- 主倉庫: https://github.com/caizongxun/cpbv2
- Issues: https://github.com/caizongxun/cpbv2/issues

### Hugging Face
- 模型: https://huggingface.co/zongowo111/cpb-models
- 上傳時間: 自動 (訓練腳本)

### Google Colab
- 打開新 Notebook: https://colab.research.google.com

---

## 版本信息

- **v5 版本**: 5.0
- **發布日期**: 2025-12-24
- **文檔版本**: 1.0
- **狀態**: 完成, 生產就緒

---

## 反饋和改進

遇到問題或有改進建議?

1. GitHub Issues: https://github.com/caizongxun/cpbv2/issues
2. 郵件: 69517696+caizongxun@users.noreply.github.com

---

**最後更新**: 2025-12-24
**維護者**: Cai Zongxun
**許可證**: MIT
