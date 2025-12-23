"""
CPB Web Application - V2 Model Inference

Local web interface to load and invoke V2 models
Supports 20 cryptocurrency pairs

Usage:
    python web_app.py
    
Then open: http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify
import numpy as np
import tensorflow as tf
import os
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ==================== Configuration ====================
CONFIG = {
    'model_version': 'V2',
    'pairs': [
        'BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'XRP_USDT', 'ADA_USDT',
        'BNB_USDT', 'DOGE_USDT', 'LINK_USDT', 'AVAX_USDT', 'MATIC_USDT',
        'ATOM_USDT', 'NEAR_USDT', 'FTM_USDT', 'ARB_USDT', 'OP_USDT',
        'LIT_USDT', 'STX_USDT', 'INJ_USDT', 'LUNC_USDT', 'LUNA_USDT'
    ],
    'model_dir': 'ALL_MODELS/MODEL_V2',
    'input_shape': (20, 4)  # (sequence_length, features)
}

# ==================== Model Cache ====================
model_cache = {}

def load_model(pair):
    """
    Load V2 model for a specific pair
    """
    if pair in model_cache:
        return model_cache[pair]
    
    model_path = f'{CONFIG["model_dir"]}/v2_model_{pair}.h5'
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        model_cache[pair] = model
        logger.info(f"Model loaded: {pair}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {pair}: {e}")
        return None

# ==================== Data Generation ====================
def generate_sample_klines(num_candles=20):
    """
    Generate sample OHLCV data
    In production, this would be replaced with real K-line data
    """
    start_price = 50000
    klines = []
    price = start_price
    
    for i in range(num_candles):
        daily_return = np.random.normal(0.001, 0.02)
        open_price = price
        close_price = price * (1 + daily_return)
        high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.01))
        low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.01))
        
        klines.append({
            'open': float(open_price),
            'high': float(high_price),
            'low': float(low_price),
            'close': float(close_price),
            'volume': float(np.random.uniform(100, 1000))
        })
        
        price = close_price
    
    return klines

def prepare_input(klines):
    """
    Prepare K-line data for model input
    Normalizes OHLC values
    """
    if len(klines) < CONFIG['input_shape'][0]:
        logger.warning(f"Not enough candles: {len(klines)} < {CONFIG['input_shape'][0]}")
        return None
    
    # Take last 20 candles
    klines = klines[-CONFIG['input_shape'][0]:]
    
    # Extract OHLC
    ohlc = np.array([
        [k['open'], k['high'], k['low'], k['close']]
        for k in klines
    ])
    
    # Normalize by first open price
    first_open = ohlc[0, 0]
    ohlc_normalized = (ohlc - first_open) / first_open * 100
    
    return ohlc_normalized.reshape(1, *CONFIG['input_shape'])

# ==================== Routes ====================
@app.route('/')
def index():
    """
    Main page
    """
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CPB V2 Model Inference</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                padding: 40px;
            }}
            
            h1 {{
                color: #333;
                margin-bottom: 10px;
                font-size: 28px;
            }}
            
            .subtitle {{
                color: #666;
                margin-bottom: 30px;
                font-size: 14px;
            }}
            
            .info-box {{
                background: #f0f4ff;
                border-left: 4px solid #667eea;
                padding: 15px;
                margin-bottom: 30px;
                border-radius: 6px;
            }}
            
            .info-box strong {{
                color: #333;
            }}
            
            .form-group {{
                margin-bottom: 20px;
            }}
            
            label {{
                display: block;
                margin-bottom: 8px;
                color: #333;
                font-weight: 500;
                font-size: 14px;
            }}
            
            select {{
                width: 100%;
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-size: 14px;
                background: white;
                cursor: pointer;
                transition: border-color 0.3s;
            }}
            
            select:hover {{
                border-color: #667eea;
            }}
            
            select:focus {{
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }}
            
            .button-group {{
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }}
            
            button {{
                flex: 1;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.3s;
            }}
            
            .btn-primary {{
                background: #667eea;
                color: white;
            }}
            
            .btn-primary:hover {{
                background: #5568d3;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }}
            
            .btn-secondary {{
                background: #f0f4ff;
                color: #667eea;
                border: 1px solid #667eea;
            }}
            
            .btn-secondary:hover {{
                background: #667eea;
                color: white;
            }}
            
            .results {{
                display: none;
                margin-top: 30px;
                padding-top: 30px;
                border-top: 2px solid #eee;
            }}
            
            .results.active {{
                display: block;
            }}
            
            .result-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }}
            
            .result-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }}
            
            .result-card h3 {{
                color: #333;
                font-size: 12px;
                text-transform: uppercase;
                margin-bottom: 10px;
                letter-spacing: 0.5px;
            }}
            
            .result-value {{
                font-size: 24px;
                font-weight: bold;
                color: #667eea;
            }}
            
            .result-unit {{
                font-size: 12px;
                color: #999;
                margin-top: 5px;
            }}
            
            .status {{
                padding: 12px 15px;
                border-radius: 6px;
                font-size: 14px;
                margin-bottom: 20px;
                display: none;
            }}
            
            .status.active {{
                display: block;
            }}
            
            .status.loading {{
                background: #e3f2fd;
                color: #1976d2;
                border-left: 4px solid #1976d2;
            }}
            
            .status.success {{
                background: #e8f5e9;
                color: #388e3c;
                border-left: 4px solid #388e3c;
            }}
            
            .status.error {{
                background: #ffebee;
                color: #c62828;
                border-left: 4px solid #c62828;
            }}
            
            .kline-data {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 20px;
                max-height: 200px;
                overflow-y: auto;
            }}
            
            .kline-item {{
                display: grid;
                grid-template-columns: repeat(5, 1fr);
                gap: 10px;
                padding: 8px;
                font-size: 12px;
                border-bottom: 1px solid #eee;
            }}
            
            .kline-item:last-child {{
                border-bottom: none;
            }}
            
            .kline-header {{
                font-weight: bold;
                background: #e8eaef;
                padding: 8px;
                border-radius: 4px;
                color: #667eea;
            }}
            
            .loading {{
                display: none;
                text-align: center;
            }}
            
            .loading.active {{
                display: block;
            }}
            
            .spinner {{
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 10px;
            }}
            
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
            .pairs-info {{
                display: grid;
                grid-template-columns: repeat(5, 1fr);
                gap: 10px;
                margin-top: 20px;
            }}
            
            .pair-tag {{
                background: #f0f4ff;
                color: #667eea;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
                text-align: center;
                font-weight: 500;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CPB V2 Model Inference</h1>
            <p class="subtitle">Real-time cryptocurrency price and volatility prediction</p>
            
            <div class="info-box">
                <strong>Model Version:</strong> {CONFIG['model_version']}<br>
                <strong>Output:</strong> [predicted_price, predicted_volatility]<br>
                <strong>Supported Pairs:</strong> {len(CONFIG['pairs'])} pairs
            </div>
            
            <div class="form-group">
                <label for="pair-select">Select Trading Pair:</label>
                <select id="pair-select">
                    <option value="">-- Choose a pair --</option>
                    {''.join([f'<option value="{pair}">{pair}</option>' for pair in CONFIG['pairs']])}
                </select>
            </div>
            
            <div class="form-group">
                <label for="num-candles">Number of Candles (for sample data):</label>
                <input type="number" id="num-candles" value="20" min="20" max="100" style="width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 6px;">
            </div>
            
            <div class="button-group">
                <button class="btn-primary" onclick="runInference()">Run Inference</button>
                <button class="btn-secondary" onclick="generateSampleData()">Generate Sample Data</button>
            </div>
            
            <div id="status" class="status"></div>
            <div id="loading" class="loading">
                <span class="spinner"></span>
                <span>Processing...</span>
            </div>
            
            <div id="klines" class="kline-data" style="display: none;"></div>
            
            <div id="results" class="results">
                <h2 style="color: #333; margin-bottom: 20px;">Prediction Results</h2>
                <div class="result-grid">
                    <div class="result-card">
                        <h3>Predicted Price</h3>
                        <div class="result-value" id="price-result">-</div>
                        <div class="result-unit">% change from first candle</div>
                    </div>
                    <div class="result-card">
                        <h3>Predicted Volatility</h3>
                        <div class="result-value" id="volatility-result">-</div>
                        <div class="result-unit">% standard deviation</div>
                    </div>
                </div>
            </div>
            
            <h3 style="color: #333; margin-top: 30px; margin-bottom: 15px;">Supported Cryptocurrency Pairs (20)</h3>
            <div class="pairs-info">
                {''.join([f'<div class="pair-tag">{pair}</div>' for pair in CONFIG['pairs']])}
            </div>
        </div>
        
        <script>
            let currentKlines = [];
            
            async function generateSampleData() {{
                const pair = document.getElementById('pair-select').value;
                if (!pair) {{
                    showStatus('Please select a pair first', 'error');
                    return;
                }}
                
                const numCandles = parseInt(document.getElementById('num-candles').value);
                
                try {{
                    showLoading(true);
                    const response = await fetch('/api/generate-sample', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{
                            pair: pair,
                            num_candles: numCandles
                        }})
                    }});
                    
                    const data = await response.json();
                    if (data.success) {{
                        currentKlines = data.klines;
                        displayKlines(data.klines);
                        showStatus(`Generated ${{numCandles}} sample candles for ${{pair}}`, 'success');
                    }} else {{
                        showStatus(`Error: ${{data.message}}`, 'error');
                    }}
                }} catch (error) {{
                    showStatus(`Error: ${{error.message}}`, 'error');
                }} finally {{
                    showLoading(false);
                }}
            }}
            
            async function runInference() {{
                const pair = document.getElementById('pair-select').value;
                if (!pair) {{
                    showStatus('Please select a pair first', 'error');
                    return;
                }}
                
                if (currentKlines.length === 0) {{
                    showStatus('Please generate sample data first', 'error');
                    return;
                }}
                
                try {{
                    showLoading(true);
                    const response = await fetch('/api/predict', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{
                            pair: pair,
                            klines: currentKlines
                        }})
                    }});
                    
                    const data = await response.json();
                    if (data.success) {{
                        displayResults(data.prediction);
                        showStatus('Prediction completed successfully', 'success');
                    }} else {{
                        showStatus(`Error: ${{data.message}}`, 'error');
                    }}
                }} catch (error) {{
                    showStatus(`Error: ${{error.message}}`, 'error');
                }} finally {{
                    showLoading(false);
                }}
            }}
            
            function displayKlines(klines) {{
                const klinesHtml = `
                    <div class="kline-header">OPEN | HIGH | LOW | CLOSE | VOLUME</div>
                    ${{klines.map(k => `
                        <div class="kline-item">
                            <div>${{k.open.toFixed(2)}}</div>
                            <div>${{k.high.toFixed(2)}}</div>
                            <div>${{k.low.toFixed(2)}}</div>
                            <div>${{k.close.toFixed(2)}}</div>
                            <div>${{{k.volume.toFixed(0)}}}</div>
                        </div>
                    `).join('')}}
                `;
                
                const klinesDiv = document.getElementById('klines');
                klinesDiv.innerHTML = klinesHtml;
                klinesDiv.style.display = 'block';
            }}
            
            function displayResults(prediction) {{
                document.getElementById('price-result').textContent = prediction.price.toFixed(4);
                document.getElementById('volatility-result').textContent = prediction.volatility.toFixed(4);
                document.getElementById('results').classList.add('active');
            }}
            
            function showStatus(message, type) {{
                const statusDiv = document.getElementById('status');
                statusDiv.textContent = message;
                statusDiv.className = `status active ${{type}}`;
                
                if (type === 'success') {{
                    setTimeout(() => {{
                        statusDiv.classList.remove('active');
                    }}, 5000);
                }}
            }}
            
            function showLoading(show) {{
                const loadingDiv = document.getElementById('loading');
                if (show) {{
                    loadingDiv.classList.add('active');
                }} else {{
                    loadingDiv.classList.remove('active');
                }}
            }}
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/api/generate-sample', methods=['POST'])
def api_generate_sample():
    """
    API to generate sample K-line data
    """
    try:
        data = request.json
        pair = data.get('pair')
        num_candles = data.get('num_candles', 20)
        
        if pair not in CONFIG['pairs']:
            return jsonify({'success': False, 'message': f'Invalid pair: {pair}'})
        
        if num_candles < 20 or num_candles > 100:
            return jsonify({'success': False, 'message': 'Number of candles must be between 20 and 100'})
        
        klines = generate_sample_klines(num_candles)
        
        return jsonify({
            'success': True,
            'pair': pair,
            'klines': klines,
            'count': len(klines)
        })
    except Exception as e:
        logger.error(f"Error in generate_sample: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API to run inference on V2 model
    """
    try:
        data = request.json
        pair = data.get('pair')
        klines = data.get('klines')
        
        if not pair or pair not in CONFIG['pairs']:
            return jsonify({'success': False, 'message': 'Invalid pair'})
        
        if not klines or len(klines) < 20:
            return jsonify({'success': False, 'message': 'Need at least 20 candles'})
        
        # Load model
        model = load_model(pair)
        if model is None:
            return jsonify({'success': False, 'message': f'Failed to load model for {pair}'})
        
        # Prepare input
        X = prepare_input(klines)
        if X is None:
            return jsonify({'success': False, 'message': 'Failed to prepare input data'})
        
        # Make prediction
        predictions = model.predict(X, verbose=0)
        price = float(predictions[0, 0])
        volatility = float(predictions[0, 1])
        
        logger.info(f"Prediction for {pair}: price={price:.6f}, volatility={volatility:.6f}")
        
        return jsonify({
            'success': True,
            'pair': pair,
            'prediction': {
                'price': price,
                'volatility': volatility,
                'timestamp': datetime.now().isoformat()
            },
            'input_shape': [X.shape[0], X.shape[1], X.shape[2]]
        })
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/status')
def api_status():
    """
    API to check system status
    """
    available_models = []
    for pair in CONFIG['pairs']:
        model_path = f'{CONFIG["model_dir"]}/v2_model_{pair}.h5'
        available_models.append({
            'pair': pair,
            'available': os.path.exists(model_path)
        })
    
    available_count = sum(1 for m in available_models if m['available'])
    
    return jsonify({
        'model_version': CONFIG['model_version'],
        'total_pairs': len(CONFIG['pairs']),
        'available_models': available_count,
        'models': available_models,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print(" "*15 + "CPB V2 Model Web Application")
    print("="*80)
    print(f"\nModel Version: {CONFIG['model_version']}")
    print(f"Supported Pairs: {len(CONFIG['pairs'])}")
    print(f"Pairs: {', '.join(CONFIG['pairs'])}")
    print(f"\nStarting server...")
    print(f"Open your browser: http://localhost:5000")
    print(f"\nAPI Endpoints:")
    print(f"  POST /api/generate-sample  - Generate sample K-line data")
    print(f"  POST /api/predict          - Run model inference")
    print(f"  GET  /api/status           - Check system status")
    print("\n" + "="*80 + "\n")
    
    app.run(debug=True, host='localhost', port=5000)
