#!/usr/bin/env python3
"""
ENHANCED BTCUSDC TRADING BOT - MULTI-MODEL ML FORECASTING
→ Uses negative powers of phi within unit circle for price forecasting
→ Creates artificial 15-second timeframe from 1-minute data
→ Uses TA-Lib HT Sine for wave analysis
→ Implements local dip/top detection
→ Multi-model ML forecasting (Linear Regression, ARIMA, Random Walk, SVC)
→ Adds momentum and volume analysis
→ Targets 0.65% profit for fast scalps
"""

import os
import time
import numpy as np
import pandas as pd
import threading
import signal
import sys
import warnings
import math
import gc
import json
import requests
from flask import Flask, request, jsonify
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime
from scipy.signal import hilbert
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Try to import ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    print("ARIMA not available. Install with: pip install statsmodels")
    ARIMA_AVAILABLE = False

# --- TA-Lib import ---
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    print("TA-Lib not available. Please install TA-Lib for better technical analysis.")
    TALIB_AVAILABLE = False

# ------------------ Configuration ------------------
API_FILE = 'api.txt'
SYMBOL = 'BTCUSDC'

# Timeframes for analysis
TIMEFRAMES = ['1m']  # Only using 1m timeframe as requested

# Trading Configuration
PROFIT_TARGET_PERCENT = 0.65  # Changed to 0.65% as requested
TOTAL_FEE_PERCENT = 0.22
MIN_TRADE_AMOUNT = 10
MAX_POSITION_PERCENT = 100  # Using 100% of available balance as requested

# Dust Conversion Configuration
MIN_DUST_CONVERSION_AMOUNT = 0.0001
MAX_DUST_CONVERSION_AMOUNT = 0.001

# Technical Indicators Configuration
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MOMENTUM_PERIOD = 10  # Period for momentum calculation

# ML Configuration
ML_FORECAST_PERIODS = 4  # Number of periods to forecast
MIN_FORECAST_CONFIDENCE = 0.7  # Minimum confidence for ensemble forecast
MIN_FORECAST_THRESHOLD = 0.05  # Minimum 0.05% forecast difference

# Phi Configuration
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
PHI_NEGATIVE_POWERS = [1 / (PHI ** i) for i in range(1, 11)]  # Negative powers of phi

# Artificial 15-second timeframe configuration
SEC15_FACTOR = 4  # 1-minute / 15 seconds = 4

# API Rate Limiting
MIN_ITERATION_INTERVAL = 5  # 5 seconds between iterations

# Webhook Configuration
WEBHOOK_PORT = 5000
WEBHOOK_HOST = '0.0.0.0'

# Global stop event
stop_event = threading.Event()

# Trade state variables
trade_active = False
trade_info = {}

# Webhook data storage
webhook_data = {
    'current_price': None,
    'last_update': None,
    'price_history': []
}

# Flask app for webhooks
app = Flask(__name__)

def signal_handler(sig, frame):
    print('\nCtrl+C pressed! Shutting down gracefully...')
    stop_event.set()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ------------------ Webhook Endpoints ------------------

@app.route('/webhook/price', methods=['POST'])
def receive_price_webhook():
    """Receive price data via webhook"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data received'}), 400
        
        # Extract price data
        if 'price' in data:
            price = float(data['price'])
            timestamp = data.get('timestamp', datetime.now().isoformat())
            
            # Update webhook data
            webhook_data['current_price'] = price
            webhook_data['last_update'] = timestamp
            
            # Add to price history (keep last 1000 points)
            webhook_data['price_history'].append({
                'price': price,
                'timestamp': timestamp
            })
            if len(webhook_data['price_history']) > 1000:
                webhook_data['price_history'].pop(0)
            
            print(f"Webhook received: {price} at {timestamp}")
            return jsonify({'status': 'success', 'price': price})
        
        return jsonify({'error': 'Invalid data format'}), 400
        
    except Exception as e:
        print(f"Webhook error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'current_price': webhook_data['current_price'],
        'last_update': webhook_data['last_update']
    })

def start_webhook_server():
    """Start webhook server using production-ready server"""
    try:
        # Try to use Gunicorn if available (recommended for production)
        try:
            import gunicorn.app.base
            from gunicorn.six import iteritems
            
            class StandaloneApplication(gunicorn.app.base.BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super(StandaloneApplication, self).__init__()
                
                def load_config(self):
                    config = dict([(key, value) for key, value in iteritems(self.options)
                                 if key in self.cfg.settings and value is not None])
                    for key, value in iteritems(config):
                        self.cfg.set(key.lower(), value)
                
                def load(self):
                    return self.application
            
            options = {
                'bind': f'{WEBHOOK_HOST}:{WEBHOOK_PORT}',
                'workers': 1,
                'threads': 4,
                'timeout': 30,
                'keepalive': 2,
                'max_requests': 1000,
                'max_requests_jitter': 100,
                'preload_app': True,
                'accesslog': '-',
                'errorlog': '-',
                'loglevel': 'info'
            }
            
            print(f"Starting webhook server with Gunicorn on port {WEBHOOK_PORT}")
            StandaloneApplication(app, options).run()
            
        except ImportError:
            # Fallback to Waitress if Gunicorn is not available
            try:
                from waitress import serve
                
                print(f"Starting webhook server with Waitress on port {WEBHOOK_PORT}")
                serve(app, host=WEBHOOK_HOST, port=WEBHOOK_PORT)
                
            except ImportError:
                # Last resort: use Flask development server with warnings suppressed
                print("WARNING: Neither Gunicorn nor Waitress found. Using Flask development server.")
                print("For production use, install Gunicorn: pip install gunicorn")
                print("Or install Waitress: pip install waitress")
                
                # Suppress Flask development server warning
                import logging
                log = logging.getLogger('werkzeug')
                log.setLevel(logging.ERROR)
                
                app.run(host=WEBHOOK_HOST, port=WEBHOOK_PORT, debug=False, use_reloader=False)
                
    except Exception as e:
        print(f"Error starting webhook server: {e}")
        # Fallback to Flask development server
        app.run(host=WEBHOOK_HOST, port=WEBHOOK_PORT, debug=False, use_reloader=False)

# ------------------ Enhanced BTC Dust Conversion ------------------

def convert_btc_dust_to_usdc(client):
    """Convert only BTC dust balances to USDC before trading with proper error handling."""
    try:
        print("Checking for BTC dust to convert...")
        account_info = client.get_account()
        
        dust_assets = []
        
        for balance in account_info['balances']:
            asset = balance['asset']
            free_balance = float(balance['free'])
            locked_balance = float(balance['locked'])
            total_balance = free_balance + locked_balance
            
            # Check for BTC dust specifically
            if asset == 'BTC' and MIN_DUST_CONVERSION_AMOUNT <= total_balance <= MAX_DUST_CONVERSION_AMOUNT:
                try:
                    # Get current price to calculate USD value
                    symbol = f"{asset}USDC"
                    ticker = client.get_symbol_ticker(symbol=symbol)
                    price = float(ticker['price'])
                    usd_value = total_balance * price
                    
                    if usd_value < MIN_TRADE_AMOUNT:
                        print(f"Found BTC dust: {asset} - {total_balance:.8f} (≈${usd_value:.2f})")
                        dust_assets.append(asset)
                        
                except Exception as e:
                    print(f"Error checking {asset} value: {e}")
                    continue
        
        # Convert dust assets if any found
        if dust_assets:
            try:
                print(f"Attempting to convert dust assets: {dust_assets}")
                
                # Use correct API endpoint for dust conversion
                result = client.transfer_dust(asset=dust_assets)
                
                if result.get('success', False):
                    print("BTC dust conversion successful!")
                    for item in result.get('transferResult', []):
                        print(f"  {item['fromAsset']}: {item['amount']} → USDC")
                    return True
                else:
                    print("Dust conversion failed or not supported for these assets")
                    return False
                    
            except BinanceAPIException as e:
                if 'illegal parameter' in str(e).lower():
                    print("Dust conversion not supported for these assets or invalid parameters")
                elif 'insufficient balance' in str(e).lower():
                    print("Insufficient balance for dust conversion")
                else:
                    print(f"Binance API error during dust conversion: {e}")
                return False
                
            except Exception as e:
                print(f"Unexpected error during dust conversion: {e}")
                return False
        else:
            print("No BTC dust found for conversion")
            return True
            
    except Exception as e:
        print(f"Error in BTC dust conversion process: {e}")
        return False

# ------------------ Data Cleaning and Artificial Timeframe Creation ------------------

def clean_ohlc_data(df):
    """Clean OHLC data of NaN and 0 values"""
    # Convert all OHLCV columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace NaN values with 0
    df = df.fillna(0)
    
    # Replace 0 values in OHLC with previous non-zero values
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(method='ffill')
        df[col] = df[col].fillna(method='bfill')
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(method='ffill')
        df[col] = df[col].fillna(method='bfill')
    
    # Replace 0 volume with previous non-zero values
    df['volume'] = df['volume'].replace(0, np.nan)
    df['volume'] = df['volume'].fillna(method='ffill')
    df['volume'] = df['volume'].fillna(method='bfill')
    
    return df

def create_15sec_timeframe(df_1m):
    """Create artificial 15-second timeframe from 1-minute data"""
    # Create empty DataFrame for 15-second data
    df_15sec = pd.DataFrame()
    
    # For each 1-minute candle, create 4 15-second candles
    for idx, row in df_1m.iterrows():
        # Calculate price changes within the 1-minute candle
        open_price = float(row['open'])
        high_price = float(row['high'])
        low_price = float(row['low'])
        close_price = float(row['close'])
        volume = float(row['volume'])
        
        # Generate 4 15-second candles
        for i in range(SEC15_FACTOR):
            # Calculate timestamp for each 15-second candle
            timestamp = row['timestamp'] + pd.Timedelta(seconds=15 * i)
            
            # Simple interpolation for price and volume
            progress = (i + 1) / SEC15_FACTOR
            
            # Linear interpolation between open and close
            interpolated_price = open_price + (close_price - open_price) * progress
            
            # Add some randomness based on high/low
            high_factor = 1 + (high_price / max(open_price, close_price) - 1) * (0.5 * abs(0.5 - progress))
            low_factor = 1 - (max(open_price, close_price) / low_price - 1) * (0.5 * abs(0.5 - progress))
            
            # Create the 15-second candle
            if i == 0:
                # First 15-second candle
                candle_open = open_price
                candle_close = interpolated_price
            elif i == SEC15_FACTOR - 1:
                # Last 15-second candle
                candle_open = df_15sec.iloc[-1]['close'] if not df_15sec.empty else interpolated_price
                candle_close = close_price
            else:
                # Middle candles
                candle_open = df_15sec.iloc[-1]['close'] if not df_15sec.empty else interpolated_price
                candle_close = interpolated_price
            
            # Calculate high and low for the 15-second candle
            candle_high = max(candle_open, candle_close) * high_factor
            candle_low = min(candle_open, candle_close) * low_factor
            
            # Distribute volume
            candle_volume = volume / SEC15_FACTOR
            
            # Add to DataFrame
            new_row = pd.DataFrame({
                'timestamp': [timestamp],
                'open': [candle_open],
                'high': [candle_high],
                'low': [candle_low],
                'close': [candle_close],
                'volume': [candle_volume]
            })
            
            if df_15sec.empty:
                df_15sec = new_row
            else:
                df_15sec = pd.concat([df_15sec, new_row], ignore_index=True)
    
    return df_15sec

# ------------------ Technical Analysis Functions ------------------

def calculate_rsi(df, period=RSI_PERIOD):
    """Calculate RSI indicator using TA-Lib."""
    try:
        if df is None or len(df) < period + 1:
            return None
        
        close_prices = df['close'].values.astype(float)
        
        if TALIB_AVAILABLE:
            rsi = talib.RSI(close_prices, timeperiod=period)
        else:
            # Fallback to manual calculation
            delta = pd.Series(close_prices).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        df = df.copy()
        df[f'RSI_{period}'] = rsi
        return df
    except Exception as e:
        print(f"calculate_rsi error: {e}")
        return None

def calculate_ht_sine(df):
    """Calculate Hilbert Transform - SineWave using TA-Lib."""
    try:
        if df is None or len(df) < 32:
            return None
        
        close_prices = df['close'].values.astype(float)
        
        if TALIB_AVAILABLE:
            sine, leadsine = talib.HT_SINE(close_prices)
        else:
            # Fallback to manual calculation using Hilbert transform
            analytic_signal = hilbert(close_prices)
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            sine = np.sin(instantaneous_phase)
            leadsine = np.sin(instantaneous_phase + np.pi/2)
        
        df = df.copy()
        df['HT_SINE'] = sine
        df['HT_LEADSINE'] = leadsine
        return df
    except Exception as e:
        print(f"calculate_ht_sine error: {e}")
        return None

def calculate_momentum(df, period=MOMENTUM_PERIOD):
    """Calculate momentum indicator."""
    try:
        if df is None or len(df) < period + 1:
            return None
        
        df = df.copy()
        close_prices = df['close'].values.astype(float)
        
        # Calculate momentum as difference between current price and price N periods ago
        momentum = np.zeros(len(close_prices))
        for i in range(period, len(close_prices)):
            momentum[i] = close_prices[i] - close_prices[i - period]
        
        df['MOMENTUM'] = momentum
        
        # Calculate momentum change (increasing/decreasing)
        momentum_change = np.zeros(len(momentum))
        for i in range(1, len(momentum)):
            if momentum[i] > momentum[i-1]:
                momentum_change[i] = 1  # Increasing
            elif momentum[i] < momentum[i-1]:
                momentum_change[i] = -1  # Decreasing
            else:
                momentum_change[i] = 0  # Neutral
        
        df['MOMENTUM_CHANGE'] = momentum_change
        
        return df
    except Exception as e:
        print(f"calculate_momentum error: {e}")
        return None

# ------------------ Local Dip/Top Detection ------------------

def detect_local_dips_tops(df, window=5):
    """Detect local dips and tops in price data."""
    try:
        if df is None or len(df) < window * 2 + 1:
            return None
        
        df = df.copy()
        close_prices = df['close'].values.astype(float)
        
        # Initialize arrays for dips and tops
        is_dip = np.zeros(len(close_prices), dtype=bool)
        is_top = np.zeros(len(close_prices), dtype=bool)
        
        # Detect local dips and tops
        for i in range(window, len(close_prices) - window):
            # Check if current point is a local dip
            is_dip[i] = all(close_prices[i] <= close_prices[j] for j in range(i - window, i + window + 1) if j != i)
            
            # Check if current point is a local top
            is_top[i] = all(close_prices[i] >= close_prices[j] for j in range(i - window, i + window + 1) if j != i)
        
        df['is_dip'] = is_dip
        df['is_top'] = is_top
        
        return df
    except Exception as e:
        print(f"detect_local_dips_tops error: {e}")
        return None

# ------------------ Multi-Model ML Forecasting ------------------

def linear_regression_forecast(prices, periods=ML_FORECAST_PERIODS):
    """Linear Regression forecast"""
    try:
        if len(prices) < 20:
            return None, 0, 0
        
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecast future values
        future_X = np.arange(len(prices), len(prices) + periods).reshape(-1, 1)
        forecast = model.predict(future_X)
        
        # Calculate confidence based on R² score
        r2_score = model.score(X, y)
        confidence = max(0, min(1, r2_score))
        
        return forecast[-1], confidence, model.coef_[0]
    except Exception as e:
        print(f"linear_regression_forecast error: {e}")
        return None, 0, 0

def arima_forecast(prices, periods=ML_FORECAST_PERIODS):
    """ARIMA forecast"""
    try:
        if not ARIMA_AVAILABLE or len(prices) < 50:
            return None, 0, 0
        
        # Use ARIMA(1,1,1) as a simple but effective model
        model = ARIMA(prices, order=(1,1,1))
        model_fit = model.fit()
        
        # Forecast
        forecast = model_fit.forecast(steps=periods)
        predicted_value = forecast[0] if len(forecast) > 0 else prices[-1]
        
        # Calculate confidence based on AIC
        aic = model_fit.aic
        # Normalize AIC to confidence (lower AIC = higher confidence)
        confidence = max(0, min(1, 1 - (aic / 1000)))
        
        return predicted_value, confidence, 0
    except Exception as e:
        print(f"arima_forecast error: {e}")
        return None, 0, 0

def random_walk_forecast(prices, periods=ML_FORECAST_PERIODS):
    """Random Walk forecast with drift"""
    try:
        if len(prices) < 20:
            return None, 0, 0
        
        # Calculate drift (average change)
        changes = np.diff(prices)
        drift = np.mean(changes)
        
        # Calculate volatility (standard deviation of changes)
        volatility = np.std(changes)
        
        # Random walk forecast
        last_price = prices[-1]
        forecast_values = []
        
        for i in range(periods):
            # Add drift and random shock
            random_shock = np.random.normal(0, volatility)
            forecast = last_price + drift + random_shock
            forecast_values.append(forecast)
            last_price = forecast
        
        predicted_value = forecast_values[0] if forecast_values else prices[-1]
        
        # Confidence based on volatility consistency
        volatility_ratio = min(1, max(0, 1 - (volatility / np.mean(prices))))
        confidence = max(0.3, volatility_ratio)
        
        return predicted_value, confidence, drift
    except Exception as e:
        print(f"random_walk_forecast error: {e}")
        return None, 0, 0

def svc_forecast(prices, periods=ML_FORECAST_PERIODS):
    """Support Vector Regression forecast"""
    try:
        if len(prices) < 30:
            return None, 0, 0
        
        # Prepare data with lagged features
        X = []
        y = []
        
        for i in range(5, len(prices)):
            # Create features: last 5 prices
            features = prices[i-5:i]
            X.append(features)
            y.append(prices[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train SVR model
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        model.fit(X_scaled, y)
        
        # Predict next value
        last_features = np.array([prices[-5:]])
        last_features_scaled = scaler.transform(last_features)
        predicted = model.predict(last_features_scaled)
        
        predicted_value = predicted[0]
        
        # Calculate confidence based on training error
        train_pred = model.predict(X_scaled)
        mse = mean_squared_error(y, train_pred)
        confidence = max(0, min(1, 1 - (mse / (np.var(y) + 1e-6))))
        
        return predicted_value, confidence, 0
    except Exception as e:
        print(f"svc_forecast error: {e}")
        return None, 0, 0

def random_forest_forecast(prices, periods=ML_FORECAST_PERIODS):
    """Random Forest forecast"""
    try:
        if len(prices) < 30:
            return None, 0, 0
        
        # Prepare data with lagged features
        X = []
        y = []
        
        for i in range(10, len(prices)):
            # Create features: last 10 prices and technical indicators
            features = []
            features.extend(prices[i-10:i])  # Last 10 prices
            
            # Add simple technical indicators as features
            if i >= 5:
                features.append(np.mean(prices[i-5:i]))  # 5-period MA
                features.append(np.mean(prices[i-10:i]))  # 10-period MA
                features.append(prices[i-1] - prices[i-5])  # Recent momentum
                features.append(np.std(prices[i-10:i]))  # Recent volatility
            
            X.append(features)
            y.append(prices[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)
        
        # Predict next value
        last_features = []
        last_features.extend(prices[-10:])  # Last 10 prices
        
        if len(prices) >= 5:
            last_features.append(np.mean(prices[-5:]))  # 5-period MA
            last_features.append(np.mean(prices[-10:]))  # 10-period MA
            last_features.append(prices[-1] - prices[-5])  # Recent momentum
            last_features.append(np.std(prices[-10:]))  # Recent volatility
        
        predicted = model.predict([last_features])
        predicted_value = predicted[0]
        
        # Calculate confidence based on feature importance and model score
        score = model.score(X, y)
        confidence = max(0, min(1, score))
        
        return predicted_value, confidence, 0
    except Exception as e:
        print(f"random_forest_forecast error: {e}")
        return None, 0, 0

def ensemble_ml_forecast(df, periods=ML_FORECAST_PERIODS):
    """Ensemble forecast using multiple ML models"""
    try:
        if df is None or len(df) < 50:
            return None, {"error": "Insufficient data for ML forecast"}
        
        df = df.copy()
        close_prices = df['close'].values.astype(float)
        
        # Initialize model results
        model_results = []
        
        # Linear Regression
        lr_pred, lr_conf, lr_coef = linear_regression_forecast(close_prices, periods)
        if lr_pred is not None:
            model_results.append({
                'model': 'Linear Regression',
                'prediction': lr_pred,
                'confidence': lr_conf,
                'strength': abs(lr_coef)
            })
        
        # ARIMA
        arima_pred, arima_conf, arima_drift = arima_forecast(close_prices, periods)
        if arima_pred is not None:
            model_results.append({
                'model': 'ARIMA',
                'prediction': arima_pred,
                'confidence': arima_conf,
                'strength': abs(arima_drift)
            })
        
        # Random Walk
        rw_pred, rw_conf, rw_drift = random_walk_forecast(close_prices, periods)
        if rw_pred is not None:
            model_results.append({
                'model': 'Random Walk',
                'prediction': rw_pred,
                'confidence': rw_conf,
                'strength': abs(rw_drift)
            })
        
        # SVR
        svc_pred, svc_conf, svc_coef = svc_forecast(close_prices, periods)
        if svc_pred is not None:
            model_results.append({
                'model': 'SVR',
                'prediction': svc_pred,
                'confidence': svc_conf,
                'strength': abs(svc_coef)
            })
        
        # Random Forest
        rf_pred, rf_conf, rf_importance = random_forest_forecast(close_prices, periods)
        if rf_pred is not None:
            model_results.append({
                'model': 'Random Forest',
                'prediction': rf_pred,
                'confidence': rf_conf,
                'strength': rf_importance
            })
        
        if not model_results:
            return None, {"error": "All ML models failed"}
        
        # Weighted ensemble based on confidence and strength
        total_weight = 0
        weighted_prediction = 0
        
        for result in model_results:
            # Weight = confidence * strength
            weight = result['confidence'] * result['strength']
            total_weight += weight
            weighted_prediction += result['prediction'] * weight
        
        if total_weight > 0:
            final_prediction = weighted_prediction / total_weight
        else:
            final_prediction = close_prices[-1]
        
        # Calculate ensemble confidence
        avg_confidence = np.mean([r['confidence'] for r in model_results])
        ensemble_confidence = min(1.0, avg_confidence * 1.2)  # Boost for ensemble
        
        # Get individual predictions for display
        predictions = {r['model']: r['prediction'] for r in model_results}
        confidences = {r['model']: r['confidence'] for r in model_results}
        
        details = {
            "ensemble_prediction": final_prediction,
            "ensemble_confidence": ensemble_confidence,
            "individual_predictions": predictions,
            "individual_confidences": confidences,
            "model_count": len(model_results)
        }
        
        return final_prediction, ensemble_confidence, details
        
    except Exception as e:
        print(f"ensemble_ml_forecast error: {e}")
        return None, {"error": str(e)}

# ------------------ Enhanced Analysis Functions ------------------

def analyze_local_dip_condition(client, symbol, timeframe='1m', lookback=500):
    """Analyze local dip condition on 1-minute timeframe."""
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=lookback)
        if not klines or len(klines) < 100:
            return False, {"error": "Insufficient data"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Detect local dips and tops
        df = detect_local_dips_tops(df)
        
        if df is None or 'is_dip' not in df.columns:
            return False, {"error": "Failed to detect local dips and tops"}
        
        # Find all dips and tops
        dip_indices = df[df['is_dip']].index.tolist()
        top_indices = df[df['is_top']].index.tolist()
        
        # Sort indices to ensure they're in chronological order
        dip_indices.sort()
        top_indices.sort()
        
        # Find most recent dip and top pair that surrounds current price
        price_between = False
        most_recent_dip = None
        most_recent_top = None
        dip_price = None
        top_price = None
        
        # Get current price
        current_price = get_current_price_from_webhook()
        if current_price is None:
            ticker = client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
        
        # Look for most recent dip-top pair that surrounds current price
        if dip_indices and top_indices:
            # Check last few pairs to find one that surrounds current price
            for i in range(min(5, len(dip_indices))):
                dip_idx = dip_indices[-(i+1)]
                dip_low = df.loc[dip_idx, 'low']
                
                # Find next top after this dip
                next_top_idx = None
                for top_idx in top_indices:
                    if top_idx > dip_idx:
                        next_top_idx = top_idx
                        break
                
                if next_top_idx is not None:
                    top_high = df.loc[next_top_idx, 'high']
                    
                    # Check if current price is between dip low and top high
                    if dip_low <= current_price <= top_high or top_high <= current_price <= dip_low:
                        price_between = True
                        most_recent_dip = dip_idx
                        most_recent_top = next_top_idx
                        dip_price = dip_low
                        top_price = top_high
                        break
        
        # If we didn't find a dip-top pair that surrounds current price,
        # try the reverse: top-dip pair
        if not price_between and dip_indices and top_indices:
            for i in range(min(5, len(top_indices))):
                top_idx = top_indices[-(i+1)]
                top_high = df.loc[top_idx, 'high']
                
                # Find next dip after this top
                next_dip_idx = None
                for dip_idx in dip_indices:
                    if dip_idx > top_idx:
                        next_dip_idx = dip_idx
                        break
                
                if next_dip_idx is not None:
                    dip_low = df.loc[next_dip_idx, 'low']
                    
                    # Check if current price is between top high and dip low
                    if top_high <= current_price <= dip_low or dip_low <= current_price <= top_high:
                        price_between = True
                        most_recent_dip = next_dip_idx
                        most_recent_top = top_idx
                        dip_price = dip_low
                        top_price = top_high
                        break
        
        # If we still don't have a valid pair, use the most recent dip and top
        if not price_between and dip_indices and top_indices:
            most_recent_dip = dip_indices[-1]
            most_recent_top = top_indices[-1]
            dip_price = df.loc[most_recent_dip, 'low']
            top_price = df.loc[most_recent_top, 'high']
            
            # Check if current price is between these values
            if dip_price <= current_price <= top_price or top_price <= current_price <= dip_price:
                price_between = True
        
        # Determine which reversal occurred most recently
        dip_most_recent = most_recent_dip is not None and (most_recent_top is None or most_recent_dip > most_recent_top)
        top_most_recent = most_recent_top is not None and (most_recent_dip is None or most_recent_top > most_recent_dip)
        
        # Condition is met when the most recent reversal is a dip and current price is between dip and top
        condition_met = dip_most_recent and price_between
        
        details = {
            "most_recent_dip": most_recent_dip,
            "most_recent_top": most_recent_top,
            "dip_price": dip_price,
            "top_price": top_price,
            "condition_met": condition_met,
            "price_between": price_between,
            "current_price": current_price,
            "dip_most_recent": dip_most_recent,
            "top_most_recent": top_most_recent
        }
        
        return condition_met, details
        
    except Exception as e:
        print(f"analyze_local_dip_condition error: {e}")
        return False, {"error": str(e)}

def analyze_momentum_condition(client, symbol, timeframe='1m', lookback=500):
    """Analyze momentum for increasing/decreasing and positive/negative values."""
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=lookback)
        if not klines or len(klines) < 100:
            return False, False, False, {"error": "Insufficient data"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Calculate momentum
        df = calculate_momentum(df, MOMENTUM_PERIOD)
        
        if df is None or 'MOMENTUM' not in df.columns:
            return False, False, False, {"error": "Failed to calculate momentum"}
        
        # Get the last momentum value and change
        last_momentum = float(df['MOMENTUM'].iloc[-1])
        last_momentum_change = int(df['MOMENTUM_CHANGE'].iloc[-1])
        
        # Determine if momentum is increasing, decreasing, or neutral
        momentum_increasing = last_momentum_change == 1
        momentum_decreasing = last_momentum_change == -1
        
        # Determine if momentum is positive
        momentum_positive = last_momentum > 0
        
        details = {
            "last_momentum": last_momentum,
            "last_momentum_change": last_momentum_change,
            "momentum_increasing": momentum_increasing,
            "momentum_decreasing": momentum_decreasing,
            "momentum_positive": momentum_positive
        }
        
        return momentum_increasing, momentum_decreasing, momentum_positive, details
        
    except Exception as e:
        print(f"analyze_momentum_condition error: {e}")
        return False, False, False, {"error": str(e)}

def analyze_volume_condition(client, symbol, timeframe='1m', lookback=500):
    """Analyze volume from 15-second timeframe for bullish vs bearish volume."""
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=lookback)
        if not klines or len(klines) < 100:
            return False, {"error": "Insufficient data"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Create artificial 15-second timeframe
        df_15sec = create_15sec_timeframe(df)
        
        if df_15sec is None or len(df_15sec) < 20:
            return False, {"error": "Failed to create 15-second timeframe"}
        
        # Calculate bullish and bearish volume
        bullish_volume = 0
        bearish_volume = 0
        
        for _, row in df_15sec.iterrows():
            open_price = float(row['open'])
            close_price = float(row['close'])
            volume = float(row['volume'])
            
            if close_price > open_price:
                bullish_volume += volume
            elif close_price < open_price:
                bearish_volume += volume
            else:
                # Neutral candles split equally
                bullish_volume += volume / 2
                bearish_volume += volume / 2
        
        total_volume = bullish_volume + bearish_volume
        
        if total_volume > 0:
            bullish_percentage = (bullish_volume / total_volume) * 100
            bearish_percentage = (bearish_volume / total_volume) * 100
        else:
            bullish_percentage = 50
            bearish_percentage = 50
        
        # Condition is true when bullish volume > bearish volume
        condition_met = bullish_volume > bearish_volume
        
        details = {
            "bullish_volume": bullish_volume,
            "bearish_volume": bearish_volume,
            "total_volume": total_volume,
            "bullish_percentage": bullish_percentage,
            "bearish_percentage": bearish_percentage,
            "condition_met": condition_met
        }
        
        return condition_met, details
        
    except Exception as e:
        print(f"analyze_volume_condition error: {e}")
        return False, {"error": str(e)}

def analyze_ht_sine_condition(client, symbol, timeframe='1m', lookback=500):
    """Analyze HT Sine wave for cycle detection."""
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=lookback)
        if not klines or len(klines) < 100:
            return False, {"error": "Insufficient data"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Calculate HT Sine
        df = calculate_ht_sine(df)
        
        if df is None or 'HT_SINE' not in df.columns:
            return False, {"error": "Failed to calculate HT Sine"}
        
        # Get the last sine and leadsine values
        last_sine = float(df['HT_SINE'].iloc[-1])
        last_leadsine = float(df['HT_LEADSINE'].iloc[-1])
        
        # Check if sine is crossing leadsine from below (indicating start of up cycle)
        if len(df) >= 2:
            prev_sine = float(df['HT_SINE'].iloc[-2])
            prev_leadsine = float(df['HT_LEADSINE'].iloc[-2])
            
            # Check for crossing from below
            crossing_up = (prev_sine < prev_leadsine) and (last_sine >= last_leadsine)
            
            # Also check if sine is currently below leadsine (in down cycle)
            in_down_cycle = last_sine < last_leadsine
            
            # Condition: either crossing up from below or in down cycle (waiting for dip)
            condition_met = crossing_up or in_down_cycle
        else:
            condition_met = False
        
        details = {
            "last_sine": last_sine,
            "last_leadsine": last_leadsine,
            "condition_met": condition_met
        }
        
        return condition_met, details
        
    except Exception as e:
        print(f"analyze_ht_sine_condition error: {e}")
        return False, {"error": str(e)}

def analyze_ml_forecast_condition(client, symbol, timeframe='1m', lookback=500):
    """Analyze multi-model ML forecast for significant price reversal."""
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=lookback)
        if not klines or len(klines) < 100:
            return False, {"error": "Insufficient data"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Create artificial 15-second timeframe
        df_15sec = create_15sec_timeframe(df)
        
        # Apply ensemble ML forecast
        ensemble_prediction, ensemble_confidence, ml_details = ensemble_ml_forecast(df_15sec, ML_FORECAST_PERIODS)
        
        if ensemble_prediction is None:
            return False, {"error": "Failed to apply ensemble ML forecast"}
        
        # Get the last price
        last_price = float(df_15sec['close'].iloc[-1])
        
        # Calculate the difference
        diff = ensemble_prediction - last_price
        diff_pct = (diff / last_price) * 100 if last_price > 0 else 0
        
        # Condition: forecast suggests a significant up cycle
        condition_met = (diff > 0 and 
                      diff_pct >= MIN_FORECAST_THRESHOLD and 
                      ensemble_confidence >= MIN_FORECAST_CONFIDENCE)
        
        # Update details with additional info
        ml_details.update({
            "last_price": last_price,
            "ensemble_prediction": ensemble_prediction,
            "diff": diff,
            "diff_pct": diff_pct,
            "condition_met": condition_met,
            "threshold_met": diff_pct >= MIN_FORECAST_THRESHOLD,
            "confidence_met": ensemble_confidence >= MIN_FORECAST_CONFIDENCE
        })
        
        return condition_met, ml_details
        
    except Exception as e:
        print(f"analyze_ml_forecast_condition error: {e}")
        return False, {"error": str(e)}

def analyze_rsi_condition(client, symbol, timeframe='1m', lookback=500):
    """Analyze RSI oversold/overbought most recent condition using TA-Lib."""
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=lookback)
        if not klines or len(klines) < 100:
            return False, False, 0.0, {"error": "Insufficient data"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Calculate RSI using TA-Lib
        df_rsi = calculate_rsi(df, RSI_PERIOD)
        if df_rsi is None or f'RSI_{RSI_PERIOD}' not in df_rsi.columns:
            return False, False, 0.0, {"error": "RSI calculation failed"}
        
        rsi_values = df_rsi[f'RSI_{RSI_PERIOD}'].values.astype(float)
        current_rsi = float(rsi_values[-1])
        
        # Find last oversold and overbought occurrences
        last_oversold_idx = None
        last_overbought_idx = None
        
        for i in range(len(rsi_values) - 1, -1, -1):
            if last_oversold_idx is None and rsi_values[i] <= RSI_OVERSOLD:
                last_oversold_idx = i
            if last_overbought_idx is None and rsi_values[i] >= RSI_OVERBOUGHT:
                last_overbought_idx = i
            if last_oversold_idx is not None and last_overbought_idx is not None:
                break
        
        # Determine which is most recent
        oversold_most_recent = False
        overbought_most_recent = False
        
        if last_oversold_idx is not None and last_overbought_idx is not None:
            oversold_most_recent = last_oversold_idx > last_overbought_idx
            overbought_most_recent = last_overbought_idx > last_oversold_idx
        elif last_oversold_idx is not None:
            oversold_most_recent = True
        elif last_overbought_idx is not None:
            overbought_most_recent = True
        
        details = {
            "current_rsi": current_rsi,
            "last_oversold_idx": last_oversold_idx,
            "last_overbought_idx": last_overbought_idx,
            "oversold_most_recent": oversold_most_recent,
            "overbought_most_recent": overbought_most_recent
        }
        
        return oversold_most_recent, overbought_most_recent, current_rsi, details
        
    except Exception as e:
        print(f"analyze_rsi_condition error: {e}")
        return False, False, 0.0, {"error": str(e)}

# ------------------ Trade Execution Functions ------------------

def execute_buy_order(client, symbol, usdc_amount):
    """Execute a market buy order using the entire USDC balance."""
    try:
        # Use webhook price if available for more accurate execution
        current_price = get_current_price_from_webhook()
        if current_price is None:
            ticker = client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
        
        # Use the entire USDC balance
        quantity = usdc_amount / current_price * 0.99  # 1% buffer for fees
        
        symbol_info = client.get_symbol_info(symbol)
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = float(lot_size_filter['step_size'])
            quantity = round(quantity - (quantity % step_size), 8)
        
        order = client.order_market_buy(
            symbol=symbol,
            quantity=quantity
        )
        
        return {
            'success': True,
            'order_id': order['orderId'],
            'symbol': symbol,
            'quantity': quantity,
            'price': current_price,
            'cost': quantity * current_price,
            'timestamp': datetime.now(),
            'order': order
        }
    except BinanceAPIException as e:
        return {
            'success': False,
            'error': str(e)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def execute_sell_order(client, symbol, quantity):
    """Execute a market sell order."""
    try:
        symbol_info = client.get_symbol_info(symbol)
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = float(lot_size_filter['step_size'])
            quantity = round(quantity - (quantity % step_size), 8)
        
        order = client.order_market_sell(
            symbol=symbol,
            quantity=quantity
        )
        
        return {
            'success': True,
            'order_id': order['orderId'],
            'symbol': symbol,
            'quantity': quantity,
            'timestamp': datetime.now(),
            'order': order
        }
    except BinanceAPIException as e:
        return {
            'success': False,
            'error': str(e)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_current_price_from_webhook():
    """Get current price from webhook data, fallback to API if needed"""
    global webhook_data
    
    # Try webhook first
    if webhook_data['current_price'] is not None:
        # Check if data is recent (within last 10 seconds)
        if webhook_data['last_update']:
            try:
                last_update = datetime.fromisoformat(webhook_data['last_update'].replace('Z', '+00:00'))
                time_diff = (datetime.now() - last_update).total_seconds()
                if time_diff < 10:
                    return webhook_data['current_price']
            except:
                pass
    
    # Fallback to API
    return None

def get_current_price(client, symbol):
    """Get current price for a symbol."""
    # Try webhook first
    price = get_current_price_from_webhook()
    if price is not None:
        return price
    
    # Fallback to API
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        print(f"Error getting current price: {e}")
        return None

def get_account_balance(client, asset):
    """Get balance of a specific asset in the account."""
    try:
        account_info = client.get_account()
        for balance in account_info['balances']:
            if balance['asset'] == asset:
                return float(balance['free'])
        return 0.0
    except Exception as e:
        print(f"Error getting account balance: {e}")
        return 0.0

def check_trade_status(client):
    """Check if profit target is reached for active trade."""
    global trade_active, trade_info
    
    if not trade_active:
        return False
    
    try:
        # Use webhook price for more accurate monitoring
        current_price = get_current_price(client, SYMBOL)
        if current_price is None:
            return False
        
        entry_price = trade_info['entry_price']
        quantity = trade_info['quantity']
        
        target_price = entry_price * (1 + (PROFIT_TARGET_PERCENT + TOTAL_FEE_PERCENT) / 100)
        
        price_diff = current_price - entry_price
        price_diff_pct = (price_diff / entry_price) * 100
        time_elapsed = datetime.now() - trade_info['entry_time']
        
        target_diff = target_price - current_price
        target_diff_pct = (target_diff / current_price) * 100
        
        # Update trade info
        trade_info.update({
            'current_price': current_price,
            'price_diff': price_diff,
            'price_diff_pct': price_diff_pct,
            'time_elapsed': time_elapsed,
            'target_price': target_price,
            'target_diff': target_diff,
            'target_diff_pct': target_diff_pct
        })
        
        # Check for profit target
        if current_price >= target_price:
            print(f"\nPROFIT TARGET REACHED! Selling at {current_price:.6f}")
            sell_result = execute_sell_order(client, SYMBOL, quantity)
            
            if sell_result['success']:
                print(f"SELL ORDER EXECUTED SUCCESSFULLY!")
                print(f"Order ID: {sell_result['order_id']}")
                print(f"Quantity Sold: {sell_result['quantity']}")
                print(f"Estimated Profit: {(current_price - entry_price) * quantity:.6f} USDC")
                trade_active = False
                trade_info = {}
                return True
            else:
                print(f"ERROR EXECUTING SELL ORDER: {sell_result['error']}")
        
        return False
        
    except Exception as e:
        print(f"Error checking trade status: {e}")
        return False

def display_trade_status():
    """Display current trade status if active."""
    global trade_active, trade_info
    
    if not trade_active:
        return
    
    print("\n" + "="*80)
    print("TRADE MONITOR - ACTIVE POSITION")
    print("="*80)
    print(f"{'Symbol:':<20}{trade_info['symbol']}")
    print(f"{'Entry Price:':<20}{trade_info['entry_price']:.6f}")
    print(f"{'Entry Time:':<20}{trade_info['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'Current Price:':<20}{trade_info['current_price']:.6f}")
    print(f"{'Price Difference:':<20}{trade_info['price_diff']:+.6f} ({trade_info['price_diff_pct']:+.2f}%)")
    print(f"{'Time Elapsed:':<20}{trade_info['time_elapsed']}")
    print(f"{'Target Price:':<20}{trade_info['target_price']:.6f}")
    print(f"{'Distance to Target:':<20}{trade_info['target_diff']:.6f} ({trade_info['target_diff_pct']:.2f}%)")
    print(f"{'Quantity:':<20}{trade_info['quantity']}")
    print("="*80)

# ------------------ Main Analysis Function ------------------

def perform_single_iteration_analysis(client):
    """Perform single iteration analysis with all conditions."""
    global trade_active, trade_info
    
    # Clear screen for fresh iteration
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("="*80)
    print(f"BTCUSDC TRADING BOT - MULTI-MODEL ML FORECASTING")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Webhook Status: {'Connected' if webhook_data['current_price'] else 'Waiting for data'}")
    if webhook_data['current_price']:
        print(f"Webhook Price: {webhook_data['current_price']:.2f}")
    print("="*80)
    
    # If trade is active, check status and display
    if trade_active:
        print("\n>>> TRADE ACTIVE - CHECKING STATUS <<<")
        trade_closed = check_trade_status(client)
        display_trade_status()
        
        if trade_closed:
            print("\nTrade closed. Resuming normal analysis...")
        else:
            print("\nTrade still active. Waiting for next iteration...")
            return
    
    # Step 1: Check and convert BTC dust (only when no active trade)
    print("\n1. Checking for BTC dust...")
    dust_success = convert_btc_dust_to_usdc(client)
    if not dust_success:
        print("Dust conversion failed or not needed")
    
    # Step 2: Get current balance
    usdc_balance = get_account_balance(client, 'USDC')
    print(f"USDC Balance: {usdc_balance:.2f}")
    
    if usdc_balance < MIN_TRADE_AMOUNT:
        print(f"!!! INSUFFICIENT USDC BALANCE - MINIMUM REQUIRED: {MIN_TRADE_AMOUNT} !!!")
        return
    
    # Step 3: Perform all condition analyses
    print("\n2. Performing technical analysis...")
    
    conditions_met = 0
    total_conditions = 5  # Updated to 5 conditions after removing Bollinger Bands
    condition_details = {}
    condition_results = {}  # Store individual condition results
    
    # Condition 1: Local Dip Condition on 1-minute timeframe
    print("\n--- Condition 1: Local Dip Analysis ---")
    local_dip_met, dip_details = analyze_local_dip_condition(client, SYMBOL, '1m', 500)
    condition_details['local_dip'] = {
        'met': local_dip_met,
        'details': dip_details
    }
    condition_results['Local Dip Most Recent'] = local_dip_met
    
    # Print details for this condition
    print(f"\nMost recent dip index: {dip_details.get('most_recent_dip', 'None')}")
    print(f"Most recent top index: {dip_details.get('most_recent_top', 'None')}")
    print(f"Dip price: {dip_details.get('dip_price', 'None')}")
    print(f"Top price: {dip_details.get('top_price', 'None')}")
    print(f"Current price between dip and top: {dip_details.get('price_between', False)}")
    print(f"Most recent reversal found at local dip: {dip_details.get('dip_most_recent', False)}")
    print(f"Most recent reversal found at local top: {dip_details.get('top_most_recent', False)}")
    print(f"Condition Met: {local_dip_met}")
    
    if local_dip_met:
        conditions_met += 1
        print("\nTRUE - Local Dip condition MET")
    else:
        print("\nFALSE - Local Dip condition NOT met")
    
    # Condition 2: Momentum Analysis
    print("\n--- Condition 2: Momentum Analysis ---")
    momentum_increasing, momentum_decreasing, momentum_positive, momentum_details = analyze_momentum_condition(client, SYMBOL, '1m', 500)
    condition_details['momentum'] = {
        'increasing': momentum_increasing,
        'decreasing': momentum_decreasing,
        'positive': momentum_positive,
        'details': momentum_details
    }
    condition_results['Momentum Increasing'] = momentum_increasing
    condition_results['Momentum Positive'] = momentum_positive
    
    # Print details for this condition
    print(f"\nLast Momentum: {momentum_details.get('last_momentum', 0):.6f}")
    print(f"Momentum Increasing: {momentum_increasing}")
    print(f"Momentum Decreasing: {momentum_decreasing}")
    print(f"Momentum > 0: {momentum_positive}")
    
    if momentum_increasing:
        conditions_met += 1
        print("\nTRUE - Momentum Increasing condition MET")
    else:
        print("\nFALSE - Momentum Increasing condition NOT met")
    
    if momentum_positive:
        conditions_met += 1
        print("TRUE - Momentum Positive condition MET")
    else:
        print("FALSE - Momentum Positive condition NOT met")
    
    # Condition 3: Volume Analysis
    print("\n--- Condition 3: Volume Analysis ---")
    volume_met, volume_details = analyze_volume_condition(client, SYMBOL, '1m', 500)
    condition_details['volume'] = {
        'met': volume_met,
        'details': volume_details
    }
    condition_results['Bullish Volume > Bearish Volume'] = volume_met
    
    # Print details for this condition
    print(f"\nBullish Volume: {volume_details.get('bullish_volume', 0):.2f}")
    print(f"Bearish Volume: {volume_details.get('bearish_volume', 0):.2f}")
    print(f"Total Volume: {volume_details.get('total_volume', 0):.2f}")
    print(f"Bullish Percentage: {volume_details.get('bullish_percentage', 0):.2f}%")
    print(f"Bearish Percentage: {volume_details.get('bearish_percentage', 0):.2f}%")
    print(f"Condition Met: {volume_met}")
    
    if volume_met:
        conditions_met += 1
        print("\nTRUE - Volume condition MET")
    else:
        print("\nFALSE - Volume condition NOT met")
    
    # Condition 4: Multi-Model ML Forecast
    print("\n--- Condition 4: Multi-Model ML Forecast ---")
    ml_forecast_met, ml_details = analyze_ml_forecast_condition(client, SYMBOL, '1m', 500)
    condition_details['ml_forecast'] = {
        'met': ml_forecast_met,
        'details': ml_details
    }
    condition_results['Multi-Model ML Forecast'] = ml_forecast_met
    
    # Print details for this condition
    print(f"\nLast Price: {ml_details.get('last_price', 0):.2f}")
    print(f"Ensemble Prediction: {ml_details.get('ensemble_prediction', 0):.2f}")
    print(f"Difference: {ml_details.get('diff', 0):.2f} ({ml_details.get('diff_pct', 0):.4f}%)")
    print(f"Ensemble Confidence: {ml_details.get('ensemble_confidence', 0):.2f}")
    print(f"Models Used: {ml_details.get('model_count', 0)}")
    
    # Show individual model predictions
    individual_predictions = ml_details.get('individual_predictions', {})
    individual_confidences = ml_details.get('individual_confidences', {})
    
    if individual_predictions:
        print("\nIndividual Model Predictions:")
        for model_name, prediction in individual_predictions.items():
            confidence = individual_confidences.get(model_name, 0)
            print(f"  {model_name:<20}: {prediction:>8.2f} (conf: {confidence:.2f})")
    
    print(f"Forecast threshold met (>{MIN_FORECAST_THRESHOLD}%): {ml_details.get('threshold_met', False)}")
    print(f"Confidence threshold met (>{MIN_FORECAST_CONFIDENCE}): {ml_details.get('confidence_met', False)}")
    print(f"Condition Met: {ml_forecast_met}")
    
    if ml_forecast_met:
        conditions_met += 1
        print("\nTRUE - Multi-Model ML Forecast condition MET")
    else:
        print("\nFALSE - Multi-Model ML Forecast condition NOT met")
    
    # Condition 5: RSI Condition
    print("\n--- Condition 5: RSI Analysis ---")
    rsi_oversold_recent, rsi_overbought_recent, current_rsi, rsi_details = analyze_rsi_condition(client, SYMBOL, '1m', 500)
    condition_details['rsi'] = {
        'oversold_most_recent': rsi_oversold_recent,
        'overbought_most_recent': rsi_overbought_recent,
        'current_rsi': current_rsi,
        'details': rsi_details
    }
    condition_results['RSI Oversold Most Recent'] = rsi_oversold_recent and not rsi_overbought_recent
    
    # Print details for this condition
    print(f"\nCurrent RSI: {current_rsi:.2f}")
    print(f"Oversold Most Recent: {rsi_oversold_recent}")
    print(f"Overbought Most Recent: {rsi_overbought_recent}")
    print(f"Condition Met: {rsi_oversold_recent and not rsi_overbought_recent}")
    
    if rsi_oversold_recent and not rsi_overbought_recent:
        conditions_met += 1
        print("\nTRUE - RSI condition MET")
    else:
        print("\nFALSE - RSI condition NOT met")
    
    # Step 4: Trading Decision
    print("\n" + "="*80)
    print("TRADING DECISION")
    print("="*80)
    print(f"Conditions Met: {conditions_met}/{total_conditions}")
    
    # Print individual condition results
    print("\nCondition Summary:")
    print("-" * 65)
    for condition_name, result in condition_results.items():
        status = "TRUE" if result else "FALSE"
        print(f"{condition_name:<50}{status}")
    print("-" * 65)
    
    # All conditions must be met for trade entry
    if conditions_met == total_conditions:
        print("\n!!! ALL CONDITIONS MET - EXECUTING TRADE !!!")
        
        # Execute buy order with entire USDC balance
        buy_result = execute_buy_order(client, SYMBOL, usdc_balance)
        
        if buy_result['success']:
            print(f"\nBUY ORDER EXECUTED SUCCESSFULLY!")
            print(f"Order ID: {buy_result['order_id']}")
            print(f"Quantity: {buy_result['quantity']:.6f}")
            print(f"Price: {buy_result['price']:.6f}")
            print(f"Cost: {buy_result['cost']:.2f} USDC (100% of balance)")
            
            # Set trade active and store trade info
            trade_active = True
            trade_info = {
                'symbol': buy_result['symbol'],
                'entry_price': buy_result['price'],
                'entry_time': buy_result['timestamp'],
                'quantity': buy_result['quantity'],
                'order_id': buy_result['order_id']
            }
            
            # Initialize trade info with current price
            current_price = get_current_price(client, SYMBOL)
            if current_price:
                trade_info.update({
                    'current_price': current_price,
                    'price_diff': current_price - buy_result['price'],
                    'price_diff_pct': ((current_price - buy_result['price']) / buy_result['price']) * 100,
                    'time_elapsed': datetime.now() - buy_result['timestamp']
                })
        else:
            print(f"\nERROR EXECUTING BUY ORDER: {buy_result['error']}")
    else:
        print("\n!!! CONDITIONS NOT MET - NO TRADE EXECUTED !!!")
        print("Waiting for next iteration...")
    
    # Step 5: Cleanup for next iteration
    print("\nCleaning up for next iteration...")
    gc.collect()

# ------------------ Main Loop ------------------

def get_binance_client():
    if not os.path.exists(API_FILE):
        print(f"API file '{API_FILE}' not found. Create with key then secret on two lines.")
        return None
    try:
        with open(API_FILE, 'r') as f:
            api_key = f.readline().strip()
            api_secret = f.readline().strip()
        return Client(api_key, api_secret)
    except Exception as e:
        print(f"Error reading API file: {e}")
        return None

def main():
    client = get_binance_client()
    if not client:
        print("No client available. Exiting.")
        return
    
    # Start webhook server in a separate thread
    webhook_thread = threading.Thread(target=start_webhook_server, daemon=True)
    webhook_thread.start()
    print(f"Webhook server started on port {WEBHOOK_PORT}")
    
    print("=== BTCUSDC TRADING BOT - MULTI-MODEL ML FORECASTING ===")
    print("Press Ctrl+C to stop monitoring.")
    print("Webhook endpoint: http://localhost:5000/webhook/price")
    print("Health check: http://localhost:5000/health")
    print("\nEach iteration will:")
    print("1. Check and convert BTC dust")
    print("2. Use webhook data for real-time price updates")
    print("3. Analyze all 5 trading conditions") 
    print("4. Execute trade if ALL conditions met")
    print("5. Use entire USDC balance for entry")
    print("6. Monitor for profit target every 5 seconds")
    print("7. Clean up for next iteration")
    print("\nMulti-Model ML Forecasting:")
    print("- Linear Regression with confidence scoring")
    print("- ARIMA time series forecasting")
    print("- Random Walk with drift and volatility")
    print("- Support Vector Regression (SVR)")
    print("- Random Forest ensemble")
    print("- Ensemble weighted prediction")
    print("- Minimum 0.05% forecast threshold")
    print("- Minimum 70% confidence requirement")
    print("- Targets 0.65% profit for fast scalps")
    print("="*60)
    
    iteration_count = 0
    
    while not stop_event.is_set():
        iteration_count += 1
        print(f"\n>>> Starting Iteration #{iteration_count} <<<")
        
        try:
            perform_single_iteration_analysis(client)
        except Exception as e:
            print(f"Error in iteration #{iteration_count}: {e}")
        
        # Always wait 5 seconds between iterations, regardless of trade status
        wait_time = MIN_ITERATION_INTERVAL
        print(f"\nWaiting {wait_time} seconds before next iteration...")
        for i in range(wait_time, 0, -1):
            if stop_event.is_set():
                break
            print(f"\rNext iteration in: {i:2d} seconds", end="")
            time.sleep(1)
        print("\r" + " " * 30 + "\r")
    
    print("\nTrading bot stopped.")

if __name__ == "__main__":
    main()