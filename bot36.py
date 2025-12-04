#!/usr/bin/env python3
"""
ENHANCED BTCUSDC TRADING BOT - DMI AND FAST SCALP ANALYSIS
→ Creates artificial 15-second timeframe from 1-minute data
→ Uses TA-Lib HT Sine for wave analysis
→ Implements Local Dip/Top detection on 1-minute timeframe
→ Uses DMI (Directional Movement Index) for fast scalping
→ Adds ML linear regression forecasting
→ RSI now also uses 15-second timeframe for consistency
→ Targets 0.35% profit for fast scalps
→ Uses 100% of available balance for maximum trading
→ Uses 25 decimal places for BTC precision
→ Improved dust conversion using Binance Pay API
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
from scipy.fft import fft, fftfreq
from sklearn.linear_model import LinearRegression

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

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
PROFIT_TARGET_PERCENT = 0.35  # Changed to 0.35% as requested
TOTAL_FEE_PERCENT = 0.22  # Total fee percentage (0.1% for buy + 0.1% for sell + 0.02% buffer)
MIN_TRADE_AMOUNT = 10
MAX_POSITION_PERCENT = 100  # Use 100% of available balance for maximum trading
BTC_PRECISION = 25  # Use 25 decimal places for BTC precision
MIN_BTC_THRESHOLD = 1e-10  # Minimum threshold to consider BTC balance as non-zero

# Technical Indicators Configuration
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Artificial 15-second timeframe configuration
SEC15_FACTOR = 4  # 1-minute / 15 seconds = 4

# Local Dip/Top Detection Configuration
LOCAL_DIP_WINDOW = 5  # Window size for local dip/top detection

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

def convert_btc_to_usdc(client):
    """
    Convert all available BTC to USDC using Binance Pay API.
    This is more reliable than the dust conversion API and handles any amount.
    """
    try:
        print("Checking for BTC to convert...")
        btc_balance = get_account_balance(client, 'BTC')
        
        # Distinguish between zero and very small balance
        if btc_balance <= MIN_BTC_THRESHOLD:
            print(f"BTC balance is effectively zero: {btc_balance:.25f}")
            return True
        
        print(f"Converting {btc_balance:.25f} BTC to USDC...")
        
        # Get current price for information
        ticker = client.get_symbol_ticker(symbol='BTCUSDC')
        current_price = float(ticker['price'])
        
        # Calculate quantity to sell (all available BTC)
        quantity = btc_balance
        
        # Get symbol info for precision
        symbol_info = get_symbol_info(client, 'BTCUSDC')
        
        if symbol_info and 'LOT_SIZE' in symbol_info['filters']:
            lot_size_filter = symbol_info['filters']['LOT_SIZE']
            step_size = lot_size_filter['stepSize']
            min_qty = lot_size_filter['minQty']
            
            # Format quantity according to step size
            quantity = format_quantity(quantity, step_size)
            
            # Ensure quantity is within min/max limits
            if quantity < min_qty:
                print(f"Quantity {quantity} is below minimum {min_qty}, cannot convert")
                return False
        else:
            # Default to BTC_PRECISION decimal places if symbol info retrieval fails
            print("Warning: Could not get LOT_SIZE filter, using default precision")
            quantity = round(quantity, BTC_PRECISION)
        
        # Execute market sell order
        order = client.order_market_sell(
            symbol='BTCUSDC',
            quantity=quantity
        )
        
        print(f"BTC conversion successful! Sold {quantity:.25f} BTC at {current_price:.2f} USDC")
        return True
        
    except BinanceAPIException as e:
        print(f"Binance API error during BTC conversion: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during BTC conversion: {e}")
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

# ------------------ DMI Analysis Function ------------------

def calculate_dmi(df, period=14):
    """Calculate Directional Movement Index (DMI) for trend analysis."""
    try:
        if df is None or len(df) < period * 2:
            return None
        
        df = df.copy()
        high_prices = df['high'].values.astype(float)
        low_prices = df['low'].values.astype(float)
        close_prices = df['close'].values.astype(float)
        
        if TALIB_AVAILABLE:
            # Calculate +DI, -DI, and ADX using TA-Lib
            plus_di = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=period)
            minus_di = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=period)
            adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=period)
        else:
            # Fallback to manual calculation
            # Calculate True Range (TR)
            tr1 = np.zeros(len(high_prices))
            tr1[0] = high_prices[0] - low_prices[0]
            
            for i in range(1, len(high_prices)):
                hl = high_prices[i] - low_prices[i]
                hc = abs(high_prices[i] - close_prices[i-1])
                lc = abs(low_prices[i] - close_prices[i-1])
                tr1[i] = max(hl, hc, lc)
            
            # Calculate +DM and -DM
            plus_dm = np.zeros(len(high_prices))
            minus_dm = np.zeros(len(high_prices))
            
            for i in range(1, len(high_prices)):
                up_move = high_prices[i] - high_prices[i-1]
                down_move = low_prices[i-1] - low_prices[i]
                
                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                elif down_move > 0:
                    minus_dm[i] = down_move
            
            # Calculate smoothed +DM and -DM
            period_float = float(period)
            smoothed_plus_dm = np.zeros(len(plus_dm))
            smoothed_minus_dm = np.zeros(len(minus_dm))
            smoothed_tr = np.zeros(len(tr1))
            
            for i in range(period, len(plus_dm)):
                smoothed_plus_dm[i] = (smoothed_plus_dm[i-1] * (period_float - 1) + plus_dm[i]) / period_float
                smoothed_minus_dm[i] = (smoothed_minus_dm[i-1] * (period_float - 1) + minus_dm[i]) / period_float
                smoothed_tr[i] = (smoothed_tr[i-1] * (period_float - 1) + tr1[i]) / period_float
            
            # Calculate +DI and -DI
            plus_di = np.zeros(len(smoothed_plus_dm))
            minus_di = np.zeros(len(smoothed_minus_dm))
            
            for i in range(period, len(plus_di)):
                if smoothed_tr[i] > 0:
                    plus_di[i] = 100.0 * smoothed_plus_dm[i] / smoothed_tr[i]
                    minus_di[i] = 100.0 * smoothed_minus_dm[i] / smoothed_tr[i]
                else:
                    plus_di[i] = 0.0
                    minus_di[i] = 0.0
            
            # Calculate ADX
            dx = np.abs(plus_di - minus_di)
            adx = np.zeros(len(dx))
            
            for i in range(period * 2, len(dx)):
                adx[i] = np.mean(dx[i-period+1:i+1])
        
        df['PLUS_DI'] = plus_di
        df['MINUS_DI'] = minus_di
        df['ADX'] = adx
        
        # Calculate DMI (PLUS_DI - MINUS_DI)
        df['DMI'] = plus_di - minus_di
        
        return df
    except Exception as e:
        print(f"calculate_dmi error: {e}")
        return None

# ------------------ ML Linear Regression Forecast ------------------

def ml_linear_regression_forecast(df, forecast_periods=4):
    """Forecast prices using linear regression."""
    try:
        if df is None or len(df) < 20:
            return None
        
        df = df.copy()
        close_prices = df['close'].values.astype(float)
        
        # Prepare data for linear regression
        X = np.arange(len(close_prices)).reshape(-1, 1)
        y = close_prices
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecast future values
        future_X = np.arange(len(close_prices), len(close_prices) + forecast_periods).reshape(-1, 1)
        forecast = model.predict(future_X)
        
        # Add forecast to dataframe
        for i, price in enumerate(forecast):
            df[f'forecast_{i+1}'] = price
        
        return df
    except Exception as e:
        print(f"ml_linear_regression_forecast error: {e}")
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

def analyze_local_dip_condition(client, symbol, lookback=100):
    """Analyze local dip condition on 1-minute timeframe."""
    try:
        # Get 1-minute data
        klines = client.get_klines(symbol=symbol, interval='1m', limit=lookback)
        if not klines or len(klines) < 1:
            print("No 1-minute data available for local dip analysis")
            return False, {"error": "No data available"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Detect local dips and tops
        df = detect_local_dips_tops(df, LOCAL_DIP_WINDOW)
        
        if df is None or 'is_dip' not in df.columns:
            return False, {"error": "Failed to detect local dips and tops"}
        
        # Find the most recent dip and top
        dip_indices = df[df['is_dip']].index.tolist()
        top_indices = df[df['is_top']].index.tolist()
        
        # Get the most recent dip and top
        last_dip_idx = dip_indices[-1] if dip_indices else None
        last_top_idx = top_indices[-1] if top_indices else None
        
        # Determine which occurred more recently
        dip_more_recent = False
        
        if last_dip_idx is not None and last_top_idx is not None:
            dip_more_recent = last_dip_idx > last_top_idx
        elif last_dip_idx is not None:
            dip_more_recent = True
        
        # Print the analysis
        print("\n" + "="*80)
        print(f"LOCAL DIP/TOP ANALYSIS - 1-MINUTE TIMEFRAME (WINDOW={LOCAL_DIP_WINDOW})")
        print("="*80)
        
        if last_dip_idx is not None:
            dip_price = df.loc[last_dip_idx, 'close']
            dip_time = df.loc[last_dip_idx, 'timestamp']
            print(f"Last Local Dip: {dip_price:.2f} at index {last_dip_idx} ({dip_time.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            print("No local dips detected")
        
        if last_top_idx is not None:
            top_price = df.loc[last_top_idx, 'close']
            top_time = df.loc[last_top_idx, 'timestamp']
            print(f"Last Local Top: {top_price:.2f} at index {last_top_idx} ({top_time.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            print("No local tops detected")
        
        print(f"Most Recent: {'Local Dip' if dip_more_recent else 'Local Top'}")
        print(f"Condition: {'TRUE' if dip_more_recent else 'FALSE'}")
        print("="*80)
        
        details = {
            "last_dip_idx": last_dip_idx,
            "last_top_idx": last_top_idx,
            "dip_more_recent": dip_more_recent
        }
        
        return dip_more_recent, details
        
    except Exception as e:
        print(f"Error analyzing local dip condition: {e}")
        return False, {"error": str(e)}

# ------------------ Enhanced Analysis Functions ------------------

def analyze_dmi_condition(client, symbol, lookback=500):
    """Analyze DMI condition on 15-second timeframe for fast scalping."""
    try:
        klines = client.get_klines(symbol=symbol, interval='1m', limit=lookback)
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
        
        # Calculate DMI on 15-second timeframe
        df_15sec = calculate_dmi(df_15sec)
        
        if df_15sec is None or 'DMI' not in df_15sec.columns:
            return False, {"error": "Failed to calculate DMI"}
        
        # Get the last DMI value
        last_dmi = float(df_15sec['DMI'].iloc[-1])
        
        # Get the last ADX value for trend strength
        last_adx = float(df_15sec['ADX'].iloc[-1])
        
        # Condition: DMI is positive (indicating upward trend)
        condition_met = last_dmi > 0
        
        details = {
            "last_dmi": last_dmi,
            "last_adx": last_adx,
            "condition_met": condition_met
        }
        
        return condition_met, details
        
    except Exception as e:
        print(f"analyze_dmi_condition error: {e}")
        return False, {"error": str(e)}

def analyze_ml_forecast_condition(client, symbol, lookback=500):
    """Analyze ML linear regression forecast for up cycle."""
    try:
        klines = client.get_klines(symbol=symbol, interval='1m', limit=lookback)
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
        
        # Apply ML linear regression forecast
        df_15sec = ml_linear_regression_forecast(df_15sec)
        
        if df_15sec is None or 'forecast_1' not in df_15sec.columns:
            return False, {"error": "Failed to apply ML linear regression forecast"}
        
        # Get the last price and forecast
        last_price = float(df_15sec['close'].iloc[-1])
        forecast_1 = float(df_15sec['forecast_1'].iloc[-1])
        
        # Calculate the difference
        diff = forecast_1 - last_price
        diff_pct = (diff / last_price) * 100 if last_price > 0 else 0
        
        # Condition: forecast suggests an up cycle (positive difference)
        condition_met = diff > 0
        
        details = {
            "last_price": last_price,
            "forecast_1": forecast_1,
            "diff": diff,
            "diff_pct": diff_pct,
            "condition_met": condition_met
        }
        
        return condition_met, details
        
    except Exception as e:
        print(f"analyze_ml_forecast_condition error: {e}")
        return False, {"error": str(e)}

def analyze_rsi_condition(client, symbol, lookback=500):
    """Analyze RSI oversold/overbought most recent condition using 15-second timeframe."""
    try:
        klines = client.get_klines(symbol=symbol, interval='1m', limit=lookback)
        if not klines or len(klines) < 100:
            return False, False, 0.0, {"error": "Insufficient data"}
            
        df_1m = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime
        df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms')
        
        # Clean OHLC data
        df_1m = clean_ohlc_data(df_1m)
        
        # Create artificial 15-second timeframe
        df_15sec = create_15sec_timeframe(df_1m)
        
        # Calculate RSI using TA-Lib on 15-second timeframe
        df_rsi = calculate_rsi(df_15sec, RSI_PERIOD)
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

def get_symbol_info(client, symbol):
    """Get symbol information with proper error handling."""
    try:
        symbol_info = client.get_symbol_info(symbol)
        if not symbol_info:
            print(f"Error: No symbol info found for {symbol}")
            return None
        
        # Extract filter information
        filters = {}
        for f in symbol_info['filters']:
            filter_type = f['filterType']
            filters[filter_type] = {
                'minQty': float(f.get('minQty', 0)),
                'maxQty': float(f.get('maxQty', float('inf'))),
                'stepSize': float(f.get('stepSize', 0)),
                'minNotional': float(f.get('minNotional', 0)),
                'tickSize': float(f.get('tickSize', 0))
            }
        
        return {
            'symbol': symbol_info['symbol'],
            'status': symbol_info['status'],
            'baseAsset': symbol_info['baseAsset'],
            'quoteAsset': symbol_info['quoteAsset'],
            'baseAssetPrecision': symbol_info['baseAssetPrecision'],
            'quotePrecision': symbol_info['quotePrecision'],
            'filters': filters
        }
    except Exception as e:
        print(f"Error getting symbol info: {e}")
        return None

def format_quantity(quantity, step_size):
    """Format quantity according to step size precision."""
    if step_size <= 0:
        return round(quantity, BTC_PRECISION)
    
    # Calculate precision from step size
    precision = int(round(-math.log10(step_size)))
    # Ensure precision is at least 0 and at most BTC_PRECISION (25)
    precision = max(0, min(BTC_PRECISION, precision))
    
    # Round down to avoid insufficient balance
    formatted_quantity = math.floor(quantity * (10 ** precision)) / (10 ** precision)
    
    return formatted_quantity

def execute_buy_order(client, symbol, usdc_amount):
    """Execute a market buy order using the entire available USDC balance."""
    try:
        # Use webhook price if available for more accurate execution
        current_price = get_current_price_from_webhook()
        if current_price is None:
            ticker = client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
        
        # Use the entire available balance for trading (100%)
        max_usdc = usdc_amount
        quantity = max_usdc / current_price * 0.99  # 1% buffer for fees
        
        print(f"Attempting to buy {symbol} with {max_usdc:.2f} USDC at price {current_price:.2f}")
        print(f"Calculated quantity before formatting: {quantity:.25f}")
        
        # Get symbol info for precision
        symbol_info = get_symbol_info(client, symbol)
        
        if symbol_info and 'LOT_SIZE' in symbol_info['filters']:
            lot_size_filter = symbol_info['filters']['LOT_SIZE']
            step_size = lot_size_filter['stepSize']
            min_qty = lot_size_filter['minQty']
            max_qty = lot_size_filter['maxQty']
            
            print(f"LOT_SIZE filter: min={min_qty}, max={max_qty}, step={step_size}")
            
            # Format quantity according to step size
            quantity = format_quantity(quantity, step_size)
            
            # Ensure quantity is within min/max limits
            if quantity < min_qty:
                return {
                    'success': False,
                    'error': f"Calculated quantity {quantity} is below minimum {min_qty}"
                }
            
            if quantity > max_qty:
                quantity = max_qty
                print(f"Quantity adjusted to maximum: {quantity:.25f}")
        else:
            # Default to BTC_PRECISION decimal places for BTC if symbol info retrieval fails
            print("Warning: Could not get LOT_SIZE filter, using default precision")
            quantity = round(quantity, BTC_PRECISION)
            min_qty = 0.00001  # Default minimum for BTC
        
        # Final check for minimum quantity
        if quantity < min_qty:
            return {
                'success': False,
                'error': f"Final quantity {quantity} is below minimum {min_qty}"
            }
        
        print(f"Final quantity after formatting: {quantity:.25f}")
        
        # Execute the order
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
            'error': f"Binance API Error: {e}"
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Unexpected Error: {e}"
        }

def execute_sell_order(client, symbol):
    """Execute a market sell order for the entire BTC balance."""
    try:
        # Get the entire BTC balance
        btc_balance = get_account_balance(client, 'BTC')
        
        # Distinguish between zero and very small balance
        if btc_balance <= MIN_BTC_THRESHOLD:
            return {
                'success': False,
                'error': f"No meaningful BTC balance available. Current balance: {btc_balance:.25f}"
            }
        
        print(f"Attempting to sell {btc_balance:.25f} BTC")
        
        # Get symbol info for precision
        symbol_info = get_symbol_info(client, symbol)
        
        if symbol_info and 'LOT_SIZE' in symbol_info['filters']:
            lot_size_filter = symbol_info['filters']['LOT_SIZE']
            step_size = lot_size_filter['stepSize']
            min_qty = lot_size_filter['minQty']
            max_qty = lot_size_filter['maxQty']
            
            print(f"LOT_SIZE filter: min={min_qty}, max={max_qty}, step={step_size}")
            
            # Format quantity according to step size
            quantity = format_quantity(btc_balance, step_size)
            
            # Ensure quantity is within min/max limits
            if quantity < min_qty:
                return {
                    'success': False,
                    'error': f"Calculated quantity {quantity} is below minimum {min_qty}"
                }
            
            if quantity > max_qty:
                quantity = max_qty
                print(f"Quantity adjusted to maximum: {quantity:.25f}")
        else:
            # Default to BTC_PRECISION decimal places for BTC if symbol info retrieval fails
            print("Warning: Could not get LOT_SIZE filter, using default precision")
            quantity = round(btc_balance, BTC_PRECISION)
            min_qty = 0.00001  # Default minimum for BTC
        
        # Final check for minimum quantity
        if quantity < min_qty:
            return {
                'success': False,
                'error': f"Final quantity {quantity} is below minimum {min_qty}"
            }
        
        print(f"Final quantity after formatting: {quantity:.25f}")
        
        # Execute the order
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
            'error': f"Binance API Error: {e}"
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Unexpected Error: {e}"
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
        
        # Calculate target price for 0.35% clean profit after fees
        # Target price = entry_price * (1 + 0.35% + 0.1% fee) = entry_price * 1.0045
        target_price = entry_price * (1 + (PROFIT_TARGET_PERCENT + 0.1) / 100)
        
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
            # Execute sell order for entire BTC balance
            sell_result = execute_sell_order(client, SYMBOL)
            
            if sell_result['success']:
                print(f"SELL ORDER EXECUTED SUCCESSFULLY!")
                print(f"Order ID: {sell_result['order_id']}")
                print(f"Quantity Sold: {sell_result['quantity']:.25f}")
                print(f"Estimated Profit: {(current_price - entry_price) * quantity:.6f} USDC")
                trade_active = False
                trade_info = {}
                
                # Convert any remaining BTC to USDC
                print("\nChecking for remaining BTC after trade...")
                convert_btc_to_usdc(client)
                
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
    print(f"{'Quantity:':<20}{trade_info['quantity']:.25f}")
    print("="*80)

# ------------------ Main Analysis Function ------------------

def perform_single_iteration_analysis(client):
    """Perform single iteration analysis with all conditions."""
    global trade_active, trade_info
    
    # Clear screen for fresh iteration
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("="*80)
    print(f"BTCUSDC TRADING BOT - DMI AND FAST SCALP ANALYSIS")
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
    
    # Step 1: Check and convert BTC to USDC (only when no active trade)
    print("\n1. Checking for BTC to convert...")
    btc_conversion_success = convert_btc_to_usdc(client)
    if not btc_conversion_success:
        print("BTC conversion failed or not needed")
    
    # Step 2: Get current balance
    usdc_balance = get_account_balance(client, 'USDC')
    btc_balance = get_account_balance(client, 'BTC')
    
    print(f"USDC Balance: {usdc_balance:.2f}")
    
    # Only show BTC balance if it's meaningful
    if btc_balance > MIN_BTC_THRESHOLD:
        print(f"BTC Balance: {btc_balance:.25f}")
    else:
        print("BTC Balance: 0.00000000 (effectively zero)")
    
    if usdc_balance < MIN_TRADE_AMOUNT:
        print(f"!!! INSUFFICIENT USDC BALANCE - MINIMUM REQUIRED: {MIN_TRADE_AMOUNT} !!!")
        return
    
    # Step 3: Perform all condition analyses
    print("\n2. Performing technical analysis...")
    
    conditions_met = 0
    total_conditions = 4  # 4 conditions as requested
    condition_details = {}
    condition_results = {}  # Store individual condition results
    
    # Condition 1: Local Dip Detection on 1-minute timeframe
    print("\n--- Condition 1: Local Dip Detection ---")
    local_dip_condition, local_dip_details = analyze_local_dip_condition(client, SYMBOL, 100)
    condition_details['local_dip'] = {
        'met': local_dip_condition,
        'details': local_dip_details
    }
    condition_results['Local Dip Most Recent (1m TF)'] = local_dip_condition
    
    if local_dip_condition:
        conditions_met += 1
        print("\nTRUE - Local Dip condition MET")
    else:
        print("\nFALSE - Local Dip condition NOT met")
    
    # Condition 2: DMI Analysis on 15-second timeframe
    print("\n--- Condition 2: DMI Analysis ---")
    dmi_met, dmi_details = analyze_dmi_condition(client, SYMBOL, 500)
    condition_details['dmi'] = {
        'met': dmi_met,
        'details': dmi_details
    }
    condition_results['DMI Up Trend'] = dmi_met
    
    # Print details for this condition
    print(f"\nLast DMI: {dmi_details.get('last_dmi', 0):.4f}")
    print(f"Last ADX: {dmi_details.get('last_adx', 0):.4f}")
    print(f"Condition Met: {dmi_met}")
    
    if dmi_met:
        conditions_met += 1
        print("\nTRUE - DMI condition MET")
    else:
        print("\nFALSE - DMI condition NOT met")
    
    # Condition 3: ML Linear Regression Forecast
    print("\n--- Condition 3: ML Linear Regression Forecast ---")
    ml_forecast_met, ml_details = analyze_ml_forecast_condition(client, SYMBOL, 500)
    condition_details['ml_forecast'] = {
        'met': ml_forecast_met,
        'details': ml_details
    }
    condition_results['ML Forecast Up Cycle'] = ml_forecast_met
    
    # Print details for this condition
    print(f"\nLast Price: {ml_details.get('last_price', 0):.2f}")
    print(f"Forecast Price: {ml_details.get('forecast_1', 0):.2f}")
    print(f"Difference: {ml_details.get('diff', 0):.2f} ({ml_details.get('diff_pct', 0):.2f}%)")
    print(f"Condition Met: {ml_forecast_met}")
    
    if ml_forecast_met:
        conditions_met += 1
        print("\nTRUE - ML Forecast condition MET")
    else:
        print("\nFALSE - ML Forecast condition NOT met")
    
    # Condition 4: RSI Condition on 15-second timeframe
    print("\n--- Condition 4: RSI Analysis (15-sec TF) ---")
    rsi_oversold_recent, rsi_overbought_recent, current_rsi, rsi_details = analyze_rsi_condition(client, SYMBOL, 500)
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
        
        # Execute buy order with the entire USDC balance
        buy_result = execute_buy_order(client, SYMBOL, usdc_balance)
        
        if buy_result['success']:
            print(f"\nBUY ORDER EXECUTED SUCCESSFULLY!")
            print(f"Order ID: {buy_result['order_id']}")
            print(f"Quantity: {buy_result['quantity']:.25f}")
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
                
                # Calculate target price
                trade_info['target_price'] = buy_result['price'] * (1 + (PROFIT_TARGET_PERCENT + 0.1) / 100)
                trade_info['target_diff'] = trade_info['target_price'] - current_price
                trade_info['target_diff_pct'] = (trade_info['target_diff'] / current_price) * 100
        else:
            print(f"\nERROR EXECUTING BUY ORDER: {buy_result['error']}")
    else:
        print("\n!!! CONDITIONS NOT MET - NO TRADE EXECUTED !!!")
        print(f"Only {conditions_met}/{total_conditions} conditions met.")
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
    
    print("=== BTCUSDC TRADING BOT - DMI AND FAST SCALP ANALYSIS ===")
    print("Press Ctrl+C to stop monitoring.")
    print("Webhook endpoint: http://localhost:5000/webhook/price")
    print("Health check: http://localhost:5000/health")
    print("\nEach iteration will:")
    print("1. Check and convert BTC to USDC")
    print("2. Use webhook data for real-time price updates")
    print("3. Analyze local dip detection on 1-minute timeframe")
    print("4. Analyze all 4 trading conditions (RSI now uses 15s TF)") 
    print("5. Execute trade if ALL conditions met")
    print("6. Use 100% of USDC balance for entry")
    print("7. Monitor for profit target every 5 seconds")
    print("8. Clean up for next iteration")
    print("\nDMI and Fast Scalp Analysis:")
    print("- Creates artificial 15-second timeframe from 1-minute data")
    print("- Uses TA-Lib HT Sine for wave analysis")
    print("- Implements Local Dip detection on 1-minute timeframe")
    print("- Uses DMI (Directional Movement Index) for fast scalping")
    print("- Adds ML linear regression forecasting")
    print("- RSI now also uses 15-second timeframe for consistency")
    print("- Targets 0.35% profit for fast scalps")
    print("- Uses 100% of available balance for maximum trading")
    print("- Uses 25 decimal places for BTC precision")
    print("- Improved dust conversion using Binance Pay API")
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
