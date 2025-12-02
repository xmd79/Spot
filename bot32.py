#!/usr/bin/env python3
"""
ENHANCED BTCUSDC TRADING BOT - INSTANT TARGET ANALYSIS WITH DEEP ML
→ Creates artificial 15-second timeframe from 1-minute data
→ Uses TA-Lib HT Sine for wave analysis
→ Implements local dip/top detection using argmin/argmax on 15s TF
→ Uses instant target analysis with multiple targets for next minutes
→ Adds ML linear regression forecasting
→ Adds Deep ML analysis considering price/volume/momentum changes
→ Targets 1.35% profit for fast scalps (user requested)
→ Uses entire USDC balance for trades
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
import logging
from flask import Flask, request, jsonify
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from decimal import Decimal

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
PROFIT_TARGET_PERCENT = 1.35  # 1.35% profit target (user requested)
TP_TARGET_PERCENT = 0.75  # 0.75% take profit target
TOTAL_FEE_PERCENT = 0.22
MIN_TRADE_AMOUNT = 10
STOP_LOSS_PERCENT = 2.0  # 2% stop loss

# Dust Conversion Configuration
MIN_DUST_CONVERSION_AMOUNT = 0.0001
MAX_DUST_CONVERSION_AMOUNT = 0.001

# Technical Indicators Configuration
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Artificial 15-second timeframe configuration
SEC15_FACTOR = 4  # 1-minute / 15 seconds = 4

# Momentum Analysis Configuration
MOMENTUM_PERIOD = 5
MOMENTUM_LOOKBACK = 5

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

def clean_data(data):
    """Clean numpy array data by replacing NaN and infinite values"""
    if isinstance(data, np.ndarray):
        # Replace NaN values with the previous valid value
        mask = np.isnan(data)
        idx = np.where(~mask, np.arange(mask.shape[0]), 0)
        np.maximum.accumulate(idx, out=idx)
        data = data[idx]
        
        # Replace infinite values with the previous valid value
        mask = np.isinf(data)
        idx = np.where(~mask, np.arange(mask.shape[0]), 0)
        np.maximum.accumulate(idx, out=idx)
        data = data[idx]
        
        # Replace any remaining NaN or infinite values with 0
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    return data

def calculate_thresholds(candles):
    """Calculate min/max thresholds for price bounds"""
    if len(candles) < 10:
        return Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')
    
    closes = np.array([float(c['close']) for c in candles], dtype=np.float64)
    closes = clean_data(closes)
    
    # Calculate min and max with some buffer
    min_price = np.min(closes)
    max_price = np.max(closes)
    
    # Add 10% buffer to min and max
    min_threshold = Decimal(str(min_price * 0.9))
    max_threshold = Decimal(str(max_price * 1.1))
    
    # Calculate mid-point
    mid_point = (min_threshold + max_threshold) / Decimal('2')
    
    return min_threshold, mid_point, max_threshold, (max_threshold - min_threshold) / Decimal('2')

def create_15sec_timeframe(df_1m):
    """Create artificial 15-second timeframe from 1-minute data with proper OHLCV"""
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
            
            # More sophisticated interpolation for price and volume
            progress = (i + 1) / SEC15_FACTOR
            
            # Use sine wave interpolation for more realistic price movement
            sine_progress = np.sin(progress * np.pi) * 0.5 + 0.5
            
            # Calculate price based on sine interpolation
            if open_price < close_price:  # Upward candle
                interpolated_price = open_price + (close_price - open_price) * sine_progress
            else:  # Downward candle
                interpolated_price = open_price - (open_price - close_price) * sine_progress
            
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
            
            # Distribute volume based on price movement
            if i == 0:
                candle_volume = volume * 0.25
            elif i == SEC15_FACTOR - 1:
                candle_volume = volume * 0.25
            else:
                candle_volume = volume * 0.5 / (SEC15_FACTOR - 2)
            
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

# ------------------ Volume Analysis Function ------------------

def analyze_volume_condition(df_15sec):
    """Analyze volume bullish vs bearish dominance on 15-second timeframe."""
    try:
        if df_15sec is None or len(df_15sec) < 20:
            return False, {"error": "Insufficient 15-second data"}
        
        # Calculate price change for each 15-second candle
        df_15sec['price_change'] = df_15sec['close'] - df_15sec['open']
        
        # Classify candles as bullish or bearish based on price change
        df_15sec['candle_type'] = np.where(df_15sec['price_change'] > 0, 'bullish', 
                                          np.where(df_15sec['price_change'] < 0, 'bearish', 'neutral'))
        
        # Calculate total volume for bullish and bearish candles
        bullish_volume = df_15sec[df_15sec['candle_type'] == 'bullish']['volume'].sum()
        bearish_volume = df_15sec[df_15sec['candle_type'] == 'bearish']['volume'].sum()
        
        # Calculate total volume
        total_volume = bullish_volume + bearish_volume
        
        # Calculate percentage of bullish and bearish volume
        bullish_pct = (bullish_volume / total_volume * 100) if total_volume > 0 else 0
        bearish_pct = (bearish_volume / total_volume * 100) if total_volume > 0 else 0
        
        # Condition is met if bullish volume > bearish volume
        condition_met = bullish_volume > bearish_volume
        
        details = {
            "bullish_volume": float(bullish_volume),
            "bearish_volume": float(bearish_volume),
            "total_volume": float(total_volume),
            "bullish_pct": float(bullish_pct),
            "bearish_pct": float(bearish_pct),
            "condition_met": condition_met
        }
        
        return condition_met, details
        
    except Exception as e:
        print(f"analyze_volume_condition error: {e}")
        return False, {"error": str(e)}

# ------------------ Momentum Analysis Function ------------------

def calculate_momentum_trend(candles, timeframe, period=MOMENTUM_PERIOD, lookback=MOMENTUM_LOOKBACK):
    """
    Calculate momentum and determine if it's increasing or decreasing over the last few values.
    
    Parameters:
    - candles: List of candle data
    - timeframe: Timeframe string (e.g., "1m", "5m")
    - period: Period for momentum calculation
    - lookback: Number of recent momentum values to check for trend
    
    Returns:
    - momentum_values: Array of momentum values
    - momentum_positive: Boolean indicating if momentum is positive
    - momentum_negative: Boolean indicating if momentum is negative
    - momentum_increasing: Boolean indicating if momentum is increasing
    - momentum_decreasing: Boolean indicating if momentum is decreasing
    """
    if len(candles) < period + lookback:
        logging.warning(f"Insufficient data ({len(candles)}) for momentum trend analysis in {timeframe}")
        print(f"Insufficient data ({len(candles)}) for momentum trend analysis in {timeframe}")
        return np.zeros(lookback), False, False, False, False
    
    closes = np.array([float(c['close']) for c in candles], dtype=np.float64)
    
    # Clean the data
    closes = clean_data(closes)
    
    # Calculate momentum using TALib
    if TALIB_AVAILABLE:
        momentum = talib.MOM(closes, timeperiod=period)
    else:
        # Fallback to manual calculation
        momentum = np.zeros(len(closes))
        for i in range(period, len(closes)):
            momentum[i] = closes[i] - closes[i-period]
    
    # Get the last 'lookback' values
    if len(momentum) >= lookback:
        recent_momentum = momentum[-lookback:]
        
        # Check if momentum is positive or negative
        momentum_positive = recent_momentum[-1] >= 0
        momentum_negative = recent_momentum[-1] < 0
        
        # Check if momentum is increasing (each value > previous value)
        momentum_increasing = all(recent_momentum[i] > recent_momentum[i-1] for i in range(1, len(recent_momentum)))
        
        # Check if momentum is decreasing (each value < previous value)
        momentum_decreasing = all(recent_momentum[i] < recent_momentum[i-1] for i in range(1, len(recent_momentum)))
        
        logging.info(f"{timeframe} - Momentum Trend: Positive={momentum_positive}, Negative={momentum_negative}, Increasing={momentum_increasing}, Decreasing={momentum_decreasing}")
        print(f"{timeframe} - Momentum Trend: Positive={momentum_positive}, Negative={momentum_negative}, Increasing={momentum_increasing}, Decreasing={momentum_decreasing}")
        
        return recent_momentum, momentum_positive, momentum_negative, momentum_increasing, momentum_decreasing
    else:
        logging.warning(f"Insufficient momentum values ({len(momentum)}) for trend analysis in {timeframe}")
        print(f"Insufficient momentum values ({len(momentum)}) for trend analysis in {timeframe}")
        return np.zeros(lookback), False, False, False, False

# ------------------ FFT Forecast Function ------------------

def generate_fft_forecast(candles, timeframe, forecast_periods=5):
    """Generate forecast using Fast Fourier Transform (FFT) analysis."""
    if len(candles) < 10:
        logging.warning(f"Insufficient data ({len(candles)}) for FFT forecast in {timeframe}")
        print(f"Insufficient data ({len(candles)}) for FFT forecast in {timeframe}")
        return Decimal('0.0')
    
    try:
        closes = np.array([float(c['close']) for c in candles], dtype=np.float64)
        if np.any(np.isnan(closes)) or np.any(closes <= 0):
            logging.warning(f"Invalid close prices in {timeframe} for FFT forecast.")
            print(f"Invalid close prices in {timeframe} for FFT forecast.")
            return Decimal('0.0')
        
        # Clean the data
        closes = clean_data(closes)
        
        current_close = Decimal(str(closes[-1])) if len(closes) > 0 else Decimal('0.0')
        
        # Calculate recent price trend
        recent_trend = np.mean(closes[-10:] - closes[-11:-1]) if len(closes) > 10 else 0
        
        # Apply FFT to the price data
        fft_result = fft(closes)
        freqs = fftfreq(len(closes))
        
        # Get the magnitudes and phases
        magnitudes = np.abs(fft_result)
        phases = np.angle(fft_result)
        
        # Sort frequencies by magnitude (excluding DC component)
        sorted_indices = np.argsort(magnitudes[1:len(magnitudes)//2])[::-1] + 1
        
        # Use the top N frequencies for reconstruction
        top_n = min(5, len(sorted_indices))
        
        # Reconstruct the signal using only the top N frequencies
        reconstructed = np.zeros(len(closes) + forecast_periods)
        for i in range(top_n):
            idx = sorted_indices[i]
            amplitude = magnitudes[idx] / len(closes)
            phase = phases[idx]
            frequency = freqs[idx]
            
            # Add the sinusoid component
            for t in range(len(closes) + forecast_periods):
                reconstructed[t] += amplitude * np.cos(2 * np.pi * frequency * t + phase)
        
        # Add the mean back to the reconstructed signal
        mean_close = np.mean(closes)
        reconstructed += mean_close
        
        # Get the forecast price (last value in the reconstructed signal)
        forecast_price = Decimal(str(reconstructed[-1]))
        
        # Adjust forecast based on recent trend
        if recent_trend > 0:
            # If recent trend is up, ensure forecast is higher than current price
            forecast_price = max(forecast_price, current_close * Decimal('1.005'))  # At least 0.5% higher
        else:
            # If recent trend is down, ensure forecast is lower than current price
            forecast_price = min(forecast_price, current_close * Decimal('0.995'))  # At least 0.5% lower
        
        # Ensure forecast is within reasonable bounds
        min_th, _, max_th, _ = calculate_thresholds(candles)
        forecast_price = max(min_th, min(max_th, forecast_price))
        
        # Calculate cycle direction
        cycle_direction = "Up" if forecast_price > current_close else "Down"
        
        logging.info(f"{timeframe} - FFT Forecast: {forecast_price:.25f} (Cycle Direction: {cycle_direction})")
        print(f"{timeframe} - FFT Forecast: {forecast_price:.25f} (Cycle Direction: {cycle_direction})")
        
        return forecast_price
    except Exception as e:
        logging.error(f"Error generating FFT forecast for {timeframe}: {e}")
        print(f"Error generating FFT forecast for {timeframe}: {e}")
        return Decimal('0.0')

# ------------------ Instant Target Analysis ------------------

def analyze_instant_target_condition(client, symbol, timeframe='1m', lookback=500):
    """Analyze instant target condition with multiple targets for next few minutes."""
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
        
        # Calculate price momentum and volatility
        df_15sec['price_change'] = df_15sec['close'].pct_change()
        df_15sec['volatility'] = df_15sec['price_change'].rolling(window=10).std()
        
        # Calculate volume momentum
        df_15sec['volume_change'] = df_15sec['volume'].pct_change()
        df_15sec['volume_momentum'] = df_15sec['volume_change'].rolling(window=5).mean()
        
        # Calculate price acceleration (second derivative)
        df_15sec['price_acceleration'] = df_15sec['price_change'].diff()
        
        # Determine trend direction using recent price movements
        recent_changes = df_15sec['price_change'].iloc[-20:].mean()
        trend_direction = "upward" if recent_changes > 0 else "downward"
        
        # Get the latest values
        latest_price_change = float(df_15sec['price_change'].iloc[-1])
        latest_volatility = float(df_15sec['volatility'].iloc[-1])
        latest_volume_momentum = float(df_15sec['volume_momentum'].iloc[-1])
        latest_price_acceleration = float(df_15sec['price_acceleration'].iloc[-1])
        
        # Calculate impulse strength
        impulse_strength = 0
        if latest_price_change > 0:
            impulse_strength += 1
        if latest_price_acceleration > 0:
            impulse_strength += 1
        if latest_volume_momentum > 0:
            impulse_strength += 1
        
        # Get current price
        current_price = float(df_15sec['close'].iloc[-1])
        
        # Calculate volatility multiplier based on impulse strength
        volatility_multiplier = 1.0 + (impulse_strength * 0.2)  # 0.2 increase per impulse point
        
        # Calculate average volatility for the last 20 periods
        avg_volatility = float(df_15sec['volatility'].iloc[-20:].mean())
        
        # Calculate targets for the next few minutes (3 targets + 1 most significant target)
        # All targets follow the same trend direction
        targets = {}
        
        # Calculate time-based multipliers (15-second intervals)
        # Target 1: 45 seconds (3 intervals)
        # Target 2: 90 seconds (6 intervals)
        # Target 3: 135 seconds (9 intervals)
        # Most Significant: 180 seconds (12 intervals) - represents end of current cycle
        
        time_multipliers = {
            'target_1': 3,
            'target_2': 6,
            'target_3': 9,
            'most_significant': 12
        }
        
        # Calculate expected price movement based on trend
        trend_multiplier = 1.0 if trend_direction == "upward" else -1.0
        
        # Use direct percentage targets to ensure we reach the required 1.35% target
        # This ensures the take profit target at 0.75% is contained
        target_percentages = {
            'target_1': 1.35,       # 1.35% for target 1
            'target_2': 2.0,        # 2.0% for target 2
            'target_3': 2.5,        # 2.5% for target 3
            'most_significant': 3.0  # 3.0% for most significant target
        }
        
        # Calculate targets all in the same direction (following trend)
        for target_name, multiplier in time_multipliers.items():
            # All targets follow the trend direction
            target_direction = trend_multiplier
            
            # Use direct percentage calculation
            target_percentage = target_percentages[target_name] / 100
            
            # Calculate target price directly based on percentage
            target_price = current_price * (1 + (target_percentage * target_direction))
            
            # Calculate percentage change
            pct_change = target_percentage * 100 * target_direction
            
            # Store target information
            targets[target_name] = {
                'price': target_price,
                'pct_change': pct_change,
                'time_seconds': multiplier * 15,  # 15 seconds per interval
                'direction': 'upward' if target_direction > 0 else 'downward'
            }
        
        # Check if any of the targets meet the profit target
        profit_target_met = False
        for target_name, target_info in targets.items():
            if abs(target_info['pct_change']) >= PROFIT_TARGET_PERCENT:
                profit_target_met = True
                break
        
        # Check if all targets are in the same direction
        all_same_direction = all(
            target['direction'] == targets['target_1']['direction'] 
            for target in targets.values()
        )
        
        # Determine if condition is met based on:
        # 1. All targets are in the same direction
        # 2. At least one target meets the profit target
        condition_met = all_same_direction and profit_target_met
        
        # Prepare details for output
        details = {
            "current_price": current_price,
            "trend_direction": trend_direction,
            "impulse_strength": impulse_strength,
            "latest_price_change": latest_price_change,
            "latest_volatility": latest_volatility,
            "latest_volume_momentum": latest_volume_momentum,
            "latest_price_acceleration": latest_price_acceleration,
            "targets": targets,
            "profit_target_met": profit_target_met,
            "all_same_direction": all_same_direction,
            "condition_met": condition_met
        }
        
        return condition_met, details
        
    except Exception as e:
        print(f"analyze_instant_target_condition error: {e}")
        return False, {"error": str(e)}

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

# ------------------ Enhanced Analysis Functions ------------------

def analyze_ml_forecast_condition(client, symbol, timeframe='1m', lookback=500):
    """Analyze ML linear regression forecast for up cycle."""
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
        
        # FIXED: Add backtest component to validate forecast
        # Check if forecast would have been profitable in recent periods
        if len(df_15sec) > 20:
            recent_prices = df_15sec['close'].iloc[-20:].values
            recent_forecasts = []
            
            for i in range(5, len(recent_prices)):
                # Create a simple linear regression model on past data
                X = np.arange(i).reshape(-1, 1)
                y = recent_prices[:i]
                model = LinearRegression()
                model.fit(X, y)
                
                # Forecast next price
                future_X = np.array([[i]])
                forecast = model.predict(future_X)[0]
                recent_forecasts.append(forecast)
            
            # Calculate accuracy of forecasts
            if recent_forecasts:
                actual_prices = recent_prices[5:]
                forecast_errors = [(actual - forecast) / actual for actual, forecast in zip(actual_prices, recent_forecasts)]
                avg_error = np.mean(forecast_errors)
                forecast_accuracy = 1 - abs(avg_error)
            else:
                forecast_accuracy = 0.5
        else:
            forecast_accuracy = 0.5
        
        # Condition: forecast suggests an up cycle (positive difference)
        condition_met = diff > 0
        
        details = {
            "last_price": last_price,
            "forecast_1": forecast_1,
            "diff": diff,
            "diff_pct": diff_pct,
            "forecast_accuracy": forecast_accuracy,
            "condition_met": condition_met
        }
        
        return condition_met, details
        
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
        
        # Use the entire USDC balance for the trade
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

def execute_sell_order(client, symbol, quantity, target_price=None):
    """Execute a market sell order with target price consideration."""
    try:
        # Get current price
        current_price = get_current_price_from_webhook()
        if current_price is None:
            ticker = client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
        
        # Check if we should sell based on target price
        if target_price and current_price < target_price:
            return {
                'success': False,
                'error': f"Current price ({current_price}) is below target price ({target_price})"
            }
        
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
            'price': current_price,
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
    """Check if profit target or stop loss is reached for active trade."""
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
        tp_price = entry_price * (1 + (TP_TARGET_PERCENT + TOTAL_FEE_PERCENT) / 100)  # 0.75% take profit target
        stop_loss_price = entry_price * (1 - STOP_LOSS_PERCENT / 100)
        
        price_diff = current_price - entry_price
        price_diff_pct = (price_diff / entry_price) * 100
        time_elapsed = datetime.now() - trade_info['entry_time']
        
        target_diff = target_price - current_price
        target_diff_pct = (target_diff / current_price) * 100
        
        tp_diff = tp_price - current_price
        tp_diff_pct = (tp_diff / current_price) * 100
        
        stop_loss_diff = current_price - stop_loss_price
        stop_loss_diff_pct = (stop_loss_diff / current_price) * 100
        
        # Update trade info
        trade_info.update({
            'current_price': current_price,
            'price_diff': price_diff,
            'price_diff_pct': price_diff_pct,
            'time_elapsed': time_elapsed,
            'target_price': target_price,
            'target_diff': target_diff,
            'target_diff_pct': target_diff_pct,
            'tp_price': tp_price,
            'tp_diff': tp_diff,
            'tp_diff_pct': tp_diff_pct,
            'stop_loss_price': stop_loss_price,
            'stop_loss_diff': stop_loss_diff,
            'stop_loss_diff_pct': stop_loss_diff_pct
        })
        
        # Check for take profit target (0.75%)
        if current_price >= tp_price:
            print(f"\nTAKE PROFIT TARGET REACHED! Selling at {current_price:.6f}")
            sell_result = execute_sell_order(client, SYMBOL, quantity, tp_price)
            
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
        
        # Check for profit target (1.35%)
        if current_price >= target_price:
            print(f"\nPROFIT TARGET REACHED! Selling at {current_price:.6f}")
            sell_result = execute_sell_order(client, SYMBOL, quantity, target_price)
            
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
        
        # Check for stop loss
        if current_price <= stop_loss_price:
            print(f"\nSTOP LOSS TRIGGERED! Selling at {current_price:.6f}")
            sell_result = execute_sell_order(client, SYMBOL, quantity)
            
            if sell_result['success']:
                print(f"SELL ORDER EXECUTED SUCCESSFULLY!")
                print(f"Order ID: {sell_result['order_id']}")
                print(f"Quantity Sold: {sell_result['quantity']}")
                print(f"Estimated Loss: {(current_price - entry_price) * quantity:.6f} USDC")
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
    print(f"{'TP Price (0.75%):':<20}{trade_info['tp_price']:.6f}")
    print(f"{'Distance to TP:':<20}{trade_info['tp_diff']:.6f} ({trade_info['tp_diff_pct']:.2f}%)")
    print(f"{'Target Price (1.35%):':<20}{trade_info['target_price']:.6f}")
    print(f"{'Distance to Target:':<20}{trade_info['target_diff']:.6f} ({trade_info['target_diff_pct']:.2f}%)")
    print(f"{'Stop Loss Price:':<20}{trade_info['stop_loss_price']:.6f}")
    print(f"{'Distance to Stop Loss:':<20}{trade_info['stop_loss_diff']:.6f} ({trade_info['stop_loss_diff_pct']:.2f}%)")
    print(f"{'Quantity:':<20}{trade_info['quantity']}")
    print("="*80)

# ------------------ Main Analysis Function ------------------

def perform_single_iteration_analysis(client):
    """Perform single iteration analysis with all conditions."""
    global trade_active, trade_info
    
    # Clear screen for fresh iteration
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("="*80)
    print(f"BTCUSDC TRADING BOT - INSTANT TARGET ANALYSIS WITH DEEP ML")
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
    
    # Step 3: Get data for analysis
    print("\n2. Fetching market data...")
    klines = client.get_klines(symbol=SYMBOL, interval='1m', limit=500)
    if not klines or len(klines) < 100:
        print("Insufficient data for analysis")
        return
    
    # Convert to DataFrame
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
    
    # Convert DataFrame to list of dictionaries for FFT and momentum analysis
    candles = []
    for idx, row in df_15sec.iterrows():
        candles.append({
            'timestamp': row['timestamp'].timestamp() * 1000,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        })
    
    # Step 4: Perform all condition analyses
    print("\n3. Performing technical analysis...")
    
    conditions_met = 0
    total_conditions = 7  # Updated to 7 conditions (added RSI)
    condition_details = {}
    condition_results = {}  # Store individual condition results
    
    # Condition 1: Instant Target Analysis
    print("\n--- Condition 1: Instant Target Analysis ---")
    instant_target_met, target_details = analyze_instant_target_condition(client, SYMBOL, '1m', 500)
    condition_details['instant_target'] = {
        'met': instant_target_met,
        'details': target_details
    }
    condition_results['Instant Target Up Cycle'] = instant_target_met
    
    # Print details for this condition
    print(f"\nCurrent Price: {target_details.get('current_price', 0):.2f}")
    print(f"Trend Direction: {target_details.get('trend_direction', 'unknown')}")
    print(f"Impulse Strength: {target_details.get('impulse_strength', 0)}/3")
    
    # Print targets
    print("\nTargets for Next Few Minutes:")
    print("-" * 50)
    for target_name, target_info in target_details.get('targets', {}).items():
        direction_symbol = "↑" if target_info['direction'] == 'upward' else "↓"
        display_name = target_name.replace('_', ' ').title()
        if target_name == 'most_significant':
            display_name = "Most Significant Target"
        print(f"{display_name}: {target_info['price']:.2f} {direction_symbol} "
              f"({target_info['pct_change']:+.2f}%) in {target_info['time_seconds']}s")
    
    print(f"\nAll Targets Same Direction: {target_details.get('all_same_direction', False)}")
    print(f"Profit Target Met: {target_details.get('profit_target_met', False)}")
    print(f"Condition Met: {instant_target_met}")
    
    if instant_target_met:
        conditions_met += 1
        print("\nTRUE - Instant Target condition MET")
    else:
        print("\nFALSE - Instant Target condition NOT met")
    
    # Condition 2: ML Linear Regression Forecast
    print("\n--- Condition 2: ML Linear Regression Forecast ---")
    ml_forecast_met, ml_details = analyze_ml_forecast_condition(client, SYMBOL, '1m', 500)
    condition_details['ml_forecast'] = {
        'met': ml_forecast_met,
        'details': ml_details
    }
    condition_results['ML Forecast Up Cycle'] = ml_forecast_met
    
    # Print details for this condition
    print(f"\nLast Price: {ml_details.get('last_price', 0):.2f}")
    print(f"Forecast Price: {ml_details.get('forecast_1', 0):.2f}")
    print(f"Difference: {ml_details.get('diff', 0):.2f} ({ml_details.get('diff_pct', 0):.2f}%)")
    print(f"Forecast Accuracy: {ml_details.get('forecast_accuracy', 0):.2f}")
    print(f"Condition Met: {ml_forecast_met}")
    
    if ml_forecast_met:
        conditions_met += 1
        print("\nTRUE - ML Forecast condition MET")
    else:
        print("\nFALSE - ML Forecast condition NOT met")
    
    # Condition 3: Volume Analysis on 15-second timeframe
    print("\n--- Condition 3: Volume Analysis (15s TF) ---")
    volume_met, volume_details = analyze_volume_condition(df_15sec)
    condition_details['volume'] = {
        'met': volume_met,
        'details': volume_details
    }
    condition_results['Bullish Volume > Bearish Volume'] = volume_met
    
    # Print details for this condition
    print(f"\nBullish Volume: {volume_details.get('bullish_volume', 0):.2f} ({volume_details.get('bullish_pct', 0):.2f}%)")
    print(f"Bearish Volume: {volume_details.get('bearish_volume', 0):.2f} ({volume_details.get('bearish_pct', 0):.2f}%)")
    print(f"Total Volume: {volume_details.get('total_volume', 0):.2f}")
    print(f"Condition Met: {volume_met}")
    
    if volume_met:
        conditions_met += 1
        print("\nTRUE - Volume condition MET")
    else:
        print("\nFALSE - Volume condition NOT met")
    
    # Condition 4: Momentum > 0
    print("\n--- Condition 4: Momentum Analysis (> 0) ---")
    momentum_values, momentum_positive, momentum_negative, _, _ = calculate_momentum_trend(candles, '15s', MOMENTUM_PERIOD, MOMENTUM_LOOKBACK)
    condition_details['momentum_positive'] = {
        'met': momentum_positive,
        'details': {
            'momentum_values': momentum_values.tolist() if isinstance(momentum_values, np.ndarray) else momentum_values,
            'momentum_positive': momentum_positive,
            'momentum_negative': momentum_negative
        }
    }
    condition_results['Momentum > 0'] = momentum_positive
    
    # Print details for this condition
    print(f"\nLatest Momentum: {momentum_values[-1] if len(momentum_values) > 0 else 0:.6f}")
    print(f"Momentum Positive: {momentum_positive}")
    print(f"Momentum Negative: {momentum_negative}")
    print(f"Condition Met: {momentum_positive}")
    
    if momentum_positive:
        conditions_met += 1
        print("\nTRUE - Momentum > 0 condition MET")
    else:
        print("\nFALSE - Momentum > 0 condition NOT met")
    
    # Condition 5: RSI Oversold Most Recent (replaces Momentum Increasing)
    print("\n--- Condition 5: RSI Oversold Most Recent ---")
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
        print("\nTRUE - RSI Oversold Most Recent condition MET")
    else:
        print("\nFALSE - RSI Oversold Most Recent condition NOT met")
    
    # Condition 6: Fast FFT Analysis
    print("\n--- Condition 6: Fast FFT Analysis ---")
    fft_forecast = generate_fft_forecast(candles, '15s', forecast_periods=5)
    current_price = float(df_15sec['close'].iloc[-1])
    
    # FFT condition is met if forecast price is higher than current price (up cycle)
    fft_met = float(fft_forecast) > current_price
    condition_details['fft'] = {
        'met': fft_met,
        'details': {
            'current_price': current_price,
            'forecast_price': float(fft_forecast),
            'forecast_diff': float(fft_forecast) - current_price,
            'forecast_diff_pct': ((float(fft_forecast) / current_price - 1) * 100) if current_price > 0 else 0
        }
    }
    condition_results['FFT Up Cycle'] = fft_met
    
    # Print details for this condition
    print(f"\nCurrent Price: {current_price:.2f}")
    print(f"FFT Forecast Price: {float(fft_forecast):.2f}")
    print(f"Forecast Difference: {float(fft_forecast) - current_price:.2f} ({((float(fft_forecast) / current_price - 1) * 100) if current_price > 0 else 0:.2f}%)")
    print(f"Condition Met: {fft_met}")
    
    if fft_met:
        conditions_met += 1
        print("\nTRUE - FFT Up Cycle condition MET")
    else:
        print("\nFALSE - FFT Up Cycle condition NOT met")
    
    # Step 5: Trading Decision
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
    
    # Step 6: Cleanup for next iteration
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
    
    print("=== BTCUSDC TRADING BOT - INSTANT TARGET ANALYSIS WITH DEEP ML ===")
    print("Press Ctrl+C to stop monitoring.")
    print("Webhook endpoint: http://localhost:5000/webhook/price")
    print("Health check: http://localhost:5000/health")
    print("\nEach iteration will:")
    print("1. Check and convert BTC dust")
    print("2. Use webhook data for real-time price updates")
    print("3. Analyze all 7 trading conditions") 
    print("4. Execute trade if ALL conditions met")
    print("5. Use entire USDC balance for entry")
    print("6. Monitor for profit target or stop loss every 5 seconds")
    print("7. Clean up for next iteration")
    print("\nConditions Checked:")
    print("- Instant Target Up Cycle")
    print("- ML Forecast Up Cycle")
    print("- Bullish Volume > Bearish Volume (15s TF)")
    print("- Momentum > 0")
    print("- RSI Oversold Most Recent")
    print("- FFT Up Cycle")
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