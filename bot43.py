"""
ENHANCED BTCUSDC TRADING BOT - ADVANCED FFT ANALYSIS WITH CONFIGURABLE CONDITIONS
→ Creates artificial 15-second timeframe from 1-minute data with realistic price movements
→ Uses TA-Lib HT Sine for wave analysis with argmin/argmax cycle detection
→ Implements improved FFT analysis with robust frequency filtering
→ Uses configurable conditions instead of requiring all 11 conditions
→ Analyzes cycles between argmin and argmax for both 1min and 15sec TF
→ Provides detailed frequency calculations and inverse FFT forecasting
→ Targets 0.75% profit for fast scalps
→ Uses 100% of available balance for maximum trading
→ Uses 25 decimal places for BTC precision
→ Improved dust conversion using Binance Pay API
→ Fixed timezone handling to use GMT+2
→ Enhanced dust management and trade recovery mechanism
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
from datetime import datetime, timezone, timedelta
from scipy.signal import hilbert, argrelextrema, find_peaks
from scipy.fft import fft, fftfreq, ifft
from sklearn.linear_model import LinearRegression
from decimal import Decimal, getcontext

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

# Timezone Configuration
LOCAL_TIMEZONE = timezone(timedelta(hours=2))  # GMT+2

# Timeframes for analysis
TIMEFRAMES = ['1m']  # Only using 1m timeframe as requested

# Trading Configuration
PROFIT_TARGET_PERCENT = 0.75  # Profits at 0.75% exit trades 
TOTAL_FEE_PERCENT = 0.22  # Total fee percentage (0.1% for buy + 0.1% for sell + 0.02% buffer)
MIN_TRADE_AMOUNT = 10
MAX_POSITION_PERCENT = 100  # Use 100% of available balance for maximum trading
BTC_PRECISION = 25  # Use 25 decimal places for BTC precision
MIN_BTC_THRESHOLD = 1e-10  # Minimum threshold to consider BTC balance as non-zero
MIN_USDC_THRESHOLD = 0.01  # Minimum threshold to consider USDC balance as non-zero

# Set decimal precision for calculations
getcontext().prec = 30

# Technical Indicators Configuration
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Artificial 15-second timeframe configuration
SEC15_FACTOR = 4  # 1-minute / 15 seconds = 4

# Local Dip/Top Detection Configuration
LOCAL_DIP_WINDOW = 5  # Window size for local dip/top detection (legacy, not used in new method)
LOCAL_DIP_LOOKBACK = 1200  # Number of candles to analyze for argmin/argmax detection
LOCAL_DIP_CONFIRM_LOOKBACK = 1200  # Number of candles for initial dip confirmation

# Harmonic Oscillator Configuration
HARMONICS_COUNT = 144  # Total number of harmonics to use
HARMONICS_NEGATIVE = -72  # Negative harmonics count (lowest low)
HARMONICS_POSITIVE = 72  # Positive harmonics count (highest high)

# Linear Regression Channel Configuration
LINEAR_REGRESSION_LENGTH = 360  # Length for linear regression channel

# Volume Analysis Configuration
VOLUME_BULLISH_THRESHOLD = 50.01  # Minimum bullish volume percentage to consider bullish predominant

# API Rate Limiting
MIN_ITERATION_INTERVAL = 5  # 5 seconds between iterations

# Webhook Configuration
WEBHOOK_PORT = 5000
WEBHOOK_HOST = '0.0.0.0'

# New configurable conditions - ALL must be met to trigger entry
CONFIG = {
    "conditions": {
        "dmi_up_trend": True,
        "ml_forecast_up_cycle": True,
        "rsi_oversold_most_recent": True,
        "volume_bullish_1m": True,
        "volume_bullish_15s": True,
        "momentum_positive": True
    },
    "min_conditions_met": 6  # ALL conditions must be met to trigger a trade
}

# Global stop event
stop_event = threading.Event()

# Trade state variables
trade_active = False
trade_info = {}
dust_converted_this_session = False  # Track if dust conversion was done this session

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
            timestamp = data.get('timestamp', datetime.now(LOCAL_TIMEZONE).isoformat())
            
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
            
            print(f"Webhook received: {price:.25f} at {timestamp}")
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

# ------------------ Enhanced Dust Conversion Functions ------------------

def convert_btc_to_usdc(client):
    """
    Convert all available BTC to USDC using market orders.
    This ensures no BTC remains after a trade exit.
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
                print(f"Quantity {quantity:.25f} is below minimum {min_qty}, cannot convert")
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
        
        print(f"BTC conversion successful! Sold {quantity:.25f} BTC at {current_price:.25f} USDC")
        return True
        
    except BinanceAPIException as e:
        print(f"Binance API error during BTC conversion: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during BTC conversion: {e}")
        return False

def convert_usdc_to_btc(client):
    """
    Convert all available USDC to BTC using market orders.
    This ensures no USDC remains after a trade entry.
    Fixed to handle small quantities properly.
    """
    try:
        print("Checking for USDC to convert...")
        usdc_balance = get_account_balance(client, 'USDC')
        
        # Distinguish between zero and very small balance
        if usdc_balance <= MIN_USDC_THRESHOLD:
            print(f"USDC balance is effectively zero: {usdc_balance:.25f}")
            return True
        
        print(f"Converting {usdc_balance:.25f} USDC to BTC...")
        
        # Get current price for information
        ticker = client.get_symbol_ticker(symbol='BTCUSDC')
        current_price = float(ticker['price'])
        
        # Calculate quantity to buy (all available USDC)
        # Use Decimal for more precise calculation with small amounts
        usdc_decimal = Decimal(str(usdc_balance))
        price_decimal = Decimal(str(current_price))
        
        # Calculate quantity with 0.99 factor for fees (1% buffer)
        quantity_decimal = (usdc_decimal / price_decimal) * Decimal('0.99')
        quantity = float(quantity_decimal)
        
        print(f"Calculated quantity before formatting: {quantity:.25f}")
        
        # Get symbol info for precision
        symbol_info = get_symbol_info(client, 'BTCUSDC')
        
        if symbol_info and 'LOT_SIZE' in symbol_info['filters']:
            lot_size_filter = symbol_info['filters']['LOT_SIZE']
            step_size = lot_size_filter['stepSize']
            min_qty = lot_size_filter['minQty']
            
            print(f"LOT_SIZE filter: min={min_qty}, max={lot_size_filter.get('maxQty', 'inf')}, step={step_size}")
            
            # For very small quantities, use rounding instead of floor to avoid zero
            if quantity < min_qty * 10:
                # Use regular rounding for small quantities
                precision = int(round(-math.log10(step_size)))
                precision = max(0, min(BTC_PRECISION, precision))
                quantity = round(quantity, precision)
            else:
                # Format quantity according to step size for larger quantities
                quantity = format_quantity(quantity, step_size)
            
            # Ensure quantity is within min/max limits
            if quantity < min_qty:
                print(f"Quantity {quantity:.25f} is below minimum {min_qty}, cannot convert")
                return False
        else:
            # Default to BTC_PRECISION decimal places if symbol info retrieval fails
            print("Warning: Could not get LOT_SIZE filter, using default precision")
            quantity = round(quantity, BTC_PRECISION)
        
        print(f"Final quantity after formatting: {quantity:.25f}")
        
        # Execute market buy order
        order = client.order_market_buy(
            symbol='BTCUSDC',
            quantity=quantity
        )
        
        print(f"USDC conversion successful! Bought {quantity:.25f} BTC at {current_price:.25f} USDC")
        return True
        
    except BinanceAPIException as e:
        print(f"Binance API error during USDC conversion: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during USDC conversion: {e}")
        return False

def check_and_convert_dust(client, in_trade=None, force_conversion=False):
    """
    Check and convert dust at the beginning of each iteration.
    If in_trade is None, determine based on current state.
    If in_trade is True, convert USDC dust to BTC.
    If in_trade is False, convert BTC dust to USDC.
    If force_conversion is True, convert dust regardless of previous conversions.
    """
    global trade_active, dust_converted_this_session
    
    # Determine trade state if not provided
    if in_trade is None:
        in_trade = trade_active
    
    # Skip dust conversion if already done this session and not forced
    if dust_converted_this_session and not force_conversion:
        print("Dust already converted this session. Skipping...")
        return True
    
    try:
        # Get current balances
        btc_balance = get_account_balance(client, 'BTC')
        usdc_balance = get_account_balance(client, 'USDC')
        
        print(f"Current balances - BTC: {btc_balance:.25f}, USDC: {usdc_balance:.25f}")
        
        if in_trade:
            # If in trade, convert any USDC dust to BTC
            if usdc_balance > MIN_USDC_THRESHOLD:
                print("Converting USDC dust to BTC while in trade...")
                if convert_usdc_to_btc(client):
                    dust_converted_this_session = True
                    return True
                else:
                    return False
            else:
                print("No USDC dust to convert while in trade")
        else:
            # If not in trade, convert any BTC dust to USDC
            if btc_balance > MIN_BTC_THRESHOLD:
                print("Converting BTC dust to USDC while not in trade...")
                if convert_btc_to_usdc(client):
                    dust_converted_this_session = True
                    return True
                else:
                    return False
            else:
                print("No BTC dust to convert while not in trade")
        
        return True
    except Exception as e:
        print(f"Error in check_and_convert_dust: {e}")
        return False

# ------------------ Trade Recovery Functions ------------------

def check_for_active_trade(client):
    """
    Check if there's an active trade by examining order history.
    Returns True if an active trade is detected, False otherwise.
    """
    try:
        # Get recent orders
        orders = client.get_all_orders(symbol=SYMBOL, limit=10)
        
        if not orders:
            return False
        
        # Sort by time (most recent first)
        orders.sort(key=lambda x: x['time'], reverse=True)
        
        # Check most recent orders
        for order in orders:
            # If we find a recent filled buy order without a corresponding sell order,
            # we might be in an active trade
            if order['status'] == 'FILLED' and order['side'] == 'BUY':
                # Check if there's a more recent sell order
                has_sell = False
                for other_order in orders:
                    if (other_order['status'] == 'FILLED' and 
                        other_order['side'] == 'SELL' and 
                        other_order['time'] > order['time']):
                        has_sell = True
                        break
                
                if not has_sell:
                    # We found a buy order without a more recent sell order
                    # This suggests we might be in an active trade
                    return True
        
        # Check current BTC balance
        btc_balance = get_account_balance(client, 'BTC')
        if btc_balance > MIN_BTC_THRESHOLD * 1000:  # Significant BTC balance
            print(f"Significant BTC balance detected: {btc_balance:.25f}")
            return True
        
        return False
    except Exception as e:
        print(f"Error checking for active trade: {e}")
        return False

def resume_active_trade(client):
    """
    Resume an active trade by reconstructing trade_info from order history.
    Returns True if successful, False otherwise.
    """
    global trade_active, trade_info
    
    try:
        # Get recent orders
        orders = client.get_all_orders(symbol=SYMBOL, limit=10)
        
        if not orders:
            return False
        
        # Sort by time (most recent first)
        orders.sort(key=lambda x: x['time'], reverse=True)
        
        # Find the most recent buy order
        buy_order = None
        for order in orders:
            if order['status'] == 'FILLED' and order['side'] == 'BUY':
                buy_order = order
                break
        
        if not buy_order:
            return False
        
        # Extract trade information
        entry_price = float(buy_order['price']) if buy_order['price'] else 0
        quantity = float(buy_order['executedQty'])
        entry_time = datetime.fromtimestamp(buy_order['time'] / 1000, tz=LOCAL_TIMEZONE)
        
        # If price is 0 (market order), get it from trades
        if entry_price == 0:
            trades = client.get_my_trades(symbol=SYMBOL, limit=10)
            for trade in trades:
                if trade['time'] == buy_order['time']:
                    entry_price = float(trade['price'])
                    break
        
        # Set trade active and store trade info
        trade_active = True
        trade_info = {
            'symbol': SYMBOL,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'quantity': quantity,
            'order_id': buy_order['orderId']
        }
        
        # Initialize trade info with current price
        current_price = get_current_price(client, SYMBOL)
        if current_price:
            trade_info.update({
                'current_price': current_price,
                'price_diff': current_price - entry_price,
                'price_diff_pct': ((current_price - entry_price) / entry_price) * 100,
                'time_elapsed': datetime.now(LOCAL_TIMEZONE) - entry_time
            })
            
            # Calculate target price with fees included
            trade_info['target_price'] = entry_price * (1 + (PROFIT_TARGET_PERCENT + TOTAL_FEE_PERCENT/100) / 100)
            trade_info['target_diff'] = trade_info['target_price'] - current_price
            trade_info['target_diff_pct'] = (trade_info['target_diff'] / current_price) * 100
            
            # Calculate actual profit after fees
            trade_info['actual_profit_pct'] = ((current_price - entry_price) / entry_price) * 100 - TOTAL_FEE_PERCENT
        
        print(f"Resumed active trade: Entry at {entry_price:.25f}, Quantity: {quantity:.25f}")
        return True
    except Exception as e:
        print(f"Error resuming active trade: {e}")
        return False

# ------------------ Data Cleaning and Artificial Timeframe Creation ------------------

def clean_ohlc_data(df):
    """Clean OHLC data of NaN and 0 values before TA-Lib implementation"""
    # Convert all OHLCV columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace NaN values with 0
    df = df.fillna(0)
    
    # Replace 0 values in OHLC with previous non-zero values
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].ffill().bfill()
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].ffill().bfill()
    
    # Replace 0 volume with previous non-zero values
    df['volume'] = df['volume'].replace(0, np.nan)
    df['volume'] = df['volume'].ffill().bfill()
    
    # Final check - ensure no NaN or 0 values remain
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].fillna(0)
        df[col] = df[col].replace(0, df[col].median())
        # If median is also 0, use a small positive value
        if df[col].median() == 0:
            df[col] = df[col].replace(0, 0.00000001)
    
    return df

def create_15sec_timeframe(df_1m):
    """Create artificial 15-second timeframe from 1-minute data with realistic price movements"""
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
        
        # Generate 4 15-second candles with more realistic price movement
        # Use a random walk with boundaries
        prices = [open_price]
        
        for i in range(1, 4):
            # Random walk with drift towards close
            drift = (close_price - prices[-1]) / (4 - i)
            volatility = (high_price - low_price) / 4
            
            # Random step with mean drift and standard deviation based on volatility
            step = np.random.normal(drift, volatility/2)
            new_price = prices[-1] + step
            
            # Ensure price stays within bounds
            new_price = max(low_price, min(high_price, new_price))
            prices.append(new_price)
        
        # Create candles
        for i in range(SEC15_FACTOR):
            # Calculate timestamp for each 15-second candle
            timestamp = row['timestamp'] + pd.Timedelta(seconds=15 * i)
            
            if i == 0:
                candle_open = open_price
            else:
                candle_open = prices[i]
            
            if i == SEC15_FACTOR - 1:
                candle_close = close_price
            else:
                candle_close = prices[i+1]
            
            # Calculate high and low with some randomness
            candle_high = max(candle_open, candle_close) * (1 + np.random.uniform(0, 0.001))
            candle_low = min(candle_open, candle_close) * (1 - np.random.uniform(0, 0.001))
            
            # Ensure high/low stay within bounds
            candle_high = min(candle_high, high_price)
            candle_low = max(candle_low, low_price)
            
            # Distribute volume with some randomness
            candle_volume = volume / SEC15_FACTOR * np.random.uniform(0.8, 1.2)
            
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
    
    # Clean the created 15-second data
    df_15sec = clean_ohlc_data(df_15sec)
    
    return df_15sec

# ------------------ Technical Analysis Functions ------------------

def calculate_rsi(df, period=RSI_PERIOD):
    """Calculate RSI indicator using TA-Lib."""
    try:
        if df is None or len(df) < period + 1:
            return None
        
        # Clean data before TA-Lib
        df_clean = clean_ohlc_data(df.copy())
        close_prices = df_clean['close'].values.astype(float)
        
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
        
        # Clean data before TA-Lib
        df_clean = clean_ohlc_data(df.copy())
        close_prices = df_clean['close'].values.astype(float)
        
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
        
        # Clean data before TA-Lib
        df_clean = clean_ohlc_data(df.copy())
        df = df.copy()
        high_prices = df_clean['high'].values.astype(float)
        low_prices = df_clean['low'].values.astype(float)
        close_prices = df_clean['close'].values.astype(float)
        
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

# ------------------ Momentum Analysis Function ------------------

def calculate_momentum(df, period=10):
    """Calculate Momentum indicator using TA-Lib."""
    try:
        if df is None or len(df) < period + 1:
            return None
        
        # Clean data before TA-Lib
        df_clean = clean_ohlc_data(df.copy())
        close_prices = df_clean['close'].values.astype(float)
        
        if TALIB_AVAILABLE:
            momentum = talib.MOM(close_prices, timeperiod=period)
        else:
            # Fallback to manual calculation
            momentum = np.zeros(len(close_prices))
            for i in range(period, len(close_prices)):
                momentum[i] = close_prices[i] - close_prices[i - period]
        
        df = df.copy()
        df['MOMENTUM'] = momentum
        return df
    except Exception as e:
        print(f"calculate_momentum error: {e}")
        return None

# ------------------ ML Linear Regression Forecast ------------------

def ml_linear_regression_forecast(df, forecast_periods=4):
    """Forecast prices using linear regression."""
    try:
        if df is None or len(df) < 20:
            return None
        
        # Clean data before analysis
        df_clean = clean_ohlc_data(df.copy())
        close_prices = df_clean['close'].values.astype(float)
        
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

# ------------------ Enhanced FFT Analysis ------------------

def improved_fft_forecast(close_prices, forecast_periods=4):
    """Improved FFT analysis with robust frequency filtering."""
    try:
        # Detrend the data
        detrended = close_prices - np.mean(close_prices)
        
        # Apply FFT
        fft_values = fft(detrended)
        fft_freq = fftfreq(len(close_prices))
        
        # Calculate power spectrum
        power = np.abs(fft_values) ** 2
        
        # Filter frequencies (keep only significant ones)
        threshold = np.max(power) * 0.1  # Keep frequencies with 10% of max power
        significant_freqs = np.where(power > threshold)[0]
        
        # Create filtered FFT
        filtered_fft = np.zeros_like(fft_values, dtype=complex)
        filtered_fft[significant_freqs] = fft_values[significant_freqs]
        
        # Inverse FFT to get forecast
        forecast = ifft(filtered_fft).real
        
        # Add trend back
        trend = np.mean(close_prices)
        forecast = forecast + trend
        
        # Return only the forecast periods
        return forecast[-forecast_periods:]
    except Exception as e:
        print(f"Error in improved FFT forecast: {e}")
        return np.array([close_prices[-1]] * forecast_periods)

def analyze_fft_cycle(client, symbol, timeframe='1m', lookback=500):
    """
    Analyze FFT cycle between argmin and argmax for both 1min and 15sec TF.
    Provides detailed frequency calculations and inverse FFT forecasting.
    """
    try:
        # Get data based on timeframe
        if timeframe == '1m':
            klines = client.get_klines(symbol=symbol, interval='1m', limit=lookback)
        else:  # 15s timeframe
            # Get 1m data and convert to 15s
            min_1m_candles = max(100, lookback // 4 + 20)
            klines_1m = client.get_klines(symbol=symbol, interval='1m', limit=min_1m_candles)
            
            if not klines_1m or len(klines_1m) < 50:
                return {"error": "Insufficient data for 15s analysis"}
                
            df_1m = pd.DataFrame(klines_1m, columns=[
                'timestamp','open','high','low','close','volume','close_time',
                'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume','ignore'])
            
            # Convert timestamp to datetime in GMT+2 timezone
            df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
            
            # Clean OHLC data
            df_1m = clean_ohlc_data(df_1m)
            
            # Create artificial 15-second timeframe
            df_15sec = create_15sec_timeframe(df_1m)
            
            # Ensure we have enough 15-second data
            if len(df_15sec) < lookback:
                return {"error": f"Insufficient 15s data: got {len(df_15sec)}, needed {lookback}"}
            
            # Trim to the requested lookback
            df_15sec = df_15sec.tail(lookback)
            
            # Convert back to klines format for consistent processing
            klines = []
            for idx, row in df_15sec.iterrows():
                klines.append([
                    int(row['timestamp'].timestamp() * 1000),  # timestamp
                    row['open'],  # open
                    row['high'],  # high
                    row['low'],  # low
                    row['close'],  # close
                    row['volume'],  # volume
                    int(row['timestamp'].timestamp() * 1000),  # close_time
                    0,  # quote_asset_volume
                    0,  # number_of_trades
                    0,  # taker_buy_base_asset_volume
                    0,  # taker_buy_quote_asset_volume
                    0   # ignore
                ])
        
        if not klines or len(klines) < 100:
            return {"error": "Insufficient data"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime in GMT+2 timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Find the index of the lowest low and highest high using argmin/argmax
        lowest_low_idx = df['low'].idxmin()
        highest_high_idx = df['high'].idxmax()
        
        # Get the values and timestamps
        lowest_low_price = df.loc[lowest_low_idx, 'low']
        lowest_low_time = df.loc[lowest_low_idx, 'timestamp']
        
        highest_high_price = df.loc[highest_high_idx, 'high']
        highest_high_time = df.loc[highest_high_idx, 'timestamp']
        
        # Determine which occurred more recently
        dip_more_recent = lowest_low_idx > highest_high_idx
        
        # Extract price data for FFT analysis
        close_prices = df['close'].values.astype(float)
        
        # Perform FFT on price data
        fft_values = fft(close_prices)
        fft_freq = fftfreq(len(close_prices))
        
        # Calculate frequency power
        power = np.abs(fft_values) ** 2
        
        # Find dominant frequencies by power
        dominant_freq_indices = np.argsort(power)[-10:]  # Top 10 frequencies by power
        dominant_freqs = fft_freq[dominant_freq_indices]
        dominant_powers = power[dominant_freq_indices]
        
        # Separate positive and negative dominant frequencies
        positive_dominant_freqs = dominant_freqs[dominant_freqs > 0]
        negative_dominant_freqs = dominant_freqs[dominant_freqs < 0]
        
        # Calculate power for positive and negative dominant frequencies
        positive_dominant_power = np.sum(power[fft_freq > 0])
        negative_dominant_power = np.sum(power[fft_freq < 0])
        total_dominant_power = positive_dominant_power + negative_dominant_power
        
        # Calculate dominance based on power
        if total_dominant_power > 0:
            positive_dominance = positive_dominant_power / total_dominant_power
            negative_dominance = negative_dominant_power / total_dominant_power
        else:
            positive_dominance = 0.5
            negative_dominance = 0.5
        
        # Get the most powerful positive and negative frequencies
        if len(positive_dominant_freqs) > 0:
            most_powerful_positive_idx = np.argmax(power[fft_freq > 0])
            most_powerful_positive_freq = fft_freq[fft_freq > 0][most_powerful_positive_idx]
            most_powerful_positive_power = power[fft_freq > 0][most_powerful_positive_idx]
        else:
            most_powerful_positive_freq = 0
            most_powerful_positive_power = 0
            
        if len(negative_dominant_freqs) > 0:
            most_powerful_negative_idx = np.argmax(power[fft_freq < 0])
            most_powerful_negative_freq = fft_freq[fft_freq < 0][most_powerful_negative_idx]
            most_powerful_negative_power = power[fft_freq < 0][most_powerful_negative_idx]
        else:
            most_powerful_negative_freq = 0
            most_powerful_negative_power = 0
        
        # Determine cycle direction based on frequency dominance
        # If positive frequencies dominate, it's an up cycle (from dip to top)
        # If negative frequencies dominate, it's a down cycle (from top to dip)
        cycle_direction = "up" if positive_dominance > negative_dominance else "down"
        
        # Use improved FFT for forecasting
        forecast_prices = improved_fft_forecast(close_prices, forecast_periods=4)
        forecast_target = forecast_prices[-1]
        
        # Ensure forecast target is realistic
        current_price = close_prices[-1]
        if forecast_target <= 0 or abs(forecast_target - current_price) > current_price * 0.05:
            # If forecast is unrealistic, use trend-adjusted current price
            if cycle_direction == "up":
                forecast_target = current_price * 1.01  # Small upward adjustment
            else:
                forecast_target = current_price * 0.99  # Small downward adjustment
        
        # Get current price from webhook if available
        current_price = get_current_price_from_webhook()
        if current_price is None:
            current_price = close_prices[-1]
        
        # Calculate percentage difference to forecast target
        forecast_diff_pct = ((forecast_target - current_price) / current_price) * 100
        
        # If argmin is most recent, start up cycle with Hilbert Transform SineWave
        ht_sine_result = None
        if dip_more_recent:
            # Calculate HT Sine for up cycle
            df_ht = calculate_ht_sine(df)
            if df_ht is not None and 'HT_SINE' in df_ht.columns:
                ht_sine = df_ht['HT_SINE'].values
                ht_leadsine = df_ht['HT_LEADSINE'].values
                
                # Get the last values
                last_ht_sine = float(ht_sine[-1])
                last_ht_leadsine = float(ht_leadsine[-1])
                
                # Determine sine wave phase
                if last_ht_sine < last_ht_leadsine:
                    sine_phase = "rising"
                else:
                    sine_phase = "falling"
                
                ht_sine_result = {
                    "last_ht_sine": last_ht_sine,
                    "last_ht_leadsine": last_ht_leadsine,
                    "sine_phase": sine_phase
                }
        
        # Analyze last 200 values for additional cycle information
        last_200_df = df.tail(200)
        last_200_low_idx = last_200_df['low'].idxmin()
        last_200_high_idx = last_200_df['high'].idxmax()
        last_200_low_price = last_200_df.loc[last_200_low_idx, 'low']
        last_200_high_price = last_200_df.loc[last_200_high_idx, 'high']
        last_200_dip_more_recent = last_200_low_idx > last_200_high_idx
        
        # Prepare results
        results = {
            "timeframe": timeframe,
            "cycle_direction": cycle_direction,
            "current_price": current_price,
            "forecast_target": forecast_target,
            "forecast_diff_pct": forecast_diff_pct,
            "lowest_low_price": lowest_low_price,
            "lowest_low_time": lowest_low_time,
            "highest_high_price": highest_high_price,
            "highest_high_time": highest_high_time,
            "dip_more_recent": dip_more_recent,
            "last_200_low_price": last_200_low_price,
            "last_200_high_price": last_200_high_price,
            "last_200_dip_more_recent": last_200_dip_more_recent,
            "frequency_analysis": {
                "most_powerful_positive_freq": most_powerful_positive_freq,
                "most_powerful_positive_power": float(most_powerful_positive_power),
                "most_powerful_negative_freq": most_powerful_negative_freq,
                "most_powerful_negative_power": float(most_powerful_negative_power),
                "positive_dominance_pct": positive_dominance * 100,
                "negative_dominance_pct": negative_dominance * 100,
                "positive_power": float(positive_dominant_power),
                "negative_power": float(negative_dominant_power)
            },
            "ht_sine": ht_sine_result
        }
        
        return results
        
    except Exception as e:
        print(f"Error analyzing FFT cycle: {e}")
        return {"error": str(e)}

# ------------------ Volume Analysis Function ------------------

def analyze_volume_condition(client, symbol, lookback=500, timeframe='1m'):
    """
    Analyze volume condition on specified timeframe.
    Calculates bullish vs bearish volume percentages and determines predominant sentiment.
    """
    try:
        # Get data based on timeframe
        if timeframe == '1m':
            klines = client.get_klines(symbol=symbol, interval='1m', limit=lookback)
        else:  # 15s timeframe
            # Get 1m data and convert to 15s
            min_1m_candles = max(100, lookback // 4 + 20)
            klines_1m = client.get_klines(symbol=symbol, interval='1m', limit=min_1m_candles)
            
            if not klines_1m or len(klines_1m) < 50:
                return False, {"error": "Insufficient data for 15s volume analysis"}
                
            df_1m = pd.DataFrame(klines_1m, columns=[
                'timestamp','open','high','low','close','volume','close_time',
                'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume','ignore'])
            
            # Convert timestamp to datetime in GMT+2 timezone
            df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
            
            # Clean OHLC data
            df_1m = clean_ohlc_data(df_1m)
            
            # Create artificial 15-second timeframe
            df_15sec = create_15sec_timeframe(df_1m)
            
            # Ensure we have enough 15-second data
            if len(df_15sec) < lookback:
                return False, {"error": f"Insufficient 15s data: got {len(df_15sec)}, needed {lookback}"}
            
            # Trim to the requested lookback
            df_15sec = df_15sec.tail(lookback)
            
            # Convert back to klines format for consistent processing
            klines = []
            for idx, row in df_15sec.iterrows():
                klines.append([
                    int(row['timestamp'].timestamp() * 1000),  # timestamp
                    row['open'],  # open
                    row['high'],  # high
                    row['low'],  # low
                    row['close'],  # close
                    row['volume'],  # volume
                    int(row['timestamp'].timestamp() * 1000),  # close_time
                    0,  # quote_asset_volume
                    0,  # number_of_trades
                    0,  # taker_buy_base_asset_volume
                    0,  # taker_buy_quote_asset_volume
                    0   # ignore
                ])
        
        if not klines or len(klines) < 100:
            return False, {"error": "Insufficient data for volume analysis"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime in GMT+2 timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Calculate volume sentiment
        bullish_volume = 0.0
        bearish_volume = 0.0
        
        for i in range(1, len(df)):
            if df.iloc[i]['close'] > df.iloc[i-1]['close']:
                bullish_volume += df.iloc[i]['volume']
            else:
                bearish_volume += df.iloc[i]['volume']
        
        total_volume = bullish_volume + bearish_volume
        if total_volume > 0:
            volume_bullish_pct = (bullish_volume / total_volume) * 100
            volume_bearish_pct = (bearish_volume / total_volume) * 100
            volume_sentiment = "bullish" if volume_bullish_pct > volume_bearish_pct else "bearish"
        else:
            volume_bullish_pct = 0.0
            volume_bearish_pct = 0.0
            volume_sentiment = "neutral"
        
        # Determine if volume is predominantly bullish
        volume_bullish_predominant = volume_sentiment == "bullish" and volume_bullish_pct >= VOLUME_BULLISH_THRESHOLD
        
        # Prepare results
        results = {
            "timeframe": timeframe,
            "bullish_volume": bullish_volume,
            "bearish_volume": bearish_volume,
            "total_volume": total_volume,
            "bullish_pct": volume_bullish_pct,
            "bearish_pct": volume_bearish_pct,
            "sentiment": volume_sentiment,
            "predominant": volume_bullish_predominant
        }
        
        return volume_bullish_predominant, results
        
    except Exception as e:
        print(f"Error analyzing volume condition: {e}")
        return False, {"error": str(e)}

# ------------------ Enhanced Analysis Functions ------------------

def analyze_dmi_condition(client, symbol, lookback=500, timeframe='1m'):
    """Analyze DMI condition on specified timeframe."""
    try:
        # Get data based on timeframe
        if timeframe == '1m':
            klines = client.get_klines(symbol=symbol, interval='1m', limit=lookback)
        else:  # 15s timeframe
            # Get 1m data and convert to 15s
            min_1m_candles = max(100, lookback // 4 + 20)
            klines_1m = client.get_klines(symbol=symbol, interval='1m', limit=min_1m_candles)
            
            if not klines_1m or len(klines_1m) < 50:
                return False, {"error": "Insufficient data for 15s DMI analysis"}
                
            df_1m = pd.DataFrame(klines_1m, columns=[
                'timestamp','open','high','low','close','volume','close_time',
                'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume','ignore'])
            
            # Convert timestamp to datetime in GMT+2 timezone
            df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
            
            # Clean OHLC data
            df_1m = clean_ohlc_data(df_1m)
            
            # Create artificial 15-second timeframe
            df_15sec = create_15sec_timeframe(df_1m)
            
            # Ensure we have enough 15-second data
            if len(df_15sec) < lookback:
                return False, {"error": f"Insufficient 15s data: got {len(df_15sec)}, needed {lookback}"}
            
            # Trim to the requested lookback
            df_15sec = df_15sec.tail(lookback)
            
            # Convert back to klines format for consistent processing
            klines = []
            for idx, row in df_15sec.iterrows():
                klines.append([
                    int(row['timestamp'].timestamp() * 1000),  # timestamp
                    row['open'],  # open
                    row['high'],  # high
                    row['low'],  # low
                    row['close'],  # close
                    row['volume'],  # volume
                    int(row['timestamp'].timestamp() * 1000),  # close_time
                    0,  # quote_asset_volume
                    0,  # number_of_trades
                    0,  # taker_buy_base_asset_volume
                    0,  # taker_buy_quote_asset_volume
                    0   # ignore
                ])
        
        if not klines or len(klines) < 50:
            return False, {"error": "Insufficient data for DMI analysis"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime in GMT+2 timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Calculate DMI
        df = calculate_dmi(df)
        
        if df is None or 'DMI' not in df.columns:
            return False, {"error": "Failed to calculate DMI"}
        
        # Get the last DMI value
        last_dmi = float(df['DMI'].iloc[-1])
        
        # Get the last ADX value for trend strength
        last_adx = float(df['ADX'].iloc[-1])
        
        # Condition: DMI is positive (indicating upward trend)
        condition_met = last_dmi > 0
        
        details = {
            "timeframe": timeframe,
            "last_dmi": last_dmi,
            "last_adx": last_adx,
            "condition_met": condition_met
        }
        
        return condition_met, details
        
    except Exception as e:
        print(f"analyze_dmi_condition error: {e}")
        return False, {"error": str(e)}

def analyze_momentum_condition(client, symbol, lookback=500, timeframe='1m'):
    """Analyze Momentum condition on specified timeframe."""
    try:
        # Get data based on timeframe
        if timeframe == '1m':
            klines = client.get_klines(symbol=symbol, interval='1m', limit=lookback)
        else:  # 15s timeframe
            # Get 1m data and convert to 15s
            min_1m_candles = max(100, lookback // 4 + 20)
            klines_1m = client.get_klines(symbol=symbol, interval='1m', limit=min_1m_candles)
            
            if not klines_1m or len(klines_1m) < 50:
                return False, {"error": "Insufficient data for 15s momentum analysis"}
                
            df_1m = pd.DataFrame(klines_1m, columns=[
                'timestamp','open','high','low','close','volume','close_time',
                'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume','ignore'])
            
            # Convert timestamp to datetime in GMT+2 timezone
            df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
            
            # Clean OHLC data
            df_1m = clean_ohlc_data(df_1m)
            
            # Create artificial 15-second timeframe
            df_15sec = create_15sec_timeframe(df_1m)
            
            # Ensure we have enough 15-second data
            if len(df_15sec) < lookback:
                return False, {"error": f"Insufficient 15s data: got {len(df_15sec)}, needed {lookback}"}
            
            # Trim to the requested lookback
            df_15sec = df_15sec.tail(lookback)
            
            # Convert back to klines format for consistent processing
            klines = []
            for idx, row in df_15sec.iterrows():
                klines.append([
                    int(row['timestamp'].timestamp() * 1000),  # timestamp
                    row['open'],  # open
                    row['high'],  # high
                    row['low'],  # low
                    row['close'],  # close
                    row['volume'],  # volume
                    int(row['timestamp'].timestamp() * 1000),  # close_time
                    0,  # quote_asset_volume
                    0,  # number_of_trades
                    0,  # taker_buy_base_asset_volume
                    0,  # taker_buy_quote_asset_volume
                    0   # ignore
                ])
        
        if not klines or len(klines) < 50:
            return False, {"error": "Insufficient data for momentum analysis"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime in GMT+2 timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Calculate Momentum
        df = calculate_momentum(df)
        
        if df is None or 'MOMENTUM' not in df.columns:
            return False, {"error": "Failed to calculate Momentum"}
        
        # Get the last Momentum value
        last_momentum = float(df['MOMENTUM'].iloc[-1])
        
        # Condition: Momentum is positive (indicating upward momentum)
        condition_met = last_momentum > 0
        
        details = {
            "timeframe": timeframe,
            "last_momentum": last_momentum,
            "condition_met": condition_met
        }
        
        return condition_met, details
        
    except Exception as e:
        print(f"analyze_momentum_condition error: {e}")
        return False, {"error": str(e)}

def analyze_ml_forecast_condition(client, symbol, lookback=500, timeframe='1m'):
    """Analyze ML linear regression forecast for up cycle."""
    try:
        # Get data based on timeframe
        if timeframe == '1m':
            klines = client.get_klines(symbol=symbol, interval='1m', limit=lookback)
        else:  # 15s timeframe
            # Get 1m data and convert to 15s
            min_1m_candles = max(100, lookback // 4 + 20)
            klines_1m = client.get_klines(symbol=symbol, interval='1m', limit=min_1m_candles)
            
            if not klines_1m or len(klines_1m) < 50:
                return False, {"error": "Insufficient data for 15s ML forecast analysis"}
                
            df_1m = pd.DataFrame(klines_1m, columns=[
                'timestamp','open','high','low','close','volume','close_time',
                'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume','ignore'])
            
            # Convert timestamp to datetime in GMT+2 timezone
            df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
            
            # Clean OHLC data
            df_1m = clean_ohlc_data(df_1m)
            
            # Create artificial 15-second timeframe
            df_15sec = create_15sec_timeframe(df_1m)
            
            # Ensure we have enough 15-second data
            if len(df_15sec) < lookback:
                return False, {"error": f"Insufficient 15s data: got {len(df_15sec)}, needed {lookback}"}
            
            # Trim to the requested lookback
            df_15sec = df_15sec.tail(lookback)
            
            # Convert back to klines format for consistent processing
            klines = []
            for idx, row in df_15sec.iterrows():
                klines.append([
                    int(row['timestamp'].timestamp() * 1000),  # timestamp
                    row['open'],  # open
                    row['high'],  # high
                    row['low'],  # low
                    row['close'],  # close
                    row['volume'],  # volume
                    int(row['timestamp'].timestamp() * 1000),  # close_time
                    0,  # quote_asset_volume
                    0,  # number_of_trades
                    0,  # taker_buy_base_asset_volume
                    0,  # taker_buy_quote_asset_volume
                    0   # ignore
                ])
        
        if not klines or len(klines) < 50:
            return False, {"error": "Insufficient data for ML forecast analysis"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime in GMT+2 timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Apply ML linear regression forecast
        df = ml_linear_regression_forecast(df)
        
        if df is None or 'forecast_1' not in df.columns:
            return False, {"error": "Failed to apply ML linear regression forecast"}
        
        # Get the last price and forecast
        last_price = float(df['close'].iloc[-1])
        forecast_1 = float(df['forecast_1'].iloc[-1])
        
        # Calculate the difference
        diff = forecast_1 - last_price
        diff_pct = (diff / last_price) * 100 if last_price > 0 else 0
        
        # Condition: forecast suggests an up cycle (positive difference)
        condition_met = diff > 0
        
        details = {
            "timeframe": timeframe,
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

def analyze_rsi_condition(client, symbol, lookback=500, timeframe='1m'):
    """Analyze RSI oversold/overbought most recent condition."""
    try:
        # Get data based on timeframe
        if timeframe == '1m':
            klines = client.get_klines(symbol=symbol, interval='1m', limit=lookback)
        else:  # 15s timeframe
            # Get 1m data and convert to 15s
            min_1m_candles = max(100, lookback // 4 + 20)
            klines_1m = client.get_klines(symbol=symbol, interval='1m', limit=min_1m_candles)
            
            if not klines_1m or len(klines_1m) < 50:
                return False, False, 0.0, {"error": "Insufficient data for 15s RSI analysis"}
                
            df_1m = pd.DataFrame(klines_1m, columns=[
                'timestamp','open','high','low','close','volume','close_time',
                'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume','ignore'])
            
            # Convert timestamp to datetime in GMT+2 timezone
            df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
            
            # Clean OHLC data
            df_1m = clean_ohlc_data(df_1m)
            
            # Create artificial 15-second timeframe
            df_15sec = create_15sec_timeframe(df_1m)
            
            # Ensure we have enough 15-second data
            if len(df_15sec) < lookback:
                return False, False, 0.0, {"error": f"Insufficient 15s data: got {len(df_15sec)}, needed {lookback}"}
            
            # Trim to the requested lookback
            df_15sec = df_15sec.tail(lookback)
            
            # Convert back to klines format for consistent processing
            klines = []
            for idx, row in df_15sec.iterrows():
                klines.append([
                    int(row['timestamp'].timestamp() * 1000),  # timestamp
                    row['open'],  # open
                    row['high'],  # high
                    row['low'],  # low
                    row['close'],  # close
                    row['volume'],  # volume
                    int(row['timestamp'].timestamp() * 1000),  # close_time
                    0,  # quote_asset_volume
                    0,  # number_of_trades
                    0,  # taker_buy_base_asset_volume
                    0,  # taker_buy_quote_asset_volume
                    0   # ignore
                ])
        
        if not klines or len(klines) < 50:
            return False, False, 0.0, {"error": "Insufficient data for RSI analysis"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime in GMT+2 timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
        
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
            "timeframe": timeframe,
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
    
    # For very small quantities, use rounding instead of floor to avoid zero
    if quantity < step_size * 10:
        return round(quantity, precision)
    
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
        
        # Use Decimal for more precise calculation with small amounts
        usdc_decimal = Decimal(str(max_usdc))
        price_decimal = Decimal(str(current_price))
        
        # Calculate quantity with 0.99 factor for fees (1% buffer)
        quantity_decimal = (usdc_decimal / price_decimal) * Decimal('0.99')
        quantity = float(quantity_decimal)
        
        print(f"Attempting to buy {symbol} with {max_usdc:.25f} USDC at price {current_price:.25f}")
        print(f"Calculated quantity before formatting: {quantity:.25f}")
        
        # Get symbol info for precision
        symbol_info = get_symbol_info(client, symbol)
        
        if symbol_info and 'LOT_SIZE' in symbol_info['filters']:
            lot_size_filter = symbol_info['filters']['LOT_SIZE']
            step_size = lot_size_filter['stepSize']
            min_qty = lot_size_filter['minQty']
            max_qty = lot_size_filter['maxQty']
            
            print(f"LOT_SIZE filter: min={min_qty}, max={max_qty}, step={step_size}")
            
            # For very small quantities, use rounding instead of floor to avoid zero
            if quantity < min_qty * 10:
                # Use regular rounding for small quantities
                precision = int(round(-math.log10(step_size)))
                precision = max(0, min(BTC_PRECISION, precision))
                quantity = round(quantity, precision)
            else:
                # Format quantity according to step size for larger quantities
                quantity = format_quantity(quantity, step_size)
            
            # Ensure quantity is within min/max limits
            if quantity < min_qty:
                return {
                    'success': False,
                    'error': f"Calculated quantity {quantity:.25f} is below minimum {min_qty}"
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
                'error': f"Final quantity {quantity:.25f} is below minimum {min_qty}"
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
            'timestamp': datetime.now(LOCAL_TIMEZONE),
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
            
            # For very small quantities, use rounding instead of floor to avoid zero
            if btc_balance < min_qty * 10:
                # Use regular rounding for small quantities
                precision = int(round(-math.log10(step_size)))
                precision = max(0, min(BTC_PRECISION, precision))
                quantity = round(btc_balance, precision)
            else:
                # Format quantity according to step size for larger quantities
                quantity = format_quantity(btc_balance, step_size)
            
            # Ensure quantity is within min/max limits
            if quantity < min_qty:
                return {
                    'success': False,
                    'error': f"Calculated quantity {quantity:.25f} is below minimum {min_qty}"
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
                'error': f"Final quantity {quantity:.25f} is below minimum {min_qty}"
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
            'timestamp': datetime.now(LOCAL_TIMEZONE),
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
                time_diff = (datetime.now(LOCAL_TIMEZONE) - last_update).total_seconds()
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
        
        # Calculate target price for 0.75% clean profit after fees
        # Target price = entry_price * (1 + 0.75% + 0.22% fee) = entry_price * 1.0097
        target_price = entry_price * (1 + (PROFIT_TARGET_PERCENT + TOTAL_FEE_PERCENT) / 100)
        
        price_diff = current_price - entry_price
        price_diff_pct = (price_diff / entry_price) * 100
        time_elapsed = datetime.now(LOCAL_TIMEZONE) - trade_info['entry_time']
        
        target_diff = target_price - current_price
        target_diff_pct = (target_diff / current_price) * 100
        
        # Calculate actual profit after fees
        actual_profit_pct = price_diff_pct - TOTAL_FEE_PERCENT
        
        # Update trade info
        trade_info.update({
            'current_price': current_price,
            'price_diff': price_diff,
            'price_diff_pct': price_diff_pct,
            'time_elapsed': time_elapsed,
            'target_price': target_price,
            'target_diff': target_diff,
            'target_diff_pct': target_diff_pct,
            'actual_profit_pct': actual_profit_pct
        })
        
        # Check for profit target
        if current_price >= target_price:
            print(f"\nPROFIT TARGET REACHED! Selling at {current_price:.25f}")
            # Execute sell order for entire BTC balance
            sell_result = execute_sell_order(client, SYMBOL)
            
            if sell_result['success']:
                print(f"SELL ORDER EXECUTED SUCCESSFULLY!")
                print(f"Order ID: {sell_result['order_id']}")
                print(f"Quantity Sold: {sell_result['quantity']:.25f}")
                print(f"Estimated Profit: {(current_price - entry_price) * quantity:.25f} USDC")
                print(f"Actual Profit After Fees: {actual_profit_pct:.25f}%")
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
    print(f"{'Entry Price:':<20}{trade_info['entry_price']:.25f}")
    print(f"{'Entry Time:':<20}{trade_info['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'Current Price:':<20}{trade_info['current_price']:.25f}")
    print(f"{'Price Difference:':<20}{trade_info['price_diff']:+.25f} ({trade_info['price_diff_pct']:+.25f}%)")
    print(f"{'Actual Profit After Fees:':<20}{trade_info['actual_profit_pct']:+.25f}%")
    print(f"{'Time Elapsed:':<20}{trade_info['time_elapsed']}")
    print(f"{'Target Price:':<20}{trade_info['target_price']:.25f}")
    print(f"{'Distance to Target:':<20}{trade_info['target_diff']:+.25f} ({trade_info['target_diff_pct']:+.25f}%)")
    print(f"{'Quantity:':<20}{trade_info['quantity']:.25f}")
    print("="*80)

# ------------------ Main Analysis Function ------------------

def perform_single_iteration_analysis(client):
    """Perform single iteration analysis with all conditions including harmonic oscillators."""
    global trade_active, trade_info, dust_converted_this_session
    
    # Clear screen for fresh iteration
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("="*80)
    print(f"ENHANCED BTCUSDC TRADING BOT - ADVANCED FFT ANALYSIS")
    print(f"Time: {datetime.now(LOCAL_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')} (GMT+2)")
    print(f"Webhook Status: {'Connected' if webhook_data['current_price'] else 'Waiting for data'}")
    if webhook_data['current_price']:
        print(f"Webhook Price: {webhook_data['current_price']:.25f}")
    print("="*80)
    
    # Step 0: Check for active trade and resume if necessary
    if not trade_active:
        print("\n0. Checking for active trade...")
        if check_for_active_trade(client):
            print("Active trade detected! Attempting to resume...")
            if resume_active_trade(client):
                print("Successfully resumed active trade!")
                # Convert dust only once when resuming a trade
                check_and_convert_dust(client, in_trade=True, force_conversion=True)
            else:
                print("Failed to resume active trade. Converting any BTC to USDC...")
                convert_btc_to_usdc(client)
        else:
            # No active trade, convert dust only once
            check_and_convert_dust(client, in_trade=False, force_conversion=True)
    
    # Get current balances
    usdc_balance = get_account_balance(client, 'USDC')
    btc_balance = get_account_balance(client, 'BTC')
    
    print(f"Current balances - BTC: {btc_balance:.25f}, USDC: {usdc_balance:.25f}")
    
    # If trade is active, check status and display
    if trade_active:
        print("\n>>> TRADE ACTIVE - CHECKING STATUS <<<")
        trade_closed = check_trade_status(client)
        display_trade_status()
        
        if trade_closed:
            print("\nTrade closed. Resuming normal analysis...")
            # Reset dust conversion flag for next trade
            dust_converted_this_session = False
        else:
            print("\nTrade still active. Continuing with full analysis...")
    
    if usdc_balance < MIN_TRADE_AMOUNT and not trade_active:
        print(f"!!! INSUFFICIENT USDC BALANCE - MINIMUM REQUIRED: {MIN_TRADE_AMOUNT} !!!")
        return
    
    # Step 1: Analyze FFT cycles for 1-minute timeframe
    print("\n1. Analyzing FFT cycles for 1-minute timeframe...")
    fft_1m = analyze_fft_cycle(client, SYMBOL, timeframe='1m', lookback=500)
    
    if 'error' not in fft_1m:
        print(f"\n--- FFT Analysis: 1-minute Timeframe ---")
        print(f"Cycle Direction: {fft_1m['cycle_direction']}")
        print(f"Current Price: {fft_1m['current_price']:.25f}")
        print(f"Forecast Target: {fft_1m['forecast_target']:.25f}")
        print(f"Forecast Difference: {fft_1m['forecast_diff_pct']:.25f}%")
        print(f"Lowest Low: {fft_1m['lowest_low_price']:.25f} at {fft_1m['lowest_low_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Highest High: {fft_1m['highest_high_price']:.25f} at {fft_1m['highest_high_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Most Recent: {'Lowest Low (Dip)' if fft_1m['dip_more_recent'] else 'Highest High (Top)'}")
        
        # Display frequency analysis
        freq_analysis = fft_1m.get('frequency_analysis', {})
        print(f"\nFrequency Analysis:")
        print(f"  Most Powerful Positive Frequency: {freq_analysis.get('most_powerful_positive_freq', 0):.25f}")
        print(f"  Most Powerful Positive Power: {freq_analysis.get('most_powerful_positive_power', 0):.25f}")
        print(f"  Most Powerful Negative Frequency: {freq_analysis.get('most_powerful_negative_freq', 0):.25f}")
        print(f"  Most Powerful Negative Power: {freq_analysis.get('most_powerful_negative_power', 0):.25f}")
        print(f"  Positive Dominance: {freq_analysis.get('positive_dominance_pct', 0):.25f}%")
        print(f"  Negative Dominance: {freq_analysis.get('negative_dominance_pct', 0):.25f}%")
        
        # Display HT Sine if available
        ht_sine = fft_1m.get('ht_sine')
        if ht_sine:
            print(f"\nHilbert Transform SineWave:")
            print(f"  Last HT Sine: {ht_sine.get('last_ht_sine', 0):.25f}")
            print(f"  Last HT Lead Sine: {ht_sine.get('last_ht_leadsine', 0):.25f}")
            print(f"  Sine Phase: {ht_sine.get('sine_phase', 'unknown')}")
    else:
        print(f"Error analyzing 1m FFT cycle: {fft_1m['error']}")
    
    # Step 2: Analyze FFT cycles for 15-second timeframe
    print("\n2. Analyzing FFT cycles for 15-second timeframe...")
    fft_15s = analyze_fft_cycle(client, SYMBOL, timeframe='15s', lookback=500)
    
    if 'error' not in fft_15s:
        print(f"\n--- FFT Analysis: 15-second Timeframe ---")
        print(f"Cycle Direction: {fft_15s['cycle_direction']}")
        print(f"Current Price: {fft_15s['current_price']:.25f}")
        print(f"Forecast Target: {fft_15s['forecast_target']:.25f}")
        print(f"Forecast Difference: {fft_15s['forecast_diff_pct']:.25f}%")
        print(f"Lowest Low: {fft_15s['lowest_low_price']:.25f} at {fft_15s['lowest_low_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Highest High: {fft_15s['highest_high_price']:.25f} at {fft_15s['highest_high_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Most Recent: {'Lowest Low (Dip)' if fft_15s['dip_more_recent'] else 'Highest High (Top)'}")
        
        # Display frequency analysis
        freq_analysis = fft_15s.get('frequency_analysis', {})
        print(f"\nFrequency Analysis:")
        print(f"  Most Powerful Positive Frequency: {freq_analysis.get('most_powerful_positive_freq', 0):.25f}")
        print(f"  Most Powerful Positive Power: {freq_analysis.get('most_powerful_positive_power', 0):.25f}")
        print(f"  Most Powerful Negative Frequency: {freq_analysis.get('most_powerful_negative_freq', 0):.25f}")
        print(f"  Most Powerful Negative Power: {freq_analysis.get('most_powerful_negative_power', 0):.25f}")
        print(f"  Positive Dominance: {freq_analysis.get('positive_dominance_pct', 0):.25f}%")
        print(f"  Negative Dominance: {freq_analysis.get('negative_dominance_pct', 0):.25f}%")
        
        # Display HT Sine if available
        ht_sine = fft_15s.get('ht_sine')
        if ht_sine:
            print(f"\nHilbert Transform SineWave:")
            print(f"  Last HT Sine: {ht_sine.get('last_ht_sine', 0):.25f}")
            print(f"  Last HT Lead Sine: {ht_sine.get('last_ht_leadsine', 0):.25f}")
            print(f"  Sine Phase: {ht_sine.get('sine_phase', 'unknown')}")
    else:
        print(f"Error analyzing 15s FFT cycle: {fft_15s['error']}")
    
    # Step 3: Analyze conditions
    print("\n3. Analyzing trading conditions...")
    
    conditions_met = 0
    total_conditions = 6  # Total number of conditions
    condition_results = {}  # Store individual condition results
    
    # Condition 1: DMI Up Trend (1m)
    print("\n--- Condition 1: DMI Up Trend (1m) ---")
    dmi_1m_met, dmi_1m_details = analyze_dmi_condition(client, SYMBOL, 500, '1m')
    condition_results['DMI Up Trend (1m)'] = dmi_1m_met
    
    # Print details for this condition
    print(f"\nLast DMI: {dmi_1m_details.get('last_dmi', 0):.25f}")
    print(f"Last ADX: {dmi_1m_details.get('last_adx', 0):.25f}")
    print(f"Condition Met: {dmi_1m_met}")
    
    if dmi_1m_met:
        conditions_met += 1
        print("\nTRUE - DMI Up Trend (1m) condition MET")
    else:
        print("\nFALSE - DMI Up Trend (1m) condition NOT met")
    
    # Condition 2: ML Forecast Up Cycle (1m)
    print("\n--- Condition 2: ML Forecast Up Cycle (1m) ---")
    ml_1m_met, ml_1m_details = analyze_ml_forecast_condition(client, SYMBOL, 500, '1m')
    condition_results['ML Forecast Up Cycle (1m)'] = ml_1m_met
    
    # Print details for this condition
    print(f"\nLast Price: {ml_1m_details.get('last_price', 0):.25f}")
    print(f"Forecast Price: {ml_1m_details.get('forecast_1', 0):.25f}")
    print(f"Difference: {ml_1m_details.get('diff', 0):.25f} ({ml_1m_details.get('diff_pct', 0):.25f}%)")
    print(f"Condition Met: {ml_1m_met}")
    
    if ml_1m_met:
        conditions_met += 1
        print("\nTRUE - ML Forecast Up Cycle (1m) condition MET")
    else:
        print("\nFALSE - ML Forecast Up Cycle (1m) condition NOT met")
    
    # Condition 3: RSI Oversold Most Recent (1m)
    print("\n--- Condition 3: RSI Oversold Most Recent (1m) ---")
    rsi_1m_oversold, rsi_1m_overbought, rsi_1m_value, rsi_1m_details = analyze_rsi_condition(client, SYMBOL, 500, '1m')
    rsi_1m_met = rsi_1m_oversold and not rsi_1m_overbought
    condition_results['RSI Oversold Most Recent (1m)'] = rsi_1m_met
    
    # Print details for this condition
    print(f"\nCurrent RSI: {rsi_1m_value:.25f}")
    print(f"Oversold Most Recent: {rsi_1m_oversold}")
    print(f"Overbought Most Recent: {rsi_1m_overbought}")
    print(f"Condition Met: {rsi_1m_met}")
    
    if rsi_1m_met:
        conditions_met += 1
        print("\nTRUE - RSI Oversold Most Recent (1m) condition MET")
    else:
        print("\nFALSE - RSI Oversold Most Recent (1m) condition NOT met")
    
    # Condition 4: Volume Bullish Predominant (1m)
    print("\n--- Condition 4: Volume Bullish Predominant (1m) ---")
    volume_1m_met, volume_1m_details = analyze_volume_condition(client, SYMBOL, 500, '1m')
    condition_results['Volume Bullish Predominant (1m)'] = volume_1m_met
    
    # Print details for this condition
    print(f"\nBullish Volume: {volume_1m_details.get('bullish_pct', 0):.25f}%")
    print(f"Bearish Volume: {volume_1m_details.get('bearish_pct', 0):.25f}%")
    print(f"Volume Sentiment: {volume_1m_details.get('sentiment', 'neutral')}")
    print(f"Dominant Volume: {'Bullish' if volume_1m_details.get('sentiment', 'neutral') == 'bullish' else 'Bearish'}")
    print(f"Condition Met: {volume_1m_met}")
    
    if volume_1m_met:
        conditions_met += 1
        print("\nTRUE - Volume Bullish Predominant (1m) condition MET")
    else:
        print("\nFALSE - Volume Bullish Predominant (1m) condition NOT met")
    
    # Condition 5: Volume Bullish Predominant (15s)
    print("\n--- Condition 5: Volume Bullish Predominant (15s) ---")
    volume_15s_met, volume_15s_details = analyze_volume_condition(client, SYMBOL, 500, '15s')
    condition_results['Volume Bullish Predominant (15s)'] = volume_15s_met
    
    # Print details for this condition
    print(f"\nBullish Volume: {volume_15s_details.get('bullish_pct', 0):.25f}%")
    print(f"Bearish Volume: {volume_15s_details.get('bearish_pct', 0):.25f}%")
    print(f"Volume Sentiment: {volume_15s_details.get('sentiment', 'neutral')}")
    print(f"Dominant Volume: {'Bullish' if volume_15s_details.get('sentiment', 'neutral') == 'bullish' else 'Bearish'}")
    print(f"Condition Met: {volume_15s_met}")
    
    if volume_15s_met:
        conditions_met += 1
        print("\nTRUE - Volume Bullish Predominant (15s) condition MET")
    else:
        print("\nFALSE - Volume Bullish Predominant (15s) condition NOT met")
    
    # Condition 6: Momentum > 0 (1m)
    print("\n--- Condition 6: Momentum > 0 (1m) ---")
    momentum_1m_met, momentum_1m_details = analyze_momentum_condition(client, SYMBOL, 500, '1m')
    condition_results['Momentum > 0 (1m)'] = momentum_1m_met
    
    # Print details for this condition
    print(f"\nLast Momentum: {momentum_1m_details.get('last_momentum', 0):.25f}")
    print(f"Condition Met: {momentum_1m_met}")
    
    if momentum_1m_met:
        conditions_met += 1
        print("\nTRUE - Momentum > 0 (1m) condition MET")
    else:
        print("\nFALSE - Momentum > 0 (1m) condition NOT met")
    
    # Step 4: Trading Decision
    print("\n" + "="*80)
    print("TRADING DECISION")
    print("="*80)
    print(f"Conditions Met: {conditions_met}/{total_conditions}")
    print(f"Minimum Required: {CONFIG['min_conditions_met']}")
    
    # Print individual condition results
    print("\nCondition Summary:")
    print("-" * 65)
    for condition_name, result in condition_results.items():
        status = "TRUE" if result else "FALSE"
        print(f"{condition_name:<50}{status}")
    print("-" * 65)
    
    # Check if ALL conditions are met for trade entry
    if conditions_met == CONFIG['min_conditions_met'] and not trade_active:
        print(f"\n!!! ALL CONDITIONS MET ({conditions_met}/{CONFIG['min_conditions_met']}) - EXECUTING TRADE !!!")
        
        # Execute buy order with entire USDC balance
        buy_result = execute_buy_order(client, SYMBOL, usdc_balance)
        
        if buy_result['success']:
            print(f"\nBUY ORDER EXECUTED SUCCESSFULLY!")
            print(f"Order ID: {buy_result['order_id']}")
            print(f"Quantity: {buy_result['quantity']:.25f}")
            print(f"Price: {buy_result['price']:.25f}")
            print(f"Cost: {buy_result['cost']:.25f} USDC (100% of balance)")
            
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
                    'time_elapsed': datetime.now(LOCAL_TIMEZONE) - buy_result['timestamp']
                })
                
                # Calculate target price with fees included
                trade_info['target_price'] = buy_result['price'] * (1 + (PROFIT_TARGET_PERCENT + TOTAL_FEE_PERCENT) / 100)
                trade_info['target_diff'] = trade_info['target_price'] - current_price
                trade_info['target_diff_pct'] = (trade_info['target_diff'] / current_price) * 100
                
                # Calculate actual profit after fees
                trade_info['actual_profit_pct'] = ((current_price - buy_result['price']) / buy_result['price']) * 100 - TOTAL_FEE_PERCENT
            
            # Convert any remaining USDC to BTC after entering trade
            print("\nConverting any remaining USDC to BTC after entering trade...")
            convert_usdc_to_btc(client)
        else:
            print(f"\nERROR EXECUTING BUY ORDER: {buy_result['error']}")
    elif trade_active:
        print("\n!!! TRADE ALREADY ACTIVE - NO NEW TRADE EXECUTED !!!")
        print(f"Monitoring existing trade for profit target...")
    else:
        print("\n!!! INSUFFICIENT CONDITIONS MET - NO TRADE EXECUTED !!!")
        print(f"Only {conditions_met}/{CONFIG['min_conditions_met']} conditions met.")
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
    
    print("=== ENHANCED BTCUSDC TRADING BOT - ADVANCED FFT ANALYSIS ===")
    print("Press Ctrl+C to stop monitoring.")
    print("Webhook endpoint: http://localhost:5000/webhook/price")
    print("Health check: http://localhost:5000/health")
    print("\nEach iteration will:")
    print("0. Check for active trade and resume if necessary")
    print("1. Check and convert dust only once when needed")
    print("2. Use webhook data for real-time price updates")
    print("3. Analyze FFT cycles for both 1m and 15s timeframes")
    print("4. Analyze 6 configurable trading conditions")
    print("5. Execute trade if ALL conditions are met")
    print("6. Use 100% of USDC balance for entry")
    print("7. Convert any remaining USDC to BTC after entering trade")
    print("8. Monitor for profit target every 5 seconds")
    print("9. Convert any remaining BTC to USDC after exiting trade")
    print("10. Clean up for next iteration")
    print("\nEnhanced Features:")
    print("- Improved FFT analysis with robust frequency filtering")
    print("- More realistic 15-second timeframe creation")
    print("- Configurable conditions instead of requiring all conditions")
    print("- FFT analysis between argmin and argmax for both timeframes")
    print("- Detailed frequency calculations and inverse FFT forecasting")
    print("- Hilbert Transform SineWave analysis for up cycles")
    print("- Analysis of last 200 values for additional cycle information")
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