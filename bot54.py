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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
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
PROFIT_TARGET_PERCENT = 1.25  # Changed from 0.25% to 1.25% as requested
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
LOCAL_DIP_WINDOW = 5  # Window size for local dip/top detection
LOCAL_DIP_LOOKBACK = 500  # Number of candles to analyze for argmin/argmax detection
LOCAL_DIP_CONFIRM_LOOKBACK = 500  # Number of candles for initial dip confirmation

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

# ML Configuration
MIN_FORECAST_THRESHOLD = 0.1  # Minimum forecast percentage to consider significant
MIN_FORECAST_CONFIDENCE = 0.6  # Minimum confidence level for ML forecasts
ML_FORECAST_PERIODS = 4  # Number of periods to forecast

# Pythagorean Harmonics Configuration
PYTHAGOREAN_TRIPLES = [
    (3, 4, 5), (5, 12, 13), (7, 24, 25), (8, 15, 17), (9, 40, 41),
    (11, 60, 61), (12, 35, 37), (13, 84, 85), (16, 63, 65), (20, 21, 29),
    (28, 45, 53), (33, 56, 65), (36, 77, 85), (39, 80, 89), (48, 55, 73)
]

# New configurable conditions - Updated to include 17 conditions (12 original + 5 new)
CONFIG = {
    "conditions": {
        "rsi_oversold_most_recent": True,
        "rsi_oversold_most_recent_15s": True,
        "fft_forecast_up_1m": True,
        "fft_forecast_up_15s": True,
        "pythagorean_harmonics_up": True,  # 1m Pythagorean Harmonics
        "pythagorean_harmonics_up_15s": True,  # 15s Pythagorean Harmonics
        "momentum_positive": True,
        "momentum_positive_15s": True,
        # Original conditions
        "close_below_middle_15s": True,  # Current close below middle threshold for 15s
        "argmin_more_recent_15s": True,  # Argmin more recent than argmax for 15s
        "close_below_middle_1m": True,  # Current close below middle threshold for 1m
        "argmin_more_recent_1m": True,  # Argmin more recent than argmax for 1m
        # New conditions
        "bullish_volume_dominance_1m": True,  # Bullish Volume Dominance (1m)
        "sma200_lowest_more_recent_1m": True,  # Lowest SMA200 more recent than highest SMA200 (1m)
        "close_below_sma200_1m": True,  # Current close below SMA200 (1m)
        "sma200_lowest_more_recent_15s": True,  # Lowest SMA200 more recent than highest SMA200 (15s)
        "close_below_sma200_15s": True  # Current close below SMA200 (15s)
    },
    "min_conditions_met": 17  # ALL 17 conditions must be met to trigger a trade
}

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

# ------------------ Pythagorean Harmonics Functions ------------------

def find_recent_extrema(df, lookback=500):
    """
    Find most recent extrema (argmin or argmax) in the last N values.
    Returns a dictionary with information about the most recent extrema.
    """
    try:
        if df is None or len(df) < lookback:
            return {"error": "Insufficient data for extrema detection"}
        
        # Get the last N values
        df_recent = df.tail(lookback)
        
        # Find the index of the lowest low and highest high using argmin and argmax
        lowest_low_idx = df_recent['low'].idxmin()
        highest_high_idx = df_recent['high'].idxmax()
        
        # Get the values and timestamps
        lowest_low_price = df_recent.loc[lowest_low_idx, 'low']
        lowest_low_time = df_recent.loc[lowest_low_idx, 'timestamp']
        
        highest_high_price = df_recent.loc[highest_high_idx, 'high']
        highest_high_time = df_recent.loc[highest_high_idx, 'timestamp']
        
        # Determine which occurred more recently
        dip_more_recent = lowest_low_idx > highest_high_idx
        
        # Get the current price
        current_price = float(df_recent['close'].iloc[-1])
        
        # Calculate the price movement from the most recent extrema
        if dip_more_recent:
            # Most recent is a dip (argmin)
            extrema_type = "dip"
            extrema_price = lowest_low_price
            extrema_time = lowest_low_time
            price_movement = current_price - extrema_price
            price_movement_pct = (price_movement / extrema_price) * 100 if extrema_price > 0 else 0
        else:
            # Most recent is a top (argmax)
            extrema_type = "top"
            extrema_price = highest_high_price
            extrema_time = highest_high_time
            price_movement = extrema_price - current_price
            price_movement_pct = (price_movement / extrema_price) * 100 if extrema_price > 0 else 0
        
        # Calculate the time elapsed since the most recent extrema
        current_time = df_recent['timestamp'].iloc[-1]
        time_elapsed = current_time - extrema_time
        time_elapsed_minutes = time_elapsed.total_seconds() / 60
        
        return {
            "extrema_type": extrema_type,
            "extrema_price": extrema_price,
            "extrema_time": extrema_time,
            "current_price": current_price,
            "price_movement": price_movement,
            "price_movement_pct": price_movement_pct,
            "time_elapsed": time_elapsed,
            "time_elapsed_minutes": time_elapsed_minutes,
            "dip_more_recent": dip_more_recent,
            "lowest_low_price": lowest_low_price,
            "lowest_low_time": lowest_low_time,
            "highest_high_price": highest_high_price,
            "highest_high_time": highest_high_time
        }
    except Exception as e:
        print(f"Error finding recent extrema: {e}")
        return {"error": str(e)}

def apply_pythagorean_harmonics(extrema_data, timeframe='1m'):
    """
    Apply Pythagorean Harmonics based on the most recent extrema.
    Returns a forecast based on the Pythagorean Theorem.
    """
    try:
        if 'error' in extrema_data:
            return {"error": extrema_data['error']}
        
        extrema_type = extrema_data['extrema_type']
        current_price = extrema_data['current_price']
        price_movement = extrema_data['price_movement']
        time_elapsed_minutes = extrema_data['time_elapsed_minutes']
        
        # Apply the Pythagorean Theorem: a² + b² = c²
        # a = Price Movement (absolute value)
        # b = Time Movement (in minutes)
        # c = Resultant Vector (energy signature)
        
        a = abs(price_movement)
        b = time_elapsed_minutes
        
        # Calculate the resultant vector
        c = math.sqrt(a**2 + b**2)
        
        # Find the closest Pythagorean triple
        closest_triple = None
        min_distance = float('inf')
        
        for triple in PYTHAGOREAN_TRIPLES:
            # Scale the triple to match our a and b values
            scale_a = a / triple[0] if triple[0] > 0 else 0
            scale_b = b / triple[1] if triple[1] > 0 else 0
            
            # Use the average scale
            scale = (scale_a + scale_b) / 2 if scale_a > 0 and scale_b > 0 else max(scale_a, scale_b)
            
            # Calculate the scaled triple
            scaled_triple = (triple[0] * scale, triple[1] * scale, triple[2] * scale)
            
            # Calculate the distance between our c and scaled triple's c
            distance = abs(c - scaled_triple[2])
            
            if distance < min_distance:
                min_distance = distance
                closest_triple = scaled_triple
        
        # Calculate the forecast based on the extrema type and closest triple
        if extrema_type == "dip":
            # Up cycle: forecast above the current price
            # Use the c value from the closest triple as the forecast movement
            forecast_movement = closest_triple[2] if closest_triple else c
            forecast_price = current_price + forecast_movement
            forecast_direction = "up"
        else:
            # Down cycle: forecast below the current price
            # Use the c value from the closest triple as the forecast movement
            forecast_movement = closest_triple[2] if closest_triple else c
            forecast_price = current_price - forecast_movement
            forecast_direction = "down"
        
        # Calculate the percentage difference to the forecast price
        forecast_diff_pct = ((forecast_price - current_price) / current_price) * 100
        
        # Calculate the confidence based on how close our a,b,c values are to a Pythagorean triple
        confidence = 1.0 - (min_distance / c) if c > 0 else 0.0
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "timeframe": timeframe,
            "extrema_type": extrema_type,
            "current_price": current_price,
            "forecast_price": forecast_price,
            "forecast_direction": forecast_direction,
            "forecast_movement": forecast_movement,
            "forecast_diff_pct": forecast_diff_pct,
            "pythagorean_values": {
                "a": a,
                "b": b,
                "c": c,
                "closest_triple": closest_triple,
                "min_distance": min_distance
            },
            "confidence": confidence,
            "extrema_data": extrema_data
        }
    except Exception as e:
        print(f"Error applying Pythagorean Harmonics: {e}")
        return {"error": str(e)}

def analyze_pythagorean_harmonics(client, symbol, timeframe='1m', lookback=500):
    """
    Analyze Pythagorean Harmonics for the specified timeframe.
    Returns a forecast based on the most recent extrema.
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
                return {"error": "Insufficient data for 15s Pythagorean Harmonics analysis"}
                
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
            return {"error": "Insufficient data for Pythagorean Harmonics analysis"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime in GMT+2 timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Find the most recent extrema
        extrema_data = find_recent_extrema(df, lookback)
        
        if 'error' in extrema_data:
            return {"error": extrema_data['error']}
        
        # Apply Pythagorean Harmonics
        harmonics_data = apply_pythagorean_harmonics(extrema_data, timeframe)
        
        if 'error' in harmonics_data:
            return {"error": harmonics_data['error']}
        
        return harmonics_data
        
    except Exception as e:
        print(f"Error analyzing Pythagorean Harmonics: {e}")
        return {"error": str(e)}

# ------------------ Fixed calculate_thresholds Function ------------------

def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3, range_distance=0.05):
    """
    Calculate thresholds based on min and max percentages using argmin and argmax.
    Fixed to properly use argmin and argmax for threshold calculation.
    """
    # Convert close_prices to numpy array
    close_prices = np.array(close_prices)
    
    # Get min/max close using argmin and argmax
    min_close_idx = np.argmin(close_prices)
    max_close_idx = np.argmax(close_prices)
    min_close = close_prices[min_close_idx]
    max_close = close_prices[max_close_idx]
    
    # Calculate momentum
    momentum = talib.MOM(close_prices, timeperiod=period)
    
    # Get min/max momentum using argmin and argmax
    min_momentum_idx = np.argmin(momentum)
    max_momentum_idx = np.argmax(momentum)
    min_momentum = momentum[min_momentum_idx]
    max_momentum = momentum[max_momentum_idx]
    
    # Calculate custom percentages 
    min_percentage_custom = minimum_percentage / 100  
    max_percentage_custom = maximum_percentage / 100

    # Calculate thresholds based on argmin and argmax values
    min_threshold = min_close - (max_close - min_close) * min_percentage_custom
    max_threshold = max_close + (max_close - min_close) * max_percentage_custom

    # Ensure thresholds are relative to current price
    current_price = close_prices[-1]
    min_threshold = min(min_threshold, current_price)
    max_threshold = max(max_threshold, current_price)

    # Calculate range of prices within a certain distance from the current close price
    range_price = np.linspace(current_price * (1 - range_distance), current_price * (1 + range_distance), num=50)

    # Filter close prices
    with np.errstate(invalid='ignore'):
        filtered_close = np.where(close_prices < min_threshold, min_threshold, close_prices)      
        filtered_close = np.where(filtered_close > max_threshold, max_threshold, filtered_close)
        
    # Calculate avg    
    avg_mtf = np.nanmean(filtered_close)

    # Get current momentum       
    current_momentum = momentum[-1]

    # Calculate % to min/max momentum    
    with np.errstate(invalid='ignore', divide='ignore'):
        percent_to_min_momentum = ((max_momentum - current_momentum) /   
                                   (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan               

        percent_to_max_momentum = ((current_momentum - min_momentum) / 
                                   (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan
 
    # Calculate combined percentages              
    percent_to_min_combined = (minimum_percentage + percent_to_min_momentum) / 2         
    percent_to_max_combined = (maximum_percentage + percent_to_max_momentum) / 2
      
    # Combined momentum signal     
    momentum_signal = percent_to_max_combined - percent_to_min_combined

    return min_threshold, max_threshold, avg_mtf, momentum_signal, range_price

# ------------------ New Function for Threshold Analysis ------------------

def analyze_thresholds(client, symbol, timeframe='1m', lookback=500):
    """
    Analyze thresholds for the specified timeframe.
    Returns min, max, and middle thresholds based on argmin and argmax.
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
                return {"error": "Insufficient data for 15s threshold analysis"}
                
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
            return {"error": "Insufficient data for threshold analysis"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime in GMT+2 timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Get close prices
        close_prices = df['close'].values.astype(float)
        
        # Find argmin and argmax
        min_close_idx = np.argmin(close_prices)
        max_close_idx = np.argmax(close_prices)
        min_close = close_prices[min_close_idx]
        max_close = close_prices[max_close_idx]
        
        # Calculate middle threshold
        middle_threshold = (min_close + max_close) / 2
        
        # Get current close price
        current_close = close_prices[-1]
        
        # Check if current close is below middle threshold
        close_below_middle = current_close < middle_threshold
        
        # Check if argmin is more recent than argmax
        argmin_more_recent = min_close_idx > max_close_idx
        
        return {
            "timeframe": timeframe,
            "min_threshold": min_close,
            "max_threshold": max_close,
            "middle_threshold": middle_threshold,
            "current_close": current_close,
            "close_below_middle": close_below_middle,
            "argmin_more_recent": argmin_more_recent,
            "min_close_idx": min_close_idx,
            "max_close_idx": max_close_idx
        }
        
    except Exception as e:
        print(f"Error analyzing thresholds: {e}")
        return {"error": str(e)}

# ------------------ Phi-Stoch Extreme Reversion Strategy Functions ------------------

def calculate_stochastic(df, k_period=13, d_period=3):
    """Calculate Stochastic oscillator using TA-Lib with error handling"""
    try:
        if df is None or len(df) < k_period:
            return None
            
        # Clean data
        df_clean = clean_ohlc_data(df.copy())
        high_prices = validate_and_clean_data(df_clean['high'].values.astype(float))
        low_prices = validate_and_clean_data(df_clean['low'].values.astype(float))
        close_prices = validate_and_clean_data(df_clean['close'].values.astype(float))
        
        if high_prices is None or low_prices is None or close_prices is None:
            return None
            
        # Manual calculation to avoid TA-Lib issues
        k_percent = np.zeros(len(close_prices))
        
        for i in range(k_period - 1, len(close_prices)):
            highest_high = np.max(high_prices[i - k_period + 1:i + 1])
            lowest_low = np.min(low_prices[i - k_period + 1:i + 1])
            
            if highest_high - lowest_low > 0:
                k_percent[i] = 100 * (close_prices[i] - lowest_low) / (highest_high - lowest_low)
            else:
                k_percent[i] = 50  # Default value
        
        # Smooth K to get D
        d_percent = np.convolve(k_percent, np.ones(d_period)/d_period, mode='same')
        
        # Clean results
        k_percent = validate_and_clean_data(k_percent, default_value=50.0)
        d_percent = validate_and_clean_data(d_percent, default_value=50.0)
        
        if k_percent is None or d_percent is None:
            return None
            
        df_result = df.copy()
        df_result['STOCH_K'] = k_percent
        df_result['STOCH_D'] = d_percent
        
        return df_result
    except Exception as e:
        print(f"Error calculating Stochastic: {e}")
        return None

def calculate_obv(df, ema_length=20, norm_length=100):
    """Calculate On-Balance Volume (OBV) oscillator using TA-Lib"""
    try:
        if df is None or len(df) < 2:
            return None
            
        # Clean data
        df_clean = clean_ohlc_data(df.copy())
        close_prices = validate_and_clean_data(df_clean['close'].values.astype(float))
        volumes = validate_and_clean_data(df_clean['volume'].values.astype(float))
        
        if close_prices is None or volumes is None:
            return None
            
        # Calculate OBV
        obv = np.zeros(len(close_prices))
        obv[0] = volumes[0]
        
        for i in range(1, len(close_prices)):
            if close_prices[i] > close_prices[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif close_prices[i] < close_prices[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]
        
        # Calculate EMA of OBV
        if TALIB_AVAILABLE:
            try:
                obv_ema = talib.EMA(obv, timeperiod=ema_length)
            except:
                # Manual EMA calculation
                obv_ema = np.zeros_like(obv)
                multiplier = 2 / (ema_length + 1)
                obv_ema[0] = obv[0]
                
                for i in range(1, len(obv)):
                    obv_ema[i] = (obv[i] * multiplier) + (obv_ema[i-1] * (1 - multiplier))
        else:
            # Manual EMA calculation
            obv_ema = np.zeros_like(obv)
            multiplier = 2 / (ema_length + 1)
            obv_ema[0] = obv[0]
            
            for i in range(1, len(obv)):
                obv_ema[i] = (obv[i] * multiplier) + (obv_ema[i-1] * (1 - multiplier))
        
        # Calculate oscillator (OBV - EMA)
        obv_osc = obv - obv_ema
        
        # Normalize
        osc_range = np.zeros(len(obv_osc))
        for i in range(norm_length - 1, len(obv_osc)):
            osc_range[i] = np.max(np.abs(obv_osc[i - norm_length + 1:i + 1]))
        
        # Calculate percentage
        osc_pct = np.zeros(len(obv_osc))
        for i in range(len(osc_pct)):
            if osc_range[i] > 0:
                osc_pct[i] = (obv_osc[i] / osc_range[i]) * 100
        
        # Clean results
        obv_osc = validate_and_clean_data(obv_osc, default_value=0.0)
        osc_pct = validate_and_clean_data(osc_pct, default_value=0.0)
        
        if obv_osc is None or osc_pct is None:
            return None
            
        df_result = df.copy()
        df_result['OBV'] = obv
        df_result['OBV_EMA'] = obv_ema
        df_result['OBV_OSC'] = obv_osc
        df_result['OBV_OSC_PCT'] = osc_pct
        
        return df_result
    except Exception as e:
        print(f"Error calculating OBV: {e}")
        return None

def calculate_volume_percentages(df, lookback=100):
    """Calculate volume percentages and bullish dominance"""
    try:
        if df is None or len(df) < lookback:
            return None
            
        # Clean data
        df_clean = clean_ohlc_data(df.copy())
        close_prices = validate_and_clean_data(df_clean['close'].values.astype(float))
        volumes = validate_and_clean_data(df_clean['volume'].values.astype(float))
        
        if close_prices is None or volumes is None:
            return None
            
        # Determine bullish vs bearish volume
        bullish_volume = np.zeros(len(close_prices))
        bearish_volume = np.zeros(len(close_prices))
        
        for i in range(1, len(close_prices)):
            if close_prices[i] > close_prices[i-1]:  # Bullish candle
                bullish_volume[i] = volumes[i]
            elif close_prices[i] < close_prices[i-1]:  # Bearish candle
                bearish_volume[i] = volumes[i]
            else:  # Doji - split volume
                bullish_volume[i] = volumes[i] / 2
                bearish_volume[i] = volumes[i] / 2
        
        # Calculate total volume over lookback period
        total_volume = np.sum(volumes[-lookback:])
        total_bullish = np.sum(bullish_volume[-lookback:])
        total_bearish = np.sum(bearish_volume[-lookback:])
        
        # Calculate percentages
        if total_volume > 0:
            bullish_pct = (total_bullish / total_volume) * 100
            bearish_pct = (total_bearish / total_volume) * 100
        else:
            bullish_pct = 50.0
            bearish_pct = 50.0
        
        # Determine dominance
        if bullish_pct > bearish_pct:
            dominance = "Bullish"
            dominance_pct = bullish_pct
        elif bearish_pct > bullish_pct:
            dominance = "Bearish"
            dominance_pct = bearish_pct
        else:
            dominance = "Neutral"
            dominance_pct = 50.0
        
        return {
            "total_volume": total_volume,
            "bullish_volume": total_bullish,
            "bearish_volume": total_bearish,
            "bullish_pct": bullish_pct,
            "bearish_pct": bearish_pct,
            "dominance": dominance,
            "dominance_pct": dominance_pct
        }
    except Exception as e:
        print(f"Error calculating volume percentages: {e}")
        return None

def analyze_sma20_extrema(client, symbol, timeframe='1m', lookback=500):
    """Analyze SMA20 extrema for the specified timeframe."""
    try:
        # Get data based on timeframe
        if timeframe == '1m':
            klines = client.get_klines(symbol=symbol, interval='1m', limit=lookback)
        else:  # 15s timeframe
            # Get 1m data and convert to 15s
            min_1m_candles = max(100, lookback // 4 + 20)
            klines_1m = client.get_klines(symbol=symbol, interval='1m', limit=min_1m_candles)
            
            if not klines_1m or len(klines_1m) < 50:
                return {"error": "Insufficient data for 15s SMA20 analysis"}
                
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
        
        if not klines or len(klines) < 50:
            return {"error": "Insufficient data for SMA20 analysis"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime in GMT+2 timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Calculate SMA20
        sma20 = ta_ema(df['close'].values, 20)
        
        # Find the index of the lowest and highest SMA20 values
        lowest_sma20_idx = np.argmin(sma20)
        highest_sma20_idx = np.argmax(sma20)
        lowest_sma20 = sma20[lowest_sma20_idx]
        highest_sma20 = sma20[highest_sma20_idx]
        
        # Get timestamps for these extrema
        lowest_sma20_time = df.iloc[lowest_sma20_idx]['timestamp']
        highest_sma20_time = df.iloc[highest_sma20_idx]['timestamp']
        
        # Check if lowest SMA20 is more recent than highest SMA20
        lowest_more_recent = lowest_sma20_idx > highest_sma20_idx
        
        # Get current close and current SMA20
        current_close = float(df['close'].iloc[-1])
        current_sma20 = float(sma20[-1])
        
        # Check if current close is below SMA20
        close_below_sma20 = current_close < current_sma20
        
        return {
            "timeframe": timeframe,
            "current_close": current_close,
            "current_sma20": current_sma20,
            "lowest_sma20": lowest_sma20,
            "highest_sma20": highest_sma20,
            "lowest_sma20_time": lowest_sma20_time,
            "highest_sma20_time": highest_sma20_time,
            "lowest_more_recent": lowest_more_recent,
            "close_below_sma20": close_below_sma20,
            "lowest_sma20_idx": lowest_sma20_idx,
            "highest_sma20_idx": highest_sma20_idx
        }
        
    except Exception as e:
        print(f"Error analyzing SMA20 extrema: {e}")
        return {"error": str(e)}

def analyze_sma200_extrema(client, symbol, timeframe='1m', lookback=500):
    """Analyze SMA200 extrema for the specified timeframe with actual SMA200 values."""
    try:
        # Get data based on timeframe
        if timeframe == '1m':
            klines = client.get_klines(symbol=symbol, interval='1m', limit=lookback)
        else:  # 15s timeframe
            # Get 1m data and convert to 15s
            min_1m_candles = max(100, lookback // 4 + 20)
            klines_1m = client.get_klines(symbol=symbol, interval='1m', limit=min_1m_candles)
            
            if not klines_1m or len(klines_1m) < 50:
                return {"error": "Insufficient data for 15s SMA200 analysis"}
                
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
        
        if not klines or len(klines) < 200:
            return {"error": "Insufficient data for SMA200 analysis"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime in GMT+2 timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Calculate SMA200 using TA-Lib if available, otherwise manual calculation
        close_prices = df['close'].values.astype(float)
        
        # Ensure we have enough data points
        if len(close_prices) < 200:
            return {"error": f"Insufficient data for SMA200 calculation: got {len(close_prices)}, need 200"}
        
        # Calculate SMA200
        if TALIB_AVAILABLE:
            try:
                sma200 = talib.SMA(close_prices, timeperiod=200)
            except Exception as e:
                print(f"TA-Lib SMA error: {e}")
                # Fall back to manual calculation
                sma200 = calculate_manual_sma(close_prices, 200)
        else:
            # Manual calculation
            sma200 = calculate_manual_sma(close_prices, 200)
        
        # Clean any NaN values in SMA200
        sma200 = np.nan_to_num(sma200, nan=np.nanmean(sma200))
        
        # Find the index of the lowest and highest SMA200 values
        lowest_sma200_idx = np.nanargmin(sma200)
        highest_sma200_idx = np.nanargmax(sma200)
        lowest_sma200 = sma200[lowest_sma200_idx]
        highest_sma200 = sma200[highest_sma200_idx]
        
        # Get timestamps for these extrema
        lowest_sma200_time = df.iloc[lowest_sma200_idx]['timestamp']
        highest_sma200_time = df.iloc[highest_sma200_idx]['timestamp']
        
        # Get the corresponding prices for these SMA200 values
        lowest_sma200_price = float(df.iloc[lowest_sma200_idx]['close'])
        highest_sma200_price = float(df.iloc[highest_sma200_idx]['close'])
        
        # Check if lowest SMA200 is more recent than highest SMA200
        lowest_more_recent = lowest_sma200_idx > highest_sma200_idx
        
        # Get current close and current SMA200
        current_close = float(df['close'].iloc[-1])
        current_sma200 = float(sma200[-1])
        
        # Check if current close is below SMA200
        close_below_sma200 = current_close < current_sma200
        
        return {
            "timeframe": timeframe,
            "current_close": current_close,
            "current_sma200": current_sma200,
            "lowest_sma200": lowest_sma200,
            "highest_sma200": highest_sma200,
            "lowest_sma200_price": lowest_sma200_price,
            "highest_sma200_price": highest_sma200_price,
            "lowest_sma200_time": lowest_sma200_time,
            "highest_sma200_time": highest_sma200_time,
            "lowest_more_recent": lowest_more_recent,
            "close_below_sma200": close_below_sma200,
            "lowest_sma200_idx": lowest_sma200_idx,
            "highest_sma200_idx": highest_sma200_idx
        }
        
    except Exception as e:
        print(f"Error analyzing SMA200 extrema: {e}")
        return {"error": str(e)}

def calculate_manual_sma(data, period):
    """Calculate Simple Moving Average (SMA) manually"""
    try:
        if len(data) < period:
            return np.full(len(data), np.nan)
        
        sma = np.zeros(len(data))
        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1:i + 1])
        
        # Fill the first period-1 values with the first calculated SMA
        if period > 0:
            sma[:period - 1] = sma[period - 1]
        
        return sma
    except Exception as e:
        print(f"Error calculating manual SMA: {e}")
        return np.full(len(data), np.nan)

def ta_ema(data, period):
    """Calculate Exponential Moving Average (EMA)"""
    try:
        if TALIB_AVAILABLE:
            return talib.EMA(data, timeperiod=period)
        else:
            # Manual EMA calculation
            ema = np.zeros_like(data)
            multiplier = 2 / (period + 1)
            ema[0] = data[0]
            
            for i in range(1, len(data)):
                ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
            
            return ema
    except Exception as e:
        print(f"Error calculating EMA: {e}")
        return np.zeros_like(data)

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
        # Try to use Gunicorn if available
        try:
            import gunicorn.app.base
            
            class StandaloneApplication(gunicorn.app.base.BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super(StandaloneApplication, self).__init__()
                
                def load_config(self):
                    config = {key: value for key, value in self.options.items()
                             if key in self.cfg.settings and value is not None}
                    for key, value in config.items():
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
                
                import logging
                log = logging.getLogger('werkzeug')
                log.setLevel(logging.ERROR)
                
                app.run(host=WEBHOOK_HOST, port=WEBHOOK_PORT, debug=False, use_reloader=False)
                
    except Exception as e:
        print(f"Error starting webhook server: {e}")
        app.run(host=WEBHOOK_HOST, port=WEBHOOK_PORT, debug=False, use_reloader=False)

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
            trade_info['target_price'] = entry_price * (1 + (PROFIT_TARGET_PERCENT + TOTAL_FEE_PERCENT) / 100)
            trade_info['target_diff'] = trade_info['target_price'] - current_price
            trade_info['target_diff_pct'] = (trade_info['target_diff'] / current_price) * 100
            
            # Calculate actual profit after fees
            trade_info['actual_profit_pct'] = ((current_price - entry_price) / entry_price) * 100 - TOTAL_FEE_PERCENT
        
        print(f"Resumed active trade: Entry at {entry_price:.25f}, Quantity: {quantity:.25f}")
        return True
    except Exception as e:
        print(f"Error resuming active trade: {e}")
        return False

# ------------------ Enhanced Data Cleaning Functions ------------------

def clean_ohlc_data(df):
    """
    Clean OHLC data of NaN and 0 values before ANY analysis.
    This is primary cleaning function that ensures only valid data is passed to any module.
    """
    if df is None or df.empty:
        return None
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Convert all OHLCV columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Replace NaN values with forward fill, then backward fill
    df_clean = df_clean.ffill().bfill()
    
    # Replace any remaining NaN with 0
    df_clean = df_clean.fillna(0)
    
    # Check for 0 values and replace with rolling average
    for col in ['open', 'high', 'low', 'close']:
        if col in df_clean.columns:
            zero_mask = df_clean[col] == 0
            if zero_mask.any():
                # Calculate rolling average (window=5) for 0 values
                rolling_avg = df_clean[col].rolling(window=5, min_periods=1).mean()
                df_clean.loc[zero_mask, col] = rolling_avg[zero_mask]
                
                # If still 0, use overall mean
                still_zero = df_clean[col] == 0
                if still_zero.any():
                    col_mean = df_clean[col][~zero_mask].mean() if (~zero_mask).any() else 1.0
                    df_clean.loc[still_zero, col] = col_mean
    
    # Handle volume separately - use exponential smoothing for 0 values
    if 'volume' in df_clean.columns:
        volume_zero_mask = df_clean['volume'] == 0
        if volume_zero_mask.any():
            # Use exponential smoothing
            df_clean['volume'] = df_clean['volume'].replace(0, np.nan)
            df_clean['volume'] = df_clean['volume'].ewm(span=3, adjust=False).mean().bfill().ffill()
            df_clean['volume'] = df_clean['volume'].fillna(1.0)  # Minimum volume
    
    # Final validation - ensure no NaN or 0 values remain
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df_clean.columns:
            # Check for NaN
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean() if df_clean[col].mean() > 0 else 1.0)
            
            # Check for 0 or negative values in price columns
            if col in ['open', 'high', 'low', 'close']:
                zero_or_neg = (df_clean[col] <= 0)
                if zero_or_neg.any():
                    # Use previous valid value
                    prev_valid = df_clean[col].shift(1)
                    df_clean.loc[zero_or_neg, col] = prev_valid[zero_or_neg]
                    
                    # If still 0, use minimum positive value
                    still_bad = (df_clean[col] <= 0)
                    if still_bad.any():
                        min_positive = df_clean[col][df_clean[col] > 0].min() if (df_clean[col] > 0).any() else 1.0
                        df_clean.loc[still_bad, col] = min_positive
    
    # Ensure all values are finite
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df_clean.columns:
            df_clean[col] = np.where(
                np.isfinite(df_clean[col]), 
                df_clean[col], 
                df_clean[col].mean() if np.isfinite(df_clean[col].mean()) else 1.0
            )
    
    return df_clean

def validate_and_clean_data(data_array, min_length=20, default_value=1.0):
    """
    Validate and clean numpy array data.
    Returns cleaned array or None if data is insufficient.
    """
    if data_array is None or len(data_array) < min_length:
        return None
    
    # Convert to numpy array if not already
    data = np.array(data_array, dtype=np.float64)
    
    # Check for NaN or infinite values
    if np.any(~np.isfinite(data)):
        # Replace NaN and infinite values with median
        median_val = np.nanmedian(data)
        if not np.isfinite(median_val):
            median_val = default_value
        data = np.where(np.isfinite(data), data, median_val)
    
    # Check for 0 or negative values
    if np.any(data <= 0):
        # Replace with median of positive values
        positive_vals = data[data > 0]
        if len(positive_vals) > 0:
            median_positive = np.median(positive_vals)
            data = np.where(data > 0, data, median_positive)
        else:
            data = np.where(data > 0, data, default_value)
    
    # Ensure data has variation (not all same values)
    if np.std(data) == 0:
        # Add small random variation
        data = data * (1 + np.random.normal(0, 0.001, len(data)))
    
    return data

def create_15sec_timeframe(df_1m):
    """Create artificial 15-second timeframe from 1-minute data with realistic price movements"""
    if df_1m is None or df_1m.empty:
        return pd.DataFrame()
    
    # First clean the 1-minute data
    df_1m_clean = clean_ohlc_data(df_1m)
    
    # Create empty DataFrame for 15-second data
    df_15sec = pd.DataFrame()
    
    # For each 1-minute candle, create 4 15-second candles
    for idx, row in df_1m_clean.iterrows():
        # Calculate price changes within the 1-minute candle
        open_price = float(row['open'])
        high_price = float(row['high'])
        low_price = float(row['low'])
        close_price = float(row['close'])
        volume = float(row['volume'])
        
        # Generate 4 15-second candles with realistic price movement
        # Use a random walk with boundaries
        prices = [open_price]
        
        for i in range(1, 4):
            # Random walk with drift towards close
            drift = (close_price - prices[-1]) / (4 - i)
            volatility = max(0.0001, (high_price - low_price) / 4)
            
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
    """Calculate RSI indicator using TA-Lib with cleaned data."""
    try:
        if df is None or len(df) < period + 1:
            return None
        
        # Clean data before TA-Lib
        df_clean = clean_ohlc_data(df.copy())
        close_prices = validate_and_clean_data(df_clean['close'].values.astype(float))
        
        if close_prices is None or len(close_prices) < period + 1:
            return None
        
        if TALIB_AVAILABLE:
            rsi = talib.RSI(close_prices, timeperiod=period)
        else:
            # Fallback to manual calculation
            delta = np.diff(close_prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = np.zeros_like(close_prices)
            avg_loss = np.zeros_like(close_prices)
            
            # First average
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            # Subsequent averages
            for i in range(period + 1, len(close_prices)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
            
            # Calculate RSI
            rs = avg_gain / np.maximum(avg_loss, 1e-10)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
        
        df_result = df.copy()
        df_result[f'RSI_{period}'] = rsi
        return df_result
    except Exception as e:
        print(f"calculate_rsi error: {e}")
        return None

# ------------------ DMI Analysis Function ------------------

def calculate_dmi(df, period=14):
    """Calculate Directional Movement Index (DMI) for trend analysis."""
    try:
        if df is None or len(df) < period * 2:
            return None
        
        # Clean data before TA-Lib
        df_clean = clean_ohlc_data(df.copy())
        high_prices = validate_and_clean_data(df_clean['high'].values.astype(float))
        low_prices = validate_and_clean_data(df_clean['low'].values.astype(float))
        close_prices = validate_and_clean_data(df_clean['close'].values.astype(float))
        
        if high_prices is None or low_prices is None or close_prices is None:
            return None
        
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
        
        # Clean results
        plus_di = validate_and_clean_data(plus_di, default_value=0.0)
        minus_di = validate_and_clean_data(minus_di, default_value=0.0)
        adx = validate_and_clean_data(adx, default_value=0.0)
        
        if plus_di is None or minus_di is None or adx is None:
            return None
        
        df_result = df.copy()
        df_result['PLUS_DI'] = plus_di
        df_result['MINUS_DI'] = minus_di
        df_result['ADX'] = adx
        
        # Calculate DMI (PLUS_DI - MINUS_DI)
        df_result['DMI'] = plus_di - minus_di
        
        return df_result
    except Exception as e:
        print(f"calculate_dmi error: {e}")
        return None

# ------------------ Momentum Analysis Function ------------------

def calculate_momentum(df, period=10):
    """Calculate Momentum indicator using TA-Lib with cleaned data."""
    try:
        if df is None or len(df) < period + 1:
            return None
        
        # Clean data before TA-Lib
        df_clean = clean_ohlc_data(df.copy())
        close_prices = validate_and_clean_data(df_clean['close'].values.astype(float))
        
        if close_prices is None or len(close_prices) < period + 1:
            return None
        
        if TALIB_AVAILABLE:
            momentum = talib.MOM(close_prices, timeperiod=period)
        else:
            # Fallback to manual calculation
            momentum = np.zeros(len(close_prices))
            for i in range(period, len(close_prices)):
                momentum[i] = close_prices[i] - close_prices[i - period]
        
        momentum = validate_and_clean_data(momentum, default_value=0.0)
        
        if momentum is None:
            return None
        
        df_result = df.copy()
        df_result['MOMENTUM'] = momentum
        return df_result
    except Exception as e:
        print(f"calculate_momentum error: {e}")
        return None

# ------------------ Enhanced FFT Analysis with Proper Frequency Gradients ------------------

def improved_fft_forecast(data, forecast_periods=4):
    """Improved FFT analysis with robust frequency filtering and proper data cleaning."""
    try:
        if data is None or len(data) < 10:
            return np.array([data[-1]] * forecast_periods) if data is not None and len(data) > 0 else np.array([1.0] * forecast_periods)
        
        # Clean and validate input data
        data_clean = validate_and_clean_data(data)
        if data_clean is None:
            return np.array([1.0] * forecast_periods)
        
        # Ensure we have enough data
        if len(data_clean) < 20:
            # Pad with last value if needed
            padding = np.full(20 - len(data_clean), data_clean[-1])
            data_clean = np.concatenate([padding, data_clean])
        
        # Detrend the data
        mean_val = np.mean(data_clean)
        detrended = data_clean - mean_val
        
        # Apply FFT
        fft_values = fft(detrended)
        fft_freq = fftfreq(len(detrended))
        
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
        forecast = forecast + mean_val
        
        # Return only the forecast periods
        forecast_result = forecast[-forecast_periods:]
        
        # Ensure forecast is valid
        forecast_result = validate_and_clean_data(forecast_result, default_value=mean_val)
        
        return forecast_result if forecast_result is not None else np.array([mean_val] * forecast_periods)
    except Exception as e:
        print(f"Error in improved FFT forecast: {e}")
        return np.array([data[-1]] * forecast_periods) if data is not None and len(data) > 0 else np.array([1.0] * forecast_periods)

def analyze_frequency_gradient(data, dip_idx, top_idx):
    """
    Analyze frequency gradient from dip to top.
    Returns frequency dominance analysis with proper gradient calculation.
    """
    try:
        if data is None or len(data) < 50:
            return {"positive_dominance_pct": 50.0, "negative_dominance_pct": 50.0, "gradient": 0.0}
        
        # Clean data
        data_clean = validate_and_clean_data(data)
        if data_clean is None:
            return {"positive_dominance_pct": 50.0, "negative_dominance_pct": 50.0, "gradient": 0.0}
        
        # Apply FFT
        fft_values = fft(data_clean)
        fft_freq = fftfreq(len(data_clean))
        
        # Calculate power spectrum
        power = np.abs(fft_values) ** 2
        
        # Separate positive and negative frequencies
        positive_mask = fft_freq > 0
        negative_mask = fft_freq < 0
        
        positive_power = np.sum(power[positive_mask])
        negative_power = np.sum(power[negative_mask])
        total_power = positive_power + negative_power
        
        if total_power > 0:
            positive_dominance = (positive_power / total_power) * 100
            negative_dominance = (negative_power / total_power) * 100
        else:
            positive_dominance = 50.0
            negative_dominance = 50.0
        
        # Calculate frequency gradient from dip to top
        if dip_idx < top_idx:
            # Up cycle: from dip to top
            # Negative frequencies should dominate at dip, positive at top
            segment_length = top_idx - dip_idx
            if segment_length > 10:
                # Analyze frequency evolution
                frequencies = []
                for i in range(dip_idx, top_idx, max(1, segment_length // 10)):
                    if i + 10 < len(data_clean):
                        segment = data_clean[i:i+10]
                        segment_fft = fft(segment - np.mean(segment))
                        segment_power = np.abs(segment_fft) ** 2
                        segment_freq = fftfreq(len(segment))
                        
                        pos_power = np.sum(segment_power[segment_freq > 0])
                        neg_power = np.sum(segment_power[segment_freq < 0])
                        total_segment_power = pos_power + neg_power
                        
                        if total_segment_power > 0:
                            # Calculate positive frequency dominance
                            pos_dominance = pos_power / total_segment_power                            
                            frequencies.append(pos_dominance)
                
                if len(frequencies) > 1:
                    # Calculate gradient of positive frequency dominance
                    gradient = np.polyfit(range(len(frequencies)), frequencies, 1)[0]
                else:
                    gradient = 0.0
            else:
                gradient = 0.0
        else:
            # Down cycle: from top to dip
            segment_length = dip_idx - top_idx
            if segment_length > 10:
                frequencies = []
                for i in range(top_idx, dip_idx, max(1, segment_length // 10)):
                    if i + 10 < len(data_clean):
                        segment = data_clean[i:i+10]
                        segment_fft = fft(segment - np.mean(segment))
                        segment_power = np.abs(segment_fft) ** 2
                        segment_freq = fftfreq(len(segment))
                        
                        pos_power = np.sum(segment_power[segment_freq > 0])
                        neg_power = np.sum(segment_power[segment_freq < 0])
                        total_segment_power = pos_power + neg_power
                        
                        if total_segment_power > 0:
                            # Calculate positive frequency dominance
                            pos_dominance = pos_power / total_segment_power
                            frequencies.append(pos_dominance)
                
                if len(frequencies) > 1:
                    # Calculate gradient of positive frequency dominance
                    gradient = np.polyfit(range(len(frequencies)), frequencies, 1)[0]
                else:
                    gradient = 0.0
            else:
                gradient = 0.0
        
        return {
            "positive_dominance_pct": positive_dominance,
            "negative_dominance_pct": negative_dominance,
            "gradient": gradient,
            "positive_power": float(positive_power),
            "negative_power": float(negative_power),
            "total_power": float(total_power)
        }
    except Exception as e:
        print(f"Error analyzing frequency gradient: {e}")
        return {"positive_dominance_pct": 50.0, "negative_dominance_pct": 50.0, "gradient": 0.0}

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
        cycle_direction = "up" if dip_more_recent else "down"
        
        # Extract price data for FFT analysis
        close_prices = df['close'].values.astype(float)
        current_price = close_prices[-1]
        
        # Analyze frequency gradient from dip to top
        freq_analysis = analyze_frequency_gradient(close_prices, lowest_low_idx, highest_high_idx)
        
        # Use improved FFT for forecasting
        forecast_prices = improved_fft_forecast(close_prices, forecast_periods=4)
        forecast_target = forecast_prices[-1]
        
        # Ensure forecast target is consistent with cycle direction
        if cycle_direction == "up" and forecast_target <= current_price:
            # If cycle is up but forecast is not above current price, adjust it
            forecast_target = current_price * 1.01  # Small upward adjustment
        elif cycle_direction == "down" and forecast_target >= current_price:
            # If cycle is down but forecast is not below current price, adjust it
            forecast_target = current_price * 0.99  # Small downward adjustment
        
        # Also ensure forecast target is realistic
        if forecast_target <= 0 or abs(forecast_target - current_price) > current_price * 0.05:
            # If forecast is unrealistic, use trend-adjusted current price
            if cycle_direction == "up":
                forecast_target = current_price * 1.01  # Small upward adjustment
            else:
                forecast_target = current_price * 0.99  # Small downward adjustment
        
        # Calculate percentage difference to forecast target
        forecast_diff_pct = ((forecast_target - current_price) / current_price) * 100
        
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
                "positive_dominance_pct": freq_analysis['positive_dominance_pct'],
                "negative_dominance_pct": freq_analysis['negative_dominance_pct'],
                "frequency_gradient": freq_analysis['gradient'],
                "positive_power": freq_analysis.get('positive_power', 0),
                "negative_power": freq_analysis.get('negative_power', 0),
                "total_power": freq_analysis.get('total_power', 0)
            }
        }
        
        return results
        
    except Exception as e:
        print(f"Error analyzing FFT cycle: {e}")
        return {"error": str(e)}

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
        
        # Execute the order using the entire balance
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
        
        # Execute the order using the entire balance
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
        # Check if data is not None and is recent (within last 10 seconds)
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
        
        # Calculate target price for 1.25% clean profit after fees (changed from 0.25%)
        # Target price = entry_price * (1 + 1.25% + 0.22% fee) = entry_price * 1.0147
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
    """Perform single iteration analysis with 17 conditions (12 original + 5 new)."""
    global trade_active, trade_info
    
    # Clear screen for fresh iteration
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("="*80)
    print(f"ENHANCED BTCUSDC TRADING BOT - 17 SPECIFIED TRIGGERS")
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
            else:
                print("Failed to resume active trade.")
    
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
        else:
            print("\nTrade still active. Continuing with full analysis...")
    
    if usdc_balance < MIN_TRADE_AMOUNT and not trade_active:
        print(f"!!! INSUFFICIENT USDC BALANCE - MINIMUM REQUIRED: {MIN_TRADE_AMOUNT} !!!")
        return
    
    # Step 1: Analyze thresholds for 15-second timeframe
    print("\n1. Analyzing thresholds for 15-second timeframe...")
    thresholds_15s = analyze_thresholds(client, SYMBOL, timeframe='15s', lookback=500)
    
    if 'error' not in thresholds_15s:
        print(f"\n--- Threshold Analysis: 15-second Timeframe ---")
        print(f"Min Threshold: {thresholds_15s['min_threshold']:.25f}")
        print(f"Max Threshold: {thresholds_15s['max_threshold']:.25f}")
        print(f"Middle Threshold: {thresholds_15s['middle_threshold']:.25f}")
        print(f"Current Close: {thresholds_15s['current_close']:.25f}")
        print(f"Close Below Middle: {thresholds_15s['close_below_middle']}")
        print(f"Argmin More Recent: {thresholds_15s['argmin_more_recent']}")
    else:
        print(f"Error analyzing 15s thresholds: {thresholds_15s['error']}")
    
    # Step 2: Analyze thresholds for 1-minute timeframe
    print("\n2. Analyzing thresholds for 1-minute timeframe...")
    thresholds_1m = analyze_thresholds(client, SYMBOL, timeframe='1m', lookback=500)
    
    if 'error' not in thresholds_1m:
        print(f"\n--- Threshold Analysis: 1-minute Timeframe ---")
        print(f"Min Threshold: {thresholds_1m['min_threshold']:.25f}")
        print(f"Max Threshold: {thresholds_1m['max_threshold']:.25f}")
        print(f"Middle Threshold: {thresholds_1m['middle_threshold']:.25f}")
        print(f"Current Close: {thresholds_1m['current_close']:.25f}")
        print(f"Close Below Middle: {thresholds_1m['close_below_middle']}")
        print(f"Argmin More Recent: {thresholds_1m['argmin_more_recent']}")
    else:
        print(f"Error analyzing 1m thresholds: {thresholds_1m['error']}")
    
    # Step 3: Analyze SMA20 extrema for 15-second timeframe
    print("\n3. Analyzing SMA20 extrema for 15-second timeframe...")
    sma20_15s = analyze_sma20_extrema(client, SYMBOL, timeframe='15s', lookback=500)
    
    if 'error' not in sma20_15s:
        print(f"\n--- SMA20 Extrema Analysis: 15-second Timeframe ---")
        print(f"Current Close: {sma20_15s['current_close']:.25f}")
        print(f"Current SMA20: {sma20_15s['current_sma20']:.25f}")
        print(f"Lowest SMA20: {sma20_15s['lowest_sma20']:.25f}")
        print(f"Highest SMA20: {sma20_15s['highest_sma20']:.25f}")
        print(f"Lowest More Recent: {sma20_15s['lowest_more_recent']}")
        print(f"Close Below SMA20: {sma20_15s['close_below_sma20']}")
    else:
        print(f"Error analyzing 15s SMA20 extrema: {sma20_15s['error']}")
    
    # Step 4: Analyze SMA20 extrema for 1-minute timeframe
    print("\n4. Analyzing SMA20 extrema for 1-minute timeframe...")
    sma20_1m = analyze_sma20_extrema(client, SYMBOL, timeframe='1m', lookback=500)
    
    if 'error' not in sma20_1m:
        print(f"\n--- SMA20 Extrema Analysis: 1-minute Timeframe ---")
        print(f"Current Close: {sma20_1m['current_close']:.25f}")
        print(f"Current SMA20: {sma20_1m['current_sma20']:.25f}")
        print(f"Lowest SMA20: {sma20_1m['lowest_sma20']:.25f}")
        print(f"Highest SMA20: {sma20_1m['highest_sma20']:.25f}")
        print(f"Lowest More Recent: {sma20_1m['lowest_more_recent']}")
        print(f"Close Below SMA20: {sma20_1m['close_below_sma20']}")
    else:
        print(f"Error analyzing 1m SMA20 extrema: {sma20_1m['error']}")
    
    # Step 5: Analyze Pythagorean Harmonics for 1-minute timeframe
    print("\n5. Analyzing Pythagorean Harmonics for 1-minute timeframe...")
    pythagorean_1m = analyze_pythagorean_harmonics(client, SYMBOL, timeframe='1m', lookback=500)
    
    if 'error' not in pythagorean_1m:
        print(f"\n--- Pythagorean Harmonics Analysis: 1-minute Timeframe ---")
        print(f"Extrema Type: {pythagorean_1m['extrema_type']}")
        print(f"Current Price: {pythagorean_1m['current_price']:.25f}")
        print(f"Forecast Price: {pythagorean_1m['forecast_price']:.25f}")
        print(f"Forecast Direction: {pythagorean_1m['forecast_direction']}")
        print(f"Forecast Difference: {pythagorean_1m['forecast_diff_pct']:.25f}%")
        print(f"Confidence: {pythagorean_1m['confidence']:.25f}")
        
        # Display Pythagorean values
        pythagorean_values = pythagorean_1m.get('pythagorean_values', {})
        print(f"\nPythagorean Values:")
        print(f"  a (Price Movement): {pythagorean_values.get('a', 0):.25f}")
        print(f"  b (Time Movement): {pythagorean_values.get('b', 0):.25f}")
        print(f"  c (Resultant Vector): {pythagorean_values.get('c', 0):.25f}")
        
        closest_triple = pythagorean_values.get('closest_triple', None)
        if closest_triple:
            # Fixed: Convert numpy float64 values to regular floats for clean display
            a_val = float(closest_triple[0])
            b_val = float(closest_triple[1])
            c_val = float(closest_triple[2])
            print(f"  Closest Triple: ({a_val:.25f}, {b_val:.25f}, {c_val:.25f})")
    else:
        print(f"Error analyzing 1m Pythagorean Harmonics: {pythagorean_1m['error']}")
    
    # Step 6: Analyze Pythagorean Harmonics for 15-second timeframe
    print("\n6. Analyzing Pythagorean Harmonics for 15-second timeframe...")
    pythagorean_15s = analyze_pythagorean_harmonics(client, SYMBOL, timeframe='15s', lookback=500)
    
    if 'error' not in pythagorean_15s:
        print(f"\n--- Pythagorean Harmonics Analysis: 15-second Timeframe ---")
        print(f"Extrema Type: {pythagorean_15s['extrema_type']}")
        print(f"Current Price: {pythagorean_15s['current_price']:.25f}")
        print(f"Forecast Price: {pythagorean_15s['forecast_price']:.25f}")
        print(f"Forecast Direction: {pythagorean_15s['forecast_direction']}")
        print(f"Forecast Difference: {pythagorean_15s['forecast_diff_pct']:.25f}%")
        print(f"Confidence: {pythagorean_15s['confidence']:.25f}")
        
        # Display Pythagorean values
        pythagorean_values = pythagorean_15s.get('pythagorean_values', {})
        print(f"\nPythagorean Values:")
        print(f"  a (Price Movement): {pythagorean_values.get('a', 0):.25f}")
        print(f"  b (Time Movement): {pythagorean_values.get('b', 0):.25f}")
        print(f"  c (Resultant Vector): {pythagorean_values.get('c', 0):.25f}")
        
        closest_triple = pythagorean_values.get('closest_triple', None)
        if closest_triple:
            # Fixed: Convert numpy float64 values to regular floats for clean display
            a_val = float(closest_triple[0])
            b_val = float(closest_triple[1])
            c_val = float(closest_triple[2])
            print(f"  Closest Triple: ({a_val:.25f}, {b_val:.25f}, {c_val:.25f})")
    else:
        print(f"Error analyzing 15s Pythagorean Harmonics: {pythagorean_15s['error']}")
    
    # Step 7: Analyze 17 specified conditions
    print("\n7. Analyzing 17 specified trading conditions...")
    
    conditions_met = 0
    total_conditions = 17  # Total number of conditions (updated to 17)
    condition_results = {}  # Store individual condition results
    
    # Condition 1: RSI Oversold Most Recent (1m)
    print("\n--- Condition 1: RSI Oversold Most Recent (1m) ---")
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
    
    # Condition 2: RSI Oversold Most Recent (15s)
    print("\n--- Condition 2: RSI Oversold Most Recent (15s) ---")
    rsi_15s_oversold, rsi_15s_overbought, rsi_15s_value, rsi_15s_details = analyze_rsi_condition(client, SYMBOL, 500, '15s')
    rsi_15s_met = rsi_15s_oversold and not rsi_15s_overbought
    condition_results['RSI Oversold Most Recent (15s)'] = rsi_15s_met
    
    # Print details for this condition
    print(f"\nCurrent RSI: {rsi_15s_value:.25f}")
    print(f"Oversold Most Recent: {rsi_15s_oversold}")
    print(f"Overbought Most Recent: {rsi_15s_overbought}")
    print(f"Condition Met: {rsi_15s_met}")
    
    if rsi_15s_met:
        conditions_met += 1
        print("\nTRUE - RSI Oversold Most Recent (15s) condition MET")
    else:
        print("\nFALSE - RSI Oversold Most Recent (15s) condition NOT met")
    
    # Condition 3: FFT Forecast Up (1m)
    print("\n--- Condition 3: FFT Forecast Up (1m) ---")
    fft_1m = analyze_fft_cycle(client, SYMBOL, timeframe='1m', lookback=500)
    fft_1m_forecast_met = False
    if 'error' not in fft_1m:
        # Check if forecast target is higher than current price
        fft_1m_forecast_met = fft_1m['forecast_target'] > fft_1m['current_price']
        condition_results['FFT Forecast Up (1m)'] = fft_1m_forecast_met
        # Print details for this condition
        print(f"\nCurrent Price: {fft_1m['current_price']:.25f}")
        print(f"Forecast Target: {fft_1m['forecast_target']:.25f}")
        print(f"Forecast Difference: {fft_1m['forecast_diff_pct']:.25f}%")
        print(f"Condition Met: {fft_1m_forecast_met}")
        
        if fft_1m_forecast_met:
            conditions_met += 1
            print("\nTRUE - FFT Forecast Up (1m) condition MET")
        else:
            print("\nFALSE - FFT Forecast Up (1m) condition NOT met")
    else:
        condition_results['FFT Forecast Up (1m)'] = False
        print(f"\nError analyzing 1m FFT cycle: {fft_1m['error']}")
        print("\nFALSE - FFT Forecast Up (1m) condition NOT met")
    
    # Condition 4: FFT Forecast Up (15s)
    print("\n--- Condition 4: FFT Forecast Up (15s) ---")
    fft_15s = analyze_fft_cycle(client, SYMBOL, timeframe='15s', lookback=500)
    fft_15s_forecast_met = False
    if 'error' not in fft_15s:
        # Check if forecast target is higher than current price
        fft_15s_forecast_met = fft_15s['forecast_target'] > fft_15s['current_price']
        condition_results['FFT Forecast Up (15s)'] = fft_15s_forecast_met
        
        # Print details for this condition
        print(f"\nCurrent Price: {fft_15s['current_price']:.25f}")
        print(f"Forecast Target: {fft_15s['forecast_target']:.25f}")
        print(f"Forecast Difference: {fft_15s['forecast_diff_pct']:.25f}%")
        print(f"Condition Met: {fft_15s_forecast_met}")
        
        if fft_15s_forecast_met:
            conditions_met += 1
            print("\nTRUE - FFT Forecast Up (15s) condition MET")
        else:
            print("\nFALSE - FFT Forecast Up (15s) condition NOT met")
    else:
        condition_results['FFT Forecast Up (15s)'] = False
        print(f"\nError analyzing 15s FFT cycle: {fft_15s['error']}")
        print("\nFALSE - FFT Forecast Up (15s) condition NOT met")
    
    # Condition 5: Pythagorean Harmonics Up (1m)
    print("\n--- Condition 5: Pythagorean Harmonics Up (1m) ---")
    pythagorean_1m_met = False
    if 'error' not in pythagorean_1m:
        # Check if forecast direction is up
        pythagorean_1m_met = pythagorean_1m['forecast_direction'] == "up"
        condition_results['Pythagorean Harmonics Up (1m)'] = pythagorean_1m_met
        
        # Print details for this condition
        print(f"\nExtrema Type: {pythagorean_1m['extrema_type']}")
        print(f"Forecast Direction: {pythagorean_1m['forecast_direction']}")
        print(f"Forecast Price: {pythagorean_1m['forecast_price']:.25f}")
        print(f"Forecast Difference: {pythagorean_1m['forecast_diff_pct']:.25f}%")
        print(f"Confidence: {pythagorean_1m['confidence']:.25f}")
        print(f"Condition Met: {pythagorean_1m_met}")
        
        if pythagorean_1m_met:
            conditions_met += 1
            print("\nTRUE - Pythagorean Harmonics Up (1m) condition MET")
        else:
            print("\nFALSE - Pythagorean Harmonics Up (1m) condition NOT met")
    else:
        condition_results['Pythagorean Harmonics Up (1m)'] = False
        print(f"\nError analyzing Pythagorean Harmonics: {pythagorean_1m['error']}")
        print("\nFALSE - Pythagorean Harmonics Up (1m) condition NOT met")
    
    # Condition 6: Pythagorean Harmonics Up (15s)
    print("\n--- Condition 6: Pythagorean Harmonics Up (15s) ---")
    pythagorean_15s_met = False
    if 'error' not in pythagorean_15s:
        # Check if forecast direction is up
        pythagorean_15s_met = pythagorean_15s['forecast_direction'] == "up"
        condition_results['Pythagorean Harmonics Up (15s)'] = pythagorean_15s_met
        
        # Print details for this condition
        print(f"\nExtrema Type: {pythagorean_15s['extrema_type']}")
        print(f"Forecast Direction: {pythagorean_15s['forecast_direction']}")
        print(f"Forecast Price: {pythagorean_15s['forecast_price']:.25f}")
        print(f"Forecast Difference: {pythagorean_15s['forecast_diff_pct']:.25f}%")
        print(f"Confidence: {pythagorean_15s['confidence']:.25f}")
        print(f"Condition Met: {pythagorean_15s_met}")
        
        if pythagorean_15s_met:
            conditions_met += 1
            print("\nTRUE - Pythagorean Harmonics Up (15s) condition MET")
        else:
            print("\nFALSE - Pythagorean Harmonics Up (15s) condition NOT met")
    else:
        condition_results['Pythagorean Harmonics Up (15s)'] = False
        print(f"\nError analyzing Pythagorean Harmonics: {pythagorean_15s['error']}")
        print("\nFALSE - Pythagorean Harmonics Up (15s) condition NOT met")
    
    # Condition 7: Momentum > 0 (1m)
    print("\n--- Condition 7: Momentum > 0 (1m) ---")
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
    
    # Condition 8: Momentum > 0 (15s)
    print("\n--- Condition 8: Momentum > 0 (15s) ---")
    momentum_15s_met, momentum_15s_details = analyze_momentum_condition(client, SYMBOL, 500, '15s')
    condition_results['Momentum > 0 (15s)'] = momentum_15s_met
    
    # Print details for this condition
    print(f"\nLast Momentum: {momentum_15s_details.get('last_momentum', 0):.25f}")
    print(f"Condition Met: {momentum_15s_met}")
    
    if momentum_15s_met:
        conditions_met += 1
        print("\nTRUE - Momentum > 0 (15s) condition MET")
    else:
        print("\nFALSE - Momentum > 0 (15s) condition NOT met")
    
    # Condition 9: Current Close Below Middle Threshold (15s)
    print("\n--- Condition 9: Current Close Below Middle Threshold (15s) ---")
    close_below_middle_15s_met = False
    if 'error' not in thresholds_15s:
        close_below_middle_15s_met = thresholds_15s['close_below_middle']
        condition_results['Close Below Middle (15s)'] = close_below_middle_15s_met
        
        # Print details for this condition
        print(f"\nCurrent Close: {thresholds_15s['current_close']:.25f}")
        print(f"Middle Threshold: {thresholds_15s['middle_threshold']:.25f}")
        print(f"Condition Met: {close_below_middle_15s_met}")
        
        if close_below_middle_15s_met:
            conditions_met += 1
            print("\nTRUE - Current Close Below Middle Threshold (15s) condition MET")
        else:
            print("\nFALSE - Current Close Below Middle Threshold (15s) condition NOT met")
    else:
        condition_results['Close Below Middle (15s)'] = False
        print(f"\nError analyzing thresholds: {thresholds_15s['error']}")
        print("\nFALSE - Current Close Below Middle Threshold (15s) condition NOT met")
    
    # Condition 10: Argmin More Recent than Argmax (15s)
    print("\n--- Condition 10: Argmin More Recent than Argmax (15s) ---")
    argmin_more_recent_15s_met = False
    if 'error' not in thresholds_15s:
        argmin_more_recent_15s_met = thresholds_15s['argmin_more_recent']
        condition_results['Argmin More Recent (15s)'] = argmin_more_recent_15s_met
        
        # Print details for this condition
        print(f"\nMin Close Index: {thresholds_15s['min_close_idx']}")
        print(f"Max Close Index: {thresholds_15s['max_close_idx']}")
        print(f"Argmin More Recent: {argmin_more_recent_15s_met}")
        print(f"Condition Met: {argmin_more_recent_15s_met}")
        
        if argmin_more_recent_15s_met:
            conditions_met += 1
            print("\nTRUE - Argmin More Recent than Argmax (15s) condition MET")
        else:
            print("\nFALSE - Argmin More Recent than Argmax (15s) condition NOT met")
    else:
        condition_results['Argmin More Recent (15s)'] = False
        print(f"\nError analyzing thresholds: {thresholds_15s['error']}")
        print("\nFALSE - Argmin More Recent than Argmax (15s) condition NOT met")
    
    # Condition 11: Current Close Below Middle Threshold (1m)
    print("\n--- Condition 11: Current Close Below Middle Threshold (1m) ---")
    close_below_middle_1m_met = False
    if 'error' not in thresholds_1m:
        close_below_middle_1m_met = thresholds_1m['close_below_middle']
        condition_results['Close Below Middle (1m)'] = close_below_middle_1m_met
        
        # Print details for this condition
        print(f"\nCurrent Close: {thresholds_1m['current_close']:.25f}")
        print(f"Middle Threshold: {thresholds_1m['middle_threshold']:.25f}")
        print(f"Condition Met: {close_below_middle_1m_met}")
        
        if close_below_middle_1m_met:
            conditions_met += 1
            print("\nTRUE - Current Close Below Middle Threshold (1m) condition MET")
        else:
            print("\nFALSE - Current Close Below Middle Threshold (1m) condition NOT met")
    else:
        condition_results['Close Below Middle (1m)'] = False
        print(f"\nError analyzing thresholds: {thresholds_1m['error']}")
        print("\nFALSE - Current Close Below Middle Threshold (1m) condition NOT met")
    
    # Condition 12: Argmin More Recent than Argmax (1m)
    print("\n--- Condition 12: Argmin More Recent than Argmax (1m) ---")
    argmin_more_recent_1m_met = False
    if 'error' not in thresholds_1m:
        argmin_more_recent_1m_met = thresholds_1m['argmin_more_recent']
        condition_results['Argmin More Recent (1m)'] = argmin_more_recent_1m_met
        
        # Print details for this condition
        print(f"\nMin Close Index: {thresholds_1m['min_close_idx']}")
        print(f"Max Close Index: {thresholds_1m['max_close_idx']}")
        print(f"Argmin More Recent: {argmin_more_recent_1m_met}")
        print(f"Condition Met: {argmin_more_recent_1m_met}")
        
        if argmin_more_recent_1m_met:
            conditions_met += 1
            print("\nTRUE - Argmin More Recent than Argmax (1m) condition MET")
        else:
            print("\nFALSE - Argmin More Recent than Argmax (1m) condition NOT met")
    else:
        condition_results['Argmin More Recent (1m)'] = False
        print(f"\nError analyzing thresholds: {thresholds_1m['error']}")
        print("\nFALSE - Argmin More Recent than Argmax (1m) condition NOT met")
    
    # Get volume analysis for 1m timeframe
    volume_1m_analysis = None
    try:
        # Get 1m data for volume analysis
        klines_1m = client.get_klines(symbol=SYMBOL, interval='1m', limit=500)
        if klines_1m:
            df_1m = pd.DataFrame(klines_1m, columns=[
                'timestamp','open','high','low','close','volume','close_time',
                'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume','ignore'])
            
            df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
            df_1m = clean_ohlc_data(df_1m)
            volume_1m_analysis = calculate_volume_percentages(df_1m, 100)
    except Exception as e:
        print(f"Error getting volume analysis: {e}")
    
    # Condition 13: Bullish Volume Dominance (1m)
    print("\n--- Condition 13: Bullish Volume Dominance (1m) ---")
    bullish_volume_1m_met = False
    if volume_1m_analysis:
        bullish_volume_1m_met = volume_1m_analysis['dominance'] == "Bullish" and \
                                volume_1m_analysis['bullish_pct'] >= 50.01
        condition_results['Bullish Volume Dominance (1m)'] = bullish_volume_1m_met
        
        print(f"\nVolume Dominance: {volume_1m_analysis['dominance']}")
        print(f"Bullish Volume %: {volume_1m_analysis['bullish_pct']:.2f}%")
        print(f"Bearish Volume %: {volume_1m_analysis['bearish_pct']:.2f}%")
        print(f"Condition Met: {bullish_volume_1m_met}")
        
        if bullish_volume_1m_met:
            conditions_met += 1
            print("\nTRUE - Bullish Volume Dominance (1m) condition MET")
        else:
            print("\nFALSE - Bullish Volume Dominance (1m) condition NOT met")
    else:
        condition_results['Bullish Volume Dominance (1m)'] = False
        print("\nFALSE - Bullish Volume Dominance (1m) condition NOT met")
    
    # Condition 14: Lowest SMA200 More Recent than Highest SMA200 (1m)
    print("\n--- Condition 14: Lowest SMA200 More Recent than Highest SMA200 (1m) ---")
    sma200_1m = analyze_sma200_extrema(client, SYMBOL, timeframe='1m', lookback=500)
    sma200_lowest_more_recent_1m_met = False
    if 'error' not in sma200_1m:
        sma200_lowest_more_recent_1m_met = sma200_1m['lowest_more_recent']
        condition_results['SMA200 Lowest More Recent (1m)'] = sma200_lowest_more_recent_1m_met
        
        print(f"\nLowest SMA200: {sma200_1m['lowest_sma200']:.25f}")
        print(f"Highest SMA200: {sma200_1m['highest_sma200']:.25f}")
        print(f"Lowest SMA200 Price: {sma200_1m['lowest_sma200_price']:.25f}")
        print(f"Highest SMA200 Price: {sma200_1m['highest_sma200_price']:.25f}")
        print(f"Lowest SMA200 More Recent: {sma200_lowest_more_recent_1m_met}")
        print(f"Condition Met: {sma200_lowest_more_recent_1m_met}")
        
        if sma200_lowest_more_recent_1m_met:
            conditions_met += 1
            print("\nTRUE - Lowest SMA200 More Recent than Highest SMA200 (1m) condition MET")
        else:
            print("\nFALSE - Lowest SMA200 More Recent than Highest SMA200 (1m) condition NOT met")
    else:
        condition_results['SMA200 Lowest More Recent (1m)'] = False
        print(f"\nError analyzing SMA200: {sma200_1m['error']}")
        print("\nFALSE - Lowest SMA200 More Recent than Highest SMA200 (1m) condition NOT met")
    
    # Condition 15: Current Close Below SMA200 (1m)
    print("\n--- Condition 15: Current Close Below SMA200 (1m) ---")
    close_below_sma200_1m_met = False
    if 'error' not in sma200_1m:
        close_below_sma200_1m_met = sma200_1m['close_below_sma200']
        condition_results['Close Below SMA200 (1m)'] = close_below_sma200_1m_met
        
        print(f"\nCurrent Close: {sma200_1m['current_close']:.25f}")
        print(f"Current SMA200: {sma200_1m['current_sma200']:.25f}")
        print(f"Close Below SMA200: {close_below_sma200_1m_met}")
        print(f"Condition Met: {close_below_sma200_1m_met}")
        
        if close_below_sma200_1m_met:
            conditions_met += 1
            print("\nTRUE - Current Close Below SMA200 (1m) condition MET")
        else:
            print("\nFALSE - Current Close Below SMA200 (1m) condition NOT met")
    else:
        condition_results['Close Below SMA200 (1m)'] = False
        print(f"\nError analyzing SMA200: {sma200_1m['error']}")
        print("\nFALSE - Current Close Below SMA200 (1m) condition NOT met")
    
    # Condition 16: Lowest SMA200 More Recent than Highest SMA200 (15s)
    print("\n--- Condition 16: Lowest SMA200 More Recent than Highest SMA200 (15s) ---")
    sma200_15s = analyze_sma200_extrema(client, SYMBOL, timeframe='15s', lookback=500)
    sma200_lowest_more_recent_15s_met = False
    if 'error' not in sma200_15s:
        sma200_lowest_more_recent_15s_met = sma200_15s['lowest_more_recent']
        condition_results['SMA200 Lowest More Recent (15s)'] = sma200_lowest_more_recent_15s_met
        
        print(f"\nLowest SMA200: {sma200_15s['lowest_sma200']:.25f}")
        print(f"Highest SMA200: {sma200_15s['highest_sma200']:.25f}")
        print(f"Lowest SMA200 Price: {sma200_15s['lowest_sma200_price']:.25f}")
        print(f"Highest SMA200 Price: {sma200_15s['highest_sma200_price']:.25f}")
        print(f"Lowest SMA200 More Recent: {sma200_lowest_more_recent_15s_met}")
        print(f"Condition Met: {sma200_lowest_more_recent_15s_met}")
        
        if sma200_lowest_more_recent_15s_met:
            conditions_met += 1
            print("\nTRUE - Lowest SMA200 More Recent than Highest SMA200 (15s) condition MET")
        else:
            print("\nFALSE - Lowest SMA200 More Recent than Highest SMA200 (15s) condition NOT met")
    else:
        condition_results['SMA200 Lowest More Recent (15s)'] = False
        print(f"\nError analyzing SMA200: {sma200_15s['error']}")
        print("\nFALSE - Lowest SMA200 More Recent than Highest SMA200 (15s) condition NOT met")
    
    # Condition 17: Current Close Below SMA200 (15s)
    print("\n--- Condition 17: Current Close Below SMA200 (15s) ---")
    close_below_sma200_15s_met = False
    if 'error' not in sma200_15s:
        close_below_sma200_15s_met = sma200_15s['close_below_sma200']
        condition_results['Close Below SMA200 (15s)'] = close_below_sma200_15s_met
        
        print(f"\nCurrent Close: {sma200_15s['current_close']:.25f}")
        print(f"Current SMA200: {sma200_15s['current_sma200']:.25f}")
        print(f"Close Below SMA200: {close_below_sma200_15s_met}")
        print(f"Condition Met: {close_below_sma200_15s_met}")
        
        if close_below_sma200_15s_met:
            conditions_met += 1
            print("\nTRUE - Current Close Below SMA200 (15s) condition MET")
        else:
            print("\nFALSE - Current Close Below SMA200 (15s) condition NOT met")
    else:
        condition_results['Close Below SMA200 (15s)'] = False
        print(f"\nError analyzing SMA200: {sma200_15s['error']}")
        print("\nFALSE - Current Close Below SMA200 (15s) condition NOT met")
    
    # Step 8: Trading Decision
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
                
                # Calculate target price with fees included (1.25% profit target)
                trade_info['target_price'] = buy_result['price'] * (1 + (PROFIT_TARGET_PERCENT + TOTAL_FEE_PERCENT) / 100)
                trade_info['target_diff'] = trade_info['target_price'] - current_price
                trade_info['target_diff_pct'] = (trade_info['target_diff'] / current_price) * 100
                
                # Calculate actual profit after fees
                trade_info['actual_profit_pct'] = ((current_price - buy_result['price']) / buy_result['price']) * 100 - TOTAL_FEE_PERCENT
        else:
            print(f"\nERROR EXECUTING BUY ORDER: {buy_result['error']}")
    elif trade_active:
        print("\n!!! TRADE ALREADY ACTIVE - NO NEW TRADE EXECUTED !!!")
        print(f"Monitoring existing trade for profit target...")
    else:
        print("\n!!! INSUFFICIENT CONDITIONS MET - NO TRADE EXECUTED !!!")
        print(f"Only {conditions_met}/{CONFIG['min_conditions_met']} conditions met.")
        print("Waiting for next iteration...")
    
    # Step 9: Cleanup for next iteration
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
    
    print("=== ENHANCED BTCUSDC TRADING BOT - 17 SPECIFIED TRIGGERS ===")
    print("Press Ctrl+C to stop monitoring.")
    print("Webhook endpoint: http://localhost:5000/webhook/price")
    print("Health check: http://localhost:5000/health")
    print("\nEach iteration will:")
    print("1. Check for active trade and resume if necessary")
    print("2. Use webhook data for real-time price updates")
    print("3. Analyze thresholds for both 1m and 15s timeframes")
    print("4. Analyze SMA20 extrema for both 1m and 15s timeframes")
    print("5. Analyze Pythagorean Harmonics for both 1m and 15s timeframes")
    print("6. Analyze 17 specified trading conditions:")
    print("   - RSI Oversold Most Recent (1m)")
    print("   - RSI Oversold Most Recent (15s)")
    print("   - FFT Forecast Up (1m)")
    print("   - FFT Forecast Up (15s)")
    print("   - Pythagorean Harmonics Up (1m)")
    print("   - Pythagorean Harmonics Up (15s)")
    print("   - Momentum > 0 (1m)")
    print("   - Momentum > 0 (15s)")
    print("   - Current Close Below Middle Threshold (15s)")
    print("   - Argmin More Recent than Argmax (15s)")
    print("   - Current Close Below Middle Threshold (1m)")
    print("   - Argmin More Recent than Argmax (1m)")
    print("   - Bullish Volume Dominance (1m)")
    print("   - Lowest SMA200 More Recent than Highest SMA200 (1m)")
    print("   - Current Close Below SMA200 (1m)")
    print("   - Lowest SMA200 More Recent than Highest SMA200 (15s)")
    print("   - Current Close Below SMA200 (15s)")
    print("7. Execute trade if ALL conditions are met")
    print("8. Use 100% of USDC balance for entry")
    print("9. Monitor for profit target every 5 seconds")
    print("10. Use 100% of BTC balance for exit")
    print("11. Clean up for next iteration")
    print("\nEnhanced Features:")
    print("- 17 specified triggers (12 original + 5 new)")
    print("- Take profit target changed to 1.25% (from 0.25%)")
    print("- Threshold analysis with argmin/argmax for both timeframes")
    print("- SMA20 extrema analysis for both timeframes")
    print("- Improved Pythagorean Harmonics analysis")
    print("- More realistic 15-second timeframe creation")
    print("- FFT analysis between argmin and argmax for both timeframes")
    print("- Detailed frequency calculations and inverse FFT forecasting")
    print("- Proper data cleaning before ANY analysis")
    print("- Correct frequency dominance calculation from dip to top")
    print("- Uses entire balance for both entry and exit trades")
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