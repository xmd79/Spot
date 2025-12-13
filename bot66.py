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
from datetime import datetime, timezone, timedelta
from scipy.signal import hilbert, argrelextrema, find_peaks
from scipy.fft import fft, fftfreq, ifft
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from decimal import Decimal, getcontext
from binance.client import Client
from binance.exceptions import BinanceAPIException

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
TIMEFRAMES = ['1m', '3m', '5m']

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

# API Rate Limiting
MIN_ITERATION_INTERVAL = 5  # 5 seconds between iterations

# Updated configurable conditions - Now 9 conditions
CONFIG = {
    "conditions": {
        "up_cycle_confirmed_1m": True,  # Up Cycle Confirmed (1m)
        "momentum_positive_1m": True,  # Momentum Positive (1m)
        "volume_bullish_dominance_1m": True,  # Volume Bullish Dominance (1m)
        "fft_forecast_up_1m": True,  # FFT Forecast Up (1m)
        "fft_forecast_up_3m": True,  # FFT Forecast Up (3m)
        "fft_forecast_up_5m": True,  # FFT Forecast Up (5m)
        "argmin_more_recent_1m": True,  # Argmin More Recent (1m)
        "argmin_more_recent_3m": True,  # Argmin More Recent (3m)
        "rsi_oversold_most_recent_5m": True,  # RSI Oversold Most Recent (5m)
    },
    "min_conditions_met": 9  # ALL 9 conditions must be met to trigger a trade
}

# Global stop event
stop_event = threading.Event()

# Trade state variables
trade_active = False
trade_info = {}

def signal_handler(sig, frame):
    print('\nCtrl+C pressed! Shutting down gracefully...')
    stop_event.set()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ------------------ API Time Synchronization Fix ------------------

def get_time_offset(client):
    """Get time offset between local and Binance server time"""
    try:
        server_time = client.get_server_time()
        server_timestamp = server_time['serverTime']
        local_timestamp = int(time.time() * 1000)
        offset = server_timestamp - local_timestamp
        return offset
    except Exception as e:
        print(f"Error getting time offset: {e}")
        return 0

def synchronize_time_with_binance(client):
    """Synchronize local time with Binance server time"""
    try:
        # Get the time offset
        offset = get_time_offset(client)
        print(f"Time offset with Binance server: {offset}ms")
        
        # Set the time offset for the client
        client.session.headers.update({
            'X-MBX-APIKEY': client.API_KEY
        })
        
        # Override the _get_request method to add the timestamp
        original_get_request = client._get_request
        
        def _get_request_with_timestamp(*args, **kwargs):
            # Add timestamp with offset
            kwargs['params'] = kwargs.get('params', {})
            kwargs['params']['timestamp'] = int(time.time() * 1000) + offset
            return original_get_request(*args, **kwargs)
        
        client._get_request = _get_request_with_timestamp
        
        # Override the _post_request method to add the timestamp
        original_post_request = client._post_request
        
        def _post_request_with_timestamp(*args, **kwargs):
            # Add timestamp with offset
            kwargs['params'] = kwargs.get('params', {})
            kwargs['params']['timestamp'] = int(time.time() * 1000) + offset
            return original_post_request(*args, **kwargs)
        
        client._post_request = _post_request_with_timestamp
        
        return True
    except Exception as e:
        print(f"Error synchronizing time with Binance: {e}")
        return False

# ------------------ Enhanced API Call Functions ------------------

def safe_api_call(func, *args, max_retries=3, **kwargs):
    """Safely make an API call with retries and timestamp synchronization"""
    for attempt in range(max_retries):
        try:
            # Add a small random delay to avoid hitting rate limits
            time.sleep(0.1 + np.random.random() * 0.2)
            return func(*args, **kwargs)
        except BinanceAPIException as e:
            if e.code == -1021:  # Timestamp error
                print(f"Timestamp error on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    # Resynchronize time
                    synchronize_time_with_binance(client)
                    time.sleep(1)  # Wait a bit before retrying
                    continue
            raise e
        except Exception as e:
            print(f"API call error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait a bit before retrying
                continue
            raise e
    return None

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

# ------------------ New Function for Threshold Analysis ------------------

def analyze_thresholds(client, symbol, timeframe='1m', lookback=500):
    """
    Analyze thresholds for the specified timeframe.
    Returns min, max, and middle thresholds based on argmin and argmax.
    Ensures proper range calculation for each timeframe.
    """
    try:
        # Get data based on timeframe with proper lookback for each timeframe
        if timeframe == '1m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='1m', limit=lookback)
        elif timeframe == '3m':
            # For 3m, we need to get more data to ensure proper range calculation
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='3m', limit=lookback)
        elif timeframe == '5m':
            # For 5m, we need even more data to ensure proper range calculation
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='5m', limit=lookback)
        
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
        
        # Calculate distance between argmin and argmax (hilo range)
        hilo_range = max_close - min_close
        
        # Determine most recent extrema
        most_recent_extrema_idx = max(min_close_idx, max_close_idx)
        most_recent_extrema_value = close_prices[most_recent_extrema_idx]
        most_recent_extrema_type = "argmin" if most_recent_extrema_idx == min_close_idx else "argmax"
        
        # Calculate distance from threshold to current price based on most recent extrema
        dist_to_current = 0.0
        if most_recent_extrema_type == "argmin":
            # Distance from min threshold to current price
            dist_to_current = current_close - min_close
        else:
            # Distance from max threshold to current price
            dist_to_current = max_close - current_close
        
        # Determine if current distance from threshold is equal or bigger than hilo range
        dist_meets_range = dist_to_current >= hilo_range
        
        # Determine up cycle confirmation
        up_cycle_confirmed = False
        down_cycle_confirmed = False
        
        if dist_meets_range:
            if most_recent_extrema_type == "argmin":
                up_cycle_confirmed = True
            elif most_recent_extrema_type == "argmax":
                down_cycle_confirmed = True
        
        return {
            "timeframe": timeframe,
            "min_threshold": min_close,
            "max_threshold": max_close,
            "middle_threshold": middle_threshold,
            "current_close": current_close,
            "close_below_middle": close_below_middle,
            "argmin_more_recent": argmin_more_recent,
            "min_close_idx": min_close_idx,
            "max_close_idx": max_close_idx,
            "hilo_range": hilo_range,
            "most_recent_extrema_idx": most_recent_extrema_idx,
            "most_recent_extrema_value": most_recent_extrema_value,
            "most_recent_extrema_type": most_recent_extrema_type,
            "dist_to_current": dist_to_current,
            "dist_meets_range": dist_meets_range,
            "up_cycle_confirmed": up_cycle_confirmed,
            "down_cycle_confirmed": down_cycle_confirmed
        }
        
    except Exception as e:
        print(f"Error analyzing thresholds: {e}")
        return {"error": str(e)}

# ------------------ NEW: Local Extrema Analysis Function ------------------

def analyze_local_extrema(client, symbol, timeframe='1m', lookback=500):
    """
    Analyze local extrema based on OHLC data with specific distance calculation rules.
    
    This function:
    1. Identifies the most recent low and high as local dip and local top
    2. Uses argmin and argmax to find the most recent minimum and maximum
    3. Determines which extrema is more recent
    4. Calculates the hilo range (distance between argmin and argmax)
    5. Calculates distance from the most recent extrema to the local reversal:
       - If argmin is more recent than argmax, calculates distance from argmin to most recent low (for up cycle)
       - If argmax is more recent than argmin, calculates distance from argmax to most recent high (for down cycle)
    6. Determines up cycle confirmation based on final rule:
       - If argmin is more recent than argmax, up cycle is confirmed if "Distance from argmin to local dip" is equal or higher than hilo range
       - If argmax is more recent than argmin, up cycle is confirmed if "Distance from argmax to local top" is equal or higher than hilo range
    7. Returns detailed information about extrema and distances for each timeframe
    
    Args:
        client: Binance client
        symbol: Trading symbol (e.g., 'BTCUSDC')
        timeframe: Timeframe for analysis ('1m', '3m', '5m')
        lookback: Number of candles to look back
        
    Returns:
        Dictionary with extrema information, local dip/top, and distance calculations
    """
    try:
        # Get data based on timeframe
        if timeframe == '1m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='1m', limit=lookback)
        elif timeframe == '3m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='3m', limit=lookback)
        elif timeframe == '5m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='5m', limit=lookback)
        
        if not klines or len(klines) < 100:
            return {"error": f"Insufficient data for {timeframe} extrema analysis"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime in GMT+2 timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Get OHLC data as arrays
        open_prices = df['open'].values.astype(float)
        high_prices = df['high'].values.astype(float)
        low_prices = df['low'].values.astype(float)
        close_prices = df['close'].values.astype(float)
        
        # Find argmin and argmax indices for close prices
        argmin_idx = np.argmin(close_prices)
        argmax_idx = np.argmax(close_prices)
        argmin_value = close_prices[argmin_idx]
        argmax_value = close_prices[argmax_idx]
        
        # Calculate hilo range (distance between argmin and argmax)
        hilo_range = argmax_value - argmin_value
        
        # Determine which is more recent
        argmin_more_recent = argmin_idx > argmax_idx
        
        # Find the most recent low and high (local dip and local top)
        # For local dip: find the most recent low that is lower than its neighbors
        # For local top: find the most recent high that is higher than its neighbors
        local_dip_idx = None
        local_top_idx = None
        
        # Find local dips (lows)
        for i in range(len(low_prices) - 2, 1, -1):
            if low_prices[i] < low_prices[i-1] and low_prices[i] < low_prices[i+1]:
                local_dip_idx = i
                break
        
        # Find local tops (highs)
        for i in range(len(high_prices) - 2, 1, -1):
            if high_prices[i] > high_prices[i-1] and high_prices[i] > high_prices[i+1]:
                local_top_idx = i
                break
        
        # If no local dip/top found, use the absolute min/max
        if local_dip_idx is None:
            local_dip_idx = np.argmin(low_prices)
        if local_top_idx is None:
            local_top_idx = np.argmax(high_prices)
        
        local_dip_value = low_prices[local_dip_idx]
        local_top_value = high_prices[local_top_idx]
        
        # Calculate distances based on the rule
        distance = 0.0
        distance_type = ""
        
        if argmin_more_recent:
            # Calculate distance from argmin to local dip (for up cycle)
            distance = local_dip_value - argmin_value
            distance_type = "up_cycle"
        else:
            # Calculate distance from argmax to local top (for down cycle)
            distance = argmax_value - local_top_value
            distance_type = "down_cycle"
        
        # Get timestamps
        argmin_time = df.iloc[argmin_idx]['timestamp']
        argmax_time = df.iloc[argmax_idx]['timestamp']
        local_dip_time = df.iloc[local_dip_idx]['timestamp']
        local_top_time = df.iloc[local_top_idx]['timestamp']
        
        # Get current close price
        current_close = close_prices[-1]
        
        # Determine up cycle confirmation based on the final rule
        up_cycle_confirmed = False
        
        if argmin_more_recent:
            # If argmin is more recent than argmax, up cycle is confirmed if 
            # "Distance from argmin to local dip" is equal or higher than hilo range
            up_cycle_confirmed = distance >= hilo_range
        else:
            # If argmax is more recent than argmin, up cycle is confirmed if 
            # "Distance from argmax to local top" is equal or higher than hilo range
            up_cycle_confirmed = distance >= hilo_range
        
        return {
            "timeframe": timeframe,
            "argmin": {
                "index": int(argmin_idx),
                "value": float(argmin_value),
                "timestamp": argmin_time
            },
            "argmax": {
                "index": int(argmax_idx),
                "value": float(argmax_value),
                "timestamp": argmax_time
            },
            "local_dip": {
                "index": int(local_dip_idx),
                "value": float(local_dip_value),
                "timestamp": local_dip_time
            },
            "local_top": {
                "index": int(local_top_idx),
                "value": float(local_top_value),
                "timestamp": local_top_time
            },
            "most_recent_extrema": "argmin" if argmin_more_recent else "argmax",
            "hilo_range": float(hilo_range),
            "distance": {
                "value": float(distance),
                "type": distance_type,
                "description": f"Distance from {'argmin to local dip' if argmin_more_recent else 'argmax to local top'}"
            },
            "current_close": float(current_close),
            "argmin_more_recent": argmin_more_recent,
            "up_cycle_confirmed": up_cycle_confirmed,
            "distance_meets_hilo_range": distance >= hilo_range
        }
        
    except Exception as e:
        print(f"Error analyzing local extrema for {timeframe}: {e}")
        return {"error": str(e)}

def analyze_all_timeframes_extrema(client, symbol, lookback=500):
    """
    Analyze local extrema for all timeframes and return comprehensive results.
    
    Args:
        client: Binance client
        symbol: Trading symbol (e.g., 'BTCUSDC')
        lookback: Number of candles to look back
        
    Returns:
        Dictionary with extrema analysis for all timeframes
    """
    timeframes = ['1m', '3m', '5m']
    results = {}
    
    for tf in timeframes:
        results[tf] = analyze_local_extrema(client, symbol, tf, lookback)
    
    return results

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

def calculate_momentum(df, period=10):
    """Calculate momentum indicator."""
    try:
        if df is None or len(df) < period + 1:
            return None
        
        # Clean data
        df_clean = clean_ohlc_data(df.copy())
        close_prices = validate_and_clean_data(df_clean['close'].values.astype(float))
        
        if close_prices is None or len(close_prices) < period + 1:
            return None
        
        # Calculate momentum (current price minus price N periods ago)
        momentum = np.zeros(len(close_prices))
        for i in range(period, len(close_prices)):
            momentum[i] = close_prices[i] - close_prices[i - period]
        
        df_result = df.copy()
        df_result[f'Momentum_{period}'] = momentum
        return df_result
    except Exception as e:
        print(f"calculate_momentum error: {e}")
        return None

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

def analyze_volume_dominance(client, symbol, timeframe='1m', lookback=100):
    """
    Analyze volume dominance for the specified timeframe.
    Returns whether bullish volume is greater than bearish volume.
    """
    try:
        # Get data based on timeframe
        if timeframe == '1m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='1m', limit=lookback)
        elif timeframe == '3m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='3m', limit=lookback)
        elif timeframe == '5m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='5m', limit=lookback)
        
        if not klines or len(klines) < 50:
            return {"error": "Insufficient data for volume analysis"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime in GMT+2 timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Calculate volume percentages
        volume_data = calculate_volume_percentages(df, lookback)
        
        if volume_data is None:
            return {"error": "Failed to calculate volume percentages"}
        
        # Determine if bullish volume is greater than bearish volume
        bullish_dominance = volume_data['bullish_pct'] > volume_data['bearish_pct']
        
        return {
            "timeframe": timeframe,
            "bullish_pct": volume_data['bullish_pct'],
            "bearish_pct": volume_data['bearish_pct'],
            "dominance": volume_data['dominance'],
            "bullish_dominance": bullish_dominance
        }
        
    except Exception as e:
        print(f"Error analyzing volume dominance: {e}")
        return {"error": str(e)}

def analyze_sma200_extrema(client, symbol, timeframe='1m', lookback=500):
    """Analyze SMA200 extrema for the specified timeframe with actual SMA200 values."""
    try:
        # Get data based on timeframe
        if timeframe == '1m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='1m', limit=lookback)
        elif timeframe == '3m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='3m', limit=lookback)
        elif timeframe == '5m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='5m', limit=lookback)
        
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

def analyze_rsi_condition(client, symbol, lookback=500, timeframe='1m'):
    """Analyze RSI oversold/overbought most recent condition."""
    try:
        # Get data based on timeframe
        if timeframe == '1m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='1m', limit=lookback)
        elif timeframe == '3m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='3m', limit=lookback)
        elif timeframe == '5m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='5m', limit=lookback)
        
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

def analyze_momentum_condition(client, symbol, lookback=500, timeframe='1m', period=10):
    """Analyze momentum condition."""
    try:
        # Get data based on timeframe
        if timeframe == '1m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='1m', limit=lookback)
        elif timeframe == '3m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='3m', limit=lookback)
        elif timeframe == '5m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='5m', limit=lookback)
        
        if not klines or len(klines) < 50:
            return False, 0.0, {"error": "Insufficient data for momentum analysis"}
            
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'])
        
        # Convert timestamp to datetime in GMT+2 timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
        
        # Clean OHLC data
        df = clean_ohlc_data(df)
        
        # Calculate momentum
        df_momentum = calculate_momentum(df, period)
        if df_momentum is None or f'Momentum_{period}' not in df_momentum.columns:
            return False, 0.0, {"error": "Momentum calculation failed"}
        
        momentum_values = df_momentum[f'Momentum_{period}'].values.astype(float)
        current_momentum = float(momentum_values[-1])
        
        # Check if momentum is positive (changed from negative as requested)
        momentum_positive = current_momentum > 0
        
        details = {
            "timeframe": timeframe,
            "current_momentum": current_momentum,
            "momentum_positive": momentum_positive,
            "period": period
        }
        
        return momentum_positive, current_momentum, details
        
    except Exception as e:
        print(f"analyze_momentum_condition error: {e}")
        return False, 0.0, {"error": str(e)}

def analyze_fft_cycle(client, symbol, timeframe='1m', lookback=500):
    """
    Analyze FFT cycle between argmin and argmax for both 1min and 15sec TF.
    Provides detailed frequency calculations and inverse FFT forecasting.
    """
    try:
        # Get data based on timeframe
        if timeframe == '1m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='1m', limit=lookback)
        elif timeframe == '3m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='3m', limit=lookback)
        elif timeframe == '5m':
            klines = safe_api_call(client.get_klines, symbol=symbol, interval='5m', limit=lookback)
        
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
        
        # Find index of the lowest low and highest high using argmin/argmax
        lowest_low_idx = df['low'].idxmin()
        highest_high_idx = df['high'].idxmax()
        
        # Get values and timestamps
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
        symbol_info = safe_api_call(client.get_symbol_info, symbol)
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
        # Get current price from API
        ticker = safe_api_call(client.get_symbol_ticker, symbol=symbol)
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
        order = safe_api_call(client.order_market_buy,
                             symbol=symbol,
                             quantity=quantity)
        
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
        order = safe_api_call(client.order_market_sell,
                             symbol=symbol,
                             quantity=quantity)
        
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

def get_current_price(client, symbol):
    """Get current price for a symbol using Binance API."""
    try:
        ticker = safe_api_call(client.get_symbol_ticker, symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        print(f"Error getting current price: {e}")
        return None

def get_account_balance(client, asset):
    """Get balance of a specific asset in the account."""
    try:
        account_info = safe_api_call(client.get_account)
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
        current_price = get_current_price(client, SYMBOL)
        if current_price is None:
            return False
        
        entry_price = trade_info['entry_price']
        quantity = trade_info['quantity']
        
        # Calculate target price for 1.25% clean profit after fees
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

def check_for_active_trade(client):
    """
    Check if there's an active trade by examining order history.
    Returns True if an active trade is detected, False otherwise.
    """
    try:
        # Get recent orders
        orders = safe_api_call(client.get_all_orders, symbol=SYMBOL, limit=10)
        
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
        orders = safe_api_call(client.get_all_orders, symbol=SYMBOL, limit=10)
        
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
            trades = safe_api_call(client.get_my_trades, symbol=SYMBOL, limit=10)
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

# ------------------ Main Analysis Function ------------------

def perform_single_iteration_analysis(client):
    """Perform single iteration analysis with all specified conditions."""
    global trade_active, trade_info
    
    # Clear screen for fresh iteration
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("="*80)
    print(f"ENHANCED BTCUSDC TRADING BOT - 9 TRIGGERS")
    print(f"Time: {datetime.now(LOCAL_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')} (GMT+2)")
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
    
    # Step 1: Fetch fresh data for all timeframes
    print("\n1. Fetching fresh data for all timeframes...")
    
    # Fetch data for all timeframes to ensure fresh data
    thresholds_1m = analyze_thresholds(client, SYMBOL, timeframe='1m', lookback=500)
    thresholds_3m = analyze_thresholds(client, SYMBOL, timeframe='3m', lookback=500)
    
    # Fetch volume data for 1m timeframe
    volume_1m = analyze_volume_dominance(client, SYMBOL, timeframe='1m', lookback=100)
    
    # Fetch FFT data for 1m, 3m, and 5m timeframes
    fft_1m = analyze_fft_cycle(client, SYMBOL, timeframe='1m', lookback=500)
    fft_3m = analyze_fft_cycle(client, SYMBOL, timeframe='3m', lookback=500)
    fft_5m = analyze_fft_cycle(client, SYMBOL, timeframe='5m', lookback=500)
    
    # Fetch local extrema data for 1m timeframe
    local_extrema_1m = analyze_local_extrema(client, SYMBOL, timeframe='1m', lookback=500)
    
    # Step 2: Analyze all specified conditions
    print("\n2. Analyzing all 9 specified trading conditions...")
    
    conditions_met = 0
    total_conditions = 9  # Total number of conditions
    condition_results = {}  # Store individual condition results
    
    # Condition 1: Up Cycle Confirmed (1m)
    print("\n--- Condition 1: Up Cycle Confirmed (1m) ---")
    up_cycle_1m_met = False
    if 'error' not in local_extrema_1m:
        up_cycle_1m_met = local_extrema_1m['up_cycle_confirmed']
        condition_results['Up Cycle Confirmed (1m)'] = up_cycle_1m_met
        
        print(f"\nMost Recent Extrema: {local_extrema_1m['most_recent_extrema']}")
        print(f"Argmin Value: {local_extrema_1m['argmin']['value']:.2f}")
        print(f"Argmax Value: {local_extrema_1m['argmax']['value']:.2f}")
        print(f"Hi-Lo Range: {local_extrema_1m['hilo_range']:.2f}")
        print(f"Local Dip Value: {local_extrema_1m['local_dip']['value']:.2f}")
        print(f"Local Top Value: {local_extrema_1m['local_top']['value']:.2f}")
        print(f"Distance: {local_extrema_1m['distance']['value']:.2f} ({local_extrema_1m['distance']['description']})")
        print(f"Distance Meets Hi-Lo Range: {local_extrema_1m['distance_meets_hilo_range']}")
        print(f"Argmin More Recent: {local_extrema_1m['argmin_more_recent']}")
        print(f"Up Cycle Confirmed: {up_cycle_1m_met}")
        print(f"Condition Met: {up_cycle_1m_met}")
        
        if up_cycle_1m_met:
            conditions_met += 1
            print("\nTRUE - Up Cycle Confirmed (1m) condition MET")
        else:
            print("\nFALSE - Up Cycle Confirmed (1m) condition NOT met")
    else:
        condition_results['Up Cycle Confirmed (1m)'] = False
        print(f"\nError analyzing local extrema: {local_extrema_1m['error']}")
        print("\nFALSE - Up Cycle Confirmed (1m) condition NOT met")
    
    # Condition 2: Momentum Positive (1m)
    print("\n--- Condition 2: Momentum Positive (1m) ---")
    momentum_1m_positive, momentum_1m_value, momentum_1m_details = analyze_momentum_condition(client, SYMBOL, 500, '1m')
    condition_results['Momentum Positive (1m)'] = momentum_1m_positive
    
    # Print details for this condition
    print(f"\nCurrent Momentum: {momentum_1m_value:.4f}")
    print(f"Momentum Positive: {momentum_1m_positive}")
    print(f"Condition Met: {momentum_1m_positive}")
    
    if momentum_1m_positive:
        conditions_met += 1
        print("\nTRUE - Momentum Positive (1m) condition MET")
    else:
        print("\nFALSE - Momentum Positive (1m) condition NOT met")
    
    # Condition 3: Volume Bullish Dominance (1m)
    print("\n--- Condition 3: Volume Bullish Dominance (1m) ---")
    volume_1m_met = False
    if 'error' not in volume_1m:
        volume_1m_met = volume_1m['bullish_dominance']
        condition_results['Volume Bullish Dominance (1m)'] = volume_1m_met
        
        print(f"\nBullish Volume %: {volume_1m['bullish_pct']:.2f}%")
        print(f"Bearish Volume %: {volume_1m['bearish_pct']:.2f}%")
        print(f"Dominance: {volume_1m['dominance']}")
        print(f"Bullish Dominance: {volume_1m_met}")
        print(f"Condition Met: {volume_1m_met}")
        
        if volume_1m_met:
            conditions_met += 1
            print("\nTRUE - Volume Bullish Dominance (1m) condition MET")
        else:
            print("\nFALSE - Volume Bullish Dominance (1m) condition NOT met")
    else:
        condition_results['Volume Bullish Dominance (1m)'] = False
        print(f"\nError analyzing volume: {volume_1m['error']}")
        print("\nFALSE - Volume Bullish Dominance (1m) condition NOT met")
    
    # Condition 4: FFT Forecast Up (1m)
    print("\n--- Condition 4: FFT Forecast Up (1m) ---")
    fft_1m_forecast_met = False
    if 'error' not in fft_1m:
        # Check if forecast target is higher than current price
        fft_1m_forecast_met = fft_1m['forecast_target'] > fft_1m['current_price']
        condition_results['FFT Forecast Up (1m)'] = fft_1m_forecast_met
        # Print details for this condition
        print(f"\nCurrent Price: {fft_1m['current_price']:.2f}")
        print(f"Forecast Target: {fft_1m['forecast_target']:.2f}")
        print(f"Forecast Difference: {fft_1m['forecast_diff_pct']:.4f}%")
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
    
    # Condition 5: FFT Forecast Up (3m)
    print("\n--- Condition 5: FFT Forecast Up (3m) ---")
    fft_3m_forecast_met = False
    if 'error' not in fft_3m:
        # Check if forecast target is higher than current price
        fft_3m_forecast_met = fft_3m['forecast_target'] > fft_3m['current_price']
        condition_results['FFT Forecast Up (3m)'] = fft_3m_forecast_met
        # Print details for this condition
        print(f"\nCurrent Price: {fft_3m['current_price']:.2f}")
        print(f"Forecast Target: {fft_3m['forecast_target']:.2f}")
        print(f"Forecast Difference: {fft_3m['forecast_diff_pct']:.4f}%")
        print(f"Condition Met: {fft_3m_forecast_met}")
        
        if fft_3m_forecast_met:
            conditions_met += 1
            print("\nTRUE - FFT Forecast Up (3m) condition MET")
        else:
            print("\nFALSE - FFT Forecast Up (3m) condition NOT met")
    else:
        condition_results['FFT Forecast Up (3m)'] = False
        print(f"\nError analyzing 3m FFT cycle: {fft_3m['error']}")
        print("\nFALSE - FFT Forecast Up (3m) condition NOT met")
    
    # Condition 6: FFT Forecast Up (5m)
    print("\n--- Condition 6: FFT Forecast Up (5m) ---")
    fft_5m_forecast_met = False
    if 'error' not in fft_5m:
        # Check if forecast target is higher than current price
        fft_5m_forecast_met = fft_5m['forecast_target'] > fft_5m['current_price']
        condition_results['FFT Forecast Up (5m)'] = fft_5m_forecast_met
        # Print details for this condition
        print(f"\nCurrent Price: {fft_5m['current_price']:.2f}")
        print(f"Forecast Target: {fft_5m['forecast_target']:.2f}")
        print(f"Forecast Difference: {fft_5m['forecast_diff_pct']:.4f}%")
        print(f"Condition Met: {fft_5m_forecast_met}")
        
        if fft_5m_forecast_met:
            conditions_met += 1
            print("\nTRUE - FFT Forecast Up (5m) condition MET")
        else:
            print("\nFALSE - FFT Forecast Up (5m) condition NOT met")
    else:
        condition_results['FFT Forecast Up (5m)'] = False
        print(f"\nError analyzing 5m FFT cycle: {fft_5m['error']}")
        print("\nFALSE - FFT Forecast Up (5m) condition NOT met")
    
    # Condition 7: Argmin More Recent (1m)
    print("\n--- Condition 7: Argmin More Recent (1m) ---")
    argmin_1m_met = False
    if 'error' not in thresholds_1m:
        argmin_1m_met = thresholds_1m['argmin_more_recent']
        condition_results['Argmin More Recent (1m)'] = argmin_1m_met
        
        print(f"\nMin Close Index: {thresholds_1m['min_close_idx']}")
        print(f"Max Close Index: {thresholds_1m['max_close_idx']}")
        print(f"Argmin More Recent: {argmin_1m_met}")
        print(f"Condition Met: {argmin_1m_met}")
        
        if argmin_1m_met:
            conditions_met += 1
            print("\nTRUE - Argmin More Recent (1m) condition MET")
        else:
            print("\nFALSE - Argmin More Recent (1m) condition NOT met")
    else:
        condition_results['Argmin More Recent (1m)'] = False
        print(f"\nError analyzing thresholds: {thresholds_1m['error']}")
        print("\nFALSE - Argmin More Recent (1m) condition NOT met")
    
    # Condition 8: Argmin More Recent (3m)
    print("\n--- Condition 8: Argmin More Recent (3m) ---")
    argmin_3m_met = False
    if 'error' not in thresholds_3m:
        argmin_3m_met = thresholds_3m['argmin_more_recent']
        condition_results['Argmin More Recent (3m)'] = argmin_3m_met
        
        print(f"\nMin Close Index: {thresholds_3m['min_close_idx']}")
        print(f"Max Close Index: {thresholds_3m['max_close_idx']}")
        print(f"Argmin More Recent: {argmin_3m_met}")
        print(f"Condition Met: {argmin_3m_met}")
        
        if argmin_3m_met:
            conditions_met += 1
            print("\nTRUE - Argmin More Recent (3m) condition MET")
        else:
            print("\nFALSE - Argmin More Recent (3m) condition NOT met")
    else:
        condition_results['Argmin More Recent (3m)'] = False
        print(f"\nError analyzing thresholds: {thresholds_3m['error']}")
        print("\nFALSE - Argmin More Recent (3m) condition NOT met")
    
    # Condition 9: RSI Oversold Most Recent (5m)
    print("\n--- Condition 9: RSI Oversold Most Recent (5m) ---")
    rsi_5m_oversold, rsi_5m_overbought, rsi_5m_value, rsi_5m_details = analyze_rsi_condition(client, SYMBOL, 500, '5m')
    rsi_5m_met = rsi_5m_oversold and not rsi_5m_overbought
    condition_results['RSI Oversold Most Recent (5m)'] = rsi_5m_met
    
    # Print details for this condition
    print(f"\nCurrent RSI: {rsi_5m_value:.2f}")
    print(f"Oversold Most Recent: {rsi_5m_oversold}")
    print(f"Overbought Most Recent: {rsi_5m_overbought}")
    print(f"Condition Met: {rsi_5m_met}")
    
    if rsi_5m_met:
        conditions_met += 1
        print("\nTRUE - RSI Oversold Most Recent (5m) condition MET")
    else:
        print("\nFALSE - RSI Oversold Most Recent (5m) condition NOT met")
    
    # Step 3: Trading Decision
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
        print(f"\n!!! ALL {conditions_met} CONDITIONS MET - EXECUTING TRADE !!!")
        
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
    
    # Step 4: Cleanup for next iteration
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
        client = Client(api_key, api_secret)
        
        # Synchronize time with Binance servers
        if not synchronize_time_with_binance(client):
            print("Warning: Failed to synchronize time with Binance servers")
        
        return client
    except Exception as e:
        print(f"Error reading API file: {e}")
        return None

def main():
    client = get_binance_client()
    if not client:
        print("No client available. Exiting.")
        return
    
    print("=== ENHANCED BTCUSDC TRADING BOT - 9 TRIGGERS ===")
    print("Press Ctrl+C to stop monitoring.")
    print("\nEach iteration will:")
    print("1. Check for active trade and resume if necessary")
    print("2. Fetch fresh data for all timeframes (1m, 3m, 5m)")
    print("3. Analyze all 9 specified trading conditions:")
    print("   - Up Cycle Confirmed (1m)")
    print("   - Momentum Positive (1m)")
    print("   - Volume Bullish Dominance (1m)")
    print("   - FFT Forecast Up (1m)")
    print("   - FFT Forecast Up (3m)")
    print("   - FFT Forecast Up (5m)")
    print("   - Argmin More Recent (1m)")
    print("   - Argmin More Recent (3m)")
    print("   - RSI Oversold Most Recent (5m)")
    print("4. Execute trade if ALL 9 conditions are met")
    print("5. Use 100% of USDC balance for entry")
    print("6. Monitor for profit target every 5 seconds")
    print("7. Use 100% of BTC balance for exit")
    print("8. Clean up for next iteration")
    print("\nEnhanced Features:")
    print("- Now 9 selected triggers")
    print("- Take profit target set to 1.25%")
    print("- Threshold analysis with argmin/argmax for all timeframes")
    print("- FFT analysis between argmin and argmax for all timeframes")
    print("- Detailed frequency calculations and inverse FFT forecasting")
    print("- Proper data cleaning before ANY analysis")
    print("- Correct frequency dominance calculation from dip to top")
    print("- Uses entire balance for both entry and exit trades")
    print("- Fresh data fetched for each iteration")
    print("- Proper range calculation for each timeframe")
    print("- Hi-Lo range analysis with distance to current price")
    print("- Up cycle confirmation based on distance and extrema type")
    print("- FIXED: Local extrema analysis with specific distance calculation rules")
    print("- FINAL RULE: Up cycle confirmed if distance meets or exceeds hilo range")
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
