#!/usr/bin/env python3
"""
COMPLETE Enhanced Trading Bot with Multi-TimeFrame Analysis
FULLY INTEGRATED VERSION WITH ALL ENHANCEMENTS:
- Enhanced MTF ArgMin/ArgMax analysis with proper timeframe naming
- Fixed Bollinger Bands 360-degree analysis with proper symmetrical distance calculations
- Enhanced geometric pattern detection (octagonal symmetry, golden triangle)
- Volume spike confirmation for fast entry identification
- Multi-model forecasting with time-to-target calculation
- Enhanced scoring system combining all factors
- Single-run mode to find and analyze best opportunity
- Multiple moving averages confirmation
- Volume analysis (bullish vs bearish)
- RSI analysis with oversold/overbought confirmation
- Golden ratio support/resistance levels
- VPA (Volume Price Analysis) integrated across all timeframes
- Fixed symmetrical distance calculations for proper position analysis
- Enhanced MTF dip filtering with proper conditions
- 100% USDC balance trading with dust conversion
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
from binance.client import Client
from binance.exceptions import BinanceAPIException
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from scipy.signal import hilbert, find_peaks
from scipy.fft import fft, fftfreq, ifft

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# --- Optional ML/Signal libraries with graceful fallbacks ---
try:
    import pandas_ta as ta
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import pmdarima as pm
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    import talib
except Exception as e:
    print("="*80)
    print("!!! IMPORT WARNING (some optional libs missing) !!!")
    print("Some features may be limited. Error:", e)
    print("Install required packages: pip install pandas-ta scikit-learn pmdarima scipy talib-binance")
    print("="*80)

# ------------------ Configuration ------------------
API_FILE = 'api.txt'
# Focus on 1m, 3m, and 5m timeframes for fast trading
MTF_SCAN_TIMEFRAMES = ['1m', '3m', '5m']
DETAILED_TIMEFRAMES = ['1m', '3m', '5m']

# Scoring & criteria
MIN_WEIGHTED_DIP_SCORE = 2.0
VOLUME_ANALYSIS_PERIOD = 56
PRICE_UPTREND_PERIOD = 5

# Optimization & Weights - Higher weights for shorter timeframes
TIMEFRAME_WEIGHTS = {'1m': 2.5, '3m': 2.2, '5m': 2.0}
ASSET_SCAN_LIMIT = 100
MIN_24H_VOLUME_USD = 500000
MIN_24H_PRICE_CHANGE_PCT = 0.5
BATCH_SIZE = 20

# ATR
ATR_PERIOD = 14
ATR_DIP_MULTIPLIER = 1.5
ATR_SPIKE_MULTIPLIER = 2.0
ATR_VOLUME_SPIKE_THRESHOLD = 1.5

# Geometric Analysis
OCTAGONAL_SEGMENTS = 8
GOLDEN_RATIO = 1.618033988749895
GOLDEN_ANGLE = 137.5077640500378

# Volume Spike Detection
VOLUME_SPIKE_THRESHOLD = 2.0
PRICE_MOMENTUM_THRESHOLD = 0.02

# Enhanced filtering for best opportunities
MIN_OCTAGONAL_STRENGTH = 0.4
MIN_TRIANGLE_STRENGTH = 0.3
MIN_UPWARD_PHASES = [0, 1, 2, 3]
MIN_CYCLE_CONFIDENCE = 0.3

# Time estimation
MIN_TIME_TO_TARGET = 60
MAX_TIME_TO_TARGET = 86400

# RSI Configuration
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Moving Averages Configuration
SMA7_PERIOD = 7
SMA12_PERIOD = 12
SMA27_PERIOD = 27
SMA56_PERIOD = 56
SMA150_PERIOD = 150

# Bollinger Bands Configuration
BB_PERIOD = 360
BB_STD = 2

# Pre-Spike Detection Configuration
BB_WIDTH_MIN = 0.05
BB_WIDTH_MAX = 0.12
ATR_PERCENTILE_MIN = 5
ATR_PERCENTILE_MAX = 10
ROC2_GROWTH_MIN = 50
ROC2_GROWTH_MAX = 200
BUY_PRESSURE_MIN = 60
LIQUIDITY_GAP_MIN = 0.5
LIQUIDITY_GAP_MAX = 1.5

# VPA Configuration
VPA_MIN_SCORE = 30

# Trade Configuration
PROFIT_TARGET_PERCENT = 1.45
TOTAL_FEE_PERCENT = 0.22
MONITOR_INTERVAL = 5
MIN_TRADE_AMOUNT = 10
MIN_DUST_AMOUNT = 1

# Misc
MAX_KLINES_LIMIT = 2000
FFT_MIN_LENGTH = 64
ENTROPY_M = 2
ENTROPY_R_SCALE = 0.2
MAX_WORKERS = 12

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

# ------------------ Enhanced Dust Conversion ------------------

def convert_dust_to_usdc(client):
    """Convert all small dust balances to USDC before trading."""
    try:
        print("Checking for dust balances to convert...")
        account_info = client.get_account()
        dust_assets = []
        
        for balance in account_info['balances']:
            asset = balance['asset']
            free_balance = float(balance['free'])
            locked_balance = float(balance['locked'])
            total_balance = free_balance + locked_balance
            
            if asset == 'USDC' or total_balance < MIN_DUST_AMOUNT:
                continue
                
            try:
                if asset != 'USDC':
                    symbol = f"{asset}USDC"
                    ticker = client.get_symbol_ticker(symbol=symbol)
                    price = float(ticker['price'])
                    usd_value = total_balance * price
                    
                    if usd_value < MIN_TRADE_AMOUNT:
                        dust_assets.append(asset)
                        print(f"Found dust: {asset} - {total_balance} (≈${usd_value:.2f})")
            except Exception:
                if total_balance < 0.1:
                    dust_assets.append(asset)
                    print(f"Found dust (no price): {asset} - {total_balance}")
        
        if dust_assets:
            print(f"Converting dust assets to USDC: {dust_assets}")
            try:
                result = client.transfer_dust(asset=dust_assets)
                if result.get('success', False):
                    print("Dust conversion successful!")
                    for item in result.get('transferResult', []):
                        print(f"  {item['asset']}: {item['amount']} → USDC")
                else:
                    print("Dust conversion failed or not supported")
            except BinanceAPIException as e:
                print(f"Dust conversion API error: {e}")
            except Exception as e:
                print(f"Dust conversion error: {e}")
        else:
            print("No dust balances found to convert")
            
    except Exception as e:
        print(f"Error in dust conversion: {e}")

# ------------------ Enhanced Bollinger Bands Analysis ------------------

def calculate_bollinger_bands(df, period=BB_PERIOD, std_dev=BB_STD):
    """Calculate Bollinger Bands with proper error handling."""
    try:
        if df is None or len(df) < period:
            return None
            
        df = df.copy()
        close_prices = df['close'].values
        
        # Calculate middle band (SMA)
        middle_band = pd.Series(close_prices).rolling(window=period, min_periods=1).mean()
        
        # Calculate standard deviation
        std = pd.Series(close_prices).rolling(window=period, min_periods=1).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        # Add to dataframe
        df[f'BB_MIDDLE_{period}'] = middle_band
        df[f'BB_UPPER_{period}'] = upper_band
        df[f'BB_LOWER_{period}'] = lower_band
        df[f'BB_WIDTH_{period}'] = (upper_band - lower_band) / middle_band
        
        return df
    except Exception as e:
        print(f"calculate_bollinger_bands error: {e}")
        return None

def analyze_bollinger_bands_extremes(df, period=BB_PERIOD):
    """
    Enhanced Bollinger Bands analysis for extreme price detection.
    Returns lowest price below lower band and highest price above upper band,
    along with their symmetrical distance percentages that sum to 100%.
    """
    try:
        if df is None or len(df) < period:
            return None
            
        df_bb = calculate_bollinger_bands(df, period)
        if df_bb is None:
            return None
            
        close_prices = df_bb['close'].values
        upper_band = df_bb[f'BB_UPPER_{period}'].values
        lower_band = df_bb[f'BB_LOWER_{period}'].values
        
        current_price = close_prices[-1]
        
        # Find prices below lower band (oversold extremes)
        below_lower_mask = close_prices < lower_band
        below_lower_prices = close_prices[below_lower_mask]
        
        # Find prices above upper band (overbought extremes)
        above_upper_mask = close_prices > upper_band
        above_upper_prices = close_prices[above_upper_mask]
        
        # Get lowest price below lower band
        lowest_below = below_lower_prices.min() if len(below_lower_prices) > 0 else None
        lowest_below_idx = np.argmin(close_prices) if len(below_lower_prices) > 0 else None
        
        # Get highest price above upper band
        highest_above = above_upper_prices.max() if len(above_upper_prices) > 0 else None
        highest_above_idx = np.argmax(close_prices) if len(above_upper_prices) > 0 else None
        
        # Calculate symmetrical distance percentages that sum to 100%
        dist_to_lowest_below_pct = None
        dist_to_highest_above_pct = None
        
        if lowest_below is not None and highest_above is not None and lowest_below > 0 and highest_above > 0:
            # Calculate raw distances
            dist_to_lowest_raw = ((current_price - lowest_below) / lowest_below * 100) if lowest_below > 0 else 0
            dist_to_highest_raw = ((highest_above - current_price) / highest_above * 100) if highest_above > 0 else 0
            
            # Normalize to sum to 100% for symmetry
            total_dist = dist_to_lowest_raw + dist_to_highest_raw
            if total_dist > 0:
                dist_to_lowest_below_pct = (dist_to_lowest_raw / total_dist) * 100
                dist_to_highest_above_pct = (dist_to_highest_raw / total_dist) * 100
            else:
                dist_to_lowest_below_pct = 50.0
                dist_to_highest_above_pct = 50.0
                
        elif lowest_below is not None and lowest_below > 0:
            # Only lowest below available
            dist_to_lowest_raw = ((current_price - lowest_below) / lowest_below * 100)
            dist_to_lowest_below_pct = min(100.0, max(0.0, dist_to_lowest_raw))
            dist_to_highest_above_pct = 100.0 - dist_to_lowest_below_pct
            
        elif highest_above is not None and highest_above > 0:
            # Only highest above available
            dist_to_highest_raw = ((highest_above - current_price) / highest_above * 100)
            dist_to_highest_above_pct = min(100.0, max(0.0, dist_to_highest_raw))
            dist_to_lowest_below_pct = 100.0 - dist_to_highest_above_pct
            
        else:
            # No extremes found, set to neutral
            dist_to_lowest_below_pct = 50.0
            dist_to_highest_above_pct = 50.0
        
        # Ensure they always sum to 100% exactly
        total_symmetrical = dist_to_lowest_below_pct + dist_to_highest_above_pct
        if total_symmetrical > 0:
            dist_to_lowest_below_pct = (dist_to_lowest_below_pct / total_symmetrical) * 100
            dist_to_highest_above_pct = (dist_to_highest_above_pct / total_symmetrical) * 100
        
        # Determine which extreme is more recent
        lowest_below_more_recent = False
        highest_above_more_recent = False
        
        if lowest_below_idx is not None and highest_above_idx is not None:
            lowest_below_more_recent = lowest_below_idx > highest_above_idx
            highest_above_more_recent = highest_above_idx > lowest_below_idx
        elif lowest_below_idx is not None:
            lowest_below_more_recent = True
        elif highest_above_idx is not None:
            highest_above_more_recent = True
        
        return {
            'lowest_below_lower': lowest_below,
            'highest_above_upper': highest_above,
            'lowest_below_idx': lowest_below_idx,
            'highest_above_idx': highest_above_idx,
            'dist_to_lowest_below_pct': dist_to_lowest_below_pct,
            'dist_to_highest_above_pct': dist_to_highest_above_pct,
            'lowest_below_more_recent': lowest_below_more_recent,
            'highest_above_more_recent': highest_above_more_recent,
            'current_price': current_price,
            'bb_middle': float(df_bb[f'BB_MIDDLE_{period}'].iloc[-1]) if f'BB_MIDDLE_{period}' in df_bb.columns else None,
            'bb_upper': float(df_bb[f'BB_UPPER_{period}'].iloc[-1]) if f'BB_UPPER_{period}' in df_bb.columns else None,
            'bb_lower': float(df_bb[f'BB_LOWER_{period}'].iloc[-1]) if f'BB_LOWER_{period}' in df_bb.columns else None
        }
    except Exception as e:
        print(f"analyze_bollinger_bands_extremes error: {e}")
        return None

# ------------------ Enhanced MTF Analysis with BB Extremes ------------------

def get_enhanced_mtf_data(client, symbol, timeframe):
    """Enhanced MTF data analysis with Bollinger Bands extremes detection."""
    if stop_event.is_set(): 
        return None
        
    try:
        limit = 500 if timeframe == '1m' else 200
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
        if not klines or len(klines) < 20:
            return None

        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
        
        # Convert all numeric columns to float
        for c in ['open','high','low','close','volume','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore']:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except:
                df[c] = 0.0
        
        df.fillna(0.0, inplace=True)

        # Calculate ATR
        df = calculate_atr(df, ATR_PERIOD)
        if df is None or f'ATR_{ATR_PERIOD}' not in df.columns:
            return None

        current_atr = float(df[f'ATR_{ATR_PERIOD}'].iloc[-1])
        current_price = float(df['close'].iloc[-1])

        # Calculate Bollinger Bands extremes with symmetrical percentages
        bb_analysis = analyze_bollinger_bands_extremes(df, BB_PERIOD)
        
        # Calculate price changes
        price_change_1period = 0.0
        price_change_3periods = 0.0  
        price_change_5periods = 0.0
        volume_change_1period = 0.0
        volume_change_3periods = 0.0
        volume_change_5periods = 0.0
        
        if len(df) >= 2:
            price_change_1period = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100 if df['close'].iloc[-2] > 0 else 0
            volume_change_1period = ((df['volume'].iloc[-1] - df['volume'].iloc[-2]) / df['volume'].iloc[-2]) * 100 if df['volume'].iloc[-2] > 0 else 0
        
        if len(df) >= 4:
            price_change_3periods = ((df['close'].iloc[-1] - df['close'].iloc[-4]) / df['close'].iloc[-4]) * 100 if df['close'].iloc[-4] > 0 else 0
            volume_change_3periods = ((df['volume'].iloc[-1] - df['volume'].iloc[-4]) / df['volume'].iloc[-4]) * 100 if df['volume'].iloc[-4] > 0 else 0
        
        if len(df) >= 6:
            price_change_5periods = ((df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6]) * 100 if df['close'].iloc[-6] > 0 else 0
            volume_change_5periods = ((df['volume'].iloc[-1] - df['volume'].iloc[-6]) / df['volume'].iloc[-6]) * 100 if df['volume'].iloc[-6] > 0 else 0

        # VPA Analysis
        vpa_dip, vpa_breakout, vpa_score = analyze_volume_price_analysis(df)

        # Dip detection logic
        recent_high = df['high'].iloc[-20:].max()
        distance_from_high_atr = (recent_high - current_price) / current_atr if current_atr > 0 else 0.0

        is_atr_dip = distance_from_high_atr >= ATR_DIP_MULTIPLIER
        p10 = np.percentile(df['close'], 10)
        is_price_dip = current_price <= p10
        ma20 = df['close'].iloc[-20:].mean()
        is_ma_dip = current_price < ma20
        recent_change = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] if len(df) >= 10 and df['close'].iloc[-10] != 0 else 0
        is_momentum_dip = recent_change < -0.02

        dip_criteria_met = sum([is_atr_dip, is_price_dip, is_ma_dip, is_momentum_dip])
        is_dip = dip_criteria_met >= 2 or is_atr_dip

        dip_strength = 0.0
        if is_dip:
            atr_factor = min(100, distance_from_high_atr * 20)
            price_factor = max(0, (p10 - current_price) / p10 * 100) if p10 > 0 else 0
            ma_factor = max(0, (ma20 - current_price) / ma20 * 100) if ma20 > 0 else 0
            momentum_factor = abs(recent_change) * 100
            dip_strength = min(100, (atr_factor * 0.5 + price_factor * 0.2 + ma_factor * 0.15 + momentum_factor * 0.15))

        time_ago_sec = None
        if is_dip:
            min_idx = df['close'].idxmin()
            dip_ts = int(df.loc[min_idx, 'timestamp'])
            last_ts = int(df['timestamp'].iloc[-1])
            time_ago_sec = (last_ts - dip_ts) / 1000.0

        return {
            'timeframe': timeframe,
            'is_dip': is_dip,
            'dip_strength': dip_strength,
            'current_price': current_price,
            # Price and volume changes
            'price_change_1period_pct': price_change_1period,
            'price_change_3periods_pct': price_change_3periods, 
            'price_change_5periods_pct': price_change_5periods,
            'volume_change_1period_pct': volume_change_1period,
            'volume_change_3periods_pct': volume_change_3periods,
            'volume_change_5periods_pct': volume_change_5periods,
            # VPA metrics
            'vpa_dip_signals': vpa_dip,
            'vpa_breakout_signals': vpa_breakout, 
            'vpa_score': vpa_score,
            'vpa_conditions_met': vpa_score > VPA_MIN_SCORE,
            # Bollinger Bands extremes with symmetrical percentages
            'bb_analysis': bb_analysis,
            'time_ago_seconds': time_ago_sec,
            'atr': current_atr,
            'distance_from_high_atr': distance_from_high_atr
        }

    except Exception as e:
        print(f"get_enhanced_mtf_data error for {symbol} {timeframe}: {e}")
        return None

# ------------------ Enhanced MTF ArgMin/ArgMax Analysis ------------------

def analyze_mtf_argmin_argmax(client, symbol):
    """
    Enhanced MTF analysis for argmin vs argmax across all timeframes.
    Returns detailed analysis for 1m, 3m, and 5m timeframes with proper symmetrical distance calculations.
    """
    results = {}
    
    for timeframe in ['1m', '3m', '5m']:
        try:
            limit = 1200
            klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
            if not klines or len(klines) < 100:
                results[timeframe] = {'error': 'Insufficient data'}
                continue
                
            df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
            
            # Convert to numeric
            for c in ['open','high','low','close','volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df.fillna(0.0, inplace=True)
            
            close_prices = df['close'].values
            
            # Find argmin and argmax in the entire dataset
            argmin_idx = np.argmin(close_prices)
            argmax_idx = np.argmax(close_prices)
            
            # Find most recent min and max (last 200 periods)
            recent_prices = close_prices[-200:] if len(close_prices) >= 200 else close_prices
            recent_argmin_idx = np.argmin(recent_prices)
            recent_argmax_idx = np.argmax(recent_prices)
            
            # Convert to global indices
            global_recent_argmin = recent_argmin_idx + (len(close_prices) - len(recent_prices))
            global_recent_argmax = recent_argmax_idx + (len(close_prices) - len(recent_prices))
            
            # Determine which is more recent
            min_more_recent = global_recent_argmin > global_recent_argmax
            max_more_recent = global_recent_argmax > global_recent_argmin
            
            # Calculate symmetrical percentage distances
            current_price = close_prices[-1]
            min_price = close_prices[argmin_idx]
            max_price = close_prices[argmax_idx]
            recent_min_price = close_prices[global_recent_argmin]
            recent_max_price = close_prices[global_recent_argmax]
            
            # Calculate symmetrical percentage distances that sum to 100%
            if recent_min_price > 0 and recent_max_price > recent_min_price:
                price_range = recent_max_price - recent_min_price
                if price_range > 0:
                    position_from_min = ((current_price - recent_min_price) / price_range) * 100
                    position_from_max = 100 - position_from_min
                else:
                    position_from_min = 50.0
                    position_from_max = 50.0
            else:
                position_from_min = 50.0
                position_from_max = 50.0
            
            # Ensure they sum to exactly 100%
            total_position = position_from_min + position_from_max
            if total_position > 0:
                position_from_min = (position_from_min / total_position) * 100
                position_from_max = (position_from_max / total_position) * 100
            
            results[timeframe] = {
                'argmin_idx': int(argmin_idx),
                'argmax_idx': int(argmax_idx),
                'recent_argmin_idx': int(global_recent_argmin),
                'recent_argmax_idx': int(global_recent_argmax),
                'min_more_recent': min_more_recent,
                'max_more_recent': max_more_recent,
                'min_price': float(min_price),
                'max_price': float(max_price),
                'recent_min_price': float(recent_min_price),
                'recent_max_price': float(recent_max_price),
                'current_price': float(current_price),
                'position_from_min': float(position_from_min),
                'position_from_max': float(position_from_max),
                'total_periods': len(close_prices)
            }
            
        except Exception as e:
            results[timeframe] = {'error': str(e)}
    
    return results

# ------------------ Enhanced Position Calculations ------------------

def calculate_enhanced_position_analysis(client, symbol):
    """
    Calculate enhanced position analysis across all timeframes including:
    - MTF ArgMin/ArgMax analysis with symmetrical percentages
    - Bollinger Bands extremes analysis with symmetrical percentages
    - Proper normalized distance calculations that sum to 100%
    """
    results = {}
    
    for timeframe in ['1m', '3m', '5m']:
        try:
            # Get MTF data
            mtf_data = get_enhanced_mtf_data(client, symbol, timeframe)
            if not mtf_data:
                continue
                
            # Get ArgMin/ArgMax analysis with symmetrical percentages
            argmin_argmax = analyze_mtf_argmin_argmax(client, symbol)
            timeframe_argmin = argmin_argmax.get(timeframe, {}) if argmin_argmax else {}
            
            # Get Bollinger Bands analysis with symmetrical percentages
            limit = 500 if timeframe == '1m' else 200
            klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
            if not klines:
                continue
                
            df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
            
            for c in ['open','high','low','close','volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df.fillna(0.0, inplace=True)
            
            bb_analysis = analyze_bollinger_bands_extremes(df, BB_PERIOD)
            
            # Calculate enhanced symmetrical position metrics
            current_price = mtf_data['current_price']
            
            # Position from MTF thresholds with symmetrical normalization
            mtf_thresholds = get_mtf_thresholds(client, symbol)
            timeframe_thresholds = mtf_thresholds.get(timeframe, {})
            
            min_price = timeframe_thresholds.get('min', current_price)
            max_price = timeframe_thresholds.get('max', current_price)
            
            # Calculate symmetrical normalized position (sum to 100%)
            if max_price > min_price:
                position_from_min = ((current_price - min_price) / (max_price - min_price)) * 100
                position_from_max = 100 - position_from_min
            else:
                position_from_min = 50.0
                position_from_max = 50.0
            
            # Ensure they sum to exactly 100%
            total_position = position_from_min + position_from_max
            if total_position > 0:
                position_from_min = (position_from_min / total_position) * 100
                position_from_max = (position_from_max / total_position) * 100
            
            results[timeframe] = {
                'current_price': current_price,
                'min_price': min_price,
                'max_price': max_price,
                'position_from_min': position_from_min,
                'position_from_max': position_from_max,
                'argmin_analysis': timeframe_argmin,
                'bb_analysis': bb_analysis,
                'mtf_data': mtf_data
            }
            
        except Exception as e:
            print(f"calculate_enhanced_position_analysis error for {timeframe}: {e}")
            continue
    
    return results

# ------------------ VPA Analysis Functions ------------------

def analyze_volume_price_analysis(df):
    """
    Enhanced Volume Price Analysis (VPA) with multiple confirmation signals.
    """
    try:
        if df is None or len(df) < 50:
            return 0.0, 0.0, 0.0
            
        close_prices = df['close'].values
        volumes = df['volume'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        
        price_changes = np.diff(close_prices)
        volume_changes = np.diff(volumes)
        
        volume_ma = pd.Series(volumes).rolling(window=20, min_periods=1).mean().values
        normalized_volumes = volumes / volume_ma
        
        volume_confirmation_signals = 0
        total_volume_checks = 0
        
        for i in range(max(0, len(price_changes)-10), len(price_changes)):
            if i < 0 or i >= len(price_changes):
                continue
                
            if price_changes[i] > 0 and normalized_volumes[i+1] > 1.2:
                volume_confirmation_signals += 1
            elif price_changes[i] < 0 and normalized_volumes[i+1] > 1.2:
                volume_confirmation_signals -= 1
            elif price_changes[i] > 0 and normalized_volumes[i+1] < 0.8:
                volume_confirmation_signals -= 0.5
            elif price_changes[i] < 0 and normalized_volumes[i+1] < 0.8:
                volume_confirmation_signals += 0.5
                
            total_volume_checks += 1
        
        volume_confirmation_score = volume_confirmation_signals / max(1, total_volume_checks) * 50
        
        volume_climax_signals = 0
        recent_volumes = normalized_volumes[-10:]
        if len(recent_volumes) >= 5:
            volume_climax = np.max(recent_volumes) > 2.0
            if volume_climax:
                recent_price_trend = np.mean(close_prices[-5:]) < np.mean(close_prices[-10:-5])
                if recent_price_trend:
                    volume_climax_signals += 25
                else:
                    volume_climax_signals -= 25
        
        volume_divergence_signals = 0
        if len(close_prices) >= 15:
            recent_lows = low_prices[-10:]
            recent_volumes_for_lows = volumes[-10:]
            
            low_indices = []
            for i in range(1, len(recent_lows)-1):
                if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1]:
                    low_indices.append(i)
            
            if len(low_indices) >= 2:
                idx1, idx2 = low_indices[-2], low_indices[-1]
                if (recent_lows[idx2] < recent_lows[idx1] and 
                    recent_volumes_for_lows[idx2] < recent_volumes_for_lows[idx1]):
                    volume_divergence_signals += 30
        
        accumulation_signals = 0
        up_days_volume = 0
        down_days_volume = 0
        up_days_count = 0
        down_days_count = 0
        
        for i in range(max(0, len(close_prices)-20), len(close_prices)-1):
            if close_prices[i+1] > close_prices[i]:
                up_days_volume += normalized_volumes[i+1]
                up_days_count += 1
            else:
                down_days_volume += normalized_volumes[i+1]  
                down_days_count += 1
        
        if up_days_count > 0 and down_days_count > 0:
            avg_up_volume = up_days_volume / up_days_count
            avg_down_volume = down_days_volume / down_days_count
            
            if avg_up_volume > avg_down_volume * 1.1:
                accumulation_signals += 25
        
        dip_signals = max(0, volume_confirmation_score + volume_divergence_signals + accumulation_signals)
        breakout_signals = max(0, volume_confirmation_score + volume_climax_signals)
        
        vpa_score = min(100, dip_signals + breakout_signals)
        
        return dip_signals, breakout_signals, vpa_score
        
    except Exception as e:
        print(f"analyze_volume_price_analysis error: {e}")
        return 0.0, 0.0, 0.0

# ------------------ Trade Execution Functions ------------------

def execute_buy_order(client, symbol, usdc_amount):
    """Execute a market buy order for the specified symbol using 100% of available USDC."""
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        
        quantity = usdc_amount / current_price * 0.99
        
        symbol_info = client.get_symbol_info(symbol)
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = float(lot_size_filter['stepSize'])
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
    """Execute a market sell order for the specified symbol."""
    try:
        symbol_info = client.get_symbol_info(symbol)
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = float(lot_size_filter['stepSize'])
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

def get_current_price(client, symbol):
    """Get the current price for a symbol."""
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        print(f"Error getting current price: {e}")
        return None

def get_account_balance(client, asset):
    """Get the balance of a specific asset in the account."""
    try:
        account_info = client.get_account()
        for balance in account_info['balances']:
            if balance['asset'] == asset:
                return float(balance['free'])
        return 0.0
    except Exception as e:
        print(f"Error getting account balance: {e}")
        return 0.0

def monitor_trade(client, symbol, entry_price, entry_time, quantity):
    """Monitor the trade and sell when profit target is reached."""
    global trade_active, trade_info
    
    target_price = entry_price * (1 + (PROFIT_TARGET_PERCENT + TOTAL_FEE_PERCENT) / 100)
    
    while trade_active and not stop_event.is_set():
        try:
            current_price = get_current_price(client, symbol)
            if current_price is None:
                time.sleep(MONITOR_INTERVAL)
                continue
            
            price_diff = current_price - entry_price
            price_diff_pct = (price_diff / entry_price) * 100
            time_elapsed = datetime.now() - entry_time
            
            target_diff = target_price - current_price
            target_diff_pct = (target_diff / current_price) * 100
            
            trade_info = {
                'symbol': symbol,
                'entry_price': entry_price,
                'current_price': current_price,
                'price_diff': price_diff,
                'price_diff_pct': price_diff_pct,
                'time_elapsed': time_elapsed,
                'target_price': target_price,
                'target_diff': target_diff,
                'target_diff_pct': target_diff_pct,
                'quantity': quantity
            }
            
            os.system('cls' if os.name == 'nt' else 'clear')
            print("="*80)
            print("TRADE MONITOR - ACTIVE POSITION")
            print("="*80)
            print(f"Symbol: {symbol}")
            print(f"Entry Price: {entry_price:.6f}")
            print(f"Entry Time: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Current Price: {current_price:.6f}")
            print(f"Price Difference: {price_diff:.6f} ({price_diff_pct:+.2f}%)")
            print(f"Time Elapsed: {time_elapsed}")
            print(f"Target Price: {target_price:.6f}")
            print(f"Distance to Target: {target_diff:.6f} ({target_diff_pct:.2f}%)")
            print(f"Quantity: {quantity}")
            print("="*80)
            
            if current_price >= target_price:
                print(f"PROFIT TARGET REACHED! Selling at {current_price:.6f}")
                sell_result = execute_sell_order(client, symbol, quantity)
                
                if sell_result['success']:
                    print(f"SELL ORDER EXECUTED SUCCESSFULLY!")
                    print(f"Order ID: {sell_result['order_id']}")
                    print(f"Quantity Sold: {sell_result['quantity']}")
                    print(f"Estimated Profit: {(current_price - entry_price) * quantity:.6f} USDC")
                    trade_active = False
                    return True
                else:
                    print(f"ERROR EXECUTING SELL ORDER: {sell_result['error']}")
            
            time.sleep(MONITOR_INTERVAL)
        except Exception as e:
            print(f"Error in trade monitoring: {e}")
            time.sleep(MONITOR_INTERVAL)
    
    return False

# ------------------ Enhanced Geometric Analysis Functions ------------------

def calculate_octagonal_symmetry(prices, timestamps):
    """Enhanced octagonal symmetry analysis with improved phase detection."""
    try:
        if len(prices) < OCTAGONAL_SEGMENTS * 2:
            return None, 0.0
            
        price_changes = np.diff(prices)
        time_changes = np.diff(timestamps)
        
        angles = []
        for i in range(len(price_changes)):
            angle = math.atan2(price_changes[i], time_changes[i]) * 180 / math.pi
            angle = angle % 360
            angles.append(angle)
        
        current_angle = angles[-1] if angles else 0
        octagonal_phase = int(current_angle / (360 / OCTAGONAL_SEGMENTS))
        
        segment_counts = [0] * OCTAGONAL_SEGMENTS
        for angle in angles:
            segment = int(angle / (360 / OCTAGONAL_SEGMENTS))
            segment_counts[segment] += 1
            
        expected_count = len(angles) / OCTAGONAL_SEGMENTS
        variance = sum((count - expected_count) ** 2 for count in segment_counts) / OCTAGONAL_SEGMENTS
        max_variance = expected_count ** 2 * (OCTAGONAL_SEGMENTS - 1)
        symmetry_strength = 1.0 - (variance / max_variance) if max_variance > 0 else 0.0
        
        recent_angles = angles[-10:] if len(angles) >= 10 else angles
        if len(recent_angles) >= 3:
            angle_changes = [abs(recent_angles[i] - recent_angles[i-1]) for i in range(1, len(recent_angles))]
            angle_changes = [min(change, 360-change) for change in angle_changes]
            consistency = 1.0 - (np.std(angle_changes) / 90.0)
            symmetry_strength = (symmetry_strength + consistency) / 2.0
        
        return octagonal_phase, symmetry_strength
    except Exception as e:
        print(f"calculate_octagonal_symmetry error: {e}")
        return None, 0.0

def detect_golden_ratio_patterns(prices, forecasted_max_price):
    """FORECASTING GOLDEN RATIO SYSTEM FOR REVERSAL SETUPS"""
    try:
        if len(prices) < 50:
            return None

        last_1200_prices = prices[-1200:] if len(prices) >= 1200 else prices
        min_idx_local = np.argmin(last_1200_prices)
        swing_low = last_1200_prices[min_idx_local]
        forecasted_swing_high = forecasted_max_price
        
        if forecasted_swing_high <= swing_low:
            forecasted_swing_high = prices[-1] * 1.05

        fib_range = forecasted_swing_high - swing_low

        levels = {
            'Level_0.000': swing_low,
            'Level_0.146': swing_low + fib_range * 0.146,
            'Level_0.236': swing_low + fib_range * 0.236,
            'Level_0.382': swing_low + fib_range * 0.382,
            'Level_0.500': swing_low + fib_range * 0.500,
            'Level_0.618': swing_low + fib_range * 0.618,
            'Level_0.786': swing_low + fib_range * 0.786,
            'Level_1.000': forecasted_swing_high,
        }
        
        levels['info_forecast_source'] = 'ML Consensus Target'
        levels['info_swing_low_source'] = f'argmin of last {len(last_1200_prices)} 1m candles'

        return levels

    except Exception as e:
        print(f"Forecasting Golden Ratio Error: {e}")
        return None

def detect_golden_triangle(prices, timestamps):
    """Enhanced golden triangle detection with more sensitive criteria."""
    try:
        if len(prices) < 30:
            return None, 0.0
            
        local_minima = []
        local_maxima = []
        
        for i in range(1, len(prices)-1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                local_minima.append((i, prices[i]))
            elif prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                local_maxima.append((i, prices[i]))
        
        if len(local_minima) < 1 or len(local_maxima) < 2:
            return None, 0.0
            
        best_score = 0.0
        best_dir = None
        
        for min_idx, min_val in local_minima:
            for max1_idx, max1_val in local_maxima:
                for max2_idx, max2_val in local_maxima:
                    if not (min_idx < max1_idx < max2_idx):
                        continue
                        
                    a = abs(max1_val - min_val)
                    b = abs(max2_val - max1_val)
                    c = abs(max2_val - min_val)
                    
                    ratio1 = a / b if b > 0 else 0
                    ratio2 = b / a if a > 0 else 0
                    ratio3 = c / a if a > 0 else 0
                    ratio4 = a / c if c > 0 else 0
                    
                    golden_ratios = [ratio1, ratio2, ratio3, ratio4]
                    for r in golden_ratios:
                        if abs(r - GOLDEN_RATIO) < 0.3 or abs(r - 0.618) < 0.3:
                            direction = "upward" if max2_val > max1_val else "downward"
                            strength = 1.0 - min(abs(r - GOLDEN_RATIO), abs(r - 0.618)) / 0.3
                            
                            if direction == "upward":
                                strength *= 1.2
                                
                            if strength > best_score:
                                best_score = strength
                                best_dir = direction

        if best_dir is None:
            if len(prices) >= 20:
                start_price = prices[-20]
                mid_price = prices[-10]
                end_price = prices[-1]
                
                if end_price > mid_price > start_price:
                    ratio1 = (mid_price - start_price) / (end_price - start_price)
                    ratio2 = (end_price - mid_price) / (end_price - start_price)
                    
                    for r in [ratio1, ratio2]:
                        if abs(r - 0.618) < 0.1:
                            best_dir = "upward"
                            best_score = 1.0 - abs(r - 0.618) / 0.1
                            break

        return best_dir, float(best_score) if best_score > 0 else 0.0
    except Exception as e:
        print(f"detect_golden_triangle error: {e}")
        return None, 0.0

def analyze_volume_spike(client, symbol):
    """Enhanced volume spike analysis for fast spike detection."""
    try:
        klines = client.get_klines(symbol=symbol, interval='1m', limit=VOLUME_ANALYSIS_PERIOD)
        if not klines or len(klines) < VOLUME_ANALYSIS_PERIOD:
            return None
            
        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
        
        for c in ['open','high','low','close','volume','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume']:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except:
                df[c] = 0.0
        
        df.fillna(0.0, inplace=True)
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[:-5].mean()
        volume_spike_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        price_change_5 = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] if len(df) >= 5 else 0
        price_change_10 = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] if len(df) >= 10 else 0
        
        volume_trend = df['volume'].iloc[-5:].mean() / df['volume'].iloc[-10:-5].mean() if len(df) >= 10 else 1.0
        
        buy_pressure = df['taker_buy_base_asset_volume'].iloc[-5:].sum()
        sell_pressure = (df['volume'].iloc[-5:].sum() - buy_pressure)
        buy_sell_ratio = buy_pressure / (buy_pressure + sell_pressure) if (buy_pressure + sell_pressure) > 0 else 0.5
        
        bullish_volume = df['taker_buy_base_asset_volume'].iloc[-5:].sum()
        bearish_volume = (df['volume'].iloc[-5:].sum() - bullish_volume)
        total_volume = bullish_volume + bearish_volume
        
        bullish_volume_pct = (bullish_volume / total_volume * 100) if total_volume > 0 else 50.0
        bearish_volume_pct = (bearish_volume / total_volume * 100) if total_volume > 0 else 50.0
        
        volume_score = min(100, (volume_spike_ratio - 1.0) * 50) if volume_spike_ratio > 1.0 else 0
        momentum_score = min(100, (abs(price_change_5) + abs(price_change_10)) * 1000)
        trend_score = min(100, (volume_trend - 1.0) * 100) if volume_trend > 1.0 else 0
        pressure_score = abs(buy_sell_ratio - 0.5) * 200
        
        spike_score = (volume_score * 0.4 + momentum_score * 0.3 + trend_score * 0.2 + pressure_score * 0.1)
        
        return {
            'spike_score': spike_score,
            'volume_spike_ratio': volume_spike_ratio,
            'price_change_5': price_change_5,
            'price_change_10': price_change_10,
            'buy_sell_ratio': buy_sell_ratio,
            'bullish_volume_pct': bullish_volume_pct,
            'bearish_volume_pct': bearish_volume_pct,
            'bullish_volume': bullish_volume,
            'bearish_volume': bearish_volume
        }
    except Exception as e:
        print(f"analyze_volume_spike error: {e}")
        return None

def analyze_pre_spike_conditions(df):
    """Enhanced pre-spike analysis using Hilbert Transform, FFT, and signal processing."""
    try:
        if df is None or len(df) < 50:
            return False, 0.0, {"error": "Not enough data for pre-spike analysis"}
        
        conditions_met = 0
        total_score = 0.0
        details = {}
        
        close_prices = df['close'].values
        volume_data = df['volume'].values
        timestamps = np.arange(len(close_prices))
        
        close_prices = np.nan_to_num(close_prices, nan=np.mean(close_prices[~np.isnan(close_prices)]))
        close_prices = np.where(close_prices == 0, np.mean(close_prices[close_prices > 0]), close_prices)
        volume_data = np.nan_to_num(volume_data, nan=np.mean(volume_data[~np.isnan(volume_data)]))
        volume_data = np.where(volume_data == 0, np.mean(volume_data[volume_data > 0]), volume_data)
        
        norm_prices = (close_prices - np.mean(close_prices)) / np.std(close_prices)
        
        # 1. Hilbert Transform Analysis
        try:
            analytic_signal = hilbert(norm_prices)
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
            
            ht_sine = np.sin(instantaneous_phase)
            
            argmin_idx = np.argmin(close_prices[-200:]) + len(close_prices) - 200
            phase_at_min = instantaneous_phase[argmin_idx]
            current_phase = instantaneous_phase[-1]
            phase_diff = (current_phase - phase_at_min) % (2 * np.pi)
            
            phase_condition = 0 < phase_diff < np.pi
            
            amplitude_at_min = amplitude_envelope[argmin_idx]
            current_amplitude = amplitude_envelope[-1]
            amplitude_growth = (current_amplitude - amplitude_at_min) / amplitude_at_min if amplitude_at_min > 0 else 0
            amplitude_condition = amplitude_growth > 0.1
            
            recent_freq = np.mean(instantaneous_frequency[-10:])
            previous_freq = np.mean(instantaneous_frequency[-20:-10])
            freq_acceleration = (recent_freq - previous_freq) / previous_freq if previous_freq != 0 else 0
            freq_condition = freq_acceleration > 0.05
            
            hilbert_condition = phase_condition and amplitude_condition and freq_condition
            
            if hilbert_condition:
                conditions_met += 1
                phase_score = 1.0 - (phase_diff / np.pi) if phase_diff < np.pi else 0
                amp_score = min(1.0, amplitude_growth * 10)
                freq_score = min(1.0, freq_acceleration * 20)
                condition_score = (phase_score * 0.4 + amp_score * 0.3 + freq_score * 0.3) * 25
                total_score += condition_score
                details['hilbert_transform'] = {
                    'met': True,
                    'phase_diff': phase_diff,
                    'phase_condition': phase_condition,
                    'amplitude_growth': amplitude_growth,
                    'amplitude_condition': amplitude_condition,
                    'freq_acceleration': freq_acceleration,
                    'freq_condition': freq_condition,
                    'score': condition_score
                }
            else:
                details['hilbert_transform'] = {
                    'met': False,
                    'phase_diff': phase_diff,
                    'phase_condition': phase_condition,
                    'amplitude_growth': amplitude_growth,
                    'amplitude_condition': amplitude_condition,
                    'freq_acceleration': freq_acceleration,
                    'freq_condition': freq_condition,
                    'score': 0
                }
        except Exception as e:
            details['hilbert_transform'] = {'met': False, 'error': str(e)}
        
        # 2. FFT-based Forecasting
        try:
            last_1200_prices = close_prices[-1200:] if len(close_prices) >= 1200 else close_prices
            argmin_idx = np.argmin(last_1200_prices)
            segment_prices = last_1200_prices[argmin_idx:]
            n = len(segment_prices)
            
            if n < FFT_MIN_LENGTH:
                fft_condition = False
                fft_score = 0
            else:
                trend = np.polyfit(np.arange(n), segment_prices, 1)
                detrended = segment_prices - np.polyval(trend, np.arange(n))
                yf = fft(detrended)
                xf = fftfreq(n, d=1.0)
                half = n // 2
                mag = np.abs(yf[:half])
                freqs = xf[:half]
                mag[0] = 0
                
                if np.max(mag) < 1e-6:
                    fft_condition = False
                    fft_score = 0
                else:
                    idx = np.argmax(mag[1:]) + 1
                    dominant_freq = freqs[idx]
                    amplitude = mag[idx] / n
                    phase = np.angle(yf[idx])
                    period = 1.0 / abs(dominant_freq) if dominant_freq != 0 else 0
                    current_time = n - 1
                    time_since_min = current_time
                    phase_position = (time_since_min / period) % 1.0 if period > 0 else 0
                    next_peak_time = period * 0.25
                    next_trough_time = period * 0.75
                    
                    if current_time < next_peak_time:
                        next_extremum_type = "peak"
                        time_to_next = next_peak_time - current_time
                    elif current_time < next_trough_time:
                        next_extremum_type = "trough"
                        time_to_next = next_trough_time - current_time
                    else:
                        next_extremum_type = "peak"
                        time_to_next = period * 1.25 - current_time
                    
                    fft_condition = next_extremum_type == "peak" and 0 < time_to_next < period * 0.5
                    time_score = 1.0 - (time_to_next / (period * 0.5)) if period > 0 else 0
                    amp_score = min(1.0, amplitude * 1000)
                    fft_score = (time_score * 0.6 + amp_score * 0.4) * 25
                    
                    details['fft_analysis'] = {
                        'met': fft_condition,
                        'dominant_freq': dominant_freq,
                        'amplitude': amplitude,
                        'phase': phase,
                        'period': period,
                        'next_extremum_type': next_extremum_type,
                        'time_to_next': time_to_next,
                        'score': fft_score if fft_condition else 0
                    }
            
            if fft_condition:
                conditions_met += 1
                total_score += fft_score
        except Exception as e:
            details['fft_analysis'] = {'met': False, 'error': str(e)}
        
        # 3. Signal Processing Metrics
        try:
            signal_energy = np.sum(norm_prices ** 2)
            signal_power = signal_energy / len(norm_prices)
            
            n = len(norm_prices)
            yf = fft(norm_prices)
            xf = fftfreq(n, d=1.0)
            half = n // 2
            mag = np.abs(yf[:half])
            freqs = xf[:half]
            dominant_freq_idx = np.argmax(mag[1:]) + 1
            dominant_freq = abs(freqs[dominant_freq_idx])
            
            vibration = np.std(norm_prices)
            
            analytic_signal = hilbert(norm_prices)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
            amplitude_envelope = np.abs(analytic_signal)
            angular_momentum = np.mean(amplitude_envelope[:-1] * instantaneous_frequency)
            
            threshold = np.std(norm_prices) * 2
            significant_changes = np.where(np.abs(np.diff(norm_prices)) > threshold)[0]
            pulse_rate = len(significant_changes) / len(norm_prices)
            
            momentum = np.diff(norm_prices)
            impulse = np.sum(np.abs(np.diff(momentum)))
            
            energy_score = min(1.0, signal_energy / 1000)
            power_score = min(1.0, signal_power * 10)
            freq_score = min(1.0, dominant_freq * 100)
            vibration_score = 1.0 - min(1.0, abs(vibration - 1.0))
            angular_momentum_score = min(1.0, angular_momentum * 10)
            pulse_score = 1.0 - min(1.0, abs(pulse_rate - 0.1) * 10)
            impulse_score = min(1.0, impulse / 10)
            
            signal_condition = (
                energy_score > 0.5 and 
                power_score > 0.5 and 
                angular_momentum_score > 0.5 and 
                impulse_score > 0.3
            )
            
            if signal_condition:
                conditions_met += 1
                condition_score = (
                    energy_score * 0.15 + 
                    power_score * 0.15 + 
                    freq_score * 0.1 + 
                    vibration_score * 0.1 + 
                    angular_momentum_score * 0.2 + 
                    pulse_score * 0.1 + 
                    impulse_score * 0.2
                ) * 25
                total_score += condition_score
                details['signal_processing'] = {
                    'met': True,
                    'signal_energy': signal_energy,
                    'signal_power': signal_power,
                    'dominant_freq': dominant_freq,
                    'vibration': vibration,
                    'angular_momentum': angular_momentum,
                    'pulse_rate': pulse_rate,
                    'impulse': impulse,
                    'score': condition_score
                }
            else:
                details['signal_processing'] = {
                    'met': False,
                    'signal_energy': signal_energy,
                    'signal_power': signal_power,
                    'dominant_freq': dominant_freq,
                    'vibration': vibration,
                    'angular_momentum': angular_momentum,
                    'pulse_rate': pulse_rate,
                    'impulse': impulse,
                    'score': 0
                }
        except Exception as e:
            details['signal_processing'] = {'met': False, 'error': str(e)}
        
        # 4. Golden Ratio Volume Analysis
        try:
            last_1200_prices = close_prices[-1200:] if len(close_prices) >= 1200 else close_prices
            last_1200_volumes = volume_data[-1200:] if len(volume_data) >= 1200 else volume_data
            
            min_idx = np.argmin(last_1200_prices)
            max_idx = np.argmax(last_1200_prices)
            min_price = last_1200_prices[min_idx]
            max_price = last_1200_prices[max_idx]
            price_range = max_price - min_price
            
            min_volume = np.min(last_1200_volumes)
            max_volume = np.max(last_1200_volumes)
            volume_range = max_volume - min_volume
            
            current_price = last_1200_prices[-1]
            price_position = (current_price - min_price) / price_range if price_range > 0 else 0.5
            
            current_volume = last_1200_volumes[-1]
            volume_position = (current_volume - min_volume) / volume_range if volume_range > 0 else 0.5
            
            golden_levels = [0.0, 0.236, 0.382, 0.618, 0.786, 1.0]
            
            closest_price_level = min(golden_levels, key=lambda x: abs(x - price_position))
            price_level_diff = abs(price_position - closest_price_level)
            
            closest_volume_level = min(golden_levels, key=lambda x: abs(x - volume_position))
            volume_level_diff = abs(volume_position - closest_volume_level)
            
            alignment_score = 1.0 - (price_level_diff + volume_level_diff) / 2.0
            at_golden_level = price_level_diff < 0.05
            volume_confirmation = (
                (price_position > 0.5 and volume_position > 0.5) or
                (price_position < 0.5 and volume_position < 0.5)
            )
            
            golden_condition = at_golden_level and volume_confirmation and alignment_score > 0.7
            
            if golden_condition:
                conditions_met += 1
                condition_score = (alignment_score * 0.6 + (1.0 - price_level_diff) * 0.2 + (1.0 - volume_level_diff) * 0.2) * 25
                total_score += condition_score
                details['golden_ratio_volume'] = {
                    'met': True,
                    'price_position': price_position,
                    'volume_position': volume_position,
                    'closest_price_level': closest_price_level,
                    'closest_volume_level': closest_volume_level,
                    'price_level_diff': price_level_diff,
                    'volume_level_diff': volume_level_diff,
                    'alignment_score': alignment_score,
                    'at_golden_level': at_golden_level,
                    'volume_confirmation': volume_confirmation,
                    'score': condition_score
                }
            else:
                details['golden_ratio_volume'] = {
                    'met': False,
                    'price_position': price_position,
                    'volume_position': volume_position,
                    'closest_price_level': closest_price_level,
                    'closest_volume_level': closest_volume_level,
                    'price_level_diff': price_level_diff,
                    'volume_level_diff': volume_level_diff,
                    'alignment_score': alignment_score,
                    'at_golden_level': at_golden_level,
                    'volume_confirmation': volume_confirmation,
                    'score': 0
                }
        except Exception as e:
            details['golden_ratio_volume'] = {'met': False, 'error': str(e)}
        
        spike_imminent = conditions_met >= 3
        
        return spike_imminent, total_score, details
    except Exception as e:
        print(f"analyze_pre_spike_conditions error: {e}")
        return False, 0.0, {"error": str(e)}

def calculate_time_to_target(current_price, target_price, cycle_period, current_phase, octagonal_phase, volume_spike_data, timeframe='1m', price_history=None):
    """Enhanced time to target calculation with volume spike consideration."""
    try:
        if cycle_period <= 0 or current_price <= 0 or target_price <= 0:
            return None, 0.0
            
        price_distance_pct = abs(target_price - current_price) / current_price
        
        timeframe_seconds = {
            '1m': 60,
            '3m': 180,
            '5m': 300
        }
        
        seconds_per_point = timeframe_seconds.get(timeframe, 60)
        cycle_period_seconds = cycle_period * seconds_per_point
        
        base_time = cycle_period_seconds * 0.5
        distance_factor = 1.0 + min(2.0, price_distance_pct * 10)
        
        volatility_factor = 1.0
        if price_history is not None and len(price_history) > 20:
            returns = np.diff(price_history) / price_history[:-1]
            volatility = np.std(returns)
            volatility_factor = 1.0 / (1.0 + volatility * 50)
        
        phase_speed_factor = 1.0
        if octagonal_phase is not None:
            if octagonal_phase in [0, 1]:
                phase_speed_factor = 0.7
            elif octagonal_phase in [2, 3]:
                phase_speed_factor = 0.85
            elif octagonal_phase in [4, 5]:
                phase_speed_factor = 1.15
            else:
                phase_speed_factor = 1.3
        
        volume_factor = 1.0
        if volume_spike_data:
            volume_spike_ratio = volume_spike_data.get('volume_spike_ratio', 1.0)
            buy_sell_ratio = volume_spike_data.get('buy_sell_ratio', 0.5)
            
            if volume_spike_ratio > VOLUME_SPIKE_THRESHOLD and buy_sell_ratio > 0.6:
                volume_factor = 0.6
            elif volume_spike_ratio > VOLUME_SPIKE_THRESHOLD:
                volume_factor = 0.8
        
        adjusted_time = base_time * distance_factor * volatility_factor * phase_speed_factor * volume_factor
        adjusted_time = max(MIN_TIME_TO_TARGET, min(MAX_TIME_TO_TARGET, adjusted_time))
        
        phase_alignment = 1.0 - abs(current_phase - 0.5) * 2
        octagonal_alignment = 1.0 - abs(octagonal_phase - OCTAGONAL_SEGMENTS/2) / (OCTAGONAL_SEGMENTS/2)
        
        volume_confidence = 0.5
        if volume_spike_data:
            volume_confidence = min(1.0, volume_spike_data.get('volume_spike_ratio', 1.0) / VOLUME_SPIKE_THRESHOLD)
        
        distance_confidence = 1.0 - min(1.0, price_distance_pct * 5)
        
        pattern_confidence = 0.5
        if price_history is not None and len(price_history) > 50:
            target_change = (target_price - current_price) / current_price
            historical_changes = []
            for i in range(20, len(price_history)):
                change = (price_history[i] - price_history[i-20]) / price_history[i-20]
                historical_changes.append(change)
            
            if historical_changes:
                similar_changes = sum(1 for change in historical_changes if abs(change) >= abs(target_change))
                pattern_confidence = min(1.0, similar_changes / len(historical_changes) * 2)
        
        confidence = (
            phase_alignment * 0.20 + 
            octagonal_alignment * 0.20 + 
            volume_confidence * 0.25 + 
            distance_confidence * 0.15 + 
            pattern_confidence * 0.20
        )
        
        confidence = max(0.1, min(1.0, confidence))
        
        return adjusted_time, confidence
    except Exception as e:
        print(f"calculate_time_to_target error: {e}")
        return None, 0.0

def analyze_sinuosidal_pattern(prices, timestamps, timeframe='1m'):
    """Enhanced sinusoidal pattern analysis with next extremum prediction."""
    try:
        if len(prices) < FFT_MIN_LENGTH:
            return None, None, None, None, None
            
        minima_indices = []
        for i in range(1, len(prices)-1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                minima_indices.append(i)
                
        if len(minima_indices) < 2:
            return None, None, None, None, None
            
        last_min_idx = minima_indices[-1]
        last_min_price = prices[last_min_idx]
        last_min_time = timestamps[last_min_idx]
        
        segment_prices = prices[last_min_idx:]
        segment_times = timestamps[last_min_idx:]
        
        trend = np.polyfit(segment_times, segment_prices, 1)
        detrended = segment_prices - np.polyval(trend, segment_times)
        
        n = len(detrended)
        yf = fft(detrended)
        xf = fftfreq(n, d=1.0)
        
        half = n // 2
        mag = np.abs(yf[:half])
        freqs = xf[:half]
        mag[0] = 0
        
        if np.max(mag) < 1e-6:
            return None, None, None, None, None
            
        idx = np.argmax(mag[1:]) + 1
        dominant_freq = freqs[idx]
        amplitude = mag[idx] / n
        
        phase = np.angle(yf[idx])
        
        period = 1.0 / abs(dominant_freq) if dominant_freq != 0 else 0
        current_time = segment_times[-1]
        
        time_since_min = current_time - last_min_time
        
        phase_position = (time_since_min / period) % 1.0 if period > 0 else 0
        
        next_peak_time = last_min_time + period * 0.25
        next_trough_time = last_min_time + period * 0.75
        
        if current_time < next_peak_time:
            next_extremum_time = next_peak_time
            next_extremum_type = "peak"
        elif current_time < next_trough_time:
            next_extremum_time = next_trough_time
            next_extremum_type = "trough"
        else:
            next_extremum_time = last_min_time + period * 1.25
            next_extremum_type = "peak"
            
        return amplitude, dominant_freq, phase, (next_extremum_time, next_extremum_type), period
    except Exception as e:
        print(f"analyze_sinuosidal_pattern error: {e}")
        return None, None, None, None, None

def detect_keops_phi_pivots(prices, timestamps):
    """Keops' time-based pivot approach using phi powers."""
    try:
        if len(prices) < 50:
            return None
        
        phi_powers = {
            'phi^-4': 0.1458980337503153,
            'phi^-3': 0.2360679774997897,
            'phi^-2': 0.3819660112501051,
            'phi^-1': 0.6180339887498948,
            '1-phi^-3': 0.7639320225002103,
            '1-phi^-4': 0.8541019662496847
        }
        
        pivots = []
        for i in range(1, len(prices)-1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                pivots.append((i, prices[i], 'low'))
            elif prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                pivots.append((i, prices[i], 'high'))
        
        if len(pivots) < 2:
            return None
        
        recent_legs = []
        for i in range(len(pivots)-1):
            if pivots[i][2] != pivots[i+1][2]:
                recent_legs.append((pivots[i], pivots[i+1]))
        
        if not recent_legs:
            return None
        
        last_leg = recent_legs[-1]
        start_pivot = last_leg[0]
        end_pivot = last_leg[1]
        
        last_1200_prices = prices[-1200:] if len(prices) >= 1200 else prices
        last_1200_min_idx = np.argmin(last_1200_prices)
        last_1200_max_idx = np.argmax(last_1200_prices)
        
        global_min_idx = last_1200_min_idx + (len(prices) - len(last_1200_prices))
        global_max_idx = last_1200_max_idx + (len(prices) - len(last_1200_prices))
        
        most_recent_extrema_idx = max(global_min_idx, global_max_idx)
        most_recent_extrema_type = 'min' if most_recent_extrema_idx == global_min_idx else 'max'
        
        is_up_leg = most_recent_extrema_type == 'min'
        
        leg_range = abs(end_pivot[1] - start_pivot[1])
        
        if is_up_leg:
            phi_pivot_price = start_pivot[1]
            phi_pivot_time = timestamps[start_pivot[0]]
        else:
            phi_pivot_price = start_pivot[1]
            phi_pivot_time = timestamps[start_pivot[0]]
        
        phi_levels = {}
        for name, ratio in phi_powers.items():
            if is_up_leg:
                level_price = phi_pivot_price + (leg_range * ratio)
            else:
                level_price = phi_pivot_price - (leg_range * ratio)
            phi_levels[name] = level_price
        
        time_range = timestamps[end_pivot[0]] - timestamps[start_pivot[0]]
        phi_time_levels = {}
        for name, ratio in phi_powers.items():
            phi_time_levels[name] = phi_pivot_time + (time_range * ratio)
        
        current_price = prices[-1]
        current_time = timestamps[-1]
        
        closest_price_level = min(phi_levels.items(), key=lambda x: abs(x[1] - current_price))
        closest_time_level = min(phi_time_levels.items(), key=lambda x: abs(x[1] - current_time))
        
        future_pivot_times = []
        for name, time_val in phi_time_levels.items():
            if time_val > current_time:
                future_pivot_times.append((name, time_val))
        
        future_pivot_times.sort(key=lambda x: x[1])
        
        golden_ratio_target = None
        if is_up_leg:
            golden_ratio_target = end_pivot[1] + (leg_range * GOLDEN_RATIO)
        else:
            golden_ratio_target = end_pivot[1] - (leg_range * GOLDEN_RATIO)
        
        return {
            'leg_type': 'up' if is_up_leg else 'down',
            'start_pivot': start_pivot,
            'end_pivot': end_pivot,
            'phi_pivot_price': phi_pivot_price,
            'phi_pivot_time': phi_pivot_time,
            'phi_price_levels': phi_levels,
            'phi_time_levels': phi_time_levels,
            'closest_price_level': closest_price_level,
            'closest_time_level': closest_time_level,
            'future_pivot_times': future_pivot_times,
            'current_price': current_price,
            'current_time': current_time,
            'most_recent_extrema_type': most_recent_extrema_type,
            'most_recent_extrema_idx': most_recent_extrema_idx,
            'golden_ratio_target': golden_ratio_target
        }
    except Exception as e:
        print(f"detect_keops_phi_pivots error: {e}")
        return None

# ------------------ Utility & Indicator Functions ------------------

def calculate_atr(df, period=ATR_PERIOD):
    """Robust ATR calculation."""
    try:
        if df is None or len(df) < 2:
            return None
        for col in ['high','low','close']:
            if col not in df.columns:
                return None
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        df = df.copy()
        df[f'ATR_{period}'] = atr
        return df
    except Exception as e:
        print(f"calculate_atr error: {type(e).__name__}: {e}")
        try:
            df.ta.atr(length=period, append=True)
            col_candidates = [c for c in df.columns if c.lower().startswith('atr')]
            if col_candidates:
                df.rename(columns={col_candidates[-1]: f'ATR_{period}'}, inplace=True)
                return df
        except Exception:
            pass
        return None

def calculate_rsi(df, period=RSI_PERIOD):
    """Calculate RSI indicator."""
    try:
        if df is None or len(df) < period + 1:
            return None
        
        if hasattr(df, 'ta'):
            try:
                df.ta.rsi(length=period, append=True)
                rsi_col = f'RSI_{period}'
                if rsi_col in df.columns:
                    return df
            except Exception:
                pass
        
        delta = df['close'].diff()
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

def calculate_moving_averages(df):
    """Calculate multiple moving averages for trend confirmation."""
    try:
        if df is None or len(df) < SMA150_PERIOD:
            return None
            
        df = df.copy()
        df['SMA7'] = df['close'].rolling(window=SMA7_PERIOD).mean()
        df['SMA12'] = df['close'].rolling(window=SMA12_PERIOD).mean()
        df['SMA27'] = df['close'].rolling(window=SMA27_PERIOD).mean()
        df['SMA56'] = df['close'].rolling(window=SMA56_PERIOD).mean()
        df['SMA150'] = df['close'].rolling(window=SMA150_PERIOD).mean()
        
        return df
    except Exception as e:
        print(f"calculate_moving_averages error: {e}")
        return None

def check_polynomial_fit(prices, timestamps):
    """Check if current price is below polynomial fit line."""
    try:
        if len(prices) < 10:
            return False
            
        x = timestamps
        y = prices
        
        best_fit_line1 = np.poly1d(np.polyfit(x, y, 1))(x)
        best_fit_line3 = best_fit_line1 * 0.99
        
        if y[-1] < best_fit_line3[-1]:
            return True
        
        return False
    except Exception as e:
        print(f"check_polynomial_fit error: {e}")
        return False

def analyze_rsi_conditions(rsi_values, current_rsi):
    """Enhanced RSI analysis with two separate conditions."""
    try:
        if rsi_values is None or len(rsi_values) < 100:
            return False, False, 0.0, {"error": "Not enough RSI data"}
        
        last_oversold_idx = None
        last_overbought_idx = None
        
        for i in range(len(rsi_values) - 1, -1, -1):
            if last_oversold_idx is None and rsi_values[i] <= RSI_OVERSOLD:
                last_oversold_idx = i
            if last_overbought_idx is None and rsi_values[i] >= RSI_OVERBOUGHT:
                last_overbought_idx = i
                
            if last_oversold_idx is not None and last_overbought_idx is not None:
                break
        
        is_oversold = current_rsi <= RSI_OVERSOLD
        is_overbought = current_rsi >= RSI_OVERBOUGHT
        
        oversold_is_most_recent = False
        if last_oversold_idx is not None:
            oversold_is_most_recent = (last_oversold_idx > last_overbought_idx) if last_overbought_idx is not None else True
        
        overbought_is_most_recent = False
        if last_overbought_idx is not None:
            overbought_is_most_recent = (last_overbought_idx > last_oversold_idx) if last_oversold_idx is not None else True
        
        oversold_score = 0.0
        if is_oversold:
            oversold_score = (RSI_OVERSOLD - current_rsi) / RSI_OVERSOLD * 50
        else:
            oversold_score = max(0, (RSI_OVERSOLD - current_rsi) / RSI_OVERSOLD * 30)
        
        overbought_score = 0.0
        if is_overbought:
            overbought_score = (current_rsi - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT) * 50
        else:
            overbought_score = max(0, (current_rsi - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT) * 30)
        
        total_score = oversold_score + overbought_score
        
        details = {
            "current_rsi": current_rsi,
            "is_oversold": is_oversold,
            "is_overbought": is_overbought,
            "oversold_is_most_recent": oversold_is_most_recent,
            "overbought_is_most_recent": overbought_is_most_recent,
            "last_oversold_idx": last_oversold_idx,
            "last_overbought_idx": last_overbought_idx,
            "oversold_score": oversold_score,
            "overbought_score": overbought_score
        }
        
        return is_oversold, is_overbought, total_score, details
    except Exception as e:
        print(f"analyze_rsi_conditions error: {e}")
        return False, False, 0.0, {"error": str(e)}

def analyze_price_dip_conditions(prices, current_price):
    """Enhanced price dip analysis using argmin vs argmax of last 1200 values."""
    try:
        if prices is None or len(prices) < 100:
            return False, 0.0, {"error": "Not enough price data"}
            
        last_1200_values = prices[-1200:] if len(prices) >= 1200 else prices
        min_idx = np.argmin(last_1200_values)
        max_idx = np.argmax(last_1200_values)
        
        if min_idx < max_idx:
            return False, 0.0, {"reason": "Most recent minimum is not more recent than most recent maximum"}
        
        min_price = last_1200_values[min_idx]
        max_price = last_1200_values[max_idx]
        
        price_range = max_price - min_price
        if price_range <= 0:
            return False, 0.0, {"error": "Invalid price range"}
            
        position_in_range = (current_price - min_price) / price_range
        score = (1.0 - position_in_range) * 100
        
        details = {
            "min_idx": min_idx,
            "max_idx": max_idx,
            "min_price": min_price,
            "max_price": max_price,
            "current_price": current_price,
            "position_in_range": position_in_range,
            "score": score
        }
        
        return True, score, details
    except Exception as e:
        print(f"analyze_price_dip_conditions error: {e}")
        return False, 0.0, {"error": str(e)}

def analyze_momentum_conditions(df):
    """Enhanced momentum analysis using TALIB's MOM function."""
    try:
        if df is None or len(df) < 100:
            return False, 0.0, {"error": "Not enough data for momentum analysis"}
        
        close_prices = df['close'].values
        
        try:
            momentum_values = talib.MOM(close_prices, timeperiod=10)
        except Exception as e:
            momentum_values = np.diff(close_prices, n=10)
            momentum_values = np.concatenate([np.zeros(10), momentum_values])
        
        momentum_values = np.nan_to_num(momentum_values, nan=0.0)
        
        last_1200_prices = close_prices[-1200:] if len(close_prices) >= 1200 else close_prices
        last_1200_momentum = momentum_values[-1200:] if len(momentum_values) >= 1200 else momentum_values
        
        min_price_idx = np.argmin(last_1200_prices)
        max_price_idx = np.argmax(last_1200_prices)
        
        min_momentum_idx = np.argmin(last_1200_momentum)
        max_momentum_idx = np.argmax(last_1200_momentum)
        most_negative_momentum = last_1200_momentum[min_momentum_idx]
        most_positive_momentum = last_1200_momentum[max_momentum_idx]
        
        current_momentum = momentum_values[-1]
        
        is_most_recent_argmin = False
        is_most_recent_argmax = False
        
        if min_momentum_idx is not None and max_momentum_idx is not None:
            if min_momentum_idx > max_momentum_idx:
                is_most_recent_argmin = True
                is_most_recent_argmax = False
            else:
                is_most_recent_argmin = False
                is_most_recent_argmax = True
        
        momentum_positive = current_momentum > 0
        
        current_price_idx = len(last_1200_prices) - 1
        price_at_min = last_1200_prices[min_price_idx]
        price_at_max = last_1200_prices[max_price_idx]
        current_price = last_1200_prices[current_price_idx]
        
        pct_from_min = ((current_price - price_at_min) / price_at_min) * 100 if price_at_min > 0 else 0
        pct_from_max = ((price_at_max - current_price) / price_at_max) * 100 if price_at_max > 0 else 0
        
        is_near_min = pct_from_min < pct_from_max
        
        min_price_more_recent = min_price_idx > max_price_idx
        max_price_more_recent = max_price_idx > min_price_idx
        
        is_bullish_reversal = min_price_more_recent and is_near_min and current_momentum < 0
        is_bearish_reversal = max_price_more_recent and not is_near_min and current_momentum > 0
        
        min_more_recent_than_max = min_momentum_idx > max_momentum_idx
        
        score = 0.0
        momentum_score = min(100, current_momentum * 1000)
        
        if min_momentum_idx is not None and max_momentum_idx is not None:
            min_max_score = 1.0 - ((max_momentum_idx - min_momentum_idx) / len(last_1200_momentum))
            score += min_max_score * 50
        
        if is_bullish_reversal or is_bearish_reversal:
            score += 25
        
        score = max(0.0, min(100.0, score))
        
        details = {
            "current_momentum": current_momentum,
            "momentum_positive": momentum_positive,
            "most_negative_momentum": most_negative_momentum,
            "most_positive_momentum": most_positive_momentum,
            "is_most_recent_argmin": is_most_recent_argmin,
            "is_most_recent_argmax": is_most_recent_argmax,
            "min_momentum_idx": min_momentum_idx,
            "max_momentum_idx": max_momentum_idx,
            "min_more_recent_than_max": min_more_recent_than_max,
            "momentum_score": momentum_score,
            "min_max_score": min_max_score if min_momentum_idx is not None and max_momentum_idx is not None else 0,
            "min_price_idx": min_price_idx,
            "max_price_idx": max_price_idx,
            "price_at_min": price_at_min,
            "price_at_max": price_at_max,
            "current_price": current_price,
            "pct_from_min": pct_from_min,
            "pct_from_max": pct_from_max,
            "is_near_min": is_near_min,
            "is_bullish_reversal": is_bullish_reversal,
            "is_bearish_reversal": is_bearish_reversal
        }
        
        return momentum_positive and min_more_recent_than_max, score, details
    except Exception as e:
        print(f"analyze_momentum_conditions error: {e}")
        return False, 0.0, {"error": str(e)}

def approx_entropy(series, m=ENTROPY_M, r_scale=ENTROPY_R_SCALE):
    """Approximate Entropy (ApEn) implementation."""
    try:
        x = np.asarray(series).astype(float)
        N = len(x)
        if N <= m + 1:
            return 0.0
        r = r_scale * np.std(x)
        def _phi(m_val):
            patterns = np.array([x[i:i+m_val] for i in range(N - m_val + 1)])
            C = []
            for i in range(len(patterns)):
                d = np.max(np.abs(patterns - patterns[i]), axis=1)
                C.append(np.sum(d <= r) / (len(patterns)))
            return np.sum(np.log(C)) / (len(patterns))
        return float(abs(_phi(m) - _phi(m+1)))
    except Exception as e:
        return 0.0

def fetch_usdc_pairs(client):
    """Fetch and pre-filter USDC pairs, with 24h volume filter."""
    try:
        print("Fetching USDC pairs...")
        exchange_info = client.get_exchange_info()
        symbols = exchange_info.get('symbols', [])
        usdc_pairs = [s['symbol'] for s in symbols if s.get('quoteAsset') == 'USDC' and s.get('status') == 'TRADING' and s.get('isSpotTradingAllowed') and not any(x in s['symbol'] for x in ['UP','DOWN','BULL','BEAR'])]
        tickers = client.get_ticker()
        ticker_map = {t['symbol']: t for t in tickers}
        filtered = []
        for sym in usdc_pairs:
            t = ticker_map.get(sym)
            if not t: continue
            try:
                vol = float(t.get('quoteVolume',0.0))
                pct = abs(float(t.get('priceChangePercent', 0.0)))
                if vol > MIN_24H_VOLUME_USD and pct > MIN_24H_PRICE_CHANGE_PCT:
                    filtered.append(sym)
            except Exception:
                continue
        filtered.sort(key=lambda s: float(ticker_map[s].get('quoteVolume',0.0)), reverse=True)
        final = filtered[:ASSET_SCAN_LIMIT]
        print(f"Selected {len(final)} pairs for scanning.")
        return final
    except Exception as e:
        print(f"fetch_usdc_pairs error: {e}")
        return []

# ------------------ Core MTF & ATR Detection ------------------

def get_mtf_data(client, symbol, timeframe):
    """Enhanced MTF data analysis with ATR-based dip detection and CLEAR timeframe naming."""
    if stop_event.is_set(): return None
    try:
        limit = 500 if timeframe == '1m' else 200
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
        if not klines or len(klines) < 20:
            return None

        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
        
        for c in ['open','high','low','close','volume','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore']:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except:
                df[c] = 0.0
        
        df.fillna(0.0, inplace=True)

        df = calculate_atr(df, ATR_PERIOD)
        if df is None or f'ATR_{ATR_PERIOD}' not in df.columns:
            return None

        current_atr = float(df[f'ATR_{ATR_PERIOD}'].iloc[-1])
        current_price = float(df['close'].iloc[-1])

        price_change_1period = 0.0
        price_change_3periods = 0.0  
        price_change_5periods = 0.0
        volume_change_1period = 0.0
        volume_change_3periods = 0.0
        volume_change_5periods = 0.0
        
        if len(df) >= 2:
            price_change_1period = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100 if df['close'].iloc[-2] > 0 else 0
            volume_change_1period = ((df['volume'].iloc[-1] - df['volume'].iloc[-2]) / df['volume'].iloc[-2]) * 100 if df['volume'].iloc[-2] > 0 else 0
        
        if len(df) >= 4:
            price_change_3periods = ((df['close'].iloc[-1] - df['close'].iloc[-4]) / df['close'].iloc[-4]) * 100 if df['close'].iloc[-4] > 0 else 0
            volume_change_3periods = ((df['volume'].iloc[-1] - df['volume'].iloc[-4]) / df['volume'].iloc[-4]) * 100 if df['volume'].iloc[-4] > 0 else 0
        
        if len(df) >= 6:
            price_change_5periods = ((df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6]) * 100 if df['close'].iloc[-6] > 0 else 0
            volume_change_5periods = ((df['volume'].iloc[-1] - df['volume'].iloc[-6]) / df['volume'].iloc[-6]) * 100 if df['volume'].iloc[-6] > 0 else 0

        vpa_dip, vpa_breakout, vpa_score = analyze_volume_price_analysis(df)

        recent_high = df['high'].iloc[-20:].max()
        distance_from_high_atr = (recent_high - current_price) / current_atr if current_atr > 0 else 0.0

        is_atr_dip = distance_from_high_atr >= ATR_DIP_MULTIPLIER
        p10 = np.percentile(df['close'], 10)
        is_price_dip = current_price <= p10
        ma20 = df['close'].iloc[-20:].mean()
        is_ma_dip = current_price < ma20
        recent_change = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] if len(df) >= 10 and df['close'].iloc[-10] != 0 else 0
        is_momentum_dip = recent_change < -0.02

        dip_criteria_met = sum([is_atr_dip, is_price_dip, is_ma_dip, is_momentum_dip])
        is_dip = dip_criteria_met >= 2 or is_atr_dip

        dip_strength = 0.0
        if is_dip:
            atr_factor = min(100, distance_from_high_atr * 20)
            price_factor = max(0, (p10 - current_price) / p10 * 100) if p10 > 0 else 0
            ma_factor = max(0, (ma20 - current_price) / ma20 * 100) if ma20 > 0 else 0
            momentum_factor = abs(recent_change) * 100
            dip_strength = min(100, (atr_factor * 0.5 + price_factor * 0.2 + ma_factor * 0.15 + momentum_factor * 0.15))

        time_ago_sec = None
        if is_dip:
            min_idx = df['close'].idxmin()
            dip_ts = int(df.loc[min_idx, 'timestamp'])
            last_ts = int(df['timestamp'].iloc[-1])
            time_ago_sec = (last_ts - dip_ts) / 1000.0

        return {
            'timeframe': timeframe,
            'is_dip': is_dip,
            'dip_strength': dip_strength,
            'current_price': current_price,
            'price_change_1period_pct': price_change_1period,
            'price_change_3periods_pct': price_change_3periods, 
            'price_change_5periods_pct': price_change_5periods,
            'volume_change_1period_pct': volume_change_1period,
            'volume_change_3periods_pct': volume_change_3periods,
            'volume_change_5periods_pct': volume_change_5periods,
            'vpa_dip_signals': vpa_dip,
            'vpa_breakout_signals': vpa_breakout, 
            'vpa_score': vpa_score,
            'vpa_conditions_met': vpa_score > VPA_MIN_SCORE,
            'time_ago_seconds': time_ago_sec,
            'atr': current_atr,
            'distance_from_high_atr': distance_from_high_atr
        }

    except Exception as e:
        return None

# ------------------ Cycle Detection and FFT Forecast ------------------

def detect_dominant_cycle(series, sample_rate=1.0):
    """Detect dominant cycle in a 1D price series using FFT."""
    try:
        x = np.asarray(series).astype(float)
        n = len(x)
        if n < FFT_MIN_LENGTH:
            return None, None, 0.0
        trend = pd.Series(x).rolling(window=max(3, int(n//10)), min_periods=1).mean().values
        y = x - trend
        y = y - np.mean(y)
        yf = fft(y)
        xf = fftfreq(n, d=sample_rate)
        half = n // 2
        mag = np.abs(yf[:half])
        freqs = xf[:half]
        mag[0] = 0
        idx = np.argmax(mag[1:]) + 1
        dominant_freq = freqs[idx]
        if dominant_freq == 0:
            return None, None, 0.0
        dominant_period = abs(1.0 / dominant_freq)
        amplitude = mag[idx] / n
        median_mag = np.median(mag)
        conf = float(min(1.0, (mag[idx] - median_mag) / (median_mag + 1e-9)))
        conf = max(0.0, conf)
        conf = conf / (conf + 1.0)
        return dominant_period, amplitude, conf
    except Exception as e:
        return None, None, 0.0

def fft_cycle_forecast(series, dominant_period, amplitude, forecast_horizon=1):
    """Simple cycle-aware forecast using sine model."""
    try:
        x = np.asarray(series).astype(float)
        n = len(x)
        if n < 10 or dominant_period is None or amplitude is None:
            return None, 0.0
        t = np.arange(n)
        P = float(dominant_period)
        omega = 2 * np.pi / P
        sin_col = np.sin(omega * t)
        cos_col = np.cos(omega * t)
        A = np.vstack([sin_col, cos_col, np.ones_like(t)]).T
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, x, rcond=None)
            a_sin, a_cos, offset = coeffs
            amp_est = np.sqrt(a_sin**2 + a_cos**2)
            phase = np.arctan2(a_cos, a_sin)
            t_future = n + forecast_horizon - 1
            pred = offset + amp_est * np.sin(omega * t_future + phase)
            cyc_conf = float(min(1.0, amp_est / (abs(amplitude) + 1e-9)))
            cyc_conf = cyc_conf / (1.0 + (np.std(x) / (amp_est + 1e-9)))
            cyc_conf = float(max(0.0, min(1.0, cyc_conf)))
            return float(pred), cyc_conf
        except Exception:
            return None, 0.0
    except Exception:
        return None, 0.0

# ------------------ ML Models (cleaned & robust) ------------------

def prepare_features(df):
    """Prepare a fixed, robust feature set for ML models."""
    d = df.copy()
    for c in ['close','volume','high','low','open']:
        if c not in d.columns:
            d[c] = 0.0
    try:
        if hasattr(d, 'ta'):
            try:
                d.ta.rsi(length=14, append=True)
                d.ta.sma(length=50, append=True)
                d.ta.ema(length=21, append=True)
                d.ta.macd(append=True)
                d.ta.bbands(append=True)
                d.ta.sma(length=50, append=True)
            except Exception:
                pass
        d['return_1'] = d['close'].pct_change(1)
        d['return_3'] = d['close'].pct_change(3)
        d['vol_ma_10'] = d['volume'].rolling(10, min_periods=1).mean()
        if f'ATR_{ATR_PERIOD}' not in d.columns:
            d = calculate_atr(d, ATR_PERIOD) or d
        d.ffill(inplace=True)
        d.dropna(inplace=True)
        keep = []
        for c in ['close','volume',f'ATR_{ATR_PERIOD}','return_1','return_3','vol_ma_10']:
            if c in d.columns:
                keep.append(c)
        for col in d.columns:
            if any(x.lower() in col.lower() for x in ['rsi','sma','ema','macd','bbu','bbl','bbm']):
                if col not in keep:
                    keep.append(col)
        X = d[keep].copy()
        X = X.select_dtypes(include=[np.number])
        X.ffill(inplace=True)
        X.dropna(inplace=True)
        return X
    except Exception as e:
        try:
            d['return_1'] = d['close'].pct_change(1)
            d['vol_ma_10'] = d['volume'].rolling(10, min_periods=1).mean()
            X = d[['close','volume','return_1','vol_ma_10']].copy()
            X.ffill(inplace=True)
            X.dropna(inplace=True)
            return X
        except Exception:
            return None

def run_random_forest(df):
    """Trains a Random Forest model and predicts the next price."""
    try:
        d = df.copy()
        d['future_close'] = d['close'].shift(-12)
        d.dropna(inplace=True)
        if len(d) < 50:
            return None
        X = prepare_features(d)
        if X is None or X.empty:
            return None
        y = d.loc[X.index, 'future_close']
        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        last_feat = X.iloc[-1:].values
        pred = model.predict(last_feat)[0]
        return float(pred)
    except Exception:
        return None

def run_arima(series):
    """Trains an ARIMA model and predicts the next price (12-step horizon)."""
    try:
        if 'pmdarima' not in sys.modules:
            return None
        s = pd.Series(series).astype(float).dropna()
        if len(s) < 50:
            return None
        model = pm.auto_arima(s, start_p=1, start_q=1, max_p=3, max_q=3, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')
        preds = model.predict(n_periods=12)
        if isinstance(preds, (np.ndarray, list, pd.Series)) and len(preds) >= 1:
            return float(preds[-1])
        return None
    except Exception:
        return None

def run_svm(df):
    """Trains an SVR model and predicts the next price."""
    try:
        d = df.copy()
        d['future_close'] = d['close'].shift(-12)
        d.dropna(inplace=True)
        if len(d) < 50:
            return None
        X = prepare_features(d)
        if X is None or X.empty:
            return None
        y = d.loc[X.index, 'future_close']
        split = int(len(X)*0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = SVR(kernel='rbf', C=1.0, gamma='scale')
        model.fit(X_train_scaled, y_train.values.ravel())
        last_scaled = scaler.transform(X.iloc[-1:].values)
        pred = model.predict(last_scaled)[0]
        return float(pred)
    except Exception:
        return None

def run_linear_regression(df):
    """Trains a Linear Regression model and predicts the next price."""
    try:
        d = df.copy()
        d['future_close'] = d['close'].shift(-12)
        d.dropna(inplace=True)
        if len(d) < 20:
            return None
        X = prepare_features(d)
        if X is None or X.empty:
            return None
        y = d.loc[X.index, 'future_close']
        split = int(len(X)*0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X.iloc[-1:].values.reshape(1,-1))[0]
        return float(pred)
    except Exception:
        return None

def run_fft_analysis(series):
    """Performs FFT to find dominant cycles and returns forecast."""
    try:
        s = pd.Series(series).astype(float).dropna()
        if len(s) < FFT_MIN_LENGTH:
            return None
        dom_period, amplitude, conf = detect_dominant_cycle(s.values)
        if dom_period is None:
            return None
        pred, cyc_conf = fft_cycle_forecast(s.values, dom_period, amplitude, forecast_horizon=1)
        if pred is None:
            return None
        return {'target': float(pred), 'period': float(dom_period), 'amplitude': float(amplitude), 'confidence': float(conf * cyc_conf)}
    except Exception:
        return None

# ------------------ Thresholds & MTF Helpers ------------------

def get_mtf_thresholds(client, symbol):
    """Calculates min, middle, max, std dev, and ATR for each timeframe."""
    thresholds = {}
    for timeframe in DETAILED_TIMEFRAMES:
        if stop_event.is_set(): break
        try:
            klines = client.get_klines(symbol=symbol, interval=timeframe, limit=1000)
            if not klines: continue
            df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
            
            for c in ['open','high','low','close','volume','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore']:
                try:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                except:
                    df[c] = 0.0
            
            df.fillna(0.0, inplace=True)
            
            df = calculate_atr(df, ATR_PERIOD)
            if df is None or f'ATR_{ATR_PERIOD}' not in df.columns:
                continue
            close_prices = df['close'].values
            current_price = float(close_prices[-1])
            current_atr = float(df[f'ATR_{ATR_PERIOD}'].iloc[-1])
            recent_high = df['high'].iloc[-20:].max()
            distance_from_high_atr = (recent_high - current_price) / current_atr if current_atr > 0 else 0.0
            thresholds[timeframe] = {
                'min': float(np.min(close_prices)),
                'max': float(np.max(close_prices)),
                'middle': float(np.mean(close_prices)),
                'std_dev': float(np.std(close_prices)),
                'current_price': current_price,
                'atr': current_atr,
                'distance_from_high_atr': distance_from_high_atr
            }
            time.sleep(0.05)
        except Exception as e:
            continue
    return thresholds

# ------------------ Table & Display ------------------

def format_time_ago(seconds):
    if seconds is None: return "N/A"
    dt = timedelta(seconds=int(seconds))
    days = dt.days
    hours, remainder = divmod(dt.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    parts = []
    if days>0: parts.append(f"{days}d")
    if hours>0: parts.append(f"{hours}h")
    if minutes>0: parts.append(f"{minutes}m")
    if not parts: parts.append("just now")
    return " ".join(parts) + " ago"

def format_time_to_target(seconds):
    """Format time to target as a future time estimate."""
    if seconds is None: return "N/A"
    dt = timedelta(seconds=int(seconds))
    days = dt.days
    hours, remainder = divmod(dt.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    parts = []
    if days>0: parts.append(f"{days}d")
    if hours>0: parts.append(f"{hours}h")
    if minutes>0: parts.append(f"{minutes}m")
    if not parts: parts.append("< 1m")
    return "in " + " ".join(parts)

def print_dynamic_table(all_results, scan_stats):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("--- Live Market Scan - Top Candidates ---")
    if all_results:
        df = pd.DataFrame(all_results)
        
        display_columns = [
            'symbol', 'current_price', 
            '1m_min_more_recent', '3m_min_more_recent', '5m_min_more_recent',
            '1m_position_from_min', '3m_position_from_min', '5m_position_from_min',
            '1m_position_from_max', '3m_position_from_max', '5m_position_from_max',
            '1m_bb_dist_to_lowest', '1m_bb_dist_to_highest',
            '3m_bb_dist_to_lowest', '3m_bb_dist_to_highest', 
            '5m_bb_dist_to_lowest', '5m_bb_dist_to_highest',
            '1m_price_change_1period_pct', '1m_price_change_3periods_pct', '1m_price_change_5periods_pct',
            '1m_volume_change_1period_pct', '1m_volume_change_3periods_pct', '1m_volume_change_5periods_pct',
            '1m_vpa_score', '1m_vpa_conditions_met',
            '3m_price_change_1period_pct', '3m_price_change_3periods_pct', '3m_price_change_5periods_pct',
            '3m_volume_change_1period_pct', '3m_volume_change_3periods_pct', '3m_volume_change_5periods_pct',
            '3m_vpa_score', '3m_vpa_conditions_met',
            '5m_price_change_1period_pct', '5m_price_change_3periods_pct', '5m_price_change_5periods_pct', 
            '5m_volume_change_1period_pct', '5m_volume_change_3periods_pct', '5m_volume_change_5periods_pct',
            '5m_vpa_score', '5m_vpa_conditions_met',
            'avg_vpa_score', 'vpa_conditions_met_count',
            'spike_score', 'volume_spike_ratio', 'buy_sell_ratio',
            'weighted_dip_score', 'power_score'
        ]
        
        available_columns = [col for col in display_columns if col in df.columns]
        df_display = df[available_columns].head(20)
        
        print(df_display.to_string(index=False, float_format="%.4f"))
    print("\n--- Scan Statistics ---")
    for k,v in scan_stats.items():
        print(f" - {k:<25}: {v}")

# ------------------ Final Analysis Pipeline ------------------

def perform_final_analysis(client, symbol):
    """Full ML + cycle + entropy + geometric analysis with VPA integration."""
    print("\n" + "="*80)
    print(f"!!! BEST MTF DIP FOUND: {symbol} - STARTING FINAL ANALYSIS !!!")
    print("="*80)

    convert_dust_to_usdc(client)
    
    mtf_thresholds = get_mtf_thresholds(client, symbol)
    
    print("Performing enhanced MTF ArgMin/ArgMax analysis with symmetrical percentages...")
    mtf_argmin_argmax = analyze_mtf_argmin_argmax(client, symbol)
    
    print("Performing enhanced Bollinger Bands analysis with symmetrical percentages...")
    enhanced_position_analysis = calculate_enhanced_position_analysis(client, symbol)
    
    try:
        klines = client.get_klines(symbol=symbol, interval='1m', limit=MAX_KLINES_LIMIT)
        if not klines:
            print("Failed to fetch historical data. Aborting.")
            return
        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
        
        for c in ['open','high','low','close','volume','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore']:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except:
                df[c] = 0.0
        
        df.fillna(0.0, inplace=True)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for c in ['open','high','low','close','volume','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore']:
            df[c] = df[c].astype(float)
        df.set_index('timestamp', inplace=True)
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return

    try:
        if hasattr(df, 'ta'):
            df.ta.rsi(length=14, append=True)
            df.ta.macd(append=True)
            df.ta.bbands(length=20, append=True)
            df.ta.sma(length=50, append=True)
    except Exception:
        pass

    df = calculate_atr(df, ATR_PERIOD)
    if df is None:
        print("Failed to calculate ATR.")
        return

    df.ffill(inplace=True)
    df.dropna(inplace=True)

    current_price = float(df['close'].iloc[-1])
    timestamps = np.arange(len(df))
    price_history = df['close'].values
    
    print("Running ML & cycle models (this may take a moment)...")
    rf_target = run_random_forest(df)
    arima_target = run_arima(df['close'])
    svm_target = run_svm(df)
    lr_target = run_linear_regression(df)
    fft_res = run_fft_analysis(df['close'])

    ap_en = approx_entropy(df['close'].values, m=ENTROPY_M, r_scale=ENTROPY_R_SCALE)
    entropy_norm = float(1.0 - (1.0 / (1.0 + ap_en)))
    model_targets = {}
    model_confidences = {}

    if rf_target:
        model_targets['RandomForest'] = rf_target
        model_confidences['RandomForest'] = 0.9 * (1.0 - entropy_norm)
    if arima_target:
        model_targets['ARIMA'] = arima_target
        model_confidences['ARIMA'] = 0.7 * (1.0 - entropy_norm)
    if svm_target:
        model_targets['SVM'] = svm_target
        model_confidences['SVM'] = 0.6 * (1.0 - entropy_norm)
    if lr_target:
        model_targets['LinearRegression'] = lr_target
        model_confidences['LinearRegression'] = 0.4 * (1.0 - entropy_norm)
    if fft_res:
        model_targets['FFT_Cycle'] = float(fft_res['target'])
        cycle_conf = float(fft_res.get('confidence', 0.0))
        model_confidences['FFT_Cycle'] = cycle_conf * (1.0 - entropy_norm) + 0.05

    consensus_numer = 0.0
    consensus_denom = 0.0
    for name, target in model_targets.items():
        conf = model_confidences.get(name, 0.1)
        atr_latest = float(df[f'ATR_{ATR_PERIOD}'].iloc[-1]) if f'ATR_{ATR_PERIOD}' in df.columns else 0.0
        atr_penalty = 1.0 / (1.0 + atr_latest)
        weight = conf * atr_penalty
        consensus_numer += weight * float(target)
        consensus_denom += weight

    consensus_target = float(consensus_numer / consensus_denom) if consensus_denom > 0 else current_price
    potential_change_pct = ((consensus_target - current_price) / current_price) * 100 if current_price != 0 else 0.0
    
    min_target = current_price * 1.015
    if consensus_target < min_target:
        consensus_target = min_target
        potential_change_pct = ((consensus_target - current_price) / current_price) * 100

    print("Performing enhanced geometric analysis...")
    octagonal_phase, octagonal_strength = calculate_octagonal_symmetry(df['close'].values, timestamps)
    golden_levels = detect_golden_ratio_patterns(df['close'].values, consensus_target)
    triangle_direction, triangle_strength = detect_golden_triangle(df['close'].values, timestamps)
    sin_amplitude, sin_freq, sin_phase, next_extremum, sin_period = analyze_sinuosidal_pattern(df['close'].values, timestamps, '1m')
    
    volume_spike_data = analyze_volume_spike(client, symbol)
    
    pre_spike_valid, pre_spike_score, pre_spike_details = analyze_pre_spike_conditions(df)
    
    print("Performing enhanced VPA analysis...")
    vpa_dip_signals, vpa_breakout_signals, vpa_score = analyze_volume_price_analysis(df)
    
    keops_phi_pivots = detect_keops_phi_pivots(df['close'].values, timestamps)
    
    df_ma = calculate_moving_averages(df)
    sma7 = float(df_ma['SMA7'].iloc[-1]) if df_ma is not None and 'SMA7' in df_ma.columns else None
    sma12 = float(df_ma['SMA12'].iloc[-1]) if df_ma is not None and 'SMA12' in df_ma.columns else None
    sma27 = float(df_ma['SMA27'].iloc[-1]) if df_ma is not None and 'SMA27' in df_ma.columns else None
    sma56 = float(df_ma['SMA56'].iloc[-1]) if df_ma is not None and 'SMA56' in df_ma.columns else None
    sma150 = float(df_ma['SMA150'].iloc[-1]) if df_ma is not None and 'SMA150' in df_ma.columns else None
    
    is_below_poly_fit = check_polynomial_fit(df['close'].values, timestamps)
    
    df_rsi = calculate_rsi(df, RSI_PERIOD)
    current_rsi = float(df_rsi[f'RSI_{RSI_PERIOD}'].iloc[-1]) if df_rsi is not None and f'RSI_{RSI_PERIOD}' in df_rsi.columns else 50.0
    
    rsi_values = df_rsi[f'RSI_{RSI_PERIOD}'].values if df_rsi is not None and f'RSI_{RSI_PERIOD}' in df_rsi.columns else None
    is_oversold, is_overbought, rsi_score, rsi_details = analyze_rsi_conditions(rsi_values, current_rsi)
    
    price_valid, price_score, price_details = analyze_price_dip_conditions(df['close'].values, current_price)
    
    momentum_valid, momentum_score, momentum_details = analyze_momentum_conditions(df)
    
    time_to_target = None
    time_confidence = 0.0
    if fft_res and sin_period:
        cycle_period = sin_period
        current_phase = sin_phase if sin_phase is not None else 0.5
        oct_phase = octagonal_phase if octagonal_phase is not None else 4
        time_to_target, time_confidence = calculate_time_to_target(
            current_price, consensus_target, cycle_period, current_phase, oct_phase, volume_spike_data, '1m', price_history
        )

    print("\n" + "="*80)
    print("!!! FINAL ANALYSIS REPORT !!!")
    print("="*80)
    print(f"Asset: {symbol}")
    print(f"Current Price: {current_price:.6f}")
    print(f"Approximate Entropy (ApEn): {ap_en:.6f} (norm predictability factor: {1.0-entropy_norm:.6f})")
    
    print("\n--- Enhanced MTF ArgMin/ArgMax Analysis with Symmetrical Percentages ---")
    for timeframe in ['1m', '3m', '5m']:
        if timeframe in mtf_argmin_argmax:
            data = mtf_argmin_argmax[timeframe]
            if 'error' not in data:
                print(f"\n{timeframe} Timeframe:")
                print(f"  Most Recent: {'MIN' if data['min_more_recent'] else 'MAX'}")
                print(f"  Current Price: {data['current_price']:.6f}")
                print(f"  Position from Min: {data['position_from_min']:.2f}%")
                print(f"  Position from Max: {data['position_from_max']:.2f}%")
                print(f"  Total Position: {data['position_from_min'] + data['position_from_max']:.2f}%")
                print(f"  Min More Recent: {data['min_more_recent']}")
    
    print("\n--- Enhanced Bollinger Bands Analysis (360 periods) with Symmetrical Percentages ---")
    for timeframe in ['1m', '3m', '5m']:
        if timeframe in enhanced_position_analysis:
            bb_data = enhanced_position_analysis[timeframe].get('bb_analysis', {})
            if bb_data:
                print(f"\n{timeframe} Bollinger Bands:")
                print(f"  Lowest Below Lower Band: {bb_data.get('lowest_below_lower', 'N/A')}")
                print(f"  Highest Above Upper Band: {bb_data.get('highest_above_upper', 'N/A')}")
                
                dist_to_lowest = bb_data.get('dist_to_lowest_below_pct')
                dist_to_highest = bb_data.get('dist_to_highest_above_pct')
                
                print(f"  Distance to Lowest Below: {dist_to_lowest:.2f}%" if dist_to_lowest is not None else "  Distance to Lowest Below: N/A")
                print(f"  Distance to Highest Above: {dist_to_highest:.2f}%" if dist_to_highest is not None else "  Distance to Highest Above: N/A")
                
                if dist_to_lowest is not None and dist_to_highest is not None:
                    total_bb_distance = dist_to_lowest + dist_to_highest
                    print(f"  Total BB Distance: {total_bb_distance:.2f}%")
                
                print(f"  Lowest Below More Recent: {bb_data.get('lowest_below_more_recent', False)}")
                print(f"  Highest Above More Recent: {bb_data.get('highest_above_more_recent', False)}")
    
    print("\n--- Enhanced Geometric Analysis ---")
    print(f"Octagonal Phase: {octagonal_phase}/7 ({octagonal_strength:.6f} strength)")
    print(f"Golden Triangle Direction: {triangle_direction} ({triangle_strength:.6f} strength)")
    if next_extremum:
        extremum_time, extremum_type = next_extremum
        print(f"Next {extremum_type}: {extremum_time:.6f} (sinusoidal analysis)")
    if golden_levels:
        print("\n--- Golden Ratio Levels ---")
        print("Golden Ratio Levels (Pure 0→1 Internal):")
        for name, price in golden_levels.items():
            if not name.startswith('info_'):
                print(f"  {name}: {price:.6f}")
    
    if keops_phi_pivots:
        print("\n--- Keops' Phi Pivot Analysis ---")
        print(f"Leg Type: {keops_phi_pivots['leg_type']}")
        print(f"Start Pivot: {keops_phi_pivots['start_pivot'][2]} at {keops_phi_pivots['start_pivot'][1]:.6f}")
        print(f"End Pivot: {keops_phi_pivots['end_pivot'][2]} at {keops_phi_pivots['end_pivot'][1]:.6f}")
        print(f"Phi Pivot Price (phi^-4): {keops_phi_pivots['phi_pivot_price']:.6f}")
        print(f"Current Price: {keops_phi_pivots['current_price']:.6f}")
        print(f"Closest Phi Price Level: {keops_phi_pivots['closest_price_level'][0]} at {keops_phi_pivots['closest_price_level'][1]:.6f}")
        print(f"Closest Phi Time Level: {keops_phi_pivots['closest_time_level'][0]}")
        
        if keops_phi_pivots['future_pivot_times']:
            print("\nFuture Expected Pivots:")
            for name, time_val in keops_phi_pivots['future_pivot_times'][:3]:
                print(f"  {name}: {time_val:.6f}")
        
        print("\nPhi Price Levels:")
        for name, price in keops_phi_pivots['phi_price_levels'].items():
            print(f"  {name}: {price:.6f}")
        
        most_recent_extrema_type = keops_phi_pivots.get('most_recent_extrema_type', 'unknown')
        most_recent_extrema_idx = keops_phi_pivots.get('most_recent_extrema_idx', -1)
        print(f"\nMost Recent Extrema: {most_recent_extrema_type} (Index: {most_recent_extrema_idx})")
        
        if triangle_direction:
            triangle_matches_leg = (triangle_direction == "upward" and keops_phi_pivots['leg_type'] == 'up') or \
                               (triangle_direction == "downward" and keops_phi_pivots['leg_type'] == 'down')
            print(f"Golden Triangle Matches Leg Type: {triangle_matches_leg}")
        
        golden_ratio_target = keops_phi_pivots.get('golden_ratio_target')
        if golden_ratio_target:
            golden_ratio_change = ((golden_ratio_target - current_price) / current_price) * 100
            print(f"Golden Ratio Target: {golden_ratio_target:.6f} ({golden_ratio_change:+.6f}%)")

    if volume_spike_data:
        print("\n--- Volume Spike Analysis ---")
        print(f"Volume Spike Ratio: {volume_spike_data.get('volume_spike_ratio', 0):.6f}x")
        print(f"Price Change (5m): {volume_spike_data.get('price_change_5', 0)*100:.6f}%")
        print(f"Price Change (10m): {volume_spike_data.get('price_change_10', 0)*100:.6f}%")
        print(f"Buy/Sell Ratio: {volume_spike_data.get('buy_sell_ratio', 0.5):.6f}")
        print(f"Bullish Volume: {volume_spike_data.get('bullish_volume', 0):.6f} ({volume_spike_data.get('bullish_volume_pct', 0):.6f}%)")
        print(f"Bearish Volume: {volume_spike_data.get('bearish_volume', 0):.6f} ({volume_spike_data.get('bearish_volume_pct', 0):.6f}%)")
    
    print("\n--- Enhanced Pre-Spike Analysis ---")
    print(f"Pre-Spike Conditions Met: {pre_spike_valid}")
    print(f"Pre-Spike Score: {pre_spike_score:.6f}")
    if pre_spike_valid:
        print("Pre-Spike Details:")
        for condition, details in pre_spike_details.items():
            if isinstance(details, dict) and 'met' in details:
                status = "✓" if details['met'] else "✗"
                print(f"  {condition.replace('_', ' ').title()}: {status} (Score: {details.get('score', 0):.6f})")

    print("\n--- Enhanced VPA Analysis ---")
    print(f"VPA Dip Signals: {vpa_dip_signals:.6f}")
    print(f"VPA Breakout Signals: {vpa_breakout_signals:.6f}")
    print(f"VPA Score: {vpa_score:.6f}")
    print(f"VPA Conditions Met: {vpa_score > VPA_MIN_SCORE}")

    print("\n--- SMA Analysis ---")
    print(f"SMA7: {sma7:.6f}")
    print(f"SMA12: {sma12:.6f}")
    print(f"SMA27: {sma27:.6f}")
    print(f"SMA56: {sma56:.6f}")
    print(f"SMA150: {sma150:.6f}")
    
    sma_condition = "PASS" if (current_price < sma7 < sma12 < sma27 < sma56 < sma150) else "FAIL"
    print(f"SMA Condition (Close < SMA7 < SMA12 < SMA27 < SMA56 < SMA150): {sma_condition}")
    
    print("\n--- Polynomial Fit Analysis ---")
    print(f"Below Poly Fit: {is_below_poly_fit}")
    
    print("\n--- Enhanced RSI Analysis ---")
    print(f"Current RSI: {current_rsi:.6f}")
    print(f"RSI Oversold is Most Recent: {rsi_details.get('oversold_is_most_recent', False)}")
    print(f"RSI Overbought is Most Recent: {rsi_details.get('overbought_is_most_recent', False)}")
    print(f"RSI Score: {rsi_score:.6f}")
    if rsi_details:
        print(f"RSI Details: {rsi_details}")
    
    print("\n--- Enhanced Price Dip Analysis ---")
    print(f"Price Dip Conditions Met: {price_valid}")
    print(f"Price Dip Score: {price_score:.6f}")
    if price_valid:
        min_idx = price_details.get('min_idx', 0)
        max_idx = price_details.get('max_idx', 0)
        min_price = price_details.get('min_price', 0)
        position_pct = price_details.get('position_in_range', 0) * 100
        print(f"  Min Price: {min_price:.6f} (Index: {min_idx})")
        print(f"  Max Price: {price_details.get('max_price', 0):.6f} (Index: {max_idx})")
        print(f"  Current Position: {position_pct:.6f}% from Min")
    
    print("\n--- Enhanced Momentum Analysis ---")
    print(f"Current Momentum: {momentum_details.get('current_momentum', 0):.6f}")
    print(f"Momentum > 0: {momentum_details.get('momentum_positive', False)}")
    print(f"Most Recent Momentum was Argmin (Most Negative): {momentum_details.get('is_most_recent_argmin', False)}")
    print(f"Most Recent Momentum was Argmax (Most Positive): {momentum_details.get('is_most_recent_argmax', False)}")
    print(f"Min Momentum Index: {momentum_details.get('min_momentum_idx', 0)}")
    print(f"Max Momentum Index: {momentum_details.get('max_momentum_idx', 0)}")
    print(f"Most Negative Momentum: {momentum_details.get('most_negative_momentum', 0):.6f}")
    print(f"Most Positive Momentum: {momentum_details.get('most_positive_momentum', 0):.6f}")
    print(f"Min More Recent Than Max: {momentum_details.get('min_more_recent_than_max', False)}")
    print(f"Momentum Score: {momentum_score:.6f}")
    
    print("\n--- Model Predictions ---")
    for name, target in model_targets.items():
        conf = model_confidences.get(name, 0.0)
        print(f" - {name:<15}: {float(target):.6f}  conf={conf:.6f}")
    
    print("\n--- Consensus Forecast ---")
    print(f"!!! CONSENSUS TARGET: {consensus_target:.6f} ({potential_change_pct:+.6f}%) !!!")
    if time_to_target:
        time_str = format_time_to_target(time_to_target)
        print(f"!!! ESTIMATED TIME TO TARGET: {time_str} (confidence: {time_confidence:.6f}) !!!")
    
    print("\n--- MTF Thresholds & Predictive Zones (sample) ---")
    if mtf_thresholds:
        order = ['1m','3m','5m']
        for tf in order:
            if tf in mtf_thresholds:
                data = mtf_thresholds[tf]
                min_p, max_p, middle_p, std_dev = data['min'], data['max'], data['middle'], data['std_dev']
                cp = data.get('current_price', 0.0)
                pct_from_min = ((cp - min_p) / (max_p - min_p) * 100) if max_p > min_p else 0.0
                pct_from_max = ((max_p - cp) / (max_p - min_p) * 100) if max_p > min_p else 100.0
                print(f" | {tf:<4} | Min:{min_p:.6f} Max:{max_p:.6f} Middle:{middle_p:.6f} Std:{std_dev:.6f}")
                print(f" |     | Current:{cp:.6f} (Pos: {pct_from_min:.6f}% from Min, {pct_from_max:.6f}% from Max) ATR:{data.get('atr',0):.6f}")
    print("="*80)
    print("Analysis complete.")
    
    print("\n--- TRADE EXECUTION DECISION ---")
    
    usdc_balance = get_account_balance(client, 'USDC')
    print(f"USDC Balance: {usdc_balance:.6f}")
    
    if usdc_balance < MIN_TRADE_AMOUNT:
        print(f"!!! INSUFFICIENT USDC BALANCE - CANNOT EXECUTE TRADE !!!")
        print(f"Minimum required: {MIN_TRADE_AMOUNT} USDC")
        return
    
    conditions_met = 0
    total_conditions = 11
    
    if is_oversold: conditions_met += 1
    if price_valid: conditions_met += 1
    if momentum_valid: conditions_met += 1
    if pre_spike_valid: conditions_met += 1
    if octagonal_phase in MIN_UPWARD_PHASES: conditions_met += 1
    if triangle_direction == "upward": conditions_met += 1
    if sma_condition == "PASS": conditions_met += 1
    if is_below_poly_fit: conditions_met += 1
    if vpa_score > VPA_MIN_SCORE: conditions_met += 1
    
    mtf_min_condition = False
    if '1m' in mtf_argmin_argmax and 'error' not in mtf_argmin_argmax['1m']:
        mtf_min_condition = mtf_argmin_argmax['1m'].get('min_more_recent', False)
    if mtf_min_condition: conditions_met += 1
    
    bb_condition = False
    if '1m' in enhanced_position_analysis:
        bb_data = enhanced_position_analysis['1m'].get('bb_analysis', {})
        if bb_data and bb_data.get('lowest_below_more_recent', False):
            bb_condition = True
    if bb_condition: conditions_met += 1
    
    conditions_score = (conditions_met / total_conditions) * 100
    
    print(f"Conditions Met: {conditions_met}/{total_conditions} ({conditions_score:.2f}%)")
    print(f"VPA Contribution: {'YES' if vpa_score > VPA_MIN_SCORE else 'NO'} (Score: {vpa_score:.2f})")
    print(f"MTF ArgMin Condition (1m): {'YES' if mtf_min_condition else 'NO'}")
    print(f"Bollinger Bands Condition (1m): {'YES' if bb_condition else 'NO'}")
    
    if conditions_met >= 7:
        print(f"\n!!! SUFFICIENT CONDITIONS MET ({conditions_met}/{total_conditions}) - EXECUTING TRADE !!!")
        
        buy_result = execute_buy_order(client, symbol, usdc_balance)
        if buy_result['success']:
            print(f"BUY ORDER EXECUTED SUCCESSFULLY!")
            print(f"Order ID: {buy_result['order_id']}")
            print(f"Quantity: {buy_result['quantity']}")
            print(f"Price: {buy_result['price']}")
            print(f"Cost: {buy_result['cost']}")
            
            global trade_active, trade_info
            trade_active = True
            monitor_trade(client, symbol, buy_result['price'], buy_result['timestamp'], buy_result['quantity'])
        else:
            print(f"ERROR EXECUTING BUY ORDER: {buy_result['error']}")
    else:
        print(f"\n!!! INSUFFICIENT CONDITIONS MET ({conditions_met}/{total_conditions}) - NO TRADE EXECUTED !!!")
        print("(But this is the best MTF dip found in the scan)")

# ------------------ Single-asset analysis wrapper ------------------

def analyze_asset_for_table(client, symbol):
    """Gathers ALL required data for an asset with enhanced dip scoring including VPA and MTF ArgMin/ArgMax."""
    if stop_event.is_set(): return None
    result = {'symbol': symbol}
    weighted_dip_score = 0.0
    spike_score = 0.0
    
    try:
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = [ex.submit(get_enhanced_mtf_data, client, symbol, tf) for tf in MTF_SCAN_TIMEFRAMES]
            for f in as_completed(futures):
                if stop_event.is_set(): return None
                data = f.result()
                if not data:
                    continue
                tf = data['timeframe']
                
                result[f'{tf}_price_change_1period_pct'] = data['price_change_1period_pct']
                result[f'{tf}_price_change_3periods_pct'] = data['price_change_3periods_pct']
                result[f'{tf}_price_change_5periods_pct'] = data['price_change_5periods_pct']
                
                result[f'{tf}_volume_change_1period_pct'] = data['volume_change_1period_pct']
                result[f'{tf}_volume_change_3periods_pct'] = data['volume_change_3periods_pct']
                result[f'{tf}_volume_change_5periods_pct'] = data['volume_change_5periods_pct']
                
                result[f'{tf}_vpa_dip_signals'] = data['vpa_dip_signals']
                result[f'{tf}_vpa_breakout_signals'] = data['vpa_breakout_signals']
                result[f'{tf}_vpa_score'] = data['vpa_score']
                result[f'{tf}_vpa_conditions_met'] = data['vpa_conditions_met']
                
                if 'current_price' not in result:
                    result['current_price'] = data['current_price']
                    
                if data['is_dip']:
                    weight = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                    dip_strength = float(data.get('dip_strength', 50) / 100.0)
                    weighted_dip_score += weight * dip_strength

                if data.get('bb_analysis'):
                    bb_data = data['bb_analysis']
                    result[f'{tf}_bb_lowest_below'] = bb_data.get('lowest_below_lower')
                    result[f'{tf}_bb_highest_above'] = bb_data.get('highest_above_upper')
                    result[f'{tf}_bb_dist_to_lowest'] = bb_data.get('dist_to_lowest_below_pct')
                    result[f'{tf}_bb_dist_to_highest'] = bb_data.get('dist_to_highest_above_pct')
                    result[f'{tf}_bb_lowest_more_recent'] = bb_data.get('lowest_below_more_recent')
                    result[f'{tf}_bb_highest_more_recent'] = bb_data.get('highest_above_more_recent')

        mtf_argmin_argmax = analyze_mtf_argmin_argmax(client, symbol)
        for timeframe in ['1m', '3m', '5m']:
            if timeframe in mtf_argmin_argmax and 'error' not in mtf_argmin_argmax[timeframe]:
                data = mtf_argmin_argmax[timeframe]
                result[f'{timeframe}_min_more_recent'] = data.get('min_more_recent', False)
                result[f'{timeframe}_position_from_min'] = data.get('position_from_min', 0)
                result[f'{timeframe}_position_from_max'] = data.get('position_from_max', 0)

        vpa_scores = []
        vpa_conditions_met_count = 0
        for tf in MTF_SCAN_TIMEFRAMES:
            vpa_key = f'{tf}_vpa_score'
            if vpa_key in result:
                vpa_scores.append(result[vpa_key])
                if result.get(f'{tf}_vpa_conditions_met', False):
                    vpa_conditions_met_count += 1
        
        result['avg_vpa_score'] = np.mean(vpa_scores) if vpa_scores else 0
        result['vpa_conditions_met_count'] = vpa_conditions_met_count
        result['vpa_conditions_met'] = vpa_conditions_met_count >= 2

        volume_spike_data = analyze_volume_spike(client, symbol)
        if volume_spike_data:
            spike_score = volume_spike_data.get('spike_score', 0)
            result['spike_score'] = spike_score
            result['volume_spike_ratio'] = volume_spike_data.get('volume_spike_ratio', 0)
            result['buy_sell_ratio'] = volume_spike_data.get('buy_sell_ratio', 0.5)
            result['bullish_volume_pct'] = volume_spike_data.get('bullish_volume_pct', 0)
            result['bearish_volume_pct'] = volume_spike_data.get('bearish_volume_pct', 0)

        klines_1h = client.get_klines(symbol=symbol, interval='1h', limit=2)
        if klines_1h and len(klines_1h) >= 2:
            current_c = float(klines_1h[-1][4])
            past_c = float(klines_1h[-2][4])
            current_v = float(klines_1h[-1][5])
            past_v = float(klines_1h[-2][5])
            result['current_price'] = current_c
            result['price_change_1h_pct'] = ((current_c - past_c) / past_c) * 100 if past_c > 0 else 0
            result['volume_change_1h_pct'] = ((current_v - past_v) / past_v) * 100 if past_v > 0 else 0

        try:
            klines_geo = client.get_klines(symbol=symbol, interval='1m', limit=200)
            if klines_geo:
                df_geo = pd.DataFrame(klines_geo, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
                
                for c in ['open','high','low','close','volume','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore']:
                    try:
                        df_geo[c] = pd.to_numeric(df_geo[c], errors='coerce')
                    except:
                        df_geo[c] = 0.0
                
                df_geo.fillna(0.0, inplace=True)
                
                timestamps_geo = np.arange(len(df_geo))
                oct_phase, oct_strength = calculate_octagonal_symmetry(df_geo['close'].values, timestamps_geo)
                tri_direction, tri_strength = detect_golden_triangle(df_geo['close'].values, timestamps_geo)
                
                result['octagonal_phase'] = oct_phase if oct_phase is not None else 0
                result['octagonal_strength'] = oct_strength if oct_strength is not None else 0
                result['triangle_direction'] = tri_direction if tri_direction is not None else 0
                result['triangle_strength'] = tri_strength if tri_strength is not None else 0
                
                result['is_below_poly_fit'] = check_polynomial_fit(df_geo['close'].values, timestamps_geo)
                
                df_geo_ma = calculate_moving_averages(df_geo)
                if df_geo_ma is not None:
                    result['sma7'] = float(df_geo_ma['SMA7'].iloc[-1]) if 'SMA7' in df_geo_ma.columns else None
                    result['sma12'] = float(df_geo_ma['SMA12'].iloc[-1]) if 'SMA12' in df_geo_ma.columns else None
                    result['sma27'] = float(df_geo_ma['SMA27'].iloc[-1]) if 'SMA27' in df_geo_ma.columns else None
                    result['sma56'] = float(df_geo_ma['SMA56'].iloc[-1]) if 'SMA56' in df_geo_ma.columns else None
                    result['sma150'] = float(df_geo_ma['SMA150'].iloc[-1]) if 'SMA150' in df_geo_ma.columns else None
                    
                    current_price = result.get('current_price', 0)
                    if (current_price < result['sma7'] < result['sma12'] < result['sma27'] < result['sma56'] < result['sma150']):
                        result['sma_condition_met'] = True
                    else:
                        result['sma_condition_met'] = False
                
                df_geo_rsi = calculate_rsi(df_geo, RSI_PERIOD)
                if df_geo_rsi is not None and f'RSI_{RSI_PERIOD}' in df_geo_rsi.columns:
                    current_rsi = float(df_geo_rsi[f'RSI_{RSI_PERIOD}'].iloc[-1])
                    result['rsi'] = current_rsi
                    rsi_values = df_geo_rsi[f'RSI_{RSI_PERIOD}'].values
                    is_oversold, is_overbought, rsi_score, rsi_details = analyze_rsi_conditions(rsi_values, current_rsi)
                    result['rsi_oversold'] = is_oversold
                    result['rsi_overbought'] = is_overbought
                    result['rsi_score'] = rsi_score
                    result['rsi_details'] = rsi_details
                
                price_valid, price_score, price_details = analyze_price_dip_conditions(df_geo['close'].values, result.get('current_price', 0))
                result['price_dip_conditions_met'] = price_valid
                result['price_dip_score'] = price_score
                result['price_dip_details'] = price_details
                
                momentum_valid, momentum_score, momentum_details = analyze_momentum_conditions(df_geo)
                result['momentum_conditions_met'] = momentum_valid
                result['momentum_score'] = momentum_score
                result['momentum_details'] = momentum_details
                
                pre_spike_valid, pre_spike_score, pre_spike_details = analyze_pre_spike_conditions(df_geo)
                result['pre_spike_conditions_met'] = pre_spike_valid
                result['pre_spike_score'] = pre_spike_score
                result['pre_spike_details'] = pre_spike_details

        except Exception as e:
            print(f"Error getting geometric data for {symbol}: {e}")
            result['octagonal_phase'] = 0
            result['octagonal_strength'] = 0
            result['triangle_direction'] = None
            result['triangle_strength'] = 0
            result['is_below_poly_fit'] = False
            result['sma_condition_met'] = False
            result['rsi_oversold'] = False
            result['rsi_overbought'] = False
            result['price_dip_conditions_met'] = False
            result['momentum_conditions_met'] = False
            result['pre_spike_conditions_met'] = False
            result['pre_spike_score'] = 0.0

    except Exception as e:
        return None

    result['weighted_dip_score'] = float(weighted_dip_score)
    result['spike_score'] = float(spike_score)
    
    octagonal_score = result.get('octagonal_strength', 0) * 20
    triangle_score = result.get('triangle_strength', 0) * 30
    volume_score = result.get('spike_score', 0) * 0.5
    poly_fit_score = 20 if result.get('is_below_poly_fit', False) else 0
    sma_condition_score = 30 if result.get('sma_condition_met', False) else 0
    rsi_score = result.get('rsi_score', 0)
    price_dip_score = result.get('price_dip_score', 0)
    momentum_score = result.get('momentum_score', 0)
    pre_spike_score = result.get('pre_spike_score', 0)
    bullish_volume_score = result.get('bullish_volume_pct', 0) * 0.5
    vpa_score = result.get('avg_vpa_score', 0)
    
    mtf_min_score = 0
    for timeframe in ['1m', '3m', '5m']:
        min_more_recent_key = f'{timeframe}_min_more_recent'
        if min_more_recent_key in result and result[min_more_recent_key]:
            if timeframe == '1m':
                mtf_min_score += 40
            elif timeframe == '3m':
                mtf_min_score += 25
            elif timeframe == '5m':
                mtf_min_score += 15
    
    bb_score = 0
    for timeframe in ['1m', '3m', '5m']:
        bb_lowest_more_recent_key = f'{timeframe}_bb_lowest_more_recent'
        if bb_lowest_more_recent_key in result and result[bb_lowest_more_recent_key]:
            if timeframe == '1m':
                bb_score += 30
            elif timeframe == '3m':
                bb_score += 20
            elif timeframe == '5m':
                bb_score += 10
    
    result['power_score'] = float(
        weighted_dip_score * 100 +
        spike_score +
        octagonal_score +
        triangle_score +
        poly_fit_score +
        sma_condition_score +
        rsi_score +
        price_dip_score +
        momentum_score +
        pre_spike_score +
        bullish_volume_score +
        vpa_score +
        mtf_min_score +
        bb_score
    )
    
    return result

# ------------------ Main Loop (SINGLE RUN) ------------------

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
    print("--- Enhanced Trading Bot with Geometric Analysis & Fast Spike Detection ---")
    print("Press Ctrl+C to stop during the scan.")
    
    convert_dust_to_usdc(client)
    
    usdc_pairs = fetch_usdc_pairs(client)
    if not usdc_pairs:
        print("No pairs found, exiting.")
        return

    start_time = time.time()
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Scanning {len(usdc_pairs)} assets...")
    
    all_results = []
    scan_stats = {'Total Assets Scanned':0,'Potential MTF Dips':0,'No Spike Pattern':0,'Other Errors':0}
    
    assets_to_scan = usdc_pairs

    for i in range(0, len(assets_to_scan), BATCH_SIZE):
        if stop_event.is_set(): break
        batch = assets_to_scan[i:i+BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(analyze_asset_for_table, client, sym): sym for sym in batch}
            for fut in as_completed(futures):
                if stop_event.is_set(): break
                sym = futures[fut]
                try:
                    res = fut.result()
                    scan_stats['Total Assets Scanned'] += 1
                    if res:
                        all_results.append(res)
                        if res.get('weighted_dip_score',0) >= MIN_WEIGHTED_DIP_SCORE:
                            scan_stats['Potential MTF Dips'] += 1
                    else:
                        scan_stats['Other Errors'] += 1
                except Exception:
                    scan_stats['Other Errors'] += 1
        time.sleep(0.2)

    for r in all_results:
        if not r.get('spike_score', 0) > 0:
            scan_stats['No Spike Pattern'] += 1

    analysis_winner = None
    if all_results:
        filtered_results = []
        for r in all_results:
            oct_phase = r.get('octagonal_phase', 0)
            oct_strength = r.get('octagonal_strength', 0)
            tri_direction = r.get('triangle_direction', None)
            tri_strength = r.get('triangle_strength', 0)
            
            has_upward_phase = oct_phase in MIN_UPWARD_PHASES
            has_upward_triangle = tri_direction == "upward"
            meets_strength_threshold = oct_strength >= MIN_OCTAGONAL_STRENGTH or tri_strength >= MIN_TRIANGLE_STRENGTH
            sma_condition_met = r.get('sma_condition_met', False)
            below_poly_fit = r.get('is_below_poly_fit', False)
            rsi_oversold = r.get('rsi_oversold', False)
            price_dip_conditions_met = r.get('price_dip_conditions_met', False)
            momentum_conditions_met = r.get('momentum_conditions_met', False)
            pre_spike_conditions_met = r.get('pre_spike_conditions_met', False)
            vpa_conditions_met = r.get('vpa_conditions_met', False)
            mtf_min_condition = r.get('1m_min_more_recent', False)
            bb_condition = r.get('1m_bb_lowest_more_recent', False)
            
            if (has_upward_phase and has_upward_triangle and meets_strength_threshold and 
                sma_condition_met and below_poly_fit and rsi_oversold and 
                price_dip_conditions_met and momentum_conditions_met and pre_spike_conditions_met and 
                vpa_conditions_met and mtf_min_condition and bb_condition):
                filtered_results.append(r)
        
        if filtered_results:
            analysis_winner = max(filtered_results, key=lambda x: x.get('power_score',0))
        else:
            analysis_winner = max(all_results, key=lambda x: x.get('power_score',0))
    
    print_dynamic_table(all_results, scan_stats)

    if analysis_winner:
        print(f"\n!!! WINNER: {analysis_winner['symbol']} (score: {analysis_winner.get('power_score'):.6f}) !!!")
        perform_final_analysis(client, analysis_winner['symbol'])
    else:
        print("\nNo suitable MTF dip found in this scan.")
        
    duration = time.time() - start_time
    print(f"\nEnhanced analysis complete in {duration:.2f}s. Exiting.")

if __name__ == "__main__": 
    main()