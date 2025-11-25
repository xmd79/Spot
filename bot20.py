#!/usr/bin/env python3
"""
COMPLETE Enhanced Trading Bot with Multi-TimeFrame Analysis
Features integrated from entire conversation:
- MTF dip detection (1m, 3m, 5m) with proper timeframe naming
- Enhanced geometric pattern detection (octagonal symmetry, golden triangle)
- Volume spike confirmation for fast entry identification
- Multi-model forecasting with time-to-target calculation
- Sinusoidal pattern analysis for cycle timing
- Enhanced scoring system combining all factors
- Single-run mode to find and analyze best opportunity
- Multiple moving averages confirmation (SMA7 < SMA12 < SMA27 < SMA56 < SMA150)
- Polynomial fit analysis for trend confirmation
- Volume analysis (bullish vs bearish)
- RSI analysis with oversold/overbought confirmation
- Golden ratio support/resistance levels with forecasted maximum
- FFT forecast price prediction
- Pre-spike detection with Hilbert Transform, FFT, and signal processing
- Keops' phi-based pivot approach for time and price analysis
- Trade execution and monitoring system
- VPA (Volume Price Analysis) integrated across all timeframes
- Clear timeframe naming convention without confusion
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
MIN_WEIGHTED_DIP_SCORE = 4.0
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
OCTAGONAL_SEGMENTS = 8  # 8 vertices of an octagon
GOLDEN_RATIO = 1.618033988749895  # φ
GOLDEN_ANGLE = 137.5077640500378  # Golden angle in degrees

# Volume Spike Detection
VOLUME_SPIKE_THRESHOLD = 2.0  # Volume must be 2x average
PRICE_MOMENTUM_THRESHOLD = 0.02  # 2% price change in short period

# Enhanced filtering for best opportunities
MIN_OCTAGONAL_STRENGTH = 0.4  # Minimum strength to consider
MIN_TRIANGLE_STRENGTH = 0.3  # Minimum strength to consider
MIN_UPWARD_PHASES = [0, 1, 2, 3]  # Upward octagonal phases
MIN_CYCLE_CONFIDENCE = 0.3  # Minimum cycle confidence for valid prediction

# Time estimation - Shorter timeframes mean faster targets
MIN_TIME_TO_TARGET = 60  # Minimum 1 minute
MAX_TIME_TO_TARGET = 86400  # Maximum 1 day

# RSI Configuration
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_MIDDLE = 50

# Moving Averages Configuration
SMA7_PERIOD = 7
SMA12_PERIOD = 12
SMA27_PERIOD = 27
SMA56_PERIOD = 56
SMA150_PERIOD = 150

# Pre-Spike Detection Configuration
BB_WIDTH_MIN = 0.05  # Minimum Bollinger Band width for squeeze
BB_WIDTH_MAX = 0.12  # Maximum Bollinger Band width for squeeze
ATR_PERCENTILE_MIN = 5  # Minimum ATR percentile for squeeze
ATR_PERCENTILE_MAX = 10  # Maximum ATR percentile for squeeze
ROC2_GROWTH_MIN = 50  # Minimum ROC2 growth percentage
ROC2_GROWTH_MAX = 200  # Maximum ROC2 growth percentage
BUY_PRESSURE_MIN = 60  # Minimum buy pressure percentage
LIQUIDITY_GAP_MIN = 0.5  # Minimum liquidity gap percentage
LIQUIDITY_GAP_MAX = 1.5  # Maximum liquidity gap percentage

# VPA Configuration
VPA_MIN_SCORE = 30  # Minimum VPA score for valid signal

# Trade Configuration
PROFIT_TARGET_PERCENT = 1.45  # Target profit percentage
TOTAL_FEE_PERCENT = 0.22  # Total fee percentage (buy + sell)
MONITOR_INTERVAL = 5  # Seconds between monitoring checks

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

# ------------------ VPA Analysis Functions ------------------

def analyze_volume_price_analysis(df):
    """
    Enhanced Volume Price Analysis (VPA) with multiple confirmation signals.
    Returns VPA dip signals, breakout signals, and overall VPA score.
    """
    try:
        if df is None or len(df) < 50:
            return 0.0, 0.0, 0.0
            
        # Extract price and volume data
        close_prices = df['close'].values
        volumes = df['volume'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        
        # Calculate price and volume changes
        price_changes = np.diff(close_prices)
        volume_changes = np.diff(volumes)
        
        # Normalize volumes for comparison
        volume_ma = pd.Series(volumes).rolling(window=20, min_periods=1).mean().values
        normalized_volumes = volumes / volume_ma
        
        # VPA Signal 1: Volume confirmation of price moves
        volume_confirmation_signals = 0
        total_volume_checks = 0
        
        # Check last 10 candles for volume-price confirmation
        for i in range(max(0, len(price_changes)-10), len(price_changes)):
            if i < 0 or i >= len(price_changes):
                continue
                
            # Up move with high volume = bullish confirmation
            if price_changes[i] > 0 and normalized_volumes[i+1] > 1.2:
                volume_confirmation_signals += 1
            # Down move with high volume = bearish confirmation  
            elif price_changes[i] < 0 and normalized_volumes[i+1] > 1.2:
                volume_confirmation_signals -= 1
            # Up move with low volume = weakness
            elif price_changes[i] > 0 and normalized_volumes[i+1] < 0.8:
                volume_confirmation_signals -= 0.5
            # Down move with low volume = potential reversal
            elif price_changes[i] < 0 and normalized_volumes[i+1] < 0.8:
                volume_confirmation_signals += 0.5
                
            total_volume_checks += 1
        
        volume_confirmation_score = volume_confirmation_signals / max(1, total_volume_checks) * 50
        
        # VPA Signal 2: Volume climax (potential reversal points)
        volume_climax_signals = 0
        recent_volumes = normalized_volumes[-10:]
        if len(recent_volumes) >= 5:
            # Check for volume climax (extremely high volume)
            volume_climax = np.max(recent_volumes) > 2.0
            if volume_climax:
                # If volume climax occurs after a downtrend, it's a buying climax
                recent_price_trend = np.mean(close_prices[-5:]) < np.mean(close_prices[-10:-5])
                if recent_price_trend:
                    volume_climax_signals += 25  # Buying climax = bullish reversal
                else:
                    volume_climax_signals -= 25  # Selling climax = bearish reversal
        
        # VPA Signal 3: Volume divergence
        volume_divergence_signals = 0
        
        # Check for bullish divergence (price makes lower low, volume decreases)
        if len(close_prices) >= 15:
            recent_lows = low_prices[-10:]
            recent_volumes_for_lows = volumes[-10:]
            
            # Find the two most recent significant lows
            low_indices = []
            for i in range(1, len(recent_lows)-1):
                if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1]:
                    low_indices.append(i)
            
            if len(low_indices) >= 2:
                # Check if second low is lower but volume is lower (bullish divergence)
                idx1, idx2 = low_indices[-2], low_indices[-1]
                if (recent_lows[idx2] < recent_lows[idx1] and 
                    recent_volumes_for_lows[idx2] < recent_volumes_for_lows[idx1]):
                    volume_divergence_signals += 30
        
        # VPA Signal 4: Accumulation/Distribution detection
        accumulation_signals = 0
        
        # Calculate accumulation: up days with higher volume, down days with lower volume
        up_days_volume = 0
        down_days_volume = 0
        up_days_count = 0
        down_days_count = 0
        
        for i in range(max(0, len(close_prices)-20), len(close_prices)-1):
            if close_prices[i+1] > close_prices[i]:  # Up day
                up_days_volume += normalized_volumes[i+1]
                up_days_count += 1
            else:  # Down day
                down_days_volume += normalized_volumes[i+1]  
                down_days_count += 1
        
        if up_days_count > 0 and down_days_count > 0:
            avg_up_volume = up_days_volume / up_days_count
            avg_down_volume = down_days_volume / down_days_count
            
            # Accumulation: up days have higher volume than down days
            if avg_up_volume > avg_down_volume * 1.1:
                accumulation_signals += 25
        
        # Calculate overall VPA scores
        dip_signals = max(0, volume_confirmation_score + volume_divergence_signals + accumulation_signals)
        breakout_signals = max(0, volume_confirmation_score + volume_climax_signals)
        
        # Combined VPA score (0-100 scale)
        vpa_score = min(100, dip_signals + breakout_signals)
        
        return dip_signals, breakout_signals, vpa_score
        
    except Exception as e:
        print(f"analyze_volume_price_analysis error: {e}")
        return 0.0, 0.0, 0.0

# ------------------ Trade Execution Functions ------------------

def execute_buy_order(client, symbol, usdc_amount=10):
    """Execute a market buy order for the specified symbol."""
    try:
        # Get current price
        ticker = client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        
        # Calculate quantity to buy (with some buffer for price fluctuations)
        quantity = usdc_amount / current_price * 0.99  # 1% buffer
        
        # Get symbol info for precision
        symbol_info = client.get_symbol_info(symbol)
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = float(lot_size_filter['stepSize'])
            quantity = round(quantity - (quantity % step_size), 8)
        
        # Execute buy order
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
        # Get symbol info for precision
        symbol_info = client.get_symbol_info(symbol)
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = float(lot_size_filter['stepSize'])
            quantity = round(quantity - (quantity % step_size), 8)
        
        # Execute sell order
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
    
    # Calculate target price (accounting for fees)
    target_price = entry_price * (1 + (PROFIT_TARGET_PERCENT + TOTAL_FEE_PERCENT) / 100)
    
    while trade_active and not stop_event.is_set():
        try:
            current_price = get_current_price(client, symbol)
            if current_price is None:
                time.sleep(MONITOR_INTERVAL)
                continue
            
            # Calculate profit/loss
            price_diff = current_price - entry_price
            price_diff_pct = (price_diff / entry_price) * 100
            time_elapsed = datetime.now() - entry_time
            
            # Calculate distance to target
            target_diff = target_price - current_price
            target_diff_pct = (target_diff / current_price) * 100
            
            # Update trade info
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
            
            # Clear screen and display trade status
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
            
            # Check if profit target is reached
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
    """
    Enhanced octagonal symmetry analysis with improved phase detection.
    Returns the current octagonal phase and strength of symmetry.
    """
    try:
        if len(prices) < OCTAGONAL_SEGMENTS * 2:
            return None, 0.0
            
        # Calculate price changes and normalize
        price_changes = np.diff(prices)
        time_changes = np.diff(timestamps)
        
        # Calculate angles of price movements (45 degrees per octagonal segment)
        angles = []
        for i in range(len(price_changes)):
            # Calculate angle based on price change and time
            angle = math.atan2(price_changes[i], time_changes[i]) * 180 / math.pi
            # Normalize to 0-360 range
            angle = angle % 360
            angles.append(angle)
        
        # Determine which octagonal segment we're in
        current_angle = angles[-1] if angles else 0
        octagonal_phase = int(current_angle / (360 / OCTAGONAL_SEGMENTS))
        
        # Calculate symmetry strength - how well the price follows octagonal patterns
        segment_counts = [0] * OCTAGONAL_SEGMENTS
        for angle in angles:
            segment = int(angle / (360 / OCTAGONAL_SEGMENTS))
            segment_counts[segment] += 1
            
        # Calculate symmetry strength as variance from uniform distribution
        expected_count = len(angles) / OCTAGONAL_SEGMENTS
        variance = sum((count - expected_count) ** 2 for count in segment_counts) / OCTAGONAL_SEGMENTS
        max_variance = expected_count ** 2 * (OCTAGONAL_SEGMENTS - 1)
        symmetry_strength = 1.0 - (variance / max_variance) if max_variance > 0 else 0.0
        
        # Enhanced strength calculation based on recent pattern consistency
        recent_angles = angles[-10:] if len(angles) >= 10 else angles
        if len(recent_angles) >= 3:
            # Check if recent angles are consistent (moving in same direction)
            angle_changes = [abs(recent_angles[i] - recent_angles[i-1]) for i in range(1, len(recent_angles))]
            # Normalize angle changes to be within 0-180 degrees
            angle_changes = [min(change, 360-change) for change in angle_changes]
            consistency = 1.0 - (np.std(angle_changes) / 90.0)  # Lower std = more consistent
            symmetry_strength = (symmetry_strength + consistency) / 2.0
        
        return octagonal_phase, symmetry_strength
    except Exception as e:
        print(f"calculate_octagonal_symmetry error: {e}")
        return None, 0.0

def detect_golden_ratio_patterns(prices, forecasted_max_price):
    """
    FORECASTING GOLDEN RATIO SYSTEM FOR REVERSAL SETUPS
    - Identifies the most recent significant low (argmin) as the starting point (0.000).
    - Uses a forecasted maximum price as the target (1.000).
    - Calculates 8 perfectly symmetrical, φ-based internal levels for the reversal.
    - This is a forward-looking system for identifying potential profit targets.
    """
    try:
        if len(prices) < 50:
            return None

        # Get the last 1200 values to find the most recent significant dip
        last_1200_prices = prices[-1200:] if len(prices) >= 1200 else prices
        
        # Find the most recent minimum (the dip) - this is our 0.000 level
        min_idx_local = np.argmin(last_1200_prices)
        swing_low = last_1200_prices[min_idx_local]
        
        # The forecasted maximum is our 1.000 level, passed in from the ML models
        forecasted_swing_high = forecasted_max_price
        
        # Ensure we have a valid forecasted range (the new high must be above the dip)
        if forecasted_swing_high <= swing_low:
            print(f"Warning: Forecasted high ({forecasted_swing_high}) is not above dip low ({swing_low}). Using current price + 5% as target.")
            forecasted_swing_high = prices[-1] * 1.05  # Fallback to 5% above current price

        fib_range = forecasted_swing_high - swing_low

        # PURE 0–1 INTERNAL GOLDEN RATIO LEVELS (8 levels, perfectly symmetrical)
        levels = {
            'Level_0.000': swing_low,                                          # Swing Low (Dip/Origin)
            'Level_0.146': swing_low + fib_range * 0.146,                      # φ⁻⁴ (deep)
            'Level_0.236': swing_low + fib_range * 0.236,                      # φ⁻³
            'Level_0.382': swing_low + fib_range * 0.382,                      # √φ
            'Level_0.500': swing_low + fib_range * 0.500,                      # Midpoint (balance)
            'Level_0.618': swing_low + fib_range * 0.618,                      # Golden Ratio (φ⁻¹) - The Golden Pocket
            'Level_0.786': swing_low + fib_range * 0.786,                      # √(φ²)
            'Level_1.000': forecasted_swing_high,                              # Forecasted Swing High (Target)
        }
        
        # Add context for the final report
        levels['info_forecast_source'] = 'ML Consensus Target'
        levels['info_swing_low_source'] = f'argmin of last {len(last_1200_prices)} 1m candles'

        return levels

    except Exception as e:
        print(f"Forecasting Golden Ratio Error: {e}")
        return None

def detect_golden_triangle(prices, timestamps):
    """
    Enhanced golden triangle detection with more sensitive criteria.
    Returns potential breakout direction and strength.
    """
    try:
        if len(prices) < 30:
            return None, 0.0
            
        # Find local minima and maxima
        local_minima = []
        local_maxima = []
        
        # Find local minima and maxima
        for i in range(1, len(prices)-1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                local_minima.append((i, prices[i]))
            elif prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                local_maxima.append((i, prices[i]))
        
        # If we don't have enough points, return None
        if len(local_minima) < 1 or len(local_maxima) < 2:
            return None, 0.0
            
        # Try to form triangles with these points
        best_score = 0.0
        best_dir = None
        
        for min_idx, min_val in local_minima:
            for max1_idx, max1_val in local_maxima:
                for max2_idx, max2_val in local_maxima:
                    # Ensure points are in chronological order
                    if not (min_idx < max1_idx < max2_idx):
                        continue
                        
                    # Calculate triangle sides using price differences instead of time
                    a = abs(max1_val - min_val)  # Price difference from min to first max
                    b = abs(max2_val - max1_val)  # Price difference between maxima
                    c = abs(max2_val - min_val)  # Price difference from min to second max
                    
                    # Check for golden ratio relationships
                    ratio1 = a / b if b > 0 else 0
                    ratio2 = b / a if a > 0 else 0
                    ratio3 = c / a if a > 0 else 0
                    ratio4 = a / c if c > 0 else 0
                    
                    # Check if any ratio is close to golden ratio
                    golden_ratios = [ratio1, ratio2, ratio3, ratio4]
                    for r in golden_ratios:
                        if abs(r - GOLDEN_RATIO) < 0.3 or abs(r - 0.618) < 0.3:  # Using 0.618 for price ratios
                            # Determine direction
                            direction = "upward" if max2_val > max1_val else "downward"
                            
                            # Calculate strength based on how close to golden ratio
                            strength = 1.0 - min(abs(r - GOLDEN_RATIO), abs(r - 0.618)) / 0.3
                            
                            # Prioritize upward triangles
                            if direction == "upward":
                                strength *= 1.2
                                
                            if strength > best_score:
                                best_score = strength
                                best_dir = direction

        # If no triangle found, try a simpler approach
        if best_dir is None:
            # Check for simple upward trend with golden ratio
            if len(prices) >= 20:
                start_price = prices[-20]
                mid_price = prices[-10]
                end_price = prices[-1]
                
                # Check if it's forming an upward pattern
                if end_price > mid_price > start_price:
                    # Calculate the ratios
                    ratio1 = (mid_price - start_price) / (end_price - start_price)
                    ratio2 = (end_price - mid_price) / (end_price - start_price)
                    
                    # Check if either ratio is close to golden ratio
                    for r in [ratio1, ratio2]:
                        if abs(r - 0.618) < 0.1:  # Using 0.618 instead of 1.618 for price ratios
                            best_dir = "upward"
                            best_score = 1.0 - abs(r - 0.618) / 0.1
                            break

        return best_dir, float(best_score) if best_score > 0 else 0.0
    except Exception as e:
        print(f"detect_golden_triangle error: {e}")
        return None, 0.0

def analyze_volume_spike(client, symbol):
    """
    Enhanced volume spike analysis for fast spike detection.
    Returns volume spike score and confidence.
    """
    try:
        # Get 1-minute data for detailed analysis
        klines = client.get_klines(symbol=symbol, interval='1m', limit=VOLUME_ANALYSIS_PERIOD)
        if not klines or len(klines) < VOLUME_ANALYSIS_PERIOD:
            return None
            
        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
        
        # Convert all numeric columns to float with robust error handling
        for c in ['open','high','low','close','volume','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume']:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except:
                df[c] = 0.0
        
        # Fill any NaN values that might have been created
        df.fillna(0.0, inplace=True)
        
        # Calculate volume metrics
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[:-5].mean()  # Exclude last 5 candles from average
        volume_spike_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Calculate price momentum
        price_change_5 = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] if len(df) >= 5 else 0
        price_change_10 = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] if len(df) >= 10 else 0
        
        # Calculate volume trend
        volume_trend = df['volume'].iloc[-5:].mean() / df['volume'].iloc[-10:-5].mean() if len(df) >= 10 else 1.0
        
        # Calculate buy/sell pressure
        buy_pressure = df['taker_buy_base_asset_volume'].iloc[-5:].sum()
        sell_pressure = (df['volume'].iloc[-5:].sum() - buy_pressure)
        buy_sell_ratio = buy_pressure / (buy_pressure + sell_pressure) if (buy_pressure + sell_pressure) > 0 else 0.5
        
        # Calculate bullish vs bearish volume
        bullish_volume = df['taker_buy_base_asset_volume'].iloc[-5:].sum()
        bearish_volume = (df['volume'].iloc[-5:].sum() - bullish_volume)
        total_volume = bullish_volume + bearish_volume
        
        bullish_volume_pct = (bullish_volume / total_volume * 100) if total_volume > 0 else 50.0
        bearish_volume_pct = (bearish_volume / total_volume * 100) if total_volume > 0 else 50.0
        
        # Calculate spike score
        volume_score = min(100, (volume_spike_ratio - 1.0) * 50) if volume_spike_ratio > 1.0 else 0
        momentum_score = min(100, (abs(price_change_5) + abs(price_change_10)) * 1000)
        trend_score = min(100, (volume_trend - 1.0) * 100) if volume_trend > 1.0 else 0
        pressure_score = abs(buy_sell_ratio - 0.5) * 200  # Deviation from balanced
        
        # Combined spike score
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
    """
    Enhanced pre-spike analysis using Hilbert Transform, FFT, and signal processing.
    Analyzes:
    1. Hilbert Transform Sine wave (HT_SINE) for phase analysis
    2. FFT-based forecasting from argmin to argmax
    3. Signal processing metrics: energy, frequency, vibration, power, angular momentum, pulse, impulse
    4. Golden ratio integration for volume analysis
    Returns a tuple of (is_valid, score, details)
    """
    try:
        if df is None or len(df) < 50:
            return False, 0.0, {"error": "Not enough data for pre-spike analysis"}
        
        # Initialize results
        conditions_met = 0
        total_score = 0.0
        details = {}
        
        # Extract price and volume data
        close_prices = df['close'].values
        volume_data = df['volume'].values
        timestamps = np.arange(len(close_prices))
        
        # Handle NaN and zero values
        close_prices = np.nan_to_num(close_prices, nan=np.mean(close_prices[~np.isnan(close_prices)]))
        close_prices = np.where(close_prices == 0, np.mean(close_prices[close_prices > 0]), close_prices)
        volume_data = np.nan_to_num(volume_data, nan=np.mean(volume_data[~np.isnan(volume_data)]))
        volume_data = np.where(volume_data == 0, np.mean(volume_data[volume_data > 0]), volume_data)
        
        # Normalize price data for signal processing
        norm_prices = (close_prices - np.mean(close_prices)) / np.std(close_prices)
        
        # 1. Hilbert Transform Analysis
        try:
            # Apply Hilbert Transform to get analytic signal
            analytic_signal = hilbert(norm_prices)
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
            
            # Calculate HT_SINE
            ht_sine = np.sin(instantaneous_phase)
            
            # Find most recent minimum (argmin) in price data
            argmin_idx = np.argmin(close_prices[-200:]) + len(close_prices) - 200
            
            # Get phase at minimum
            phase_at_min = instantaneous_phase[argmin_idx]
            
            # Current phase
            current_phase = instantaneous_phase[-1]
            
            # Calculate phase difference from minimum to current
            phase_diff = (current_phase - phase_at_min) % (2 * np.pi)
            
            # Phase condition: we're in upward phase (0 to π)
            phase_condition = 0 < phase_diff < np.pi
            
            # Calculate amplitude growth from minimum
            amplitude_at_min = amplitude_envelope[argmin_idx]
            current_amplitude = amplitude_envelope[-1]
            amplitude_growth = (current_amplitude - amplitude_at_min) / amplitude_at_min if amplitude_at_min > 0 else 0
            
            # Amplitude condition: amplitude is growing
            amplitude_condition = amplitude_growth > 0.1  # 10% growth threshold
            
            # Frequency condition: frequency is increasing (acceleration)
            recent_freq = np.mean(instantaneous_frequency[-10:])
            previous_freq = np.mean(instantaneous_frequency[-20:-10])
            freq_acceleration = (recent_freq - previous_freq) / previous_freq if previous_freq != 0 else 0
            freq_condition = freq_acceleration > 0.05  # 5% acceleration threshold
            
            # Overall Hilbert condition
            hilbert_condition = phase_condition and amplitude_condition and freq_condition
            
            if hilbert_condition:
                conditions_met += 1
                phase_score = 1.0 - (phase_diff / np.pi) if phase_diff < np.pi else 0  # Closer to 0 is better
                amp_score = min(1.0, amplitude_growth * 10)  # Scale amplitude growth
                freq_score = min(1.0, freq_acceleration * 20)  # Scale frequency acceleration
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
            # Get last 1200 values or all if less than 1200
            last_1200_prices = close_prices[-1200:] if len(close_prices) >= 1200 else close_prices
            
            # Find most recent minimum (argmin) in price data
            argmin_idx = np.argmin(last_1200_prices)
            
            # Extract data from minimum to now
            segment_prices = last_1200_prices[argmin_idx:]
            
            # Apply FFT to find dominant frequencies
            n = len(segment_prices)
            if n < FFT_MIN_LENGTH:
                fft_condition = False
                fft_score = 0
            else:
                # Detrend data
                trend = np.polyfit(np.arange(n), segment_prices, 1)
                detrended = segment_prices - np.polyval(trend, np.arange(n))
                
                # Apply FFT
                yf = fft(detrended)
                xf = fftfreq(n, d=1.0)
                
                # Find dominant frequency (excluding zero frequency)
                half = n // 2
                mag = np.abs(yf[:half])
                freqs = xf[:half]
                mag[0] = 0  # Remove DC component
                
                if np.max(mag) < 1e-6:
                    fft_condition = False
                    fft_score = 0
                else:
                    # Get dominant frequency and amplitude
                    idx = np.argmax(mag[1:]) + 1
                    dominant_freq = freqs[idx]
                    amplitude = mag[idx] / n
                    
                    # Calculate phase
                    phase = np.angle(yf[idx])
                    
                    # Calculate next expected peak
                    period = 1.0 / abs(dominant_freq) if dominant_freq != 0 else 0
                    current_time = n - 1
                    
                    # Time since last minimum
                    time_since_min = current_time
                    
                    # Determine if we're heading to a peak
                    phase_position = (time_since_min / period) % 1.0 if period > 0 else 0
                    
                    # Next peak at 0.25, trough at 0.75
                    next_peak_time = period * 0.25
                    next_trough_time = period * 0.75
                    
                    # Determine which comes next
                    if current_time < next_peak_time:
                        next_extremum_type = "peak"
                        time_to_next = next_peak_time - current_time
                    elif current_time < next_trough_time:
                        next_extremum_type = "trough"
                        time_to_next = next_trough_time - current_time
                    else:
                        next_extremum_type = "peak"
                        time_to_next = period * 1.25 - current_time
                    
                    # FFT condition: we're heading to a peak within a reasonable time
                    fft_condition = next_extremum_type == "peak" and 0 < time_to_next < period * 0.5
                    
                    # Calculate FFT score
                    time_score = 1.0 - (time_to_next / (period * 0.5)) if period > 0 else 0
                    amp_score = min(1.0, amplitude * 1000)  # Scale amplitude
                    fft_score = (time_score * 0.6 + amp_score * 0.4) * 25
                    
                    # Store FFT details
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
            # Calculate signal energy (sum of squared values)
            signal_energy = np.sum(norm_prices ** 2)
            
            # Calculate signal power (energy per unit time)
            signal_power = signal_energy / len(norm_prices)
            
            # Calculate dominant frequency using FFT
            n = len(norm_prices)
            yf = fft(norm_prices)
            xf = fftfreq(n, d=1.0)
            half = n // 2
            mag = np.abs(yf[:half])
            freqs = xf[:half]
            dominant_freq_idx = np.argmax(mag[1:]) + 1
            dominant_freq = abs(freqs[dominant_freq_idx])
            
            # Calculate vibration (standard deviation of the signal)
            vibration = np.std(norm_prices)
            
            # Calculate angular momentum (phase velocity * amplitude)
            analytic_signal = hilbert(norm_prices)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
            amplitude_envelope = np.abs(analytic_signal)
            angular_momentum = np.mean(amplitude_envelope[:-1] * instantaneous_frequency)
            
            # Calculate pulse (rate of significant changes)
            threshold = np.std(norm_prices) * 2
            significant_changes = np.where(np.abs(np.diff(norm_prices)) > threshold)[0]
            pulse_rate = len(significant_changes) / len(norm_prices)
            
            # Calculate impulse (sudden changes in momentum)
            momentum = np.diff(norm_prices)
            impulse = np.sum(np.abs(np.diff(momentum)))
            
            # Calculate signal processing score
            # Higher energy, power, and angular momentum are positive indicators
            # Moderate vibration and pulse rate are ideal
            # High impulse indicates potential breakout
            
            energy_score = min(1.0, signal_energy / 1000)  # Normalize
            power_score = min(1.0, signal_power * 10)  # Normalize
            freq_score = min(1.0, dominant_freq * 100)  # Normalize
            vibration_score = 1.0 - min(1.0, abs(vibration - 1.0))  # Ideal around 1.0
            angular_momentum_score = min(1.0, angular_momentum * 10)  # Normalize
            pulse_score = 1.0 - min(1.0, abs(pulse_rate - 0.1) * 10)  # Ideal around 0.1
            impulse_score = min(1.0, impulse / 10)  # Normalize
            
            # Overall signal processing condition
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
            # Get the last 1200 values or all if less than 1200
            last_1200_prices = close_prices[-1200:] if len(close_prices) >= 1200 else close_prices
            last_1200_volumes = volume_data[-1200:] if len(volume_data) >= 1200 else volume_data
            
            # Find the most recent minimum and maximum in the price data
            min_idx = np.argmin(last_1200_prices)
            max_idx = np.argmax(last_1200_prices)
            
            # Calculate price range
            min_price = last_1200_prices[min_idx]
            max_price = last_1200_prices[max_idx]
            price_range = max_price - min_price
            
            # Calculate golden ratio levels for volume
            # We expect volume to follow golden ratio patterns relative to price movement
            min_volume = np.min(last_1200_volumes)
            max_volume = np.max(last_1200_volumes)
            volume_range = max_volume - min_volume
            
            # Calculate current position in price range (0 to 1)
            current_price = last_1200_prices[-1]
            price_position = (current_price - min_price) / price_range if price_range > 0 else 0.5
            
            # Calculate current position in volume range (0 to 1)
            current_volume = last_1200_volumes[-1]
            volume_position = (current_volume - min_volume) / volume_range if volume_range > 0 else 0.5
            
            # Golden ratio levels (0, 0.236, 0.382, 0.618, 0.786, 1.0)
            golden_levels = [0.0, 0.236, 0.382, 0.618, 0.786, 1.0]
            
            # Find closest golden level to current price position
            closest_price_level = min(golden_levels, key=lambda x: abs(x - price_position))
            price_level_diff = abs(price_position - closest_price_level)
            
            # Find closest golden level to current volume position
            closest_volume_level = min(golden_levels, key=lambda x: abs(x - volume_position))
            volume_level_diff = abs(volume_position - closest_volume_level)
            
            # Check if price and volume are aligned with golden ratio
            alignment_score = 1.0 - (price_level_diff + volume_level_diff) / 2.0
            
            # Check if we're at a golden ratio support/resistance level
            at_golden_level = price_level_diff < 0.05  # Within 5% of a golden level
            
            # Check if volume is confirming price movement
            volume_confirmation = (
                (price_position > 0.5 and volume_position > 0.5) or  # Both above middle
                (price_position < 0.5 and volume_position < 0.5)      # Both below middle
            )
            
            # Overall golden ratio condition
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
        
        # Determine if spike is imminent based on conditions met
        spike_imminent = conditions_met >= 3  # At least 3 of 4 conditions
        
        return spike_imminent, total_score, details
    except Exception as e:
        print(f"analyze_pre_spike_conditions error: {e}")
        return False, 0.0, {"error": str(e)}

def calculate_time_to_target(current_price, target_price, cycle_period, current_phase, octagonal_phase, volume_spike_data, timeframe='1m', price_history=None):
    """
    Enhanced time to target calculation with volume spike consideration.
    Returns estimated time in seconds and confidence.
    """
    try:
        if cycle_period <= 0 or current_price <= 0 or target_price <= 0:
            return None, 0.0
            
        # Calculate price distance as percentage
        price_distance_pct = abs(target_price - current_price) / current_price
        
        # Convert cycle period from data points to actual time in seconds
        # This depends on the timeframe of the data
        timeframe_seconds = {
            '1m': 60,
            '3m': 180,
            '5m': 300
        }
        
        # Get the seconds per data point based on timeframe
        seconds_per_point = timeframe_seconds.get(timeframe, 60)  # Default to 1m
        
        # Convert cycle period to seconds
        cycle_period_seconds = cycle_period * seconds_per_point
        
        # Enhanced base time calculation with historical volatility consideration
        base_time = cycle_period_seconds * 0.5
        
        # Adjust for price distance - larger movements take proportionally more time
        distance_factor = 1.0 + min(2.0, price_distance_pct * 10)  # Cap at 3x base time
        
        # Calculate historical volatility if price history is available
        volatility_factor = 1.0
        if price_history is not None and len(price_history) > 20:
            returns = np.diff(price_history) / price_history[:-1]
            volatility = np.std(returns)
            # Higher volatility means faster price movements
            volatility_factor = 1.0 / (1.0 + volatility * 50)  # Normalize volatility impact
        
        # Adjust based on octagonal phase
        # Phases 0 and 1 (uptrend) are fastest, 4 and 5 (downtrend) are slowest
        phase_speed_factor = 1.0
        if octagonal_phase is not None:
            if octagonal_phase in [0, 1]:  # Fast uptrend
                phase_speed_factor = 0.7
            elif octagonal_phase in [2, 3]:  # Moderate uptrend
                phase_speed_factor = 0.85
            elif octagonal_phase in [4, 5]:  # Moderate downtrend
                phase_speed_factor = 1.15
            else:  # Fast downtrend
                phase_speed_factor = 1.3
        
        # Adjust based on volume spike
        volume_factor = 1.0
        if volume_spike_data:
            volume_spike_ratio = volume_spike_data.get('volume_spike_ratio', 1.0)
            buy_sell_ratio = volume_spike_data.get('buy_sell_ratio', 0.5)
            
            # High volume spike and buy pressure reduce time to target
            if volume_spike_ratio > VOLUME_SPIKE_THRESHOLD and buy_sell_ratio > 0.6:
                volume_factor = 0.6  # 40% faster with strong volume spike and buy pressure
            elif volume_spike_ratio > VOLUME_SPIKE_THRESHOLD:
                volume_factor = 0.8  # 20% faster with volume spike
        
        # Calculate adjusted time to target
        adjusted_time = base_time * distance_factor * volatility_factor * phase_speed_factor * volume_factor
        
        # Ensure time is within reasonable bounds
        adjusted_time = max(MIN_TIME_TO_TARGET, min(MAX_TIME_TO_TARGET, adjusted_time))
        
        # Enhanced confidence calculation
        phase_alignment = 1.0 - abs(current_phase - 0.5) * 2  # 1.0 at peak, 0.0 at trough
        octagonal_alignment = 1.0 - abs(octagonal_phase - OCTAGONAL_SEGMENTS/2) / (OCTAGONAL_SEGMENTS/2)
        
        volume_confidence = 0.5
        if volume_spike_data:
            volume_confidence = min(1.0, volume_spike_data.get('volume_spike_ratio', 1.0) / VOLUME_SPIKE_THRESHOLD)
        
        # Price distance confidence - closer targets are more confident
        distance_confidence = 1.0 - min(1.0, price_distance_pct * 5)  # Scale down confidence for distant targets
        
        # Historical pattern confidence - check if similar price movements occurred before
        pattern_confidence = 0.5
        if price_history is not None and len(price_history) > 50:
            # Look for similar price movements in history
            target_change = (target_price - current_price) / current_price
            historical_changes = []
            for i in range(20, len(price_history)):
                change = (price_history[i] - price_history[i-20]) / price_history[i-20]
                historical_changes.append(change)
            
            if historical_changes:
                # Count how many times similar or larger changes occurred
                similar_changes = sum(1 for change in historical_changes if abs(change) >= abs(target_change))
                pattern_confidence = min(1.0, similar_changes / len(historical_changes) * 2)
        
        # Combine all confidence factors
        confidence = (
            phase_alignment * 0.20 + 
            octagonal_alignment * 0.20 + 
            volume_confidence * 0.25 + 
            distance_confidence * 0.15 + 
            pattern_confidence * 0.20
        )
        
        # Ensure confidence is within 0-1 range with minimum of 0.1
        confidence = max(0.1, min(1.0, confidence))
        
        return adjusted_time, confidence
    except Exception as e:
        print(f"calculate_time_to_target error: {e}")
        return None, 0.0

def analyze_sinuosidal_pattern(prices, timestamps, timeframe='1m'):
    """
    Enhanced sinusoidal pattern analysis with next extremum prediction.
    Returns amplitude, frequency, phase, and next expected peak/trough.
    """
    try:
        if len(prices) < FFT_MIN_LENGTH:
            return None, None, None, None, None
            
        # Find major minima
        minima_indices = []
        for i in range(1, len(prices)-1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                minima_indices.append(i)
                
        if len(minima_indices) < 2:
            return None, None, None, None, None
            
        # Use the last major minimum as starting point
        last_min_idx = minima_indices[-1]
        last_min_price = prices[last_min_idx]
        last_min_time = timestamps[last_min_idx]
        
        # Extract data from last minimum to now
        segment_prices = prices[last_min_idx:]
        segment_times = timestamps[last_min_idx:]
        
        # Detrend the data
        trend = np.polyfit(segment_times, segment_prices, 1)
        detrended = segment_prices - np.polyval(trend, segment_times)
        
        # Apply FFT to find dominant frequency
        n = len(detrended)
        yf = fft(detrended)
        xf = fftfreq(n, d=1.0)  # Using 1.0 as d since timestamps are just indices
        
        # Find dominant frequency (excluding zero frequency)
        half = n // 2
        mag = np.abs(yf[:half])
        freqs = xf[:half]
        mag[0] = 0  # Remove DC component
        
        if np.max(mag) < 1e-6:
            return None, None, None, None, None
            
        # Get dominant frequency and amplitude
        idx = np.argmax(mag[1:]) + 1
        dominant_freq = freqs[idx]
        amplitude = mag[idx] / n
        
        # Calculate phase
        phase = np.angle(yf[idx])
        
        # Calculate next expected peak or trough
        period = 1.0 / abs(dominant_freq) if dominant_freq != 0 else 0
        current_time = segment_times[-1]
        
        # Time since last minimum
        time_since_min = current_time - last_min_time
        
        # Determine if we're heading to a peak or trough
        # Using the phase to determine position in the cycle
        phase_position = (time_since_min / period) % 1.0 if period > 0 else 0
        
        # Next peak at 0.25, trough at 0.75
        next_peak_time = last_min_time + period * 0.25
        next_trough_time = last_min_time + period * 0.75
        
        # Determine which comes next
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
    """
    Keops' time-based pivot approach using phi powers.
    - Identifies pivot points based on phi ratios
    - Sets the first high/low or low/high leg at phi^-4 ratio
    - Uses phi powers for time and price analysis
    Returns a dictionary with pivot points and phi levels.
    """
    try:
        if len(prices) < 50:
            return None
        
        # Define phi powers
        phi_powers = {
            'phi^-4': 0.1458980337503153,  # 1/phi^4
            'phi^-3': 0.2360679774997897,  # 1/phi^3
            'phi^-2': 0.3819660112501051,  # 1/phi^2
            'phi^-1': 0.6180339887498948,  # 1/phi
            '1-phi^-3': 0.7639320225002103,  # 1 - 1/phi^3
            '1-phi^-4': 0.8541019662496847   # 1 - 1/phi^4
        }
        
        # Find significant pivot points (local minima and maxima)
        pivots = []
        for i in range(1, len(prices)-1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                pivots.append((i, prices[i], 'low'))
            elif prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                pivots.append((i, prices[i], 'high'))
        
        if len(pivots) < 2:
            return None
        
        # Find the most recent significant leg (high-low or low-high)
        recent_legs = []
        for i in range(len(pivots)-1):
            if pivots[i][2] != pivots[i+1][2]:  # Different pivot types
                recent_legs.append((pivots[i], pivots[i+1]))
        
        if not recent_legs:
            return None
        
        # Get the most recent leg
        last_leg = recent_legs[-1]
        start_pivot = last_leg[0]
        end_pivot = last_leg[1]
        
        # Check if the most recent extrema was an argmin or argmax in the last 1200 values
        last_1200_prices = prices[-1200:] if len(prices) >= 1200 else prices
        last_1200_min_idx = np.argmin(last_1200_prices)
        last_1200_max_idx = np.argmax(last_1200_prices)
        
        # Convert to global indices
        global_min_idx = last_1200_min_idx + (len(prices) - len(last_1200_prices))
        global_max_idx = last_1200_max_idx + (len(prices) - len(last_1200_prices))
        
        # Determine if the most recent extrema was a min or max
        most_recent_extrema_idx = max(global_min_idx, global_max_idx)
        most_recent_extrema_type = 'min' if most_recent_extrema_idx == global_min_idx else 'max'
        
        # Determine if it's an up leg or down leg based on most recent extrema
        # If the most recent extrema was a minimum (argmin), then we're in an up cycle
        # If the most recent extrema was a maximum (argmax), then we're in a down cycle
        is_up_leg = most_recent_extrema_type == 'min'
        
        # Calculate the price range of the leg
        leg_range = abs(end_pivot[1] - start_pivot[1])
        
        # Set the first pivot point at phi^-4 ratio
        if is_up_leg:
            # For up leg, the low point is at phi^-4
            phi_pivot_price = start_pivot[1]
            phi_pivot_time = timestamps[start_pivot[0]]
        else:
            # For down leg, the high point is at phi^-4
            phi_pivot_price = start_pivot[1]
            phi_pivot_time = timestamps[start_pivot[0]]
        
        # Calculate phi levels for the leg
        phi_levels = {}
        for name, ratio in phi_powers.items():
            if is_up_leg:
                # For up leg, calculate levels above the low
                level_price = phi_pivot_price + (leg_range * ratio)
            else:
                # For down leg, calculate levels below the high
                level_price = phi_pivot_price - (leg_range * ratio)
            phi_levels[name] = level_price
        
        # Calculate time projections using phi ratios
        time_range = timestamps[end_pivot[0]] - timestamps[start_pivot[0]]
        phi_time_levels = {}
        for name, ratio in phi_powers.items():
            phi_time_levels[name] = phi_pivot_time + (time_range * ratio)
        
        # Determine the current position relative to phi levels
        current_price = prices[-1]
        current_time = timestamps[-1]
        
        # Find which phi level is closest to current price
        closest_price_level = min(phi_levels.items(), key=lambda x: abs(x[1] - current_price))
        
        # Find which phi time level is closest to current time
        closest_time_level = min(phi_time_levels.items(), key=lambda x: abs(x[1] - current_time))
        
        # Calculate the next expected pivot based on phi time levels
        future_pivot_times = []
        for name, time_val in phi_time_levels.items():
            if time_val > current_time:
                future_pivot_times.append((name, time_val))
        
        # Sort by time
        future_pivot_times.sort(key=lambda x: x[1])
        
        # Calculate golden ratio target price based on the leg type
        golden_ratio_target = None
        if is_up_leg:
            # For up leg, target is above the high
            golden_ratio_target = end_pivot[1] + (leg_range * GOLDEN_RATIO)
        else:
            # For down leg, target is below the low
            golden_ratio_target = end_pivot[1] - (leg_range * GOLDEN_RATIO)
        
        # Return the analysis results
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
    """Robust ATR calculation; returns DataFrame with 'ATR_{period}' column or None."""
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
        
        # Use pandas_ta if available
        if hasattr(df, 'ta'):
            try:
                df.ta.rsi(length=period, append=True)
                rsi_col = f'RSI_{period}'
                if rsi_col in df.columns:
                    return df
            except Exception:
                pass
        
        # Manual calculation if pandas_ta fails
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
        
        # Calculate SMA7 (changed from MA7)
        df['SMA7'] = df['close'].rolling(window=SMA7_PERIOD).mean()
        
        # Calculate other SMAs
        df['SMA12'] = df['close'].rolling(window=SMA12_PERIOD).mean()
        df['SMA27'] = df['close'].rolling(window=SMA27_PERIOD).mean()
        df['SMA56'] = df['close'].rolling(window=SMA56_PERIOD).mean()
        df['SMA150'] = df['close'].rolling(window=SMA150_PERIOD).mean()
        
        return df
    except Exception as e:
        print(f"calculate_moving_averages error: {e}")
        return None

def check_polynomial_fit(prices, timestamps):
    """
    Check if current price is below polynomial fit line.
    Returns boolean indicating if condition is met.
    """
    try:
        if len(prices) < 10:
            return False
            
        x = timestamps
        y = prices
        
        # Fit polynomial of degree 1 (linear regression)
        best_fit_line1 = np.poly1d(np.polyfit(x, y, 1))(x)
        best_fit_line2 = best_fit_line1 * 1.01  # 1% above
        best_fit_line3 = best_fit_line1 * 0.99  # 1% below
        
        # Check if current price is below best_fit_line3
        if y[-1] < best_fit_line3[-1]:
            return True
        
        return False
    except Exception as e:
        print(f"check_polynomial_fit error: {e}")
        return False

def analyze_rsi_conditions(rsi_values, current_rsi):
    """
    Enhanced RSI analysis with two separate conditions:
    1. Is the most recent RSI value oversold (true/false)
    2. Is the most recent RSI value overbought (true/false)
    Returns a tuple of (is_oversold, is_overbought, score, details)
    """
    try:
        if rsi_values is None or len(rsi_values) < 100:
            return False, False, 0.0, {"error": "Not enough RSI data"}
        
        # Find the most recent oversold and overbought occurrences
        last_oversold_idx = None
        last_overbought_idx = None
        
        for i in range(len(rsi_values) - 1, -1, -1):
            if last_oversold_idx is None and rsi_values[i] <= RSI_OVERSOLD:
                last_oversold_idx = i
            if last_overbought_idx is None and rsi_values[i] >= RSI_OVERBOUGHT:
                last_overbought_idx = i
                
            if last_oversold_idx is not None and last_overbought_idx is not None:
                break
        
        # Check if current RSI is oversold or overbought
        is_oversold = current_rsi <= RSI_OVERSOLD
        is_overbought = current_rsi >= RSI_OVERBOUGHT
        
        # Check if oversold is the most recent occurrence
        oversold_is_most_recent = False
        if last_oversold_idx is not None:
            oversold_is_most_recent = (last_oversold_idx > last_overbought_idx) if last_overbought_idx is not None else True
        
        # Check if overbought is the most recent occurrence
        overbought_is_most_recent = False
        if last_overbought_idx is not None:
            overbought_is_most_recent = (last_overbought_idx > last_oversold_idx) if last_oversold_idx is not None else True
        
        # Calculate score based on how oversold or overbought RSI is
        oversold_score = 0.0
        if is_oversold:
            # More oversold = higher score
            oversold_score = (RSI_OVERSOLD - current_rsi) / RSI_OVERSOLD * 50
        else:
            # If not oversold, give partial score based on how close to oversold
            oversold_score = max(0, (RSI_OVERSOLD - current_rsi) / RSI_OVERSOLD * 30)
        
        overbought_score = 0.0
        if is_overbought:
            # More overbought = higher score
            overbought_score = (current_rsi - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT) * 50
        else:
            # If not overbought, give partial score based on how close to overbought
            overbought_score = max(0, (current_rsi - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT) * 30)
        
        # Combined score
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
    """
    Enhanced price dip analysis using argmin vs argmax of last 1200 values.
    Returns a tuple of (is_valid, score, details)
    """
    try:
        if prices is None or len(prices) < 100:
            return False, 0.0, {"error": "Not enough price data"}
            
        # Get the last 1200 values or all if less than 1200
        last_1200_values = prices[-1200:] if len(prices) >= 1200 else prices
        
        # Find the index of the absolute minimum and maximum in the last 1200 values
        min_idx = np.argmin(last_1200_values)
        max_idx = np.argmax(last_1200_values)
        
        # Check if the minimum is more recent than the maximum
        if min_idx < max_idx:
            return False, 0.0, {"reason": "Most recent minimum is not more recent than most recent maximum"}
        
        # Calculate how much the current price is above the minimum
        min_price = last_1200_values[min_idx]
        max_price = last_1200_values[max_idx]
        
        # Calculate score based on position between min and max
        # If current price is closer to min, score is higher
        price_range = max_price - min_price
        if price_range <= 0:
            return False, 0.0, {"error": "Invalid price range"}
            
        position_in_range = (current_price - min_price) / price_range
        score = (1.0 - position_in_range) * 100  # Closer to min = higher score
        
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
    """
    Enhanced momentum analysis using TALIB's MOM function.
    Returns a tuple of (is_valid, score, details)
    """
    try:
        if df is None or len(df) < 100:
            return False, 0.0, {"error": "Not enough data for momentum analysis"}
        
        # Extract price data
        close_prices = df['close'].values
        
        # Calculate momentum using TALIB's MOM function
        try:
            momentum_values = talib.MOM(close_prices, timeperiod=10)  # 10-period momentum
        except Exception as e:
            print(f"TALIB MOM error: {e}")
            # Fallback to manual calculation
            momentum_values = np.diff(close_prices, n=10)  # 10-period price change
            momentum_values = np.concatenate([np.zeros(10), momentum_values])  # Pad with zeros
        
        # Handle NaN values
        momentum_values = np.nan_to_num(momentum_values, nan=0.0)
        
        # Get the last 1200 values of price and momentum
        last_1200_prices = close_prices[-1200:] if len(close_prices) >= 1200 else close_prices
        last_1200_momentum = momentum_values[-1200:] if len(momentum_values) >= 1200 else momentum_values
        
        # Find the index of the absolute minimum and maximum in the last 1200 PRICES
        min_price_idx = np.argmin(last_1200_prices)
        max_price_idx = np.argmax(last_1200_prices)
        
        # Get the most negative and most positive momentum values
        min_momentum_idx = np.argmin(last_1200_momentum)
        max_momentum_idx = np.argmax(last_1200_momentum)
        most_negative_momentum = last_1200_momentum[min_momentum_idx]
        most_positive_momentum = last_1200_momentum[max_momentum_idx]
        
        # Current momentum value
        current_momentum = momentum_values[-1]
        
        # Condition 1: Current momentum > 0
        momentum_positive = current_momentum > 0
        
        # Check if the current momentum corresponds to a price minimum or maximum
        # Find the index of the current price in the last 1200 values
        current_price_idx = len(last_1200_prices) - 1  # Current price is the last element
        
        # Check if current momentum aligns with a price reversal
        # If we're at a price minimum, momentum should be negative (starting to turn up)
        # If we're at a price maximum, momentum should be positive (starting to turn down)
        
        # Check if current momentum is at or near a price minimum
        price_at_min = last_1200_prices[min_price_idx]
        price_at_max = last_1200_prices[max_price_idx]
        current_price = last_1200_prices[current_price_idx]
        
        # Calculate how close current price is to the minimum and maximum
        pct_from_min = ((current_price - price_at_min) / price_at_min) * 100 if price_at_min > 0 else 0
        pct_from_max = ((price_at_max - current_price) / price_at_max) * 100 if price_at_max > 0 else 0
        
        # Determine if we're closer to a minimum or maximum
        is_near_min = pct_from_min < pct_from_max
        
        # Check if the momentum direction aligns with the price position
        # If near a price minimum, momentum should be negative (about to turn positive)
        # If near a price maximum, momentum should be positive (about to turn negative)
        
        # Check if the current momentum is the most negative or most positive in the recent window
        recent_window = 20
        if len(momentum_values) < recent_window:
            recent_window = len(momentum_values)
        
        recent_momentum = momentum_values[-recent_window:]
        is_most_recent_argmin = current_momentum <= np.min(recent_momentum)
        is_most_recent_argmax = current_momentum >= np.max(recent_momentum)
        
        # Check if we're at a reversal point
        # If the most recent price minimum is more recent than the most recent price maximum
        # and we're near the minimum with negative momentum, that's a bullish reversal
        # If the most recent price maximum is more recent than the most recent price minimum
        # and we're near the maximum with positive momentum, that's a bearish reversal
        
        min_price_more_recent = min_price_idx > max_price_idx
        max_price_more_recent = max_price_idx > min_price_idx
        
        # Determine if we're at a bullish or bearish reversal
        is_bullish_reversal = min_price_more_recent and is_near_min and current_momentum < 0
        is_bearish_reversal = max_price_more_recent and not is_near_min and current_momentum > 0
        
        # Condition 2: Most recent momentum minimum value is more recent than the most recent momentum maximum value
        min_more_recent_than_max = min_momentum_idx > max_momentum_idx
        
        # Calculate score based on how well conditions are met
        score = 0.0
        
        # Score based on how high current momentum is (higher is better)
        momentum_score = min(100, current_momentum * 1000)  # Scale momentum to score
        
        # Score based on how recent the minimum is compared to maximum
        if min_momentum_idx is not None and max_momentum_idx is not None:
            min_max_score = 1.0 - ((max_momentum_idx - min_momentum_idx) / len(last_1200_momentum))
            score += min_max_score * 50
        
        # Bonus score for being at a reversal point
        if is_bullish_reversal or is_bearish_reversal:
            score += 25
        
        # Ensure score is between 0 and 100
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
        # Get more data for 1min timeframe for better analysis
        limit = 500 if timeframe == '1m' else 200
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
        if not klines or len(klines) < 20:
            return None

        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
        
        # Convert all numeric columns to float with robust error handling
        for c in ['open','high','low','close','volume','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore']:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except:
                df[c] = 0.0
        
        # Fill any NaN values that might have been created
        df.fillna(0.0, inplace=True)

        df = calculate_atr(df, ATR_PERIOD)
        if df is None or f'ATR_{ATR_PERIOD}' not in df.columns:
            return None

        current_atr = float(df[f'ATR_{ATR_PERIOD}'].iloc[-1])
        current_price = float(df['close'].iloc[-1])

        # Calculate price change metrics with CLEAR naming
        price_change_1period = 0.0
        price_change_3periods = 0.0  
        price_change_5periods = 0.0
        volume_change_1period = 0.0
        volume_change_3periods = 0.0
        volume_change_5periods = 0.0
        
        if len(df) >= 2:
            # 1-period change (most recent)
            price_change_1period = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100 if df['close'].iloc[-2] > 0 else 0
            volume_change_1period = ((df['volume'].iloc[-1] - df['volume'].iloc[-2]) / df['volume'].iloc[-2]) * 100 if df['volume'].iloc[-2] > 0 else 0
        
        if len(df) >= 4:
            # 3-period change
            price_change_3periods = ((df['close'].iloc[-1] - df['close'].iloc[-4]) / df['close'].iloc[-4]) * 100 if df['close'].iloc[-4] > 0 else 0
            volume_change_3periods = ((df['volume'].iloc[-1] - df['volume'].iloc[-4]) / df['volume'].iloc[-4]) * 100 if df['volume'].iloc[-4] > 0 else 0
        
        if len(df) >= 6:
            # 5-period change
            price_change_5periods = ((df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6]) * 100 if df['close'].iloc[-6] > 0 else 0
            volume_change_5periods = ((df['volume'].iloc[-1] - df['volume'].iloc[-6]) / df['volume'].iloc[-6]) * 100 if df['volume'].iloc[-6] > 0 else 0

        # VPA Analysis for THIS timeframe
        vpa_dip, vpa_breakout, vpa_score = analyze_volume_price_analysis(df)

        # Rest of dip detection logic
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
            # CLEAR naming: timeframe + periods
            'price_change_1period_pct': price_change_1period,
            'price_change_3periods_pct': price_change_3periods, 
            'price_change_5periods_pct': price_change_5periods,
            'volume_change_1period_pct': volume_change_1period,
            'volume_change_3periods_pct': volume_change_3periods,
            'volume_change_5periods_pct': volume_change_5periods,
            # VPA metrics for this timeframe
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
            
            # Convert all numeric columns to float with error handling
            for c in ['open','high','low','close','volume','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore']:
                try:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                except:
                    df[c] = 0.0
            
            # Fill any NaN values that might have been created
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
        
        # Select columns to display with CLEAR naming
        display_columns = [
            'symbol', 'current_price', 
            # 1m timeframe metrics
            '1m_price_change_1period_pct', '1m_price_change_3periods_pct', '1m_price_change_5periods_pct',
            '1m_volume_change_1period_pct', '1m_volume_change_3periods_pct', '1m_volume_change_5periods_pct',
            '1m_vpa_score', '1m_vpa_conditions_met',
            # 3m timeframe metrics  
            '3m_price_change_1period_pct', '3m_price_change_3periods_pct', '3m_price_change_5periods_pct',
            '3m_volume_change_1period_pct', '3m_volume_change_3periods_pct', '3m_volume_change_5periods_pct',
            '3m_vpa_score', '3m_vpa_conditions_met',
            # 5m timeframe metrics
            '5m_price_change_1period_pct', '5m_price_change_3periods_pct', '5m_price_change_5periods_pct', 
            '5m_volume_change_1period_pct', '5m_volume_change_3periods_pct', '5m_volume_change_5periods_pct',
            '5m_vpa_score', '5m_vpa_conditions_met',
            # Combined metrics
            'avg_vpa_score', 'vpa_conditions_met_count',
            'spike_score', 'volume_spike_ratio', 'buy_sell_ratio',
            'weighted_dip_score', 'power_score'
        ]
        
        # Filter to available columns
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

    mtf_thresholds = get_mtf_thresholds(client, symbol)
    try:
        klines = client.get_klines(symbol=symbol, interval='1m', limit=MAX_KLINES_LIMIT)
        if not klines:
            print("Failed to fetch historical data. Aborting.")
            return
        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
        
        # Convert all numeric columns to float with error handling
        for c in ['open','high','low','close','volume','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore']:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except:
                df[c] = 0.0
        
        # Fill any NaN values that might have been created
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
    price_history = df['close'].values  # Store for time-to-target calculation
    
    # --- ML models must run BEFORE we can calculate the golden ratio levels ---
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
    
    # Ensure minimum 1.5% profit target
    min_target = current_price * 1.015  # 1.5% minimum
    if consensus_target < min_target:
        consensus_target = min_target
        potential_change_pct = ((consensus_target - current_price) / current_price) * 100

    # --- NOW, calculate golden ratio levels using the forecasted consensus target ---
    print("Performing enhanced geometric analysis...")
    octagonal_phase, octagonal_strength = calculate_octagonal_symmetry(df['close'].values, timestamps)
    golden_levels = detect_golden_ratio_patterns(df['close'].values, consensus_target)
    triangle_direction, triangle_strength = detect_golden_triangle(df['close'].values, timestamps)
    sin_amplitude, sin_freq, sin_phase, next_extremum, sin_period = analyze_sinuosidal_pattern(df['close'].values, timestamps, '1m')
    
    # Get volume spike data
    volume_spike_data = analyze_volume_spike(client, symbol)
    
    # Analyze pre-spike conditions with new enhanced function
    pre_spike_valid, pre_spike_score, pre_spike_details = analyze_pre_spike_conditions(df)
    
    # --- VPA Analysis - Integrated with other conditions ---
    print("Performing enhanced VPA analysis...")
    vpa_dip_signals, vpa_breakout_signals, vpa_score = analyze_volume_price_analysis(df)
    
    # --- Keops' Phi Pivot Analysis ---
    keops_phi_pivots = detect_keops_phi_pivots(df['close'].values, timestamps)
    
    # Calculate moving averages
    df_ma = calculate_moving_averages(df)
    sma7 = float(df_ma['SMA7'].iloc[-1]) if df_ma is not None and 'SMA7' in df_ma.columns else None
    sma12 = float(df_ma['SMA12'].iloc[-1]) if df_ma is not None and 'SMA12' in df_ma.columns else None
    sma27 = float(df_ma['SMA27'].iloc[-1]) if df_ma is not None and 'SMA27' in df_ma.columns else None
    sma56 = float(df_ma['SMA56'].iloc[-1]) if df_ma is not None and 'SMA56' in df_ma.columns else None
    sma150 = float(df_ma['SMA150'].iloc[-1]) if df_ma is not None and 'SMA150' in df_ma.columns else None
    
    # Check polynomial fit
    is_below_poly_fit = check_polynomial_fit(df['close'].values, timestamps)
    
    # Calculate RSI
    df_rsi = calculate_rsi(df, RSI_PERIOD)
    current_rsi = float(df_rsi[f'RSI_{RSI_PERIOD}'].iloc[-1]) if df_rsi is not None and f'RSI_{RSI_PERIOD}' in df_rsi.columns else 50.0
    
    # Enhanced RSI analysis with new conditions
    rsi_values = df_rsi[f'RSI_{RSI_PERIOD}'].values if df_rsi is not None and f'RSI_{RSI_PERIOD}' in df_rsi.columns else None
    is_oversold, is_overbought, rsi_score, rsi_details = analyze_rsi_conditions(rsi_values, current_rsi)
    
    # Enhanced price dip analysis
    price_valid, price_score, price_details = analyze_price_dip_conditions(df['close'].values, current_price)
    
    # Enhanced momentum analysis
    momentum_valid, momentum_score, momentum_details = analyze_momentum_conditions(df)
    
    # Enhanced time to target calculation with volume spike data
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
            for name, time_val in keops_phi_pivots['future_pivot_times'][:3]:  # Show next 3 pivots
                print(f"  {name}: {time_val:.6f}")
        
        print("\nPhi Price Levels:")
        for name, price in keops_phi_pivots['phi_price_levels'].items():
            print(f"  {name}: {price:.6f}")
        
        # Check if most recent extrema was argmin or argmax
        most_recent_extrema_type = keops_phi_pivots.get('most_recent_extrema_type', 'unknown')
        most_recent_extrema_idx = keops_phi_pivots.get('most_recent_extrema_idx', -1)
        print(f"\nMost Recent Extrema: {most_recent_extrema_type} (Index: {most_recent_extrema_idx})")
        
        # Check if golden triangle direction matches leg type
        if triangle_direction:
            triangle_matches_leg = (triangle_direction == "upward" and keops_phi_pivots['leg_type'] == 'up') or \
                               (triangle_direction == "downward" and keops_phi_pivots['leg_type'] == 'down')
            print(f"Golden Triangle Matches Leg Type: {triangle_matches_leg}")
        
        # Show golden ratio target price
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
    
    # Updated SMA condition check (removed SMA360 requirement)
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
        # Simplified price dip details display
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
    
    # ENHANCED TRADE EXECUTION LOGIC - Execute even if not all conditions are met for the best MTF dip
    print("\n--- TRADE EXECUTION DECISION ---")
    
    # Check USDC balance first
    usdc_balance = get_account_balance(client, 'USDC')
    print(f"USDC Balance: {usdc_balance:.6f}")
    
    if usdc_balance < 10:  # Minimum trade amount
        print("!!! INSUFFICIENT USDC BALANCE - CANNOT EXECUTE TRADE !!!")
        return
    
    # Calculate overall conditions score INCLUDING VPA
    conditions_met = 0
    total_conditions = 9  # Increased from 8 to 9 to include VPA
    
    if is_oversold: conditions_met += 1
    if price_valid: conditions_met += 1
    if momentum_valid: conditions_met += 1
    if pre_spike_valid: conditions_met += 1
    if octagonal_phase in MIN_UPWARD_PHASES: conditions_met += 1
    if triangle_direction == "upward": conditions_met += 1
    if sma_condition == "PASS": conditions_met += 1
    if is_below_poly_fit: conditions_met += 1
    if vpa_score > VPA_MIN_SCORE: conditions_met += 1  # VPA condition
    
    conditions_score = (conditions_met / total_conditions) * 100
    
    print(f"Conditions Met: {conditions_met}/{total_conditions} ({conditions_score:.2f}%)")
    print(f"VPA Contribution: {'YES' if vpa_score > VPA_MIN_SCORE else 'NO'} (Score: {vpa_score:.2f})")
    
    # ENHANCED: Execute trade for best MTF dip even if not all conditions are met
    # Only check for sufficient USDC balance, not perfect conditions
    if conditions_met >= 5:  # At least 55% of conditions met (adjusted for new total)
        print(f"\n!!! SUFFICIENT CONDITIONS MET ({conditions_met}/{total_conditions}) - EXECUTING TRADE !!!")
        
        # Execute buy order
        buy_result = execute_buy_order(client, symbol, usdc_balance * 0.9)  # Use 90% of balance
        if buy_result['success']:
            print(f"BUY ORDER EXECUTED SUCCESSFULLY!")
            print(f"Order ID: {buy_result['order_id']}")
            print(f"Quantity: {buy_result['quantity']}")
            print(f"Price: {buy_result['price']}")
            print(f"Cost: {buy_result['cost']}")
            
            # Start monitoring
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
    """Gathers ALL required data for an asset with enhanced dip scoring including VPA."""
    if stop_event.is_set(): return None
    result = {'symbol': symbol}
    weighted_dip_score = 0.0
    spike_score = 0.0
    
    try:
        # Get MTF data using concurrent processing for ALL timeframes
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = [ex.submit(get_mtf_data, client, symbol, tf) for tf in MTF_SCAN_TIMEFRAMES]
            for f in as_completed(futures):
                if stop_event.is_set(): return None
                data = f.result()
                if not data:
                    continue
                tf = data['timeframe']
                
                # Store metrics with CLEAR timeframe prefix
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
                
                # Store current_price from MTF data
                if 'current_price' not in result:
                    result['current_price'] = data['current_price']
                    
                if data['is_dip']:
                    weight = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                    dip_strength = float(data.get('dip_strength', 50) / 100.0)
                    weighted_dip_score += weight * dip_strength

        # Calculate AVERAGE VPA scores across all timeframes
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
        result['vpa_conditions_met'] = vpa_conditions_met_count >= 2  # At least 2/3 timeframes

        # Get volume spike data (existing)
        volume_spike_data = analyze_volume_spike(client, symbol)
        if volume_spike_data:
            spike_score = volume_spike_data.get('spike_score', 0)
            result['spike_score'] = spike_score
            result['volume_spike_ratio'] = volume_spike_data.get('volume_spike_ratio', 0)
            result['buy_sell_ratio'] = volume_spike_data.get('buy_sell_ratio', 0.5)
            result['bullish_volume_pct'] = volume_spike_data.get('bullish_volume_pct', 0)
            result['bearish_volume_pct'] = volume_spike_data.get('bearish_volume_pct', 0)

        # Get 1h quick price/volume change (existing)
        klines_1h = client.get_klines(symbol=symbol, interval='1h', limit=2)
        if klines_1h and len(klines_1h) >= 2:
            current_c = float(klines_1h[-1][4])
            past_c = float(klines_1h[-2][4])
            current_v = float(klines_1h[-1][5])
            past_v = float(klines_1h[-2][5])
            result['current_price'] = current_c
            result['price_change_1h_pct'] = ((current_c - past_c) / past_c) * 100 if past_c > 0 else 0
            result['volume_change_1h_pct'] = ((current_v - past_v) / past_v) * 100 if past_v > 0 else 0

        # Get geometric data for enhanced scoring
        try:
            klines_geo = client.get_klines(symbol=symbol, interval='1m', limit=200)
            if klines_geo:
                df_geo = pd.DataFrame(klines_geo, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
                
                # Convert all numeric columns to float with error handling
                for c in ['open','high','low','close','volume','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore']:
                    try:
                        df_geo[c] = pd.to_numeric(df_geo[c], errors='coerce')
                    except:
                        df_geo[c] = 0.0
                
                # Fill any NaN values that might have been created
                df_geo.fillna(0.0, inplace=True)
                
                timestamps_geo = np.arange(len(df_geo))
                oct_phase, oct_strength = calculate_octagonal_symmetry(df_geo['close'].values, timestamps_geo)
                tri_direction, tri_strength = detect_golden_triangle(df_geo['close'].values, timestamps_geo)
                
                result['octagonal_phase'] = oct_phase if oct_phase is not None else 0
                result['octagonal_strength'] = oct_strength if oct_strength is not None else 0
                result['triangle_direction'] = tri_direction if tri_direction is not None else 0
                result['triangle_strength'] = tri_strength if tri_strength is not None else 0
                
                # Check polynomial fit
                result['is_below_poly_fit'] = check_polynomial_fit(df_geo['close'].values, timestamps_geo)
                
                # Calculate moving averages
                df_geo_ma = calculate_moving_averages(df_geo)
                if df_geo_ma is not None:
                    result['sma7'] = float(df_geo_ma['SMA7'].iloc[-1]) if 'SMA7' in df_geo_ma.columns else None
                    result['sma12'] = float(df_geo_ma['SMA12'].iloc[-1]) if 'SMA12' in df_geo_ma.columns else None
                    result['sma27'] = float(df_geo_ma['SMA27'].iloc[-1]) if 'SMA27' in df_geo_ma.columns else None
                    result['sma56'] = float(df_geo_ma['SMA56'].iloc[-1]) if 'SMA56' in df_geo_ma.columns else None
                    result['sma150'] = float(df_geo_ma['SMA150'].iloc[-1]) if 'SMA150' in df_geo_ma.columns else None
                    
                    # Check updated SMA condition (removed SMA360 requirement)
                    current_price = result.get('current_price', 0)
                    if (current_price < result['sma7'] < result['sma12'] < result['sma27'] < result['sma56'] < result['sma150']):
                        result['sma_condition_met'] = True
                    else:
                        result['sma_condition_met'] = False
                
                # Calculate RSI and enhanced RSI conditions
                df_geo_rsi = calculate_rsi(df_geo, RSI_PERIOD)
                if df_geo_rsi is not None and f'RSI_{RSI_PERIOD}' in df_geo_rsi.columns:
                    current_rsi = float(df_geo_rsi[f'RSI_{RSI_PERIOD}'].iloc[-1])
                    result['rsi'] = current_rsi
                    # Enhanced RSI analysis
                    rsi_values = df_geo_rsi[f'RSI_{RSI_PERIOD}'].values
                    is_oversold, is_overbought, rsi_score, rsi_details = analyze_rsi_conditions(rsi_values, current_rsi)
                    result['rsi_oversold'] = is_oversold
                    result['rsi_overbought'] = is_overbought
                    result['rsi_score'] = rsi_score
                    result['rsi_details'] = rsi_details
                
                # Enhanced price dip analysis
                price_valid, price_score, price_details = analyze_price_dip_conditions(df_geo['close'].values, result.get('current_price', 0))
                result['price_dip_conditions_met'] = price_valid
                result['price_dip_score'] = price_score
                result['price_dip_details'] = price_details
                
                # Enhanced momentum analysis
                momentum_valid, momentum_score, momentum_details = analyze_momentum_conditions(df_geo)
                result['momentum_conditions_met'] = momentum_valid
                result['momentum_score'] = momentum_score
                result['momentum_details'] = momentum_details
                
                # Pre-spike analysis with new function
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

    # Calculate enhanced power score
    result['weighted_dip_score'] = float(weighted_dip_score)
    result['spike_score'] = float(spike_score)
    
    # Enhanced scoring with geometric factors including VPA
    octagonal_score = result.get('octagonal_strength', 0) * 20
    triangle_score = result.get('triangle_strength', 0) * 30
    volume_score = result.get('spike_score', 0) * 0.5
    poly_fit_score = 20 if result.get('is_below_poly_fit', False) else 0
    sma_condition_score = 30 if result.get('sma_condition_met', False) else 0
    rsi_score = result.get('rsi_score', 0)  # Use the new RSI score
    price_dip_score = result.get('price_dip_score', 0)  # Use the new price dip score
    momentum_score = result.get('momentum_score', 0)  # Use the new momentum score
    pre_spike_score = result.get('pre_spike_score', 0)  # Use the new pre-spike score
    bullish_volume_score = result.get('bullish_volume_pct', 0) * 0.5
    vpa_score = result.get('avg_vpa_score', 0)  # Use AVERAGE VPA across all timeframes
    
    # Calculate power score with all factors including multi-timeframe VPA
    result['power_score'] = float(
        weighted_dip_score * 100 +  # Base dip score
        spike_score +  # Volume spike score
        octagonal_score +  # Octagonal strength
        triangle_score +  # Golden triangle strength
        poly_fit_score +  # Polynomial fit score
        sma_condition_score +  # Moving averages condition score
        rsi_score +  # Enhanced RSI score
        price_dip_score +  # Enhanced price dip score
        momentum_score +  # Enhanced momentum score
        pre_spike_score +  # Pre-spike detection score
        bullish_volume_score +  # Bullish volume score
        vpa_score  # Multi-timeframe VPA score
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
    
    usdc_pairs = fetch_usdc_pairs(client)
    if not usdc_pairs:
        print("No pairs found, exiting.")
        return

    start_time = time.time()
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Scanning {len(usdc_pairs)} assets...")
    
    all_results = []
    scan_stats = {'Total Assets Scanned':0,'Not Enough MTF Dips':0,'No Spike Pattern':0,'Other Errors':0}
    
    assets_to_scan = usdc_pairs # Scan all assets once

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
                    else:
                        scan_stats['Other Errors'] += 1
                except Exception:
                    scan_stats['Other Errors'] += 1
        time.sleep(0.2) # Be gentle with API

    # Categorize
    for r in all_results:
        if r.get('weighted_dip_score',0) < MIN_WEIGHTED_DIP_SCORE:
            scan_stats['Not Enough MTF Dips'] += 1
        elif not r.get('spike_score', 0) > 0:
            scan_stats['No Spike Pattern'] += 1

    # Enhanced winner selection with geometric filtering including VPA
    analysis_winner = None
    if all_results:
        # Filter for assets with clear geometric patterns indicating upward movement
        filtered_results = []
        for r in all_results:
            oct_phase = r.get('octagonal_phase', 0)
            oct_strength = r.get('octagonal_strength', 0)
            tri_direction = r.get('triangle_direction', None)
            tri_strength = r.get('triangle_strength', 0)
            
            # Only consider assets with:
            # 1. Upward octagonal phase (0, 1, 2, 3)
            # 2. Upward golden triangle direction
            # 3. Minimum strength thresholds
            # 4. SMA condition met
            # 5. Below poly fit
            # 6. Enhanced RSI conditions met
            # 7. Enhanced price dip conditions met
            # 8. Enhanced momentum conditions met
            # 9. Pre-spike conditions met
            # 10. VPA conditions met
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
            
            if (has_upward_phase and has_upward_triangle and meets_strength_threshold and 
                sma_condition_met and below_poly_fit and rsi_oversold and 
                price_dip_conditions_met and momentum_conditions_met and pre_spike_conditions_met and vpa_conditions_met):
                filtered_results.append(r)
        
        if filtered_results:
            # Select the best from the filtered results
            analysis_winner = max(filtered_results, key=lambda x: x.get('power_score',0))
        else:
            # If no assets meet the enhanced criteria, fall back to regular selection
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
