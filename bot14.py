#!/usr/bin/env python3
"""
Enhanced trading analysis bot with geometric pattern confirmation and fast spike detection.
Features:
 - MTF dip detection with ATR weighting for 1m, 3m, and 5m timeframes
 - Enhanced geometric pattern detection (octagonal symmetry, golden triangle)
 - Volume spike confirmation for fast entry identification
 - Multi-model forecasting with time-to-target calculation
 - Sinusoidal pattern analysis for cycle timing
 - Enhanced scoring system combining all factors
 - Single-run mode to find and analyze the best opportunity
 - Optimized concurrent scanning for faster execution
 - Multiple moving averages confirmation (MA7 < SMA12 < SMA27 < SMA56 < SMA150 < SMA360)
 - Polynomial fit analysis for trend confirmation
 - Volume analysis (bullish vs bearish)
 - RSI analysis with oversold confirmation
 - Golden ratio support/resistance levels
 - FFT forecast price prediction
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# --- Optional ML/Signal libraries with graceful fallbacks ---
try:
    import pandas_ta as ta
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import pmdarima as pm
    from scipy.fft import fft, fftfreq
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    print("="*80)
    print("!!! IMPORT WARNING (some optional libs missing) !!!")
    print("Some features may be limited. Error:", e)
    print("Install required packages: pip install pandas-ta scikit-learn pmdarima scipy")
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
MA7_PERIOD = 7
SMA12_PERIOD = 12
SMA27_PERIOD = 27
SMA56_PERIOD = 56
SMA150_PERIOD = 150
SMA360_PERIOD = 360

# Misc
MAX_KLINES_LIMIT = 2000
FFT_MIN_LENGTH = 64
ENTROPY_M = 2
ENTROPY_R_SCALE = 0.2
MAX_WORKERS = 12

# Global stop event
stop_event = threading.Event()

def signal_handler(sig, frame):
    print('\nCtrl+C pressed! Shutting down gracefully...')
    stop_event.set()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

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

def detect_golden_ratio_patterns(prices, timestamps=None):
    """
    PURE 0 to 1.000 GOLDEN RATIO SYSTEM
    - Only levels INSIDE the most recent swing (0.000 → 1.000)
    - 8 perfectly symmetrical, φ-based internal levels
    - No extensions above 1.000
    - Used by ICT, SMC, and harmonic pattern masters
    """
    try:
        if len(prices) < 50:
            return None

        # Find most recent significant swing low and high
        window = prices[-140:]
        low_idx_local  = np.argmin(window)
        high_idx_local = np.argmax(window)

        low_idx  = len(prices) - len(window) + low_idx_local
        high_idx = len(prices) - len(window) + high_idx_local

        swing_low  = float(prices[low_idx])
        swing_high = float(prices[high_idx])

        if swing_high <= swing_low:
            return None

        fib_range = swing_high - swing_low

        # PURE 0–1 INTERNAL GOLDEN RATIO LEVELS (8 levels, perfectly symmetrical)
        levels = {
            'Level_0.000': swing_low,                                          # Swing Low (Origin)
            'Level_0.146': swing_low + fib_range * 0.146,                      # φ⁻⁴ (deep)
            'Level_0.236': swing_low + fib_range * 0.236,                      # φ⁻³
            'Level_0.382': swing_low + fib_range * 0.382,                      # √φ
            'Level_0.500': swing_low + fib_range * 0.500,                      # Midpoint (balance)
            'Level_0.618': swing_low + fib_range * 0.618,                      # Golden Ratio (φ⁻¹) - The Golden Pocket
            'Level_0.786': swing_low + fib_range * 0.786,                      # √(φ²)
            'Level_1.000': swing_high,                                         # Swing High (Target)
        }

        return levels

    except Exception as e:
        print(f"Golden Ratio (0–1) Error: {e}")
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
        idx = np.argmax(mag)
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
        if df is None or len(df) < SMA360_PERIOD:
            return None
            
        df = df.copy()
        
        # Calculate MA7
        df['MA7'] = df['close'].rolling(window=MA7_PERIOD).mean()
        
        # Calculate SMAs
        df['SMA12'] = df['close'].rolling(window=SMA12_PERIOD).mean()
        df['SMA27'] = df['close'].rolling(window=SMA27_PERIOD).mean()
        df['SMA56'] = df['close'].rolling(window=SMA56_PERIOD).mean()
        df['SMA150'] = df['close'].rolling(window=SMA150_PERIOD).mean()
        df['SMA360'] = df['close'].rolling(window=SMA360_PERIOD).mean()
        
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
    """Enhanced MTF data analysis with ATR-based dip detection."""
    if stop_event.is_set(): return None
    try:
        # Get more data for 1min timeframe for better analysis
        limit = 500 if timeframe == '1m' else 200
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
        if not klines or len(klines) < 20:
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

        df = calculate_atr(df, ATR_PERIOD)
        if df is None or f'ATR_{ATR_PERIOD}' not in df.columns:
            return None

        current_atr = float(df[f'ATR_{ATR_PERIOD}'].iloc[-1])
        current_price = float(df['close'].iloc[-1])

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

        price_change_pct = 0.0
        volume_change_pct = 0.0
        if len(df) >= 2:
            past_price, past_vol = df['close'].iloc[-2], df['volume'].iloc[-2]
            if past_price > 0: price_change_pct = ((current_price - past_price) / past_price) * 100
            if past_vol > 0: volume_change_pct = ((df['volume'].iloc[-1] - past_vol) / past_vol) * 100

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
            'price_change_pct': price_change_pct,
            'volume_change_pct': volume_change_pct,
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
            df['return_1'] = df['close'].pct_change(1)
            df['vol_ma_10'] = df['volume'].rolling(10, min_periods=1).mean()
            X = df[['close','volume','return_1','vol_ma_10']].copy()
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
            for c in ['open','high','low','close','volume','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume']:
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
        # Fixed sorting: first by power_score, then by weighted_dip_score
        if 'power_score' in df.columns:
            df = df.sort_values(by=['power_score', 'weighted_dip_score'], ascending=[False, False])
        else:
            df = df.sort_values(by='weighted_dip_score', ascending=False)
        print(df.head(20).to_string(index=False, float_format="%.25f"))
    print("\n--- Scan Statistics ---")
    for k,v in scan_stats.items():
        print(f" - {k:<25}: {v}")

# ------------------ Final Analysis Pipeline ------------------

def perform_final_analysis(client, symbol):
    """Full ML + cycle + entropy + geometric analysis."""
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
        for c in ['open','high','low','close','volume','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume']:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except:
                df[c] = 0.0
        
        # Fill any NaN values that might have been created
        df.fillna(0.0, inplace=True)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for c in ['open','high','low','close','volume']:
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
    
    print("Performing enhanced geometric analysis...")
    octagonal_phase, octagonal_strength = calculate_octagonal_symmetry(df['close'].values, timestamps)
    golden_levels = detect_golden_ratio_patterns(df['close'].values, timestamps)
    triangle_direction, triangle_strength = detect_golden_triangle(df['close'].values, timestamps)
    sin_amplitude, sin_freq, sin_phase, next_extremum, sin_period = analyze_sinuosidal_pattern(df['close'].values, timestamps, '1m')
    
    # Get volume spike data
    volume_spike_data = analyze_volume_spike(client, symbol)
    
    # Calculate moving averages
    df_ma = calculate_moving_averages(df)
    ma7 = float(df_ma['MA7'].iloc[-1]) if df_ma is not None and 'MA7' in df_ma.columns else None
    sma12 = float(df_ma['SMA12'].iloc[-1]) if df_ma is not None and 'SMA12' in df_ma.columns else None
    sma27 = float(df_ma['SMA27'].iloc[-1]) if df_ma is not None and 'SMA27' in df_ma.columns else None
    sma56 = float(df_ma['SMA56'].iloc[-1]) if df_ma is not None and 'SMA56' in df_ma.columns else None
    sma150 = float(df_ma['SMA150'].iloc[-1]) if df_ma is not None and 'SMA150' in df_ma.columns else None
    sma360 = float(df_ma['SMA360'].iloc[-1]) if df_ma is not None and 'SMA360' in df_ma.columns else None
    
    # Check polynomial fit
    is_below_poly_fit = check_polynomial_fit(df['close'].values, timestamps)
    
    # Calculate RSI
    df_rsi = calculate_rsi(df, RSI_PERIOD)
    current_rsi = float(df_rsi[f'RSI_{RSI_PERIOD}'].iloc[-1]) if df_rsi is not None and f'RSI_{RSI_PERIOD}' in df_rsi.columns else 50.0
    is_oversold = current_rsi < RSI_OVERSOLD
    rsi_to_oversold = max(0, RSI_OVERSOLD - current_rsi)
    rsi_to_middle = abs(RSI_MIDDLE - current_rsi)
    
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
    print(f"Current Price: {current_price:.25f}")
    print(f"Approximate Entropy (ApEn): {ap_en:.25f} (norm predictability factor: {1.0-entropy_norm:.25f})")
    
    print("\n--- Enhanced Geometric Analysis ---")
    print(f"Octagonal Phase: {octagonal_phase}/7 ({octagonal_strength:.25f} strength)")
    print(f"Golden Triangle Direction: {triangle_direction} ({triangle_strength:.25f} strength)")
    if next_extremum:
        extremum_time, extremum_type = next_extremum
        print(f"Next {extremum_type}: {extremum_time:.25f} (sinusoidal analysis)")
    if golden_levels:
        print("\n--- Golden Ratio Levels ---")
        print("Golden Ratio Levels (Pure 0→1 Internal):")
        for name, price in golden_levels.items():
            print(f"  {name}: {price:.25f}")
    
    if volume_spike_data:
        print("\n--- Volume Spike Analysis ---")
        print(f"Volume Spike Ratio: {volume_spike_data.get('volume_spike_ratio', 0):.25f}x")
        print(f"Price Change (5m): {volume_spike_data.get('price_change_5', 0)*100:.25f}%")
        print(f"Price Change (10m): {volume_spike_data.get('price_change_10', 0)*100:.25f}%")
        print(f"Buy/Sell Ratio: {volume_spike_data.get('buy_sell_ratio', 0.5):.25f}")
        print(f"Bullish Volume: {volume_spike_data.get('bullish_volume', 0):.25f} ({volume_spike_data.get('bullish_volume_pct', 0):.25f}%)")
        print(f"Bearish Volume: {volume_spike_data.get('bearish_volume', 0):.25f} ({volume_spike_data.get('bearish_volume_pct', 0):.25f}%)")
    
    print("\n--- Moving Averages Analysis ---")
    print(f"MA7: {ma7:.25f}")
    print(f"SMA12: {sma12:.25f}")
    print(f"SMA27: {sma27:.25f}")
    print(f"SMA56: {sma56:.25f}")
    print(f"SMA150: {sma150:.25f}")
    print(f"SMA360: {sma360:.25f}")
    
    ma_condition = "PASS" if (current_price < ma7 < sma12 < sma27 < sma56 < sma150 < sma360) else "FAIL"
    print(f"MA Condition (Close < MA7 < SMA12 < SMA27 < SMA56 < SMA150 < SMA360): {ma_condition}")
    
    print("\n--- Polynomial Fit Analysis ---")
    print(f"Below Poly Fit: {is_below_poly_fit}")
    
    print("\n--- RSI Analysis ---")
    print(f"Current RSI: {current_rsi:.25f}")
    print(f"RSI Oversold: {is_oversold}")
    print(f"Distance to Oversold: {rsi_to_oversold:.25f}")
    print(f"Distance to Middle: {rsi_to_middle:.25f}")
    
    print("\n--- Model Predictions ---")
    for name, target in model_targets.items():
        conf = model_confidences.get(name, 0.0)
        print(f" - {name:<15}: {float(target):.25f}  conf={conf:.25f}")
    
    print("\n--- Consensus Forecast ---")
    print(f"!!! CONSENSUS TARGET: {consensus_target:.25f} ({potential_change_pct:+.25f}%) !!!")
    if time_to_target:
        time_str = format_time_to_target(time_to_target)
        print(f"!!! ESTIMATED TIME TO TARGET: {time_str} (confidence: {time_confidence:.25f}) !!!")
    
    print("\n--- MTF Thresholds & Predictive Zones (sample) ---")
    if mtf_thresholds:
        order = ['1m','3m','5m']
        for tf in order:
            if tf in mtf_thresholds:
                data = mtf_thresholds[tf]
                min_p, max_p, middle_p, std_dev = data['min'], data['max'], data['middle'], data['std_dev']
                cp = data.get('current_price', 0.0)
                
                # Calculate symmetrical percentages within the min-max range
                if max_p > min_p:
                    total_range = max_p - min_p
                    pct_from_min = ((cp - min_p) / total_range) * 100
                    pct_from_max = ((max_p - cp) / total_range) * 100
                else:
                    # Handle edge case where min and max are the same
                    pct_from_min = 0.0
                    pct_from_max = 100.0
                
                print(f" | {tf:<4} | Min:{min_p:.25f} Max:{max_p:.25f} Middle:{middle_p:.25f} Std:{std_dev:.25f}")
                print(f" |     | Current:{cp:.25f} (Pos: {pct_from_min:.25f}% from Min, {pct_from_max:.25f}% from Max) ATR:{data.get('atr',0):.25f}")
    print("="*80)
    print("Analysis complete.")


# ------------------ Single-asset analysis wrapper ------------------

def analyze_asset_for_table(client, symbol):
    """Gathers all required data for an asset with enhanced dip scoring."""
    if stop_event.is_set(): return None
    result = {'symbol': symbol}
    weighted_dip_score = 0.0
    spike_score = 0.0
    
    try:
        # Get MTF data using concurrent processing
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = [ex.submit(get_mtf_data, client, symbol, tf) for tf in MTF_SCAN_TIMEFRAMES]
            for f in as_completed(futures):
                if stop_event.is_set(): return None
                data = f.result()
                if not data:
                    continue
                tf = data['timeframe']
                result[f'{tf}_price_change_pct'] = data['price_change_pct']
                result[f'{tf}_volume_change_pct'] = data['volume_change_pct']
                result['current_price'] = data['current_price']
                if data['is_dip']:
                    weight = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                    dip_strength = float(data.get('dip_strength', 50)) / 100.0
                    weighted_dip_score += weight * dip_strength

        # Get volume spike data
        volume_spike_data = analyze_volume_spike(client, symbol)
        if volume_spike_data:
            spike_score = volume_spike_data.get('spike_score', 0)
            result['spike_score'] = spike_score
            result['volume_spike_ratio'] = volume_spike_data.get('volume_spike_ratio', 0)
            result['buy_sell_ratio'] = volume_spike_data.get('buy_sell_ratio', 0.5)
            result['bullish_volume_pct'] = volume_spike_data.get('bullish_volume_pct', 0)
            result['bearish_volume_pct'] = volume_spike_data.get('bearish_volume_pct', 0)

        # Get 1h quick price/volume change
        klines_1h = client.get_klines(symbol=symbol, interval='1h', limit=2)
        if klines_1h and len(klines_1h) >= 2:
            current_c = float(klines_1h[-1][4]); past_c = float(klines_1h[-2][4])
            current_v = float(klines_1h[-1][5]); past_v = float(klines_1h[-2][5])
            result['current_price'] = current_c
            result['price_change_1h_pct'] = ((current_c - past_c) / past_c) * 100 if past_c > 0 else 0
            result['volume_change_1h_pct'] = ((current_v - past_v) / past_v) * 100 if past_v > 0 else 0

        # Get geometric data for enhanced scoring
        try:
            klines_geo = client.get_klines(symbol=symbol, interval='1m', limit=200)
            if klines_geo:
                df_geo = pd.DataFrame(klines_geo, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
                
                # Convert all numeric columns to float with error handling
                for c in ['open','high','low','close','volume','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume']:
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
                result['triangle_direction'] = tri_direction
                result['triangle_strength'] = tri_strength if tri_strength is not None else 0
                
                # Check polynomial fit
                result['is_below_poly_fit'] = check_polynomial_fit(df_geo['close'].values, timestamps_geo)
                
                # Calculate moving averages
                df_geo_ma = calculate_moving_averages(df_geo)
                if df_geo_ma is not None:
                    result['ma7'] = float(df_geo_ma['MA7'].iloc[-1]) if 'MA7' in df_geo_ma.columns else None
                    result['sma12'] = float(df_geo_ma['SMA12'].iloc[-1]) if 'SMA12' in df_geo_ma.columns else None
                    result['sma27'] = float(df_geo_ma['SMA27'].iloc[-1]) if 'SMA27' in df_geo_ma.columns else None
                    result['sma56'] = float(df_geo_ma['SMA56'].iloc[-1]) if 'SMA56' in df_geo_ma.columns else None
                    result['sma150'] = float(df_geo_ma['SMA150'].iloc[-1]) if 'SMA150' in df_geo_ma.columns else None
                    result['sma360'] = float(df_geo_ma['SMA360'].iloc[-1]) if 'SMA360' in df_geo_ma.columns else None
                    
                    # Check MA condition
                    current_price = result.get('current_price', 0)
                    if (current_price < result['ma7'] < result['sma12'] < result['sma27'] < 
                        result['sma56'] < result['sma150'] < result['sma360']):
                        result['ma_condition_met'] = True
                    else:
                        result['ma_condition_met'] = False
                
                # Calculate RSI
                df_geo_rsi = calculate_rsi(df_geo, RSI_PERIOD)
                if df_geo_rsi is not None and f'RSI_{RSI_PERIOD}' in df_geo_rsi.columns:
                    current_rsi = float(df_geo_rsi[f'RSI_{RSI_PERIOD}'].iloc[-1])
                    result['rsi'] = current_rsi
                    result['is_oversold'] = current_rsi < RSI_OVERSOLD
                    result['rsi_to_oversold'] = max(0, RSI_OVERSOLD - current_rsi)
                    result['rsi_to_middle'] = abs(RSI_MIDDLE - current_rsi)
                
        except Exception as e:
            print(f"Error getting geometric data for {symbol}: {e}")
            result['octagonal_phase'] = 0
            result['octagonal_strength'] = 0
            result['triangle_direction'] = None
            result['triangle_strength'] = 0
            result['is_below_poly_fit'] = False
            result['ma_condition_met'] = False
            result['rsi'] = 50.0
            result['is_oversold'] = False
            result['rsi_to_oversold'] = 0.0
            result['rsi_to_middle'] = 0.0

    except Exception as e:
        return None

    # Calculate enhanced power score
    result['weighted_dip_score'] = float(weighted_dip_score)
    result['spike_score'] = float(spike_score)
    
    # Enhanced scoring with geometric factors
    octagonal_score = result.get('octagonal_strength', 0) * 20
    triangle_score = result.get('triangle_strength', 0) * 30
    volume_score = result.get('spike_score', 0) * 0.5
    poly_fit_score = 20 if result.get('is_below_poly_fit', False) else 0
    ma_condition_score = 30 if result.get('ma_condition_met', False) else 0
    rsi_oversold_score = result.get('rsi_to_oversold', 0) * 2
    rsi_middle_score = result.get('rsi_to_middle', 0)
    bullish_volume_score = result.get('bullish_volume_pct', 0) * 0.5
    
    # Calculate power score with all factors
    result['power_score'] = float(
        weighted_dip_score * 100 +  # Base dip score
        spike_score +  # Volume spike score
        octagonal_score +  # Octagonal strength
        triangle_score +  # Golden triangle strength
        poly_fit_score +  # Polynomial fit score
        ma_condition_score +  # Moving averages condition score
        rsi_oversold_score +  # RSI oversold score
        rsi_middle_score +  # RSI to middle score
        bullish_volume_score  # Bullish volume score
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

    # Enhanced winner selection with geometric filtering
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
            # 4. MA condition met
            # 5. Below poly fit
            # 6. RSI oversold or close to middle
            has_upward_phase = oct_phase in MIN_UPWARD_PHASES
            has_upward_triangle = tri_direction == "upward"
            meets_strength_threshold = oct_strength >= MIN_OCTAGONAL_STRENGTH or tri_strength >= MIN_TRIANGLE_STRENGTH
            ma_condition_met = r.get('ma_condition_met', False)
            below_poly_fit = r.get('is_below_poly_fit', False)
            rsi_oversold = r.get('is_oversold', False)
            rsi_to_middle = r.get('rsi_to_middle', 0) < 10  # Close to middle
            
            if (has_upward_phase and has_upward_triangle and meets_strength_threshold and 
                ma_condition_met and below_poly_fit and (rsi_oversold or rsi_to_middle)):
                filtered_results.append(r)
        
        if filtered_results:
            # Select the best from the filtered results
            analysis_winner = max(filtered_results, key=lambda x: x.get('power_score',0))
        else:
            # If no assets meet the enhanced criteria, fall back to regular selection
            analysis_winner = max(all_results, key=lambda x: x.get('power_score',0))
    
    print_dynamic_table(all_results, scan_stats)

    if analysis_winner:
        print(f"\n!!! WINNER: {analysis_winner['symbol']} (score: {analysis_winner.get('power_score'):.25f}) !!!")
        perform_final_analysis(client, analysis_winner['symbol'])
    else:
        print("\nNo suitable MTF dip found in this scan.")
        
    duration = time.time() - start_time
    print(f"\nEnhanced analysis complete in {duration:.2f}s. Exiting.")

if __name__ == "__main__":
    main()