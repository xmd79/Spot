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

# suppress warnings for clean output per your original script
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
    # We'll try to continue; functions will check availability.

# ------------------ Configuration ------------------
API_FILE = 'api.txt'
MTF_SCAN_TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h', '1d']
DETAILED_1M_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

# Scoring & criteria
MIN_WEIGHTED_DIP_SCORE = 4.0
VOLUME_ANALYSIS_PERIOD = 56
PRICE_UPTREND_PERIOD = 5

# Optimization & Weights
TIMEFRAME_WEIGHTS = {'5m': 1.0, '15m': 1.2, '30m': 1.4, '1h': 1.6, '4h': 1.8, '1d': 2.0}
ASSET_SCAN_LIMIT = 100
MIN_24H_VOLUME_USD = 500000
MIN_24H_PRICE_CHANGE_PCT = 0.5
BATCH_SIZE = 20

# ATR
ATR_PERIOD = 14
ATR_DIP_MULTIPLIER = 1.5
ATR_SPIKE_MULTIPLIER = 2.0
ATR_VOLUME_SPIKE_THRESHOLD = 1.5

# Octagonal and Geometric Analysis
OCTAGONAL_SEGMENTS = 8  # 8 vertices of an octagon
GOLDEN_RATIO = 1.618033988749895  # φ
GOLDEN_ANGLE = 137.5077640500378  # Golden angle in degrees

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

signal.signal(signal.SIGINT, signal_handler)

# ------------------ Geometric Analysis Functions ------------------

def calculate_octagonal_symmetry(prices, timestamps):
    """
    Analyze price movements in octagonal symmetry (8 segments of 45 degrees each).
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
        
        return octagonal_phase, symmetry_strength
    except Exception as e:
        print(f"calculate_octagonal_symmetry error: {e}")
        return None, 0.0

def detect_golden_ratio_patterns(prices, timestamps):
    """
    Detect golden ratio patterns in price movements.
    Returns potential support/resistance levels based on golden ratio.
    """
    try:
        if len(prices) < 20:
            return None
            
        # Find significant highs and lows
        highs = []
        lows = []
        
        # Simple peak detection
        for i in range(1, len(prices)-1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                highs.append((i, prices[i]))
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                lows.append((i, prices[i]))
                
        if not highs or not lows:
            return None
            
        # Calculate potential retracement levels based on golden ratio
        current_price = prices[-1]
        last_high = max(highs, key=lambda x: x[1])[1]
        last_low = min(lows, key=lambda x: x[1])[1]
        
        # Golden ratio levels
        if current_price > last_low:  # Uptrend
            fib_levels = {
                'support_0.618': last_low + (last_high - last_low) * 0.382,  # 61.8% support
                'support_0.5': last_low + (last_high - last_low) * 0.5,     # 50% support
                'support_0.382': last_low + (last_high - last_low) * 0.618, # 38.2% support
                'resistance_1.618': last_high + (last_high - last_low) * 0.618, # 161.8% extension
                'resistance_2.618': last_high + (last_high - last_low) * 1.618  # 261.8% extension
            }
        else:  # Downtrend
            fib_levels = {
                'resistance_0.618': last_high - (last_high - last_low) * 0.382,  # 61.8% resistance
                'resistance_0.5': last_high - (last_high - last_low) * 0.5,     # 50% resistance
                'resistance_0.382': last_high - (last_high - last_low) * 0.618, # 38.2% resistance
                'support_1.618': last_low - (last_high - last_low) * 0.618,     # 161.8% extension
                'support_2.618': last_low - (last_high - last_low) * 1.618      # 261.8% extension
            }
            
        return fib_levels
    except Exception as e:
        print(f"detect_golden_ratio_patterns error: {e}")
        return None

def detect_golden_triangle(prices, timestamps):
    """
    Detect golden triangle patterns in price movements.
    Returns potential breakout direction and strength.
    """
    try:
        if len(prices) < 20:
            return None, 0.0
            
        # Find three points that form a potential triangle
        n = len(prices)
        best_triangle = None
        best_score = 0.0
        
        # Look for potential triangles in the last portion of the data
        for i in range(int(n*0.6), n-10):
            for j in range(i+5, n-5):
                for k in range(j+5, n):
                    # Get the three points
                    p1 = (timestamps[i], prices[i])
                    p2 = (timestamps[j], prices[j])
                    p3 = (timestamps[k], prices[k])
                    
                    # Calculate side lengths
                    a = math.sqrt((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2)
                    b = math.sqrt((p1[0]-p3[0])**2 + (p1[1]-p3[1])**2)
                    c = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                    
                    # Check if it's a valid triangle
                    if a <= 0 or b <= 0 or c <= 0:
                        continue
                        
                    # Calculate ratios to check for golden triangle
                    ratios = [a/b, b/c, a/c]
                    golden_matches = sum(1 for r in ratios if abs(r - GOLDEN_RATIO) < 0.1 or abs(1/r - GOLDEN_RATIO) < 0.1)
                    
                    if golden_matches > 0:
                        # Calculate score based on how close to golden ratio
                        score = sum(1 - min(abs(r - GOLDEN_RATIO), abs(1/r - GOLDEN_RATIO)) / 0.1 for r in ratios) / 3
                        if score > best_score:
                            best_score = score
                            best_triangle = (p1, p2, p3)
                            
        if best_triangle:
            # Determine breakout direction
            p1, p2, p3 = best_triangle
            # If the last point is higher than the previous, potential upward breakout
            direction = "upward" if p3[1] > p2[1] else "downward"
            return direction, best_score
            
        return None, 0.0
    except Exception as e:
        print(f"detect_golden_triangle error: {e}")
        return None, 0.0

def calculate_time_to_target(current_price, target_price, cycle_period, current_phase, octagonal_phase):
    """
    Calculate time to reach target price using Pythagorean theorem and cycle analysis.
    Returns estimated time in seconds and confidence.
    """
    try:
        if cycle_period <= 0 or current_price <= 0 or target_price <= 0:
            return None, 0.0
            
        # Calculate price distance
        price_distance = abs(target_price - current_price)
        
        # Calculate time distance based on cycle period and current phase
        # Using Pythagorean theorem: c^2 = a^2 + b^2
        # Where a is price distance, b is time distance, c is the hypotenuse
        
        # Estimate time distance based on cycle period
        # Adjust based on octagonal phase (0-7)
        phase_adjustment = 1.0 + (octagonal_phase / OCTAGONAL_SEGMENTS) * 0.5
        
        # Calculate the hypotenuse (price-time vector)
        # Normalize price to make it comparable to time
        normalized_price = price_distance / current_price
        hypotenuse = math.sqrt(normalized_price**2 + cycle_period**2)
        
        # Calculate time to target
        time_to_target = hypotenuse * phase_adjustment
        
        # Calculate confidence based on how well we're aligned with the cycle
        phase_alignment = 1.0 - abs(current_phase - 0.5) * 2  # 1.0 at peak, 0.0 at trough
        octagonal_alignment = 1.0 - abs(octagonal_phase - OCTAGONAL_SEGMENTS/2) / (OCTAGONAL_SEGMENTS/2)
        
        confidence = (phase_alignment + octagonal_alignment) / 2
        
        return time_to_target, confidence
    except Exception as e:
        print(f"calculate_time_to_target error: {e}")
        return None, 0.0

def analyze_sinuosidal_pattern(prices, timestamps):
    """
    Analyze sinusoidal patterns between major minima and current price.
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
        xf = fftfreq(n, d=(segment_times[1] - segment_times[0]))
        
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
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=200)
        if not klines or len(klines) < 20:
            return None

        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
        for c in ['open','high','low','close','volume']:
            df[c] = df[c].astype(float)

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

def analyze_atr_spike_pattern(client, symbol):
    """Analyzes 1m chart for signs of an imminent price spike using ATR."""
    if stop_event.is_set(): return None
    try:
        limit_req = max(VOLUME_ANALYSIS_PERIOD + 50, 100)
        klines = client.get_klines(symbol=symbol, interval='1m', limit=limit_req)
        if not klines or len(klines) < VOLUME_ANALYSIS_PERIOD:
            return None

        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
        for c in ['open','high','low','close','volume']:
            df[c] = df[c].astype(float)

        df = calculate_atr(df, ATR_PERIOD)
        if df is None or f'ATR_{ATR_PERIOD}' not in df.columns:
            return None

        current_price = df['close'].iloc[-1]
        current_atr = float(df[f'ATR_{ATR_PERIOD}'].iloc[-1])
        current_volume = float(df['volume'].iloc[-1])
        avg_volume = float(df['volume'].iloc[-VOLUME_ANALYSIS_PERIOD:].mean())

        is_volume_spike = current_volume > avg_volume * ATR_VOLUME_SPIKE_THRESHOLD
        recent_closes = df['close'].iloc[-PRICE_UPTREND_PERIOD:]
        is_price_increasing = len(recent_closes) >= 2 and all(recent_closes.iloc[i] > recent_closes.iloc[i-1] for i in range(1,len(recent_closes)))
        price_change_atr = (current_price - df['close'].iloc[-10]) / current_atr if current_atr > 0 else 0.0
        is_atr_breakout = price_change_atr >= ATR_SPIKE_MULTIPLIER
        ma20 = df['close'].iloc[-20:].mean() if len(df) >= 20 else df['close'].mean()
        is_above_ma = current_price > ma20

        spike_criteria_met = sum([is_volume_spike, is_price_increasing, is_atr_breakout, is_above_ma])
        is_spike = spike_criteria_met >= 3

        if is_spike:
            volume_factor = min(100, (current_volume / avg_volume - 1) * 100) if avg_volume > 0 else 0
            atr_factor = min(100, price_change_atr * 25)
            score = (volume_factor * 0.6 + atr_factor * 0.4) * 10
            return {'score': score, 'current_price': current_price}
        return None
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
    for timeframe in DETAILED_1M_TIMEFRAMES:
        if stop_event.is_set(): break
        try:
            klines = client.get_klines(symbol=symbol, interval=timeframe, limit=1000)
            if not klines: continue
            df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
            for c in ['open','high','low','close','volume']:
                df[c] = df[c].astype(float)
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

def print_dynamic_table(all_results, near_misses, scan_stats):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("--- Live Market Scan - Top Candidates ---")
    if all_results:
        df = pd.DataFrame(all_results)
        if 'power_score' in df.columns:
            df = df.sort_values(by='power_score', ascending=False)
        else:
            df = df.sort_values(by='weighted_dip_score', ascending=False)
        print(df.head(20).to_string(index=False, float_format="%.2f"))
    print("\n--- Near Misses ---")
    if near_misses:
        dfn = pd.DataFrame(near_misses)
        print(dfn.head(15).to_string(index=False, float_format="%.2f"))
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
        klines = client.get_klines(symbol=symbol, interval='1h', limit=MAX_KLINES_LIMIT)
        if not klines:
            print("Failed to fetch historical data. Aborting.")
            return
        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
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
    
    print("Performing geometric analysis...")
    octagonal_phase, octagonal_strength = calculate_octagonal_symmetry(df['close'].values, timestamps)
    golden_levels = detect_golden_ratio_patterns(df['close'].values, timestamps)
    triangle_direction, triangle_strength = detect_golden_triangle(df['close'].values, timestamps)
    sin_amplitude, sin_freq, sin_phase, next_extremum, sin_period = analyze_sinuosidal_pattern(df['close'].values, timestamps)
    
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
    
    time_to_target = None
    time_confidence = 0.0
    if fft_res and sin_period:
        cycle_period = sin_period
        current_phase = sin_phase if sin_phase is not None else 0.5
        oct_phase = octagonal_phase if octagonal_phase is not None else 4
        time_to_target, time_confidence = calculate_time_to_target(
            current_price, consensus_target, cycle_period, current_phase, oct_phase
        )

    print("\n" + "="*80)
    print("!!! FINAL ANALYSIS REPORT !!!")
    print("="*80)
    print(f"Asset: {symbol}")
    print(f"Current Price: {current_price:.8f}")
    print(f"Approximate Entropy (ApEn): {ap_en:.6f} (norm predictability factor: {1.0-entropy_norm:.3f})")
    
    print("\n--- Geometric Analysis ---")
    print(f"Octagonal Phase: {octagonal_phase}/7 ({octagonal_strength:.2f} strength)")
    print(f"Golden Triangle Direction: {triangle_direction} ({triangle_strength:.2f} strength)")
    if next_extremum:
        extremum_time, extremum_type = next_extremum
        print(f"Next {extremum_type}: {extremum_time:.2f} (sinusoidal analysis)")
    if golden_levels:
        print("Golden Ratio Levels:")
        for level, price in golden_levels.items():
            print(f"  {level}: {price:.8f}")
    
    print("\n--- Model Predictions ---")
    for name, target in model_targets.items():
        conf = model_confidences.get(name, 0.0)
        print(f" - {name:<15}: {float(target):.8f}  conf={conf:.3f}")
    
    print("\n--- Consensus Forecast ---")
    print(f"!!! CONSENSUS TARGET: {consensus_target:.8f} ({potential_change_pct:+.2f}%) !!!")
    if time_to_target:
        time_str = format_time_ago(time_to_target)
        print(f"!!! ESTIMATED TIME TO TARGET: {time_str} (confidence: {time_confidence:.2f}) !!!")
    
    print("\n--- MTF Thresholds & Predictive Zones (sample) ---")
    if mtf_thresholds:
        order = ['1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d']
        for tf in order:
            if tf in mtf_thresholds:
                data = mtf_thresholds[tf]
                min_p, max_p, middle_p, std_dev = data['min'], data['max'], data['middle'], data['std_dev']
                cp = data.get('current_price', 0.0)
                dist_to_min = ((cp - min_p)/min_p)*100 if min_p>0 else 0
                dist_to_max = ((max_p - cp)/max_p)*100 if max_p>0 else 0
                print(f" | {tf:<4} | Min:{min_p:.8f} Max:{max_p:.8f} Middle:{middle_p:.8f} Std:{std_dev:.8f}")
                print(f" |     | Current:{cp:.8f} ({dist_to_min:+.2f}% from Min, {dist_to_max:+.2f}% from Max) ATR:{data.get('atr',0):.8f}")
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
        with ThreadPoolExecutor(max_workers=6) as ex:
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

        spike_res = analyze_atr_spike_pattern(client, symbol)
        if spike_res:
            spike_score = float(spike_res['score'])
            result['spike_score'] = spike_score
            result['current_price'] = spike_res.get('current_price', result.get('current_price'))

        klines_1h = client.get_klines(symbol=symbol, interval='1h', limit=2)
        if klines_1h and len(klines_1h) >= 2:
            current_c = float(klines_1h[-1][4]); past_c = float(klines_1h[-2][4])
            current_v = float(klines_1h[-1][5]); past_v = float(klines_1h[-2][5])
            result['current_price'] = current_c
            result['price_change_1h_pct'] = ((current_c - past_c) / past_c) * 100 if past_c > 0 else 0
            result['volume_change_1h_pct'] = ((current_v - past_v) / past_v) * 100 if past_v > 0 else 0

    except Exception as e:
        return None

    result['weighted_dip_score'] = float(weighted_dip_score)
    result['spike_score'] = float(spike_score)
    result['power_score'] = float(weighted_dip_score * 100 + spike_score)
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
    print("--- Binance Single-Run Analysis Bot with Geometric Analysis ---")
    print("Press Ctrl+C to stop during the scan.")
    
    usdc_pairs = fetch_usdc_pairs(client)
    if not usdc_pairs:
        print("No pairs found, exiting.")
        return

    start_time = time.time()
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Scanning {len(usdc_pairs)} assets...")
    
    all_results = []
    near_misses = []
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

    # categorize
    for r in all_results:
        if r.get('weighted_dip_score',0) < MIN_WEIGHTED_DIP_SCORE:
            near_misses.append(r)
        elif not r.get('spike_score', 0) > 0:
            scan_stats['No Spike Pattern'] += 1

    # winner selection
    analysis_winner = None
    if all_results:
        analysis_winner = max(all_results, key=lambda x: x.get('weighted_dip_score',0))
    
    print_dynamic_table(all_results, near_misses, scan_stats)

    if analysis_winner:
        print(f"\n!!! WINNER: {analysis_winner['symbol']} (score: {analysis_winner.get('weighted_dip_score'):.2f}) !!!")
        perform_final_analysis(client, analysis_winner['symbol'])
    else:
        print("\nNo suitable MTF dip found in this scan.")
        
    duration = time.time() - start_time
    print(f"\nSingle-run analysis complete in {duration:.2f}s. Exiting.")

if __name__ == "__main__":
    main()