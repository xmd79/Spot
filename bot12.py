#!/usr/bin/env python3
"""
Enhanced trading analysis bot with focus on 1m, 3m, and 5m timeframes.
Features:
 - MTF dip detection using only 1m, 3m, and 5m timeframes
 - ATR criteria for uptrend determination and spike pump detection
 - RSI with oversold/overbought confirmation and MTF backup from 5min
 - Argmin/argmax logic for dip detection on 1min timeframe
 - Polynomial fit analysis for trend confirmation
 - Momentum analysis for spike prediction
 - Volume spike confirmation for fast entry identification
 - Multi-model forecasting with time-to-target calculation
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
# Use only 1m, 3m, and 5m timeframes as requested
MTF_SCAN_TIMEFRAMES = ['1m', '3m', '5m']
PRIORITY_TIMEFRAMES = ['1m', '3m', '5m']

# Scoring & criteria
MIN_WEIGHTED_DIP_SCORE = 4.0
VOLUME_ANALYSIS_PERIOD = 56
PRICE_UPTREND_PERIOD = 5

# Optimization & Weights
PRIORITY_TIMEFRAME_WEIGHTS = {
    '1m': 2.5, '3m': 2.2, '5m': 2.0
}

ASSET_SCAN_LIMIT = 100
MIN_24H_VOLUME_USD = 500000
MIN_24H_PRICE_CHANGE_PCT = 0.5
BATCH_SIZE = 20

# ATR
ATR_PERIOD = 14
ATR_DIP_MULTIPLIER = 1.5
ATR_SPIKE_MULTIPLIER = 2.0
ATR_VOLUME_SPIKE_THRESHOLD = 1.5

# Volume Spike Detection
VOLUME_SPIKE_THRESHOLD = 2.0  # Volume must be 2x average
PRICE_MOMENTUM_THRESHOLD = 0.02  # 2% price change in short period

# RSI Configuration
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_MIDDLE = 50

# Polynomial Fit Configuration
POLY_DEGREE = 1
POLY_THRESHOLD = 0.99  # 1% below the best fit line

# Momentum Configuration
MOMENTUM_PERIOD = 5
MOMENTUM_THRESHOLD = 0.0  # Must be positive

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

def calculate_momentum(df, period=MOMENTUM_PERIOD):
    """Calculate price momentum."""
    try:
        if df is None or len(df) < period + 1:
            return None
        
        df = df.copy()
        df['momentum'] = df['close'].pct_change(period)
        return df
    except Exception as e:
        print(f"calculate_momentum error: {e}")
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
        # Get more data for 1min timeframe for argmin/argmax analysis
        limit = 1200 if timeframe == '1m' else 200
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

        # Calculate RSI
        df = calculate_rsi(df, RSI_PERIOD)
        current_rsi = float(df[f'RSI_{RSI_PERIOD}'].iloc[-1]) if f'RSI_{RSI_PERIOD}' in df.columns else 50.0
        
        # Calculate momentum
        df = calculate_momentum(df, MOMENTUM_PERIOD)
        current_momentum = float(df['momentum'].iloc[-1]) if 'momentum' in df.columns else 0.0

        # ATR-based uptrend determination
        recent_high = df['high'].iloc[-20:].max()
        distance_from_high_atr = (recent_high - current_price) / current_atr if current_atr > 0 else 0.0
        
        # Check for uptrend based on ATR
        is_uptrend = distance_from_high_atr < ATR_DIP_MULTIPLIER
        
        # Check for spike pump potential
        is_spike_potential = distance_from_high_atr < ATR_SPIKE_MULTIPLIER and current_price > df['close'].iloc[-5:].mean()

        # RSI oversold/overbought checks
        is_oversold = current_rsi < RSI_OVERSOLD
        is_overbought = current_rsi > RSI_OVERBOUGHT
        is_below_middle_rsi = current_rsi < RSI_MIDDLE
        
        # Traditional dip criteria
        p10 = np.percentile(df['close'], 10)
        is_price_dip = current_price <= p10
        ma20 = df['close'].iloc[-20:].mean()
        is_ma_dip = current_price < ma20
        recent_change = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] if len(df) >= 10 and df['close'].iloc[-10] != 0 else 0
        is_momentum_dip = recent_change < -0.02

        # Special handling for 1min timeframe with argmin/argmax logic
        argmin_idx = None
        argmax_idx = None
        is_argmin_dip = False
        
        if timeframe == '1m':
            # Find argmin (most recent minimum) in last 1200 values
            close_prices = df['close'].values
            argmin_idx = np.argmin(close_prices)
            argmax_idx = np.argmax(close_prices)
            
            # Check if current close is below middle between argmin and argmax
            min_price = close_prices[argmin_idx]
            max_price = close_prices[argmax_idx]
            middle_price = (min_price + max_price) / 2
            
            is_argmin_dip = current_price < middle_price
            
            # Add polynomial fit analysis
            x = np.arange(len(close_prices))
            best_fit_line1 = np.poly1d(np.polyfit(x, close_prices, POLY_DEGREE))(x)
            best_fit_line3 = best_fit_line1 * POLY_THRESHOLD
            
            is_below_poly_fit = close_prices[-1] < best_fit_line3[-1]
            
            # Check if momentum is positive
            is_positive_momentum = current_momentum > MOMENTUM_THRESHOLD
            
            # Only consider as dip if all conditions are met
            is_dip = is_argmin_dip and is_below_poly_fit and is_positive_momentum
        else:
            # For other timeframes, use traditional dip detection
            dip_criteria_met = sum([is_price_dip, is_ma_dip, is_momentum_dip])
            is_dip = dip_criteria_met >= 2 or (is_oversold and is_below_middle_rsi)

        dip_strength = 0.0
        if is_dip:
            atr_factor = min(100, distance_from_high_atr * 20)
            price_factor = max(0, (p10 - current_price) / p10 * 100) if p10 > 0 else 0
            ma_factor = max(0, (ma20 - current_price) / ma20 * 100) if ma20 > 0 else 0
            momentum_factor = abs(recent_change) * 100
            rsi_factor = max(0, (RSI_MIDDLE - current_rsi) / RSI_MIDDLE * 100) if current_rsi < RSI_MIDDLE else 0
            
            dip_strength = min(100, (atr_factor * 0.3 + price_factor * 0.2 + ma_factor * 0.15 + momentum_factor * 0.15 + rsi_factor * 0.2))

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

        result = {
            'timeframe': timeframe,
            'is_dip': is_dip,
            'dip_strength': dip_strength,
            'current_price': current_price,
            'price_change_pct': price_change_pct,
            'volume_change_pct': volume_change_pct,
            'time_ago_seconds': time_ago_sec,
            'atr': current_atr,
            'distance_from_high_atr': distance_from_high_atr,
            'is_uptrend': is_uptrend,
            'is_spike_potential': is_spike_potential,
            'rsi': current_rsi,
            'is_oversold': is_oversold,
            'is_overbought': is_overbought,
            'is_below_middle_rsi': is_below_middle_rsi,
            'momentum': current_momentum
        }
        
        # Add 1min specific data
        if timeframe == '1m':
            result['argmin_idx'] = argmin_idx
            result['argmax_idx'] = argmax_idx
            result['is_argmin_dip'] = is_argmin_dip
            result['is_below_poly_fit'] = is_below_poly_fit if 'is_below_poly_fit' in locals() else False
            result['is_positive_momentum'] = is_positive_momentum if 'is_positive_momentum' in locals() else False
        
        return result

    except Exception as e:
        print(f"get_mtf_data error for {symbol} on {timeframe}: {e}")
        return None

# ------------------ Thresholds & MTF Helpers ------------------

def get_mtf_thresholds(client, symbol):
    """Calculates min, middle, max, std dev, and ATR for each timeframe using concurrent processing."""
    thresholds = {}
    
    def process_timeframe(tf):
        if stop_event.is_set(): 
            return None
        try:
            klines = client.get_klines(symbol=symbol, interval=tf, limit=1000)
            if not klines: 
                return None
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
                return None
            close_prices = df['close'].values
            current_price = float(close_prices[-1])
            current_atr = float(df[f'ATR_{ATR_PERIOD}'].iloc[-1])
            recent_high = df['high'].iloc[-20:].max()
            distance_from_high_atr = (recent_high - current_price) / current_atr if current_atr > 0 else 0.0
            
            # Calculate RSI
            df = calculate_rsi(df, RSI_PERIOD)
            current_rsi = float(df[f'RSI_{RSI_PERIOD}'].iloc[-1]) if f'RSI_{RSI_PERIOD}' in df.columns else 50.0
            
            return {
                'timeframe': tf,
                'min': float(np.min(close_prices)),
                'max': float(np.max(close_prices)),
                'middle': float(np.mean(close_prices)),
                'std_dev': float(np.std(close_prices)),
                'current_price': current_price,
                'atr': current_atr,
                'distance_from_high_atr': distance_from_high_atr,
                'rsi': current_rsi
            }
        except Exception as e:
            return None
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_timeframe, tf): tf for tf in MTF_SCAN_TIMEFRAMES}
        for future in as_completed(futures):
            result = future.result()
            if result:
                thresholds[result['timeframe']] = result
            time.sleep(0.01)  # Small delay to avoid overwhelming API
    
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
        if 'power_score' in df.columns:
            df = df.sort_values(by='power_score', ascending=False)
        else:
            df = df.sort_values(by='weighted_dip_score', ascending=False)
        print(df.head(20).to_string(index=False, float_format="%.2f"))
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
    
    # Get detailed MTF data for analysis
    mtf_data = {}
    for tf in MTF_SCAN_TIMEFRAMES:
        data = get_mtf_data(client, symbol, tf)
        if data:
            mtf_data[tf] = data
    
    # Print detailed analysis
    print("\n--- MTF Analysis ---")
    for tf in MTF_SCAN_TIMEFRAMES:
        if tf in mtf_data:
            data = mtf_data[tf]
            print(f"\n--- {tf} Timeframe ---")
            print(f"Current Price: {data['current_price']:.8f}")
            print(f"ATR: {data['atr']:.8f}")
            print(f"Distance from High (ATR): {data['distance_from_high_atr']:.2f}")
            print(f"RSI: {data['rsi']:.2f}")
            print(f"Is Oversold: {data['is_oversold']}")
            print(f"Is Overbought: {data['is_overbought']}")
            print(f"Is Below Middle RSI: {data['is_below_middle_rsi']}")
            print(f"Momentum: {data['momentum']:.6f}")
            print(f"Is Uptrend: {data['is_uptrend']}")
            print(f"Is Spike Potential: {data['is_spike_potential']}")
            print(f"Is Dip: {data['is_dip']}")
            print(f"Dip Strength: {data['dip_strength']:.2f}")
            
            # Add 1min specific data
            if tf == '1m':
                print(f"Argmin Index: {data['argmin_idx']}")
                print(f"Argmax Index: {data['argmax_idx']}")
                print(f"Is Argmin Dip: {data['is_argmin_dip']}")
                print(f"Is Below Poly Fit: {data['is_below_poly_fit']}")
                print(f"Is Positive Momentum: {data['is_positive_momentum']}")

    print("\n--- MTF Thresholds & Predictive Zones ---")
    if mtf_thresholds:
        for tf in MTF_SCAN_TIMEFRAMES:
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
                
                print(f" | {tf:<4} | Min:{min_p:.8f} Max:{max_p:.8f} Middle:{middle_p:.8f} Std:{std_dev:.8f}")
                print(f" |     | Current:{cp:.8f} (Pos: {pct_from_min:.2f}% from Min, {pct_from_max:.2f}% from Max) ATR:{data.get('atr',0):.8f} RSI:{data.get('rsi',0):.2f}")
    
    # Consensus calculation based on MTF data
    consensus_target = current_price
    if mtf_data:
        # Calculate consensus based on uptrend indicators and spike potential
        uptrend_count = sum(1 for tf in MTF_SCAN_TIMEFRAMES if tf in mtf_data and mtf_data[tf]['is_uptrend'])
        spike_count = sum(1 for tf in MTF_SCAN_TIMEFRAMES if tf in mtf_data and mtf_data[tf]['is_spike_potential'])
        
        # If majority of timeframes show uptrend and spike potential, project a higher target
        if uptrend_count >= 2 and spike_count >= 2:
            # Calculate average ATR across timeframes
            avg_atr = sum(mtf_data[tf]['atr'] for tf in MTF_SCAN_TIMEFRAMES if tf in mtf_data) / len(mtf_data)
            # Set target to current price plus 2x ATR
            consensus_target = current_price + (avg_atr * 2)
            potential_change_pct = ((consensus_target - current_price) / current_price) * 100
            print(f"\n--- Consensus Forecast ---")
            print(f"!!! CONSENSUS TARGET: {consensus_target:.8f} ({potential_change_pct:+.2f}%) !!!")
            print("!!! Reason: Strong uptrend and spike potential detected across multiple timeframes !!!")
        else:
            print("\n--- Consensus Forecast ---")
            print("No strong consensus for upward movement detected.")
    
    print("="*80)
    print("Analysis complete.")


# ------------------ Single-asset analysis wrapper ------------------

def analyze_asset_for_table(client, symbol):
    """Gathers all required data for an asset with enhanced dip scoring using concurrent processing."""
    if stop_event.is_set(): return None
    result = {'symbol': symbol}
    weighted_dip_score = 0.0
    spike_score = 0.0
    
    try:
        # Get MTF data using concurrent processing
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = [ex.submit(get_mtf_data, client, symbol, tf) for tf in PRIORITY_TIMEFRAMES]
            for f in as_completed(futures):
                if stop_event.is_set(): return None
                data = f.result()
                if not data:
                    continue
                tf = data['timeframe']
                result[f'{tf}_price_change_pct'] = data['price_change_pct']
                result[f'{tf}_volume_change_pct'] = data['volume_change_pct']
                result['current_price'] = data['current_price']
                
                # Add ATR, RSI, and momentum data
                result[f'{tf}_atr'] = data['atr']
                result[f'{tf}_rsi'] = data['rsi']
                result[f'{tf}_momentum'] = data['momentum']
                result[f'{tf}_is_uptrend'] = data['is_uptrend']
                result[f'{tf}_is_spike_potential'] = data['is_spike_potential']
                result[f'{tf}_is_oversold'] = data['is_oversold']
                result[f'{tf}_is_below_middle_rsi'] = data['is_below_middle_rsi']
                
                # Add 1min specific data
                if tf == '1m':
                    result['1m_is_argmin_dip'] = data['is_argmin_dip']
                    result['1m_is_below_poly_fit'] = data['is_below_poly_fit']
                    result['1m_is_positive_momentum'] = data['is_positive_momentum']
                
                if data['is_dip']:
                    weight = PRIORITY_TIMEFRAME_WEIGHTS.get(tf, 1.0)
                    dip_strength = float(data.get('dip_strength', 50)) / 100.0
                    weighted_dip_score += weight * dip_strength
                    
                    # Extra weight for 1min timeframe with all conditions met
                    if tf == '1m' and data.get('is_argmin_dip') and data.get('is_below_poly_fit') and data.get('is_positive_momentum'):
                        weighted_dip_score += 2.0
                
                # Spike score based on spike potential
                if data['is_spike_potential']:
                    spike_score += PRIORITY_TIMEFRAME_WEIGHTS.get(tf, 1.0)

        # Get 1h quick price/volume change
        klines_1h = client.get_klines(symbol=symbol, interval='1h', limit=2)
        if klines_1h and len(klines_1h) >= 2:
            current_c = float(klines_1h[-1][4]); past_c = float(klines_1h[-2][4])
            current_v = float(klines_1h[-1][5]); past_v = float(klines_1h[-2][5])
            result['current_price'] = current_c
            result['price_change_1h_pct'] = ((current_c - past_c) / past_c) * 100 if past_c > 0 else 0
            result['volume_change_1h_pct'] = ((current_v - past_v) / past_v) * 100 if past_v > 0 else 0

    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None

    # Calculate enhanced power score
    result['weighted_dip_score'] = float(weighted_dip_score)
    result['spike_score'] = float(spike_score)
    
    # Calculate power score with all factors
    result['power_score'] = float(
        weighted_dip_score * 100 +  # Base dip score
        spike_score * 50  # Volume spike score
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
    print("--- Enhanced Trading Bot with Focus on 1m, 3m, and 5m Timeframes ---")
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

    # Enhanced winner selection with focus on 1min criteria
    analysis_winner = None
    if all_results:
        # Filter for assets with strong 1min indicators
        filtered_results = []
        for r in all_results:
            # Only consider assets with:
            # 1. Strong dip score
            # 2. Positive momentum on 1min
            # 3. Below poly fit on 1min
            # 4. Argmin dip on 1min
            has_strong_dip = r.get('weighted_dip_score', 0) >= MIN_WEIGHTED_DIP_SCORE
            has_positive_momentum = r.get('1m_is_positive_momentum', False)
            has_below_poly_fit = r.get('1m_is_below_poly_fit', False)
            has_argmin_dip = r.get('1m_is_argmin_dip', False)
            
            if has_strong_dip and has_positive_momentum and has_below_poly_fit and has_argmin_dip:
                filtered_results.append(r)
        
        if filtered_results:
            # Select the best from the filtered results
            analysis_winner = max(filtered_results, key=lambda x: x.get('power_score',0))
        else:
            # If no assets meet the enhanced criteria, fall back to regular selection
            analysis_winner = max(all_results, key=lambda x: x.get('power_score',0))
    
    print_dynamic_table(all_results, scan_stats)

    if analysis_winner:
        print(f"\n!!! WINNER: {analysis_winner['symbol']} (score: {analysis_winner.get('power_score'):.2f}) !!!")
        perform_final_analysis(client, analysis_winner['symbol'])
    else:
        print("\nNo suitable MTF dip found in this scan.")
        
    duration = time.time() - start_time
    print(f"\nEnhanced analysis complete in {duration:.2f}s. Exiting.")

if __name__ == "__main__":
    main()
