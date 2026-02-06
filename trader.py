import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.fft import fft
from statsmodels.tsa.stattools import adfuller
from binance.client import Client
import time
import sys
import os

# ==========================
# LOAD API
# ==========================
def load_api_keys(filepath="api.txt"):
    try:
        if not os.path.exists(filepath):
            print("Error: api.txt not found.")
            # Create dummy file for testing if you don't have one, otherwise raise error
            # For production, raise the error.
            raise FileNotFoundError("api.txt missing")
            
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if len(lines) < 2:
            raise ValueError("api.txt must contain API_KEY on line1 and API_SECRET on line2")
        return lines[0], lines[1]
    except Exception as e:
        print(f"API Load Error: {e}")
        sys.exit(1)

API_KEY, API_SECRET = load_api_keys()
client = Client(API_KEY, API_SECRET)

# ==========================
# CONFIGURATION
# ==========================
SYMBOL = "BTCUSDC"
TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w"]
LIMIT = 1000 

# ==========================
# DATA FETCHING (FRESH EVERY TIME)
# ==========================
def get_fresh_data(symbol, interval):
    """Fetches strictly fresh OHLCV data. No caching."""
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=LIMIT)
        if not klines:
            return None
            
        df = pd.DataFrame(klines, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'ct', 'qav', 'trades', 'tb', 'tq', 'ig'])
        
        # Convert types
        for col in ['close', 'high', 'low', 'open', 'volume']:
            df[col] = df[col].astype(float)
            
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return df
    except Exception as e:
        # print(f"Fetch Error {interval}: {e}") # Optional: Silent fail to keep UI clean
        return None

# ==========================
# ANALYSIS ENGINE
# ==========================
def stationary_series(series):
    if len(series) < 50: return series
    try:
        result = adfuller(series)
        if result[1] > 0.05: return series.diff().dropna()
    except: pass
    return series

def analyze_tf(df, tf_name):
    if df is None or len(df) < 20: 
        return None

    prices = df['close'].values
    current = prices[-1]
    
    # --- 1. Volume Analysis (Bullish vs Bearish) ---
    # Symmetrical: BullVol% + BearVol% = 100%
    bull_mask = df['close'] > df['open']
    bear_mask = df['close'] < df['open']
    
    total_bull_vol = df.loc[bull_mask, 'volume'].sum()
    total_bear_vol = df.loc[bear_mask, 'volume'].sum()
    total_vol = total_bull_vol + total_bear_vol
    
    if total_vol > 0:
        bull_vol_pct = (total_bull_vol / total_vol) * 100
        bear_vol_pct = (total_bear_vol / total_vol) * 100
    else:
        bull_vol_pct, bear_vol_pct = 50.0, 50.0

    # --- 2. Support & Resistance (Pivots) ---
    # Dynamic order based on timeframe length to avoid noise
    order = max(5, int(len(df) / 40)) 
    
    try:
        mins = argrelextrema(prices, np.less, order=order)[0]
        maxs = argrelextrema(prices, np.greater, order=order)[0]
    except:
        mins, maxs = [], []

    # Support = most recent local min, Resistance = most recent local max
    support = prices[mins[-1]] if len(mins) > 0 else np.min(prices)
    resistance = prices[maxs[-1]] if len(maxs) > 0 else np.max(prices)

    # --- 3. ArgMin / ArgMax (Absolute Extremes) ---
    abs_min_idx = np.argmin(prices)
    abs_max_idx = np.argmax(prices)
    
    # What occurred most recently?
    recent_extreme = "MIN" if abs_min_idx > abs_max_idx else "MAX"

    # --- 4. Trend & Forecast ---
    # Logic: Price relative to S/R Midpoint
    mid_range = (support + resistance) / 2
    trend = "BULLISH" if current > mid_range else "BEARISH"
    
    # Simple FFT Forecast
    try:
        series = stationary_series(df['close'])
        if len(series) > 20:
            spectrum = np.abs(fft(series))
            upper = min(50, len(spectrum) - 1)
            freq_idx = np.argmax(spectrum[1:upper]) + 1
            returns = df['close'].pct_change().dropna()
            amp = np.std(returns) * current
            omega = 2 * np.pi / (freq_idx + 1)
            phase = 0 if trend == "BULLISH" else np.pi
            forecast = current + amp * np.sin(phase + omega * 1)
        else:
            forecast = current
    except:
        forecast = current

    return {
        "TF": tf_name,
        "Price": round(current, 2),
        "Sup": round(support, 2),
        "Res": round(resistance, 2),
        "BullVol%": round(bull_vol_pct, 1),
        "BearVol%": round(bear_vol_pct, 1),
        "RecentExt": recent_extreme,
        "Trend": trend, # This is the key value for the summary
        "Forecast": round(forecast, 2)
    }

# ==========================
# MAIN LOOP
# ==========================
def run_mtf_scan():
    print(f"Initializing MTF Scanner for {SYMBOL}...")
    print(f"Sequence: {' -> '.join(TIMEFRAMES)}")
    
    try:
        while True:
            loop_start = time.time()
            
            # 1. Fresh Container for this iteration
            results = []
            loaded_tfs = []
            
            # 2. Sequential Fetch & Process (Ensuring strict order and fresh data)
            for tf in TIMEFRAMES:
                # Force fetch new data every loop
                df = get_fresh_data(SYMBOL, tf)
                
                if df is not None:
                    res = analyze_tf(df, tf)
                    if res:
                        results.append(res)
                        loaded_tfs.append(tf)
            
            if results:
                # 3. Build DataFrame
                df_res = pd.DataFrame(results)
                
                # 4. MTF DOMINANCE CALCULATION (MATCHING TABLE TREND)
                # We count based on the 'Trend' column to ensure numbers match visually
                bull_score = sum(1 for r in results if r['Trend'] == "BULLISH")
                bear_score = sum(1 for r in results if r['Trend'] == "BEARISH")
                
                # Determine Color and Status
                if bull_score > bear_score:
                    overall_status = "STRONG BULLISH"
                    status_color = "\033[92m" # Green
                elif bear_score > bull_score:
                    overall_status = "STRONG BEARISH"
                    status_color = "\033[91m" # Red
                else:
                    overall_status = "NEUTRAL"
                    status_color = "\033[93m" # Yellow

                # 5. Display Output
                # Clear terminal
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print(f"=== {SYMBOL} LIVE MTF SCANNER ===")
                print(f"Last Update: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 95)
                
                # Format the table string
                table_str = df_res.to_string(index=False)
                print(table_str)
                
                print("-" * 95)
                print(f"DATA LOADED: {len(loaded_tfs)}/{len(TIMEFRAMES)} Timeframes")
                print(f"MTF DOMINANCE (Based on Trend): {status_color}{overall_status}\033[0m")
                print(f"Bullish Trends: {bull_score} | Bearish Trends: {bear_score}")
                print("-" * 95)
                print("Press Ctrl+C to Exit...")

            # 6. Wait 5 Seconds
            elapsed = time.time() - loop_start
            wait_time = max(0, 5.0 - elapsed)
            time.sleep(wait_time)

    except KeyboardInterrupt:
        print("\n\nStopping Scanner...")
        time.sleep(0.5)
        sys.exit(0)

if __name__ == "__main__":
    run_mtf_scan()