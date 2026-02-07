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
            raise FileNotFoundError("api.txt missing")
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if len(lines) < 2:
            raise ValueError("api.txt must contain API_KEY and API_SECRET")
        return lines[0], lines[1]
    except Exception as e:
        print(f"API Load Error: {e}")
        sys.exit(1)

API_KEY, API_SECRET = load_api_keys()
client = Client(API_KEY, API_SECRET)

# ==========================
# CONFIG
# ==========================
SYMBOL = "BTCUSDC"
TIMEFRAMES = ["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","1w"]
LIMIT = 1200

# ==========================
# HELPERS
# ==========================
def fmt25(x):
    return f"{float(x):.25f}"

def get_fresh_data(symbol, interval):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=LIMIT)
        if not klines:
            return None

        df = pd.DataFrame(klines, columns=[
            'time','open','high','low','close','volume',
            'ct','qav','trades','tb','tq','ig'])

        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)

        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return df
    except:
        return None

def stationary_series(series):
    if len(series) < 50:
        return series
    try:
        result = adfuller(series)
        if result[1] > 0.05:
            return series.diff().dropna()
    except:
        pass
    return series

# ==========================
# ANALYSIS
# ==========================
def analyze_tf(df, tf_name):

    if df is None or len(df) < 20:
        return None

    prices = df['close'].values
    current = prices[-1]

    # ===== VOLUME =====
    bull_mask = df['close'] > df['open']
    bear_mask = df['close'] < df['open']

    bull_vol = df.loc[bull_mask,'volume'].sum()
    bear_vol = df.loc[bear_mask,'volume'].sum()
    total_vol = bull_vol + bear_vol

    if total_vol > 0:
        bull_pct = (bull_vol/total_vol)*100
        bear_pct = (bear_vol/total_vol)*100
    else:
        bull_pct = bear_pct = 50.0

    # ===== LOCAL EXTREMA =====
    order = max(5, int(len(df)/40))
    mins = argrelextrema(prices, np.less, order=order)[0]
    maxs = argrelextrema(prices, np.greater, order=order)[0]

    support = prices[mins[-1]] if len(mins)>0 else np.min(prices)
    resistance = prices[maxs[-1]] if len(maxs)>0 else np.max(prices)

    # ===== ABS EXTREMA LAST 1200 =====
    window_prices = prices[-1200:] if len(prices)>=1200 else prices
    offset = len(prices) - len(window_prices)

    abs_min_local = np.argmin(window_prices)
    abs_max_local = np.argmax(window_prices)

    abs_min_idx = abs_min_local + offset
    abs_max_idx = abs_max_local + offset

    abs_min_val = prices[abs_min_idx]
    abs_max_val = prices[abs_max_idx]

    # ===== TREND BASED ON MOST RECENT EXTREME =====
    if abs_min_idx > abs_max_idx:
        trend = "UP CYCLE"
        recent_extreme = "ARGMIN"
    else:
        trend = "DOWN CYCLE"
        recent_extreme = "ARGMAX"

    # ===== FFT FORECAST =====
    try:
        series = stationary_series(df['close'])
        if len(series)>20:
            spectrum = np.abs(fft(series))
            upper = min(50,len(spectrum)-1)
            freq_idx = np.argmax(spectrum[1:upper])+1
            returns = df['close'].pct_change().dropna()
            amp = np.std(returns)*current
            omega = 2*np.pi/(freq_idx+1)
            phase = 0 if trend=="UP CYCLE" else np.pi
            forecast = current + amp*np.sin(phase + omega)
        else:
            forecast = current
    except:
        forecast = current

    return {
        "TF": tf_name,
        "Price": fmt25(current),
        "Sup": fmt25(support),
        "Res": fmt25(resistance),
        "BullVol%": fmt25(bull_pct),
        "BearVol%": fmt25(bear_pct),
        "ArgMinIdx": abs_min_idx,
        "ArgMaxIdx": abs_max_idx,
        "ArgMinVal": fmt25(abs_min_val),
        "ArgMaxVal": fmt25(abs_max_val),
        "RecentExt": recent_extreme,
        "Trend": trend,
        "Forecast": fmt25(forecast)
    }

# ==========================
# MAIN LOOP
# ==========================
def run_mtf_scan():

    print(f"Initializing Scanner for {SYMBOL}")

    try:
        while True:

            loop_start = time.time()
            results = []
            loaded = []

            for tf in TIMEFRAMES:
                df = get_fresh_data(SYMBOL, tf)
                if df is not None:
                    res = analyze_tf(df, tf)
                    if res:
                        results.append(res)
                        loaded.append(tf)

            if results:

                df_res = pd.DataFrame(results)

                up_score = sum(1 for r in results if r['Trend']=="UP CYCLE")
                down_score = sum(1 for r in results if r['Trend']=="DOWN CYCLE")

                if up_score>down_score:
                    overall="UP DOMINANT"
                    color="\033[92m"
                elif down_score>up_score:
                    overall="DOWN DOMINANT"
                    color="\033[91m"
                else:
                    overall="NEUTRAL"
                    color="\033[93m"

                os.system('cls' if os.name=='nt' else 'clear')

                print(f"=== {SYMBOL} LIVE SCANNER ===")
                print(pd.Timestamp.now())
                print("-"*130)
                print(df_res.to_string(index=False))
                print("-"*130)
                print(f"Loaded TF: {len(loaded)}/{len(TIMEFRAMES)}")
                print(f"MTF DOMINANCE: {color}{overall}\033[0m")
                print(f"UP: {up_score} | DOWN: {down_score}")

            elapsed=time.time()-loop_start
            time.sleep(max(0,5-elapsed))

    except KeyboardInterrupt:
        print("Stopping...")
        sys.exit(0)

if __name__=="__main__":
    run_mtf_scan()
