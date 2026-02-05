import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import eig
from scipy.signal import hilbert
import math
from binance.client import Client
from datetime import datetime

# =========================================
# CONSTANTS
# =========================================
PHI = (1 + math.sqrt(5)) / 2
GOLDEN_ANGLE = 2 * np.pi * (1 - 1 / PHI)

# =====================================
# 1. DATA FETCHING (Real-time)
# =====================================
def get_binance_data(symbol='BTCUSDT', interval='1h', limit=1200):
    client = Client()
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(klines, columns=[
        'time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
    ])
    
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['Open'] = pd.to_numeric(df['Open'])
    df['Close'] = pd.to_numeric(df['Close'])
    df['Volume'] = pd.to_numeric(df['Volume'])
    df['returns'] = np.log(df['Close']).diff()
    df = df.dropna()
    df.set_index('time', inplace=True)
    return df

# =====================================
# 2. BULLISH VS BEARISH VOLUME SPLIT
# =====================================
def calculate_volume_distribution(df):
    """
    Calculates the percentage of volume that was Bullish (Close > Open) 
    vs Bearish (Close < Open) over the dataframe.
    """
    # Identify Bullish and Bearish candles
    is_bull = df['Close'] > df['Open']
    is_bear = df['Close'] < df['Open']
    
    total_vol = df['Volume'].sum()
    
    if total_vol == 0:
        return 50.0, 50.0
        
    bull_vol = df[is_bull]['Volume'].sum()
    bear_vol = df[is_bear]['Volume'].sum()
    
    # Calculate percentages
    bull_pct = (bull_vol / total_vol) * 100
    bear_pct = (bear_vol / total_vol) * 100
    
    return bull_pct, bear_pct

# =====================================
# 3. FFT CYCLE FORECAST
# =====================================
def fft_cycle_forecast(prices, window=64):
    if len(prices) < window:
        window = len(prices)
    data = prices[-window:].values
    x = np.arange(window)
    z = np.polyfit(x, data, 1)
    p = np.poly1d(z)
    detrended = data - p(x)
    fft_res = np.fft.rfft(detrended)
    threshold = np.max(np.abs(fft_res)) * 0.1 
    fft_res_filtered = fft_res * (np.abs(fft_res) > threshold)
    reconstructed = np.fft.irfft(fft_res_filtered, n=window)
    clean_cycle = reconstructed + p(x)
    last_val = clean_cycle[-1]
    prev_val = clean_cycle[-2]
    slope = last_val - prev_val
    forecast_next = last_val + slope
    cycle_direction = "BULLISH" if slope > 0 else "BEARISH"
    return forecast_next, cycle_direction

# =====================================
# 4. FORECAST & EXTREMA LOGIC (With Volume Confirmation)
# =====================================
def detect_extrema_and_forecast(df):
    analysis_df = df.tail(1200) if len(df) >= 1200 else df
    min_time = analysis_df['Close'].idxmin()
    max_time = analysis_df['Close'].idxmax()
    min_price = analysis_df.loc[min_time, 'Close']
    max_price = analysis_df.loc[max_time, 'Close']
    current_price = df['Close'].iloc[-1]
    current_vol = df['Volume'].iloc[-1]
    vol_avg = df['Volume'].rolling(window=20).mean().iloc[-1]
    
    if min_time > max_time:
        recent_type = "MINIMA"
        recent_price = min_price
        forecast_target = max_price 
        direction = "LONG"
        if current_vol > vol_avg * 1.2:
            vol_conf = "HIGH (Support)"
        elif current_vol > vol_avg * 0.9:
            vol_conf = "MODERATE"
        else:
            vol_conf = "LOW"
    else:
        recent_type = "MAXIMA"
        recent_price = max_price
        forecast_target = min_price 
        direction = "SHORT"
        if current_vol > vol_avg * 1.2:
            vol_conf = "HIGH (Pressure)"
        elif current_vol > vol_avg * 0.9:
            vol_conf = "MODERATE"
        else:
            vol_conf = "LOW"

    dist_pct = abs(forecast_target - current_price) / current_price * 100
    return {
        'direction': direction,
        'recent_type': recent_type,
        'recent_price': recent_price,
        'current_price': current_price,
        'forecast_target': forecast_target,
        'dist_pct': dist_pct,
        'vol_conf': vol_conf
    }

# =====================================
# 5. GEOMETRY MATH
# =====================================
def phi_spectral(df, window=50):
    mats = []
    if len(df) < window: return np.array([])
    for i in range(len(df) - window):
        segment = df['returns'].iloc[i:i + window].values
        M = np.outer(segment, segment)
        w, _ = eig(M)
        mats.append(np.real(np.max(w)) / PHI)
    return np.array(mats)

def quasicrystal_projection(series):
    angles = np.arange(len(series)) * GOLDEN_ANGLE
    x = series * np.cos(angles)
    y = series * np.sin(angles)
    return x, y

def log_spiral_phase(series):
    analytic = hilbert(series)
    phase = np.unwrap(np.angle(analytic))
    radius = np.exp(series)
    return phase, radius

def spin_network(series, threshold=0.001):
    G = nx.Graph()
    for i in range(len(series) - 1):
        spin = 1 if series[i] > 0 else -1
        next_spin = 1 if series[i + 1] > 0 else -1
        if abs(series[i]) > threshold:
            G.add_edge(i, i + 1, weight=spin * next_spin)
    return G

def topology_detector(series, window=30):
    topo = []
    for i in range(len(series) - window):
        seg = series[i:i + window]
        entropy = np.std(seg) / (np.mean(np.abs(seg)) + 1e-6)
        topo.append(entropy)
    return np.array(topo)

# =====================================
# MASTER ANALYSIS ENGINE
# =====================================
def analyze_timeframe(symbol, interval):
    df = get_binance_data(symbol, interval)
    if len(df) < 50: return None

    # 1. Math
    phi_spec = phi_spectral(df)
    qx, qy = quasicrystal_projection(df['returns'].values)
    spiral_phase, spiral_radius = log_spiral_phase(df['returns'].values)
    G = spin_network(df['returns'].values)
    topo = topology_detector(df['returns'].values)
    
    # 2. Forecasts
    forecast = detect_extrema_and_forecast(df)
    fft_target, fft_dir = fft_cycle_forecast(df['Close'])
    bull_pct, bear_pct = calculate_volume_distribution(df)
    
    # 3. Consensus Logic with Volume confirmation
    # Strong Signal: Geometry + FFT + Volume Agrees
    consensus = "NEUTRAL"
    if forecast['direction'] == "LONG" and fft_dir == "BULLISH":
        if bull_pct > 55.0:
            consensus = "STRONG BUY (Vol Confirmed)"
        elif bull_pct > 50.0:
            consensus = "BUY (Weak Vol)"
        else:
            consensus = "BUY (Bearish Vol - Risky)"
            
    elif forecast['direction'] == "SHORT" and fft_dir == "BEARISH":
        if bear_pct > 55.0:
            consensus = "STRONG SELL (Vol Confirmed)"
        elif bear_pct > 50.0:
            consensus = "SELL (Weak Vol)"
        else:
            consensus = "SELL (Bullish Vol - Risky)"
            
    elif forecast['direction'] == "LONG" and fft_dir == "BEARISH":
        consensus = "WEAK BUY (Cycle Conflict)"
    elif forecast['direction'] == "SHORT" and fft_dir == "BULLISH":
        consensus = "WEAK SELL (Cycle Conflict)"
        
    geometric_energy = phi_spec[-1] if len(phi_spec) > 0 else 0
    
    return {
        'interval': interval,
        'forecast': forecast,
        'fft_target': fft_target,
        'fft_dir': fft_dir,
        'consensus': consensus,
        'bull_pct': bull_pct,
        'bear_pct': bear_pct,
        'geo_energy': geometric_energy
    }

def print_dashboard(analysis_results):
    print("\n" + "="*135)
    print(f"{'META-GEOMETRY, FFT & VOLUME PROFILE FORECAST':^135}")
    print("="*135)
    
    # Header
    header = f"{'TF':<6} | {'CONSENSUS':<25} | {'EXTREMA TGT':<11} | {'FFT TGT':<11} | {'BULL VOL%':<10} | {'BEAR VOL%':<10} | {'VOL CONFIRMATION'}"
    print(header)
    print("-"*135)
    
    for res in analysis_results:
        f = res['forecast']
        tf = res['interval']
        vol_conf = f['vol_conf']
        
        # Visual cue for volume imbalance
        bull_str = f"{res['bull_pct']:.1f}%"
        bear_str = f"{res['bear_pct']:.1f}%"
        if res['bull_pct'] > 60: bull_str = f"\033[92m{bull_str}\033[0m" # Green
        if res['bear_pct'] > 60: bear_str = f"\033[91m{bear_str}\033[0m" # Red
        
        print(f"{tf:<6} | {res['consensus']:<25} | {f['forecast_target']:<11.2f} | {res['fft_target']:<11.2f} | {bull_str:<10} | {bear_str:<10} | {vol_conf}")

    # --- CYCLE SYNTHESIS ---
    # We focus on 1h, 4h, 1d for the "Overall Target" as they define the cycle
    cycle_tfs = [r for r in analysis_results if r['interval'] in ['1h', '4h', '1d']]
    micro_tfs = [r for r in analysis_results if r['interval'] in ['1m', '5m', '15m', '30m']]
    
    print("\n" + "="*135)
    print("OVERALL CYCLE & TARGET ANALYSIS")
    print("="*135)
    
    # Determine the dominant cycle based on the strongest confirmed signal in the key timeframes
    dominant_signal = None
    dominant_target = None
    dominant_tf = None
    highest_confidence = 0
    
    # Simple scoring: Strong Buy/Sell = 2, Weak = 1. Add volume % diff as weight.
    for r in cycle_tfs:
        score = 0
        if "STRONG BUY" in r['consensus']: score = 3
        elif "BUY" in r['consensus']: score = 2
        elif "STRONG SELL" in r['consensus']: score = -3
        elif "SELL" in r['consensus']: score = -2
        
        # Weight by volume imbalance
        vol_weight = abs(r['bull_pct'] - 50) * 0.1 
        
        total_conf = abs(score) + vol_weight
        
        if total_conf > highest_confidence:
            highest_confidence = total_conf
            dominant_signal = r['consensus']
            dominant_target = r['forecast']['forecast_target'] # Prefer Extrema target for the "Cycle"
            dominant_tf = r['interval']
            
    # If no strong cycle signal, look at 1h
    if dominant_signal is None:
        for r in analysis_results:
            if r['interval'] == '1h':
                dominant_signal = r['consensus']
                dominant_target = r['forecast']['forecast_target']
                dominant_tf = "1h (Default)"

    print(f"\n>>> DOMINANT CYCLE TIMEFRAME: {dominant_tf}")
    print(f">>> CYCLE DIRECTION: {dominant_signal}")
    print(f">>> OVERALL CYCLE TARGET: {dominant_target:.2f} USDC")
    
    # Micro confirmation
    bull_micro = sum(1 for r in micro_tfs if 'BUY' in r['consensus'])
    bear_micro = sum(1 for r in micro_tfs if 'SELL' in r['consensus'])
    
    print("\n--- MICRO ENTRY CONFIRMATION (Reversal Triggers) ---")
    if "BUY" in dominant_signal:
        if bull_micro > bear_micro:
            print(">>> STATUS: Optimal. Micro timeframes are aligning with the Bullish Cycle.")
            print(">>> ACTION: Look for Long entries on 15m/30m dips.")
        else:
            print(">>> STATUS: Divergence. Cycle is Bullish but Micro timeframes are weak/down.")
            print(">>> ACTION: Wait for Micro timeframes to flip Bullish before entering.")
            
    elif "SELL" in dominant_signal:
        if bear_micro > bull_micro:
            print(">>> STATUS: Optimal. Micro timeframes are aligning with the Bearish Cycle.")
            print(">>> ACTION: Look for Short entries on 15m/30m pumps.")
        else:
            print(">>> STATUS: Divergence. Cycle is Bearish but Micro timeframes are weak/up.")
            print(">>> ACTION: Wait for Micro timeframes to flip Bearish before entering.")
    else:
        print(">>> STATUS: Neutral Cycle. Wait for direction.")

    print("="*135 + "\n")

if __name__ == "__main__":
    TRADING_PAIR = "BTCUSDT"
    TIMEFRAMES = [
        '1m', '3m', '5m', '15m', '30m', 
        '1h', '2h', '4h', '6h', '8h', '12h', 
        '1d', '3d', '1w'
    ]

    all_results = []
    print(f"\nRunning Meta-Geometry & Volume Profile Engine for {TRADING_PAIR}...")
    
    for tf in TIMEFRAMES:
        try:
            result = analyze_timeframe(TRADING_PAIR, tf)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error processing timeframe {tf}: {e}")

    if all_results:
        print_dashboard(all_results)
    else:
        print("No data retrieved. Check connection.")