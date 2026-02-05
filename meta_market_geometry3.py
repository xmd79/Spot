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
    """
    Fetches real-time kline data from Binance.
    """
    client = Client()
    # print(f"Fetching {limit} candles of {symbol} - {interval}...") # Silence for cleaner output
    
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
    df['Close'] = pd.to_numeric(df['Close'])
    df['returns'] = np.log(df['Close']).diff()
    df = df.dropna()
    df.set_index('time', inplace=True)
    
    return df

# =====================================
# 2. FORECAST & EXTREMA LOGIC
# =====================================
def detect_extrema_and_forecast(df):
    """
    Analyzes the last values to find Min/Max and determine target.
    """
    analysis_df = df.tail(1200) if len(df) >= 1200 else df
        
    min_time = analysis_df['Close'].idxmin()
    max_time = analysis_df['Close'].idxmax()
    min_price = analysis_df.loc[min_time, 'Close']
    max_price = analysis_df.loc[max_time, 'Close']
    current_price = df['Close'].iloc[-1]
    
    # Determine Recent Extrema and Direction
    if min_time > max_time:
        # Most recent was a Low -> Expecting Upward Reversal
        recent_type = "MINIMA (Support)"
        recent_price = min_price
        forecast_target = max_price 
        direction = "LONG / BULLISH"
        reversal_type = "Rejection of Low"
    else:
        # Most recent was a High -> Expecting Downward Reversal
        recent_type = "MAXIMA (Resistance)"
        recent_price = max_price
        forecast_target = min_price 
        direction = "SHORT / BEARISH"
        reversal_type = "Rejection of High"

    # Calculate distance to target in percentage
    dist_pct = abs(forecast_target - current_price) / current_price * 100
        
    return {
        'direction': direction,
        'recent_type': recent_type,
        'recent_price': recent_price,
        'current_price': current_price,
        'forecast_target': forecast_target,
        'dist_pct': dist_pct,
        'reversal_type': reversal_type
    }

# =====================================
# 3. φ EIGENVALUE SPECTRAL DECOMPOSITION
# =====================================
def phi_spectral(df, window=50):
    mats = []
    # Reduced computation slightly for speed in text mode
    for i in range(len(df) - window):
        segment = df['returns'].iloc[i:i + window].values
        M = np.outer(segment, segment)
        w, _ = eig(M)
        mats.append(np.real(np.max(w)) / PHI)
    return np.array(mats)

# =====================================
# 4. QUASICRYSTAL PRICE LATTICE PROJECTION
# =====================================
def quasicrystal_projection(series):
    angles = np.arange(len(series)) * GOLDEN_ANGLE
    x = series * np.cos(angles)
    y = series * np.sin(angles)
    return x, y

# =====================================
# 5. LOG SPIRAL ATTRACTOR DETECTION
# =====================================
def log_spiral_phase(series):
    analytic = hilbert(series)
    phase = np.unwrap(np.angle(analytic))
    radius = np.exp(series)
    return phase, radius

# =====================================
# 6. QUANTUM SPIN MARKET GRAPH
# =====================================
def spin_network(series, threshold=0.001):
    G = nx.Graph()
    for i in range(len(series) - 1):
        spin = 1 if series[i] > 0 else -1
        next_spin = 1 if series[i + 1] > 0 else -1
        if abs(series[i]) > threshold:
            G.add_edge(i, i + 1, weight=spin * next_spin)
    return G

# =====================================
# 7. TOPOLOGICAL PHASE TRANSITION DETECTOR
# =====================================
def topology_detector(series, window=30):
    topo = []
    for i in range(len(series) - window):
        seg = series[i:i + window]
        entropy = np.std(seg) / (np.mean(np.abs(seg)) + 1e-6)
        topo.append(entropy)
    return np.array(topo)

# =====================================
# MASTER META ENGINE (TEXT ONLY)
# =====================================
def analyze_timeframe(symbol, interval):
    """
    Runs math calculations and returns a summary dictionary without plotting.
    """
    df = get_binance_data(symbol, interval)
    
    if len(df) < 50:
        return None

    # 1. Run Geometry Math (Silent Calculation)
    phi_spec = phi_spectral(df)
    qx, qy = quasicrystal_projection(df['returns'].values)
    spiral_phase, spiral_radius = log_spiral_phase(df['returns'].values)
    G = spin_network(df['returns'].values)
    topo = topology_detector(df['returns'].values)
    
    # 2. Forecast Logic
    forecast = detect_extrema_and_forecast(df)
    
    # 3. Add Geometric Strength (Optional synthetic metric based on Phi Energy)
    # If the last Phi spectral value is high, the geometry is "active"
    geometric_energy = phi_spec[-1] if len(phi_spec) > 0 else 0
    
    return {
        'interval': interval,
        'forecast': forecast,
        'geo_energy': geometric_energy,
        'topo_entropy': topo[-1] if len(topo) > 0 else 0
    }

def print_dashboard(analysis_results):
    """
    Prints the consolidated report connecting Small TFs (Reversals) and Big TFs (Trend).
    """
    print("\n" + "="*90)
    print(f"{'META-GEOMETRY FORECAST REPORT':^90}")
    print("="*90)
    
    # --- PART 1: DETAILED TABLE ---
    print(f"\n{'TF':<6} | {'DIRECTION':<15} | {'CURRENT PRICE':<12} | {'TARGET PRICE':<12} | {'MOVE POTENTIAL':<10} | {'GEOMETRY STATUS'}")
    print("-"*90)
    
    for res in analysis_results:
        f = res['forecast']
        tf = res['interval']
        
        # Determine Geometry Status based on Spectral Energy
        energy = res['geo_energy']
        if energy > 1e-6:
            geo_status = "ACTIVE (High Volatility)"
        else:
            geo_status = "STABLE (Low Volatility)"
            
        print(f"{tf:<6} | {f['direction']:<15} | {f['current_price']:<12.2f} | {f['forecast_target']:<12.2f} | {f['dist_pct']:<10.2f}% | {geo_status}")

    # --- PART 2: CONNECTING THE DOTS ---
    
    # Define Micro (Fast Reversals) and Macro (Big Trend)
    micro_tfs = [r for r in analysis_results if r['interval'] in ['1m', '3m', '5m', '15m', '30m']]
    macro_tfs = [r for r in analysis_results if r['interval'] in ['1d', '3d', '1w']]
    
    print("\n" + "="*90)
    print("MULTI-TIMEFRAME SYNTHESIS")
    print("="*90)
    
    # 1. Analyze Micro (Immediate Reversals)
    bull_micro = sum(1 for r in micro_tfs if 'LONG' in r['forecast']['direction'])
    bear_micro = len(micro_tfs) - bull_micro
    
    print(f"\n[FAST REVERSAL INCOMING - MICRO TFs (1m to 30m)]")
    if bull_micro > bear_micro:
        print(f"--> SIGNAL: IMMEDIATE BULLISH REVERSAL DETECTED")
        print(f"--> {bull_micro}/{len(micro_tfs)} small timeframes are targeting HIGHER prices.")
        print(f"--> Action: Watch for rejection of lows on small charts.")
    elif bear_micro > bull_micro:
        print(f"--> SIGNAL: IMMEDIATE BEARISH REVERSAL DETECTED")
        print(f"--> {bear_micro}/{len(micro_tfs)} small timeframes are targeting LOWER prices.")
        print(f"--> Action: Watch for rejection of highs on small charts.")
    else:
        print(f"--> SIGNAL: CONSOLIDATION / SIDEWAYS")
        print(f"--> Small TFs are mixed. Wait for break of structure.")

    # 2. Analyze Macro (Big Trend)
    bull_macro = sum(1 for r in macro_tfs if 'LONG' in r['forecast']['direction'])
    bear_macro = len(macro_tfs) - bull_macro
    
    print(f"\n[BIGGER TREND - MACRO TFs (1d to 1w)]")
    if bull_macro > bear_macro:
        print(f"--> MACRO TREND: BULLISH (Uptrend)")
        print(f"--> Larger timeframes suggest buying dips and targeting higher highs.")
        macro_bias = "BULLISH"
    elif bear_macro > bull_macro:
        print(f"--> MACRO TREND: BEARISH (Downtrend)")
        print(f"--> Larger timeframes suggest selling rallies and targeting lower lows.")
        macro_bias = "BEARISH"
    else:
        print(f"--> MACRO TREND: TRANSITIONING")
        macro_bias = "NEUTRAL"

    # 3. The Connection (Synthesis)
    print("\n" + "="*90)
    print("FINAL CONCLUSION")
    print("="*90)
    
    if ('bull_micro > bear_micro' in locals() and bull_micro > bear_micro) and macro_bias == "BULLISH":
        print(">>> ALIGNMENT: Fast Reversals are UP and Macro Trend is UP.")
        print(">>> STRONG BUY OPPORTUNITY. Look for Long entries on Small TFs.")
        
    elif ('bear_micro > bull_micro' in locals() and bear_micro > bull_micro) and macro_bias == "BEARISH":
        print(">>> ALIGNMENT: Fast Reversals are DOWN and Macro Trend is DOWN.")
        print(">>> STRONG SELL OPPORTUNITY. Look for Short entries on Small TFs.")
        
    elif ('bull_micro > bear_micro' in locals() and bull_micro > bear_micro) and macro_bias == "BEARISH":
        print(">>> CONFLICT: Small TFs say UP, but Macro Trend is DOWN.")
        print(">>> EXPECTATION: This is likely a 'Dead Cat Bounce' or Pullback.")
        print(">>> RECOMMENDATION: Do not chase. Look for Short setup at Key Resistance.")
        
    elif ('bear_micro > bull_micro' in locals() and bear_micro > bull_micro) and macro_bias == "BULLISH":
        print(">>> CONFLICT: Small TFs say DOWN, but Macro Trend is UP.")
        print(">>> EXPECTATION: This is likely a 'Dip' or Correction.")
        print(">>> RECOMMENDATION: Do not Panic Sell. Look for Long setup at Key Support.")
        
    else:
        print(">>> MARKET STATE: CHOPPY / UNCERTAIN")
        print(">>> Wait for clearer signals on 1h or 4h timeframes.")

    print("="*90 + "\n")


if __name__ == "__main__":
    TRADING_PAIR = "BTCUSDT"
    
    # Timeframes from 1 minute to Weekly
    TIMEFRAMES = [
        '1m', '3m', '5m', '15m', '30m', 
        '1h', '2h', '4h', '6h', '8h', '12h', 
        '1d', '3d', '1w'
    ]

    all_results = []

    print(f"\nRunning Advanced Meta Geometry Engine for {TRADING_PAIR}...")
    
    for tf in TIMEFRAMES:
        try:
            result = analyze_timeframe(TRADING_PAIR, tf)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error processing timeframe {tf}: {e}")

    # Only print dashboard if we have data
    if all_results:
        print_dashboard(all_results)
    else:
        print("No data retrieved. Check internet connection.")