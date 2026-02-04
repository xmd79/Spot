import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import eig
from scipy.signal import hilbert
import math
from binance.client import Client
from datetime import datetime
import matplotlib.dates as mdates

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
    limit is set to 1200 as requested for extrema analysis.
    """
    client = Client()
    print(f"Fetching {limit} candles of {symbol} - {interval} from Binance...")
    
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(klines, columns=[
        'time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
    ])
    
    # Convert Time to Local Datetime objects for plotting
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    
    # Convert Close to numeric
    df['Close'] = pd.to_numeric(df['Close'])
    
    # Calculate returns
    df['returns'] = np.log(df['Close']).diff()
    df = df.dropna()
    
    # Set time as index for easy time-series plotting
    df.set_index('time', inplace=True)
    
    return df

# =====================================
# 2. FORECAST & EXTREMA LOGIC
# =====================================
def detect_extrema_and_forecast(df):
    """
    Analyzes the last 1200 values to find Min/Max.
    Determines the most recent extrema and sets a forecast target.
    """
    if len(df) < 1200:
        analysis_df = df
    else:
        analysis_df = df.tail(1200)
        
    # Get indices (timestamps) of absolute Min and Max in the window
    min_time = analysis_df['Close'].idxmin()
    max_time = analysis_df['Close'].idxmax()
    min_price = analysis_df.loc[min_time, 'Close']
    max_price = analysis_df.loc[max_time, 'Close']
    
    # Determine which occurred more recently
    if min_time > max_time:
        recent_type = "MINIMA (Low)"
        recent_time = min_time
        recent_price = min_price
        forecast_target = max_price # Forecast a return to the high
        forecast_nature = "Reversal Target (High)"
    else:
        recent_type = "MAXIMA (High)"
        recent_time = max_time
        recent_price = max_price
        forecast_target = min_price # Forecast a return to the low
        forecast_nature = "Reversal Target (Low)"
        
    return {
        'recent_type': recent_type,
        'recent_time': recent_time,
        'recent_price': recent_price,
        'forecast_target': forecast_target,
        'forecast_nature': forecast_nature
    }

# =====================================
# 3. φ EIGENVALUE SPECTRAL DECOMPOSITION
# =====================================
def phi_spectral(df, window=50):
    mats = []
    # We need to handle the index alignment later, so we store raw values
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
# MASTER META ENGINE
# =====================================
def run_meta_geometry(symbol, interval):

    print(f"\n=== ADVANCED META MARKET GEOMETRY ENGINE: {symbol} {interval} ===")

    # 1. Load Real-time Data
    df = get_binance_data(symbol, interval)
    
    if len(df) < 50:
        print("Not enough data for analysis window.")
        return

    # 2. Calculate Indicators
    print("Running φ spectral decomposition...")
    phi_spec = phi_spectral(df)

    print("Building quasicrystal lattice...")
    qx, qy = quasicrystal_projection(df['returns'].values)

    print("Detecting log-spiral attractors...")
    spiral_phase, spiral_radius = log_spiral_phase(df['returns'].values)

    print("Constructing spin network...")
    G = spin_network(df['returns'].values)

    print("Detecting topological transitions...")
    topo = topology_detector(df['returns'].values)
    
    # 3. Extrema & Forecast Analysis
    forecast_data = detect_extrema_and_forecast(df)
    
    print(f"\n--- FORECAST REPORT ({interval}) ---")
    print(f"Analysis Window: Last 1200 Candles")
    print(f"Most Recent Extrema : {forecast_data['recent_type']}")
    print(f"Extrema Price       : {forecast_data['recent_price']:.2f} USDC")
    print(f"Extrema Time        : {forecast_data['recent_time']}")
    print(f"Forecast Target     : {forecast_data['forecast_target']:.2f} USDC ({forecast_data['forecast_nature']})")
    print(f"Current Price       : {df['Close'].iloc[-1]:.2f} USDC")
    print(f"-------------------------------\n")

    # 4. Align Indices for Time-Series Plots
    # Indicators are shorter than DF due to windowing. 
    # We align them to the END of the window (the current time).
    window_phi = 50
    window_topo = 30
    window_spiral = len(df) - len(spiral_phase) # Hilbert reduces length slightly or stays same depending on padding, but here usually same
    
    # Create aligned time indices
    time_phi = df.index[window_phi:]
    time_topo = df.index[window_topo:]
    time_spiral = df.index[-len(spiral_phase):]

    # ======================
    # PLOTS
    # ======================
    
    # Setup Figure with GridSpec to organize Price vs Time vs Meta Indicators
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 2)
    
    # --- MAIN CHART: PRICE vs TIME (Vertical Scale: Real Price) ---
    ax_price = fig.add_subplot(gs[0, :]) # Top row, spans both columns
    ax_price.plot(df.index, df['Close'], label='BTC/USDC Price', color='black', alpha=0.8)
    
    # Plot Forecast Elements
    ax_price.axhline(y=forecast_data['forecast_target'], color='green', linestyle='--', alpha=0.5, label='Forecast Target')
    ax_price.scatter(forecast_data['recent_time'], forecast_data['recent_price'], color='red', s=100, zorder=5, label='Recent Extrema')
    
    ax_price.set_title(f"REAL-TIME PRICE GEOMETRY: {symbol} [{interval}]", fontsize=14, fontweight='bold')
    ax_price.set_ylabel("Price (USDC)", fontsize=12)
    ax_price.legend(loc='upper left')
    ax_price.grid(True, alpha=0.3)
    
    # Format X-axis for Time
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45)

    # --- SUBPLOTS: META INDICATORS ---
    
    # 1. φ Spectral Signal
    ax_phi = fig.add_subplot(gs[1, 0], sharex=ax_price)
    ax_phi.plot(time_phi, phi_spec, color='purple')
    ax_phi.set_title("φ Spectral Signal")
    ax_phi.grid(True, alpha=0.3)
    
    # 2. Topological Phase
    ax_topo = fig.add_subplot(gs[1, 1], sharex=ax_price)
    ax_topo.plot(time_topo, topo, color='orange')
    ax_topo.set_title("Topological Phase Metric")
    ax_topo.grid(True, alpha=0.3)

    # 3. Log Spiral Phase
    ax_spiral = fig.add_subplot(gs[2, 0], sharex=ax_price)
    ax_spiral.plot(time_spiral, spiral_phase, color='blue')
    ax_spiral.set_title("Log Spiral Phase")
    ax_spiral.grid(True, alpha=0.3)

    # 4. Quasicrystal Projection (Geometry X/Y)
    # This doesn't fit Price vs Time, so we keep it as a Geometry plot
    ax_crystal = fig.add_subplot(gs[2, 1])
    ax_crystal.scatter(qx, qy, s=2, c='green', alpha=0.6)
    ax_crystal.set_title("Quasicrystal Projection")
    ax_crystal.set_xlabel("X (Price * Cos)")
    ax_crystal.set_ylabel("Y (Price * Sin)")
    ax_crystal.grid(True, alpha=0.3)

    # 5. Quantum Spin Network
    ax_net = fig.add_subplot(gs[3, :])
    nx.draw(G, node_size=10, node_color='red', ax=ax_net, with_labels=False)
    ax_net.set_title("Quantum Spin Network Graph")

    plt.tight_layout()
    plt.show()

    print(f"=== META GEOMETRY ANALYSIS COMPLETE FOR {interval} ===\n")


if __name__ == "__main__":
    TRADING_PAIR = "BTCUSDT"
    
    # Timeframes from 1 minute to Weekly
    TIMEFRAMES = [
        '1m', '3m', '5m', '15m', '30m', 
        '1h', '2h', '4h', '6h', '8h', '12h', 
        '1d', '3d', '1w'
    ]

    for tf in TIMEFRAMES:
        try:
            run_meta_geometry(TRADING_PAIR, tf)
        except Exception as e:
            print(f"Error processing timeframe {tf}: {e}")