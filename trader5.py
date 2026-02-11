import numpy as np
import pandas as pd
import time
import signal
import threading
from datetime import datetime
from scipy.signal import argrelextrema, hilbert
from scipy.fft import fft
from statsmodels.tsa.stattools import adfuller
from binance.client import Client

# ==========================
# CONFIG
# ==========================

SYMBOL = "BTCUSDC"

FAST_TF = ["1m", "3m", "5m"]
INTER_TF = ["15m", "30m", "1h", "2h"]
MAJOR_TF = ["4h", "6h", "8h"]
BIGGEST_TF = ["12h", "1d", "1w"]

LOOKBACK = 1200 
SCAN_INTERVAL = 5

client = Client()

# ==========================
# STOP EVENT (Ctrl+C SAFE)
# ==========================

stop_event = threading.Event()

def signal_handler(sig, frame):
    print("\nStopping Predictive Cycle Engine...")
    stop_event.set()

signal.signal(signal.SIGINT, signal_handler)

# ==========================
# STATE ENGINE
# ==========================

class CycleEngineState:
    def __init__(self):
        self.current_major_cycle = None
        self.current_inter_cycle = None
        self.current_fast_cycle = None
        self.current_biggest_cycle = None  

        self.major_target = None
        self.inter_target = None
        self.fast_target = None
        self.biggest_target = None         

        self.phase_score = {}
        self.resonance_score = {}
        self.vol_comp = {}
        self.bull_bear_vol = {}
        
        # Liquidity & Exhaustion States
        self.liquidity_sweeps = {}
        self.stop_hunt_prob = {}
        self.magnet_zones = {}
        self.exhaustion_levels = {}
        self.vol_regimes = {}
        self.divergence_pressure = 0

state = CycleEngineState()

# ==========================
# DATA FETCH
# ==========================

def get_data(symbol, tf):
    try:
        klines = client.get_klines(symbol=symbol, interval=tf, limit=LOOKBACK)
        df = pd.DataFrame(klines, columns=[
            "time","open","high","low","close","vol",
            "ct","qv","nt","tb","tq","ig"
        ])
        for col in ["open", "high", "low", "close", "vol", "tb"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["close", "high", "low", "vol"], inplace=True)
        return df
    except Exception as e:
        print(f"Data fetch error for {tf}: {e}")
        return pd.DataFrame()

# ==========================================================
# ADVANCED CYCLIC CORE & ORBITAL ENGINE LEVEL 2
# ==========================================================

def make_stationary_series(close):
    """
    Converts price into a stationary cyclic signal.
    Steps: log transform -> first difference (returns) -> z-score normalization.
    """
    close = np.array(close, dtype=float)
    if len(close) < 5:
        return close
    
    # Log transform
    log_price = np.log(close + 1e-12)
    # First difference (log returns)
    returns = np.diff(log_price)
    
    if len(returns) < 5:
        return returns
    
    # Z-score normalization
    mean = np.mean(returns)
    std = np.std(returns)
    
    if std == 0:
        return returns
    
    stationary = (returns - mean) / std
    return stationary

def complex_rotating_vector(close):
    """
    Converts price into a rotating complex vector using stationary data.
    """
    stationary = make_stationary_series(close)
    
    if len(stationary) < 20:
        return None, None, None
    
    analytic = hilbert(stationary)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    
    return analytic, amplitude, phase

def instantaneous_frequency(phase):
    """
    Extract instantaneous frequency with smoothing.
    """
    if len(phase) < 10:
        return 0.0, 0.0
    
    freq = np.diff(phase)
    recent = freq[-30:] if len(freq) >= 30 else freq
    
    inst_freq = np.median(recent)
    stability = 1.0 / (1.0 + np.std(recent))
    
    return inst_freq, stability

def nonlinear_rotational_forecast(state_vec, omega, growth=0.0):
    """
    Spiral / nonlinear oscillator projection.
    Allows amplitude expansion or contraction.
    """
    R = np.array([
        [np.cos(omega), -np.sin(omega)],
        [np.sin(omega),  np.cos(omega)]
    ])
    
    # Apply growth factor for spiral dynamics
    next_state = (1.0 + growth) * (R @ state_vec)
    return next_state

def spectral_phase(close):
    """
    Calculate FFT phase on STATIONARY series (Fixed).
    """
    stationary = make_stationary_series(close)
    if len(stationary) < 10:
        return 0.0
    
    f = fft(stationary)
    # Return angle of first non-DC component
    return float(np.angle(f[1]))

# --- ORBITAL ENGINE LEVEL 2 FUNCTIONS ---

def dominant_cycle_period(phase):
    """
    Estimates the dominant cycle period (bars) from phase velocity.
    """
    if len(phase) < 20:
        return 0.0
    
    freq = np.diff(phase)
    if len(freq) < 5:
        return 0.0
    
    omega = np.median(freq)
    if omega == 0:
        return 0.0
    
    period = (2 * np.pi) / abs(omega)
    return float(period)

def orbital_resonance_strength(amplitude):
    """
    Measures stability of the orbit's energy (amplitude).
    High resonance = stable amplitude (low variance).
    """
    if len(amplitude) < 10:
        return 0.0
    
    recent = amplitude[-30:] if len(amplitude) >= 30 else amplitude
    mean_amp = np.mean(recent)
    std_amp = np.std(recent)
    
    if mean_amp == 0:
        return 0.0
    
    # Inverse coefficient of variation
    resonance = 1.0 - (std_amp / (mean_amp + 1e-9))
    return float(max(0.0, resonance)) # Clamp to 0-1

def multi_orbit_phase_lock(phase):
    """
    Checks if short, mid, and long term phase velocities are aligned.
    """
    if len(phase) < 30:
        return 0.0
    
    short = np.diff(phase[-10:])
    mid   = np.diff(phase[-20:])
    long  = np.diff(phase[-30:])
    
    s = np.median(short)
    m = np.median(mid)
    l = np.median(long)
    
    # Low variance between timeframes = high lock
    alignment = 1.0 / (1.0 + np.std([s, m, l]))
    return float(alignment)

def toroidal_projection(state_vec, omega):
    """
    Projects state using two interacting rotations (R1, R2) to model 
    complex harmonic interaction (Toroidal dynamics).
    """
    # Primary Rotation
    R1 = np.array([
        [np.cos(omega), -np.sin(omega)],
        [np.sin(omega),  np.cos(omega)]
    ])
    
    # Secondary Rotation (Harmonic)
    R2 = np.array([
        [np.cos(omega*0.5), -np.sin(omega*0.5)],
        [np.sin(omega*0.5),  np.cos(omega*0.5)]
    ])
    
    v1 = R1 @ state_vec
    v2 = R2 @ state_vec
    
    # Average of primary and harmonic projection
    toroidal = (v1 + v2) / 2.0
    return toroidal

# --- MAIN ANALYSIS ENTRY POINT ---

def analyze_rotational_symmetry(close):
    """
    Main Engine function.
    Returns a dictionary of advanced cyclic features.
    """
    analytic, amp, phase = complex_rotating_vector(close)
    
    if analytic is None:
        return {
            "phase_deg": 0, "inst_freq": 0, "coherence": 0,
            "forecast_component": 0, "amplitude": 0,
            "dominant_cycle": 0, "resonance": 0, "phase_lock": 0,
            "spectral_phase": 0
        }
    
    # 1. Frequency & Coherence
    inst_freq, coherence = instantaneous_frequency(phase)
    
    # 2. State Vector Construction
    real = np.real(analytic)
    imag = np.imag(analytic)
    state_vec = np.array([real[-1], imag[-1]])
    
    # 3. Growth Factor (Spiral Dynamics)
    amp_recent = amp[-20:] if len(amp) >= 20 else amp
    growth = np.median(np.diff(amp_recent)) if len(amp_recent) > 5 else 0.0
    
    # 4. Nonlinear Forecast
    predicted_state = nonlinear_rotational_forecast(state_vec, inst_freq, growth=growth)
    
    # 5. Toroidal Projection
    toroidal_state = toroidal_projection(predicted_state, inst_freq)
    
    # 6. Orbital Metrics
    dominant_cycle = dominant_cycle_period(phase)
    resonance = orbital_resonance_strength(amp)
    phase_lock = multi_orbit_phase_lock(phase)
    spec_phase = spectral_phase(close)
    
    # 7. Format Results
    phase_deg = (phase[-1] % (2*np.pi)) * (180/np.pi)
    
    return {
        "phase_deg": float(phase_deg),
        "inst_freq": float(inst_freq),
        "coherence": float(coherence),
        "forecast_component": float(toroidal_state[0]),
        "amplitude": float(amp[-1]),
        "dominant_cycle": float(dominant_cycle),
        "resonance": float(resonance),
        "phase_lock": float(phase_lock),
        "spectral_phase": float(spec_phase)
    }

# ==========================
# STANDARD ANALYSIS
# ==========================

def extrema(close):
    arr = np.array(close)
    mins = argrelextrema(arr, np.less_equal, order=5)[0]
    maxs = argrelextrema(arr, np.greater_equal, order=5)[0]
    return mins, maxs

def detect_cycle(phase):
    return "UP" if phase > 0 else "DOWN"

def vol_compression(close):
    vol = np.std(close[-50:])
    base = np.std(close)
    return float(vol / base) if base > 0 else 1.0

def bullish_bearish_vol(df):
    if len(df) == 0 or "vol" not in df.columns:
        return 50.0, 50.0
    
    total_vol = df["vol"].sum()
    if total_vol == 0:
        return 50.0, 50.0
    
    buy_vol = df["tb"].sum() 
    sell_vol = total_vol - buy_vol
    
    bull_pct = (buy_vol / total_vol) * 100.0
    bear_pct = (sell_vol / total_vol) * 100.0
    
    return float(bull_pct), float(bear_pct)

def support_resistance(close):
    mins, maxs = extrema(close)
    support = np.mean(close[mins]) if len(mins) > 0 else np.min(close)
    resistance = np.mean(close[maxs]) if len(maxs) > 0 else np.max(close)
    return support, resistance

# ==========================
# LIQUIDITY & SWEEP LOGIC
# ==========================

def detect_sweep_and_stop_hunt(df, cycle):
    if len(df) < 10:
        return "NONE", 0.0

    last = df.iloc[-1]
    taker_buy = last['tb']
    taker_sell = last['vol'] - last['tb']
    delta = taker_buy - taker_sell
    delta_strength = abs(delta) / last['vol'] if last['vol'] > 0 else 0
    
    candle_range = last['high'] - last['low']
    if candle_range == 0:
        return "NONE", 0.0

    upper_wick = last['high'] - max(last['open'], last['close'])
    lower_wick = min(last['open'], last['close']) - last['low']
    
    wick_ratio_up = upper_wick / candle_range
    wick_ratio_down = lower_wick / candle_range
    
    status = "NONE"
    prob = 0.0
    
    if wick_ratio_up > 0.4 and last['close'] < last['open']:
        if delta < 0:
            status = "BEAR_SWEEP"
            prob = 0.7 + (delta_strength * 0.3)
            
    elif wick_ratio_down > 0.4 and last['close'] > last['open']:
        if delta > 0:
            status = "BULL_SWEEP"
            prob = 0.7 + (delta_strength * 0.3)

    return status, prob

def forced_liquidation_magnet(df, cycle):
    if len(df) == 0: return 0
    close = df["close"].values
    mins, maxs = extrema(close)
    
    liquidity_highs = close[maxs] if len(maxs) > 0 else np.array([close[-1]])
    liquidity_lows = close[mins] if len(mins) > 0 else np.array([close[-1]])
    
    recent_highs = liquidity_highs[-5:] if len(liquidity_highs) >= 5 else liquidity_highs
    recent_lows = liquidity_lows[-5:] if len(liquidity_lows) >= 5 else liquidity_lows
    
    high_magnet = np.mean(recent_highs)
    low_magnet = np.mean(recent_lows)
    
    current_price = close[-1]
    
    if cycle == "UP":
        return high_magnet if high_magnet > current_price else current_price * 1.005
    else:
        return low_magnet if low_magnet < current_price else current_price * 0.995

def reversal_exhaustion_detection(df):
    if len(df) < 20:
        return 0.0
    recent = df.tail(5)
    older = df.iloc[-20:-5]
    price_change_rate = abs(recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
    recent_vol_avg = recent['vol'].mean()
    older_vol_avg = older['vol'].mean()
    vol_dry_up = older_vol_avg > 0 and (recent_vol_avg / older_vol_avg < 0.8)
    score = 0.0
    if vol_dry_up and price_change_rate > 0.001:
        score = 0.8
    elif price_change_rate < 0.0005:
        score = 0.5
    return score

def volatility_regime_switching(close):
    if len(close) < 100:
        return "Neutral"
    short_vol = np.std(close[-20:])
    long_vol = np.std(close[-100:])
    ratio = short_vol / long_vol if long_vol > 0 else 1
    if ratio > 1.3:
        return "Expansion"
    elif ratio < 0.7:
        return "Compression"
    else:
        return "Transition"

# ==========================
# TARGET ENGINE
# ==========================

def compute_target(close, cycle):
    return float(np.percentile(close, 90)) if cycle == "UP" else float(np.percentile(close, 10))

def update_stable_target(current_price, target_attr, cycle_attr, new_cycle, new_target, price):
    try:
        price = float(price)
        stable_target = getattr(state, target_attr)
        stable_target = float(stable_target) if stable_target is not None else None
    except:
        return

    current_cycle = getattr(state, cycle_attr)
    if current_cycle != new_cycle:
        setattr(state, cycle_attr, new_cycle)
        setattr(state, target_attr, new_target)
        return
    if stable_target is None:
        setattr(state, target_attr, new_target)
        return
    if new_cycle == "UP" and price >= stable_target:
        setattr(state, target_attr, new_target)
    if new_cycle == "DOWN" and price <= stable_target:
        setattr(state, target_attr, new_target)

# ==========================
# PREDICTIVE FEATURES
# ==========================

def phase_transition_score(close):
    # Uses the fixed spectral_phase (stationary)
    phase = spectral_phase(close)
    vol_comp = vol_compression(close)
    score = (phase * (1 - vol_comp))
    return score, vol_comp

def resonance_alignment(cycles):
    up = cycles.count("UP")
    down = cycles.count("DOWN")
    total = len(cycles)
    if total == 0: return 0
    score = (up - down) / total
    return score

def interpret_resonance(score):
    if score == 1:
        return "Strongly bullish alignment"
    elif score > 0.3:
        return "Mostly bullish"
    elif score > 0:
        return "Slightly bullish"
    elif score == 0:
        return "Neutral / mixed"
    elif score > -0.3:
        return "Slightly bearish"
    elif score > -1:
        return "Mostly bearish"
    elif score == -1:
        return "Strongly bearish"
    else:
        return "Unknown"

# ==========================
# PROCESS TF GROUP
# ==========================

def process_group(tf_list):
    cycles = []
    targets = []
    prices = []
    vol_bull = []
    vol_bear = []
    support_list = []
    resistance_list = []
    
    # Feature Lists
    sweep_list = []
    sh_prob_list = []
    magnet_list = []
    exhaustion_list = []
    regime_list = []
    
    # Rotational Features (Expanded)
    rot_phase_list = []
    rot_velocity_list = []
    rot_coherence_list = []
    rot_prediction_list = []
    rot_dominant_cycle_list = []
    rot_resonance_list = []
    rot_phase_lock_list = []

    # Range Rule Lists
    abs_low_list = []
    abs_high_list = []
    recent_extrema_type_list = []
    recent_extrema_val_list = []

    for tf in tf_list:
        if stop_event.is_set():
            break
        df = get_data(SYMBOL, tf)
        if len(df) < 10: continue 
        
        close = df["close"].values
        low = df["low"].values
        high = df["high"].values

        # CYCLE LOGIC: ABSOLUTE ARGMIN/MAX
        abs_high = np.max(high)
        abs_low = np.min(low)
        
        idx_high = np.argmax(high)
        idx_low = np.argmin(low)
        
        if idx_low > idx_high:
            cycle = "UP"
            most_recent_type = "LOW"
            most_recent_val = abs_low
        else:
            cycle = "DOWN"
            most_recent_type = "HIGH"
            most_recent_val = abs_high
            
        target = compute_target(close, cycle)
        bull, bear = bullish_bearish_vol(df)
        
        support = abs_low
        resistance = abs_high
        
        abs_low_list.append(abs_low)
        abs_high_list.append(abs_high)
        recent_extrema_type_list.append(most_recent_type)
        recent_extrema_val_list.append(most_recent_val)

        vcomp = vol_compression(close)

        sweep_status, sh_prob = detect_sweep_and_stop_hunt(df, cycle)
        state.liquidity_sweeps[tf] = sweep_status
        state.stop_hunt_prob[tf] = sh_prob
        
        magnet_zone = forced_liquidation_magnet(df, cycle)
        state.magnet_zones[tf] = magnet_zone
        
        exh_score = reversal_exhaustion_detection(df)
        state.exhaustion_levels[tf] = exh_score
        
        regime = volatility_regime_switching(close)
        state.vol_regimes[tf] = regime
        
        sweep_list.append(sweep_status)
        sh_prob_list.append(sh_prob)
        magnet_list.append(magnet_zone)
        exhaustion_list.append(exh_score)
        regime_list.append(regime)

        # === ROTATIONAL SYMMETRY CALCULATION (UPGRADED) ===
        rot_data = analyze_rotational_symmetry(close)
        
        rot_phase_list.append(rot_data['phase_deg'])
        rot_velocity_list.append(rot_data['inst_freq'])
        rot_coherence_list.append(rot_data['coherence'])
        rot_prediction_list.append(rot_data['forecast_component'])
        rot_dominant_cycle_list.append(rot_data['dominant_cycle'])
        rot_resonance_list.append(rot_data['resonance'])
        rot_phase_lock_list.append(rot_data['phase_lock'])
        # ==================================================

        state.phase_score[tf], state.vol_comp[tf] = phase_transition_score(close)
        state.bull_bear_vol[tf] = (bull, bear)

        cycles.append(cycle)
        targets.append(target)
        prices.append(close[-1])
        vol_bull.append(bull)
        vol_bear.append(bear)
        support_list.append(support)
        resistance_list.append(resistance)

    if not cycles:
        return None

    price = np.mean(prices)
    overall_cycle = "UP" if cycles.count("UP") >= cycles.count("DOWN") else "DOWN"
    target = np.mean(targets)
    resonance = resonance_alignment(cycles)
    state.resonance_score["_".join(tf_list)] = resonance

    return {
        "cycle": overall_cycle,
        "target": target,
        "price": price,
        "vol_bull": vol_bull,
        "vol_bear": vol_bear,
        "support": support_list,
        "resistance": resistance_list,
        "per_tf_cycle": cycles,
        "per_tf_target": targets,
        "sweeps": sweep_list,
        "sh_probs": sh_prob_list,
        "magnets": magnet_list,
        "exhaustions": exhaustion_list,
        "regimes": regime_list,
        "abs_lows": abs_low_list,
        "abs_highs": abs_high_list,
        "recent_types": recent_extrema_type_list,
        "recent_vals": recent_extrema_val_list,
        # Rotational Data
        "rot_phases": rot_phase_list,
        "rot_velocities": rot_velocity_list,
        "rot_coherences": rot_coherence_list,
        "rot_predictions": rot_prediction_list,
        "rot_dominant_cycles": rot_dominant_cycle_list,
        "rot_resonances": rot_resonance_list,
        "rot_phase_locks": rot_phase_lock_list
    }

# ==========================
# MOST LIKELY REVERSAL TF
# ==========================

def most_likely_reversal(tf_list):
    scores = {}
    for tf in tf_list:
        phase_score = state.phase_score.get(tf, 0)
        vol_ratio = state.vol_comp.get(tf, 1)
        bull, bear = state.bull_bear_vol.get(tf, (50.0, 50.0))
        imbalance = abs(bull - bear) / 100.0
        predictive_score = abs(phase_score) * (1/vol_ratio) * imbalance
        scores[tf] = predictive_score
    if scores:
        likely_tf = max(scores, key=lambda k: scores[k])
        return likely_tf, scores[likely_tf]
    return None, 0

# ==========================
# SMART MTF REVERSAL LOGIC
# ==========================

def get_smart_reversal_scan(fast_data, inter_data, major_data):
    major_cycle = major_data['cycle']
    fast_cycle = fast_data['cycle']
    
    major_support = np.mean(major_data['support'])
    major_resistance = np.mean(major_data['resistance'])
    
    reversal_type = "NEUTRAL"
    reversal_target = 0.0
    
    if major_cycle == "UP" and fast_cycle == "DOWN":
        reversal_type = "MTF DIP"
        reversal_target = major_support
    elif major_cycle == "DOWN" and fast_cycle == "UP":
        reversal_type = "MTF TOP"
        reversal_target = major_resistance
    elif major_cycle == "UP" and fast_cycle == "UP":
        reversal_type = "MTF TOP"
        reversal_target = major_resistance
    elif major_cycle == "DOWN" and fast_cycle == "DOWN":
        reversal_type = "MTF DIP"
        reversal_target = major_support
    
    if reversal_type == "MTF DIP":
        fast_target_val = np.mean(fast_data['support'])
    else:
        fast_target_val = np.mean(fast_data['resistance'])

    return reversal_type, reversal_target, fast_target_val

# ==========================
# MAIN ENGINE LOOP
# ==========================

print("=== INSTITUTIONAL PREDICTIVE CYCLE ENGINE (ORBITAL L2) STARTED ===")

while not stop_event.is_set():
    try:
        fast_data = process_group(FAST_TF)
        inter_data = process_group(INTER_TF)
        major_data = process_group(MAJOR_TF)      
        biggest_data = process_group(BIGGEST_TF)  

        if fast_data is None or inter_data is None or major_data is None or biggest_data is None:
            continue

        price = fast_data["price"]

        update_stable_target(price, "fast_target", "current_fast_cycle",
                             fast_data["cycle"], fast_data["target"], price)
        update_stable_target(price, "inter_target", "current_inter_cycle",
                             inter_data["cycle"], inter_data["target"], price)
        update_stable_target(price, "major_target", "current_major_cycle",
                             major_data["cycle"], major_data["target"], price)
        update_stable_target(price, "biggest_target", "current_biggest_cycle",
                             biggest_data["cycle"], biggest_data["target"], price)
                             
        fast_bias = 1 if fast_data["cycle"] == "UP" else -1
        major_bias = 1 if major_data["cycle"] == "UP" else -1
        divergence_val = 0
        div_text = "None"
        if fast_bias != major_bias:
            divergence_val = abs(fast_bias - major_bias)
            div_text = f"HIGH DIVERGENCE (Fast:{fast_data['cycle']} vs Major:{major_data['cycle']})"
        else:
            div_text = "Aligned"
        state.divergence_pressure = divergence_val

        print("\n", datetime.now())
        print(f"Price: {price:.25f}")
        print("FAST Cycle:", fast_data["cycle"], "StableTarget:", f"{state.fast_target:.25f}")
        print("INTER Cycle:", inter_data["cycle"], "StableTarget:", f"{state.inter_target:.25f}")
        print("MAJOR Cycle:", major_data["cycle"], "StableTarget:", f"{state.major_target:.25f}")
        print("BIGGEST Cycle:", biggest_data["cycle"], "StableTarget:", f"{state.biggest_target:.25f}")
        print(f"Cross-TF Pressure: {div_text}")

        # ==========================================
        # SMART MTF REVERSAL SCAN OUTPUT
        # ==========================================
        rev_type, rev_target, fast_aligned = get_smart_reversal_scan(fast_data, inter_data, major_data)
        
        print("\n=== MTF SMART REVERSAL SCAN ===")
        print(f"INCOMING REVERSAL: {rev_type}")
        print(f"MAJOR REVERSAL TARGET: {rev_target:.25f}")
        print(f"FAST TARGET (Aligned): {fast_aligned:.25f}")
        # ==========================================

        print("\nPer-TF Analysis:")
        all_data_map = [
            (FAST_TF, fast_data), 
            (INTER_TF, inter_data), 
            (MAJOR_TF, major_data),
            (BIGGEST_TF, biggest_data)
        ]
        
        for tf_list, data in all_data_map:
            for i, tf in enumerate(tf_list):
                print(f"\nTF: {tf}")
                print(f"Cycle: {data['per_tf_cycle'][i]}")
                print(f"Target: {data['per_tf_target'][i]:.25f}")
                
                # --- VOLUME PRINTING ---
                bull_vol_val = data['vol_bull'][i]
                bear_vol_val = data['vol_bear'][i]
                print(f"BullVol: {bull_vol_val:.25f}%")
                print(f"BearVol: {bear_vol_val:.25f}%")
                
                # --- ROTATIONAL SYMMETRY PRINT (UPGRADED) ---
                rot_p = data['rot_phases'][i]
                rot_v = data['rot_velocities'][i]
                rot_c = data['rot_coherences'][i]
                rot_pr = data['rot_predictions'][i]
                rot_dc = data['rot_dominant_cycles'][i]
                rot_r = data['rot_resonances'][i]
                rot_pl = data['rot_phase_locks'][i]
                
                print(f"  [ROTATIONAL ENGINE L2]")
                print(f"  Phase: {rot_p:.2f} deg | Freq: {rot_v:.5f} rad/c | Coherence: {rot_c:.3f}")
                print(f"  Dominant Cycle: {rot_dc:.1f} bars | Resonance: {rot_r:.3f} | PhaseLock: {rot_pl:.3f}")
                print(f"  Next Forecast (Toroidal): {rot_pr:.5f}")
                # ---------------------------------

                # --- RANGE RULE PRINT ---
                abs_low = data['abs_lows'][i]
                abs_high = data['abs_highs'][i]
                recent_type = data['recent_types'][i]
                recent_val = data['recent_vals'][i]
                
                print(f"  [RANGE RULE]")
                print(f"  Support: {abs_low:.25f} | Resistance: {abs_high:.25f}")
                print(f"  Recent Extrema: {recent_type} ({recent_val:.25f})")
                # ------------------------------

                print(f"  [LIQUIDITY ENGINE]")
                print(f"  Regime: {data['regimes'][i]}")
                sweep = data['sweeps'][i]
                if sweep != "NONE":
                    print(f"  ALERT: {sweep} (Prob: {data['sh_probs'][i]:.3f})")
                print(f"  Magnet Zone: {data['magnets'][i]:.2f}")
                if data['exhaustions'][i] > 0.6:
                    print(f"  EXHAUSTION DETECTED: {data['exhaustions'][i]:.3f}")

        targets = {
            "FAST": state.fast_target,
            "INTER": state.inter_target,
            "MAJOR": state.major_target
        }
        priority_order = sorted(targets.items(), key=lambda x: abs(x[1] - price))
        print("\nTarget Priority (Closest First):")
        for rank, (name, t) in enumerate(priority_order, start=1):
            print(f"{rank}. {name} Target: {t:.25f} (distance: {abs(t - price):.2f})")

        print("\nResonance Interpretation:")
        for layer, score in state.resonance_score.items():
            status = interpret_resonance(score)
            print(f"{layer}: {status}")

        likely_tf, rev_score = most_likely_reversal(FAST_TF + INTER_TF + MAJOR_TF + BIGGEST_TF)
        if likely_tf:
            print(f"\nMost Likely Reversal TF: {likely_tf} (Score: {rev_score:.5f})")

        stop_event.wait(SCAN_INTERVAL)

    except Exception as e:
        print("Engine Error:", e)
        import traceback
        traceback.print_exc()
        stop_event.wait(2)

print("Engine stopped cleanly.")