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

# Updated MAJOR_TF to include 6h as requested
MAJOR_TF = ["4h", "6h", "8h"]

# Added BIGGEST_TF for 12h, Daily, and Weekly
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

# ==========================
# ANALYSIS
# ==========================

def spectral_phase(close):
    # Calculate FFT on the full 1200 values
    f = fft(close)
    return float(np.angle(f[1]))

def extrema(close):
    arr = np.array(close)
    # Using order=5 for local extrema
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
    """
    Returns percentages (0.0 to 100.0) 
    representing the symmetrical distribution of Bull vs Bear volume.
    """
    if len(df) == 0 or "vol" not in df.columns:
        return 50.0, 50.0
    
    total_vol = df["vol"].sum()
    
    if total_vol == 0:
        return 50.0, 50.0
    
    # Taker Buy Volume (Bullish Volume)
    buy_vol = df["tb"].sum() 
    # Taker Sell Volume (Bearish Volume) = Total - Buy
    sell_vol = total_vol - buy_vol
    
    # Calculate percentages 0-100
    bull_pct = (buy_vol / total_vol) * 100.0
    bear_pct = (sell_vol / total_vol) * 100.0
    
    return float(bull_pct), float(bear_pct)

def support_resistance(close):
    mins, maxs = extrema(close)
    support = np.mean(close[mins]) if len(mins) > 0 else np.min(close)
    resistance = np.mean(close[maxs]) if len(maxs) > 0 else np.max(close)
    return support, resistance

# ==========================
# ROTATIONAL SYMMETRY ENGINE
# ==========================

def get_rotation_matrix(theta):
    """
    Returns a 2D rotation matrix R(theta).
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

def analyze_rotational_symmetry(close):
    """
    Converts price to Analytic Signal (Hilbert) to get a rotating vector.
    Projects future position using estimated angular velocity.
    Returns phase, coherence (stationarity of frequency), and predicted value.
    """
    if len(close) < 50:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # 1. Create Analytic Signal (Real=Price, Imag=Hilbert)
    # This places the price series onto a complex plane (rotating orbit)
    analytic_signal = hilbert(close)
    
    # 2. Extract Instantaneous Phase and Amplitude
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    
    # Real part is price, Imaginary part is quadrature (phase shifted)
    real_part = np.real(analytic_signal)
    imag_part = np.imag(analytic_signal)
    
    # Current State Vector [Real, Imag] = [x, y]
    current_state = np.array([real_part[-1], imag_part[-1]])
    
    # 3. Calculate Angular Velocity (Omega)
    # Phase derivative = frequency
    inst_freq = np.diff(instantaneous_phase)
    
    # Use robust average for velocity (median reduces noise impact)
    # We look at recent velocity to match current market speed
    recent_freq = inst_freq[-20:] if len(inst_freq) >= 20 else inst_freq
    avg_angular_vel = np.median(recent_freq)
    
    # 4. Coherence Score (Stability Measure)
    # If frequency variance is low, the orbit is stable (stationary circuit)
    freq_std = np.std(recent_freq)
    coherence = 1.0 / (1.0 + freq_std) # Normalized 0 to 1 (approx)
    
    # 5. Apply Rotation Matrix to Predict Future
    # state(t+1) = R(omega) * state(t)
    R = get_rotation_matrix(avg_angular_vel)
    predicted_state = R @ current_state # Matrix dot product
    
    # Predicted Price is the Real part of the new vector
    predicted_price = predicted_state[0]
    
    # Current Phase Angle (0 to 2pi)
    current_phase_deg = (instantaneous_phase[-1] % (2 * np.pi)) * (180.0 / np.pi)
    
    return current_phase_deg, avg_angular_vel, coherence, predicted_price, np.abs(analytic_signal[-1])

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
    
    body_size = abs(last['close'] - last['open'])
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
    
    # Rotational Features
    rot_phase_list = []
    rot_velocity_list = []
    rot_coherence_list = []
    rot_prediction_list = []

    # New Range Rule Lists
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

        # CYCLE LOGIC: ABSOLUTE ARGMIN/MAX OF LAST 1200 VALUES
        
        # Get the absolute values and their indices
        abs_high = np.max(high)
        abs_low = np.min(low)
        
        idx_high = np.argmax(high)
        idx_low = np.argmin(low)
        
        # Determine most recent extrema
        if idx_low > idx_high:
            cycle = "UP"
            most_recent_type = "LOW"
            most_recent_val = abs_low
        else:
            cycle = "DOWN"
            most_recent_type = "HIGH"
            most_recent_val = abs_high
            
        phase = spectral_phase(close)
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

        # === ROTATIONAL SYMMETRY CALCULATION ===
        rot_phase, rot_vel, rot_coh, rot_pred, rot_amp = analyze_rotational_symmetry(close)
        
        rot_phase_list.append(rot_phase)
        rot_velocity_list.append(rot_vel)
        rot_coherence_list.append(rot_coh)
        rot_prediction_list.append(rot_pred)
        # =======================================

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
        "rot_predictions": rot_prediction_list
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
        # Normalize imbalance to 0-1 range by dividing by 100 since bull/bear are now 0-100
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
    """
    Determines the incoming Major Reversal (MTF DIP or MTF TOP)
    based on divergence between cycles and trend alignment.
    Returns: Reversal Type, Major Target Value, Fast Target Value
    """
    # Determine Cycle Directions
    major_cycle = major_data['cycle']     # The Big Trend
    fast_cycle = fast_data['cycle']       # Current Momentum
    
    # Extract Support/Resistance levels for targeting
    # We use the mean of the group's levels for the "Reversal Target"
    major_support = np.mean(major_data['support'])
    major_resistance = np.mean(major_data['resistance'])
    
    # --- ML SCAN LOGIC ---
    reversal_type = "NEUTRAL"
    reversal_target = 0.0
    
    # SCENARIO 1: DIVERGENCE (Reversal imminent)
    # Major is UP, but Fast is turning DOWN -> Expecting a DIP to Support
    if major_cycle == "UP" and fast_cycle == "DOWN":
        reversal_type = "MTF DIP"
        reversal_target = major_support
    
    # Major is DOWN, but Fast is turning UP -> Expecting a TOP (Rally) to Resistance
    elif major_cycle == "DOWN" and fast_cycle == "UP":
        reversal_type = "MTF TOP"
        reversal_target = major_resistance
        
    # SCENARIO 2: ALIGNMENT (Continuation)
    # Both UP -> Price continues to Resistance (Top of move)
    elif major_cycle == "UP" and fast_cycle == "UP":
        reversal_type = "MTF TOP"
        reversal_target = major_resistance
        
    # Both DOWN -> Price continues to Support (Bottom of move)
    elif major_cycle == "DOWN" and fast_cycle == "DOWN":
        reversal_type = "MTF DIP"
        reversal_target = major_support

    # Determine aligned target for Fast only (Inter is excluded from print)
    # If we are targeting a DIP, Fast target is Support. 
    # If we are targeting a TOP, Fast target is Resistance.
    
    if reversal_type == "MTF DIP":
        fast_target_val = np.mean(fast_data['support'])
    else: # MTF TOP or NEUTRAL
        fast_target_val = np.mean(fast_data['resistance'])

    return reversal_type, reversal_target, fast_target_val

# ==========================
# MAIN ENGINE LOOP
# ==========================

print("=== INSTITUTIONAL PREDICTIVE CYCLE ENGINE (ROTATIONAL) STARTED ===")

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
                
                # --- UPDATED VOLUME PRINTING (0-100%, 25 decimals) ---
                bull_vol_val = data['vol_bull'][i]
                bear_vol_val = data['vol_bear'][i]
                print(f"BullVol: {bull_vol_val:.25f}%")
                print(f"BearVol: {bear_vol_val:.25f}%")
                # ------------------------------------------------------
                
                # --- ROTATIONAL SYMMETRY PRINT ---
                rot_p = data['rot_phases'][i]
                rot_v = data['rot_velocities'][i]
                rot_c = data['rot_coherences'][i]
                rot_pr = data['rot_predictions'][i]
                
                print(f"  [ROTATIONAL ENGINE]")
                print(f"  Phase (Degrees): {rot_p:.25f}")
                print(f"  Angular Velocity: {rot_v:.25f} rad/candle")
                print(f"  Cycle Coherence: {rot_c:.25f} (0=Chaotic, 1=Stable)")
                print(f"  Next Rotational Forecast: {rot_pr:.25f}")
                # ---------------------------------

                # --- UPDATED RANGE RULE PRINT ---
                abs_low = data['abs_lows'][i]
                abs_high = data['abs_highs'][i]
                recent_type = data['recent_types'][i]
                recent_val = data['recent_vals'][i]
                
                print(f"  [1200-CANDLE RANGE RULE (ARGMIN/MAX)]")
                print(f"  Support (Lowest Low): {abs_low:.25f}")
                print(f"  Resistance (Highest High): {abs_high:.25f}")
                print(f"  Most Recent Extrema: {recent_type} ({recent_val:.25f})")
                
                is_between = abs_low <= price <= abs_high
                if not is_between:
                    print(f"  Close Inside Range: NO (New Range Extreme)")
                # ------------------------------

                print(f"  [LIQUIDITY ENGINE]")
                print(f"  Regime: {data['regimes'][i]}")
                sweep = data['sweeps'][i]
                if sweep != "NONE":
                    print(f"  ALERT: {sweep} (Prob: {data['sh_probs'][i]:.25f})")
                print(f"  Magnet Zone: {data['magnets'][i]:.25f}")
                if data['exhaustions'][i] > 0.6:
                    print(f"  EXHAUSTION DETECTED: {data['exhaustions'][i]:.25f}")
                
                print(f"  PhaseScore: {state.phase_score[tf]:.25f}")

        targets = {
            "FAST": state.fast_target,
            "INTER": state.inter_target,
            "MAJOR": state.major_target
        }
        priority_order = sorted(targets.items(), key=lambda x: abs(x[1] - price))
        print("\nTarget Priority (Incoming Reversal - Closest First):")
        for rank, (name, t) in enumerate(priority_order, start=1):
            print(f"{rank}. {name} Target: {t:.25f} (distance: {abs(t - price):.25f})")

        print("\nResonance Score:", state.resonance_score)
        print("\nResonance Interpretation:")
        for layer, score in state.resonance_score.items():
            status = interpret_resonance(score)
            print(f"{layer}: Score={score:.25f} -> {status}")

        likely_tf, rev_score = most_likely_reversal(FAST_TF + INTER_TF + MAJOR_TF + BIGGEST_TF)
        if likely_tf:
            print(f"\nMost Likely Reversal TF: {likely_tf} (Predictive Score: {rev_score:.25f})")

        stop_event.wait(SCAN_INTERVAL)

    except Exception as e:
        print("Engine Error:", e)
        import traceback
        traceback.print_exc()
        stop_event.wait(2)

print("Engine stopped cleanly.")