import numpy as np
import pandas as pd
import time
import signal
import threading
from datetime import datetime
from scipy.signal import argrelextrema
from scipy.fft import fft, fftfreq
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

# LOOKBACK is 500 to provide data for FFT
LOOKBACK = 500 
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
        
        # Forecasting State
        self.fft_forecasts = {}
        
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

def get_fft_forecast(close, forecast_len=10, top_k=3):
    """
    Advanced FFT Forecast:
    1. Detrends data (Linear).
    2. Applies Hann window to reduce edge artifacts.
    3. Identifies top_k dominant frequencies.
    4. Projects those sine waves forward.
    5. Re-applies trend.
    """
    n = len(close)
    if n < 50: return close[-1] 

    # 1. Detrend
    t = np.arange(n)
    try:
        coeffs = np.polyfit(t, close, 1)
        trend = np.polyval(coeffs, t)
        detrended = close - trend
    except:
        detrended = close - np.mean(close)
        trend = np.linspace(0,0,n)

    # 2. Windowing (Hann window)
    window = np.hanning(n)
    windowed_data = detrended * window

    # 3. FFT
    fft_vals = fft(windowed_data)
    freqs = fftfreq(n)
    
    # 4. Filter Dominant Cycles
    magnitudes = np.abs(fft_vals[1:n//2])
    top_indices = np.argsort(magnitudes)[-top_k:] + 1
    
    # 5. Reconstruct Signal for Future
    future_t = np.arange(n, n + forecast_len)
    forecast = np.zeros(forecast_len)
    
    for idx in top_indices:
        amp = np.abs(fft_vals[idx]) / (n/2)
        phase = np.angle(fft_vals[idx])
        freq = freqs[idx]
        
        if freq != 0:
            forecast += amp * np.cos(2 * np.pi * freq * future_t + phase)
            
    # 6. Re-apply Trend
    future_trend = np.polyval(coeffs, future_t)
    forecast_price = forecast + future_trend
    
    # Return the mean of the forecast period as the target
    return float(np.mean(forecast_price))

def spectral_phase(close):
    f = fft(close)
    return float(np.angle(f[1]))

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
    # For FFT targets, we update them constantly as the projection evolves
    # But we still respect cycle flips
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
    
    # Forecast List
    forecast_list = []

    # Range Rule Lists
    abs_low_list = []
    abs_high_list = []
    recent_extrema_type_list = []
    recent_extrema_val_list = []

    for tf in tf_list:
        if stop_event.is_set():
            break
        df = get_data(SYMBOL, tf)
        if len(df) < 200: 
            print(f"Skipping {tf}: not enough data for 200-candle range rule.")
            continue 
        
        close = df["close"].values
        low = df["low"].values
        high = df["high"].values

        # --- FFT FORECAST (Now used as PRIMARY TARGET) ---
        forecast_price = get_fft_forecast(close, forecast_len=10, top_k=3)
        state.fft_forecasts[tf] = forecast_price
        forecast_list.append(forecast_price)

        # --- RANGE RULE LOGIC (Strictly Last 200 Values) ---
        window_len = 200
        low_200 = low[-window_len:]
        high_200 = high[-window_len:]
        
        abs_low = float(np.min(low_200))
        abs_high = float(np.max(high_200))
        
        idx_low_rel = int(np.argmin(low_200))
        idx_high_rel = int(np.argmax(high_200))
        
        if idx_low_rel > idx_high_rel:
            cycle = "UP"
            most_recent_type = "LOW"
            most_recent_val = abs_low
        else:
            cycle = "DOWN"
            most_recent_type = "HIGH"
            most_recent_val = abs_high
            
        phase = spectral_phase(close)
        
        # CHANGE: Target is now the FFT Prediction, not percentile
        target = forecast_price 
        
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
        "forecasts": forecast_list
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

print("=== INSTITUTIONAL PREDICTIVE CYCLE ENGINE STARTED ===")

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
        print("FAST Cycle:", fast_data["cycle"], "FFT Target:", f"{state.fast_target:.25f}")
        print("INTER Cycle:", inter_data["cycle"], "FFT Target:", f"{state.inter_target:.25f}")
        print("MAJOR Cycle:", major_data["cycle"], "FFT Target:", f"{state.major_target:.25f}")
        print("BIGGEST Cycle:", biggest_data["cycle"], "FFT Target:", f"{state.biggest_target:.25f}")
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
                # This target is now explicitly the FFT Forecast
                print(f"Target (FFT): {data['per_tf_target'][i]:.25f}")
                
                bull_vol_val = data['vol_bull'][i]
                bear_vol_val = data['vol_bear'][i]
                print(f"BullVol: {bull_vol_val:.25f}%")
                print(f"BearVol: {bear_vol_val:.25f}%")
                
                # --- RANGE RULE PRINT ---
                abs_low = data['abs_lows'][i]
                abs_high = data['abs_highs'][i]
                recent_type = data['recent_types'][i]
                recent_val = data['recent_vals'][i]
                
                print(f"  [200-CANDLE RANGE RULE]")
                print(f"  Support (Lowest Low): {abs_low:.25f}")
                print(f"  Resistance (Highest High): {abs_high:.25f}")
                print(f"  Most Recent Extrema: {recent_type} ({recent_val:.25f})")
                
                is_between = abs_low <= price <= abs_high
                if not is_between:
                    print(f"  Close Inside Range: NO (New Range Extreme)")

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
        print("\nTarget Priority (FFT Forecast - Closest First):")
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