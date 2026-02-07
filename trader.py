import numpy as np
import pandas as pd
import time
import signal
import threading
from datetime import datetime
from scipy.signal import argrelextrema
from scipy.fft import fft
from statsmodels.tsa.stattools import adfuller
from binance.client import Client

# ==========================
# CONFIG
# ==========================

SYMBOL = "BTCUSDC"

FAST_TF = ["1m", "3m", "5m"]
INTER_TF = ["15m", "30m", "1h", "2h"]  # added 2h to mid-term group
MAJOR_TF = ["4h", "8h", "12h", "1d"]    # only high TFs

LOOKBACK = 1000
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

        self.major_target = None
        self.inter_target = None
        self.fast_target = None

        self.phase_score = {}
        self.resonance_score = {}
        self.vol_comp = {}
        self.bull_bear_vol = {}

state = CycleEngineState()

# ==========================
# DATA FETCH
# ==========================

def get_data(symbol, tf):
    klines = client.get_klines(symbol=symbol, interval=tf, limit=LOOKBACK)
    df = pd.DataFrame(klines, columns=[
        "time","open","high","low","close","vol",
        "ct","qv","nt","tb","tq","ig"
    ])
    for col in ["open", "high", "low", "close", "vol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["close", "high", "low", "vol"], inplace=True)
    return df

# ==========================
# ANALYSIS
# ==========================

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
    return float(vol / base)

def bullish_bearish_vol(df):
    bullish = df[df.close >= df.open]["vol"].sum()
    bearish = df[df.close < df.open]["vol"].sum()
    total = bullish + bearish
    return float(bullish / total) if total > 0 else 0.5, float(bearish / total) if total > 0 else 0.5

def support_resistance(close):
    mins, maxs = extrema(close)
    support = np.mean(close[mins]) if len(mins) > 0 else np.min(close)
    resistance = np.mean(close[maxs]) if len(maxs) > 0 else np.max(close)
    return support, resistance

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

    for tf in tf_list:
        if stop_event.is_set():
            break
        df = get_data(SYMBOL, tf)
        close = df["close"]

        phase = spectral_phase(close)
        cycle = detect_cycle(phase)

        target = compute_target(close, cycle)
        bull, bear = bullish_bearish_vol(df)
        support, resistance = support_resistance(close)
        vcomp = vol_compression(close)

        # save for predictive analysis
        state.phase_score[tf], state.vol_comp[tf] = phase_transition_score(close)
        state.bull_bear_vol[tf] = (bull, bear)

        cycles.append(cycle)
        targets.append(target)
        prices.append(close.iloc[-1])
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
        "per_tf_target": targets
    }

# ==========================
# MOST LIKELY REVERSAL TF
# ==========================

def most_likely_reversal(tf_list):
    scores = {}
    for tf in tf_list:
        phase_score = state.phase_score.get(tf, 0)
        vol_ratio = state.vol_comp.get(tf, 1)
        bull, bear = state.bull_bear_vol.get(tf, (0.5, 0.5))
        # predictive metric: PhaseScore magnitude * vol expansion * imbalance
        imbalance = abs(bull - bear)
        predictive_score = abs(phase_score) * (1/vol_ratio) * imbalance
        scores[tf] = predictive_score
    if scores:
        likely_tf = max(scores, key=lambda k: scores[k])
        return likely_tf, scores[likely_tf]
    return None, 0

# ==========================
# MAIN ENGINE LOOP
# ==========================

print("=== INSTITUTIONAL PREDICTIVE CYCLE ENGINE v2 STARTED ===")

while not stop_event.is_set():
    try:
        fast_data = process_group(FAST_TF)
        inter_data = process_group(INTER_TF)
        major_data = process_group(MAJOR_TF)

        if fast_data is None or inter_data is None or major_data is None:
            continue

        price = fast_data["price"]

        update_stable_target(price, "fast_target", "current_fast_cycle",
                             fast_data["cycle"], fast_data["target"], price)
        update_stable_target(price, "inter_target", "current_inter_cycle",
                             inter_data["cycle"], inter_data["target"], price)
        update_stable_target(price, "major_target", "current_major_cycle",
                             major_data["cycle"], major_data["target"], price)

        # ==========================
        # Print cycles & stable targets
        # ==========================
        print("\n", datetime.now())
        print(f"Price: {price:.2f}")
        print("FAST Cycle:", fast_data["cycle"], "StableTarget:", state.fast_target)
        print("INTER Cycle:", inter_data["cycle"], "StableTarget:", state.inter_target)
        print("MAJOR Cycle:", major_data["cycle"], "StableTarget:", state.major_target)

        # ==========================
        # Per-TF Analysis (vertical)
        # ==========================
        print("\nPer-TF Analysis:")
        all_data_map = [(FAST_TF, fast_data), (INTER_TF, inter_data), (MAJOR_TF, major_data)]
        for tf_list, data in all_data_map:
            for i, tf in enumerate(tf_list):
                print(f"\nTF: {tf}")
                print(f"Cycle: {data['per_tf_cycle'][i]}")
                print(f"Target: {data['per_tf_target'][i]:.2f}")
                print(f"BullVol: {data['vol_bull'][i]:.2f}")
                print(f"BearVol: {data['vol_bear'][i]:.2f}")
                print(f"Support: {data['support'][i]:.2f}")
                print(f"Resistance: {data['resistance'][i]:.2f}")
                print(f"PhaseScore: {state.phase_score[tf]:.4f}")

        # ==========================
        # Target Priority
        # ==========================
        targets = {
            "FAST": state.fast_target,
            "INTER": state.inter_target,
            "MAJOR": state.major_target
        }
        priority_order = sorted(targets.items(), key=lambda x: abs(x[1] - price))
        print("\nTarget Priority (closest first):")
        for rank, (name, t) in enumerate(priority_order, start=1):
            print(f"{rank}. {name} Target: {t:.2f} (distance: {abs(t - price):.2f})")

        # ==========================
        # Resonance Score & Interpretation
        # ==========================
        print("\nResonance Score:", state.resonance_score)
        print("\nResonance Interpretation:")
        for layer, score in state.resonance_score.items():
            status = interpret_resonance(score)
            print(f"{layer}: Score={score:.4f} -> {status}")

        # ==========================
        # Most Likely Reversal TF
        # ==========================
        likely_tf, rev_score = most_likely_reversal(FAST_TF + INTER_TF + MAJOR_TF)
        if likely_tf:
            print(f"\nMost Likely Reversal TF: {likely_tf} (Predictive Score: {rev_score:.4f})")

        stop_event.wait(SCAN_INTERVAL)

    except Exception as e:
        print("Engine Error:", e)
        stop_event.wait(2)

print("Engine stopped cleanly.")
