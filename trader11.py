import numpy as np
import pandas as pd
import time
import signal
import threading
import talib as ta
from datetime import datetime
from scipy.signal import argrelextrema
from scipy.fft import fft, fftfreq
from scipy.stats import linregress
from binance.client import Client

# ==========================
# CONFIG
# ==========================
FAST_TF = ["1m", "3m", "5m"]
INTER_TF = ["15m", "30m", "1h", "2h"]
MAJOR_TF = ["4h", "6h", "8h"]
BIGGEST_TF = ["12h", "1d", "1w"]
LOOKBACK = 500
client = Client()

# ==========================
# USER INPUT FOR SYMBOL
# ==========================
SYMBOL = input("Enter symbol (e.g., BTCUSDC, ETHUSDC): ").upper().strip()
if not SYMBOL.endswith("USDC"):
    print("Warning: Symbol should end with USDC (spot pair assumed)")
    time.sleep(1.5)

# ==========================
# STOP EVENT
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
        self.fft_forecasts = {}
        self.liquidity_sweeps = {}
        self.stop_hunt_prob = {}
        self.magnet_zones = {}
        self.exhaustion_levels = {}
        self.vol_regimes = {}
        self.divergence_pressure = 0
        self.dominant_cycles = {}
        self.ht_sine_signals = {}
        self.turn_projections = {}

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
# ANALYSIS FUNCTIONS
# ==========================
def get_dominant_cycle_fft(price):
    n = len(price)
    if n < 50:
        return 20
    detrended = price - ta.SMA(price, 20)
    fft_vals = fft(detrended)
    power = np.abs(fft_vals)**2
    freqs = fftfreq(n)
    positive_mask = freqs > 0
    if not np.any(positive_mask):
        return 20
    dominant_freq = freqs[positive_mask][np.argmax(power[positive_mask])]
    if dominant_freq != 0:
        period = int(abs(1 / dominant_freq))
    else:
        period = 20
    return max(10, min(period, 80))

def detect_ht_sine_reversal(close):
    sine, leadsine = ta.HT_SINE(close)
    sine = np.nan_to_num(sine)
    leadsine = np.nan_to_num(leadsine)
    cycle_period = get_dominant_cycle_fft(close)
    signal_type = None
    last = len(close) - 1
    if (sine[last-1] > leadsine[last-1] and sine[last] < leadsine[last] and sine[last] > 0.8):
        signal_type = "TOP"
    elif (sine[last-1] < leadsine[last-1] and sine[last] > leadsine[last] and sine[last] < -0.8):
        signal_type = "BOTTOM"
    return signal_type, cycle_period

def get_fft_forecast(close, forecast_len=20, top_k=3):
    n = len(close)
    if n < 50:
        return close[-1], close[-1], close[-1]
    t = np.arange(n)
    try:
        coeffs = np.polyfit(t, close, 1)
        trend = np.polyval(coeffs, t)
        detrended = close - trend
    except:
        detrended = close - np.mean(close)
        trend = np.linspace(0,0,n)
    window = np.hanning(n)
    windowed_data = detrended * window
    fft_vals = fft(windowed_data)
    freqs = fftfreq(n)
    magnitudes = np.abs(fft_vals[1:n//2])
    top_indices = np.argsort(magnitudes)[-top_k:] + 1
    future_t = np.arange(n, n + forecast_len)
    forecast = np.zeros(forecast_len)
    for idx in top_indices:
        amp = np.abs(fft_vals[idx]) / (n/2)
        phase = np.angle(fft_vals[idx])
        freq = freqs[idx]
        if freq != 0:
            forecast += amp * np.cos(2 * np.pi * freq * future_t + phase)
    future_trend = np.polyval(coeffs, future_t)
    forecast_price = forecast + future_trend
    return float(np.mean(forecast_price)), float(np.max(forecast_price)), float(np.min(forecast_price))

def spectral_phase(close):
    f = fft(close)
    return float(np.angle(f[1]))

def extrema(close):
    arr = np.array(close)
    mins = argrelextrema(arr, np.less_equal, order=5)[0]
    maxs = argrelextrema(arr, np.greater_equal, order=5)[0]
    return mins, maxs

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

def update_stable_target(current_price, target_attr, cycle_attr, new_cycle, new_target, price):
    try:
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
    setattr(state, target_attr, new_target)

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
    if score == 1: return "Strongly bullish alignment"
    elif score > 0.3: return "Mostly bullish"
    elif score > 0: return "Slightly bullish"
    elif score == 0: return "Neutral / mixed"
    elif score > -0.3: return "Slightly bearish"
    elif score > -1: return "Mostly bearish"
    elif score == -1: return "Strongly bearish"
    else: return "Unknown"

def mtf_harmonic_target_full(fast_data, inter_data, major_data, biggest_data,
                             weight_fast=0.4, weight_inter=0.3, weight_major=0.2, weight_biggest=0.1):
    if not all([fast_data, inter_data, major_data, biggest_data]):
        return None, {}, None, {}
    price_fast = np.mean(fast_data['forecasts'])
    price_inter = np.mean(inter_data['forecasts'])
    price_major = np.mean(major_data['forecasts'])
    price_biggest = np.mean(biggest_data['forecasts'])
    harmonic_target = (price_fast * weight_fast +
                       price_inter * weight_inter +
                       price_major * weight_major +
                       price_biggest * weight_biggest)
    cycles = (fast_data['per_tf_cycle'] +
              inter_data['per_tf_cycle'] +
              major_data['per_tf_cycle'] +
              biggest_data['per_tf_cycle'])
    consensus_cycle = "UP" if cycles.count("UP") >= cycles.count("DOWN") else "DOWN"
    details = {
        "weighted_fast": price_fast * weight_fast,
        "weighted_inter": price_inter * weight_inter,
        "weighted_major": price_major * weight_major,
        "weighted_biggest": price_biggest * weight_biggest,
        "consensus_cycle": consensus_cycle
    }
    fib_fast = np.mean(fast_data['fib_bests'])
    fib_inter = np.mean(inter_data['fib_bests'])
    fib_major = np.mean(major_data['fib_bests'])
    fib_biggest = np.mean(biggest_data['fib_bests'])
    harmonic_fib_target = (fib_fast * weight_fast +
                           fib_inter * weight_inter +
                           fib_major * weight_major +
                           fib_biggest * weight_biggest)
    fib_details = {
        "weighted_fast": fib_fast * weight_fast,
        "weighted_inter": fib_inter * weight_inter,
        "weighted_major": fib_major * weight_major,
        "weighted_biggest": fib_biggest * weight_biggest,
        "consensus_cycle": consensus_cycle
    }
    return harmonic_target, details, harmonic_fib_target, fib_details

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
    sweep_list = []
    sh_prob_list = []
    magnet_list = []
    exhaustion_list = []
    regime_list = []
    forecast_list = []
    abs_low_list = []
    abs_high_list = []
    recent_extrema_type_list = []
    recent_extrema_val_list = []
    fib_levels_list = []
    fib_bests = []
    abs_low_500_list = []
    abs_high_500_list = []
    range_200_list = []
    range_500_list = []
    recent_type_500_list = []
    recent_val_500_list = []
    extrema_comp_list = []
    idx_low_200_list = []
    idx_high_200_list = []
    idx_low_500_list = []
    idx_high_500_list = []
    reg_upper_500_list = []
    reg_lower_500_list = []
    reg_slope_500_list = []
    reg_upper_200_list = []
    reg_lower_200_list = []
    reg_slope_200_list = []
    processed_tfs = []

    for tf in tf_list:
        if stop_event.is_set():
            break
        df = get_data(SYMBOL, tf)
        if len(df) < 30:
            print(f"Skipping {tf} — insufficient data ({len(df)} candles)")
            continue
        processed_tfs.append(tf)
        close = df["close"].values
        low = df["low"].values
        high = df["high"].values
        current_price = close[-1]
        reversal_signal, dom_cycle = detect_ht_sine_reversal(close)
        state.dominant_cycles[tf] = dom_cycle
        state.ht_sine_signals[tf] = reversal_signal
        bars_to_turn = dom_cycle // 2 if reversal_signal != "NONE" else dom_cycle // 4
        state.turn_projections[tf] = bars_to_turn
        window_len_200 = min(200, len(df))
        low_200 = low[-window_len_200:]
        high_200 = high[-window_len_200:]
        abs_low = float(np.min(low_200))
        abs_high = float(np.max(high_200))
        idx_low_200 = int(np.argmin(low_200))
        idx_high_200 = int(np.argmax(high_200))
        range_200 = abs_high - abs_low
        if idx_low_200 > idx_high_200:
            cycle = "UP"
            most_recent_type = "LOW"
            most_recent_val = abs_low
        else:
            cycle = "DOWN"
            most_recent_type = "HIGH"
            most_recent_val = abs_high
        window_len_500 = min(500, len(df))
        low_500 = low[-window_len_500:]
        high_500 = high[-window_len_500:]
        abs_low_500 = float(np.min(low_500))
        abs_high_500 = float(np.max(high_500))
        idx_low_500 = int(np.argmin(low_500))
        idx_high_500 = int(np.argmax(high_500))
        range_500 = abs_high_500 - abs_low_500
        if idx_low_500 > idx_high_500:
            most_recent_type_500 = "LOW"
            most_recent_val_500 = abs_low_500
        else:
            most_recent_type_500 = "HIGH"
            most_recent_val_500 = abs_high_500
        extrema_comp = f"200: {most_recent_type} ({most_recent_val:.4f}) @ idx {idx_low_200 if most_recent_type=='LOW' else idx_high_200} | 500: {most_recent_type_500} ({most_recent_val_500:.4f}) @ idx {idx_low_500 if most_recent_type_500=='LOW' else idx_high_500}"
        fft_mean, fft_high, fft_low = get_fft_forecast(close, forecast_len=10, top_k=3)
        state.fft_forecasts[tf] = (fft_mean, fft_high, fft_low)
        bull_vol_pct, bear_vol_pct = bullish_bearish_vol(df)
        sweep_status, sh_prob = detect_sweep_and_stop_hunt(df, cycle)
        state.liquidity_sweeps[tf] = sweep_status
        state.stop_hunt_prob[tf] = sh_prob
        magnet_zone = forced_liquidation_magnet(df, cycle)
        state.magnet_zones[tf] = magnet_zone
        if cycle == "UP":
            candidate = fft_high
            if candidate < current_price:
                candidate = magnet_zone if magnet_zone > current_price else current_price * 1.005
            if bull_vol_pct < bear_vol_pct and sweep_status != "BULL_SWEEP":
                if fft_mean > current_price:
                    candidate = fft_mean
            target = max(candidate, current_price * 1.001)
        else:
            candidate = fft_low
            if candidate > current_price:
                candidate = magnet_zone if magnet_zone < current_price else current_price * 0.995
            if bear_vol_pct < bull_vol_pct and sweep_status != "BEAR_SWEEP":
                if fft_mean < current_price:
                    candidate = fft_mean
            target = min(candidate, current_price * 0.999)
        state.phase_score[tf], state.vol_comp[tf] = phase_transition_score(close)
        state.bull_bear_vol[tf] = (bull_vol_pct, bear_vol_pct)
        exh_score = reversal_exhaustion_detection(df)
        state.exhaustion_levels[tf] = exh_score
        regime = volatility_regime_switching(close)
        state.vol_regimes[tf] = regime
        if cycle == "UP":
            fib_0 = abs_low
            fib_1 = abs_high
            direction = 1
        else:
            fib_0 = abs_high
            fib_1 = abs_low
            direction = -1
        fib_diff = abs(fib_1 - fib_0)
        fib_levels = {}
        ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 0.886, 1.0]
        for r in ratios:
            fib_levels[r] = fib_0 + direction * r * fib_diff
        fib_levels_values = list(fib_levels.values())
        distances = [abs(v - fft_mean) for v in fib_levels_values]
        fib_best = fib_levels_values[np.argmin(distances)]
        if window_len_500 >= 2:
            t_500 = np.arange(window_len_500)
            slope_500, intercept_500, _, _, _ = linregress(t_500, close[-window_len_500:])
            reg_line_500 = intercept_500 + slope_500 * t_500
            res_500 = close[-window_len_500:] - reg_line_500
            std_500 = np.std(res_500) if len(res_500) > 1 else 0
            reg_upper_500 = reg_line_500[-1] + 2 * std_500
            reg_lower_500 = reg_line_500[-1] - 2 * std_500
        else:
            reg_upper_500 = current_price
            reg_lower_500 = current_price
            slope_500 = 0
        if window_len_200 >= 2:
            t_200 = np.arange(window_len_200)
            slope_200, intercept_200, _, _, _ = linregress(t_200, close[-window_len_200:])
            reg_line_200 = intercept_200 + slope_200 * t_200
            res_200 = close[-window_len_200:] - reg_line_200
            std_200 = np.std(res_200) if len(res_200) > 1 else 0
            reg_upper_200 = reg_line_200[-1] + 2 * std_200
            reg_lower_200 = reg_line_200[-1] - 2 * std_200
        else:
            reg_upper_200 = current_price
            reg_lower_200 = current_price
            slope_200 = 0
        cycles.append(cycle)
        targets.append(target)
        prices.append(current_price)
        vol_bull.append(bull_vol_pct)
        vol_bear.append(bear_vol_pct)
        support_list.append(abs_low)
        resistance_list.append(abs_high)
        sweep_list.append(sweep_status)
        sh_prob_list.append(sh_prob)
        magnet_list.append(magnet_zone)
        exhaustion_list.append(exh_score)
        regime_list.append(regime)
        forecast_list.append(target)
        abs_low_list.append(abs_low)
        abs_high_list.append(abs_high)
        recent_extrema_type_list.append(most_recent_type)
        recent_extrema_val_list.append(most_recent_val)
        fib_levels_list.append(fib_levels)
        fib_bests.append(fib_best)
        abs_low_500_list.append(abs_low_500)
        abs_high_500_list.append(abs_high_500)
        range_200_list.append(range_200)
        range_500_list.append(range_500)
        recent_type_500_list.append(most_recent_type_500)
        recent_val_500_list.append(most_recent_val_500)
        extrema_comp_list.append(extrema_comp)
        idx_low_200_list.append(idx_low_200)
        idx_high_200_list.append(idx_high_200)
        idx_low_500_list.append(idx_low_500)
        idx_high_500_list.append(idx_high_500)
        reg_upper_500_list.append(reg_upper_500)
        reg_lower_500_list.append(reg_lower_500)
        reg_slope_500_list.append(slope_500)
        reg_upper_200_list.append(reg_upper_200)
        reg_lower_200_list.append(reg_lower_200)
        reg_slope_200_list.append(slope_200)

    if not cycles:
        return None

    price = np.mean(prices) if prices else 0.0
    overall_cycle = "UP" if cycles.count("UP") >= cycles.count("DOWN") else "DOWN"

    valid_targets = [targets[i] for i in range(len(cycles)) if cycles[i] == overall_cycle]
    if valid_targets:
        target = np.mean(valid_targets)
    else:
        raw_avg = np.mean(targets) if targets else 0.0
        if overall_cycle == "UP":
            target = max(raw_avg, price * 1.001) if price > 0 else 0.0
        else:
            target = min(raw_avg, price * 0.999) if price > 0 else 0.0

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
        "forecasts": forecast_list,
        "fib_levels": fib_levels_list,
        "fib_bests": fib_bests,
        "abs_lows_500": abs_low_500_list,
        "abs_highs_500": abs_high_500_list,
        "ranges_200": range_200_list,
        "ranges_500": range_500_list,
        "recent_types_500": recent_type_500_list,
        "recent_vals_500": recent_val_500_list,
        "extrema_comps": extrema_comp_list,
        "idx_low_200": idx_low_200_list,
        "idx_high_200": idx_high_200_list,
        "idx_low_500": idx_low_500_list,
        "idx_high_500": idx_high_500_list,
        "reg_uppers_500": reg_upper_500_list,
        "reg_lowers_500": reg_lower_500_list,
        "reg_slopes_500": reg_slope_500_list,
        "reg_uppers_200": reg_upper_200_list,
        "reg_lowers_200": reg_lower_200_list,
        "reg_slopes_200": reg_slope_200_list
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
        likely_tf = max(scores, key=scores.get)
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
# MAIN ENGINE – ONE SHOT
# ==========================
print("=== INSTITUTIONAL PREDICTIVE CYCLE ENGINE STARTED ===")

def fmt(val):
    return f"{val:.4f}" if val is not None and val != 0 else "N/A"

try:
    fast_data = process_group(FAST_TF)
    inter_data = process_group(INTER_TF)
    major_data = process_group(MAJOR_TF)
    biggest_data = process_group(BIGGEST_TF)

    if fast_data is None or inter_data is None or major_data is None or biggest_data is None:
        print("Data fetch failed for one or more groups.")
    else:
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

        print("\n", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"Symbol: {SYMBOL}    Current Price: {price:.4f}\n")

        print(f"FAST   Cycle: {fast_data['cycle']}   Target: {fmt(state.fast_target)}")
        print(f"INTER  Cycle: {inter_data['cycle']}   Target: {fmt(state.inter_target)}")
        print(f"MAJOR  Cycle: {major_data['cycle']}   Target: {fmt(state.major_target)}")
        print(f"BIGGEST Cycle: {biggest_data['cycle']}   Target: {fmt(state.biggest_target)}")
        print(f"Cross-TF Pressure: {div_text}")

        rev_type, rev_target, fast_aligned = get_smart_reversal_scan(fast_data, inter_data, major_data)

        print("\n=== MTF SMART REVERSAL SCAN ===")
        print(f"INCOMING REVERSAL     : {rev_type}")
        print(f"MAJOR REVERSAL TARGET : {fmt(rev_target)}")
        print(f"FAST TARGET (Aligned) : {fmt(fast_aligned)}")

        harmonic_target, harmonic_details, harmonic_fib_target, harmonic_fib_details = mtf_harmonic_target_full(
            fast_data, inter_data, major_data, biggest_data
        )

        if harmonic_target is not None:
            print("\n=== MTF HARMONIC TARGET (4-LAYER) ===")
            print(f"Projected Target Price: {harmonic_target:.4f}")
            print("Contributions Breakdown:")
            for k, v in harmonic_details.items():
                if k != "consensus_cycle":
                    print(f" {k:>15}: {v:.4f}")
            print(f"Consensus Cycle Across All Layers: {harmonic_details['consensus_cycle']}")

            print("\n=== MTF FIBO HARMONIC TARGET (4-LAYER) ===")
            print(f"Projected Fib Best Target Price: {harmonic_fib_target:.4f}")
            print("Fib Contributions Breakdown:")
            for k, v in harmonic_fib_details.items():
                if k != "consensus_cycle":
                    print(f" {k:>15}: {v:.4f}")
            print(f"Consensus Cycle Across All Layers: {harmonic_fib_details['consensus_cycle']}")

        print("\nPer-TF Analysis:")
        all_data_map = [
            (FAST_TF, fast_data),
            (INTER_TF, inter_data),
            (MAJOR_TF, major_data),
            (BIGGEST_TF, biggest_data)
        ]

        for tf_group, data in all_data_map:
            if data is None or not data.get('per_tf_cycle'):
                print(f"\nGroup {tf_group} — no valid data available")
                continue

            for idx in range(len(data['per_tf_cycle'])):
                tf = tf_group[idx] if idx < len(tf_group) else f"TF-{idx+1}"
                print(f"\nTF: {tf}")
                print(f"Cycle: {data['per_tf_cycle'][idx]}")
                print(f"Target (Directional Fixed): {data['per_tf_target'][idx]:.4f}")

                ht_sig = state.ht_sine_signals.get(tf, "NONE")
                dom_cyc = state.dominant_cycles.get(tf, 0)
                bars_left = state.turn_projections.get(tf, 0)

                if ht_sig != "NONE":
                    print(f" >>> HT_SINE REVERSAL DETECTED: {ht_sig} <<<")
                    print(f" >>> DOMINANT CYCLE: {dom_cyc} bars | NEXT TURN EST: {bars_left} bars <<<")
                else:
                    print(f" HT_SINE Phase: No reversal signal (Cycle: {dom_cyc})")

                bull_vol_val = data['vol_bull'][idx]
                bear_vol_val = data['vol_bear'][idx]
                print(f" BullVol: {bull_vol_val:.2f}% vs BearVol: {bear_vol_val:.2f}%")

                abs_low = data['abs_lows'][idx]
                abs_high = data['abs_highs'][idx]
                recent_type = data['recent_types'][idx]
                recent_val = data['recent_vals'][idx]

                print(f" [200-CANDLE RANGE RULE]")
                print(f" Support (Lowest Low): {abs_low:.4f}")
                print(f" Resistance (Highest High): {abs_high:.4f}")
                print(f" Range: {data['ranges_200'][idx]:.4f}")
                print(f" Argmin Low Index (rel): {data['idx_low_200'][idx]}")
                print(f" Argmax High Index (rel): {data['idx_high_200'][idx]}")
                print(f" Most Recent Extrema: {recent_type} ({recent_val:.4f})")

                print(f" [500-CANDLE RANGE RULE (or max avail)]")
                print(f" Support (Lowest Low): {data['abs_lows_500'][idx]:.4f}")
                print(f" Resistance (Highest High): {data['abs_highs_500'][idx]:.4f}")
                print(f" Range: {data['ranges_500'][idx]:.4f}")
                print(f" Argmin Low Index (rel): {data['idx_low_500'][idx]}")
                print(f" Argmax High Index (rel): {data['idx_high_500'][idx]}")
                print(f" Most Recent Extrema: {data['recent_types_500'][idx]} ({data['recent_vals_500'][idx]:.4f})")

                print(f" Extrema Comparison: {data['extrema_comps'][idx]}")

                is_between = abs_low <= price <= abs_high
                if not is_between:
                    print(f" Close Inside Range: NO (New Range Extreme)")

                print(f" [LIQUIDITY ENGINE]")
                print(f" Regime: {data['regimes'][idx]}")
                sweep = data['sweeps'][idx]
                if sweep != "NONE":
                    print(f" ALERT: {sweep} (Prob: {data['sh_probs'][idx]:.2f})")
                print(f" Magnet Zone: {data['magnets'][idx]:.4f}")
                if data['exhaustions'][idx] > 0.6:
                    print(f" EXHAUSTION DETECTED: {data['exhaustions'][idx]:.2f}")

                print(f" PhaseScore: {state.phase_score.get(tf, 0.0):.4f}")

                fib_levels = data['fib_levels'][idx]
                fib_best = data['fib_bests'][idx]
                print(f" [FIBONACCI REVERSAL LEVELS (0.000 to 1.000)]")
                for r, level in fib_levels.items():
                    print(f" {r:.3f}: {level:.4f}")
                print(f" Forecast Fib Best Incoming Target: {fib_best:.4f}")

                print(f" [REGRESSION CHANNEL (200)]")
                print(f" Upper: {data['reg_uppers_200'][idx]:.4f}")
                print(f" Lower: {data['reg_lowers_200'][idx]:.4f}")
                print(f" Slope: {data['reg_slopes_200'][idx]:.4f}")

                print(f" [REGRESSION CHANNEL (500 / All Data)]")
                print(f" Upper: {data['reg_uppers_500'][idx]:.4f}")
                print(f" Lower: {data['reg_lowers_500'][idx]:.4f}")
                print(f" Slope: {data['reg_slopes_500'][idx]:.4f}")

        targets = {
            "FAST": state.fast_target,
            "INTER": state.inter_target,
            "MAJOR": state.major_target,
            "HARMONIC": harmonic_target
        }
        priority_order = sorted(targets.items(), key=lambda x: abs(x[1] - price) if x[1] is not None else float('inf'))
        print("\nTarget Priority (Closest First):")
        for rank, (name, t) in enumerate(priority_order, start=1):
            if t is not None:
                print(f"{rank}. {name} Target: {t:.4f} (distance: {abs(t - price):.4f})")

        print("\nResonance Score:", state.resonance_score)
        print("\nResonance Interpretation:")
        for layer, score in state.resonance_score.items():
            status = interpret_resonance(score)
            print(f"{layer}: Score={score:.2f} -> {status}")

        likely_tf, rev_score = most_likely_reversal(FAST_TF + INTER_TF + MAJOR_TF + BIGGEST_TF)
        if likely_tf:
            print(f"\nMost Likely Reversal TF: {likely_tf} (Predictive Score: {rev_score:.4f})")

except Exception as e:
    print("Engine Error:", e)
    import traceback
    traceback.print_exc()

print("Engine stopped cleanly.")