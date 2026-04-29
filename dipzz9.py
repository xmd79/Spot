import gc
from binance.client import Client
import numpy as np
import talib as ta
import time
import sys
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


# ==========================================
# PHI / GOLDEN HARMONIC CONSTANTS
# ==========================================

PHI = (1.0 + 5.0 ** 0.5) / 2.0
PHI_INV = 1.0 / PHI
PHI_SQ  = PHI * PHI

FIB_RATIOS = {
    "F236": 0.236, "F382": 0.382, "F500": 0.500,
    "F618": PHI_INV, "F786": PHI_INV ** 0.5,
}


# ==========================================
# GOLDEN HARMONIC ENGINE
# ==========================================

def golden_signal(t: np.ndarray, omega0: float = 1.0, N: int = 3) -> np.ndarray:
    x = np.zeros_like(t, dtype=float)
    for n in range(-N, N + 1):
        omega = omega0 * (PHI ** n)
        A = 1.0 / (PHI ** abs(n))
        x += A * np.sin(omega * t)
    return x

def golden_fft_detect(signal: np.ndarray, dt: float = 1.0, epsilon: float = 0.18) -> Tuple[np.ndarray, np.ndarray, float]:
    n = len(signal)
    fft_vals, freqs, magnitudes = np.fft.rfft(signal), np.fft.rfftfreq(n, dt), np.abs(np.fft.rfft(signal))
    idx = np.argsort(magnitudes)[-10:]
    peak_freqs = np.sort(freqs[idx])
    peak_freqs = peak_freqs[peak_freqs > 0]
    if len(peak_freqs) < 2: return peak_freqs, np.array([]), 0.0
    ratios = peak_freqs[1:] / np.maximum(peak_freqs[:-1], 1e-12)
    golden_targets = np.array([PHI, PHI_SQ, PHI_INV])
    hits = sum(float(np.min(np.abs(r - golden_targets))) < epsilon for r in ratios)
    return peak_freqs, ratios, hits / len(ratios)

def compute_phase_alignment(close_prices: List[float], dt: float = 1.0, omega0: float = None, N: int = 3, epsilon: float = 0.18) -> Dict:
    if len(close_prices) < 64:
        return {"golden_score": 0.0, "energy_state": "INSUFFICIENT", "energy_ratio": 1.0, "spike_prob": 0.0, "phase_aligned": False, "near_min": False, "ratios": [], "pos_in_range": 0.5}
    arr = np.array(close_prices, dtype=float)
    arr_norm = arr - np.mean(arr)
    if omega0 is None: omega0 = 2.0 * np.pi / len(arr_norm)
    peak_freqs, ratios, golden_score = golden_fft_detect(arr_norm, dt, epsilon)
    energy = arr_norm ** 2
    mid = len(energy) // 2
    early_e, recent_e = float(np.mean(energy[:mid])), float(np.mean(energy[mid:]))
    energy_ratio = recent_e / (early_e + 1e-9)
    if energy_ratio < 0.40: energy_state = "COMPRESSION"
    elif energy_ratio < 0.75: energy_state = "BUILDING"
    elif energy_ratio < 1.40: energy_state = "EQUILIBRIUM"
    elif energy_ratio < 2.50: energy_state = "EXPANSION"
    else: energy_state = "PEAK"
    arr_min, arr_max = float(arr.min()), float(arr.max())
    rng = arr_max - arr_min
    pos_in_range = (float(arr[-1]) - arr_min) / (rng + 1e-9)
    near_min = pos_in_range < 0.25
    phase_aligned = (golden_score > 0.30 and energy_state in ("COMPRESSION", "BUILDING"))
    energy_bonus = {"COMPRESSION": 1.0, "BUILDING": 0.75, "EQUILIBRIUM": 0.40, "EXPANSION": 0.20, "PEAK": 0.05}.get(energy_state, 0.0)
    spike_prob = float(np.clip(0.45 * golden_score + 0.35 * energy_bonus + 0.20 * float(near_min), 0.0, 1.0))
    return {"golden_score": float(golden_score), "energy_state": energy_state, "energy_ratio": float(energy_ratio), "spike_prob": spike_prob, "phase_aligned": bool(phase_aligned), "near_min": bool(near_min), "ratios": [float(r) for r in ratios], "pos_in_range": float(pos_in_range)}

def golden_fib_proximity(current_price: float, ref_low: float, ref_high: float) -> Dict:
    rng = ref_high - ref_low
    if rng <= 0: return {"nearest": "NONE", "dist_pct": 0.0, "level_price": current_price}
    results = {}
    for label, ratio in FIB_RATIOS.items():
        level_price = ref_low + rng * ratio
        dist_pct = abs(current_price - level_price) / current_price * 100.0
        results[label] = {"price": level_price, "ratio": ratio, "dist_pct": dist_pct}
    nearest = min(results, key=lambda k: results[k]["dist_pct"])
    return {"nearest": nearest, "dist_pct": results[nearest]["dist_pct"], "level_price": results[nearest]["price"], "all_levels": results}


# ==========================================
# RATE LIMITER & TRADER
# ==========================================

class RateLimiter:
    def __init__(self, requests_per_second: float = 15, burst: int = 25):
        self.rate, self.burst, self.tokens, self.last_update, self.lock = requests_per_second, burst, burst, time.time(), Lock()
    def acquire(self):
        while True:
            with self.lock:
                now, elapsed = time.time(), time.time() - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_update = now
                if self.tokens >= 1: self.tokens -= 1; return
            time.sleep(0.05)

class Trader:
    def __init__(self, credentials_file: str):
        self.connect(credentials_file)
        self.rate_limiter = RateLimiter(requests_per_second=15, burst=30)
    def connect(self, file: str):
        with open(file) as f: lines = [line.strip() for line in f if line.strip()]
        if len(lines) < 2: raise ValueError("credentials.txt must contain API key on line 1 and secret on line 2")
        self.client = Client(lines[0], lines[1])
    def get_usdc_pairs(self) -> List[str]:
        exchange_info = self.client.get_exchange_info()
        pairs = [s['symbol'] for s in exchange_info['symbols'] if s['quoteAsset'] == 'USDC' and s['status'] == 'TRADING']
        print(f"Found {len(pairs)} USDC trading pairs"); return pairs
    def get_klines(self, symbol: str, interval: str, limit: int = 500, return_raw: bool = False, start_time: int = None, end_time: int = None):
        self.rate_limiter.acquire()
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        if start_time is not None: params['startTime'] = start_time
        if end_time is not None: params['endTime'] = end_time
        for attempt in range(3):
            try:
                klines = self.client.get_klines(**params)
                return klines if return_raw else [float(k[4]) for k in klines]
            except Exception as e: time.sleep(2 ** attempt * 2 if 'rate limit' in str(e).lower() else 0.5)
        return []
    def get_klines_extended(self, symbol: str, interval: str, total: int = 1200):
        MAX = 1000
        if total <= MAX: return self.get_klines(symbol, interval, limit=total, return_raw=True)
        first = self.get_klines(symbol, interval, limit=MAX, return_raw=True)
        if not first: return []
        second = self.get_klines(symbol, interval, limit=total - MAX, return_raw=True, end_time=int(first[0][0]) - 1)
        return (second + first) if second else first


# ==========================================
# VOLUME BREAKDOWN (per-TF)
# ==========================================

def get_volume_breakdown(trader: Trader, symbol: str, interval: str, limit: int = 100) -> Dict:
    klines = trader.get_klines(symbol, interval, limit=limit, return_raw=True)
    if not klines: return {'bull_pct': 50.0, 'bear_pct': 50.0, 'total': 0}
    bull = sum(float(k[5]) for k in klines if float(k[4]) >= float(k[1]))
    bear = sum(float(k[5]) for k in klines if float(k[4]) < float(k[1]))
    tot = bull + bear
    return {'bull_pct': bull / tot * 100 if tot > 0 else 50.0, 'bear_pct': bear / tot * 100 if tot > 0 else 50.0, 'bull': bull, 'bear': bear, 'total': tot}


# ==========================================
# CANDIDATE FILTER INDICATORS
# ==========================================

def is_confirmed_dip(close: list, high_tf: bool = False) -> bool:
    """Deep MA stack on higher TF, exhaustion on lower TFs."""
    if len(close) < 200: return False
    arr = np.array(close, dtype=float)
    sma12, sma27, sma56, sma200 = ta.SMA(arr, 12), ta.SMA(arr, 27), ta.SMA(arr, 56), ta.SMA(arr, 200)
    if high_tf:
        sma360 = ta.SMA(arr, 360) if len(arr) >= 360 else sma200
        return (arr[-1] < sma12[-1] and sma12[-1] < sma27[-1] < sma56[-1] < sma200[-1] and arr[-1] < sma360[-1])
    rsi = ta.RSI(arr, 14)
    macd, macdsignal, macdhist = ta.MACD(arr)
    oversold = rsi[-1] < 35
    momentum_shift = (macdhist[-1] > macdhist[-2]) and (macdhist[-1] > -0.5)
    return oversold and momentum_shift and rsi[-1] < 30

def is_below_regression_low(close: List[float], deviation: float = 0.01) -> bool:
    """STRICT ENFORCEMENT: Ensure price is strictly below the lowest regression line."""
    if len(close) < 20: return False
    x = np.arange(len(close))
    slope, intercept = np.polyfit(x, close, 1)
    trend = slope * x + intercept
    lower_band = trend * (1 - deviation)
    return close[-1] < lower_band[-1]

def get_sinusoidal_dip_timing(close_prices: list, lookback: int = 500) -> Dict:
    """
    STRICT UPGRADE: Lowest sine extrema confirmed AND up cycle with pump incoming.
    Analyzes the last 20 bars of the wave macro-trend to avoid 1-bar noise bounces.
    """
    if len(close_prices) < lookback: lookback = len(close_prices)
    arr = np.array(close_prices[-lookback:], dtype=float)
    arr_norm = arr - np.mean(arr)
    
    golden = compute_phase_alignment(arr_norm.tolist(), dt=1.0, N=3, epsilon=0.18)
    t = np.arange(len(arr_norm))
    wave = golden_signal(t, omega0=2*np.pi/len(arr_norm), N=2)
    
    # 1. Confirm we are strictly at the absolute lowest area (< 20% of range)
    current_phase_pos = (arr_norm[-1] - np.min(arr_norm)) / (np.max(arr_norm) - np.min(arr_norm) + 1e-9)
    near_bottom = current_phase_pos < 0.20  
    
    # 2. Confirm lowest sine extrema was just hit and moving UP
    lookback_window = 20  
    recent_wave = wave[-lookback_window:]
    wave_abs_min = np.min(recent_wave)
    wave_abs_min_idx = np.argmin(recent_wave)
    
    # The absolute minimum of the wave must have been hit in the last 5 bars
    just_hit_bottom = (lookback_window - wave_abs_min_idx) <= 5
    
    # The current wave must be strictly higher than the absolute minimum 
    # AND higher than it was 3 bars ago to confirm macro up-cycle (not micro-noise)
    moving_upward = (wave[-1] > wave_abs_min) and (wave[-1] > wave[-3])
    
    turning_up = just_hit_bottom and moving_upward
    
    # Combine both for the ultimate strict trigger. 
    # If wave_near_bottom is True here, it means BOTH conditions are 100% met.
    confirmed_dip_pump = near_bottom and turning_up
    
    cycle_length = len(arr_norm) / 3.0
    bars_to_up = int(cycle_length * (0.75 - current_phase_pos)) if near_bottom else int(cycle_length * 0.6)
    
    return {
        "wave_near_bottom": confirmed_dip_pump, 
        "turning_up": turning_up,
        "near_bottom_raw": near_bottom,
        "est_bars_to_pump": max(5, bars_to_up),
        "phase_pos": float(current_phase_pos),
        **golden
    }

def has_bullish_rejection_volume(raw_klines: list, window: int = 10) -> Tuple[bool, float]:
    if not raw_klines or len(raw_klines) < window: return False, 0.0
    recent = raw_klines[-window:]
    bull_vol = bear_vol = 0.0
    for k in recent:
        o, c, v = float(k[1]), float(k[4]), float(k[5])
        if v > 0:
            if c > o: bull_vol += v
            elif c < o: bear_vol += v
    total = bull_vol + bear_vol
    if total == 0: return False, 0.0
    ratio = bull_vol / total
    return ratio > 0.65, ratio

def calculate_effort_result_metrics(close: List[float], volumes: List[float], window: int = 20) -> Dict:
    if len(close) < window + 2: return {"R": 0, "C": 0, "E": 0}
    ca, va = np.array(close[-window:], dtype='float64'), np.array(volumes[-window:], dtype='float64')
    dp, tv, eps = abs(ca[-1] - ca[0]), np.sum(va), 1e-9
    return {"R": tv / (dp + eps), "C": tv / (np.std(ca) + eps), "E": tv / ((dp * window) + eps)}

def ml_spike_probability(R, C, E, bull_ratio, cmo, vratio) -> float:
    score = (0.30 * np.log1p(R) + 0.25 * np.log1p(C) + 0.20 * np.log1p(E) + 0.15 * bull_ratio + 0.05 * (-cmo / 100.0) + 0.05 * min(vratio / 5.0, 1.0))
    return 1 / (1 + np.exp(-score))


# ==========================================
# STRUCTURAL RANGE ENGINE (multi-lookback)
# ==========================================

def get_structural_extremes(close: np.ndarray, highs: np.ndarray, lows: np.ndarray, lookback: int) -> Dict:
    n, start = len(close), max(0, len(close) - lookback)
    c, h, l = close[start:], highs[start:], lows[start:]
    sl = len(c)
    amax_i, amin_i = int(np.argmax(c)), int(np.argmin(c))
    g_high, g_low = float(c[amax_i]), float(c[amin_i])
    high_age, low_age = sl - amax_i, sl - amin_i
    if low_age < high_age: more_recent, mr_label = "ARGMIN", "🟢 ARGMIN (low is fresher → floor established recently)"
    elif high_age < low_age: more_recent, mr_label = "ARGMAX", "🔴 ARGMAX (high is fresher → ceiling established recently)"
    else: more_recent, mr_label = "EQUAL", "⚪ EQUAL (both extremes same age)"
    rng = g_high - g_low
    rng_pct = (rng / g_low * 100) if g_low > 0 else 0
    pos = (close[-1] - g_low) / rng if rng > 0 else 0.5
    return {'high': g_high, 'low': g_low, 'high_bar': float(h[amax_i]), 'low_bar': float(l[amin_i]), 'high_age': high_age, 'low_age': low_age, 'more_recent': more_recent, 'mr_label': mr_label, 'range_size': rng, 'range_pct': rng_pct, 'position': pos, 'bars_used': sl}

def build_fib_grid(extremes: Dict, current_price: float) -> List[Dict]:
    lo, hi, rng = extremes['low'], extremes['high'], extremes['range_size']
    if rng <= 0: return []
    fibs = [(0.000, "ARGMIN"), (0.236, "F236"), (0.382, "F382"), (0.500, "F500"), (0.618, "F618"), (0.786, "F786"), (1.000, "ARGMAX")]
    grid = []
    for fib, label in fibs:
        price = lo + rng * fib
        dist = (price - current_price) / current_price * 100
        direction = 'UP' if price > current_price else ('DOWN' if price < current_price else 'AT')
        grid.append({'price': price, 'fib': fib, 'label': label, 'dist_pct': dist, 'direction': direction})
    return grid

def volume_profile_at_level(level_price: float, raw_klines: list, tolerance: float) -> Dict:
    bull_vol = bear_vol = 0.0
    touches = bull_rej = bear_rej = 0
    vol_seq = []
    lo, hi = level_price - tolerance, level_price + tolerance
    for k in raw_klines:
        o, h, l, c, v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
        if l <= hi and h >= lo:
            touches += 1; vol_seq.append(v)
            if v > 0:
                if c >= o: bull_vol += v
                else: bear_vol += v
            if l < lo and c >= level_price: bull_rej += 1
            elif h > hi and c <= level_price: bear_rej += 1
    total = bull_vol + bear_vol
    bp = bull_vol / total if total > 0 else 0.5
    exh, exh_detail = 0.0, "N/A"
    if len(vol_seq) >= 6:
        mid = len(vol_seq) // 2
        first_avg, second_avg = np.mean(vol_seq[:mid]), np.mean(vol_seq[mid:])
        if first_avg > 0:
            ratio = second_avg / first_avg
            exh = max(0.0, min(1.0, 1.0 - ratio))
            if ratio < 0.5: exh_detail = f"STRONG ({ratio:.0%})"
            elif ratio < 0.8: exh_detail = f"MODERATE ({ratio:.0%})"
            else: exh_detail = f"Weak ({ratio:.0%})"
    rej_vol_total = 0.0
    for k in raw_klines:
        o, h, l, c, v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
        if l <= hi and h >= lo:
            if (l < lo and c >= level_price) or (h > hi and c <= level_price): rej_vol_total += v
    rej_int = rej_vol_total / total if total > 0 else 0.0
    verdict = "SUPPORT" if bp > 0.58 else ("RESISTANCE" if bp < 0.42 else "NEUTRAL")
    return {'bull_pct': bp, 'total_volume': total, 'touches': touches, 'bull_rej': bull_rej, 'bear_rej': bear_rej, 'total_rej': bull_rej + bear_rej, 'exhaustion': exh, 'exhaustion_detail': exh_detail, 'rej_intensity': rej_int, 'verdict': verdict}


# ==========================================
# REJECTION CLUSTER ENERGY & SPIKE DETECTION
# ==========================================

def compute_rejection_cluster_score(vp: Dict, range_size: float) -> Dict:
    N, RI, V = vp['total_rej'], vp['rej_intensity'], vp['total_volume']
    if range_size <= 0 or V == 0: return {"score": 0.0, "raw": 0.0, "N": N, "state": "INVALID"}
    energy = (N ** 2) * RI * (V / range_size)
    norm_energy = np.log1p(energy)
    state = "NOISE" if N < 3 else ("BUILDING" if N <= 4 else ("COMPRESSION" if N <= 6 else "UNSTABLE"))
    return {"score": norm_energy, "raw": energy, "N": N, "state": state}

def detect_spike_trigger(curr_vp: Dict, prev_vp: Dict) -> bool:
    if not prev_vp: return False
    return curr_vp['total_rej'] <= prev_vp['total_rej'] and curr_vp['rej_intensity'] < prev_vp['rej_intensity'] and curr_vp['total_volume'] > prev_vp['total_volume'] * 0.8

def is_valid_spike(cluster: Dict, vp: Dict, vol_bias: float) -> bool:
    return cluster['state'] in ["COMPRESSION", "UNSTABLE"] and cluster['score'] > 1.5 and vp['rej_intensity'] > 0.2 and 0.45 < vol_bias < 0.65

def detect_cluster_transition(cluster: Dict, prev_cluster: Dict) -> bool:
    return bool(prev_cluster and prev_cluster['state'] == "COMPRESSION" and cluster['state'] == "UNSTABLE")

def detect_extreme_exhaustion(extreme_price: float, direction: str, raw_klines: list, zone_pct: float = 0.04) -> Dict:
    z_lo = extreme_price * (1 - zone_pct) if direction == 'high' else extreme_price * (1 - zone_pct * 0.5)
    z_hi = extreme_price * (1 + zone_pct * 0.5) if direction == 'high' else extreme_price * (1 + zone_pct)
    zv, zc = [], []
    for k in raw_klines:
        h, l, c, v = float(k[2]), float(k[3]), float(k[4]), float(k[5])
        if l <= z_hi and h >= z_lo: zv.append(v); zc.append(c)
    if len(zv) < 8: return {'exhaustion': 0.0, 'detail': 'Insufficient data', 'pattern': 'NONE', 'approach_vol': 0, 'final_vol': 0}
    split = int(len(zv) * 0.6)
    app_vol, fin_vol = np.mean(zv[:split]), np.mean(zv[split:])
    reached = (max(zc[split:]) >= extreme_price * 0.998) if direction == 'high' else (min(zc[split:]) <= extreme_price * 1.002)
    exh, pattern, detail = 0.0, "NONE", ""
    if app_vol > 0:
        ratio = fin_vol / app_vol
        if direction == 'high':
            if ratio < 0.4 and reached: exh, pattern, detail = 0.9, "CLIMAX_EXHAUSTION", f"Vol collapsed to {ratio:.0%} at peak → rejection likely"
            elif ratio < 0.65 and reached: exh, pattern, detail = 0.6, "FADE", f"Vol faded to {ratio:.0%} near high"
            elif ratio < 0.85: exh, pattern, detail = 0.3, "MILD_FADE", f"Slight fade to {ratio:.0%}"
            else: detail = f"No exhaustion (vol at {ratio:.0%})"
        else:
            if ratio < 0.35 and reached: exh, pattern, detail = 0.9, "CAPITULATION", f"Vol died to {ratio:.0%} after low → bounce likely"
            elif ratio < 0.6 and reached: exh, pattern, detail = 0.6, "SELLING_EXHAUSTION", f"Selling exhausted at {ratio:.0%}"
            elif ratio < 0.85: exh, pattern, detail = 0.3, "MILD_EXHAUSTION", f"Mild exhaustion at {ratio:.0%}"
            else: detail = f"No exhaustion (vol at {ratio:.0%})"
    else: detail = "No approach volume"
    return {'exhaustion': exh, 'detail': detail, 'pattern': pattern, 'approach_vol': app_vol, 'final_vol': fin_vol}


def score_level(level: Dict, vp: Dict, extreme_exh: float, range_pct: float, is_above: bool) -> Tuple[float, Dict]:
    bp, touches, rej, ri, exh = vp['bull_pct'], vp['touches'], vp['total_rej'], vp['rej_intensity'], vp['exhaustion']
    pressure = max(0.0, (0.5 - bp) * 2.0) if is_above else max(0.0, (bp - 0.5) * 2.0)
    rej_bonus = (vp['bear_rej'] if is_above else vp['bull_rej']) / max(touches, 1)
    touch_sc, rej_sc, ri_sc, exh_sc = min(touches / 12.0, 1.0) if touches > 0 else 0.0, min(rej_bonus * 3.0, 1.0), min(ri * 5.0, 1.0), extreme_exh
    fib = level['fib']
    ext_prox = max(0.0, 1.0 - abs(fib - 1.0) * 2.0) if is_above else max(0.0, 1.0 - abs(fib - 0.0) * 2.0)
    score = (0.25 * pressure + 0.15 * touch_sc + 0.15 * rej_sc + 0.15 * ri_sc + 0.20 * exh_sc + 0.10 * ext_prox)
    return score, {'pressure': pressure, 'touches': touch_sc, 'rejection_candles': rej_sc, 'rejection_intensity': ri_sc, 'exhaustion': exh_sc, 'extreme_prox': ext_prox}


def estimate_eta(dist_pct: float, range_pct: float, vol_bias: float) -> str:
    if range_pct <= 0: return "N/A"
    bias = vol_bias if vol_bias >= 0.5 else (1.0 - vol_bias)
    speed = (0.7 + 0.6 * bias) if ((dist_pct > 0 and vol_bias > 0.5) or (dist_pct < 0 and vol_bias < 0.5)) else (1.2 + 0.8 * (1.0 - bias))
    mins = abs(dist_pct) / max(range_pct, 0.01) * 360 * speed
    if mins < 5: return "~1-5 min"
    elif mins < 15: return "~5-15 min"
    elif mins < 30: return "~15-30 min"
    elif mins < 60: return "~30-60 min"
    elif mins < 120: return "~1-2 hrs"
    elif mins < 240: return "~2-4 hrs"
    else: return f"~{mins/60:.1f}+ hrs"


def analyze_lookback(raw_klines: list, close: np.ndarray, highs: np.ndarray, lows: np.ndarray, current_price: float, lookback: int, avg_range_pct: float, vol_bias: float) -> Dict:
    ext = get_structural_extremes(close, highs, lows, lookback)
    if ext['range_pct'] < 0.1: return {'lookback': lookback, 'extremes': ext, 'targets_up': [], 'targets_down': [], 'exh_high': {}, 'exh_low': {}, 'grid': [], 'min_dist': 0}
    grid = build_fib_grid(ext, current_price)
    tolerance = max(ext['range_size'] * 0.025, avg_range_pct / 100 * current_price * 1.5)
    klines_slice = raw_klines[max(0, len(close) - lookback):]
    exh_high, exh_low = detect_extreme_exhaustion(ext['high'], 'high', klines_slice), detect_extreme_exhaustion(ext['low'], 'low', klines_slice)
    min_dist = max(ext['range_pct'] * 0.05, 0.08)
    targets_up, targets_down, grid_out = [], [], []
    prev_vp, prev_cluster = None, None
    for level in grid:
        vp = volume_profile_at_level(level['price'], klines_slice, tolerance)
        cluster = compute_rejection_cluster_score(vp, ext['range_size'])
        trigger = detect_spike_trigger(vp, prev_vp)
        valid_spike = is_valid_spike(cluster, vp, vol_bias)
        explosion = detect_cluster_transition(cluster, prev_cluster)
        prev_vp, prev_cluster = vp, cluster
        lev_exh = exh_high['exhaustion'] if level['fib'] >= 0.618 else (exh_low['exhaustion'] if level['fib'] <= 0.382 else 0.0)
        dist, is_above, is_below = abs(level['dist_pct']), level['direction'] == 'UP', level['direction'] == 'DOWN'
        if dist < min_dist:
            grid_out.append({**level, **vp, 'score': 0, 'status': 'TOO_CLOSE', 'cluster_score': cluster['score'], 'cluster_state': cluster['state'], 'valid_spike': valid_spike, 'trigger': trigger, 'explosion': explosion})
            continue
        score, details = score_level(level, vp, lev_exh, ext['range_pct'], is_above)
        score += min(cluster['score'] * 0.15, 0.3)
        eta = estimate_eta(level['dist_pct'], ext['range_pct'], vol_bias)
        entry = {'price': level['price'], 'score': score, 'dist_pct': level['dist_pct'], 'label': level['label'], 'fib': level['fib'], 'verdict': vp['verdict'], 'bull_pct': vp['bull_pct'], 'touches': vp['touches'], 'rejections': vp['total_rej'], 'rej_intensity': vp['rej_intensity'], 'cluster_score': cluster['score'], 'cluster_state': cluster['state'], 'cluster_raw': cluster['raw'], 'trigger': trigger, 'valid_spike': valid_spike, 'explosion': explosion, 'eta': eta, 'details': details}
        grid_out.append({**level, **vp, **entry, 'status': 'ACTIVE'})
        if is_above: targets_up.append(entry)
        elif is_below: targets_down.append(entry)
    targets_up.sort(key=lambda t: t['score'], reverse=True)
    targets_down.sort(key=lambda t: t['score'], reverse=True)
    return {'lookback': lookback, 'extremes': ext, 'targets_up': sorted(targets_up[:4], key=lambda t: t['dist_pct']), 'targets_down': sorted(targets_down[:4], key=lambda t: -t['dist_pct']), 'exh_high': exh_high, 'exh_low': exh_low, 'grid': grid_out, 'min_dist': min_dist}


def get_sr_targets(raw_klines: list, current_price: float) -> Dict:
    if len(raw_klines) < 100: return {'lookbacks': [], 'vol_bias': 0.5, 'avg_range': 0}
    highs, lows, closes, volumes = np.array([float(k[2]) for k in raw_klines]), np.array([float(k[3]) for k in raw_klines]), np.array([float(k[4]) for k in raw_klines]), np.array([float(k[5]) for k in raw_klines])
    candle_ranges = (highs - lows) / (closes + 1e-12) * 100.0
    avg_range = float(np.mean(candle_ranges[-50:]))
    closed_vols = [v for v in volumes[-21:-1] if v > 0]
    vol_bias = 0.5
    if closed_vols:
        rec = raw_klines[-21:-1]
        bv = sum(float(k[5]) for k in rec if float(k[4]) >= float(k[1]) and float(k[5]) > 0)
        bear_v = sum(float(k[5]) for k in rec if float(k[4]) < float(k[1]) and float(k[5]) > 0)
        tv = bv + bear_v
        vol_bias = bv / tv if tv > 0 else 0.5
    lookbacks = [analyze_lookback(raw_klines, closes, highs, lows, current_price, lb, avg_range, vol_bias) for lb in [500, 800, 1200] if len(raw_klines) >= lb]
    return {'lookbacks': lookbacks, 'vol_bias': vol_bias, 'avg_range': avg_range}


# ==========================================
# CONCURRENT FILTER FUNCTIONS
# ==========================================

def check_tf_dip(trader, symbol, interval):
    close = trader.get_klines(symbol, interval, limit=500)
    return (symbol, is_confirmed_dip(close, high_tf=True))

def check_5m_regression(trader, symbol):
    close = trader.get_klines(symbol, '5m', limit=100)
    return (symbol, is_below_regression_low(close, deviation=0.01))

def check_1m_final(trader, symbol):
    klines = trader.get_klines(symbol, '1m', limit=500, return_raw=True)
    default_dict = {"golden_score": 0.0, "energy_state": "INSUFFICIENT", "spike_prob": 0.0, "phase_aligned": False, "near_min": False, "wave_near_bottom": False, "turning_up": False, "est_bars_to_pump": 0, "phase_pos": 0.5}
    if not klines or len(klines) < 100:
        return (symbol, 0.0, 0.0, False, 0.0, 0.0, default_dict)

    close, volumes = [float(k[4]) for k in klines], [float(k[5]) for k in klines]
    cmo = ta.CMO(np.asarray(close), timeperiod=14)
    cmo_val = float(cmo[-1]) if not np.isnan(cmo[-1]) else 0.0

    closed_vols = [v for v in volumes[:-1] if v > 0]
    vratio = 0.0
    if closed_vols:
        avg_vol, last_closed = np.mean(closed_vols[-50:]), closed_vols[-1]
        vratio = last_closed / avg_vol if avg_vol > 0 else 0.0

    is_rej, bull_ratio = has_bullish_rejection_volume(klines, window=10)
    metrics = calculate_effort_result_metrics(close, volumes, window=20)
    prob = ml_spike_probability(metrics["R"], metrics["C"], metrics["E"], bull_ratio, cmo_val, vratio)

    golden = compute_phase_alignment(close, dt=1.0, N=3, epsilon=0.18)
    sinusoidal = get_sinusoidal_dip_timing(close, 500)
    combined = {**golden, **sinusoidal}

    # ABSOLUTE STRICT GATEKEEPERS:
    # 1. RSI/MACD Oversold dip
    cond_1 = is_confirmed_dip(close, high_tf=False)
    # 2. Below lowest regression channel line
    cond_2 = is_below_regression_low(close, deviation=0.01)
    # 3. Lowest Sine Extrema confirmed AND up cycle initiated
    cond_3 = combined.get('wave_near_bottom', False) == True
    cond_4 = combined.get('turning_up', False) == True
    
    is_strong = cond_1 and cond_2 and cond_3 and cond_4

    return (symbol, cmo_val, vratio, is_strong, bull_ratio, prob, combined)


class ProgressTracker:
    def __init__(self, total, label):
        self.total, self.label, self.completed, self.passed = total, label, 0, 0
        self.lock, self.start_time = Lock(), time.time()
    def update(self, passed=False):
        with self.lock:
            self.completed += 1
            if passed: self.passed += 1
    def get_stats(self):
        with self.lock:
            e = time.time() - self.start_time
            r = self.completed / e if e > 0 else 0
            rem = (self.total - self.completed) / r if r > 0 else 0
            return (f"\r{self.label}: {self.completed}/{self.total} | ✓{self.passed} | {r:.1f}/s | ETA: {rem:.0f}s")

def run_tf_filter(trader, symbols, interval, max_workers=20):
    passed, tracker = [], ProgressTracker(len(symbols), f"{interval} filter")
    print(f"Running {interval} filter on {len(symbols)} pairs...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(check_tf_dip, trader, s, interval): s for s in symbols}
        for f in as_completed(futures):
            try:
                sym, ok = f.result()
                if ok: passed.append(sym)
                tracker.update(passed=ok)
                print(tracker.get_stats(), end="", flush=True)
            except: tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20)
    return passed

def run_5m_regression_filter(trader, symbols, max_workers=20):
    passed, tracker = [], ProgressTracker(len(symbols), "5m Reg filter")
    print(f"Running 5m Lowest Regression filter on {len(symbols)} pairs...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(check_5m_regression, trader, s): s for s in symbols}
        for f in as_completed(futures):
            try:
                sym, ok = f.result()
                if ok: passed.append(sym)
                tracker.update(passed=ok)
                print(tracker.get_stats(), end="", flush=True)
            except: tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20)
    return passed

def run_1m_filter(trader, symbols, max_workers=15):
    results, tracker = [], ProgressTracker(len(symbols), "1m filter")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(check_1m_final, trader, s): s for s in symbols}
        for f in as_completed(futures):
            try:
                res = f.result()
                results.append(res)
                tracker.update(passed=res[3])
                print(tracker.get_stats(), end="", flush=True)
            except: tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20)
    return results


# ==========================================
# OUTPUT FORMATTER
# ==========================================

def format_golden_block(golden: Dict, W: int = 74):
    gs, estate, sp = golden.get("golden_score", 0.0), golden.get("energy_state", "N/A"), golden.get("spike_prob", 0.0)
    pa, nm, er = golden.get("phase_aligned", False), golden.get("near_min", False), golden.get("energy_ratio", 1.0)
    pos, ratios = golden.get("pos_in_range", 0.5), golden.get("ratios", [])
    wave_near_bottom, turning_up = golden.get("wave_near_bottom", False), golden.get("turning_up", False)
    est_bars_to_pump, phase_pos = golden.get("est_bars_to_pump", 0), golden.get("phase_pos", 0.5)
    estate_icon = {"COMPRESSION": "🔵", "BUILDING": "🟡", "EQUILIBRIUM": "⚪", "EXPANSION": "🟠", "PEAK": "🔴"}.get(estate, "⚪")
    print("─" * W)
    print("  ✨  GOLDEN HARMONIC ENGINE  (φ = 1.6180…)")
    print("─" * W)
    print(f"  φ Score (FFT)    : {gs*100:.1f}%  ({'φ-structure detected' if gs > 0.3 else 'weak φ-structure'})")
    print(f"  Energy State     : {estate_icon} {estate}  (ratio={er:.3f})")
    print(f"  Phase Aligned    : {'✅ YES — harmonics converging' if pa else '❌ NO'}")
    print(f"  Near Cycle Min   : {'✅ YES — stationary floor proximity' if nm else '❌ NO'}")
    print(f"  Pos in Range     : {pos*100:.1f}%")
    print(f"  Golden Spike Prob: {sp*100:.1f}%")
    print("─" * W)
    print("  🌊  SINUSOIDAL DIP TIMING (STRICT EXTREMA CHECK)")
    print("─" * W)
    print(f"  Wave Near Bottom : {'✅ YES — Confirmed Lowest Extrema' if wave_near_bottom else '❌ NO'}")
    print(f"  Turning Up       : {'✅ YES — Macro Up-Cycle Initiated' if turning_up else '❌ NO'}")
    print(f"  Est. Bars to Pump: ~{est_bars_to_pump} bars (1m)")
    print(f"  Phase Position   : {phase_pos*100:.1f}%")
    if ratios:
        ratio_str = "  ".join(f"{r:.3f}" for r in ratios[:6])
        phi_hits = sum(1 for r in ratios if abs(r - PHI) < 0.18 or abs(r - PHI_SQ) < 0.18 or abs(r - PHI_INV) < 0.18)
        print(f"  FFT Ratios       : {ratio_str}")
        print(f"  φ-ratio hits     : {phi_hits}/{len(ratios)}  (φ≈{PHI:.3f}  φ²≈{PHI_SQ:.3f}  1/φ≈{PHI_INV:.3f})")
    states = ["COMPRESSION", "BUILDING", "EQUILIBRIUM", "EXPANSION", "PEAK"]
    bar = "  Flow: "
    for s in states:
        bar += f"[{s[:4]}]→" if s == estate else f" {s[:4]} →"
    print(f"{bar[:-1]}\n")

def format_sr_output(symbol, sr, current_price, cmo_val, vratio, bull_ratio, ml_prob, tf_volumes, golden: Dict = None):
    vb, avg_r = sr['vol_bias'], sr['avg_range']
    bp_pct = vb * 100
    bias_lbl = "🟢 BULLISH" if vb > 0.55 else ("🔴 BEARISH" if vb < 0.45 else "⚪ NEUTRAL")
    W = 74
    print("\n" + "=" * W)
    print(f"  ★  STRUCTURAL RANGE S/R  —  {symbol}  ★")
    print(f"  (argmin/argmax anchored · multi-lookback · volume exhaustion · φ-harmonics)")
    print("=" * W)
    print(f"  Entry Price    : {current_price:.10f}")
    print(f"  1m CMO         : {cmo_val:+.2f}  (< -50 = oversold)")
    print(f"  Vol Ratio      : x{vratio:.2f}")
    print(f"  Bull Rej Vol   : {bull_ratio*100:.1f}%")
    print(f"  ML Spike Prob  : {ml_prob*100:.1f}%")
    print(f"  1m Vol Bias    : {bias_lbl}  ({bp_pct:.1f}% bull / {100-bp_pct:.1f}% bear)")
    print(f"  Avg 1m Range   : {avg_r:.4f}%")
    print("-" * W)
    print("  📊  VOLUME BREAKDOWN BY TIMEFRAME")
    for tf, vd in tf_volumes.items():
        bar_len, bull_len = 30, int(vd['bull_pct'] / 100 * 30)
        bar = "🟢" * bull_len + "🔴" * (bar_len - bull_len)
        print(f"  {tf:>4s}  [{bar}]  Bull: {vd['bull_pct']:.1f}%  Bear: {vd['bear_pct']:.1f}%")
    if golden: format_golden_block(golden, W)

    all_signals = []
    if golden:
        if golden.get("phase_aligned"): all_signals.append(f"φ Phase Aligned (score={golden['golden_score']*100:.0f}%, state={golden['energy_state']})")
        if golden.get("near_min") and golden.get("energy_state") == "COMPRESSION": all_signals.append("φ COMPRESSION at cycle minimum → bounce setup")
        if golden.get("spike_prob", 0) > 0.65: all_signals.append(f"φ Golden spike prob ({golden['spike_prob']*100:.0f}%)")
        if golden.get("wave_near_bottom") and golden.get("turning_up"): all_signals.append("🌊 LOWEST EXTREMA CONFIRMED & UP CYCLE PUMP INCOMING")
        if golden.get("est_bars_to_pump", 0) < 15: all_signals.append(f"🌊 Estimated <{golden['est_bars_to_pump']} bars to pump")

    for lb_data in sr['lookbacks']:
        lb, ext, exh_h, exh_l = lb_data['lookback'], lb_data['extremes'], lb_data['exh_high'], lb_data['exh_low']
        rng_pct, pos, min_d = ext['range_pct'], ext['position'], lb_data['min_dist']
        print("\n" + "─" * W)
        print(f"  📐  LOOKBACK: {lb} BARS  ({lb} min)")
        print("─" * W)
        print(f"  Global High     : {ext['high']:.10f}  ({ext['high_age']} bars ago)")
        print(f"  Global Low      : {ext['low']:.10f}  ({ext['low_age']} bars ago)")
        print(f"  True Range      : {rng_pct:.3f}%")
        print(f"  More Recent     : {ext['mr_label']}")
        print(f"  Min Target Dist : {min_d:.3f}%")
        pos_pct, blen, bpos = pos * 100, 40, int(pos * 40)
        pbar = "─" * bpos + "▲" + "─" * (blen - bpos - 1)
        pos_txt = 'near LOW' if pos < 0.25 else ('near HIGH' if pos > 0.75 else 'mid-range')
        print(f"  Position        : [{pbar}]  {pos_pct:.1f}%  ({pos_txt})")
        phi_prox = golden_fib_proximity(current_price, ext['low'], ext['high'])
        print(f"  φ Nearest Level : {phi_prox['nearest']}  @ {phi_prox['level_price']:.10f}  (dist {phi_prox['dist_pct']:.3f}%)")
        print(f"\n  🫁  Exhaustion at HIGH: ", end="")
        if exh_h.get('exhaustion', 0) > 0.5: print(f"🔴 {exh_h['pattern']} ({exh_h['exhaustion']:.2f})\n     {exh_h['detail']}")
        elif exh_h.get('exhaustion', 0) > 0.2: print(f"🟡 {exh_h['pattern']} ({exh_h['exhaustion']:.2f})\n     {exh_h['detail']}")
        else: print(f"⚪ {exh_h.get('pattern', 'NONE')} ({exh_h.get('exhaustion', 0):.2f})\n     {exh_h.get('detail', '')}")
        print(f"  🫁  Exhaustion at LOW : ", end="")
        if exh_l.get('exhaustion', 0) > 0.5:
            print(f"🟢 {exh_l['pattern']} ({exh_l['exhaustion']:.2f})\n     {exh_l['detail']}")
            if pos < 0.5: all_signals.append(f"[{lb}] Selling exhaustion at low ({exh_l['pattern']})")
        elif exh_l.get('exhaustion', 0) > 0.2: print(f"🟡 {exh_l['pattern']} ({exh_l['exhaustion']:.2f})\n     {exh_l['detail']}")
        else: print(f"⚪ {exh_l.get('pattern', 'NONE')} ({exh_l.get('exhaustion', 0):.2f})\n     {exh_l.get('detail', '')}")
        if ext['more_recent'] == 'ARGMIN': all_signals.append(f"[{lb}] ARGMIN more recent → recent floor")
        elif ext['more_recent'] == 'ARGMAX': all_signals.append(f"[{lb}] ARGMAX more recent → recent ceiling")
        grid = lb_data['grid']
        if grid:
            print(f"\n  📊  Fibonacci Grid (volume profile + cluster energy + φ levels)")
            print(f"  {'Level':<8} {'Price':>14} {'Dist%':>8} {'Bull%':>6} {'Tch':>4} {'Rej':>4} {'Exh':>5} {'ClSt':<12} {'ClSc':>5} {'Verdict':<10} {'St'}")
            print("  " + "─" * 88)
            for g in grid:
                st, cs, csc, vs = g.get('status', '?'), g.get('cluster_state', '—'), g.get('cluster_score', 0.0), g.get('valid_spike', False)
                cs_icon = "💣" if cs == "UNSTABLE" else ("🔥" if cs == "COMPRESSION" else ("⚡" if cs == "BUILDING" else "·"))
                m = "·" if st == 'TOO_CLOSE' else ("►" if g.get('direction') == 'UP' else "◄")
                print(f"  {m}{g['label']:<7} {g['price']:>14.8f} {g['dist_pct']:>+7.3f}% {g['bull_pct']*100:>5.0f}% {g['touches']:>4} {g['total_rej']:>4} {g['exhaustion']:>4.2f} {cs_icon}{cs:<11} {csc:>5.2f} {g['verdict']:<10} {'✅SPIKE' if vs else st}")
        for direction, tgt_list, label_prefix in [("UP", lb_data['targets_up'], "📈 RESISTANCE"), ("DOWN", lb_data['targets_down'], "📉 SUPPORT")]:
            if tgt_list:
                print(f"\n  {label_prefix} TARGETS ({lb} bars)\n")
                for i, t in enumerate(tgt_list, 1):
                    bar = "█" * int(t['score'] * 10) + "░" * (10 - int(t['score'] * 10))
                    vi = "🔴" if t['verdict'] == "RESISTANCE" else ("🟢" if t['verdict'] == "SUPPORT" else "⚪")
                    cs, csc, vs, trig, expl = t.get('cluster_state', '—'), t.get('cluster_score', 0.0), t.get('valid_spike', False), t.get('trigger', False), t.get('explosion', False)
                    spike_tag = "  💣 EXPLOSION SETUP" if expl else ("  🔥 VALID SPIKE" if vs else ("  ⚡ TRIGGER" if trig else ""))
                    print(f"  {label_prefix[0]}{i}  {t['label']:5s}  {t['price']:.10f}  ({t['dist_pct']:+.3f}%)  ETA: {t['eta']}{spike_tag}")
                    print(f"       [{bar}] {t['score']:.2f}  {vi} {t['verdict']}  BullVol: {t['bull_pct']*100:.0f}%  Tch: {t['touches']}  Rej: {t['rejections']}  RejInt: {t['rej_intensity']:.2f}")
                    print(f"       ClusterState: {cs:<12}  ClusterScore: {csc:.3f}")
                    d, parts = t.get('details', {}), []
                    if d.get('exhaustion', 0) > 0.2: parts.append(f"Exh:{d['exhaustion']:.2f}")
                    if d.get('rejection_candles', 0) > 0.2: parts.append(f"Rej:{d['rejection_candles']:.2f}")
                    if d.get('extreme_prox', 0) > 0.3: parts.append(f"NearExt")
                    if parts: print(f"             + {' | '.join(parts)}")
                    print()
            else:
                print(f"\n  {label_prefix} targets beyond {min_d:.3f}% minimum not found.\n")

    print("=" * W)
    print("  ⚡  CONSOLIDATED TRADE BIAS")
    print("=" * W)
    argmin_count = sum(1 for lb in sr['lookbacks'] if lb['extremes']['more_recent'] == 'ARGMIN')
    argmax_count = sum(1 for lb in sr['lookbacks'] if lb['extremes']['more_recent'] == 'ARGMAX')
    total_lb = len(sr['lookbacks'])
    print(f"\n  Recency Across Lookbacks:\n    ARGMIN more recent : {argmin_count}/{total_lb}\n    ARGMAX more recent : {argmax_count}/{total_lb}")
    if argmin_count > argmax_count: all_signals.append(f"ARGMIN dominant across lookbacks ({argmin_count}/{total_lb})")
    best_cluster_score, best_cluster_state, explosion_found, spike_found = 0.0, None, False, False
    for lb_data in sr['lookbacks']:
        for tgt_list in [lb_data['targets_up'], lb_data['targets_down']]:
            for t in tgt_list:
                cs = t.get('cluster_score', 0.0)
                if cs > best_cluster_score: best_cluster_score, best_cluster_state = cs, t.get('cluster_state')
                if t.get('explosion'): explosion_found = True
                if t.get('valid_spike'): spike_found = True
    if explosion_found: all_signals.append(f"💣 EXPLOSION SETUP: COMPRESSION→UNSTABLE transition detected")
    elif spike_found: all_signals.append(f"🔥 VALID SPIKE zone (cluster state={best_cluster_state}, score={best_cluster_score:.2f})")
    elif best_cluster_state in ("COMPRESSION", "UNSTABLE") and best_cluster_score > 1.0: all_signals.append(f"⚡ Cluster energy building ({best_cluster_state}, score={best_cluster_score:.2f})")
    if vb > 0.55: all_signals.append(f"1m Bullish vol bias ({vb*100:.0f}%)")
    if cmo_val < -50: all_signals.append(f"CMO oversold ({cmo_val:.0f})")
    if bull_ratio > 0.65: all_signals.append(f"Bull rejection vol ({bull_ratio*100:.0f}%)")
    if ml_prob > 0.65: all_signals.append(f"ML spike prob ({ml_prob*100:.0f}%)")
    cluster_prob, trigger_bonus = min(best_cluster_score / 3.0, 1.0), (1 if explosion_found else (0.5 if spike_found else 0))
    golden_contrib = golden.get("spike_prob", 0.0) if golden else 0.0
    sinusoidal_contrib = 0.0
    if golden:
        if golden.get("wave_near_bottom") and golden.get("turning_up"): sinusoidal_contrib = 0.8
        elif golden.get("wave_near_bottom"): sinusoidal_contrib = 0.5
        if golden.get("est_bars_to_pump", 0) < 15: sinusoidal_contrib = min(sinusoidal_contrib + 0.2, 1.0)
    enhanced_prob = (0.30 * ml_prob + 0.20 * cluster_prob + 0.15 * trigger_bonus + 0.15 * golden_contrib + 0.20 * sinusoidal_contrib)
    print(f"\n  ⚡  Enhanced Spike Probability (φ + sinusoidal-augmented):")
    print(f"     ML Prob        : {ml_prob*100:.1f}%")
    print(f"     Cluster Prob   : {cluster_prob*100:.1f}%  (best score={best_cluster_score:.2f}, state={best_cluster_state or 'N/A'})")
    print(f"     Trigger Bonus  : {'EXPLOSION' if explosion_found else ('SPIKE' if spike_found else 'none')}")
    if golden:
        print(f"     φ Golden Prob  : {golden_contrib*100:.1f}%  (state={golden.get('energy_state','N/A')})")
        print(f"     Sinusoidal Prob: {sinusoidal_contrib*100:.1f}%  (lowest extrema & up cycle: {golden.get('wave_near_bottom', False)})")
    print(f"     FINAL PROB     : {enhanced_prob*100:.1f}%")
    best_up, best_dn = None, None
    for lb in reversed(sr['lookbacks']):
        if not best_up and lb['targets_up']: best_up = lb['targets_up'][0]
        if not best_dn and lb['targets_down']: best_dn = lb['targets_down'][0]
        if best_up and best_dn: break
    if best_up and best_dn:
        rr = abs(best_up['dist_pct']) / max(abs(best_dn['dist_pct']), 0.0001)
        print(f"\n  Best Target : {best_up['label']:5s}  {best_up['price']:.10f}  ({best_up['dist_pct']:+.3f}%)  ETA: {best_up['eta']}")
        print(f"  Best Stop   : {best_dn['label']:5s}  {best_dn['price']:.10f}  ({best_dn['dist_pct']:+.3f}%)")
        print(f"  R:R         : {rr:.2f}x")
        if rr >= 1.5: all_signals.append(f"R:R favorable ({rr:.1f}x)")
    elif best_up: print(f"\n  Target only : {best_up['label']:5s}  {best_up['price']:.10f}  ({best_up['dist_pct']:+.3f}%)")
    elif best_dn: print(f"\n  Support only: {best_dn['label']:5s}  {best_dn['price']:.10f}  ({best_dn['dist_pct']:+.3f}%)")
    else: print("\n  No structural levels found.")
    ns = len(all_signals)
    print(f"\n  Structural Signals ({ns}):")
    for s in all_signals: print(f"    ✅  {s}")
    print()
    if ns >= 4: v = "✅  STRONG LONG  —  Multiple structural confirmations"
    elif ns >= 3: v = "✅  LONG  —  Good structural alignment"
    elif ns >= 2: v = "⏳  PROBABLE LONG  —  Awaiting final confirmation"
    elif ns >= 1: v = "⏳  WEAK SIGNAL  —  Insufficient confirmation"
    else: v = "⚪  NEUTRAL  —  No clear structural bias"
    print(f"  VERDICT : {v}")
    print("\n" + "=" * W)
    print(f"  CURRENT PRICE     : {current_price:.8f} USDC")
    print(f"  ARGMIN vs ARGMAX  : {argmin_count}/{total_lb} lookbacks show recent floor")
    if golden:
        print(f"  Sinusoidal Timing : Extrema & Up Cycle: {'YES' if golden.get('wave_near_bottom') else 'NO'}")
        print(f"  Est. Bars to Pump : ~{golden.get('est_bars_to_pump', 'N/A')} bars (1m)")
    print(f"  Expected Move     : Strong reversal spike expected (φ-compression + exhaustion)")
    print("=" * W + "\n")
    return v, ns


def print_scan_header(scan_count: int):
    W = 78
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "╔" + "═" * W + "╗")
    print("║" + " " * W + "║")
    print("║" + f"  🚀  SCAN #{scan_count} STARTED  —  {timestamp}".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("║" + "  🔍  CASCADE: 4H→2H→1H→30M→15M→5M(REG)→1M(REG+SINE UP)".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("╚" + "═" * W + "╝")


# ==========================================
# MAIN SCAN LOOP
# ==========================================

def main():
    CREDENTIALS_FILE = "credentials.txt"
    MIN_SIGNALS_REQUIRED = 3
    max_scans = 10
    
    print("=" * 78)
    print("  🌊  STRICT MTF CASCADE DIP DETECTOR (LOWEST EXTREMA + SINE UP)  🌊")
    print("=" * 78)
    print()
    
    try:
        trader = Trader(CREDENTIALS_FILE)
    except Exception as e:
        print(f"❌ Failed to initialize trader: {e}")
        return
    
    scan_count = 0
    
    while scan_count < max_scans:
        scan_count += 1
        print_scan_header(scan_count)
        
        try:
            symbols = trader.get_usdc_pairs()
            if not symbols:
                print("❌ No USDC pairs found. Retrying...")
                time.sleep(10)
                continue
            
            four_h_passed = run_tf_filter(trader, symbols, '4h', max_workers=20)
            if not four_h_passed:
                print("❌ No pairs passed 4h dip filter. Retrying...")
                time.sleep(5)
                continue
            
            two_h_passed = run_tf_filter(trader, four_h_passed, '2h', max_workers=20)
            if not two_h_passed:
                print("❌ No pairs passed 2h dip filter. Retrying...")
                time.sleep(5)
                continue
            
            one_h_passed = run_tf_filter(trader, two_h_passed, '1h', max_workers=20)
            if not one_h_passed:
                print("❌ No pairs passed 1h dip filter. Retrying...")
                time.sleep(5)
                continue
            
            thirty_m_passed = run_tf_filter(trader, one_h_passed, '30m', max_workers=20)
            if not thirty_m_passed:
                print("❌ No pairs passed 30m dip filter. Retrying...")
                time.sleep(5)
                continue
            
            fifteen_m_passed = run_tf_filter(trader, thirty_m_passed, '15m', max_workers=20)
            if not fifteen_m_passed:
                print("❌ No pairs passed 15m dip filter. Retrying...")
                time.sleep(5)
                continue
            
            five_m_passed = run_5m_regression_filter(trader, fifteen_m_passed, max_workers=20)
            if not five_m_passed:
                print("❌ No pairs below 5m lowest regression line. Retrying...")
                time.sleep(5)
                continue
            
            results_1m = run_1m_filter(trader, five_m_passed, max_workers=15)
            if not results_1m:
                print("❌ No 1m analysis results (Sine Extrema + Up Cycle failed). Retrying...")
                time.sleep(5)
                continue
            
            def final_score(r):
                ml = r[5]
                gs = r[6].get("spike_prob", 0.0)
                near_bottom = r[6].get("wave_near_bottom", False)
                turning_up = r[6].get("turning_up", False)
                est_bars_to_pump = r[6].get("est_bars_to_pump", 0)
                score = ml * 0.4 + gs * 0.3
                if near_bottom: score += 0.15
                if turning_up: score += 0.1
                if est_bars_to_pump < 15: score += 0.1
                score -= r[1] * 0.0015
                return score
            
            results_1m.sort(key=final_score, reverse=True)
            
            top_candidate = results_1m[0]
            symbol, cmo_val, vratio = top_candidate[0], top_candidate[1], top_candidate[2]
            bull_ratio, ml_prob, golden = top_candidate[4], top_candidate[5], top_candidate[6]
            
            print(f"\n🏆 TOP CANDIDATE: {symbol}")
            print(f"   Final Score: {final_score(top_candidate):.4f}")
            
            klines_1m = trader.get_klines(symbol, '1m', limit=1200, return_raw=True)
            
            if klines_1m:
                current_price = float(klines_1m[-1][4])
                sr = get_sr_targets(klines_1m, current_price)
                
                tf_volumes = {
                    '5m': get_volume_breakdown(trader, symbol, '5m'),
                    '15m': get_volume_breakdown(trader, symbol, '15m'),
                    '1h': get_volume_breakdown(trader, symbol, '1h'),
                    '4h': get_volume_breakdown(trader, symbol, '4h'),
                }
                
                verdict, signal_count = format_sr_output(
                    symbol, sr, current_price, cmo_val, vratio,
                    bull_ratio, ml_prob, tf_volumes, golden
                )
                
                if signal_count >= MIN_SIGNALS_REQUIRED:
                    W = 78
                    print("\n" + "╔" + "═" * W + "╗")
                    print("║" + " " * W + "║")
                    print("║" + "  ✅  SETUP FOUND — BOT STOPPING".ljust(W) + "║")
                    print("║" + " " * W + "║")
                    print("║" + f"  Asset: {symbol}".ljust(W) + "║")
                    print("║" + f"  Price: {current_price:.8f} USDC".ljust(W) + "║")
                    print("║" + f"  Signals: {signal_count}".ljust(W) + "║")
                    print("║" + f"  Verdict: {verdict}".ljust(W) + "║")
                    print("║" + " " * W + "║")
                    print("║" + "  🎯  Review the analysis above and make your decision.".ljust(W) + "║")
                    print("║" + " " * W + "║")
                    print("╚" + "═" * W + "╝")
                    print("\n✅ Bot completed successfully. Exiting...")
                    return
                else:
                    print(f"\n⚠️  Found {signal_count} signals (need {MIN_SIGNALS_REQUIRED}). Continuing search...")
                    time.sleep(3)
            else:
                print(f"\n❌ Failed to get detailed analysis for {symbol}")
                time.sleep(3)
            
            gc.collect()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Scan interrupted by user. Exiting...")
            return
        except Exception as e:
            print(f"\n❌ Error in scan: {e}")
            time.sleep(5)
    
    W = 78
    print("\n" + "╔" + "═" * W + "╗")
    print("║" + " " * W + "║")
    print("║" + "  ❌  MAX SCANS REACHED — NO SETUP FOUND".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("║" + f"  Completed {scan_count} scans without finding".ljust(W) + "║")
    print("║" + f"  a setup with {MIN_SIGNALS_REQUIRED}+ signals.".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("║" + "  Try again later when market conditions improve.".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("╚" + "═" * W + "╝")
    print("\n❌ Bot completed without finding setup. Exiting...")
    return


if __name__ == "__main__":
    main()