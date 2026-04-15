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

PHI = (1.0 + 5.0 ** 0.5) / 2.0          # ≈ 1.6180339887
PHI_INV = 1.0 / PHI                       # ≈ 0.6180339887  (1/φ = φ - 1)
PHI_SQ  = PHI * PHI                       # ≈ 2.6180339887  (φ² = φ + 1)

# Fibonacci inner ratios (retracement levels)
FIB_RATIOS = {
    "F236": 0.236,
    "F382": 0.382,
    "F500": 0.500,
    "F618": PHI_INV,   # 0.618 — the golden ratio
    "F786": PHI_INV ** 0.5,  # ≈ 0.786 = √0.618
}


# ==========================================
# GOLDEN HARMONIC ENGINE
# ==========================================

def golden_signal(t: np.ndarray, omega0: float = 1.0, N: int = 3) -> np.ndarray:
    """
    φ-scaled harmonic stack:
        x(t) = Σ A_n · sin(ω₀ · φⁿ · t)   for n in [-N, N]

    Symmetry conditions enforced:
        A_n  = A_{-n}  = 1 / φ^|n|   (amplitude symmetry)
        Phase is zero for all harmonics  → no drift, perfect loop

    n < 0 : slow macro waves
    n = 0 : base cycle
    n > 0 : fast microstructure / spikes
    """
    x = np.zeros_like(t, dtype=float)
    energy_check = 0.0
    for n in range(-N, N + 1):
        omega = omega0 * (PHI ** n)
        A = 1.0 / (PHI ** abs(n))          # symmetric amplitude decay
        x += A * np.sin(omega * t)
        energy_check += A ** 2              # Σ A_n² = constant (energy conservation)
    return x


def golden_fft_detect(signal: np.ndarray, dt: float = 1.0,
                       epsilon: float = 0.18) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    FFT decomposition + φ-structure detector.

    Checks: f_{k+1} / f_k ≈ φ   (golden frequency ladder)

    Returns:
        peak_freqs  : dominant frequencies (sorted)
        ratios      : consecutive frequency ratios
        golden_score: fraction of ratios within ε of φ
    """
    n = len(signal)
    fft_vals   = np.fft.rfft(signal)
    freqs      = np.fft.rfftfreq(n, dt)
    magnitudes = np.abs(fft_vals)

    # Top-10 dominant peaks
    idx = np.argsort(magnitudes)[-10:]
    peak_freqs = np.sort(freqs[idx])
    peak_freqs = peak_freqs[peak_freqs > 0]   # discard DC / zero

    if len(peak_freqs) < 2:
        return peak_freqs, np.array([]), 0.0

    ratios = peak_freqs[1:] / np.maximum(peak_freqs[:-1], 1e-12)

    # Count ratios close to φ (or φ² = φ+1, or 1/φ)
    golden_targets = np.array([PHI, PHI_SQ, PHI_INV])
    hits = sum(
        float(np.min(np.abs(r - golden_targets))) < epsilon
        for r in ratios
    )
    golden_score = hits / len(ratios)

    return peak_freqs, ratios, float(golden_score)


def compute_phase_alignment(close_prices: List[float],
                             dt: float = 1.0,
                             omega0: float = None,
                             N: int = 3,
                             epsilon: float = 0.18) -> Dict:
    """
    Full golden-harmonic phase analysis on a price series.

    Steps:
      1. Normalise to zero-mean (remove trend bias)
      2. FFT → detect φ-ratio structure in dominant frequencies
      3. Measure energy state: E(t) = x²  (compression vs expansion)
      4. Assess symmetry: is price near stationary min? (bounce setup)
      5. Blend into spike_prob [0..1]

    Energy flow model:
        COMPRESSION → BUILDING → EQUILIBRIUM → EXPANSION → collapse → repeat
    """
    if len(close_prices) < 64:
        return {
            "golden_score":  0.0,
            "energy_state":  "INSUFFICIENT",
            "energy_ratio":  1.0,
            "spike_prob":    0.0,
            "phase_aligned": False,
            "near_min":      False,
            "ratios":        [],
            "pos_in_range":  0.5,
        }

    arr = np.array(close_prices, dtype=float)
    arr_norm = arr - np.mean(arr)               # zero-mean

    if omega0 is None:
        omega0 = 2.0 * np.pi / len(arr_norm)    # base period = full window

    peak_freqs, ratios, golden_score = golden_fft_detect(arr_norm, dt, epsilon)

    # --- Energy state (E(t) = x²) ---
    energy = arr_norm ** 2
    mid = len(energy) // 2
    early_e  = float(np.mean(energy[:mid]))
    recent_e = float(np.mean(energy[mid:]))
    energy_ratio = recent_e / (early_e + 1e-9)

    if energy_ratio < 0.40:
        energy_state = "COMPRESSION"      # energy collapsed → buildup imminent
    elif energy_ratio < 0.75:
        energy_state = "BUILDING"         # energy gathering
    elif energy_ratio < 1.40:
        energy_state = "EQUILIBRIUM"      # balanced oscillation
    elif energy_ratio < 2.50:
        energy_state = "EXPANSION"        # active spike / trend
    else:
        energy_state = "PEAK"             # overextended → reversal risk

    # --- Position in range (how close to stationary minimum?) ---
    arr_min, arr_max = float(arr.min()), float(arr.max())
    rng = arr_max - arr_min
    pos_in_range = (float(arr[-1]) - arr_min) / (rng + 1e-9)
    near_min = pos_in_range < 0.25

    # --- Phase alignment: high golden_score + compression/building ---
    phase_aligned = (golden_score > 0.30 and
                     energy_state in ("COMPRESSION", "BUILDING"))

    # --- Spike probability ---
    # compression factor: highest when energy is compressed
    energy_bonus = {
        "COMPRESSION": 1.0,
        "BUILDING":    0.75,
        "EQUILIBRIUM": 0.40,
        "EXPANSION":   0.20,
        "PEAK":        0.05,
    }.get(energy_state, 0.0)

    spike_prob = (0.45 * golden_score +
                  0.35 * energy_bonus +
                  0.20 * float(near_min))
    spike_prob = float(np.clip(spike_prob, 0.0, 1.0))

    return {
        "golden_score":  float(golden_score),
        "energy_state":  energy_state,
        "energy_ratio":  float(energy_ratio),
        "spike_prob":    spike_prob,
        "phase_aligned": bool(phase_aligned),
        "near_min":      bool(near_min),
        "ratios":        [float(r) for r in ratios],
        "pos_in_range":  float(pos_in_range),
    }


def golden_fib_proximity(current_price: float,
                          ref_low: float,
                          ref_high: float) -> Dict:
    """
    Measures how close current_price is to each φ-derived Fibonacci level.
    Returns the nearest level name + distance%.
    Used to give golden-ratio context to structural S/R zones.
    """
    rng = ref_high - ref_low
    if rng <= 0:
        return {"nearest": "NONE", "dist_pct": 0.0, "level_price": current_price}

    results = {}
    for label, ratio in FIB_RATIOS.items():
        level_price = ref_low + rng * ratio
        dist_pct = abs(current_price - level_price) / current_price * 100.0
        results[label] = {"price": level_price, "ratio": ratio, "dist_pct": dist_pct}

    nearest = min(results, key=lambda k: results[k]["dist_pct"])
    return {
        "nearest":     nearest,
        "dist_pct":    results[nearest]["dist_pct"],
        "level_price": results[nearest]["price"],
        "all_levels":  results,
    }


# ==========================================
# RATE LIMITER & TRADER
# ==========================================

class RateLimiter:
    def __init__(self, requests_per_second: float = 15, burst: int = 25):
        self.rate = requests_per_second
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self.lock = Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_update = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            time.sleep(0.05)


class Trader:
    def __init__(self, credentials_file: str):
        self.connect(credentials_file)
        self.rate_limiter = RateLimiter(requests_per_second=15, burst=30)

    def connect(self, file: str):
        with open(file) as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) < 2:
            raise ValueError("credentials.txt must contain API key on line 1 and secret on line 2")
        self.client = Client(lines[0], lines[1])

    def get_usdc_pairs(self) -> List[str]:
        exchange_info = self.client.get_exchange_info()
        pairs = [
            symbol['symbol']
            for symbol in exchange_info['symbols']
            if symbol['quoteAsset'] == 'USDC' and symbol['status'] == 'TRADING'
        ]
        print(f"Found {len(pairs)} USDC trading pairs")
        return pairs

    def get_klines(self, symbol: str, interval: str, limit: int = 500,
                   return_raw: bool = False, start_time: int = None, end_time: int = None):
        self.rate_limiter.acquire()
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        if start_time is not None:
            params['startTime'] = start_time
        if end_time is not None:
            params['endTime'] = end_time
        for attempt in range(3):
            try:
                klines = self.client.get_klines(**params)
                if return_raw:
                    return klines
                return [float(k[4]) for k in klines]
            except Exception as e:
                if 'rate limit' in str(e).lower():
                    time.sleep(2 ** attempt * 2)
                else:
                    time.sleep(0.5)
        return []

    def get_klines_extended(self, symbol: str, interval: str, total: int = 1200):
        MAX = 1000
        if total <= MAX:
            return self.get_klines(symbol, interval, limit=total, return_raw=True)
        first = self.get_klines(symbol, interval, limit=MAX, return_raw=True)
        if not first:
            return []
        remaining = total - MAX
        end_time = int(first[0][0]) - 1
        second = self.get_klines(symbol, interval, limit=remaining,
                                 return_raw=True, end_time=end_time)
        return (second + first) if second else first


# ==========================================
# VOLUME BREAKDOWN (per-TF)
# ==========================================

def get_volume_breakdown(trader: Trader, symbol: str, interval: str,
                         limit: int = 100) -> Dict:
    klines = trader.get_klines(symbol, interval, limit=limit, return_raw=True)
    if not klines:
        return {'bull_pct': 50.0, 'bear_pct': 50.0, 'total': 0}
    bull = sum(float(k[5]) for k in klines if float(k[4]) >= float(k[1]))
    bear = sum(float(k[5]) for k in klines if float(k[4]) < float(k[1]))
    tot = bull + bear
    return {
        'bull_pct': bull / tot * 100 if tot > 0 else 50.0,
        'bear_pct': bear / tot * 100 if tot > 0 else 50.0,
        'bull': bull, 'bear': bear, 'total': tot,
    }


# ==========================================
# CANDIDATE FILTER INDICATORS
# ==========================================

def linear_regression_dip(close: List[float], deviation: float = 0.01) -> bool:
    if len(close) < 20:
        return False
    x = np.arange(len(close))
    slope, intercept = np.polyfit(x, close, 1)
    trend = slope * x + intercept
    lower_band = trend * (1 - deviation)
    return close[-1] < lower_band[-1]

def has_bullish_rejection_volume(raw_klines: list, window: int = 10) -> Tuple[bool, float]:
    if not raw_klines or len(raw_klines) < window:
        return False, 0.0
    recent = raw_klines[-window:]
    bull_vol = bear_vol = 0.0
    for k in recent:
        o, c, v = float(k[1]), float(k[4]), float(k[5])
        if v > 0:
            if c > o:
                bull_vol += v
            elif c < o:
                bear_vol += v
    total = bull_vol + bear_vol
    if total == 0:
        return False, 0.0
    ratio = bull_vol / total
    return ratio > 0.65, ratio


def calculate_effort_result_metrics(close: List[float], volumes: List[float],
                                     window: int = 20) -> Dict:
    if len(close) < window + 2:
        return {"R": 0, "C": 0, "E": 0}
    ca = np.array(close[-window:], dtype='float64')
    va = np.array(volumes[-window:], dtype='float64')
    dp = abs(ca[-1] - ca[0])
    tv = np.sum(va)
    eps = 1e-9
    return {"R": tv / (dp + eps), "C": tv / (np.std(ca) + eps),
            "E": tv / ((dp * window) + eps)}


def ml_spike_probability(R, C, E, bull_ratio, cmo, vratio) -> float:
    score = (0.30 * np.log1p(R) + 0.25 * np.log1p(C) + 0.20 * np.log1p(E) +
             0.15 * bull_ratio + 0.05 * (-cmo / 100.0) +
             0.05 * min(vratio / 5.0, 1.0))
    return 1 / (1 + np.exp(-score))


# ==========================================
# STRUCTURAL RANGE ENGINE (multi-lookback)
# ==========================================

def get_structural_extremes(close: np.ndarray, highs: np.ndarray,
                            lows: np.ndarray, lookback: int) -> Dict:
    n = len(close)
    start = max(0, n - lookback)
    c = close[start:]
    h = highs[start:]
    l = lows[start:]
    sl = len(c)

    amax_i = int(np.argmax(c))
    amin_i = int(np.argmin(c))
    g_high = float(c[amax_i])
    g_low  = float(c[amin_i])

    high_age = sl - amax_i
    low_age  = sl - amin_i
    if low_age < high_age:
        more_recent = "ARGMIN"
        mr_label = "🟢 ARGMIN (low is fresher → floor established recently)"
    elif high_age < low_age:
        more_recent = "ARGMAX"
        mr_label = "🔴 ARGMAX (high is fresher → ceiling established recently)"
    else:
        more_recent = "EQUAL"
        mr_label = "⚪ EQUAL (both extremes same age)"

    rng = g_high - g_low
    rng_pct = (rng / g_low * 100) if g_low > 0 else 0
    pos = (close[-1] - g_low) / rng if rng > 0 else 0.5

    return {
        'high': g_high, 'low': g_low,
        'high_bar': float(h[amax_i]), 'low_bar': float(l[amin_i]),
        'high_age': high_age, 'low_age': low_age,
        'more_recent': more_recent, 'mr_label': mr_label,
        'range_size': rng, 'range_pct': rng_pct,
        'position': pos, 'bars_used': sl,
    }


def build_fib_grid(extremes: Dict, current_price: float) -> List[Dict]:
    lo, hi = extremes['low'], extremes['high']
    rng = extremes['range_size']
    if rng <= 0:
        return []
    fibs = [(0.000, "ARGMIN"), (0.236, "F236"), (0.382, "F382"),
            (0.500, "F500"), (0.618, "F618"), (0.786, "F786"), (1.000, "ARGMAX")]
    grid = []
    for fib, label in fibs:
        price = lo + rng * fib
        dist = (price - current_price) / current_price * 100
        direction = 'UP' if price > current_price else ('DOWN' if price < current_price else 'AT')
        grid.append({'price': price, 'fib': fib, 'label': label,
                     'dist_pct': dist, 'direction': direction})
    return grid


def volume_profile_at_level(level_price: float, raw_klines: list,
                             tolerance: float) -> Dict:
    bull_vol = bear_vol = 0.0
    touches = bull_rej = bear_rej = 0
    vol_seq = []

    lo = level_price - tolerance
    hi = level_price + tolerance

    for k in raw_klines:
        o, h, l, c, v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
        if l <= hi and h >= lo:
            touches += 1
            vol_seq.append(v)
            if v > 0:
                if c >= o:
                    bull_vol += v
                else:
                    bear_vol += v
            if l < lo and c >= level_price:
                bull_rej += 1
            elif h > hi and c <= level_price:
                bear_rej += 1

    total = bull_vol + bear_vol
    bp = bull_vol / total if total > 0 else 0.5

    exh = 0.0
    exh_detail = "N/A"
    if len(vol_seq) >= 6:
        mid = len(vol_seq) // 2
        first_avg = np.mean(vol_seq[:mid])
        second_avg = np.mean(vol_seq[mid:])
        if first_avg > 0:
            ratio = second_avg / first_avg
            exh = max(0.0, min(1.0, 1.0 - ratio))
            if ratio < 0.5:
                exh_detail = f"STRONG ({ratio:.0%})"
            elif ratio < 0.8:
                exh_detail = f"MODERATE ({ratio:.0%})"
            else:
                exh_detail = f"Weak ({ratio:.0%})"

    rej_vol_total = 0.0
    for k in raw_klines:
        o, h, l, c, v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
        if l <= hi and h >= lo:
            if (l < lo and c >= level_price) or (h > hi and c <= level_price):
                rej_vol_total += v

    rej_int = rej_vol_total / total if total > 0 else 0.0

    if bp > 0.58:
        verdict = "SUPPORT"
    elif bp < 0.42:
        verdict = "RESISTANCE"
    else:
        verdict = "NEUTRAL"

    return {'bull_pct': bp, 'total_volume': total, 'touches': touches,
            'bull_rej': bull_rej, 'bear_rej': bear_rej,
            'total_rej': bull_rej + bear_rej,
            'exhaustion': exh, 'exhaustion_detail': exh_detail,
            'rej_intensity': rej_int, 'verdict': verdict}


# ==========================================
# REJECTION CLUSTER ENERGY & SPIKE DETECTION
# ==========================================

def compute_rejection_cluster_score(vp: Dict, range_size: float) -> Dict:
    """
    Converts raw rejection data into spike energy score.
    Score = N² · RI · (V / range_size), log-normalised.
    """
    N  = vp['total_rej']
    RI = vp['rej_intensity']
    V  = vp['total_volume']

    if range_size <= 0 or V == 0:
        return {"score": 0.0, "raw": 0.0, "N": N, "state": "INVALID"}

    energy = (N ** 2) * RI * (V / range_size)
    norm_energy = np.log1p(energy)

    if N < 3:
        state = "NOISE"
    elif N <= 4:
        state = "BUILDING"
    elif N <= 6:
        state = "COMPRESSION"
    else:
        state = "UNSTABLE"

    return {"score": norm_energy, "raw": energy, "N": N, "state": state}


def detect_spike_trigger(curr_vp: Dict, prev_vp: Dict) -> bool:
    """
    Detects the release moment: last rejection weakening while volume stays high.
    """
    if not prev_vp:
        return False
    weakening        = curr_vp['total_rej']    <= prev_vp['total_rej']
    weaker_intensity = curr_vp['rej_intensity'] <  prev_vp['rej_intensity']
    pressure         = curr_vp['total_volume']  >  prev_vp['total_volume'] * 0.8
    return weakening and weaker_intensity and pressure


def is_valid_spike(cluster: Dict, vp: Dict, vol_bias: float) -> bool:
    """
    Final gate: only COMPRESSION / UNSTABLE zones with real energy and
    neutral volume bias (accumulation, not already trending) pass.
    """
    return (
        cluster['state'] in ["COMPRESSION", "UNSTABLE"] and
        cluster['score'] > 1.5 and
        vp['rej_intensity'] > 0.2 and
        0.45 < vol_bias < 0.65
    )


def detect_cluster_transition(cluster: Dict, prev_cluster: Dict) -> bool:
    """
    Detects the key COMPRESSION → UNSTABLE transition.
    This is the highest-probability spike setup.
    """
    if not prev_cluster:
        return False
    return (
        prev_cluster['state'] == "COMPRESSION" and
        cluster['state'] == "UNSTABLE"
    )


def detect_extreme_exhaustion(extreme_price: float, direction: str,
                               raw_klines: list, zone_pct: float = 0.04) -> Dict:
    if direction == 'high':
        z_lo = extreme_price * (1 - zone_pct)
        z_hi = extreme_price * (1 + zone_pct * 0.5)
    else:
        z_lo = extreme_price * (1 - zone_pct * 0.5)
        z_hi = extreme_price * (1 + zone_pct)

    zv = []
    zc = []
    for k in raw_klines:
        h, l, c, v = float(k[2]), float(k[3]), float(k[4]), float(k[5])
        if l <= z_hi and h >= z_lo:
            zv.append(v)
            zc.append(c)

    if len(zv) < 8:
        return {'exhaustion': 0.0, 'detail': 'Insufficient data',
                'pattern': 'NONE', 'approach_vol': 0, 'final_vol': 0}

    split   = int(len(zv) * 0.6)
    app_vol = np.mean(zv[:split])
    fin_vol = np.mean(zv[split:])

    final_c = zc[split:]
    if direction == 'high':
        reached = max(final_c) >= extreme_price * 0.998
    else:
        reached = min(final_c) <= extreme_price * 1.002

    exh     = 0.0
    pattern = "NONE"
    detail  = ""

    if app_vol > 0:
        ratio = fin_vol / app_vol
        if direction == 'high':
            if ratio < 0.4 and reached:
                exh, pattern = 0.9, "CLIMAX_EXHAUSTION"
                detail = f"Vol collapsed to {ratio:.0%} at peak → rejection likely"
            elif ratio < 0.65 and reached:
                exh, pattern = 0.6, "FADE"
                detail = f"Vol faded to {ratio:.0%} near high"
            elif ratio < 0.85:
                exh, pattern = 0.3, "MILD_FADE"
                detail = f"Slight fade to {ratio:.0%}"
            else:
                detail = f"No exhaustion (vol at {ratio:.0%})"
        else:
            if ratio < 0.35 and reached:
                exh, pattern = 0.9, "CAPITULATION"
                detail = f"Vol died to {ratio:.0%} after low → bounce likely"
            elif ratio < 0.6 and reached:
                exh, pattern = 0.6, "SELLING_EXHAUSTION"
                detail = f"Selling exhausted at {ratio:.0%}"
            elif ratio < 0.85:
                exh, pattern = 0.3, "MILD_EXHAUSTION"
                detail = f"Mild exhaustion at {ratio:.0%}"
            else:
                detail = f"No exhaustion (vol at {ratio:.0%})"
    else:
        detail = "No approach volume"

    return {'exhaustion': exh, 'detail': detail, 'pattern': pattern,
            'approach_vol': app_vol, 'final_vol': fin_vol}


def score_level(level: Dict, vp: Dict, extreme_exh: float,
                range_pct: float, is_above: bool) -> Tuple[float, Dict]:
    bp     = vp['bull_pct']
    touches = vp['touches']
    rej    = vp['total_rej']
    ri     = vp['rej_intensity']
    exh    = vp['exhaustion']

    pressure  = max(0.0, (0.5 - bp) * 2.0) if is_above else max(0.0, (bp - 0.5) * 2.0)
    rej_bonus = (vp['bear_rej'] if is_above else vp['bull_rej']) / max(touches, 1)
    touch_sc  = min(touches / 12.0, 1.0) if touches > 0 else 0.0
    rej_sc    = min(rej_bonus * 3.0, 1.0)
    ri_sc     = min(ri * 5.0, 1.0)
    exh_sc    = extreme_exh

    fib = level['fib']
    if is_above:
        ext_prox = max(0.0, 1.0 - abs(fib - 1.0) * 2.0)
    else:
        ext_prox = max(0.0, 1.0 - abs(fib - 0.0) * 2.0)

    score = (0.25 * pressure + 0.15 * touch_sc + 0.15 * rej_sc +
             0.15 * ri_sc + 0.20 * exh_sc + 0.10 * ext_prox)

    details = {'pressure': pressure, 'touches': touch_sc, 'rejection_candles': rej_sc,
               'rejection_intensity': ri_sc, 'exhaustion': exh_sc, 'extreme_prox': ext_prox}
    return score, details


def estimate_eta(dist_pct: float, range_pct: float, vol_bias: float) -> str:
    if range_pct <= 0:
        return "N/A"
    bias = vol_bias if vol_bias >= 0.5 else (1.0 - vol_bias)
    if (dist_pct > 0 and vol_bias > 0.5) or (dist_pct < 0 and vol_bias < 0.5):
        speed = 0.7 + 0.6 * bias
    else:
        speed = 1.2 + 0.8 * (1.0 - bias)
    mins = abs(dist_pct) / max(range_pct, 0.01) * 360 * speed
    if mins < 5:     return "~1-5 min"
    elif mins < 15:  return "~5-15 min"
    elif mins < 30:  return "~15-30 min"
    elif mins < 60:  return "~30-60 min"
    elif mins < 120: return "~1-2 hrs"
    elif mins < 240: return "~2-4 hrs"
    else:            return f"~{mins/60:.1f}+ hrs"


def analyze_lookback(raw_klines: list, close: np.ndarray, highs: np.ndarray,
                      lows: np.ndarray, current_price: float, lookback: int,
                      avg_range_pct: float, vol_bias: float) -> Dict:
    ext = get_structural_extremes(close, highs, lows, lookback)

    if ext['range_pct'] < 0.1:
        return {'lookback': lookback, 'extremes': ext, 'targets_up': [],
                'targets_down': [], 'exh_high': {}, 'exh_low': {},
                'grid': [], 'min_dist': 0}

    grid      = build_fib_grid(ext, current_price)
    tolerance = max(ext['range_size'] * 0.025,
                    avg_range_pct / 100 * current_price * 1.5)

    n     = len(close)
    start = max(0, n - lookback)
    klines_slice = raw_klines[start:]

    exh_high = detect_extreme_exhaustion(ext['high'], 'high', klines_slice)
    exh_low  = detect_extreme_exhaustion(ext['low'],  'low',  klines_slice)

    min_dist = max(ext['range_pct'] * 0.05, 0.08)

    targets_up   = []
    targets_down = []
    grid_out     = []

    prev_vp      = None
    prev_cluster = None

    for level in grid:
        vp = volume_profile_at_level(level['price'], klines_slice, tolerance)

        # Rejection cluster energy
        cluster     = compute_rejection_cluster_score(vp, ext['range_size'])
        trigger     = detect_spike_trigger(vp, prev_vp)
        valid_spike = is_valid_spike(cluster, vp, vol_bias)
        explosion   = detect_cluster_transition(cluster, prev_cluster)

        prev_vp      = vp
        prev_cluster = cluster

        if level['fib'] >= 0.618:
            lev_exh = exh_high['exhaustion']
        elif level['fib'] <= 0.382:
            lev_exh = exh_low['exhaustion']
        else:
            lev_exh = 0.0

        dist     = abs(level['dist_pct'])
        is_above = level['direction'] == 'UP'
        is_below = level['direction'] == 'DOWN'

        if dist < min_dist:
            grid_out.append({**level, **vp, 'score': 0, 'status': 'TOO_CLOSE',
                             'cluster_score': cluster['score'],
                             'cluster_state': cluster['state'],
                             'valid_spike': valid_spike, 'trigger': trigger,
                             'explosion': explosion})
            continue

        score, details = score_level(level, vp, lev_exh, ext['range_pct'], is_above)

        # Boost score with cluster energy (max +0.30)
        score += min(cluster['score'] * 0.15, 0.3)

        eta = estimate_eta(level['dist_pct'], ext['range_pct'], vol_bias)

        entry = {'price': level['price'], 'score': score, 'dist_pct': level['dist_pct'],
                 'label': level['label'], 'fib': level['fib'],
                 'verdict': vp['verdict'], 'bull_pct': vp['bull_pct'],
                 'touches': vp['touches'], 'rejections': vp['total_rej'],
                 'rej_intensity': vp['rej_intensity'],
                 'cluster_score': cluster['score'], 'cluster_state': cluster['state'],
                 'cluster_raw': cluster['raw'], 'trigger': trigger,
                 'valid_spike': valid_spike, 'explosion': explosion,
                 'eta': eta, 'details': details}

        grid_out.append({**level, **vp, **entry, 'status': 'ACTIVE'})

        if is_above:
            targets_up.append(entry)
        elif is_below:
            targets_down.append(entry)

    targets_up.sort(key=lambda t: t['score'], reverse=True)
    targets_down.sort(key=lambda t: t['score'], reverse=True)
    top_up = sorted(targets_up[:4],  key=lambda t: t['dist_pct'])
    top_dn = sorted(targets_down[:4], key=lambda t: -t['dist_pct'])

    return {'lookback': lookback, 'extremes': ext,
            'targets_up': top_up, 'targets_down': top_dn,
            'exh_high': exh_high, 'exh_low': exh_low,
            'grid': grid_out, 'min_dist': min_dist}


def get_sr_targets(raw_klines: list, current_price: float) -> Dict:
    if len(raw_klines) < 100:
        return {'lookbacks': [], 'vol_bias': 0.5, 'avg_range': 0}

    highs   = np.array([float(k[2]) for k in raw_klines])
    lows    = np.array([float(k[3]) for k in raw_klines])
    closes  = np.array([float(k[4]) for k in raw_klines])
    volumes = np.array([float(k[5]) for k in raw_klines])

    candle_ranges = (highs - lows) / (closes + 1e-12) * 100.0
    avg_range     = float(np.mean(candle_ranges[-50:]))

    closed_vols = [v for v in volumes[-21:-1] if v > 0]
    if closed_vols:
        rec    = raw_klines[-21:-1]
        bv     = sum(float(k[5]) for k in rec if float(k[4]) >= float(k[1]) and float(k[5]) > 0)
        bear_v = sum(float(k[5]) for k in rec if float(k[4]) < float(k[1]) and float(k[5]) > 0)
        tv     = bv + bear_v
        vol_bias = bv / tv if tv > 0 else 0.5
    else:
        vol_bias = 0.5

    lookbacks = []
    for lb in [500, 800, 1200]:
        if len(raw_klines) >= lb:
            result = analyze_lookback(raw_klines, closes, highs, lows,
                                       current_price, lb, avg_range, vol_bias)
            lookbacks.append(result)

    return {'lookbacks': lookbacks, 'vol_bias': vol_bias, 'avg_range': avg_range}


# ==========================================
# CONCURRENT FILTER FUNCTIONS
# ==========================================

def check_tf_dip(trader, symbol, interval):
    close = trader.get_klines(symbol, interval, limit=300)
    return (symbol, linear_regression_dip(close, 0.01))


def check_1m_final(trader, symbol):
    """
    Extended 1m analysis including:
    - CMO oversold check
    - Volume ratio
    - Bullish rejection volume
    - ML spike probability (effort/result model)
    - Golden harmonic phase alignment (new φ engine)
    """
    klines = trader.get_klines(symbol, '1m', limit=200, return_raw=True)
    if not klines or len(klines) < 50:
        return (symbol, 0.0, 0.0, False, 0.0, 0.0,
                {"golden_score": 0.0, "energy_state": "INSUFFICIENT",
                 "spike_prob": 0.0, "phase_aligned": False})

    close   = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    cmo     = ta.CMO(np.asarray(close), timeperiod=14)
    cmo_val = float(cmo[-1]) if not np.isnan(cmo[-1]) else 0.0

    closed_vols = [v for v in volumes[:-1] if v > 0]
    if closed_vols:
        avg_vol     = np.mean(closed_vols[-50:])
        last_closed = closed_vols[-1]
        vratio      = last_closed / avg_vol if avg_vol > 0 else 0.0
    else:
        vratio = 0.0

    is_rej, bull_ratio = has_bullish_rejection_volume(klines, window=10)
    metrics = calculate_effort_result_metrics(close, volumes, window=20)
    prob    = ml_spike_probability(metrics["R"], metrics["C"], metrics["E"],
                                   bull_ratio, cmo_val, vratio)

    is_strong = linear_regression_dip(close, 0.01)

    # ─── Golden Harmonic Phase Alignment ───────────────────────────────────
    golden = compute_phase_alignment(close, dt=1.0, N=3, epsilon=0.18)
    # ───────────────────────────────────────────────────────────────────────

    return (symbol, cmo_val, vratio, is_strong, bull_ratio, prob, golden)


class ProgressTracker:
    def __init__(self, total, label):
        self.total, self.label = total, label
        self.completed = self.passed = 0
        self.lock = Lock()
        self.start_time = time.time()

    def update(self, passed=False):
        with self.lock:
            self.completed += 1
            if passed:
                self.passed += 1

    def get_stats(self):
        with self.lock:
            e   = time.time() - self.start_time
            r   = self.completed / e if e > 0 else 0
            rem = (self.total - self.completed) / r if r > 0 else 0
            return (f"\r{self.label}: {self.completed}/{self.total} | "
                    f"✓{self.passed} | {r:.1f}/s | ETA: {rem:.0f}s")


def run_tf_filter(trader, symbols, interval, max_workers=20):
    passed  = []
    tracker = ProgressTracker(len(symbols), f"{interval} filter")
    print(f"Running {interval} filter on {len(symbols)} pairs...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(check_tf_dip, trader, s, interval): s for s in symbols}
        for f in as_completed(futures):
            try:
                sym, ok = f.result()
                if ok:
                    passed.append(sym)
                tracker.update(passed=ok)
                print(tracker.get_stats(), end="", flush=True)
            except:
                tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20)
    return passed


def run_1m_filter(trader, symbols, max_workers=15):
    results = []
    tracker = ProgressTracker(len(symbols), "1m filter")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(check_1m_final, trader, s): s for s in symbols}
        for f in as_completed(futures):
            try:
                res = f.result()
                results.append(res)
                tracker.update(passed=res[3])
                print(tracker.get_stats(), end="", flush=True)
            except:
                tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20)
    return results


# ==========================================
# OUTPUT FORMATTER
# ==========================================

def format_golden_block(golden: Dict, W: int = 74):
    """Prints the golden harmonic analysis block."""
    gs     = golden.get("golden_score",  0.0)
    estate = golden.get("energy_state",  "N/A")
    sp     = golden.get("spike_prob",    0.0)
    pa     = golden.get("phase_aligned", False)
    nm     = golden.get("near_min",      False)
    er     = golden.get("energy_ratio",  1.0)
    pos    = golden.get("pos_in_range",  0.5)
    ratios = golden.get("ratios",        [])

    estate_icon = {
        "COMPRESSION": "🔵",
        "BUILDING":    "🟡",
        "EQUILIBRIUM": "⚪",
        "EXPANSION":   "🟠",
        "PEAK":        "🔴",
    }.get(estate, "⚪")

    print("─" * W)
    print("  ✨  GOLDEN HARMONIC ENGINE  (φ = 1.6180…)")
    print("─" * W)
    print(f"  φ Score (FFT)    : {gs*100:.1f}%  "
          f"({'φ-structure detected' if gs > 0.3 else 'weak φ-structure'})")
    print(f"  Energy State     : {estate_icon} {estate}  (ratio={er:.3f})")
    print(f"  Phase Aligned    : {'✅ YES — harmonics converging' if pa else '❌ NO'}")
    print(f"  Near Cycle Min   : {'✅ YES — stationary floor proximity' if nm else '❌ NO'}")
    print(f"  Pos in Range     : {pos*100:.1f}%")
    print(f"  Golden Spike Prob: {sp*100:.1f}%")

    if ratios:
        ratio_str = "  ".join(f"{r:.3f}" for r in ratios[:6])
        phi_hits  = sum(1 for r in ratios if abs(r - PHI) < 0.18 or
                                              abs(r - PHI_SQ) < 0.18 or
                                              abs(r - PHI_INV) < 0.18)
        print(f"  FFT Ratios       : {ratio_str}")
        print(f"  φ-ratio hits     : {phi_hits}/{len(ratios)}  "
              f"(φ≈{PHI:.3f}  φ²≈{PHI_SQ:.3f}  1/φ≈{PHI_INV:.3f})")

    # Energy flow diagram
    states = ["COMPRESSION", "BUILDING", "EQUILIBRIUM", "EXPANSION", "PEAK"]
    bar    = "  Flow: "
    for s in states:
        if s == estate:
            bar += f"[{s[:4]}]→"
        else:
            bar += f" {s[:4]} →"
    print(f"{bar[:-1]}")
    print()


def format_sr_output(symbol, sr, current_price, cmo_val, vratio,
                     bull_ratio, ml_prob, tf_volumes, golden: Dict = None):
    vb      = sr['vol_bias']
    avg_r   = sr['avg_range']
    bp_pct  = vb * 100
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
        bar_len  = 30
        bull_len = int(vd['bull_pct'] / 100 * bar_len)
        bear_len = bar_len - bull_len
        bar      = "🟢" * bull_len + "🔴" * bear_len
        print(f"  {tf:>4s}  [{bar}]  Bull: {vd['bull_pct']:.1f}%  Bear: {vd['bear_pct']:.1f}%")

    # ─── Golden Harmonic Block ────────────────────────────────────────────
    if golden:
        format_golden_block(golden, W)
    # ─────────────────────────────────────────────────────────────────────

    all_signals = []

    # Carry golden signals forward
    if golden:
        if golden.get("phase_aligned"):
            all_signals.append(
                f"φ Phase Aligned (score={golden['golden_score']*100:.0f}%, "
                f"state={golden['energy_state']})"
            )
        if golden.get("near_min") and golden.get("energy_state") == "COMPRESSION":
            all_signals.append("φ COMPRESSION at cycle minimum → bounce setup")
        if golden.get("spike_prob", 0) > 0.65:
            all_signals.append(f"φ Golden spike prob ({golden['spike_prob']*100:.0f}%)")

    for lb_data in sr['lookbacks']:
        lb      = lb_data['lookback']
        ext     = lb_data['extremes']
        exh_h   = lb_data['exh_high']
        exh_l   = lb_data['exh_low']
        rng_pct = ext['range_pct']
        pos     = ext['position']
        min_d   = lb_data['min_dist']

        print("\n" + "─" * W)
        print(f"  📐  LOOKBACK: {lb} BARS  ({lb} min)")
        print("─" * W)
        print(f"  Global High     : {ext['high']:.10f}  ({ext['high_age']} bars ago)")
        print(f"  Global Low      : {ext['low']:.10f}  ({ext['low_age']} bars ago)")
        print(f"  True Range      : {rng_pct:.3f}%")
        print(f"  More Recent     : {ext['mr_label']}")
        print(f"  Min Target Dist : {min_d:.3f}%")

        # Position bar
        pos_pct = pos * 100
        blen    = 40
        bpos    = int(pos * blen)
        pbar    = "─" * bpos + "▲" + "─" * (blen - bpos - 1)
        pos_txt = ('near LOW' if pos < 0.25 else
                   'near HIGH' if pos > 0.75 else 'mid-range')
        print(f"  Position        : [{pbar}]  {pos_pct:.1f}%  ({pos_txt})")

        # φ proximity to current price within this lookback range
        phi_prox = golden_fib_proximity(current_price, ext['low'], ext['high'])
        print(f"  φ Nearest Level : {phi_prox['nearest']}  "
              f"@ {phi_prox['level_price']:.10f}  "
              f"(dist {phi_prox['dist_pct']:.3f}%)")

        print(f"\n  🫁  Exhaustion at HIGH: ", end="")
        if exh_h.get('exhaustion', 0) > 0.5:
            print(f"🔴 {exh_h['pattern']} ({exh_h['exhaustion']:.2f})")
            print(f"     {exh_h['detail']}")
        elif exh_h.get('exhaustion', 0) > 0.2:
            print(f"🟡 {exh_h['pattern']} ({exh_h['exhaustion']:.2f})")
            print(f"     {exh_h['detail']}")
        else:
            print(f"⚪ {exh_h.get('pattern', 'NONE')} ({exh_h.get('exhaustion', 0):.2f})")
            print(f"     {exh_h.get('detail', '')}")

        print(f"  🫁  Exhaustion at LOW : ", end="")
        if exh_l.get('exhaustion', 0) > 0.5:
            print(f"🟢 {exh_l['pattern']} ({exh_l['exhaustion']:.2f})")
            print(f"     {exh_l['detail']}")
            if pos < 0.5:
                all_signals.append(f"[{lb}] Selling exhaustion at low ({exh_l['pattern']})")
        elif exh_l.get('exhaustion', 0) > 0.2:
            print(f"🟡 {exh_l['pattern']} ({exh_l['exhaustion']:.2f})")
            print(f"     {exh_l['detail']}")
        else:
            print(f"⚪ {exh_l.get('pattern', 'NONE')} ({exh_l.get('exhaustion', 0):.2f})")
            print(f"     {exh_l.get('detail', '')}")

        if ext['more_recent'] == 'ARGMIN':
            all_signals.append(f"[{lb}] ARGMIN more recent → recent floor")
        elif ext['more_recent'] == 'ARGMAX':
            all_signals.append(f"[{lb}] ARGMAX more recent → recent ceiling")

        grid = lb_data['grid']
        if grid:
            print(f"\n  📊  Fibonacci Grid (volume profile + cluster energy + φ levels)")
            print(f"  {'Level':<8} {'Price':>14} {'Dist%':>8} {'Bull%':>6} "
                  f"{'Tch':>4} {'Rej':>4} {'Exh':>5} {'ClSt':<12} {'ClSc':>5} {'Verdict':<10} {'St'}")
            print("  " + "─" * 88)
            for g in grid:
                st   = g.get('status', '?')
                cs   = g.get('cluster_state', '—')
                csc  = g.get('cluster_score', 0.0)
                vs   = g.get('valid_spike', False)
                cs_icon = ("💣" if cs == "UNSTABLE" else
                           "🔥" if cs == "COMPRESSION" else
                           "⚡" if cs == "BUILDING" else "·")
                m = "·" if st == 'TOO_CLOSE' else ("►" if g.get('direction') == 'UP' else "◄")
                print(f"  {m}{g['label']:<7} {g['price']:>14.8f} {g['dist_pct']:>+7.3f}% "
                      f"{g['bull_pct']*100:>5.0f}% {g['touches']:>4} {g['total_rej']:>4} "
                      f"{g['exhaustion']:>4.2f} {cs_icon}{cs:<11} {csc:>5.2f} {g['verdict']:<10} "
                      f"{'✅SPIKE' if vs else st}")

        up = lb_data['targets_up']
        if up:
            print(f"\n  📈  RESISTANCE TARGETS ({lb} bars)\n")
            for i, t in enumerate(up, 1):
                bar  = "█" * int(t['score'] * 10) + "░" * (10 - int(t['score'] * 10))
                vi   = "🔴" if t['verdict'] == "RESISTANCE" else ("🟢" if t['verdict'] == "SUPPORT" else "⚪")
                cs   = t.get('cluster_state', '—')
                csc  = t.get('cluster_score', 0.0)
                vs   = t.get('valid_spike', False)
                trig = t.get('trigger', False)
                expl = t.get('explosion', False)
                spike_tag = ""
                if expl:   spike_tag = "  💣 EXPLOSION SETUP"
                elif vs:   spike_tag = "  🔥 VALID SPIKE"
                elif trig: spike_tag = "  ⚡ TRIGGER"
                print(f"  T{i}  {t['label']:5s}  {t['price']:.10f}  ({t['dist_pct']:+.3f}%)  ETA: {t['eta']}{spike_tag}")
                print(f"       [{bar}] {t['score']:.2f}  {vi} {t['verdict']}  "
                      f"BullVol: {t['bull_pct']*100:.0f}%  Tch: {t['touches']}  "
                      f"Rej: {t['rejections']}  RejInt: {t['rej_intensity']:.2f}")
                print(f"       ClusterState: {cs:<12}  ClusterScore: {csc:.3f}")
                d     = t.get('details', {})
                parts = []
                if d.get('exhaustion', 0) > 0.2:
                    parts.append(f"Exh:{d['exhaustion']:.2f}")
                if d.get('rejection_candles', 0) > 0.2:
                    parts.append(f"Rej:{d['rejection_candles']:.2f}")
                if d.get('extreme_prox', 0) > 0.3:
                    parts.append(f"NearExt")
                if parts:
                    print(f"             + {' | '.join(parts)}")
                print()
        else:
            print(f"\n  📈  No resistance targets beyond {min_d:.3f}% minimum.\n")

        dn = lb_data['targets_down']
        if dn:
            print(f"  📉  SUPPORT LEVELS ({lb} bars)\n")
            for i, t in enumerate(dn, 1):
                bar  = "█" * int(t['score'] * 10) + "░" * (10 - int(t['score'] * 10))
                vi   = "🟢" if t['verdict'] == "SUPPORT" else ("🔴" if t['verdict'] == "RESISTANCE" else "⚪")
                cs   = t.get('cluster_state', '—')
                csc  = t.get('cluster_score', 0.0)
                vs   = t.get('valid_spike', False)
                trig = t.get('trigger', False)
                expl = t.get('explosion', False)
                spike_tag = ""
                if expl:   spike_tag = "  💣 EXPLOSION SETUP"
                elif vs:   spike_tag = "  🔥 VALID SPIKE"
                elif trig: spike_tag = "  ⚡ TRIGGER"
                print(f"  S{i}  {t['label']:5s}  {t['price']:.10f}  ({t['dist_pct']:+.3f}%)  ETA: {t['eta']}{spike_tag}")
                print(f"       [{bar}] {t['score']:.2f}  {vi} {t['verdict']}  "
                      f"BullVol: {t['bull_pct']*100:.0f}%  Tch: {t['touches']}  "
                      f"Rej: {t['rejections']}  RejInt: {t['rej_intensity']:.2f}")
                print(f"       ClusterState: {cs:<12}  ClusterScore: {csc:.3f}")
                d     = t.get('details', {})
                parts = []
                if d.get('exhaustion', 0) > 0.2:
                    parts.append(f"Exh:{d['exhaustion']:.2f}")
                if d.get('rejection_candles', 0) > 0.2:
                    parts.append(f"Rej:{d['rejection_candles']:.2f}")
                if d.get('extreme_prox', 0) > 0.3:
                    parts.append(f"NearExt")
                if parts:
                    print(f"             + {' | '.join(parts)}")
                print()
        else:
            print(f"  📉  No support levels beyond {min_d:.3f}% minimum.\n")

    print("=" * W)
    print("  ⚡  CONSOLIDATED TRADE BIAS")
    print("=" * W)

    argmin_count = sum(1 for lb in sr['lookbacks']
                       if lb['extremes']['more_recent'] == 'ARGMIN')
    argmax_count = sum(1 for lb in sr['lookbacks']
                       if lb['extremes']['more_recent'] == 'ARGMAX')
    total_lb = len(sr['lookbacks'])

    print(f"\n  Recency Across Lookbacks:")
    print(f"    ARGMIN more recent : {argmin_count}/{total_lb}")
    print(f"    ARGMAX more recent : {argmax_count}/{total_lb}")

    if argmin_count > argmax_count:
        all_signals.append(f"ARGMIN dominant across lookbacks ({argmin_count}/{total_lb})")

    # Cluster energy signals
    best_cluster_score = 0.0
    best_cluster_state = None
    explosion_found    = False
    spike_found        = False
    for lb_data in sr['lookbacks']:
        for tgt_list in [lb_data['targets_up'], lb_data['targets_down']]:
            for t in tgt_list:
                cs = t.get('cluster_score', 0.0)
                if cs > best_cluster_score:
                    best_cluster_score = cs
                    best_cluster_state = t.get('cluster_state')
                if t.get('explosion'):
                    explosion_found = True
                if t.get('valid_spike'):
                    spike_found = True

    if explosion_found:
        all_signals.append(f"💣 EXPLOSION SETUP: COMPRESSION→UNSTABLE transition detected")
    elif spike_found:
        all_signals.append(f"🔥 VALID SPIKE zone (cluster state={best_cluster_state}, score={best_cluster_score:.2f})")
    elif best_cluster_state in ("COMPRESSION", "UNSTABLE") and best_cluster_score > 1.0:
        all_signals.append(f"⚡ Cluster energy building ({best_cluster_state}, score={best_cluster_score:.2f})")

    if vb > 0.55:
        all_signals.append(f"1m Bullish vol bias ({vb*100:.0f}%)")

    if cmo_val < -50:
        all_signals.append(f"CMO oversold ({cmo_val:.0f})")

    if bull_ratio > 0.65:
        all_signals.append(f"Bull rejection vol ({bull_ratio*100:.0f}%)")

    if ml_prob > 0.65:
        all_signals.append(f"ML spike prob ({ml_prob*100:.0f}%)")

    # Enhanced probability: blend ML + cluster energy + trigger + golden
    cluster_prob   = min(best_cluster_score / 3.0, 1.0)
    trigger_bonus  = 1 if explosion_found else (0.5 if spike_found else 0)
    golden_contrib = golden.get("spike_prob", 0.0) if golden else 0.0
    enhanced_prob  = (0.35 * ml_prob +
                      0.25 * cluster_prob +
                      0.20 * trigger_bonus +
                      0.20 * golden_contrib)

    print(f"\n  ⚡  Enhanced Spike Probability (φ-augmented):")
    print(f"     ML Prob        : {ml_prob*100:.1f}%")
    print(f"     Cluster Prob   : {cluster_prob*100:.1f}%  (best score={best_cluster_score:.2f}, state={best_cluster_state or 'N/A'})")
    print(f"     Trigger Bonus  : {'EXPLOSION' if explosion_found else ('SPIKE' if spike_found else 'none')}")
    if golden:
        print(f"     φ Golden Prob  : {golden_contrib*100:.1f}%  (state={golden.get('energy_state','N/A')})")
    print(f"     FINAL PROB     : {enhanced_prob*100:.1f}%")

    best_up = best_dn = None
    for lb in reversed(sr['lookbacks']):
        if not best_up and lb['targets_up']:
            best_up = lb['targets_up'][0]
        if not best_dn and lb['targets_down']:
            best_dn = lb['targets_down'][0]
        if best_up and best_dn:
            break

    if best_up and best_dn:
        rr = abs(best_up['dist_pct']) / max(abs(best_dn['dist_pct']), 0.0001)
        print(f"\n  Best Target : {best_up['label']:5s}  {best_up['price']:.10f}  "
              f"({best_up['dist_pct']:+.3f}%)  ETA: {best_up['eta']}")
        print(f"  Best Stop   : {best_dn['label']:5s}  {best_dn['price']:.10f}  "
              f"({best_dn['dist_pct']:+.3f}%)")
        print(f"  R:R         : {rr:.2f}x")
        if rr >= 1.5:
            all_signals.append(f"R:R favorable ({rr:.1f}x)")
    elif best_up:
        print(f"\n  Target only : {best_up['label']:5s}  {best_up['price']:.10f}  "
              f"({best_up['dist_pct']:+.3f}%)")
    elif best_dn:
        print(f"\n  Support only: {best_dn['label']:5s}  {best_dn['price']:.10f}  "
              f"({best_dn['dist_pct']:+.3f}%)")
    else:
        print("\n  No structural levels found.")

    ns = len(all_signals)
    print(f"\n  Structural Signals ({ns}):")
    for s in all_signals:
        print(f"    ✅  {s}")

    print()
    if ns >= 4:   v = "✅  STRONG LONG  —  Multiple structural confirmations"
    elif ns >= 3: v = "✅  LONG  —  Good structural alignment"
    elif ns >= 2: v = "⏳  PROBABLE LONG  —  Awaiting final confirmation"
    elif ns >= 1: v = "⏳  WEAK SIGNAL  —  Insufficient confirmation"
    else:         v = "⚪  NEUTRAL  —  No clear structural bias"

    print(f"  VERDICT : {v}")
    print("=" * W + "\n")

    return v, ns


def print_rescan_banner(scan_count: int, reason: str):
    W = 78
    print("\n" + "╔" + "═" * W + "╗")
    print("║" + " " * W + "║")
    print("║" + f"  🔄  RESCAN #{scan_count} INITIATED".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("║" + f"  Reason: {reason}".ljust(W)[:W] + "║")
    print("║" + " " * W + "║")
    print("║" + "  ⏰  Clearing memory & fetching fresh data in 5s...".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("╚" + "═" * W + "╝")


def print_scan_header(scan_count: int):
    W = 78
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "╔" + "═" * W + "╗")
    print("║" + " " * W + "║")
    print("║" + f"  🔍  MTF SCANNER  —  SCAN #{scan_count}".ljust(W) + "║")
    print("║" + f"  📅  {timestamp}".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("║" + "  Multi-Lookback Structural Range Engine + φ-Harmonic Engine".ljust(W) + "║")
    print("║" + "  (500/800/1200 bar argmin·argmax · fib grid · exhaustion · φ FFT)".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("╚" + "═" * W + "╝\n")


# ==========================================
# MAIN WITH WHILE LOOP RESCAN & GC CLEANUP
# ==========================================

def main():
    total_start_time = time.time()
    scan_count = 0

    RESCAN_INTERVAL_SECONDS = 5

    W = 78
    print("=" * W)
    print("  🚀  MTF DIP SCANNER  —  CONTINUOUS MODE  (φ-Harmonic Edition)")
    print("  Scans until valid MTF dip is found, then EXITS")
    print("=" * W)
    print(f"\n  Configuration:")
    print(f"    Rescan Interval    : {RESCAN_INTERVAL_SECONDS}s (Aggressive)")
    print(f"    Memory Management  : Forced GC + Fresh Client per iteration")
    print(f"    Filter Logic       : Uniform 'linear_regression_dip' across all TFs")
    print(f"    Exit Logic         : Instant exit on MTF dip (No Warnings checked)")
    print(f"    Golden Engine      : φ={PHI:.6f}  φ²={PHI_SQ:.6f}  1/φ={PHI_INV:.6f}")
    print(f"\n  Bot will keep rescanning until a dip is found.")
    print(f"  Press Ctrl+C to manually exit.\n")

    while True:
        scan_count += 1
        scan_start_time = time.time()

        print_scan_header(scan_count)

        trader         = None
        trading_pairs  = None
        filtered1      = None
        filtered2      = None
        filtered3      = None
        results_1m     = None
        tf_volumes     = None
        klines_1m      = None
        sr             = None

        try:
            trader        = Trader('credentials.txt')
            trading_pairs = trader.get_usdc_pairs()

            filtered1 = run_tf_filter(trader, trading_pairs, '2h', 20)
            if not filtered1:
                reason = "No 2h dips found"
                print(f"\n  ⚠️  {reason}")
                print_rescan_banner(scan_count, reason)
                del trader, trading_pairs, filtered1
                gc.collect()
                time.sleep(RESCAN_INTERVAL_SECONDS)
                continue

            filtered2 = run_tf_filter(trader, filtered1, '15m', 15)
            if not filtered2:
                reason = f"No 15m dips found (from {len(filtered1)} 2h dips)"
                print(f"\n  ⚠️  {reason}")
                print_rescan_banner(scan_count, reason)
                del trader, trading_pairs, filtered1, filtered2
                gc.collect()
                time.sleep(RESCAN_INTERVAL_SECONDS)
                continue

            filtered3 = run_tf_filter(trader, filtered2, '5m', 15)
            if not filtered3:
                reason = f"No 5m dips found (from {len(filtered2)} 15m dips)"
                print(f"\n  ⚠️  {reason}")
                print_rescan_banner(scan_count, reason)
                del trader, trading_pairs, filtered1, filtered2, filtered3
                gc.collect()
                time.sleep(RESCAN_INTERVAL_SECONDS)
                continue

            results_1m = run_1m_filter(trader, filtered3, 15)

            if not results_1m:
                reason = "Failed to fetch 1m data for final filtering"
                print(f"\n  ⚠️  {reason}")
                print_rescan_banner(scan_count, reason)
                del trader, trading_pairs, filtered1, filtered2, filtered3, results_1m
                gc.collect()
                time.sleep(RESCAN_INTERVAL_SECONDS)
                continue

            # ─── Symbol selection with golden harmonic score ──────────────
            # results_1m tuple: (sym, cmo_val, vratio, is_strong, bull_ratio, ml_prob, golden)
            strong = [r for r in results_1m if r[3] is True]

            if strong:
                # Among strong, rank by: combined_score = ml_prob*0.5 + golden_spike*0.3 + (-cmo/100)*0.2
                def combined_score(r):
                    ml  = r[5]
                    gs  = r[6].get("spike_prob", 0.0) if len(r) > 6 else 0.0
                    cmo = r[1]
                    return ml * 0.5 + gs * 0.3 + (-cmo / 100.0) * 0.2

                final = max(strong, key=combined_score)
                mode  = "STRONG + ML + φ-HARMONIC"
            else:
                # Fallback: best CMO but boosted by golden alignment
                def fallback_score(r):
                    cmo_score    = -r[1]           # more negative = better
                    golden_bonus = r[6].get("spike_prob", 0.0) * 50 if len(r) > 6 else 0.0
                    return cmo_score + golden_bonus

                final = max(results_1m, key=fallback_score)
                mode  = "FALLBACK (Best CMO + φ)"
            # ──────────────────────────────────────────────────────────────

            golden_data = final[6] if len(final) > 6 else {}
            sym, cmo_val, vratio, is_strong_1m, live_bull_ratio, ml_prob = final[:6]

            print("\n" + "-" * W)
            print(f"  SELECTED SYMBOL : {sym}")
            print(f"  SELECTION MODE  : {mode}")
            print(f"  1m CMO          : {cmo_val:.4f}")
            print(f"  1m Vol Ratio    : x{vratio:.4f}")
            print(f"  Bull Rej Vol    : {live_bull_ratio*100:.2f}%")
            print(f"  ML Spike Prob   : {ml_prob*100:.2f}%")
            print(f"  1m Strong       : {'YES' if is_strong_1m else 'NO'}")
            if golden_data:
                print(f"  φ Energy State  : {golden_data.get('energy_state','N/A')}")
                print(f"  φ Golden Score  : {golden_data.get('golden_score',0)*100:.1f}%")
                print(f"  φ Spike Prob    : {golden_data.get('spike_prob',0)*100:.1f}%")
                print(f"  φ Phase Aligned : {'YES' if golden_data.get('phase_aligned') else 'NO'}")
            print("-" * W)

            tf_volumes = {}
            for tf in ['1m', '5m', '15m', '1h', '2h']:
                tf_volumes[tf] = get_volume_breakdown(trader, sym, tf, limit=100)

            klines_1m = trader.get_klines_extended(sym, '1m', total=1200)
            if not klines_1m:
                reason = f"Could not fetch 1m klines for {sym}"
                print(f"\n  ⚠️  {reason}")
                print_rescan_banner(scan_count, reason)
                del trader, trading_pairs, filtered1, filtered2, filtered3, results_1m, tf_volumes
                gc.collect()
                time.sleep(RESCAN_INTERVAL_SECONDS)
                continue

            print(f"Retrieved {len(klines_1m)} klines")

            current_price = float(klines_1m[-1][4])
            sr = get_sr_targets(klines_1m, current_price)

            # Recompute golden on the full 1200-bar window for final report
            full_close  = [float(k[4]) for k in klines_1m]
            golden_full = compute_phase_alignment(full_close, dt=1.0, N=3, epsilon=0.18)

            verdict, num_signals = format_sr_output(
                sym, sr, current_price,
                cmo_val, vratio, live_bull_ratio, ml_prob,
                tf_volumes, golden=golden_full
            )

            # MTF DIP FOUND → EXIT
            total_time = time.time() - total_start_time
            print("\n" + "╔" + "═" * W + "╗")
            print("║" + " " * W + "║")
            print("║" + "  🎯🎯🎯  MTF DIP FOUND — BOT EXITING  🎯🎯🎯".ljust(W) + "║")
            print("║" + " " * W + "║")
            print("║" + f"  Symbol: {sym}".ljust(W) + "║")
            print("║" + f"  Verdict: {verdict}".ljust(W)[:W] + "║")
            print("║" + " " * W + "║")
            print("║" + f"  φ Energy: {golden_full.get('energy_state','N/A')}  "
                  f"Score: {golden_full.get('golden_score',0)*100:.0f}%".ljust(W - 2) + "║")
            print("║" + " " * W + "║")
            print("║" + f"  Total Scans: {scan_count}".ljust(W) + "║")
            print("║" + f"  Total Time: {total_time:.1f}s ({total_time/60:.1f} min)".ljust(W) + "║")
            print("║" + " " * W + "║")
            print("╚" + "═" * W + "╝\n")

            sys.exit(0)

        except KeyboardInterrupt:
            total_time = time.time() - total_start_time
            print("\n\n" + "╔" + "═" * W + "╗")
            print("║" + " " * W + "║")
            print("║" + "  ⛔  BOT STOPPED BY USER (Ctrl+C)".ljust(W) + "║")
            print("║" + " " * W + "║")
            print("║" + f"  Total Scans Completed: {scan_count}".ljust(W) + "║")
            print("║" + f"  Total Runtime: {total_time:.1f}s ({total_time/60:.1f} min)".ljust(W) + "║")
            print("║" + " " * W + "║")
            print("╚" + "═" * W + "╝\n")
            sys.exit(0)

        except Exception as e:
            error_msg = str(e)[:60]
            print(f"\n  ❌  Error during scan #{scan_count}: {error_msg}")

            if trader:        del trader
            if trading_pairs: del trading_pairs
            if filtered1:     del filtered1
            if filtered2:     del filtered2
            if filtered3:     del filtered3
            if results_1m:    del results_1m
            if tf_volumes:    del tf_volumes
            if klines_1m:     del klines_1m
            if sr:            del sr

            gc.collect()
            time.sleep(RESCAN_INTERVAL_SECONDS)
            continue


if __name__ == "__main__":
    main()
