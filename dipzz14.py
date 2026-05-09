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

def talib_regression_dip(close: np.ndarray, highs: np.ndarray,
                          lows: np.ndarray, period: int = 500) -> bool:
    """
    TA-Lib LINEARREG 500-bar dual-channel dip filter.

    Passes when BOTH conditions hold:
      1. Recency:  the minimum of the lower regression line (lows)
                   occurs at a more recent bar than the maximum of the
                   upper regression line (highs) — i.e. the floor was
                   formed later than the ceiling.
      2. Proximity: the current close is closer to the lower regression
                   line than to the upper regression line — price is
                   hugging the floor, not floating near the ceiling.
    """
    if len(close) < period + 2:
        return False

    lower_line = ta.LINEARREG(lows.astype('float64'),  timeperiod=period)
    upper_line = ta.LINEARREG(highs.astype('float64'), timeperiod=period)

    # Strip leading NaNs — both arrays are the same length
    valid_mask = (~np.isnan(lower_line)) & (~np.isnan(upper_line))
    if valid_mask.sum() < 2:
        return False

    ll = lower_line[valid_mask]
    ul = upper_line[valid_mask]

    # --- Condition 1: recency ---
    # argmin of lower line must be MORE RECENT (higher index) than argmax of upper line
    lower_min_idx = int(np.argmin(ll))
    upper_max_idx = int(np.argmax(ul))
    recency_ok = lower_min_idx > upper_max_idx

    # --- Condition 2: proximity ---
    # Current close must be closer to the lower line's last value than to the upper line's
    last_close = float(close[-1])
    dist_lower = abs(last_close - float(ll[-1]))
    dist_upper = abs(last_close - float(ul[-1]))
    proximity_ok = dist_lower < dist_upper

    return recency_ok and proximity_ok


# Legacy scalar wrapper kept for any residual internal calls
def linear_regression_dip(close, deviation=0.01):
    arr = np.asarray(close, dtype='float64')
    return talib_regression_dip(arr, arr, arr)  # degenerate: same series for all three

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


def cosine_volume_cycle(volumes: List[float], window: int = 50) -> float:
    """
    Cosine as volume-per-time mapped onto a 360° circle.

    Cosine is the LEADING phase (cos(0) = 1 = maximum at the very start
    of the cycle).  A positive score means volume is front-loaded — buying
    pressure concentrates at the beginning of the window.  A negative score
    means volume is back-loaded — late-cycle distribution or climax.

    The full window maps to one complete revolution (0 → 2π), so this is
    literally 'volume per degree of the cycle'.

    Returns a value in [-1, +1].
    """
    if len(volumes) < window:
        return 0.0
    v = np.array(volumes[-window:], dtype='float64')
    total = float(np.sum(v))
    if total == 0.0:
        return 0.0
    # Distribute n points uniformly across [0, 2π)
    angles = np.linspace(0.0, 2.0 * np.pi, len(v), endpoint=False)
    cos_weights = np.cos(angles)
    # Volume-weighted dot product: each bar's share × its cosine position
    return float(np.dot(v / total, cos_weights))


def sine_price_correlation(close: List[float], window: int = 50) -> float:
    """
    Sine as price correlation in the stationary circuit.

    Sine is the LAGGED quadrature phase (sin(0) = 0, 90° behind cosine).
    This function projects mean-centered price onto the sine wave and
    returns the Pearson correlation — how tightly price oscillation matches
    the sine component of the cycle.

    A value near +1 means price is rising in sync with the ascending half
    of the sine wave (bullish harmonic alignment).  Near -1 means price is
    falling through the descending half (bearish).  Near 0 means price has
    no sinusoidal structure (noise or trend).

    Returns a value in [-1, +1].
    """
    if len(close) < window:
        return 0.0
    c = np.array(close[-window:], dtype='float64')
    deviations = c - np.mean(c)
    std_d = float(np.std(deviations))
    if std_d == 0.0:
        return 0.0
    angles = np.linspace(0.0, 2.0 * np.pi, len(c), endpoint=False)
    sin_wave = np.sin(angles)
    corr = float(np.corrcoef(deviations, sin_wave)[0, 1])
    return 0.0 if np.isnan(corr) else corr


def ml_spike_probability(R, C, E, bull_ratio, cmo, vratio,
                          cos_vol: float = 0.0, sin_price: float = 0.0) -> float:
    """
    Logistic spike-probability score.

    cos_vol  : cosine volume cycle [-1, +1] — positive = front-loaded volume (bullish)
    sin_price: sine price correlation [-1, +1] — positive = price in ascending sine arc
    """
    # cos_vol >0 adds bullish weight; sin_price >0 adds bullish weight
    cos_term   = max(cos_vol,   0.0)          # only reward positive phase alignment
    sin_term   = max(sin_price, 0.0)          # only reward positive correlation
    score = (0.25 * np.log1p(R) + 0.20 * np.log1p(C) + 0.15 * np.log1p(E) +
             0.15 * bull_ratio  + 0.05 * (-cmo / 100.0) +
             0.05 * min(vratio / 5.0, 1.0)   +
             0.10 * cos_term    + 0.05 * sin_term)
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
    g_low = float(c[amin_i])

    high_age = sl - amax_i
    low_age = sl - amin_i
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

    split = int(len(zv) * 0.6)
    app_vol = np.mean(zv[:split])
    fin_vol = np.mean(zv[split:])

    final_c = zc[split:]
    if direction == 'high':
        reached = max(final_c) >= extreme_price * 0.998
    else:
        reached = min(final_c) <= extreme_price * 1.002

    exh = 0.0
    pattern = "NONE"
    detail = ""

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
    bp = vp['bull_pct']
    touches = vp['touches']
    rej = vp['total_rej']
    ri = vp['rej_intensity']
    exh = vp['exhaustion']

    pressure = max(0.0, (0.5 - bp) * 2.0) if is_above else max(0.0, (bp - 0.5) * 2.0)
    rej_bonus = (vp['bear_rej'] if is_above else vp['bull_rej']) / max(touches, 1)
    touch_sc = min(touches / 12.0, 1.0) if touches > 0 else 0.0
    rej_sc = min(rej_bonus * 3.0, 1.0)
    ri_sc = min(ri * 5.0, 1.0)
    exh_sc = extreme_exh

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
    if mins < 5:    return "~1-5 min"
    elif mins < 15: return "~5-15 min"
    elif mins < 30: return "~15-30 min"
    elif mins < 60: return "~30-60 min"
    elif mins < 120:return "~1-2 hrs"
    elif mins < 240:return "~2-4 hrs"
    else:           return f"~{mins/60:.1f}+ hrs"


def analyze_lookback(raw_klines: list, close: np.ndarray, highs: np.ndarray,
                      lows: np.ndarray, current_price: float, lookback: int,
                      avg_range_pct: float, vol_bias: float) -> Dict:
    ext = get_structural_extremes(close, highs, lows, lookback)

    if ext['range_pct'] < 0.1:
        return {'lookback': lookback, 'extremes': ext, 'targets_up': [],
                'targets_down': [], 'exh_high': {}, 'exh_low': {},
                'grid': [], 'min_dist': 0}

    grid = build_fib_grid(ext, current_price)
    tolerance = max(ext['range_size'] * 0.025,
                    avg_range_pct / 100 * current_price * 1.5)

    n = len(close)
    start = max(0, n - lookback)
    klines_slice = raw_klines[start:]

    exh_high = detect_extreme_exhaustion(ext['high'], 'high', klines_slice)
    exh_low = detect_extreme_exhaustion(ext['low'], 'low', klines_slice)

    min_dist = max(ext['range_pct'] * 0.05, 0.08)

    targets_up = []
    targets_down = []
    grid_out = []

    for level in grid:
        vp = volume_profile_at_level(level['price'], klines_slice, tolerance)

        if level['fib'] >= 0.618:
            lev_exh = exh_high['exhaustion']
        elif level['fib'] <= 0.382:
            lev_exh = exh_low['exhaustion']
        else:
            lev_exh = 0.0

        dist = abs(level['dist_pct'])
        is_above = level['direction'] == 'UP'
        is_below = level['direction'] == 'DOWN'

        if dist < min_dist:
            grid_out.append({**level, **vp, 'score': 0, 'status': 'TOO_CLOSE'})
            continue

        score, details = score_level(level, vp, lev_exh, ext['range_pct'], is_above)
        eta = estimate_eta(level['dist_pct'], ext['range_pct'], vol_bias)

        entry = {'price': level['price'], 'score': score, 'dist_pct': level['dist_pct'],
                 'label': level['label'], 'fib': level['fib'],
                 'verdict': vp['verdict'], 'bull_pct': vp['bull_pct'],
                 'touches': vp['touches'], 'rejections': vp['total_rej'],
                 'rej_intensity': vp['rej_intensity'], 'eta': eta, 'details': details}

        grid_out.append({**level, **vp, **entry, 'status': 'ACTIVE'})

        if is_above:
            targets_up.append(entry)
        elif is_below:
            targets_down.append(entry)

    targets_up.sort(key=lambda t: t['score'], reverse=True)
    targets_down.sort(key=lambda t: t['score'], reverse=True)
    top_up = sorted(targets_up[:4], key=lambda t: t['dist_pct'])
    top_dn = sorted(targets_down[:4], key=lambda t: -t['dist_pct'])

    return {'lookback': lookback, 'extremes': ext,
            'targets_up': top_up, 'targets_down': top_dn,
            'exh_high': exh_high, 'exh_low': exh_low,
            'grid': grid_out, 'min_dist': min_dist}


def get_sr_targets(raw_klines: list, current_price: float) -> Dict:
    if len(raw_klines) < 100:
        return {'lookbacks': [], 'vol_bias': 0.5, 'avg_range': 0}

    highs = np.array([float(k[2]) for k in raw_klines])
    lows = np.array([float(k[3]) for k in raw_klines])
    closes = np.array([float(k[4]) for k in raw_klines])
    volumes = np.array([float(k[5]) for k in raw_klines])

    candle_ranges = (highs - lows) / (closes + 1e-12) * 100.0
    avg_range = float(np.mean(candle_ranges[-50:]))

    closed_vols = [v for v in volumes[-21:-1] if v > 0]
    if closed_vols:
        rec = raw_klines[-21:-1]
        bv = sum(float(k[5]) for k in rec if float(k[4]) >= float(k[1]) and float(k[5]) > 0)
        bear_v = sum(float(k[5]) for k in rec if float(k[4]) < float(k[1]) and float(k[5]) > 0)
        tv = bv + bear_v
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
# CONCURRENT FILTER FUNCTIONS (ALL USE SAME LOGIC)
# ==========================================

def check_tf_dip(trader, symbol, interval):
    # Need 600 bars: 500 for LINEARREG period + ~100 warm-up buffer
    klines = trader.get_klines(symbol, interval, limit=600, return_raw=True)
    if not klines or len(klines) < 502:
        return (symbol, False)
    highs  = np.array([float(k[2]) for k in klines], dtype='float64')
    lows   = np.array([float(k[3]) for k in klines], dtype='float64')
    closes = np.array([float(k[4]) for k in klines], dtype='float64')
    return (symbol, talib_regression_dip(closes, highs, lows))


def check_1m_final(trader, symbol):
    # 700 bars: 600 for talib_regression_dip + 100 for other indicators
    klines = trader.get_klines(symbol, '1m', limit=700, return_raw=True)
    if not klines or len(klines) < 502:
        return (symbol, 0.0, 0.0, False, 0.0, 0.0, 0.0, 0.0)

    close   = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    cmo = ta.CMO(np.asarray(close), timeperiod=14)
    cmo_val = float(cmo[-1]) if not np.isnan(cmo[-1]) else 0.0

    closed_vols = [v for v in volumes[:-1] if v > 0]
    if closed_vols:
        avg_vol = np.mean(closed_vols[-50:])
        last_closed = closed_vols[-1]
        vratio = last_closed / avg_vol if avg_vol > 0 else 0.0
    else:
        vratio = 0.0

    # --- Harmonic cycle metrics ---
    cos_vol   = cosine_volume_cycle(volumes, window=50)   # volume across 360° circle
    sin_price = sine_price_correlation(close,  window=50)  # price in stationary circuit

    is_rej, bull_ratio = has_bullish_rejection_volume(klines, window=10)
    metrics = calculate_effort_result_metrics(close, volumes, window=20)
    prob = ml_spike_probability(metrics["R"], metrics["C"], metrics["E"],
                                bull_ratio, cmo_val, vratio,
                                cos_vol=cos_vol, sin_price=sin_price)

    highs  = np.array([float(k[2]) for k in klines], dtype='float64')
    lows   = np.array([float(k[3]) for k in klines], dtype='float64')
    closes = np.array(close, dtype='float64')
    is_strong = talib_regression_dip(closes, highs, lows)

    return (symbol, cmo_val, vratio, is_strong, bull_ratio, prob, cos_vol, sin_price)


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
            e = time.time() - self.start_time
            r = self.completed / e if e > 0 else 0
            rem = (self.total - self.completed) / r if r > 0 else 0
            return (f"\r{self.label}: {self.completed}/{self.total} | "
                    f"✓{self.passed} | {r:.1f}/s | ETA: {rem:.0f}s")


def run_tf_filter(trader, symbols, interval, max_workers=20):
    passed = []
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
                tracker.update(passed=res[3])   # is_strong still at index 3
                print(tracker.get_stats(), end="", flush=True)
            except:
                tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20)
    return results


# ==========================================
# OUTPUT FORMATTER
# ==========================================

def format_sr_output(symbol, sr, current_price, cmo_val, vratio,
                     bull_ratio, ml_prob, tf_volumes,
                     cos_vol: float = 0.0, sin_price: float = 0.0):
    vb = sr['vol_bias']
    avg_r = sr['avg_range']
    bp_pct = vb * 100
    bias_lbl = "🟢 BULLISH" if vb > 0.55 else ("🔴 BEARISH" if vb < 0.45 else "⚪ NEUTRAL")

    W = 74
    print("\n" + "=" * W)
    print(f"  ★  STRUCTURAL RANGE S/R  —  {symbol}  ★")
    print(f"  (argmin/argmax anchored · multi-lookback · volume exhaustion)")
    print("=" * W)
    print(f"  Entry Price    : {current_price:.10f}")
    print(f"  1m CMO         : {cmo_val:+.2f}  (< -50 = oversold)")
    print(f"  Vol Ratio      : x{vratio:.2f}")
    print(f"  Bull Rej Vol   : {bull_ratio*100:.1f}%")
    print(f"  ML Spike Prob  : {ml_prob*100:.1f}%")
    print(f"  1m Vol Bias    : {bias_lbl}  ({bp_pct:.1f}% bull / {100-bp_pct:.1f}% bear)")
    print(f"  Avg 1m Range   : {avg_r:.4f}%")

    # ---- Harmonic Cycle Display -----------------------------------------------
    print("-" * W)
    print("  🌀  HARMONIC CYCLE METRICS  (cos = leading · sin = quadrature −90°)")
    # Cosine bar: maps [-1, +1] onto a 40-char bar
    BAR = 40
    mid = BAR // 2
    cos_pos = int((cos_vol + 1.0) / 2.0 * BAR)
    cos_pos = max(0, min(BAR - 1, cos_pos))
    cos_bar = list("─" * BAR)
    cos_bar[mid] = "│"
    cos_bar[cos_pos] = "◆"
    cos_lbl = "front-loaded 🟢" if cos_vol > 0.05 else ("back-loaded 🔴" if cos_vol < -0.05 else "neutral ⚪")
    print(f"  cos(vol/time)  : [{''.join(cos_bar)}]  {cos_vol:+.4f}  {cos_lbl}")
    print(f"    Volume across 360° circle — positive = vol concentrated at cycle start")

    sin_pos = int((sin_price + 1.0) / 2.0 * BAR)
    sin_pos = max(0, min(BAR - 1, sin_pos))
    sin_bar = list("─" * BAR)
    sin_bar[mid] = "│"
    sin_bar[sin_pos] = "◆"
    sin_lbl = "ascending arc 🟢" if sin_price > 0.15 else ("descending arc 🔴" if sin_price < -0.15 else "flat/noise ⚪")
    print(f"  sin(price/corr): [{''.join(sin_bar)}]  {sin_price:+.4f}  {sin_lbl}")
    print(f"    Price corr vs sine wave — positive = price in rising half of cycle")

    # Phase relationship
    phase_deg = np.degrees(np.arctan2(sin_price, cos_vol)) % 360
    quad = ("Q1 [0–90°]   cos↑ sin↑  — early cycle, vol building, price rising" if phase_deg < 90 else
            "Q2 [90–180°] cos↓ sin↑  — mid cycle, vol fading, price still rising" if phase_deg < 180 else
            "Q3 [180–270°] cos↓ sin↓  — late cycle, vol falling, price turning" if phase_deg < 270 else
            "Q4 [270–360°] cos↑ sin↓  — reset, vol re-loading, price bottoming")
    print(f"  Phase Angle    : {phase_deg:.1f}°  →  {quad}")
    # ---- End Harmonic -----------------------------------------------------------

    print("-" * W)
    print("  📊  VOLUME BREAKDOWN BY TIMEFRAME")
    for tf, vd in tf_volumes.items():
        bar_len = 30
        bull_len = int(vd['bull_pct'] / 100 * bar_len)
        bear_len = bar_len - bull_len
        bar = "🟢" * bull_len + "🔴" * bear_len
        print(f"  {tf:>4s}  [{bar}]  Bull: {vd['bull_pct']:.1f}%  Bear: {vd['bear_pct']:.1f}%")

    # ONLY TRACK SIGNALS NOW
    all_signals = []

    for lb_data in sr['lookbacks']:
        lb = lb_data['lookback']
        ext = lb_data['extremes']
        exh_h = lb_data['exh_high']
        exh_l = lb_data['exh_low']
        rng_pct = ext['range_pct']
        pos = ext['position']
        min_d = lb_data['min_dist']

        print("\n" + "─" * W)
        print(f"  📐  LOOKBACK: {lb} BARS  ({lb} min)")
        print("─" * W)
        print(f"  Global High     : {ext['high']:.10f}  ({ext['high_age']} bars ago)")
        print(f"  Global Low      : {ext['low']:.10f}  ({ext['low_age']} bars ago)")
        print(f"  True Range      : {rng_pct:.3f}%")
        print(f"  More Recent     : {ext['mr_label']}")
        print(f"  Min Target Dist : {min_d:.3f}%")

        pos_pct = pos * 100
        blen = 40
        bpos = int(pos * blen)
        pbar = "─" * bpos + "▲" + "─" * (blen - bpos - 1)
        pos_txt = ('near LOW' if pos < 0.25 else
                   'near HIGH' if pos > 0.75 else 'mid-range')
        print(f"  Position        : [{pbar}]  {pos_pct:.1f}%  ({pos_txt})")

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
            print(f"\n  📊  Fibonacci Grid (volume profile)")
            print(f"  {'Level':<8} {'Price':>14} {'Dist%':>8} {'Bull%':>6} "
                  f"{'Tch':>4} {'Rej':>4} {'Exh':>5} {'Verdict':<10} {'St'}")
            print("  " + "─" * 68)
            for g in grid:
                st = g.get('status', '?')
                m = "·" if st == 'TOO_CLOSE' else ("►" if g.get('direction') == 'UP' else "◄")
                print(f"  {m}{g['label']:<7} {g['price']:>14.8f} {g['dist_pct']:>+7.3f}% "
                      f"{g['bull_pct']*100:>5.0f}% {g['touches']:>4} {g['total_rej']:>4} "
                      f"{g['exhaustion']:>4.2f} {g['verdict']:<10} {st}")

        up = lb_data['targets_up']
        if up:
            print(f"\n  📈  RESISTANCE TARGETS ({lb} bars)\n")
            for i, t in enumerate(up, 1):
                bar = "█" * int(t['score'] * 10) + "░" * (10 - int(t['score'] * 10))
                vi = "🔴" if t['verdict'] == "RESISTANCE" else ("🟢" if t['verdict'] == "SUPPORT" else "⚪")
                print(f"  T{i}  {t['label']:5s}  {t['price']:.10f}  ({t['dist_pct']:+.3f}%)  ETA: {t['eta']}")
                print(f"       [{bar}] {t['score']:.2f}  {vi} {t['verdict']}  "
                      f"BullVol: {t['bull_pct']*100:.0f}%  Tch: {t['touches']}  "
                      f"Rej: {t['rejections']}  RejInt: {t['rej_intensity']:.2f}")
                d = t.get('details', {})
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
                bar = "█" * int(t['score'] * 10) + "░" * (10 - int(t['score'] * 10))
                vi = "🟢" if t['verdict'] == "SUPPORT" else ("🔴" if t['verdict'] == "RESISTANCE" else "⚪")
                print(f"  S{i}  {t['label']:5s}  {t['price']:.10f}  ({t['dist_pct']:+.3f}%)  ETA: {t['eta']}")
                print(f"       [{bar}] {t['score']:.2f}  {vi} {t['verdict']}  "
                      f"BullVol: {t['bull_pct']*100:.0f}%  Tch: {t['touches']}  "
                      f"Rej: {t['rejections']}  RejInt: {t['rej_intensity']:.2f}")
                d = t.get('details', {})
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

    if vb > 0.55:
        all_signals.append(f"1m Bullish vol bias ({vb*100:.0f}%)")

    if cmo_val < -50:
        all_signals.append(f"CMO oversold ({cmo_val:.0f})")

    if bull_ratio > 0.65:
        all_signals.append(f"Bull rejection vol ({bull_ratio*100:.0f}%)")

    if ml_prob > 0.65:
        all_signals.append(f"ML spike prob ({ml_prob*100:.0f}%)")

    # Harmonic cycle signals
    if cos_vol > 0.10:
        all_signals.append(f"Cos vol cycle front-loaded ({cos_vol:+.3f}) — buying pressure early in cycle")
    if sin_price > 0.20:
        all_signals.append(f"Sin price corr positive ({sin_price:+.3f}) — price in ascending sine arc")
    phase_deg = np.degrees(np.arctan2(sin_price, cos_vol)) % 360
    if phase_deg < 90:
        all_signals.append(f"Harmonic phase Q1 ({phase_deg:.0f}°) — early-cycle bullish alignment")

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

    # ONLY PRINT SIGNALS (WARNINGS COMPLETELY REMOVED)
    ns = len(all_signals)
    print(f"\n  Structural Signals ({ns}):")
    for s in all_signals:
        print(f"    ✅  {s}")

    print()
    if ns >= 4: v = "✅  STRONG LONG  —  Multiple structural confirmations"
    elif ns >= 3: v = "✅  LONG  —  Good structural alignment"
    elif ns >= 2: v = "⏳  PROBABLE LONG  —  Awaiting final confirmation"
    elif ns >= 1: v = "⏳  WEAK SIGNAL  —  Insufficient confirmation"
    else: v = "⚪  NEUTRAL  —  No clear structural bias"

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
    print("║" + "  Multi-Lookback Structural Range Engine".ljust(W) + "║")
    print("║" + "  (500 / 800 / 1200 bar argmin·argmax · fib grid · exhaustion)".ljust(W) + "║")
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
    print("  🚀  MTF DIP SCANNER  —  CONTINUOUS MODE")
    print("  Scans until valid MTF dip is found, then EXITS")
    print("=" * W)
    print(f"\n  Configuration:")
    print(f"    Rescan Interval    : {RESCAN_INTERVAL_SECONDS}s (Aggressive)")
    print(f"    Memory Management  : Forced GC + Fresh Client per iteration")
    print(f"    Filter Logic       : Uniform 'linear_regression_dip' across all TFs")
    print(f"    Exit Logic         : Instant exit on MTF dip (No Warnings checked)")
    print(f"\n  Bot will keep rescanning until a dip is found.")
    print(f"  Press Ctrl+C to manually exit.\n")
    
    while True:
        scan_count += 1
        scan_start_time = time.time()
        
        print_scan_header(scan_count)
        
        trader = None
        trading_pairs = None
        filtered1 = None
        filtered2 = None
        filtered3 = None
        results_1m = None
        tf_volumes = None
        klines_1m = None
        sr = None
        
        try:
            trader = Trader('credentials.txt')
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
            
            strong = [r for r in results_1m if r[3] is True]
            if strong:
                final = max(strong, key=lambda x: (x[5], x[6], -x[1]))  # ml_prob, cos_vol, best cmo
                mode = "STRONG + ML ENERGY"
            else:
                final = min(results_1m, key=lambda x: x[1])
                mode = "FALLBACK (Best CMO)"
            
            sym, cmo_val, vratio, is_strong_1m, live_bull_ratio, ml_prob, cos_vol, sin_price = final
            
            print("\n" + "-" * W)
            print(f"  SELECTED SYMBOL : {sym}")
            print(f"  SELECTION MODE  : {mode}")
            print(f"  1m CMO          : {cmo_val:.4f}")
            print(f"  1m Vol Ratio    : x{vratio:.4f}")
            print(f"  Bull Rej Vol    : {live_bull_ratio*100:.2f}%")
            print(f"  ML Spike Prob   : {ml_prob*100:.2f}%")
            print(f"  1m Strong       : {'YES' if is_strong_1m else 'NO'}")
            cos_lbl = "front-loaded 🟢" if cos_vol > 0.05 else ("back-loaded 🔴" if cos_vol < -0.05 else "neutral ⚪")
            sin_lbl = "ascending arc 🟢" if sin_price > 0.15 else ("descending arc 🔴" if sin_price < -0.15 else "flat ⚪")
            print(f"  Cos Vol Cycle   : {cos_vol:+.4f}  ({cos_lbl})")
            print(f"  Sin Price Corr  : {sin_price:+.4f}  ({sin_lbl})")
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
            
            verdict, num_signals = format_sr_output(
                sym, sr, current_price,
                cmo_val, vratio, live_bull_ratio, ml_prob, tf_volumes,
                cos_vol=cos_vol, sin_price=sin_price
            )
            
            # ==========================================
            # MTF DIP FOUND -> EXIT IMMEDIATELY 
            # ==========================================
            total_time = time.time() - total_start_time
            print("\n" + "╔" + "═" * W + "╗")
            print("║" + " " * W + "║")
            print("║" + "  🎯🎯🎯  MTF DIP FOUND — BOT EXITING  🎯🎯🎯".ljust(W) + "║")
            print("║" + " " * W + "║")
            print("║" + f"  Symbol: {sym}".ljust(W) + "║")
            print("║" + f"  Verdict: {verdict}".ljust(W)[:W] + "║")
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
            
            if trader: del trader
            if trading_pairs: del trading_pairs
            if filtered1: del filtered1
            if filtered2: del filtered2
            if filtered3: del filtered3
            if results_1m: del results_1m
            if tf_volumes: del tf_volumes
            if klines_1m: del klines_1m
            if sr: del sr
            
            gc.collect()
            time.sleep(RESCAN_INTERVAL_SECONDS)
            continue


if __name__ == "__main__":
    main()