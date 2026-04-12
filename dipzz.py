from binance.client import Client
import numpy as np
import talib as ta
import time
import sys
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


# ==========================================
# RATE LIMITER & TRADER (unchanged)
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

    def get_klines(self, symbol: str, interval: str, limit: int = 500, return_raw: bool = False):
        self.rate_limiter.acquire()
        for attempt in range(3):
            try:
                klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
                if return_raw:
                    return klines
                close = [float(k[4]) for k in klines]
                return close
            except Exception as e:
                if 'rate limit' in str(e).lower():
                    time.sleep(2 ** attempt * 2)
                else:
                    time.sleep(0.5)
        return []


# ==========================================
# CANDIDATE FILTER INDICATORS (unchanged)
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
    recent_klines = raw_klines[-window:]
    bull_vol, bear_vol = 0.0, 0.0
    for k in recent_klines:
        o, c, v = float(k[1]), float(k[4]), float(k[5])
        if c > o:
            bull_vol += v
        elif c < o:
            bear_vol += v
    total_dir_vol = bull_vol + bear_vol
    if total_dir_vol == 0:
        return False, 0.0
    bull_ratio = bull_vol / total_dir_vol
    return bull_ratio > 0.65, bull_ratio


def calculate_effort_result_metrics(close: List[float], volumes: List[float], window: int = 20) -> Dict:
    if len(close) < window + 2:
        return {"R": 0, "C": 0, "E": 0}
    close_arr = np.array(close[-window:], dtype='float64')
    vol_arr   = np.array(volumes[-window:], dtype='float64')
    delta_p   = abs(close_arr[-1] - close_arr[0])
    total_vol = np.sum(vol_arr)
    eps = 1e-9
    R = total_vol / (delta_p + eps)
    C = total_vol / (np.std(close_arr) + eps)
    E = total_vol / ((delta_p * window) + eps)
    return {"R": R, "C": C, "E": E}


def ml_spike_probability(R: float, C: float, E: float,
                         bull_ratio: float, cmo: float, vratio: float) -> float:
    Rn = np.log1p(R)
    Cn = np.log1p(C)
    En = np.log1p(E)
    score = (
        0.30 * Rn +
        0.25 * Cn +
        0.20 * En +
        0.15 * bull_ratio +
        0.05 * (-cmo / 100.0) +
        0.05 * min(vratio / 5.0, 1.0)
    )
    return 1 / (1 + np.exp(-score))


# ==========================================
# NEW: 1m S/R ZONE ENGINE
# ==========================================

def find_swing_points(
    highs: np.ndarray,
    lows:  np.ndarray,
    lookback: int = 5
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Detect swing highs (argmax) and swing lows (argmin) over a rolling window.
    A bar is a swing high if its HIGH equals the max of the surrounding window.
    A bar is a swing low  if its LOW  equals the min of the surrounding window.
    Returns lists of (index, price).
    """
    n = len(highs)
    swing_highs: List[Tuple[int, float]] = []
    swing_lows:  List[Tuple[int, float]] = []

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback: i + lookback + 1]
        window_l = lows [i - lookback: i + lookback + 1]

        if highs[i] == np.max(window_h):
            swing_highs.append((i, highs[i]))

        if lows[i] == np.min(window_l):
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def cluster_price_levels(
    points: List[Tuple[int, float]],
    cluster_pct: float = 0.003
) -> List[Dict]:
    """
    Merge nearby price levels within cluster_pct of each other.
    Returns list of {price, touches, indices}.
    """
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda x: x[1])
    clusters: List[List[Tuple[int, float]]] = [[sorted_pts[0]]]

    for pt in sorted_pts[1:]:
        ref = np.mean([p[1] for p in clusters[-1]])
        if abs(pt[1] - ref) / ref <= cluster_pct:
            clusters[-1].append(pt)
        else:
            clusters.append([pt])

    result = []
    for cl in clusters:
        result.append({
            'price':   float(np.mean([p[1] for p in cl])),
            'touches': len(cl),
            'indices': [p[0] for p in cl],
        })
    return result


def hilo_range_significance(
    highs: np.ndarray,
    lows:  np.ndarray,
    volumes: np.ndarray,
    window: int = 20
) -> np.ndarray:
    """
    Per-bar significance = (bar_range / avg_range) * (bar_vol / avg_vol).
    Captures wide-range + high-volume bars → structurally important.
    """
    n = len(highs)
    ranges = highs - lows
    scores = np.ones(n)
    for i in range(window, n):
        r_win = ranges [i - window: i + 1]
        v_win = volumes[i - window: i + 1]
        r_norm = ranges [i] / (np.mean(r_win)  + 1e-12)
        v_norm = volumes[i] / (np.mean(v_win)  + 1e-12)
        scores[i] = r_norm * v_norm
    return scores


def volume_pressure_at_zone(
    zone_price: float,
    highs: np.ndarray, lows: np.ndarray,
    closes: np.ndarray, opens: np.ndarray,
    volumes: np.ndarray,
    tol_pct: float = 0.005
) -> Dict:
    """
    Bull vs bear volume on candles that touched zone_price ± tol_pct.
    """
    bull_vol = bear_vol = 0.0
    interactions = 0
    lo_bound = zone_price * (1 - tol_pct)
    hi_bound = zone_price * (1 + tol_pct)

    for i in range(len(closes)):
        if lows[i] <= hi_bound and highs[i] >= lo_bound:
            interactions += 1
            if closes[i] >= opens[i]:
                bull_vol += volumes[i]
            else:
                bear_vol += volumes[i]

    total = bull_vol + bear_vol
    bull_pct = bull_vol / total if total > 0 else 0.5
    if bull_pct > 0.55:
        z_type = 'SUPPORT'
    elif bull_pct < 0.45:
        z_type = 'RESISTANCE'
    else:
        z_type = 'NEUTRAL'

    return {
        'bull_vol': bull_vol, 'bear_vol': bear_vol,
        'bull_pct': bull_pct, 'interactions': interactions,
        'zone_type': z_type,
    }


def score_support_zone(
    zone: Dict,
    total_candles: int,
    vp: Dict,
    sig_scores: np.ndarray,
) -> float:
    """
    Score a support zone purely by structural bullish volume significance.
    No distance cap — the market decides what matters, not an arbitrary %.

    Weights
    -------
    Bullish vol pressure  : 40%  — how dominantly bull vol defended this level
    Hi-Lo range sig       : 25%  — wide-range + high-vol bars at the pivot = real reaction
    Touch count           : 20%  — repeated holds = validated floor
    Recency               : 15%  — fresher pivots are more actionable
    """
    touches   = zone['touches']
    bull_pct  = vp.get('bull_pct', 0.5)

    # Bull pressure: 0.5 = neutral, 1.0 = all bull vol — map to [0,1]
    pressure_sc = max(0.0, (bull_pct - 0.5) * 2.0)

    most_recent = max(zone['indices']) if zone['indices'] else 0
    recency_sc  = most_recent / max(total_candles - 1, 1)

    touch_sc    = min(touches / 4.0, 1.0)

    valid_idx   = [i for i in zone['indices'] if i < len(sig_scores)]
    # Raw sig scores can be > 1; cap at 3× average → normalise to [0,1]
    raw_sig     = float(np.mean([sig_scores[i] for i in valid_idx])) if valid_idx else 1.0
    vol_sig_sc  = min(raw_sig / 3.0, 1.0)

    return (0.40 * pressure_sc +
            0.25 * vol_sig_sc  +
            0.20 * touch_sc    +
            0.15 * recency_sc)


def score_resistance_zone(
    zone: Dict,
    total_candles: int,
    vp: Dict,
    sig_scores: np.ndarray,
) -> float:
    """
    Score a resistance zone purely by structural bearish volume significance.
    No distance cap — the market decides what matters, not an arbitrary %.

    Weights
    -------
    Bearish vol pressure  : 40%  — how dominantly bear vol rejected this level
    Hi-Lo range sig       : 25%  — wide-range + high-vol bars at the pivot = real rejection
    Touch count           : 20%  — repeated failures = validated ceiling
    Recency               : 15%  — fresher pivots are more actionable
    """
    touches  = zone['touches']
    bull_pct = vp.get('bull_pct', 0.5)

    # Bear pressure: 0.5 = neutral, 0.0 = all bull — invert and map to [0,1]
    pressure_sc = max(0.0, (0.5 - bull_pct) * 2.0)

    most_recent = max(zone['indices']) if zone['indices'] else 0
    recency_sc  = most_recent / max(total_candles - 1, 1)

    touch_sc    = min(touches / 4.0, 1.0)

    valid_idx   = [i for i in zone['indices'] if i < len(sig_scores)]
    raw_sig     = float(np.mean([sig_scores[i] for i in valid_idx])) if valid_idx else 1.0
    vol_sig_sc  = min(raw_sig / 3.0, 1.0)

    return (0.40 * pressure_sc +
            0.25 * vol_sig_sc  +
            0.20 * touch_sc    +
            0.15 * recency_sc)


def estimate_eta(dist_pct: float, avg_1m_range_pct: float, vol_bias: float) -> str:
    """
    Rough minutes-to-target based on average 1m candle range and direction bias.
    """
    if avg_1m_range_pct <= 0:
        return "N/A"
    bias = vol_bias if vol_bias >= 0.5 else (1.0 - vol_bias)
    effective_range = avg_1m_range_pct * (0.5 + 0.5 * bias)
    candles = abs(dist_pct) / max(effective_range, 0.0001)
    if   candles <   5: return "~1-5 min"
    elif candles <  15: return "~5-15 min"
    elif candles <  30: return "~15-30 min"
    elif candles <  60: return "~30-60 min"
    elif candles < 120: return "~1-2 hrs"
    elif candles < 240: return "~2-4 hrs"
    else:               return "~4+ hrs"


def get_sr_targets(
    raw_klines: list,
    current_price: float,
    swing_lookback: int = 5,
    n_targets: int = 4,
) -> Dict:
    """
    1m S/R target engine — range determined entirely by volume significance.

    No fixed distance filter is applied. Every swing high above and every swing
    low below current price is evaluated. Levels surface purely because the market
    defended them (bullish vol for support) or rejected them (bearish vol for
    resistance) with structurally significant hi-lo range and volume. The n_targets
    with the highest directional vol significance are returned per side, sorted
    nearest-first for readability (distance is display-only, not a scoring input).

    Steps
    -----
    1. Build OHLCV arrays.
    2. Detect swing highs (argmax) / swing lows (argmin) via rolling window.
    3. Cluster nearby pivots — same price tested multiple times = stronger zone.
    4. Compute per-bar hi-lo × volume significance scores.
    5. For each zone measure directional volume pressure at the exact level.
    6. Score supports by BULLISH vol dominance; score resistance by BEARISH vol
       dominance — completely independent, no shared proximity term.
    7. Return top-N per side ranked by vol significance score, then price-sorted
       (nearest first) for display.
    """
    if len(raw_klines) < 30:
        return {'targets_up': [], 'targets_down': [], 'vol_bias': 0.5,
                'avg_1m_range_pct': 0.0}

    highs   = np.array([float(k[2]) for k in raw_klines])
    lows    = np.array([float(k[3]) for k in raw_klines])
    closes  = np.array([float(k[4]) for k in raw_klines])
    opens   = np.array([float(k[1]) for k in raw_klines])
    volumes = np.array([float(k[5]) for k in raw_klines])
    n       = len(closes)

    # --- Average 1m candle range (%) — used for ETA only ---
    candle_ranges    = (highs - lows) / (closes + 1e-12) * 100.0
    avg_1m_range_pct = float(np.mean(candle_ranges[-50:]))

    # --- Recent volume bias (last 20 candles) ---
    rec    = raw_klines[-20:]
    bull_v = sum(float(k[5]) for k in rec if float(k[4]) >= float(k[1]))
    bear_v = sum(float(k[5]) for k in rec if float(k[4]) <  float(k[1]))
    tot_v  = bull_v + bear_v
    vol_bias = bull_v / tot_v if tot_v > 0 else 0.5

    # --- Swing detection via rolling argmax / argmin ---
    swing_highs, swing_lows = find_swing_points(highs, lows, lookback=swing_lookback)

    # --- Cluster adjacent pivots into zones ---
    res_zones = cluster_price_levels(swing_highs, cluster_pct=0.003)
    sup_zones = cluster_price_levels(swing_lows,  cluster_pct=0.003)

    # --- Per-bar hi-lo × volume significance ---
    sig_scores = hilo_range_significance(highs, lows, volumes, window=20)

    # ----------------------------------------------------------------
    # Score resistance zones: ranked by BEARISH volume dominance
    # No distance gate — every swing high above price is evaluated
    # ----------------------------------------------------------------
    targets_up: List[Dict] = []
    for zone in res_zones:
        if zone['price'] <= current_price * 1.001:
            continue                              # must be above current price
        dist = (zone['price'] - current_price) / current_price * 100.0
        vp   = volume_pressure_at_zone(
            zone['price'], highs, lows, closes, opens, volumes)
        score = score_resistance_zone(zone, n, vp, sig_scores)
        eta   = estimate_eta(dist, avg_1m_range_pct, vol_bias)
        targets_up.append({
            'price': zone['price'], 'score': score, 'dist_pct': dist,
            'touches': zone['touches'], 'zone_type': vp['zone_type'],
            'bull_pct': vp['bull_pct'], 'interactions': vp['interactions'],
            'eta': eta,
        })

    # ----------------------------------------------------------------
    # Score support zones: ranked by BULLISH volume dominance
    # No distance gate — every swing low below price is evaluated
    # ----------------------------------------------------------------
    targets_down: List[Dict] = []
    for zone in sup_zones:
        if zone['price'] >= current_price * 0.999:
            continue                              # must be below current price
        dist = (zone['price'] - current_price) / current_price * 100.0  # negative
        vp   = volume_pressure_at_zone(
            zone['price'], highs, lows, closes, opens, volumes)
        score = score_support_zone(zone, n, vp, sig_scores)
        eta   = estimate_eta(dist, avg_1m_range_pct, vol_bias)
        targets_down.append({
            'price': zone['price'], 'score': score, 'dist_pct': dist,
            'touches': zone['touches'], 'zone_type': vp['zone_type'],
            'bull_pct': vp['bull_pct'], 'interactions': vp['interactions'],
            'eta': eta,
        })

    # --- Select top-N by vol significance score, then sort nearest-first ---
    targets_up.sort(  key=lambda t: t['score'], reverse=True)
    targets_down.sort(key=lambda t: t['score'], reverse=True)

    top_up   = sorted(targets_up[:n_targets],   key=lambda t:  t['dist_pct'])
    top_down = sorted(targets_down[:n_targets],  key=lambda t: -t['dist_pct'])

    return {
        'targets_up':       top_up,
        'targets_down':     top_down,
        'vol_bias':         vol_bias,
        'bull_vol':         bull_v,
        'bear_vol':         bear_v,
        'avg_1m_range_pct': avg_1m_range_pct,
    }


# ==========================================
# CONCURRENT FILTER FUNCTIONS (unchanged)
# ==========================================

def check_tf_dip(trader: Trader, symbol: str, interval: str) -> Tuple[str, bool]:
    close = trader.get_klines(symbol, interval, limit=300)
    return (symbol, linear_regression_dip(close, 0.01))


def check_5m_rejection(trader: Trader, symbol: str) -> Tuple[str, bool, float]:
    klines = trader.get_klines(symbol, '5m', limit=300, return_raw=True)
    if not klines:
        return (symbol, False, 0.0)
    close = [float(k[4]) for k in klines]
    if linear_regression_dip(close, 0.01):
        is_rejection, ratio = has_bullish_rejection_volume(klines, window=10)
        return (symbol, is_rejection, ratio)
    return (symbol, False, 0.0)


def check_1m_final(trader: Trader, symbol: str) -> Tuple[str, float, float, bool, float, float]:
    klines = trader.get_klines(symbol, '1m', limit=200, return_raw=True)
    if not klines or len(klines) < 50:
        return (symbol, 0.0, 0.0, False, 0.0, 0.0)

    close   = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    cmo     = ta.CMO(np.asarray(close), timeperiod=14)
    cmo_val = cmo[-1] if not np.isnan(cmo[-1]) else 0.0

    avg_vol = np.mean(volumes[-51:-1]) if len(volumes) > 51 else np.mean(volumes[:-1])
    vratio  = volumes[-1] / avg_vol if avg_vol > 0 else 0.0

    is_rejection, bull_ratio = has_bullish_rejection_volume(klines, window=10)
    metrics = calculate_effort_result_metrics(close, volumes, window=20)
    prob    = ml_spike_probability(
        metrics["R"], metrics["C"], metrics["E"],
        bull_ratio, cmo_val, vratio
    )

    is_strong = (
        (cmo_val < -50) and
        is_rejection and
        (metrics["C"] > np.mean(volumes[-50:])) and
        (prob > 0.65)
    )
    return (symbol, cmo_val, vratio, is_strong, bull_ratio, prob)


class ProgressTracker:
    def __init__(self, total: int, label: str):
        self.total, self.label = total, label
        self.completed = self.passed = 0
        self.lock = Lock()
        self.start_time = time.time()

    def update(self, passed: bool = False):
        with self.lock:
            self.completed += 1
            if passed:
                self.passed += 1

    def get_stats(self) -> str:
        with self.lock:
            elapsed   = time.time() - self.start_time
            rate      = self.completed / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.completed) / rate if rate > 0 else 0
            return (f"\r{self.label}: {self.completed}/{self.total} | "
                    f"✓{self.passed} | {rate:.1f}/s | ETA: {remaining:.0f}s")


def run_tf_filter_concurrent(trader: Trader, symbols: List[str],
                              interval: str, max_workers: int = 20) -> List[str]:
    passed = []
    tracker = ProgressTracker(len(symbols), f"{interval} filter")
    print(f"Running {interval} filter on {len(symbols)} pairs...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(check_tf_dip, trader, s, interval): s for s in symbols}
        for f in as_completed(futures):
            try:
                sym, is_dip = f.result()
                if is_dip:
                    passed.append(sym)
                tracker.update(passed=is_dip)
                print(tracker.get_stats(), end="", flush=True)
            except:
                tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20)
    return passed


def run_5m_filter_concurrent(trader: Trader, symbols: List[str],
                              max_workers: int = 15) -> Tuple[List[str], float]:
    passed, best_ratio = [], 0.0
    tracker = ProgressTracker(len(symbols), "5m+Rej filter")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(check_5m_rejection, trader, s): s for s in symbols}
        for f in as_completed(futures):
            try:
                sym, ok, ratio = f.result()
                if ok:
                    passed.append(sym)
                    if ratio > best_ratio:
                        best_ratio = ratio
                tracker.update(passed=ok)
                print(tracker.get_stats(), end="", flush=True)
            except:
                tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20)
    return passed, best_ratio


def run_1m_filter_concurrent(trader: Trader, symbols: List[str],
                              max_workers: int = 15):
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

def format_sr_output(symbol: str, sr: Dict, current_price: float,
                     cmo_val: float, vratio: float,
                     bull_ratio: float, ml_prob: float):
    vol_bias  = sr['vol_bias']
    avg_range = sr['avg_1m_range_pct']
    bull_pct_overall = vol_bias * 100
    bias_label = "🟢 BULLISH" if vol_bias > 0.55 else ("🔴 BEARISH" if vol_bias < 0.45 else "⚪ NEUTRAL")

    print("\n" + "=" * 62)
    print(f"  ★  1m S/R TARGET MAP  —  {symbol}  ★")
    print(f"  (range determined by vol-significant rejection points)")
    print("=" * 62)
    print(f"  Entry Price    : {current_price:.10f}")
    print(f"  1m CMO         : {cmo_val:+.2f}  (< -50 = oversold)")
    print(f"  Vol Ratio      : x{vratio:.2f}")
    print(f"  Bull Rej Vol   : {bull_ratio*100:.1f}%  (>65% = confirmed)")
    print(f"  ML Spike Prob  : {ml_prob*100:.1f}%")
    print(f"  Vol Pressure   : {bias_label}  ({bull_pct_overall:.1f}% bull / "
          f"{100-bull_pct_overall:.1f}% bear, last 20 candles)")
    print(f"  Avg 1m Range   : {avg_range:.4f}%")
    print("-" * 62)

    # ---- RESISTANCE / TARGETS UP ----
    up = sr['targets_up']
    if up:
        print(f"\n  📈  RESISTANCE TARGETS  (nearest → furthest)\n")
        for i, t in enumerate(up, 1):
            bar  = "█" * int(t['score'] * 10) + "░" * (10 - int(t['score'] * 10))
            sign = "🔴" if t['zone_type'] == 'RESISTANCE' else ("🟢" if t['zone_type'] == 'SUPPORT' else "⚪")
            print(f"  T{i}  {t['price']:.10f}  ({t['dist_pct']:+.3f}%)  ETA: {t['eta']}")
            print(f"       Score [{bar}] {t['score']:.2f}  "
                  f"Touches: {t['touches']}  {sign} {t['zone_type']}  "
                  f"BullVol@zone: {t['bull_pct']*100:.0f}%  "
                  f"Interactions: {t['interactions']}\n")
    else:
        print("\n  📈  No resistance targets within 5% range.\n")

    # ---- SUPPORT / TARGETS DOWN ----
    dn = sr['targets_down']
    if dn:
        print(f"  📉  SUPPORT LEVELS  (nearest → furthest)\n")
        for i, t in enumerate(dn, 1):
            bar  = "█" * int(t['score'] * 10) + "░" * (10 - int(t['score'] * 10))
            sign = "🟢" if t['zone_type'] == 'SUPPORT' else ("🔴" if t['zone_type'] == 'RESISTANCE' else "⚪")
            print(f"  S{i}  {t['price']:.10f}  ({t['dist_pct']:+.3f}%)  ETA: {t['eta']}")
            print(f"       Score [{bar}] {t['score']:.2f}  "
                  f"Touches: {t['touches']}  {sign} {t['zone_type']}  "
                  f"BullVol@zone: {t['bull_pct']*100:.0f}%  "
                  f"Interactions: {t['interactions']}\n")
    else:
        print("  📉  No support levels within 5% range.\n")

    # ---- IMMEDIATE TRADE BIAS ----
    print("-" * 62)
    print("  ⚡  TRADE BIAS SUMMARY")
    if up and dn:
        nearest_up   = up[0]
        nearest_dn   = dn[0]
        rr           = abs(nearest_up['dist_pct']) / max(abs(nearest_dn['dist_pct']), 0.0001)
        bias_ok      = vol_bias > 0.55 and bull_ratio > 0.55
        print(f"  Nearest target : {nearest_up['price']:.10f}  ({nearest_up['dist_pct']:+.3f}%)"
              f"  ETA {nearest_up['eta']}")
        print(f"  Nearest stop   : {nearest_dn['price']:.10f}  ({nearest_dn['dist_pct']:+.3f}%)")
        print(f"  R:R (target/stop) : {rr:.2f}x")
        if bias_ok and rr >= 1.5:
            print("  Signal         : ✅  LONG — volume + rejection + R:R confirm")
        elif rr < 1.0:
            print("  Signal         : ⚠️   SKIP — R:R unfavorable")
        else:
            print("  Signal         : ⏳  WAIT — bias not fully confirmed")
    elif up:
        print(f"  Target only: {up[0]['price']:.10f} ({up[0]['dist_pct']:+.3f}%)  ETA {up[0]['eta']}")
    elif dn:
        print(f"  Support only: {dn[0]['price']:.10f} ({dn[0]['dist_pct']:+.3f}%)  ETA {dn[0]['eta']}")
    else:
        print("  No clear levels found in range.")
    print("=" * 62 + "\n")


# ==========================================
# MAIN
# ==========================================

def main():
    start_time = time.time()
    trader = Trader('credentials.txt')
    trading_pairs = trader.get_usdc_pairs()

    print("=" * 62)
    print("  MTF SCANNER  +  1m S/R ZONE TARGET ENGINE")
    print("=" * 62 + "\n")

    # ---- Multi-TF candidate filtering (unchanged) ----
    filtered1 = run_tf_filter_concurrent(trader, trading_pairs, '2h', 20)
    if not filtered1:
        print("No 2h dips. Exiting.")
        sys.exit(0)

    filtered2 = run_tf_filter_concurrent(trader, filtered1, '15m', 15)
    if not filtered2:
        print("No 15m dips. Exiting.")
        sys.exit(0)

    filtered3, best_5m_ratio = run_5m_filter_concurrent(trader, filtered2, 15)
    if not filtered3:
        print("\nNo 5m dips with Bullish Rejection Volume. Exiting.")
        sys.exit(0)

    results_1m = run_1m_filter_concurrent(trader, filtered3, 15)

    # ---- Select best candidate ----
    strong = [r for r in results_1m if r[3] is True]
    if strong:
        final = max(strong, key=lambda x: (x[5], -x[1]))
        mode  = "STRONG + ML ENERGY CONFIRMATION"
    elif results_1m:
        final = min(results_1m, key=lambda x: x[1])
        mode  = "FALLBACK (Best CMO)"
    else:
        print("\nFailed to fetch 1m data. Exiting.")
        sys.exit(0)

    sym, cmo_val, vratio, _, live_bull_ratio, ml_prob = final

    print("\n" + "-" * 62)
    print(f"  SELECTED SYMBOL : {sym}")
    print(f"  SELECTION MODE  : {mode}")
    print(f"  1m CMO          : {cmo_val:.4f}")
    print(f"  1m Volume Ratio : x{vratio:.4f}")
    print(f"  Bull Rej Vol    : {live_bull_ratio*100:.2f}%")
    print(f"  ML Spike Prob   : {ml_prob*100:.2f}%")
    print("-" * 62)

    # ---- Fetch 200 1m raw klines for S/R engine ----
    klines_1m = trader.get_klines(sym, '1m', limit=200, return_raw=True)
    if not klines_1m:
        print("Could not fetch 1m klines. Exiting.")
        sys.exit(0)

    current_price = float(klines_1m[-1][4])

    # ---- Run S/R target engine ----
    sr = get_sr_targets(
        raw_klines     = klines_1m,
        current_price  = current_price,
        swing_lookback = 5,   # bars each side for argmin/argmax
        n_targets      = 4,   # top N zones per direction by vol significance
    )

    # ---- Print results ----
    format_sr_output(sym, sr, current_price,
                     cmo_val, vratio, live_bull_ratio, ml_prob)

    print(f"Total Execution Time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
