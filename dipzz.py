from binance.client import Client
import numpy as np
import talib as ta
import time
import sys
from typing import List, Tuple, Optional, Dict
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
# FFT CYCLE ANALYSIS ENGINE
# ==========================================

def fft_cycle_analysis(
    close: np.ndarray,
    min_period: int = 20,
    max_period: int = 250
) -> Dict:
    """
    Perform FFT cycle analysis on price data to detect dominant cycles.
    
    Uses proper FFT with Hanning window to reduce spectral leakage.
    Detects cycles in the 20-250 bar range (20min to ~4hrs on 1m).
    
    Returns:
        - dominant_cycles: List of cycle dicts with period, amplitude, phase
        - trend_slope/intercept: Linear trend for detrending
        - cycle_strength: Overall cycle signal strength (0-1)
    """
    n = len(close)
    if n < min_period * 2:
        return {
            'dominant_cycles': [],
            'trend_slope': 0.0,
            'trend_intercept': float(close[-1]) if len(close) > 0 else 0.0,
            'cycle_strength': 0.0,
            'detrended': close,
            'spectral_entropy': 0.0,
        }
    
    # Detrend: remove linear regression
    x = np.arange(n, dtype='float64')
    slope, intercept = np.polyfit(x, close, 1)
    trend = slope * x + intercept
    detrended = close - trend
    
    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(n)
    windowed = detrended * window
    
    # Compute window correction factor
    window_sum = np.sum(window)
    window_sq_sum = np.sum(window ** 2)
    amplitude_correction = window_sum / window_sq_sum
    
    # FFT
    fft_vals = np.fft.rfft(windowed)
    n_freq = len(fft_vals)
    
    # Power spectrum (corrected for windowing)
    power = (np.abs(fft_vals) ** 2 / n) * (amplitude_correction ** 2)
    
    # Frequency bins
    freqs = np.fft.rfftfreq(n)
    
    # Convert frequency to period (in bars)
    with np.errstate(divide='ignore', invalid='ignore'):
        periods = np.where(freqs > 1e-10, 1.0 / freqs, np.inf)
    
    # Filter to meaningful cycle range
    valid_mask = (periods >= min_period) & (periods <= max_period)
    valid_periods = periods[valid_mask]
    valid_power = power[valid_mask]
    valid_fft = fft_vals[valid_mask]
    
    if len(valid_power) == 0:
        return {
            'dominant_cycles': [],
            'trend_slope': float(slope),
            'trend_intercept': float(intercept),
            'cycle_strength': 0.0,
            'detrended': detrended,
            'spectral_entropy': 0.0,
        }
    
    # Find peaks in power spectrum using prominence
    mean_power = np.mean(valid_power)
    std_power = np.std(valid_power)
    threshold = mean_power + 1.5 * std_power
    
    # Simple peak detection
    peak_indices = []
    for i in range(1, len(valid_power) - 1):
        if (valid_power[i] > valid_power[i-1] and 
            valid_power[i] > valid_power[i+1] and
            valid_power[i] > threshold):
            peak_indices.append(i)
    
    if not peak_indices:
        # Fallback: take highest power point
        peak_indices = [np.argmax(valid_power)]
    
    # Sort by power, take top cycles
    peak_indices.sort(key=lambda i: valid_power[i], reverse=True)
    peak_indices = peak_indices[:5]
    
    # Extract cycle parameters
    dominant_cycles = []
    for idx in peak_indices:
        period = valid_periods[idx]
        raw_amplitude = np.abs(valid_fft[idx]) * 2.0 * amplitude_correction / n
        
        # Phase: angle of complex FFT coefficient
        phase = np.angle(valid_fft[idx])
        
        # Convert phase to bars from END of data (where next peak occurs)
        # phase = 0 means peak at start, phase = -π means peak at middle
        # We want bars until next peak from the end of our data
        phase_fraction = -phase / (2 * np.pi)  # Fraction of cycle completed
        if phase_fraction < 0:
            phase_fraction += 1.0
        
        bars_to_next_peak = (1.0 - phase_fraction) * period
        if bars_to_next_peak < 1:
            bars_to_next_peak += period
        
        bars_to_next_trough = bars_to_next_peak + period / 2
        
        dominant_cycles.append({
            'period': float(period),
            'amplitude': float(raw_amplitude),
            'power': float(valid_power[idx]),
            'bars_to_peak': float(bars_to_next_peak),
            'bars_to_trough': float(bars_to_next_trough),
            'relative_power': float(valid_power[idx] / np.max(valid_power)),
        })
    
    # Sort by period (shortest first)
    dominant_cycles.sort(key=lambda c: c['period'])
    
    # Calculate spectral entropy (lower = more structured/cyclical)
    total_power = np.sum(valid_power)
    if total_power > 0:
        probs = valid_power / total_power
        probs = probs[probs > 0]
        spectral_entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(probs)) if len(probs) > 1 else 1.0
        normalized_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 1.0
    else:
        normalized_entropy = 1.0
    
    # Cycle strength: ratio of dominant cycle power to total power
    if dominant_cycles:
        cycle_strength = sum(c['relative_power'] for c in dominant_cycles[:3]) / 3.0
    else:
        cycle_strength = 0.0
    
    return {
        'dominant_cycles': dominant_cycles,
        'trend_slope': float(slope),
        'trend_intercept': float(intercept),
        'cycle_strength': float(cycle_strength),
        'detrended': detrended,
        'spectral_entropy': float(normalized_entropy),
    }


def fft_project_targets(
    close: np.ndarray,
    fft_result: Dict,
    current_price: float,
    n_targets: int = 5
) -> Tuple[List[Dict], List[Dict]]:
    """
    Project realistic price targets based on FFT cycle analysis.
    
    Uses dominant cycle amplitudes to estimate where price might reach
    at future peaks (resistance) and troughs (support).
    
    Returns:
        - projected_up: List of resistance target dicts
        - projected_down: List of support target dicts
    """
    projected_up = []
    projected_down = []
    
    if not fft_result['dominant_cycles']:
        return projected_up, projected_down
    
    n = len(close)
    slope = fft_result['trend_slope']
    intercept = fft_result['trend_intercept']
    
    seen_prices_up = set()
    seen_prices_down = set()
    
    for cycle in fft_result['dominant_cycles']:
        period = cycle['period']
        amplitude = cycle['amplitude']
        bars_to_peak = cycle['bars_to_peak']
        bars_to_trough = cycle['bars_to_trough']
        
        # Project price at next peak (resistance)
        future_idx_peak = n + bars_to_peak
        trend_at_peak = slope * future_idx_peak + intercept
        peak_price = trend_at_peak + amplitude
        
        # Project price at next trough (support)
        future_idx_trough = n + bars_to_trough
        trend_at_trough = slope * future_idx_trough + intercept
        trough_price = trend_at_trough - amplitude
        
        # Round to avoid near-duplicates
        peak_key = round(peak_price, 8)
        trough_key = round(trough_price, 8)
        
        if peak_price > current_price * 1.0001 and peak_key not in seen_prices_up:
            seen_prices_up.add(peak_key)
            projected_up.append({
                'price': peak_price,
                'dist_pct': (peak_price - current_price) / current_price * 100,
                'bars_ahead': bars_to_peak,
                'eta_minutes': bars_to_peak,  # 1m candles
                'cycle_period': period,
                'amplitude_pct': amplitude / current_price * 100,
                'source': 'fft',
                'confidence': cycle['relative_power'],
            })
        
        if trough_price < current_price * 0.9999 and trough_key not in seen_prices_down:
            seen_prices_down.add(trough_key)
            projected_down.append({
                'price': trough_price,
                'dist_pct': (trough_price - current_price) / current_price * 100,
                'bars_ahead': bars_to_trough,
                'eta_minutes': bars_to_trough,
                'cycle_period': period,
                'amplitude_pct': amplitude / current_price * 100,
                'source': 'fft',
                'confidence': cycle['relative_power'],
            })
    
    # Sort by distance (nearest first)
    projected_up.sort(key=lambda p: p['dist_pct'])
    projected_down.sort(key=lambda p: p['dist_pct'], reverse=True)
    
    return projected_up[:n_targets], projected_down[:n_targets]


# ==========================================
# ARGMIN/ARGMAX EXTREME DETECTION
# ==========================================

def find_significant_extremes(
    close: np.ndarray,
    lookback: int = 500
) -> Dict:
    """
    Find the absolute argmin and argmax positions in the last N values.
    
    These represent the most significant swing points:
    - argmax: highest price reached → potential resistance origin
    - argmin: lowest price reached → potential support origin
    
    Also finds local argmin/argmax using rolling windows for
    intermediate swing points.
    """
    if len(close) < 10:
        return {
            'global_argmax': {'idx': 0, 'price': 0, 'age': 0},
            'global_argmin': {'idx': 0, 'price': 0, 'age': 0},
            'local_maxima': [],
            'local_minima': [],
            'range_pct': 0,
        }
    
    # Use last N values
    data = close[-lookback:] if len(close) >= lookback else close
    n = len(data)
    
    # Global extremes (absolute argmin/argmax)
    argmax_idx = int(np.argmax(data))
    argmin_idx = int(np.argmin(data))
    
    # Local extrema using rolling window argmin/argmax
    window = 10  # 5 bars each side
    local_maxima = []
    local_minima = []
    
    for i in range(window, n - window):
        segment = data[i - window: i + window + 1]
        
        if data[i] == np.max(segment):
            local_maxima.append({'idx': i, 'price': float(data[i])})
        
        if data[i] == np.min(segment):
            local_minima.append({'idx': i, 'price': float(data[i])})
    
    # Calculate range
    high = data[argmax_idx]
    low = data[argmin_idx]
    range_pct = (high - low) / low * 100 if low > 0 else 0
    
    return {
        'global_argmax': {
            'idx': argmax_idx,
            'price': float(high),
            'age': n - argmax_idx,  # How many bars ago
        },
        'global_argmin': {
            'idx': argmin_idx,
            'price': float(low),
            'age': n - argmin_idx,
        },
        'local_maxima': local_maxima,
        'local_minima': local_minima,
        'range_pct': range_pct,
        'high': float(high),
        'low': float(low),
    }


# ==========================================
# IMPROVED S/R ZONE ENGINE WITH FFT VALIDATION
# ==========================================

def find_swing_points(
    highs: np.ndarray,
    lows: np.ndarray,
    close: np.ndarray,
    lookback: int = 5,
    min_amplitude_pct: float = 0.0005
) -> Tuple[List[Tuple[int, float, float]], List[Tuple[int, float, float]]]:
    """
    Detect swing highs and lows using rolling argmax/argmin.
    
    A bar is a swing high if its HIGH equals the max of surrounding window.
    A bar is a swing low if its LOW equals the min of surrounding window.
    
    Additionally validates against close prices and filters by minimum amplitude.
    
    Returns lists of (index, price, amplitude_pct).
    """
    n = len(highs)
    swing_highs: List[Tuple[int, float, float]] = []
    swing_lows: List[Tuple[int, float, float]] = []
    
    for i in range(lookback, n - lookback):
        # Swing High Detection
        window_high = highs[i - lookback: i + lookback + 1]
        window_close = close[i - lookback: i + lookback + 1]
        
        if highs[i] == np.max(window_high) and close[i] == np.max(window_close):
            # Calculate amplitude from preceding trough
            lookback_start = max(0, i - lookback * 3)
            prev_low = np.min(lows[lookback_start:i])
            amplitude = highs[i] - prev_low
            amp_pct = amplitude / highs[i] if highs[i] > 0 else 0
            
            if amp_pct >= min_amplitude_pct:
                swing_highs.append((i, highs[i], amp_pct))
        
        # Swing Low Detection
        window_low = lows[i - lookback: i + lookback + 1]
        
        if lows[i] == np.min(window_low) and close[i] == np.min(window_close):
            # Calculate amplitude from preceding peak
            lookback_start = max(0, i - lookback * 3)
            prev_high = np.max(highs[lookback_start:i])
            amplitude = prev_high - lows[i]
            amp_pct = amplitude / lows[i] if lows[i] > 0 else 0
            
            if amp_pct >= min_amplitude_pct:
                swing_lows.append((i, lows[i], amp_pct))
    
    return swing_highs, swing_lows


def cluster_price_levels(
    points: List[Tuple[int, float, float]],
    cluster_pct: float = 0.002
) -> List[Dict]:
    """
    Merge nearby price levels within cluster_pct of each other.
    Returns list of {price, touches, indices, avg_amplitude}.
    """
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda x: x[1])
    clusters: List[List[Tuple[int, float, float]]] = [[sorted_pts[0]]]

    for pt in sorted_pts[1:]:
        ref = np.mean([p[1] for p in clusters[-1]])
        if abs(pt[1] - ref) / ref <= cluster_pct:
            clusters[-1].append(pt)
        else:
            clusters.append([pt])

    result = []
    for cl in clusters:
        result.append({
            'price': float(np.mean([p[1] for p in cl])),
            'touches': len(cl),
            'indices': [p[0] for p in cl],
            'avg_amplitude': float(np.mean([p[2] for p in cl])),
            'max_amplitude': float(np.max([p[2] for p in cl])),
        })
    return result


def hilo_range_significance(
    highs: np.ndarray,
    lows: np.ndarray,
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
        r_norm = ranges [i] / (np.mean(r_win) + 1e-12)
        v_norm = volumes[i] / (np.mean(v_win) + 1e-12)
        scores[i] = r_norm * v_norm
    return scores


def volume_pressure_at_zone(
    zone_price: float,
    highs: np.ndarray, lows: np.ndarray,
    closes: np.ndarray, opens: np.ndarray,
    volumes: np.ndarray,
    tol_pct: float = 0.004
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


def check_fft_alignment(
    zone_price: float,
    fft_targets: List[Dict],
    tolerance_pct: float = 0.003
) -> Tuple[bool, float, Dict]:
    """
    Check if a zone aligns with any FFT-projected target.
    Returns (is_aligned, alignment_score, best_target).
    """
    best_alignment = 0.0
    best_target = None
    
    for target in fft_targets:
        dist_pct = abs(zone_price - target['price']) / target['price']
        if dist_pct <= tolerance_pct:
            # Perfect alignment = 1.0, degrades linearly to 0 at tolerance boundary
            alignment = 1.0 - (dist_pct / tolerance_pct)
            # Boost by FFT confidence
            weighted_alignment = alignment * (0.5 + 0.5 * target.get('confidence', 0.5))
            if weighted_alignment > best_alignment:
                best_alignment = weighted_alignment
                best_target = target
    
    return (best_alignment > 0.1, best_alignment, best_target)


def check_argmin_argmax_relevance(
    zone_price: float,
    extremes: Dict,
    current_price: float,
) -> Tuple[float, str]:
    """
    Score zone based on relevance to global argmin/argmax.
    
    Zones near the global high/low are structurally significant.
    Returns (relevance_score, reason).
    """
    global_high = extremes['global_argmax']['price']
    global_low = extremes['global_argmin']['price']
    high_age = extremes['global_argmax']['age']
    low_age = extremes['global_argmin']['age']
    
    score = 0.0
    reason = ""
    
    # Check proximity to global high (resistance relevance)
    dist_to_high = abs(zone_price - global_high) / global_high
    if dist_to_high < 0.005 and zone_price > current_price:
        # Very close to the highest point
        recency = max(0, 1.0 - high_age / 500)
        score = 0.8 * recency + 0.2
        reason = f"Near global high ({high_age} bars ago)"
    
    # Check proximity to global low (support relevance)
    dist_to_low = abs(zone_price - global_low) / global_low
    if dist_to_low < 0.005 and zone_price < current_price:
        recency = max(0, 1.0 - low_age / 500)
        score = 0.8 * recency + 0.2
        reason = f"Near global low ({low_age} bars ago)"
    
    return score, reason


def score_support_zone(
    zone: Dict,
    total_candles: int,
    vp: Dict,
    sig_scores: np.ndarray,
    fft_alignment: Tuple[bool, float, Dict],
    argmax_relevance: Tuple[float, str],
) -> Tuple[float, Dict]:
    """
    Score a support zone with FFT and argmin/argmax integration.
    
    Weights:
    - Bullish vol pressure  : 30%
    - Hi-Lo range sig       : 20%
    - Touch count           : 15%
    - Recency               : 10%
    - FFT cycle alignment   : 15%
    - Argmin/argmax relevance: 10%
    """
    touches   = zone['touches']
    bull_pct  = vp.get('bull_pct', 0.5)
    
    # Bull pressure: 0.5 = neutral, 1.0 = all bull vol
    pressure_sc = max(0.0, (bull_pct - 0.5) * 2.0)
    
    most_recent = max(zone['indices']) if zone['indices'] else 0
    recency_sc  = most_recent / max(total_candles - 1, 1)
    
    touch_sc    = min(touches / 4.0, 1.0)
    
    valid_idx   = [i for i in zone['indices'] if i < len(sig_scores)]
    raw_sig     = float(np.mean([sig_scores[i] for i in valid_idx])) if valid_idx else 1.0
    vol_sig_sc  = min(raw_sig / 3.0, 1.0)
    
    # FFT alignment score
    _, fft_score, fft_target = fft_alignment
    fft_sc = fft_score if fft_score > 0 else 0.0
    
    # Argmin/argmax relevance
    arg_sc, arg_reason = argmax_relevance
    
    final_score = (
        0.30 * pressure_sc +
        0.20 * vol_sig_sc  +
        0.15 * touch_sc    +
        0.10 * recency_sc  +
        0.15 * fft_sc      +
        0.10 * arg_sc
    )
    
    details = {
        'pressure': pressure_sc,
        'vol_sig': vol_sig_sc,
        'touch': touch_sc,
        'recency': recency_sc,
        'fft': fft_sc,
        'arg_relevance': arg_sc,
        'arg_reason': arg_reason,
        'fft_target': fft_target,
    }
    
    return final_score, details


def score_resistance_zone(
    zone: Dict,
    total_candles: int,
    vp: Dict,
    sig_scores: np.ndarray,
    fft_alignment: Tuple[bool, float, Dict],
    argmax_relevance: Tuple[float, str],
) -> Tuple[float, Dict]:
    """
    Score a resistance zone with FFT and argmin/argmax integration.
    
    Weights:
    - Bearish vol pressure  : 30%
    - Hi-Lo range sig       : 20%
    - Touch count           : 15%
    - Recency               : 10%
    - FFT cycle alignment   : 15%
    - Argmin/argmax relevance: 10%
    """
    touches  = zone['touches']
    bull_pct = vp.get('bull_pct', 0.5)
    
    # Bear pressure: 0.5 = neutral, 0.0 = all bull → invert
    pressure_sc = max(0.0, (0.5 - bull_pct) * 2.0)
    
    most_recent = max(zone['indices']) if zone['indices'] else 0
    recency_sc  = most_recent / max(total_candles - 1, 1)
    
    touch_sc    = min(touches / 4.0, 1.0)
    
    valid_idx   = [i for i in zone['indices'] if i < len(sig_scores)]
    raw_sig     = float(np.mean([sig_scores[i] for i in valid_idx])) if valid_idx else 1.0
    vol_sig_sc  = min(raw_sig / 3.0, 1.0)
    
    # FFT alignment score
    _, fft_score, fft_target = fft_alignment
    fft_sc = fft_score if fft_score > 0 else 0.0
    
    # Argmin/argmax relevance
    arg_sc, arg_reason = argmax_relevance
    
    final_score = (
        0.30 * pressure_sc +
        0.20 * vol_sig_sc  +
        0.15 * touch_sc    +
        0.10 * recency_sc  +
        0.15 * fft_sc      +
        0.10 * arg_sc
    )
    
    details = {
        'pressure': pressure_sc,
        'vol_sig': vol_sig_sc,
        'touch': touch_sc,
        'recency': recency_sc,
        'fft': fft_sc,
        'arg_relevance': arg_sc,
        'arg_reason': arg_reason,
        'fft_target': fft_target,
    }
    
    return final_score, details


def format_eta_minutes(bars_ahead: float) -> str:
    """Convert bars to human-readable ETA."""
    if bars_ahead < 5:
        return "~1-5 min"
    elif bars_ahead < 15:
        return "~5-15 min"
    elif bars_ahead < 30:
        return "~15-30 min"
    elif bars_ahead < 60:
        return "~30-60 min"
    elif bars_ahead < 120:
        return "~1-2 hrs"
    elif bars_ahead < 240:
        return "~2-4 hrs"
    else:
        return f"~{bars_ahead/60:.1f}+ hrs"


def get_sr_targets(
    raw_klines: list,
    current_price: float,
    swing_lookback: int = 5,
    n_targets: int = 4,
) -> Dict:
    """
    Enhanced 1m S/R target engine with FFT cycle validation and
    argmin/argmax structural significance.
    
    Steps:
    1. Build OHLCV arrays from 500 klines
    2. Run FFT cycle analysis to detect dominant periods
    3. Project FFT-based price targets (realistic cycle-based levels)
    4. Find global argmin/argmax for structural anchoring
    5. Detect swing highs/lows via rolling argmax/argmin
    6. Cluster nearby pivots into zones
    7. Score zones with FFT alignment and argmin/argmax relevance
    8. Return top-N zones per direction, sorted nearest-first
    """
    if len(raw_klines) < 50:
        return {
            'targets_up': [], 'targets_down': [],
            'vol_bias': 0.5, 'avg_1m_range_pct': 0.0,
            'fft_targets_up': [], 'fft_targets_down': [],
            'extremes': {}, 'fft_info': {},
        }
    
    highs   = np.array([float(k[2]) for k in raw_klines])
    lows    = np.array([float(k[3]) for k in raw_klines])
    closes  = np.array([float(k[4]) for k in raw_klines])
    opens   = np.array([float(k[1]) for k in raw_klines])
    volumes = np.array([float(k[5]) for k in raw_klines])
    n       = len(closes)
    
    # --- Average 1m candle range (%) ---
    candle_ranges    = (highs - lows) / (closes + 1e-12) * 100.0
    avg_1m_range_pct = float(np.mean(candle_ranges[-50:]))
    
    # --- Recent volume bias ---
    rec    = raw_klines[-20:]
    bull_v = sum(float(k[5]) for k in rec if float(k[4]) >= float(k[1]))
    bear_v = sum(float(k[5]) for k in rec if float(k[4]) <  float(k[1]))
    tot_v  = bull_v + bear_v
    vol_bias = bull_v / tot_v if tot_v > 0 else 0.5
    
    # ========================================
    # FFT CYCLE ANALYSIS
    # ========================================
    fft_result = fft_cycle_analysis(closes, min_period=15, max_period=250)
    fft_up, fft_down = fft_project_targets(closes, fft_result, current_price, n_targets=8)
    
    fft_info = {
        'dominant_cycles': fft_result['dominant_cycles'],
        'cycle_strength': fft_result['cycle_strength'],
        'spectral_entropy': fft_result['spectral_entropy'],
        'trend_slope': fft_result['trend_slope'],
    }
    
    # ========================================
    # ARGMIN/ARGMAX EXTREMES
    # ========================================
    extremes = find_significant_extremes(closes, lookback=min(500, n))
    
    # ========================================
    # SWING DETECTION
    # ========================================
    swing_highs, swing_lows = find_swing_points(
        highs, lows, closes, lookback=swing_lookback, min_amplitude_pct=0.0003
    )
    
    # ========================================
    # CLUSTER INTO ZONES
    # ========================================
    res_zones = cluster_price_levels(swing_highs, cluster_pct=0.002)
    sup_zones = cluster_price_levels(swing_lows,  cluster_pct=0.002)
    
    # ========================================
    # PER-BAR SIGNIFICANCE
    # ========================================
    sig_scores = hilo_range_significance(highs, lows, volumes, window=20)
    
    # ========================================
    # SCORE RESISTANCE ZONES
    # ========================================
    targets_up: List[Dict] = []
    for zone in res_zones:
        if zone['price'] <= current_price * 1.0005:
            continue
        
        dist = (zone['price'] - current_price) / current_price * 100.0
        vp   = volume_pressure_at_zone(
            zone['price'], highs, lows, closes, opens, volumes)
        
        # FFT alignment check
        fft_align = check_fft_alignment(zone['price'], fft_up, tolerance_pct=0.003)
        
        # Argmin/argmax relevance
        arg_rel = check_argmin_argmax_relevance(zone['price'], extremes, current_price)
        
        # Score with all factors
        score, details = score_resistance_zone(
            zone, n, vp, sig_scores, fft_align, arg_rel
        )
        
        # ETA: use FFT if aligned, otherwise estimate from distance
        if fft_align[0] and fft_align[2]:
            eta = format_eta_minutes(fft_align[2]['bars_ahead'])
            eta_source = 'fft'
        else:
            est_bars = abs(dist) / max(avg_1m_range_pct, 0.001)
            eta = format_eta_minutes(est_bars)
            eta_source = 'est'
        
        targets_up.append({
            'price': zone['price'],
            'score': score,
            'dist_pct': dist,
            'touches': zone['touches'],
            'zone_type': vp['zone_type'],
            'bull_pct': vp['bull_pct'],
            'interactions': vp['interactions'],
            'eta': eta,
            'eta_source': eta_source,
            'details': details,
        })
    
    # ========================================
    # SCORE SUPPORT ZONES
    # ========================================
    targets_down: List[Dict] = []
    for zone in sup_zones:
        if zone['price'] >= current_price * 0.9995:
            continue
        
        dist = (zone['price'] - current_price) / current_price * 100.0
        vp   = volume_pressure_at_zone(
            zone['price'], highs, lows, closes, opens, volumes)
        
        # FFT alignment check
        fft_align = check_fft_alignment(zone['price'], fft_down, tolerance_pct=0.003)
        
        # Argmin/argmax relevance
        arg_rel = check_argmin_argmax_relevance(zone['price'], extremes, current_price)
        
        # Score with all factors
        score, details = score_support_zone(
            zone, n, vp, sig_scores, fft_align, arg_rel
        )
        
        # ETA
        if fft_align[0] and fft_align[2]:
            eta = format_eta_minutes(fft_align[2]['bars_ahead'])
            eta_source = 'fft'
        else:
            est_bars = abs(dist) / max(avg_1m_range_pct, 0.001)
            eta = format_eta_minutes(est_bars)
            eta_source = 'est'
        
        targets_down.append({
            'price': zone['price'],
            'score': score,
            'dist_pct': dist,
            'touches': zone['touches'],
            'zone_type': vp['zone_type'],
            'bull_pct': vp['bull_pct'],
            'interactions': vp['interactions'],
            'eta': eta,
            'eta_source': eta_source,
            'details': details,
        })
    
    # ========================================
    # SELECT TOP-N BY SCORE, SORT NEAREST-FIRST
    # ========================================
    targets_up.sort(key=lambda t: t['score'], reverse=True)
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
        'fft_targets_up':   fft_up[:n_targets],
        'fft_targets_down': fft_down[:n_targets],
        'extremes':         extremes,
        'fft_info':         fft_info,
    }


# ==========================================
# CONCURRENT FILTER FUNCTIONS
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
    
    fft_info = sr.get('fft_info', {})
    extremes = sr.get('extremes', {})
    cycle_strength = fft_info.get('cycle_strength', 0)
    spec_entropy = fft_info.get('spectral_entropy', 1)
    dominant_cycles = fft_info.get('dominant_cycles', [])

    print("\n" + "=" * 70)
    print(f"  ★  1m S/R TARGET MAP  —  {symbol}  ★")
    print(f"  (FFT cycle-validated + argmin/argmax structural anchors)")
    print("=" * 70)
    print(f"  Entry Price    : {current_price:.10f}")
    print(f"  1m CMO         : {cmo_val:+.2f}  (< -50 = oversold)")
    print(f"  Vol Ratio      : x{vratio:.2f}")
    print(f"  Bull Rej Vol   : {bull_ratio*100:.1f}%  (>65% = confirmed)")
    print(f"  ML Spike Prob  : {ml_prob*100:.1f}%")
    print(f"  Vol Pressure   : {bias_label}  ({bull_pct_overall:.1f}% bull / "
          f"{100-bull_pct_overall:.1f}% bear, last 20 candles)")
    print(f"  Avg 1m Range   : {avg_range:.4f}%")
    
    # FFT Info
    print("-" * 70)
    print("  📊  FFT CYCLE ANALYSIS")
    print(f"  Cycle Strength : {cycle_strength:.2f}  (higher = more cyclical)")
    print(f"  Spectral Entropy: {spec_entropy:.2f}  (lower = more structured)")
    if dominant_cycles:
        print(f"  Dominant Cycles:")
        for i, c in enumerate(dominant_cycles[:3], 1):
            print(f"    C{i}: {c['period']:.0f} bars ({c['period']:.0f}min) | "
                  f"Amp: {c['amplitude']:.6f} | "
                  f"Next peak: {c['bars_to_peak']:.0f} bars | "
                  f"Power: {c['relative_power']:.2f}")
    else:
        print("  No dominant cycles detected")
    
    # Argmin/Argmax Info
    print("-" * 70)
    print("  📍  STRUCTURAL EXTREMES (argmin/argmax of last 500)")
    if extremes:
        print(f"  Global High    : {extremes.get('high', 0):.10f}  "
              f"({extremes['global_argmax'].get('age', 0)} bars ago)")
        print(f"  Global Low     : {extremes.get('low', 0):.10f}  "
              f"({extremes['global_argmin'].get('age', 0)} bars ago)")
        print(f"  Range          : {extremes.get('range_pct', 0):.3f}%")
    
    # FFT Projected Targets (pure cycle-based)
    fft_up = sr.get('fft_targets_up', [])
    fft_down = sr.get('fft_targets_down', [])
    if fft_up or fft_down:
        print("-" * 70)
        print("  🔮  FFT PROJECTED TARGETS (cycle-based, unvalidated)")
        for i, t in enumerate(fft_up[:2], 1):
            print(f"  FFT-R{i}: {t['price']:.10f}  ({t['dist_pct']:+.3f}%)  "
                  f"ETA: {format_eta_minutes(t['bars_ahead'])}  "
                  f"Conf: {t['confidence']:.2f}")
        for i, t in enumerate(fft_down[:2], 1):
            print(f"  FFT-S{i}: {t['price']:.10f}  ({t['dist_pct']:+.3f}%)  "
                  f"ETA: {format_eta_minutes(t['bars_ahead'])}  "
                  f"Conf: {t['confidence']:.2f}")
    
    print("-" * 70)

    # ---- RESISTANCE / TARGETS UP ----
    up = sr['targets_up']
    if up:
        print(f"\n  📈  RESISTANCE TARGETS  (FFT-validated, nearest → furthest)\n")
        for i, t in enumerate(up, 1):
            bar  = "█" * int(t['score'] * 10) + "░" * (10 - int(t['score'] * 10))
            sign = "🔴" if t['zone_type'] == 'RESISTANCE' else ("🟢" if t['zone_type'] == 'SUPPORT' else "⚪")
            fft_mark = "⚡" if t.get('details', {}).get('fft', 0) > 0.1 else "  "
            arg_mark = "📍" if t.get('details', {}).get('arg_relevance', 0) > 0.1 else "  "
            eta_src = "(FFT)" if t.get('eta_source') == 'fft' else "(est)"
            
            print(f"  T{i}  {t['price']:.10f}  ({t['dist_pct']:+.3f}%)  ETA: {t['eta']} {eta_src}")
            print(f"       {fft_mark}{arg_mark} Score [{bar}] {t['score']:.2f}  "
                  f"Touches: {t['touches']}  {sign} {t['zone_type']}  "
                  f"BullVol@zone: {t['bull_pct']*100:.0f}%  "
                  f"Interactions: {t['interactions']}")
            # Show score breakdown
            d = t.get('details', {})
            if d:
                parts = []
                if d.get('fft', 0) > 0.05:
                    parts.append(f"FFT:{d['fft']:.2f}")
                if d.get('arg_relevance', 0) > 0.05:
                    parts.append(f"ARG:{d['arg_relevance']:.2f}({d.get('arg_reason','')[:20]})")
                if parts:
                    print(f"             + {' | '.join(parts)}")
            print()
    else:
        print("\n  📈  No resistance targets found.\n")

    # ---- SUPPORT / TARGETS DOWN ----
    dn = sr['targets_down']
    if dn:
        print(f"  📉  SUPPORT LEVELS  (FFT-validated, nearest → furthest)\n")
        for i, t in enumerate(dn, 1):
            bar  = "█" * int(t['score'] * 10) + "░" * (10 - int(t['score'] * 10))
            sign = "🟢" if t['zone_type'] == 'SUPPORT' else ("🔴" if t['zone_type'] == 'RESISTANCE' else "⚪")
            fft_mark = "⚡" if t.get('details', {}).get('fft', 0) > 0.1 else "  "
            arg_mark = "📍" if t.get('details', {}).get('arg_relevance', 0) > 0.1 else "  "
            eta_src = "(FFT)" if t.get('eta_source') == 'fft' else "(est)"
            
            print(f"  S{i}  {t['price']:.10f}  ({t['dist_pct']:+.3f}%)  ETA: {t['eta']} {eta_src}")
            print(f"       {fft_mark}{arg_mark} Score [{bar}] {t['score']:.2f}  "
                  f"Touches: {t['touches']}  {sign} {t['zone_type']}  "
                  f"BullVol@zone: {t['bull_pct']*100:.0f}%  "
                  f"Interactions: {t['interactions']}")
            d = t.get('details', {})
            if d:
                parts = []
                if d.get('fft', 0) > 0.05:
                    parts.append(f"FFT:{d['fft']:.2f}")
                if d.get('arg_relevance', 0) > 0.05:
                    parts.append(f"ARG:{d['arg_relevance']:.2f}({d.get('arg_reason','')[:20]})")
                if parts:
                    print(f"             + {' | '.join(parts)}")
            print()
    else:
        print("  📉  No support levels found.\n")

    # ---- IMMEDIATE TRADE BIAS ----
    print("-" * 70)
    print("  ⚡  TRADE BIAS SUMMARY")
    if up and dn:
        nearest_up   = up[0]
        nearest_dn   = dn[0]
        rr           = abs(nearest_up['dist_pct']) / max(abs(nearest_dn['dist_pct']), 0.0001)
        bias_ok      = vol_bias > 0.55 and bull_ratio > 0.55
        
        # Check FFT alignment for nearest targets
        up_fft_aligned = nearest_up.get('details', {}).get('fft', 0) > 0.1
        dn_fft_aligned = nearest_dn.get('details', {}).get('fft', 0) > 0.1
        
        print(f"  Nearest target : {nearest_up['price']:.10f}  ({nearest_up['dist_pct']:+.3f}%)"
              f"  ETA {nearest_up['eta']} {'⚡FFT' if up_fft_aligned else ''}")
        print(f"  Nearest stop   : {nearest_dn['price']:.10f}  ({nearest_dn['dist_pct']:+.3f}%)"
              f" {'⚡FFT' if dn_fft_aligned else ''}")
        print(f"  R:R (target/stop) : {rr:.2f}x")
        
        signals = []
        if bias_ok:
            signals.append("Volume+Rejection✓")
        if rr >= 1.5:
            signals.append("R:R≥1.5✓")
        if up_fft_aligned:
            signals.append("Target-FFT✓")
        if dn_fft_aligned:
            signals.append("Stop-FFT✓")
        if cycle_strength > 0.3:
            signals.append("Cyclical✓")
        
        if len(signals) >= 3:
            print(f"  Signal         : ✅  LONG — {' | '.join(signals)}")
        elif len(signals) >= 2:
            print(f"  Signal         : ⏳  LIKELY — {' | '.join(signals)}")
        elif rr < 1.0:
            print("  Signal         : ⚠️   SKIP — R:R unfavorable")
        else:
            print(f"  Signal         : ⏳  WAIT — {' | '.join(signals) if signals else 'no confirmation'}")
    elif up:
        print(f"  Target only: {up[0]['price']:.10f} ({up[0]['dist_pct']:+.3f}%)  ETA {up[0]['eta']}")
    elif dn:
        print(f"  Support only: {dn[0]['price']:.10f} ({dn[0]['dist_pct']:+.3f}%)  ETA {dn[0]['eta']}")
    else:
        print("  No clear levels found.")
    
    print("=" * 70 + "\n")


# ==========================================
# MAIN
# ==========================================

def main():
    start_time = time.time()
    trader = Trader('credentials.txt')
    trading_pairs = trader.get_usdc_pairs()

    print("=" * 70)
    print("  MTF SCANNER  +  FFT-VALIDATED 1m S/R ZONE TARGET ENGINE")
    print("=" * 70 + "\n")

    # ---- Multi-TF candidate filtering ----
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

    print("\n" + "-" * 70)
    print(f"  SELECTED SYMBOL : {sym}")
    print(f"  SELECTION MODE  : {mode}")
    print(f"  1m CMO          : {cmo_val:.4f}")
    print(f"  1m Volume Ratio : x{vratio:.4f}")
    print(f"  Bull Rej Vol    : {live_bull_ratio*100:.2f}%")
    print(f"  ML Spike Prob   : {ml_prob*100:.2f}%")
    print("-" * 70)

    # ---- Fetch 500 1m raw klines for S/R engine ----
    print("\nFetching 500 1m klines for FFT + argmin/argmax analysis...")
    klines_1m = trader.get_klines(sym, '1m', limit=500, return_raw=True)
    if not klines_1m:
        print("Could not fetch 1m klines. Exiting.")
        sys.exit(0)
    
    print(f"Retrieved {len(klines_1m)} klines")

    current_price = float(klines_1m[-1][4])

    # ---- Run enhanced S/R target engine ----
    sr = get_sr_targets(
        raw_klines     = klines_1m,
        current_price  = current_price,
        swing_lookback = 5,
        n_targets      = 4,
    )

    # ---- Print results ----
    format_sr_output(sym, sr, current_price,
                     cmo_val, vratio, live_bull_ratio, ml_prob)

    print(f"Total Execution Time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
