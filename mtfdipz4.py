import gc
import sys
from binance.client import Client
import numpy as np
import talib as ta
import time
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
from scipy.signal import argrelextrema


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

    # ==========================================
    # NEW: MAXIMUM HISTORICAL DATA PAGINATOR
    # ==========================================
    def get_max_klines(self, symbol: str, interval: str, max_candles: int = 10000) -> list:
        """
        Paginates backward in time to fetch the maximum allowed historical data.
        Stops when max_candles is reached or no more historical data exists.
        """
        MAX_PER_REQ = 1000
        all_klines = []
        end_time = None
        
        print(f"  ⏳ Fetching maximum historical data (up to {max_candles} candles)...", end="", flush=True)
        start_fetch_time = time.time()
        
        while len(all_klines) < max_candles:
            self.rate_limiter.acquire()
            params = {'symbol': symbol, 'interval': interval, 'limit': MAX_PER_REQ}
            if end_time is not None:
                params['endTime'] = end_time
                
            try:
                klines = self.client.get_klines(**params)
                if not klines:
                    break # No more historical data available
                    
                # Prepend older data to the front of our array
                all_klines = klines + all_klines
                # Set end_time to exactly 1ms before the oldest candle we just fetched
                end_time = int(klines[0][0]) - 1
                
            except Exception as e:
                if 'rate limit' in str(e).lower():
                    time.sleep(2)
                else:
                    break
                    
        elapsed = time.time() - start_fetch_time
        # Trim to exact max_candles if we overshot
        final_klines = all_klines[-max_candles:] 
        print(f" Done! Fetched {len(final_klines)} candles in {elapsed:.1f}s")
        return final_klines


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


# ==========================================
# ML FORECAST ENGINE (ZERO RAM/HDD USAGE)
# ==========================================
def ml_forecast_probability_and_target(R, C, E, bull_ratio, cmo, vratio, 
                                       argmin_gt, rsi, stoch, reg_div, hid_div, 
                                       dbl_bot, fib_z, delta_d, sweep_r):
    rsi_score = max(0.0, (100 - rsi) / 100.0) if rsi < 40 else 0.0
    stoch_score = max(0.0, (100 - stoch) / 100.0) if stoch < 30 else 0.0
    
    score = (
        0.15 * argmin_gt +
        0.10 * rsi_score +
        0.10 * stoch_score +
        0.15 * reg_div +
        0.10 * hid_div +
        0.05 * dbl_bot +
        0.05 * fib_z +
        0.15 * delta_d +
        0.15 * sweep_r +
        0.10 * bull_ratio +
        0.05 * max(0, -cmo / 100.0) +
        0.05 * min(vratio / 5.0, 1.0)
    )
    
    probability = 1 / (1 + np.exp(-(score - 0.5) * 12))
    
    if probability > 0.8:      forecast_mult = 2.0
    elif probability > 0.6:    forecast_mult = 1.5
    elif probability > 0.4:    forecast_mult = 1.0
    else:                      forecast_mult = 0.0
        
    return probability, forecast_mult


# ==========================================
# REJECTION PATTERN STACK
# ==========================================
def detect_rejection_patterns(raw_klines: list, lookback: int = 15) -> Dict:
    if not raw_klines or len(raw_klines) < 30:
        return {'talib_hits': {}, 'pin_bar': False, 'tweezer_bottom': False,
                'rejection_score': 0, 'detail': 'insufficient bars'}

    o = np.array([float(k[1]) for k in raw_klines], dtype='float64')
    h = np.array([float(k[2]) for k in raw_klines], dtype='float64')
    l = np.array([float(k[3]) for k in raw_klines], dtype='float64')
    c = np.array([float(k[4]) for k in raw_klines], dtype='float64')

    pattern_fns = {
        'HAMMER': ta.CDLHAMMER, 'INV_HAMMER': ta.CDLINVERTEDHAMMER,
        'BULL_ENGULFING': ta.CDLENGULFING, 'PIERCING': ta.CDLPIERCING,
        'MORNING_STAR': ta.CDLMORNINGSTAR, 'MORNING_DOJI_STAR': ta.CDLMORNINGDOJISTAR,
        'DRAGONFLY_DOJI': ta.CDLDRAGONFLYDOJI, 'THREE_WHITE_SOLDIERS': ta.CDL3WHITESOLDIERS,
        'HARAMI': ta.CDLHARAMI, 'BELT_HOLD': ta.CDLBELTHOLD, 'TAKURI': ta.CDLTAKURI,
        'HOMING_PIGEON': ta.CDLHOMINGPIGEON, 'MAT_HOLD': ta.CDLMATHOLD,
    }

    hits = {}
    for name, fn in pattern_fns.items():
        try:
            arr = fn(o, h, l, c)
            recent = arr[-lookback:]
            nz = recent[recent != 0]
            if len(nz) > 0 and nz[-1] > 0: hits[name] = int(nz[-1])
        except Exception: continue

    body = abs(c[-1] - o[-1])
    lower_wick = min(o[-1], c[-1]) - l[-1]
    upper_wick = h[-1] - max(o[-1], c[-1])
    rng = h[-1] - l[-1]
    pin_bar = bool(rng > 0 and (lower_wick / rng) > 0.6 and (body / rng) < 0.30 and c[-1] >= o[-1])

    tweezer = False
    if len(l) >= 2 and l[-2] > 0:
        low_diff = abs(l[-1] - l[-2]) / l[-2]
        if low_diff < 0.0015 and c[-2] < o[-2] and c[-1] > o[-1]: tweezer = True

    spring = False
    if len(l) >= 20:
        recent_low = float(np.min(l[-21:-1]))
        vols = np.array([float(k[5]) for k in raw_klines[-21:]], dtype='float64')
        avg_vol = float(np.mean(vols[:-1])) if len(vols) > 1 else 0.0
        if l[-1] < recent_low and c[-1] > recent_low and vols[-1] > avg_vol * 1.2: spring = True

    score = sum(1 for v in hits.values() if v > 0) + int(pin_bar) + int(tweezer) + int(spring)
    return {'talib_hits': hits, 'pin_bar': pin_bar, 'tweezer_bottom': tweezer,
            'wyckoff_spring': spring, 'rejection_score': score,
            'detail': f"{score} rejection pattern(s) confirmed on last closed bar"}


# ==========================================
# ENTRY-ZONE EXHAUSTION
# ==========================================
def detect_entry_exhaustion(raw_klines: list, current_price: float,
                             zone_pct: float = 0.015, min_bars: int = 8) -> Dict:
    if not raw_klines: return {'exhausted': False, 'confidence': 0.0, 'detail': 'no data'}
    z_lo, z_hi = current_price * (1 - zone_pct), current_price * (1 + zone_pct)
    zv, zc, zo = [], [], []
    for k in raw_klines[-80:]:
        o, h, l, c, v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
        if l <= z_hi and h >= z_lo: zv.append(v); zc.append(c); zo.append(o)

    if len(zv) < min_bars: return {'exhausted': False, 'confidence': 0.0, 'detail': f'only {len(zv)} bars touched zone'}
    split = max(1, int(len(zv) * 0.5))
    early_vol, late_vol = float(np.mean(zv[:split])), float(np.mean(zv[split:]))
    vol_ratio = late_vol / early_vol if early_vol > 0 else 1.0
    early_std, late_std = float(np.std(zc[:split])), float(np.std(zc[split:]))
    coiling = late_std < early_std * 0.75 if early_std > 0 else False
    bull_closes = sum(1 for cc, oo in zip(zc[split:], zo[split:]) if cc >= oo)
    bull_ratio_in_zone = bull_closes / max(len(zc[split:]), 1)
    exhausted = (vol_ratio < 0.60) and coiling and (bull_ratio_in_zone >= 0.50)
    confidence = max(0.0, min(1.0, (1 - min(vol_ratio, 1.0)) * 0.5 + (0.3 if coiling else 0.0) + bull_ratio_in_zone * 0.2))
    return {'exhausted': exhausted, 'confidence': confidence, 'vol_ratio': vol_ratio, 'coiling': coiling, 'bull_ratio_in_zone': bull_ratio_in_zone, 'bars_in_zone': len(zv), 'detail': f"Vol {vol_ratio:.0%} of earlier pace, coiling={coiling}, bull closes={bull_ratio_in_zone:.0%}"}


# ==========================================
# DYNAMIC 360-DEGREE CYCLIC FORECAST ENGINE
# ==========================================
def dynamic_360_cycle_forecast(close: List[float], min_cycle: int = 20, max_cycle: int = 500) -> Dict:
    n = len(close)
    if n < max_cycle + 10:
        return {'direction': 'UNKNOWN', 'phase_deg': 0, 'cycle_length': 0, 'cycle_target': 0, 'detail': 'Insufficient data for FFT'}

    sma_window = max_cycle
    sma = np.convolve(close, np.ones(sma_window)/sma_window, mode='valid')
    detrended = np.array(close[-len(sma):]) - sma

    if np.std(detrended) < 1e-10:
        return {'direction': 'FLAT', 'phase_deg': 0, 'cycle_length': 0, 'cycle_target': 0, 'detail': 'No cyclical variance (Flat)'}

    fft_vals = np.fft.rfft(detrended)
    fft_freqs = np.fft.rfftfreq(len(detrended), d=1.0)
    
    valid_idx = (fft_freqs > 1.0/max_cycle) & (fft_freqs < 1.0/min_cycle)
    if not np.any(valid_idx):
        return {'direction': 'UNKNOWN', 'phase_deg': 0, 'cycle_length': 0, 'cycle_target': 0, 'detail': 'No valid cycle found in range'}

    filtered_fft = fft_vals.copy()
    filtered_fft[~valid_idx] = 0

    dominant_idx = np.argmax(np.abs(filtered_fft))
    dominant_freq = fft_freqs[dominant_idx]
    cycle_length = int(round(1.0 / dominant_freq))
    
    phase_rad = np.angle(fft_vals[dominant_idx])
    phase_deg = float(np.degrees(phase_rad)) % 360
    
    amplitude = np.abs(fft_vals[dominant_idx]) / len(detrended) * 2
    
    if 0 <= phase_deg < 180:
        direction = "UP"
        remaining_gain = amplitude * (np.sin(np.pi/2) - np.sin(phase_rad))
        cycle_target = close[-1] + max(0, remaining_gain)
    else:
        direction = "DOWN"
        remaining_drop = amplitude * (np.sin(3*np.pi/2) - np.sin(phase_rad))
        cycle_target = close[-1] + min(0, remaining_drop)

    return {
        'direction': direction,
        'phase_deg': phase_deg,
        'cycle_length': cycle_length,
        'cycle_target': float(cycle_target),
        'amplitude': float(amplitude),
        'detail': f"Phase {phase_deg:.1f}° | {direction} cycle | Wavelength: {cycle_length} bars"
    }


# ==========================================
# STRUCTURAL RANGE ENGINE
# ==========================================
def get_structural_extremes(close: np.ndarray, highs: np.ndarray, lows: np.ndarray, lookback: int) -> Dict:
    n = len(close); start = max(0, n - lookback); c, h, l = close[start:], highs[start:], lows[start:]; sl = len(c)
    amax_i, amin_i = int(np.argmax(c)), int(np.argmin(c))
    g_high, g_low = float(c[amax_i]), float(c[amin_i])
    high_age, low_age = sl - amax_i, sl - amin_i
    if low_age < high_age: more_recent, mr_label = "ARGMIN", "🟢 ARGMIN (low is fresher -> floor established recently)"
    elif high_age < low_age: more_recent, mr_label = "ARGMAX", "🔴 ARGMAX (high is fresher -> ceiling established recently)"
    else: more_recent, mr_label = "EQUAL", "⚪ EQUAL"
    rng, rng_pct = g_high - g_low, ((g_high - g_low) / g_low * 100) if g_low > 0 else 0
    pos = (close[-1] - g_low) / rng if rng > 0 else 0.5
    return {'high': g_high, 'low': g_low, 'high_bar': float(h[amax_i]), 'low_bar': float(l[amin_i]), 'high_age': high_age, 'low_age': low_age, 'more_recent': more_recent, 'mr_label': mr_label, 'range_size': rng, 'range_pct': rng_pct, 'position': pos, 'bars_used': sl}

def build_fib_grid(extremes: Dict, current_price: float) -> List[Dict]:
    lo, hi, rng = extremes['low'], extremes['high'], extremes['range_size']
    if rng <= 0: return []
    fibs = [(0.000, "ARGMIN"), (0.236, "F236"), (0.382, "F382"), (0.500, "F500"), (0.618, "F618"), (0.786, "F786"), (1.000, "ARGMAX")]
    grid = []
    for fib, label in fibs:
        price = lo + rng * fib; dist = (price - current_price) / current_price * 100
        direction = 'UP' if price > current_price else ('DOWN' if price < current_price else 'AT')
        grid.append({'price': price, 'fib': fib, 'label': label, 'dist_pct': dist, 'direction': direction})
    return grid

def volume_profile_at_level(level_price: float, raw_klines: list, tolerance: float) -> Dict:
    bull_vol = bear_vol = 0.0; touches = bull_rej = bear_rej = 0; vol_seq = []
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
    total = bull_vol + bear_vol; bp = bull_vol / total if total > 0 else 0.5
    exh, exh_detail = 0.0, "N/A"
    if len(vol_seq) >= 6:
        mid = len(vol_seq) // 2; first_avg, second_avg = np.mean(vol_seq[:mid]), np.mean(vol_seq[mid:])
        if first_avg > 0:
            ratio = second_avg / first_avg; exh = max(0.0, min(1.0, 1.0 - ratio))
            exh_detail = f"STRONG ({ratio:.0%})" if ratio < 0.5 else (f"MODERATE ({ratio:.0%})" if ratio < 0.8 else f"Weak ({ratio:.0%})")
    rej_vol_total = sum(float(k[5]) for k in raw_klines if float(k[3]) <= hi and float(k[2]) >= lo and ((float(k[3]) < lo and float(k[4]) >= level_price) or (float(k[2]) > hi and float(k[4]) <= level_price)))
    rej_int = rej_vol_total / total if total > 0 else 0.0
    verdict = "SUPPORT" if bp > 0.58 else ("RESISTANCE" if bp < 0.42 else "NEUTRAL")
    return {'bull_pct': bp, 'total_volume': total, 'touches': touches, 'bull_rej': bull_rej, 'bear_rej': bear_rej, 'total_rej': bull_rej + bear_rej, 'exhaustion': exh, 'exhaustion_detail': exh_detail, 'rej_intensity': rej_int, 'verdict': verdict}

def detect_extreme_exhaustion(extreme_price: float, direction: str, raw_klines: list, zone_pct: float = 0.04) -> Dict:
    z_lo = extreme_price * (1 - zone_pct) if direction == 'high' else extreme_price * (1 - zone_pct * 0.5)
    z_hi = extreme_price * (1 + zone_pct * 0.5) if direction == 'high' else extreme_price * (1 + zone_pct)
    zv, zc = [], []
    for k in raw_klines:
        h, l, c, v = float(k[2]), float(k[3]), float(k[4]), float(k[5])
        if l <= z_hi and h >= z_lo: zv.append(v); zc.append(c)
    if len(zv) < 8: return {'exhaustion': 0.0, 'detail': 'Insufficient data', 'pattern': 'NONE', 'approach_vol': 0, 'final_vol': 0}
    split = int(len(zv) * 0.6); app_vol, fin_vol = np.mean(zv[:split]), np.mean(zv[split:])
    final_c = zc[split:]; reached = (max(final_c) >= extreme_price * 0.998) if direction == 'high' else (min(final_c) <= extreme_price * 1.002)
    exh, pattern, detail = 0.0, "NONE", ""
    if app_vol > 0:
        ratio = fin_vol / app_vol
        if direction == 'high':
            if ratio < 0.4 and reached: exh, pattern, detail = 0.9, "CLIMAX_EXHAUSTION", f"Vol collapsed to {ratio:.0%} at peak"
            elif ratio < 0.65 and reached: exh, pattern, detail = 0.6, "FADE", f"Vol faded to {ratio:.0%} near high"
            elif ratio < 0.85: exh, pattern, detail = 0.3, "MILD_FADE", f"Slight fade to {ratio:.0%}"
            else: detail = f"No exhaustion (vol at {ratio:.0%})"
        else:
            if ratio < 0.35 and reached: exh, pattern, detail = 0.9, "CAPITULATION", f"Vol died to {ratio:.0%} after low"
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
    fib = level['fib']; ext_prox = max(0.0, 1.0 - abs(fib - 1.0) * 2.0) if is_above else max(0.0, 1.0 - abs(fib - 0.0) * 2.0)
    score = 0.25 * pressure + 0.15 * touch_sc + 0.15 * rej_sc + 0.15 * ri_sc + 0.20 * exh_sc + 0.10 * ext_prox
    return score, {'pressure': pressure, 'touches': touch_sc, 'rejection_candles': rej_sc, 'rejection_intensity': ri_sc, 'exhaustion': exh_sc, 'extreme_prox': ext_prox}

def estimate_eta(dist_pct: float, range_pct: float, vol_bias: float) -> str:
    if range_pct <= 0: return "N/A"
    bias = vol_bias if vol_bias >= 0.5 else (1.0 - vol_bias)
    speed = 0.7 + 0.6 * bias if ((dist_pct > 0 and vol_bias > 0.5) or (dist_pct < 0 and vol_bias < 0.5)) else 1.2 + 0.8 * (1.0 - bias)
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
    grid = build_fib_grid(ext, current_price); tolerance = max(ext['range_size'] * 0.025, avg_range_pct / 100 * current_price * 1.5)
    n = len(close); start = max(0, n - lookback); klines_slice = raw_klines[start:]
    exh_high, exh_low = detect_extreme_exhaustion(ext['high'], 'high', klines_slice), detect_extreme_exhaustion(ext['low'], 'low', klines_slice)
    min_dist = max(ext['range_pct'] * 0.05, 0.08); targets_up, targets_down, grid_out = [], [], []
    for level in grid:
        vp = volume_profile_at_level(level['price'], klines_slice, tolerance)
        lev_exh = exh_high['exhaustion'] if level['fib'] >= 0.618 else (exh_low['exhaustion'] if level['fib'] <= 0.382 else 0.0)
        dist, is_above, is_below = abs(level['dist_pct']), level['direction'] == 'UP', level['direction'] == 'DOWN'
        if dist < min_dist: grid_out.append({**level, **vp, 'score': 0, 'status': 'TOO_CLOSE'}); continue
        score, details = score_level(level, vp, lev_exh, ext['range_pct'], is_above); eta = estimate_eta(level['dist_pct'], ext['range_pct'], vol_bias)
        entry = {'price': level['price'], 'score': score, 'dist_pct': level['dist_pct'], 'label': level['label'], 'fib': level['fib'], 'verdict': vp['verdict'], 'bull_pct': vp['bull_pct'], 'touches': vp['touches'], 'rejections': vp['total_rej'], 'rej_intensity': vp['rej_intensity'], 'eta': eta, 'details': details}
        grid_out.append({**level, **vp, **entry, 'status': 'ACTIVE'})
        if is_above: targets_up.append(entry)
        elif is_below: targets_down.append(entry)
    targets_up.sort(key=lambda t: t['score'], reverse=True); targets_down.sort(key=lambda t: t['score'], reverse=True)
    return {'lookback': lookback, 'extremes': ext, 'targets_up': sorted(targets_up[:4], key=lambda t: t['dist_pct']), 'targets_down': sorted(targets_down[:4], key=lambda t: -t['dist_pct']), 'exh_high': exh_high, 'exh_low': exh_low, 'grid': grid_out, 'min_dist': min_dist}

def get_sr_targets(raw_klines: list, current_price: float) -> Dict:
    if len(raw_klines) < 100: return {'lookbacks': [], 'vol_bias': 0.5, 'avg_range': 0}
    highs, lows, closes, volumes = np.array([float(k[2]) for k in raw_klines]), np.array([float(k[3]) for k in raw_klines]), np.array([float(k[4]) for k in raw_klines]), np.array([float(k[5]) for k in raw_klines])
    candle_ranges = (highs - lows) / (closes + 1e-12) * 100.0; avg_range = float(np.mean(candle_ranges[-50:]))
    closed_vols = [v for v in volumes[-21:-1] if v > 0]
    if closed_vols:
        rec = raw_klines[-21:-1]; bv = sum(float(k[5]) for k in rec if float(k[4]) >= float(k[1]) and float(k[5]) > 0); bear_v = sum(float(k[5]) for k in rec if float(k[4]) < float(k[1]) and float(k[5]) > 0); tv = bv + bear_v; vol_bias = bv / tv if tv > 0 else 0.5
    else: vol_bias = 0.5

    n = len(raw_klines)
    dynamic_lookbacks = [n // 4, n // 2, n]
    valid_lookbacks = [lb for lb in dynamic_lookbacks if lb >= 100]

    return {'lookbacks': [analyze_lookback(raw_klines, closes, highs, lows, current_price, lb, avg_range, vol_bias) for lb in valid_lookbacks], 'vol_bias': vol_bias, 'avg_range': avg_range}


# ==========================================
# CONCURRENT FILTER FUNCTIONS
# ==========================================
def check_tf_dip(trader, symbol, interval):
    return (symbol, linear_regression_dip(trader.get_klines(symbol, interval, limit=300), 0.01))

def check_1m_final(trader, symbol):
    # Keep initial scan fast using limited data
    klines = trader.get_klines(symbol, '1m', limit=200, return_raw=True)
    if not klines or len(klines) < 50: return (symbol, 0.0, 0.0, False, 0.0, 0.0, {}, {}, {}, 0.0, {})

    c = np.array([float(k[4]) for k in klines], dtype='float64')
    h = np.array([float(k[2]) for k in klines], dtype='float64')
    l = np.array([float(k[3]) for k in klines], dtype='float64')
    v = np.array([float(k[5]) for k in klines], dtype='float64')
    o = np.array([float(k[1]) for k in klines], dtype='float64')

    lookback = 20
    idx_min, idx_max = np.argmin(l[-lookback:]), np.argmax(h[-lookback:])
    argmin_gt_argmax = 1.0 if idx_min > idx_max else 0.0
    rsi = float(ta.RSI(c, timeperiod=14)[-1])
    stoch_k, _ = ta.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowd_period=3)
    stoch_k = float(stoch_k[-1])
    rsi_vals = ta.RSI(c, timeperiod=14)
    p_lows_idx, r_lows_idx = argrelextrema(l[-30:], np.less, order=3)[0], argrelextrema(rsi_vals[-30:], np.less, order=3)[0]
    reg_div, hid_div = 0.0, 0.0
    if len(p_lows_idx) >= 2 and len(r_lows_idx) >= 2:
        p_l1, p_l2, r_l1, r_l2 = l[-30:][p_lows_idx[-2:]][0], l[-30:][p_lows_idx[-2:]][1], rsi_vals[-30:][r_lows_idx[-2:]][0], rsi_vals[-30:][r_lows_idx[-2:]][1]
        if p_l1 > p_l2 and r_l1 < r_l2: reg_div = 1.0 
        if p_l1 < p_l2 and r_l1 < r_l2: hid_div = 1.0 
    dbl_bottom = 0.0
    if len(p_lows_idx) >= 2:
        low1, low2 = l[-30:][p_lows_idx[-2]], l[-30:][p_lows_idx[-1]]
        if abs(low1 - low2) <= ((np.mean(h[-30:]) - np.mean(l[-30:])) * 0.02) and c[-1] > max(low1, low2): dbl_bottom = 1.0
    fib_zone = 0.0; rng = np.max(h[-40:]) - np.min(l[-40:])
    if rng > 0:
        if np.max(h[-40:]) - (rng * 0.786) <= c[-1] <= np.max(h[-40:]) - (rng * 0.618): fib_zone = 1.0 
    deltas = np.zeros(len(klines))
    for i in range(len(klines)):
        rng_k = h[i] - l[i]
        if rng_k > 0: deltas[i] = v[i] * (abs(c[i] - o[i]) / rng_k) * (1 if c[i] > o[i] else -1)
    cum_delta = np.cumsum(deltas); pl_idx = np.argmin(l[-20:])
    delta_div = 1.0 if (l[-1] < l[pl_idx] and cum_delta[-1] > cum_delta[pl_idx]) else 0.0
    sl_20 = np.min(l[-21:-1]); sweep_rec = 1.0 if (l[-1] < sl_20 and c[-1] > sl_20) else 0.0

    new_feats = {'argmin': argmin_gt_argmax, 'rsi': rsi, 'stoch': stoch_k, 'reg_div': reg_div, 'hid_div': hid_div, 'dbl_bot': dbl_bottom, 'fib': fib_zone, 'delta_div': delta_div, 'sweep': sweep_rec}

    close = c.tolist(); volumes = v.tolist()
    cmo = ta.CMO(np.asarray(close), timeperiod=14); cmo_val = float(cmo[-1]) if not np.isnan(cmo[-1]) else 0.0
    closed_vols = [vol for vol in volumes[:-1] if vol > 0]
    vratio = (closed_vols[-1] / np.mean(closed_vols[-50:])) if (closed_vols and np.mean(closed_vols[-50:]) > 0) else 0.0
    is_rej, bull_ratio = has_bullish_rejection_volume(klines, window=10)
    metrics = calculate_effort_result_metrics(close, volumes, window=20)
    
    prob, forecast_atr_mult = ml_forecast_probability_and_target(
        metrics["R"], metrics["C"], metrics["E"], bull_ratio, cmo_val, vratio, 
        argmin_gt_argmax, rsi, stoch_k, reg_div, hid_div, dbl_bottom, fib_zone, delta_div, sweep_rec
    )
    
    is_strong = linear_regression_dip(close, 0.01)
    current_price = close[-1]
    rejection = detect_rejection_patterns(klines, lookback=15)
    entry_exh = detect_entry_exhaustion(klines, current_price, zone_pct=0.015)
    circuit = dynamic_360_cycle_forecast(close, min_cycle=15, max_cycle=150)

    return (symbol, cmo_val, vratio, is_strong, bull_ratio, prob, rejection, entry_exh, circuit, forecast_atr_mult, new_feats)


class ProgressTracker:
    def __init__(self, total, label):
        self.total = total
        self.label = label
        self.completed = 0
        self.passed = 0
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
            return f"\r{self.label}: {self.completed}/{self.total} | ✓{self.passed} | {r:.1f}/s | ETA: {rem:.0f}s"


def run_tf_filter(trader, symbols, interval, max_workers=20):
    passed = []; tracker = ProgressTracker(len(symbols), f"{interval} filter"); print(f"Running {interval} filter on {len(symbols)} pairs...")
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
    print(f"\r{tracker.get_stats()}" + " " * 20); 
    return passed

def run_1m_filter(trader, symbols, max_workers=15):
    results = []; tracker = ProgressTracker(len(symbols), "1m filter")
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
# LIVE BACKTESTER (ZERO RAM BLOAT)
# ==========================================
def verify_live_predictions(trader: Trader, predictions: List[Dict]) -> str:
    wins, losses = 0, 0
    for pred in predictions:
        if pred['status'] == 'PENDING':
            if time.time() - pred['time'] > 900: 
                pred['status'] = 'LOSS'
            else:
                curr = trader.get_klines(pred['sym'], '1m', limit=1, return_raw=False)
                if curr and curr[-1] >= pred['target']: 
                    pred['status'] = 'WIN'
        if pred['status'] == 'WIN': wins += 1
        elif pred['status'] == 'LOSS': losses += 1
            
    if len(predictions) > 20: 
        del predictions[0:len(predictions) - 20]
        
    total = wins + losses
    return f"✅W:{wins} ❌L:{losses} 📈Live Acc:{(wins/total)*100:.0f}%" if total > 0 else "⏳ Awaiting backtest verification..."


# ==========================================
# OUTPUT FORMATTER
# ==========================================
def format_sr_output(symbol, sr, current_price, cmo_val, vratio, bull_ratio, ml_prob, tf_volumes, rejection, entry_exh, circuit, forecast_price, new_feats, live_bt_stats):
    vb = sr['vol_bias']; avg_r = sr['avg_range']; bp_pct = vb * 100; bias_lbl = "🟢 BULLISH" if vb > 0.55 else ("🔴 BEARISH" if vb < 0.45 else "⚪ NEUTRAL")
    W = 74
    print("\n" + "=" * W)
    print(f"  ★  STRUCTURAL RANGE S/R  —  {symbol}  ★")
    print("=" * W)
    
    cycle_target = circuit.get('cycle_target', 0)
    primary_target = cycle_target if (circuit.get('direction') == 'UP' and cycle_target > current_price) else forecast_price
    target_label = "🌀 CYCLIC" if (circuit.get('direction') == 'UP' and cycle_target > current_price) else "🧠 ML ATR"
    
    print(f"  🎯 {target_label} FORECAST: {primary_target:.10f}")
    print(f"  🧠 ML Confidence: {ml_prob*100:.1f}%")
    print(f"  📊 Live Backtest: {live_bt_stats}")
    print("-" * W)
    print(f"  Entry Price    : {current_price:.10f}")
    print(f"  1m CMO         : {cmo_val:+.2f}  |  Vol Ratio: x{vratio:.2f}  |  Bull Rej Vol: {bull_ratio*100:.1f}%")
    print(f"  1m Vol Bias    : {bias_lbl}  ({bp_pct:.1f}% bull / {100-bp_pct:.1f}% bear)  |  Avg Range: {avg_r:.4f}%")
    
    print("-" * W)
    print("  🧬 ML NEURAL FEATURE ACTIVATIONS:")
    feat_str = "  ".join([f"{k.upper()}:{int(v)}" for k, v in new_feats.items() if v > 0])
    print(f"  {feat_str if feat_str else 'No strong micro-features triggered'}")

    print("-" * W)
    print("  🔎  ENTRY-ZONE EXHAUSTION"); tag = "🟢 EXHAUSTED" if entry_exh.get('exhausted') else "⚪ NOT CONFIRMED"
    print(f"  {tag}  ({entry_exh.get('confidence',0)*100:.0f}%)  |  {entry_exh.get('detail','')}")

    print("-" * W)
    print("  🕯  REJECTION PATTERN STACK"); print(f"  Score: {rejection.get('rejection_score',0)}  |  {rejection.get('detail','')}")
    if rejection.get('talib_hits'): print(f"    TA-Lib: {', '.join(rejection['talib_hits'].keys())}")
    extras = []; 
    if rejection.get('pin_bar'): extras.append('PIN_BAR')
    if rejection.get('tweezer_bottom'): extras.append('TWEEZER_BOTTOM')
    if rejection.get('wyckoff_spring'): extras.append('WYCKOFF_SPRING')
    if extras: print(f"    Custom: {', '.join(extras)}")

    print("-" * W)
    print("  🌀  DYNAMIC 360° CYCLIC CIRCUIT (FFT EXTRACTED)")
    c_dir = circuit.get('direction', 'UNKNOWN')
    c_tag = "🟢 UP CYCLE" if c_dir == "UP" else ("🔴 DOWN CYCLE" if c_dir == "DOWN" else "⚪ FLAT/UNKNOWN")
    print(f"  State: {c_tag}")
    print(f"  Phase: {circuit.get('phase_deg', 0):.1f}° / 360°  |  Wavelength: {circuit.get('cycle_length', 0)} bars")
    print(f"  Wave Amplitude: {circuit.get('amplitude', 0):.10f}")
    print(f"  {circuit.get('detail', '')}")

    print("-" * W); print("  📊  VOLUME BREAKDOWN BY TIMEFRAME")
    for tf, vd in tf_volumes.items():
        bar_len, bull_len = 30, int(vd['bull_pct'] / 100 * 30); bar = "🟢" * bull_len + "🔴" * (30 - bull_len)
        print(f"  {tf:>4s}  [{bar}]  Bull: {vd['bull_pct']:.1f}%")

    all_signals = []
    if entry_exh.get('exhausted'): all_signals.append(f"Entry-zone exhaustion ({entry_exh['confidence']*100:.0f}%)")
    if rejection.get('rejection_score', 0) >= 2: all_signals.append(f"Rejection stack ({rejection['rejection_score']} hits)")
    if c_dir == "UP": all_signals.append(f"360° Cycle UP phase ({circuit.get('phase_deg', 0):.0f}°)")
    if new_feats.get('sweep') == 1.0: all_signals.append("1m Liquidity Sweep Recovery")
    if new_feats.get('delta_div') == 1.0: all_signals.append("1m Delta Divergence (Absorption)")
    if new_feats.get('reg_div') == 1.0: all_signals.append("1m Regular RSI Divergence")

    for lb_data in sr['lookbacks']:
        lb, ext, exh_h, exh_l = lb_data['lookback'], lb_data['extremes'], lb_data['exh_high'], lb_data['exh_low']
        rng_pct, pos, min_d = ext['range_pct'], ext['position'], lb_data['min_dist']
        print("\n" + "─" * W); print(f"  📐  LOOKBACK: {lb} BARS (MAX DATA SLICE)"); print("─" * W)
        print(f"  High: {ext['high']:.10f} ({ext['high_age']} bars ago)  |  Low: {ext['low']:.10f} ({ext['low_age']} bars ago)")
        print(f"  True Range: {rng_pct:.3f}%  |  More Recent: {ext['mr_label']}")
        if ext['more_recent'] == 'ARGMIN': all_signals.append(f"[{lb}] ARGMIN more recent")
        if exh_l.get('exhaustion', 0) > 0.5 and pos < 0.5: all_signals.append(f"[{lb}] Selling exhaustion ({exh_l['pattern']})")
        
        grid = lb_data['grid']
        if grid:
            print(f"\n  {'Level':<8} {'Price':>14} {'Dist%':>8} {'Bull%':>6} {'Tch':>4} {'Rej':>4} {'Exh':>5} {'Verdict':<10} {'St'}")
            print("  " + "─" * 68)
            for g in grid:
                st, m = g.get('status', '?'), "·" if g.get('status') == 'TOO_CLOSE' else ("►" if g.get('direction') == 'UP' else "◄")
                print(f"  {m}{g['label']:<7} {g['price']:>14.8f} {g['dist_pct']:>+7.3f}% {g['bull_pct']*100:>5.0f}% {g['touches']:>4} {g['total_rej']:>4} {g['exhaustion']:>4.2f} {g['verdict']:<10} {st}")

        up, dn = lb_data['targets_up'], lb_data['targets_down']
        if up:
            print(f"\n  📈  RESISTANCE TARGETS\n")
            for i, t in enumerate(up, 1):
                bar = "█" * int(t['score'] * 10) + "░" * (10 - int(t['score'] * 10)); vi = "🔴" if t['verdict'] == "RESISTANCE" else ("🟢" if t['verdict'] == "SUPPORT" else "⚪")
                print(f"  T{i} {t['label']:5s} {t['price']:.10f} ({t['dist_pct']:+.3f}%) ETA: {t['eta']}\n       [{bar}] {t['score']:.2f} {vi} {t['verdict']} BullVol:{t['bull_pct']*100:.0f}% Tch:{t['touches']} Rej:{t['rejections']}\n")
        if dn:
            print(f"  📉  SUPPORT LEVELS\n")
            for i, t in enumerate(dn, 1):
                bar = "█" * int(t['score'] * 10) + "░" * (10 - int(t['score'] * 10)); vi = "🟢" if t['verdict'] == "SUPPORT" else ("🔴" if t['verdict'] == "RESISTANCE" else "⚪")
                print(f"  S{i} {t['label']:5s} {t['price']:.10f} ({t['dist_pct']:+.3f}%) ETA: {t['eta']}\n       [{bar}] {t['score']:.2f} {vi} {t['verdict']} BullVol:{t['bull_pct']*100:.0f}% Tch:{t['touches']} Rej:{t['rejections']}\n")

    print("=" * W); print("  ⚡  CONSOLIDATED TRADE BIAS"); print("=" * W)
    argmin_count = sum(1 for lb in sr['lookbacks'] if lb['extremes']['more_recent'] == 'ARGMIN')
    if argmin_count > 0: all_signals.append(f"ARGMIN dominant across lookbacks")
    if vb > 0.55: all_signals.append(f"Bullish vol bias ({vb*100:.0f}%)")
    if cmo_val < -50: all_signals.append(f"CMO oversold ({cmo_val:.0f})")
    if ml_prob > 0.65: all_signals.append(f"ML spike prob ({ml_prob*100:.0f}%)")

    best_up = best_dn = None
    for lb in reversed(sr['lookbacks']):
        if not best_up and lb['targets_up']: best_up = lb['targets_up'][0]
        if not best_dn and lb['targets_down']: best_dn = lb['targets_down'][0]
        if best_up and best_dn: break

    if best_up and best_dn:
        rr = abs(best_up['dist_pct']) / max(abs(best_dn['dist_pct']), 0.0001)
        print(f"\n  Best Target : {best_up['label']:5s} {best_up['price']:.10f} ({best_up['dist_pct']:+.3f}%) ETA: {best_up['eta']}")
        print(f"  Best Stop   : {best_dn['label']:5s} {best_dn['price']:.10f} ({best_dn['dist_pct']:+.3f}%)")
        print(f"  R:R         : {rr:.2f}x")
        if rr >= 1.5: all_signals.append(f"R:R favorable ({rr:.1f}x)")

    print(f"\n  Signals ({len(all_signals)}):")
    for s in all_signals: print(f"    ✅  {s}")
    print()
    if len(all_signals) >= 6: v = "✅  STRONG LONG  —  Full-stack confluence"
    elif len(all_signals) >= 4: v = "✅  LONG  —  Good alignment"
    elif len(all_signals) >= 2: v = "⏳  PROBABLE LONG"
    else: v = "⚪  NEUTRAL"
    print(f"  VERDICT : {v}"); 
    
    # ==========================================
    # FINAL SUMMARY BLOCK
    # ==========================================
    print("\n" + "╔" + "═" * W + "╗")
    print("║" + " " * W + "║")
    print(f"║  💰  CURRENT PRICE : {current_price:<20.10f}                      ║".ljust(W+1))
    if best_up:
        print(f"║  🎯  BEST TARGET  : {best_up['price']:<20.10f}                      ║".ljust(W+1))
        print(f"║  ⏱️  ESTIMATED ETA: {best_up['eta']:<35}                   ║".ljust(W+1))
    else:
        print(f"║  🎯  BEST TARGET  : N/A                                        ║".ljust(W+1))
        print(f"║  ⏱️  ESTIMATED ETA: N/A                                        ║".ljust(W+1))
    print("║" + " " * W + "║")
    print("╚" + "═" * W + "╝\n")
    
    return v, len(all_signals)


def print_rescan_banner(scan_count: int, reason: str):
    W = 78; print("\n" + "╔" + "═" * W + "╗\n║" + " " * W + "║\n║" + f"  🔄  RESCAN #{scan_count} INITIATED".ljust(W) + "║\n║" + " " * W + "║\n║" + f"  Reason: {reason}".ljust(W)[:W] + "║\n║" + " " * W + "║\n║" + "  ⏰  Clearing memory & fetching fresh data in 5s...".ljust(W) + "║\n║" + " " * W + "║\n╚" + "═" * W + "╝")

def print_scan_header(scan_count: int):
    W = 78; timestamp = time.strftime('%Y-%m-%d %H:%M:%S'); print("\n" + "╔" + "═" * W + "╗\n║" + " " * W + "║\n║" + f"  🔍  MTF SCANNER  —  SCAN #{scan_count}".ljust(W) + "║\n║" + f"  📅  {timestamp}".ljust(W) + "║\n║" + " " * W + "║\n║" + "  Engine: ML Neural Sigmoid + Dynamic 360° FFT Cycle Extraction".ljust(W) + "║\n║" + "  Action: WILL STOP AND CLOSE UPON FINDING DIP.".ljust(W) + "║\n║" + " " * W + "║\n╚" + "═" * W + "╝\n")


# ==========================================
# MAIN WITH WHILE LOOP RESCAN & GC CLEANUP
# ==========================================
def main():
    scan_count = 0
    RESCAN_INTERVAL_SECONDS = 5
    W = 78
    live_predictions = []
    
    print("=" * W)
    print("  🚀  MTF DIP SCANNER  —  FIND & SHUTDOWN MODE")
    print("=" * W + "\n")

    while True:
        scan_count += 1
        print_scan_header(scan_count)
        trader = Trader('credentials.txt')
        
        try:
            trading_pairs = trader.get_usdc_pairs()
            filtered1 = run_tf_filter(trader, trading_pairs, '2h', 20)
            if not filtered1: 
                reason = "No 2h dips"
                print(f"\n  ⚠️  {reason}")
                print_rescan_banner(scan_count, reason)
                del trader, trading_pairs, filtered1
                gc.collect()
                time.sleep(RESCAN_INTERVAL_SECONDS)
                continue
                
            filtered2 = run_tf_filter(trader, filtered1, '15m', 15)
            if not filtered2: 
                reason = "No 15m dips"
                print(f"\n  ⚠️  {reason}")
                print_rescan_banner(scan_count, reason)
                del trader, trading_pairs, filtered1, filtered2
                gc.collect()
                time.sleep(RESCAN_INTERVAL_SECONDS)
                continue
                
            filtered3 = run_tf_filter(trader, filtered2, '5m', 15)
            if not filtered3: 
                reason = "No 5m dips"
                print(f"\n  ⚠️  {reason}")
                print_rescan_banner(scan_count, reason)
                del trader, trading_pairs, filtered1, filtered2, filtered3
                gc.collect()
                time.sleep(RESCAN_INTERVAL_SECONDS)
                continue

            results_1m = run_1m_filter(trader, filtered3, 15)
            if not results_1m: 
                reason = "1m filter failed"
                print_rescan_banner(scan_count, reason)
                del trader, trading_pairs, filtered1, filtered2, filtered3, results_1m
                gc.collect()
                time.sleep(RESCAN_INTERVAL_SECONDS)
                continue

            confluent = [r for r in results_1m if r[3] is True and (r[6].get('rejection_score', 0) >= 1 or r[7].get('exhausted', False) or r[8].get('direction') == 'UP')]
            if confluent: 
                final = max(confluent, key=lambda x: (x[5], x[6].get('rejection_score', 0), -x[1]))
                mode = "STRONG + CONFLUENCE"
            else:
                strong = [r for r in results_1m if r[3] is True]
                if strong: 
                    final = max(strong, key=lambda x: (x[5], -x[1]))
                    mode = "STRONG + ML ENERGY"
                else: 
                    final = min(results_1m, key=lambda x: x[1])
                    mode = "FALLBACK"

            (sym, cmo_val, vratio, is_strong_1m, live_bull_ratio, ml_prob, rejection, entry_exh, circuit, forecast_atr_mult, new_feats) = final

            print("\n" + "-" * W)
            print(f"  SELECTED : {sym} via {mode}")
            print(f"  ML PROB  : {ml_prob*100:.2f}%")
            print("-" * W)

            tf_volumes = {tf: get_volume_breakdown(trader, sym, tf, limit=100) for tf in ['1m', '5m', '15m', '1h', '2h']}
            
            # ==========================================
            # TRIGGER MAXIMUM HISTORICAL FETCH FOR WINNER
            # ==========================================
            klines_1m = trader.get_max_klines(sym, '1m', max_candles=10000)
            
            if not klines_1m: 
                print_rescan_banner(scan_count, "No 1m klines")
                del trader, trading_pairs, filtered1, filtered2, filtered3, results_1m, tf_volumes
                gc.collect()
                time.sleep(RESCAN_INTERVAL_SECONDS)
                continue
            
            current_price = float(klines_1m[-1][4])
            
            atr_1m = float(ta.ATR(np.array([float(k[2]) for k in klines_1m], dtype='float64'), np.array([float(k[3]) for k in klines_1m], dtype='float64'), np.array([float(k[4]) for k in klines_1m], dtype='float64'), timeperiod=14)[-1])
            forecast_price = current_price + (atr_1m * forecast_atr_mult)

            live_bt_stats = verify_live_predictions(trader, live_predictions)
            
            actual_target = forecast_price
            if circuit.get('direction') == 'UP' and circuit.get('cycle_target', 0) > current_price:
                actual_target = circuit['cycle_target']
                
            if ml_prob > 0.6 and forecast_atr_mult > 0:
                live_predictions.append({'sym': sym, 'target': actual_target, 'entry': current_price, 'time': time.time(), 'status': 'PENDING'})

            sr = get_sr_targets(klines_1m, current_price)
            format_sr_output(sym, sr, current_price, cmo_val, vratio, live_bull_ratio, ml_prob, tf_volumes, rejection, entry_exh, circuit, forecast_price, new_feats, live_bt_stats)

            print("\n" + "╔" + "═" * 78 + "╗")
            print("║" + " " * 78 + "║")
            print("║" + "  🛑  MTF DIP FOUND & ANALYZED. BOT TASK COMPLETE.".ljust(78) + "║")
            print("║" + "  🧹  CLEANING MEMORY AND SHUTTING DOWN...".ljust(78) + "║")
            print("║" + " " * 78 + "║")
            print("╚" + "═" * 78 + "╝\n")
            
            del trading_pairs, filtered1, filtered2, filtered3, results_1m, tf_volumes, klines_1m, sr, final
            del live_predictions
            gc.collect()
            
            trader = None
            sys.exit(0)

        except SystemExit:
            raise  
        except Exception as e:
            print(f"\n⚠️  System error: {str(e)}")
        finally:
            if 'trader' in locals() and trader is not None:
                del trader
            gc.collect()
            time.sleep(RESCAN_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()