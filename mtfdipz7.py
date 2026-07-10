import gc
import os
import sys
from binance.client import Client
import numpy as np
import talib as ta
import time
from datetime import datetime
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
from scipy.signal import argrelextrema

MAX_CANDLES = 1200

def local_timestamp() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z').strip() or datetime.now().strftime('%Y-%m-%d %H:%M:%S')

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
        if len(lines) < 2: raise ValueError("credentials.txt must contain API key on line 1 and secret on line 2")
        self.client = Client(lines[0], lines[1])

    def get_usdc_pairs(self) -> List[str]:
        exchange_info = self.client.get_exchange_info()
        return [s['symbol'] for s in exchange_info['symbols'] if s['quoteAsset'] == 'USDC' and s['status'] == 'TRADING']

    def get_klines(self, symbol: str, interval: str, limit: int = 500, return_raw: bool = False, start_time: int = None, end_time: int = None):
        self.rate_limiter.acquire()
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        if start_time is not None: params['startTime'] = start_time
        if end_time is not None: params['endTime'] = end_time
        for attempt in range(3):
            try:
                klines = self.client.get_klines(**params)
                return klines if return_raw else [float(k[4]) for k in klines]
            except Exception as e:
                time.sleep(2 ** attempt * 2 if 'rate limit' in str(e).lower() else 0.5)
        return []

    def get_realtime_price(self, symbol: str) -> float:
        self.rate_limiter.acquire()
        for attempt in range(3):
            try:
                bt = self.client.get_orderbook_ticker(symbol=symbol)
                bid, ask = float(bt['bidPrice']), float(bt['askPrice'])
                if bid > 0 and ask > 0: return (bid + ask) / 2.0
                return float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            except Exception: time.sleep(0.3)
        return 0.0

    def get_max_klines(self, symbol: str, interval: str, max_candles: int = 10000, verbose: bool = False) -> list:
        MAX_PER_REQ = 1000
        all_klines, end_time = [], None
        if verbose: print(f"  ⏳ Fetching max historical data...", end="", flush=True)
        start_fetch_time = time.time()
        while len(all_klines) < max_candles:
            self.rate_limiter.acquire()
            params = {'symbol': symbol, 'interval': interval, 'limit': MAX_PER_REQ}
            if end_time is not None: params['endTime'] = end_time
            try:
                klines = self.client.get_klines(**params)
                if not klines: break
                all_klines = klines + all_klines
                end_time = int(klines[0][0]) - 1
            except Exception as e:
                if 'rate limit' in str(e).lower(): time.sleep(2)
                else: break
        final_klines = all_klines[-max_candles:]
        if verbose: print(f" Done! {len(final_klines)} candles in {time.time() - start_fetch_time:.1f}s")
        return final_klines

# ==========================================
# MACRO TREND CONTEXT
# ==========================================
def add_trend_context(close: np.ndarray, lookbacks: List[int] = [100, 200, 500]) -> Dict:
    n = len(close); context = {}
    for lb in lookbacks:
        if n < lb: continue
        slice_data = close[-lb:]
        sma_50 = np.mean(slice_data[-50:]) if len(slice_data) >= 50 else np.mean(slice_data)
        sma_200 = np.mean(slice_data) if len(slice_data) >= 200 else None
        recent_highs, older_highs = slice_data[-20:], slice_data[-40:-20] if len(slice_data) >= 40 else slice_data[:-20]
        if len(recent_highs) > 0 and len(older_highs) > 0:
            hh_hl = np.max(recent_highs) > np.max(older_highs) and np.min(recent_highs) > np.min(older_highs)
            lh_ll = np.max(recent_highs) < np.max(older_highs) and np.min(recent_highs) < np.min(older_highs)
        else: hh_hl = lh_ll = False
        if sma_200 and close[-1] > sma_200 and hh_hl: trend = "STRONG_UPTREND"
        elif sma_200 and close[-1] > sma_200: trend = "UPTREND"
        elif sma_200 and close[-1] < sma_200 and lh_ll: trend = "STRONG_DOWNTREND"
        elif sma_200 and close[-1] < sma_200: trend = "DOWNTREND"
        else: trend = "RANGING"
        context[f'{lb}bar'] = {'trend': trend, 'above_sma200': close[-1] > sma_200 if sma_200 else None, 'hh_hl': hh_hl}
    if any(c['trend'] in ['STRONG_UPTREND', 'UPTREND'] for c in context.values()): tradeability = "PULLBACK_BUY"
    elif any(c['trend'] == 'RANGING' for c in context.values()): tradeability = "RANGE_DIP_BUY"
    elif any(c['trend'] == 'DOWNTREND' and c.get('above_sma200') for c in context.values()): tradeability = "CAUTIOUS"
    else: tradeability = "AVOID"
    return {'context': context, 'tradeability': tradeability, 'tradeability_score': {"PULLBACK_BUY": 1.0, "RANGE_DIP_BUY": 0.7, "CAUTIOUS": 0.3, "AVOID": 0.0}[tradeability]}

# ==========================================
# DUMP MATURITY METRICS
# ==========================================
def calculate_dump_metrics(close: np.ndarray, window: int = 50) -> Dict:
    n = len(close)
    if n < window: return {'valid': False, 'drop_pct': 0}
    recent = close[-window:]
    drop_pct = (recent[0] - recent[-1]) / recent[0] * 100
    mid = len(recent) // 2
    first_half_drop = (recent[0] - recent[mid]) / recent[0] * 100
    second_half_drop = (recent[mid] - recent[-1]) / recent[mid] * 100
    half_window = mid if mid > 0 else 1
    acceleration = (second_half_drop - first_half_drop) / half_window
    dump_phase = "ACCELERATING" if acceleration < -0.1 else ("DECELERATING" if acceleration > 0.1 else "STEADY")
    returns = np.diff(recent) / recent[:-1]
    typical_move = np.std(returns) * 100
    z_score_drop = drop_pct / typical_move if typical_move > 0 else 0
    extendedness = "EXTREME" if z_score_drop > 4 else ("HIGH" if z_score_drop > 2.5 else ("MODERATE" if z_score_drop > 1.5 else "NORMAL"))
    return {'valid': True, 'drop_pct': drop_pct, 'acceleration': acceleration, 'dump_phase': dump_phase, 'z_score_drop': z_score_drop, 'extendedness': extendedness, 'favorable': dump_phase == "DECELERATING" and z_score_drop > 2.0}

# ==========================================
# ROBUST DIVERGENCES V2
# ==========================================
def detect_divergences_v2(highs: np.ndarray, lows: np.ndarray, rsi: np.ndarray, lookback: int = 50) -> Dict:
    order = 5
    p_lows_idx = argrelextrema(lows[-lookback:], np.less, order=order)[0]
    rsi_lows_idx = argrelextrema(rsi[-lookback:], np.less, order=order)[0]
    p_highs_idx = argrelextrema(highs[-lookback:], np.greater, order=order)[0]
    rsi_highs_idx = argrelextrema(rsi[-lookback:], np.greater, order=order)[0]
    reg_bull = hid_bull = reg_bear = hid_bear = 0.0
    if len(p_lows_idx) >= 2 and len(rsi_lows_idx) >= 2:
        p_l, r_l = lows[-lookback:][p_lows_idx[-2:]], rsi[-lookback:][rsi_lows_idx[-2:]]
        if len(p_l) == 2 and len(r_l) == 2:
            if p_l[0] > p_l[1] and r_l[0] < r_l[1]: reg_bull = 1.0 
            if p_l[0] < p_l[1] and r_l[0] > r_l[1]: hid_bull = 1.0 
    if len(p_highs_idx) >= 2 and len(rsi_highs_idx) >= 2:
        p_h, r_h = highs[-lookback:][p_highs_idx[-2:]], rsi[-lookback:][rsi_highs_idx[-2:]]
        if len(p_h) == 2 and len(r_h) == 2:
            if p_h[0] < p_h[1] and r_h[0] > r_h[1]: reg_bear = 1.0
            if p_h[0] > p_h[1] and r_h[0] < r_h[1]: hid_bear = 1.0
    return {'reg_bull_div': reg_bull, 'hid_bull_div': hid_bull, 'reg_bear_div': reg_bear, 'hid_bear_div': hid_bear}

# ==========================================
# TRADEABILITY & SPREAD CHECK
# ==========================================
def check_tradeability(trader: Trader, symbol: str, expected_profit_pct: float, min_profit: float = 0.3) -> Dict:
    trader.rate_limiter.acquire()
    try:
        ticker = trader.client.get_orderbook_ticker(symbol=symbol)
        bid, ask = float(ticker['bidPrice']), float(ticker['askPrice'])
        spread_pct = (ask - bid) / bid * 100
        slippage_est = spread_pct * 0.5
        total_cost_pct = (spread_pct + slippage_est) * 2
        net_profit_pct = expected_profit_pct - total_cost_pct
        return {'tradeable': net_profit_pct >= min_profit, 'spread_pct': spread_pct, 'total_cost_pct': total_cost_pct, 'net_profit_pct': net_profit_pct, 'detail': f"Spread: {spread_pct:.3f}% | Net: {net_profit_pct:.3f}%"}
    except Exception as e: return {'tradeable': False, 'detail': f"Error: {e}"}

# ==========================================
# POSITION SIZING
# ==========================================
def calculate_position_size(entry: float, stop: float, risk_pct: float = 0.01, balance: float = 1000) -> Dict:
    stop_dist = abs(entry - stop) / entry
    if stop_dist < 0.001: stop_dist = 0.001
    size_usd = (balance * risk_pct) / stop_dist
    size_usd = min(size_usd, balance * 0.10)
    return {'size_usd': size_usd, 'size_coins': size_usd / entry, 'stop_dist_pct': stop_dist * 100}

# ==========================================
# VOLUME BREAKDOWN
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
def linear_regression_dip(close: List[float], deviation: float = 0.01) -> bool:
    if len(close) < 20: return False
    x = np.arange(len(close))
    slope, intercept = np.polyfit(x, close, 1)
    trend = slope * x + intercept
    lower_band = trend * (1 - deviation)
    return close[-1] < lower_band[-1]

def sma_stack_gate(close: np.ndarray, fast: int = 12, slow: int = 56) -> Dict:
    n = len(close)
    if n < slow + 2: return {'pass': False, 'sma_fast': None, 'sma_slow': None, 'detail': 'insufficient bars'}
    sma_f, sma_s = ta.SMA(close, timeperiod=fast), ta.SMA(close, timeperiod=slow)
    c, f, s = float(close[-1]), float(sma_f[-1]), float(sma_s[-1])
    if np.isnan(f) or np.isnan(s): return {'pass': False, 'sma_fast': None, 'sma_slow': None, 'detail': 'sma warmup'}
    ok = (c < f) and (f < s)
    return {'pass': bool(ok), 'sma_fast': f, 'sma_slow': s, 'close': c, 'detail': f"close {c:.6g} {'<' if c < f else '>='} SMA{fast} {f:.6g} {'<' if f < s else '>='} SMA{slow} {s:.6g}"}

def regression_channel_dip_gate(close: np.ndarray, band_mult: float = 2.0) -> Dict:
    n = len(close)
    if n < 50: return {'valid': False, 'pass': False, 'detail': 'insufficient bars'}
    x = np.arange(n, dtype='float64')
    slope, intercept = np.polyfit(x, close, 1)
    trend = slope * x + intercept
    resid = close - trend
    std = float(np.std(resid))
    if std <= 0: return {'valid': False, 'pass': False, 'detail': 'zero std'}
    argmin_i, argmax_i = int(np.argmin(close)), int(np.argmax(close))
    argmin_zscore = (close[argmin_i] - trend[argmin_i]) / std
    argmax_zscore = (close[argmax_i] - trend[argmax_i]) / std
    current_zscore = (close[-1] - trend[-1]) / std
    gate_pass = (argmin_zscore < -1.5 and argmax_zscore > 1.5 and argmin_i > argmax_i and current_zscore < 0 and abs(current_zscore - argmin_zscore) < 0.5)
    return {'valid': True, 'pass': gate_pass, 'argmin_zscore': argmin_zscore, 'argmax_zscore': argmax_zscore, 'current_zscore': current_zscore, 'detail': f"argmin_z={argmin_zscore:.2f} argmax_z={argmax_zscore:.2f} curr_z={current_zscore:.2f}"}

def fear_greed_wave(close: np.ndarray) -> Dict:
    n = len(close)
    if n < 30: return {'valid': False}
    argmin_i, argmax_i = int(np.argmin(close)), int(np.argmax(close))
    argmin_more_recent = argmin_i > argmax_i
    half_period = max(abs(argmin_i - argmax_i), 5)
    period = half_period * 2.0
    lo, hi = float(np.min(close)), float(np.max(close))
    amplitude = (hi - lo) / 2.0
    center = (hi + lo) / 2.0
    if amplitude <= 0: return {'valid': False}
    idx = np.arange(n, dtype='float64')
    theta = 2 * np.pi * (idx - argmin_i) / period - np.pi / 2
    sine_wave = center + amplitude * np.sin(theta)
    fg_norm = float(np.clip((close[-1] - center) / amplitude, -1.0, 1.0))
    fg_score = (fg_norm + 1.0) / 2.0 * 100.0
    turning_up = bool(sine_wave[-1] > sine_wave[-2] and sine_wave[-2] <= sine_wave[-3]) if n >= 3 else False
    energy_exhausted = bool(fg_score < 25.0 and turning_up)
    return {'valid': True, 'period': period, 'amplitude': amplitude, 'center': center, 'argmin_more_recent': argmin_more_recent, 'fg_score': fg_score, 'turning_up': turning_up, 'energy_exhausted': energy_exhausted, 'sine_last': float(sine_wave[-1]), 'label': ('EXTREME FEAR' if fg_score < 20 else 'FEAR' if fg_score < 40 else 'NEUTRAL' if fg_score < 60 else 'GREED' if fg_score < 80 else 'EXTREME GREED')}

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
    ca = np.array(close[-window:], dtype='float64')
    va = np.array(volumes[-window:], dtype='float64')
    dp = abs(ca[-1] - ca[0])
    tv = np.sum(va)
    eps = 1e-9
    return {"R": tv / (dp + eps), "C": tv / (np.std(ca) + eps), "E": tv / ((dp * window) + eps)}

def ml_forecast_probability_and_target(R, C, E, bull_ratio, cmo, vratio, argmin_gt, rsi, stoch, reg_div, hid_div, dbl_bot, fib_z, delta_d, sweep_r):
    rsi_score = np.clip((40 - rsi) / 40, 0, 1)
    stoch_score = np.clip((30 - stoch) / 30, 0, 1)
    cmo_score = np.clip(-cmo / 100, 0, 1)
    vratio_score = np.clip((vratio - 1) / 4, 0, 1)
    features = {'argmin_gt': argmin_gt, 'rsi': rsi_score, 'stoch': stoch_score, 'reg_div': reg_div, 'hid_div': hid_div, 'dbl_bot': dbl_bot, 'fib_z': fib_z, 'delta_d': delta_d, 'sweep_r': sweep_r, 'bull_ratio': np.clip(bull_ratio, 0, 1), 'cmo': cmo_score, 'vratio': vratio_score}
    weights = {'argmin_gt': 0.15, 'rsi': 0.10, 'stoch': 0.08, 'reg_div': 0.12, 'hid_div': 0.08, 'dbl_bot': 0.05, 'fib_z': 0.05, 'delta_d': 0.12, 'sweep_r': 0.12, 'bull_ratio': 0.08, 'cmo': 0.03, 'vratio': 0.02}
    raw_score = sum(features[k] * weights[k] for k in features)
    probability = 1 / (1 + np.exp(-(raw_score - 0.45) * 10))
    forecast_mult = 2.5 if probability > 0.80 else (1.8 if probability > 0.65 else (1.0 if probability > 0.50 else 0.0))
    return probability, forecast_mult

# ==========================================
# REJECTION & EXHAUSTION
# ==========================================
def detect_rejection_patterns(raw_klines: list, lookback: int = 15) -> Dict:
    if not raw_klines or len(raw_klines) < 30: return {'talib_hits': {}, 'pin_bar': False, 'tweezer_bottom': False, 'rejection_score': 0, 'detail': 'insufficient bars'}
    o = np.array([float(k[1]) for k in raw_klines], dtype='float64')
    h = np.array([float(k[2]) for k in raw_klines], dtype='float64')
    l = np.array([float(k[3]) for k in raw_klines], dtype='float64')
    c = np.array([float(k[4]) for k in raw_klines], dtype='float64')
    pattern_fns = {'HAMMER': ta.CDLHAMMER, 'INV_HAMMER': ta.CDLINVERTEDHAMMER, 'BULL_ENGULFING': ta.CDLENGULFING, 'PIERCING': ta.CDLPIERCING, 'MORNING_STAR': ta.CDLMORNINGSTAR, 'MORNING_DOJI_STAR': ta.CDLMORNINGDOJISTAR, 'DRAGONFLY_DOJI': ta.CDLDRAGONFLYDOJI, 'THREE_WHITE_SOLDIERS': ta.CDL3WHITESOLDIERS, 'HARAMI': ta.CDLHARAMI, 'BELT_HOLD': ta.CDLBELTHOLD, 'TAKURI': ta.CDLTAKURI, 'HOMING_PIGEON': ta.CDLHOMINGPIGEON, 'MAT_HOLD': ta.CDLMATHOLD}
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
    return {'talib_hits': hits, 'pin_bar': pin_bar, 'tweezer_bottom': tweezer, 'wyckoff_spring': spring, 'rejection_score': score, 'detail': f"{score} rejection pattern(s) confirmed"}

def detect_entry_exhaustion(raw_klines: list, current_price: float, zone_pct: float = 0.015, min_bars: int = 8) -> Dict:
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
    return {'exhausted': exhausted, 'confidence': confidence, 'vol_ratio': vol_ratio, 'coiling': coiling, 'bull_ratio_in_zone': bull_ratio_in_zone, 'bars_in_zone': len(zv), 'detail': f"Vol {vol_ratio:.0%} of earlier pace, coiling={coiling}"}

def dynamic_360_cycle_forecast(close: List[float], min_cycle: int = 20, max_cycle: int = 500) -> Dict:
    n = len(close)
    if n < max_cycle + 50: return {'direction': 'UNKNOWN', 'phase_deg': 0, 'cycle_length': 0, 'cycle_target': 0, 'detail': 'Insufficient data', 'confidence': 0}
    detrend_window = min(max_cycle // 2, n // 3)
    sma = np.convolve(close, np.ones(detrend_window)/detrend_window, mode='valid')
    detrended = np.array(close[-len(sma):]) - sma
    if np.std(detrended) < 1e-10: return {'direction': 'FLAT', 'phase_deg': 0, 'cycle_length': 0, 'cycle_target': 0, 'detail': 'No cyclical variance', 'confidence': 0}
    window = np.hanning(len(detrended))
    detrended_windowed = detrended * window
    fft_vals = np.fft.rfft(detrended_windowed)
    fft_freqs = np.fft.rfftfreq(len(detrended_windowed), d=1.0)
    power = np.abs(fft_vals) ** 2
    valid_idx = (fft_freqs > 1.0/max_cycle) & (fft_freqs < 1.0/min_cycle)
    if not np.any(valid_idx): return {'direction': 'UNKNOWN', 'phase_deg': 0, 'cycle_length': 0, 'cycle_target': 0, 'detail': 'No valid cycle', 'confidence': 0}
    filtered_power = power.copy()
    filtered_power[~valid_idx] = 0
    dominant_idx = np.argmax(filtered_power)
    dominant_freq = fft_freqs[dominant_idx]
    cycle_length = int(round(1.0 / dominant_freq))
    total_power = np.sum(filtered_power[valid_idx])
    peak_power = power[dominant_idx]
    snr = peak_power / (total_power - peak_power + 1e-10)
    confidence = min(1.0, snr / 10)
    phase_rad = np.angle(fft_vals[dominant_idx])
    phase_deg = float(np.degrees(phase_rad)) % 360
    amplitude = np.abs(fft_vals[dominant_idx]) / np.sum(window) * 2
    if 0 <= phase_deg < 180:
        direction = "UP"
        gain_to_peak = amplitude * (1 - np.sin(phase_rad))
        cycle_target = close[-1] + max(0, gain_to_peak)
    else:
        direction = "DOWN"
        drop_to_trough = amplitude * (1 + np.sin(phase_rad))
        cycle_target = close[-1] - max(0, drop_to_trough)
    return {'direction': direction, 'phase_deg': phase_deg, 'cycle_length': cycle_length, 'cycle_target': float(cycle_target), 'amplitude': float(amplitude), 'confidence': confidence, 'detail': f"Phase {phase_deg:.1f}° | {direction} | {cycle_length} bars | Conf: {confidence:.0%}"}

# ==========================================
# STRUCTURAL RANGE ENGINE
# ==========================================
def get_structural_extremes(close: np.ndarray, highs: np.ndarray, lows: np.ndarray, lookback: int) -> Dict:
    n = len(close); start = max(0, n - lookback); c, h, l = close[start:], highs[start:], lows[start:]; sl = len(c)
    amax_i, amin_i = int(np.argmax(c)), int(np.argmin(c))
    g_high, g_low = float(c[amax_i]), float(c[amin_i])
    high_age, low_age = sl - amax_i, sl - amin_i
    more_recent = "ARGMIN" if low_age < high_age else ("ARGMAX" if high_age < low_age else "EQUAL")
    rng, rng_pct = g_high - g_low, ((g_high - g_low) / g_low * 100) if g_low > 0 else 0
    pos = (close[-1] - g_low) / rng if rng > 0 else 0.5
    return {'high': g_high, 'low': g_low, 'high_age': high_age, 'low_age': low_age, 'more_recent': more_recent, 'range_size': rng, 'range_pct': rng_pct, 'position': pos}

def build_fib_grid(extremes: Dict, current_price: float) -> List[Dict]:
    lo, hi, rng = extremes['low'], extremes['high'], extremes['range_size']
    if rng <= 0: return []
    return [{'price': lo + rng * f, 'fib': f, 'label': l, 'dist_pct': (lo + rng*f - current_price) / current_price * 100, 'direction': 'UP' if lo+rng*f > current_price else 'DOWN'} for f, l in [(0.0,"ARGMIN"),(0.236,"F236"),(0.382,"F382"),(0.5,"F500"),(0.618,"F618"),(0.786,"F786"),(1.0,"ARGMAX")]]

def volume_profile_at_level(level_price: float, raw_klines: list, tolerance: float) -> Dict:
    bull_vol = bear_vol = 0.0; touches = 0; lo, hi = level_price - tolerance, level_price + tolerance
    for k in raw_klines:
        o, h, l, c, v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
        if l <= hi and h >= lo:
            touches += 1
            if v > 0:
                if c >= o: bull_vol += v
                else: bear_vol += v
    total = bull_vol + bear_vol; bp = bull_vol / total if total > 0 else 0.5
    verdict = "SUPPORT" if bp > 0.58 else ("RESISTANCE" if bp < 0.42 else "NEUTRAL")
    return {'bull_pct': bp, 'total_volume': total, 'touches': touches, 'verdict': verdict}

def get_sr_targets(raw_klines: list, current_price: float) -> Dict:
    if len(raw_klines) < 100: return {'lookbacks': [], 'vol_bias': 0.5, 'avg_range': 0}
    highs, lows, closes, volumes = np.array([float(k[2]) for k in raw_klines]), np.array([float(k[3]) for k in raw_klines]), np.array([float(k[4]) for k in raw_klines]), np.array([float(k[5]) for k in raw_klines])
    avg_range = float(np.mean(((highs - lows) / (closes + 1e-12) * 100.0)[-50:]))
    rec = raw_klines[-21:-1]
    bv = sum(float(k[5]) for k in rec if float(k[4]) >= float(k[1]) and float(k[5]) > 0)
    bear_v = sum(float(k[5]) for k in rec if float(k[4]) < float(k[1]) and float(k[5]) > 0)
    tv = bv + bear_v; vol_bias = bv / tv if tv > 0 else 0.5
    n = len(raw_klines)
    lookbacks_data = []
    for lb in [n // 4, n // 2, n]:
        if lb >= 100:
            ext = get_structural_extremes(closes, highs, lows, lb)
            grid = build_fib_grid(ext, current_price)
            tol = max(ext['range_size'] * 0.025, avg_range / 100 * current_price * 1.5)
            for g in grid: g.update(volume_profile_at_level(g['price'], raw_klines[-lb:], tol))
            lookbacks_data.append({'lookback': lb, 'extremes': ext, 'grid': grid, 'targets_up': [g for g in grid if g['direction']=='UP'], 'targets_down': [g for g in grid if g['direction']=='DOWN']})
    return {'lookbacks': lookbacks_data, 'vol_bias': vol_bias, 'avg_range': avg_range}

# ==========================================
# CONCURRENT FILTER FUNCTIONS
# ==========================================
def check_tf_dip(trader, symbol, interval):
    raw = trader.get_max_klines(symbol, interval, max_candles=MAX_CANDLES)
    if not raw or len(raw) < 60: return (symbol, False, False)
    close = np.array([float(k[4]) for k in raw], dtype='float64')
    dip_ok = linear_regression_dip(close.tolist(), 0.01)
    sma_gate = sma_stack_gate(close, fast=12, slow=56)
    return (symbol, bool(dip_ok and sma_gate['pass']), sma_gate['pass'])

def check_1m_final(trader, symbol):
    klines = trader.get_max_klines(symbol, '1m', max_candles=MAX_CANDLES)
    empty_adv = {'trend_ctx': {'tradeability': 'NA'}, 'dump': {'valid': False, 'drop_pct': 0}, 'tradeability': {'tradeable': False, 'detail': 'No data'}, 'divs': {}, 'position_size': {}}
    empty = (symbol, 0.0, 0.0, False, 0.0, 0.0, {}, {}, {}, 0.0, {}, {}, {}, {}, 0.0, empty_adv)
    if not klines or len(klines) < 60: return empty
    c = np.array([float(k[4]) for k in klines], dtype='float64')
    h = np.array([float(k[2]) for k in klines], dtype='float64')
    l = np.array([float(k[3]) for k in klines], dtype='float64')
    v = np.array([float(k[5]) for k in klines], dtype='float64')
    o = np.array([float(k[1]) for k in klines], dtype='float64')
    trend_ctx = add_trend_context(c)
    if trend_ctx['tradeability'] == 'AVOID':
        return (symbol, 0.0, 0.0, False, 0.0, 0.0, {}, {}, {}, 0.0, {}, {}, {}, {}, 0.0, {'trend_ctx': trend_ctx, 'dump': {'valid': False, 'drop_pct': 0}, 'tradeability': {'tradeable': False, 'detail': 'Strong downtrend - AVOIDED'}, 'divs': {}, 'position_size': {}})
    dump_met = calculate_dump_metrics(c)
    idx_min, idx_max = np.argmin(l), np.argmax(h)
    argmin_gt_argmax = 1.0 if idx_min > idx_max else 0.0
    rsi = float(ta.RSI(c, timeperiod=14)[-1])
    stoch_k, _ = ta.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowd_period=3)
    stoch_k = float(stoch_k[-1])
    rsi_vals = ta.RSI(c, timeperiod=14)
    divs = detect_divergences_v2(h, l, rsi_vals)
    dbl_bottom = 0.0
    p_lows_idx = argrelextrema(l[-30:], np.less, order=3)[0]
    if len(p_lows_idx) >= 2:
        low1, low2 = l[-30:][p_lows_idx[-2]], l[-30:][p_lows_idx[-1]]
        if abs(low1 - low2) <= ((np.mean(h[-30:]) - np.mean(l[-30:])) * 0.02) and c[-1] > max(low1, low2): dbl_bottom = 1.0
    fib_zone = 0.0; rng = np.max(h[-40:]) - np.min(l[-40:])
    if rng > 0 and np.max(h[-40:]) - (rng * 0.786) <= c[-1] <= np.max(h[-40:]) - (rng * 0.618): fib_zone = 1.0
    deltas = np.zeros(len(klines))
    for i in range(len(klines)):
        rng_k = h[i] - l[i]
        if rng_k > 0: deltas[i] = v[i] * (abs(c[i] - o[i]) / rng_k) * (1 if c[i] > o[i] else -1)
    cum_delta = np.cumsum(deltas); pl_idx = np.argmin(l[-20:])
    delta_div = 1.0 if (l[-1] < l[pl_idx] and cum_delta[-1] > cum_delta[pl_idx]) else 0.0
    sl_20 = np.min(l[-21:-1]); sweep_rec = 1.0 if (l[-1] < sl_20 and c[-1] > sl_20) else 0.0
    new_feats = {'argmin': argmin_gt_argmax, 'rsi': rsi, 'stoch': stoch_k, 'reg_div': divs['reg_bull_div'], 'hid_div': divs['hid_bull_div'], 'dbl_bot': dbl_bottom, 'fib': fib_zone, 'delta_div': delta_div, 'sweep': sweep_rec}
    close_list = c.tolist(); volumes = v.tolist()
    cmo_val = float(ta.CMO(np.asarray(close_list), timeperiod=14)[-1]) if not np.isnan(ta.CMO(np.asarray(close_list), timeperiod=14)[-1]) else 0.0
    closed_vols = [vol for vol in volumes[:-1] if vol > 0]
    vratio = (closed_vols[-1] / np.mean(closed_vols[-50:])) if (closed_vols and np.mean(closed_vols[-50:]) > 0) else 0.0
    is_rej, bull_ratio = has_bullish_rejection_volume(klines, window=10)
    metrics = calculate_effort_result_metrics(close_list, volumes, window=20)
    prob, forecast_atr_mult = ml_forecast_probability_and_target(metrics["R"], metrics["C"], metrics["E"], bull_ratio, cmo_val, vratio, argmin_gt_argmax, rsi, stoch_k, divs['reg_bull_div'], divs['hid_bull_div'], dbl_bottom, fib_zone, delta_div, sweep_rec)
    sma_gate = sma_stack_gate(c, fast=12, slow=56)
    reg_channel = regression_channel_dip_gate(c, band_mult=2.0)
    fg_wave = fear_greed_wave(c)
    realtime_price = trader.get_realtime_price(symbol)
    if not realtime_price or realtime_price <= 0: realtime_price = float(c[-1])
    linreg_dip = linear_regression_dip(close_list, 0.01)
    mandatory_1m = bool(sma_gate.get('pass', False) and reg_channel.get('pass', False) and fg_wave.get('valid', False) and fg_wave.get('argmin_more_recent', False) and realtime_price < sma_gate.get('sma_fast', float('inf')))
    is_strong = bool(linreg_dip and mandatory_1m)
    rejection = detect_rejection_patterns(klines, lookback=15)
    if fg_wave.get('valid') and fg_wave.get('energy_exhausted'): rejection = {**rejection, 'rejection_score': rejection.get('rejection_score', 0) + 1}
    entry_exh = detect_entry_exhaustion(klines, realtime_price, zone_pct=0.015)
    circuit = dynamic_360_cycle_forecast(close_list, min_cycle=15, max_cycle=min(500, len(close_list) - 20))
    avg_range_pct = float(np.mean((h[-20:] - l[-20:]) / c[-20:] * 100)) if len(c) >= 20 else 0.5
    expected_profit = forecast_atr_mult * avg_range_pct
    tradeability = check_tradeability(trader, symbol, expected_profit) if is_strong else {'tradeable': False, 'detail': 'Failed gates'}
    stop_price = min(l[-20:]) * 0.999 if len(l) >= 20 else realtime_price * 0.985
    pos_size = calculate_position_size(realtime_price, stop_price) if is_strong else {}
    advanced_metrics = {'trend_ctx': trend_ctx, 'dump': dump_met, 'divs': divs, 'tradeability': tradeability, 'position_size': pos_size}
    return (symbol, cmo_val, vratio, is_strong, bull_ratio, prob, rejection, entry_exh, circuit, forecast_atr_mult, new_feats, sma_gate, reg_channel, fg_wave, realtime_price, advanced_metrics)

class ProgressTracker:
    def __init__(self, total, label): self.total, self.label, self.completed, self.passed, self.lock, self.start_time = total, label, 0, 0, Lock(), time.time()
    def update(self, passed=False):
        with self.lock: self.completed += 1
        if passed: self.passed += 1
    def get_stats(self):
        with self.lock: e = time.time() - self.start_time; r = self.completed / e if e > 0 else 0; rem = (self.total - self.completed) / r if r > 0 else 0
        return f"\r{self.label}: {self.completed}/{self.total} | ✓{self.passed} | {r:.1f}/s | ETA: {rem:.0f}s"

def run_tf_filter(trader, symbols, interval, max_workers=20):
    passed = []; tracker = ProgressTracker(len(symbols), f"{interval} filter"); print(f"Running {interval} filter on {len(symbols)} pairs...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(check_tf_dip, trader, s, interval): s for s in symbols}
        for f in as_completed(futures):
            try:
                sym, ok, _ = f.result()
                if ok: passed.append(sym)
                tracker.update(passed=ok)
                print(tracker.get_stats(), end="", flush=True)
            except: tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20); return passed

def run_1m_filter(trader, symbols, max_workers=15):
    results = []; tracker = ProgressTracker(len(symbols), "1m analysis")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(check_1m_final, trader, s): s for s in symbols}
        for f in as_completed(futures):
            try: 
                res = f.result()
                results.append(res)
                tracker.update(passed=res[3]) # Tracks strict passes just for stats
                print(tracker.get_stats(), end="", flush=True)
            except: tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20)
    return results

# ==========================================
# OUTPUT FORMATTER
# ==========================================
def format_sr_output(symbol, sr, current_price, cmo_val, vratio, bull_ratio, ml_prob, tf_volumes, rejection, entry_exh, circuit, forecast_price, new_feats, live_bt_stats, sma_gate=None, reg_channel=None, fg_wave=None, adv_metrics=None):
    if not adv_metrics: adv_metrics = {}
    W = 74
    print("\n" + "=" * W)
    print(f"  ★  STRUCTURAL RANGE S/R  —  {symbol}  ★")
    print(f"  🕒  Local time: {local_timestamp()}")
    print("=" * W)
    t_ctx = adv_metrics.get('trend_ctx', {})
    print(f"  🧭  MACRO TREND CONTEXT: {t_ctx.get('tradeability', 'N/A')} (Score: {t_ctx.get('tradeability_score', 0):.1f})")
    dump = adv_metrics.get('dump', {})
    if dump.get('valid'):
        dump_tag = "🟢 FAVORABLE" if dump.get('favorable') else ("🔴 DANGEROUS" if dump.get('dump_phase') == 'ACCELERATING' else "⚪ NEUTRAL")
        print(f"  📉  DUMP MATURITY: {dump['dump_phase']} | Extended: {dump['extendedness']} (Z-Score: {dump['z_score_drop']:.1f}) {dump_tag}")
    trade = adv_metrics.get('tradeability', {})
    trade_tag = "🟢 TRADEABLE" if trade.get('tradeable') else "🔴 UNTRADEABLE"
    print(f"  💹  SPREAD/COSTS: {trade_tag} | {trade.get('detail', 'N/A')}")
    cycle_target = circuit.get('cycle_target', 0)
    primary_target = cycle_target if (circuit.get('direction') == 'UP' and cycle_target > current_price) else forecast_price
    target_label = "🌀 CYCLIC" if (circuit.get('direction') == 'UP' and cycle_target > current_price) else "🧠 ML ATR"
    print(f"  🎯 {target_label} FORECAST: {primary_target:.10f}")
    print(f"  🧠 ML Confidence: {ml_prob*100:.1f}%  |  📊 Live Backtest: {live_bt_stats}")
    if sma_gate is not None:
        print("-" * W)
        tag = "🟢 PASS" if sma_gate.get('pass') else "🔴 FAIL"
        print(f"  📉  SMA STACK GATE: {tag} | {sma_gate.get('detail', '')}")
    if reg_channel is not None and reg_channel.get('valid'):
        print("-" * W)
        tag = "🟢 PASS" if reg_channel.get('pass') else "🔴 FAIL"
        print(f"  📐  REGRESSION Z-SCORE GATE: {tag} | {reg_channel.get('detail', '')}")
    if fg_wave is not None and fg_wave.get('valid'):
        print("-" * W)
        print(f"  🌊  FEAR/GREED WAVE: {fg_wave['label']} ({fg_wave['fg_score']:.1f}/100) | Turning: {fg_wave.get('turning_up')}")
    divs = adv_metrics.get('divs', {})
    print("-" * W)
    print("  📊  DIVERGENCE STACK:")
    print(f"      Reg Bull: {'🟢 YES' if divs.get('reg_bull_div') else '⚪ No'} | Hid Bull: {'🟢 YES' if divs.get('hid_bull_div') else '⚪ No'} | Reg Bear: {'🔴 YES' if divs.get('reg_bear_div') else '⚪ No'}")
    print("-" * W)
    print(f"  Entry Price    : {current_price:.10f}  |  1m CMO: {cmo_val:+.2f}  |  Bull Rej Vol: {bull_ratio*100:.1f}%")
    print("-" * W)
    print("  🧬 ML NEURAL FEATURE ACTIVATIONS:")
    feat_str = "  ".join([f"{k.upper()}:{int(v)}" for k, v in new_feats.items() if v > 0])
    print(f"  {feat_str if feat_str else 'No strong micro-features triggered'}")
    print("-" * W)
    tag = "🟢 EXHAUSTED" if entry_exh.get('exhausted') else "⚪ NOT CONFIRMED"
    print(f"  🔎  ENTRY-ZONE EXHAUSTION: {tag} ({entry_exh.get('confidence',0)*100:.0f}%)  |  {entry_exh.get('detail','')}")
    print("-" * W)
    print(f"  🕯  REJECTION STACK: Score {rejection.get('rejection_score',0)}  |  {rejection.get('detail','')}")
    if rejection.get('talib_hits'): print(f"      TA-Lib: {', '.join(rejection['talib_hits'].keys())}")
    print("-" * W)
    c_dir = circuit.get('direction', 'UNKNOWN')
    c_tag = "🟢 UP" if c_dir == "UP" else ("🔴 DOWN" if c_dir == "DOWN" else "⚪ FLAT")
    print(f"  🌀  360° CYCLIC CIRCUIT: {c_tag} | Phase: {circuit.get('phase_deg', 0):.1f}° | Conf: {circuit.get('confidence', 0):.0%}")
    pos = adv_metrics.get('position_size', {})
    if pos:
        print("-" * W)
        print(f"  🎯  POSITION SIZING (1% Risk):")
        print(f"      Size: ${pos.get('size_usd', 0):.2f} ({pos.get('size_coins', 0):.4f} coins) | Stop Dist: {pos.get('stop_dist_pct', 0):.2f}%")
    print("-" * W); print("  📊  VOLUME BREAKDOWN BY TIMEFRAME")
    for tf, vd in tf_volumes.items():
        bull_len = int(vd['bull_pct'] / 100 * 30); bar = "🟢" * bull_len + "🔴" * (30 - bull_len)
        print(f"  {tf:>4s}  [{bar}]  Bull: {vd['bull_pct']:.1f}%")
    for lb_data in sr['lookbacks']:
        ext = lb_data['extremes']
        print("\n" + "─" * W); print(f"  📐  LOOKBACK: {lb_data['lookback']} BARS")
        print(f"  High: {ext['high']:.10f} ({ext['high_age']} bars ago)  |  Low: {ext['low']:.10f} ({ext['low_age']} bars ago) | More Recent: {ext['more_recent']}")
        grid = lb_data['grid']
        if grid:
            print(f"\n  {'Level':<8} {'Price':>14} {'Dist%':>8} {'Bull%':>6} {'Tch':>4} {'Verdict':<10}")
            print("  " + "─" * 56)
            for g in grid:
                m = "►" if g.get('direction') == 'UP' else "◄"
                print(f"  {m}{g['label']:<7} {g['price']:>14.8f} {g['dist_pct']:>+7.3f}% {g['bull_pct']*100:>5.0f}% {g['touches']:>4} {g['verdict']:<10}")
    print("=" * W)

# ==========================================
# MAIN EXECUTION (Updated Logic)
# ==========================================
def main():
    trader = Trader("credentials.txt")
    print("Fetching USDC pairs...")
    symbols = trader.get_usdc_pairs()
    
    LOOP_INTERVAL = 5
    print(f"\n🚀 Starting infinite MTF dip scanner. Scanning {len(symbols)} pairs.")
    print(f"⏱ Loop interval set to {LOOP_INTERVAL}s between scans.")
    time.sleep(3)
    
    scan_count = 0
    
    while True:
        scan_count += 1
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print(f"{'='*60}")
        print(f"  🔄 SCAN #{scan_count} | {local_timestamp()}")
        print(f"{'='*60}")
        
        # 1. HTF Funnels
        print("\n--- STAGE 1: HIGHER TIMEFRAME FUNNELS ---")
        h2_pass = run_tf_filter(trader, symbols, '2h', max_workers=20)
        if not h2_pass:
            print(f"\n⏳ No 2h dips found. Sleeping for {LOOP_INTERVAL}s...")
            time.sleep(LOOP_INTERVAL)
            continue
            
        m15_pass = run_tf_filter(trader, h2_pass, '15m', max_workers=20)
        if not m15_pass:
            print(f"\n⏳ No 15m dips found. Sleeping for {LOOP_INTERVAL}s...")
            time.sleep(LOOP_INTERVAL)
            continue
            
        m5_pass = run_tf_filter(trader, m15_pass, '5m', max_workers=20)
        
        if not m5_pass:
            print(f"\n⏳ No symbols survived the 5m funnel. Sleeping for {LOOP_INTERVAL}s...")
            time.sleep(LOOP_INTERVAL)
            continue
        
        print("\n" + "="*60)
        print(f"  ✅ STAGE 1 COMPLETE: Found {len(m5_pass)} 5m dip candidates!")
        print(f"  📋 5m Survivors: {', '.join(m5_pass)}")
        print("="*60)
        
        # -------------------------------------------------------------
        # NEW LOGIC: 1m IS NOW ONLY A SORTER OR BYPASSED COMPLETELY
        # -------------------------------------------------------------
        
        if len(m5_pass) == 1:
            # EXACTLY 1 FOUND: Skip 1m gates, give data, and STOP BOT
            print("\n🎯 Exactly ONE 5m candidate found. Bypassing 1m gates, generating forecast...")
            sym = m5_pass[0]
            
            # Fetch 1m data to populate forecast numbers (ignoring the pass/fail gates)
            res = run_1m_filter(trader, [sym], max_workers=1)[0]
            sym, cmo_val, vratio, is_strong, bull_ratio, prob, rejection, entry_exh, circuit, forecast_mult, new_feats, sma_gate, reg_channel, fg_wave, rt_price, adv_metrics = res
            
            tf_volumes = {tf: get_volume_breakdown(trader, sym, tf, limit=50) for tf in ['1m', '5m', '15m', '2h']}
            sr_data = get_sr_targets(trader.get_max_klines(sym, '1m', max_candles=MAX_CANDLES, verbose=False), rt_price)
            
            print(f"\n⚠️ NOTE: 1m strict gates bypassed because this was the sole 5m survivor.")
            format_sr_output(
                symbol=sym, sr=sr_data, current_price=rt_price, cmo_val=cmo_val, 
                vratio=vratio, bull_ratio=bull_ratio, ml_prob=prob, tf_volumes=tf_volumes,
                rejection=rejection, entry_exh=entry_exh, circuit=circuit, 
                forecast_price=rt_price * (1 + (forecast_mult * sr_data.get('avg_range', 0.5)/100)), 
                new_feats=new_feats, live_bt_stats="Sole 5m Survivor", 
                sma_gate=sma_gate, reg_channel=reg_channel, fg_wave=fg_wave,
                adv_metrics=adv_metrics
            )
            print("\n🛑 BOT STOPPED: Single 5m MTF dip identified and forecast provided.")
            break # EXIT THE INFINITE LOOP
            
        else:
            # MULTIPLE FOUND: Use 1m strictly to SORT by Lowest RSI
            print("\n--- STAGE 2: RUNNING 1-MINUTE SORTER ON 5m CANDIDATES ---")
            results_1m = run_1m_filter(trader, m5_pass, max_workers=15)
            
            # Sort by lowest RSI (index 10 is new_feats dict, key is 'rsi')
            results_1m.sort(key=lambda x: x[10].get('rsi', 100))
            
            # Pick the absolute best one
            best_pick = results_1m[0]
            sym, cmo_val, vratio, is_strong, bull_ratio, prob, rejection, entry_exh, circuit, forecast_mult, new_feats, sma_gate, reg_channel, fg_wave, rt_price, adv_metrics = best_pick
            
            tf_volumes = {tf: get_volume_breakdown(trader, sym, tf, limit=50) for tf in ['1m', '5m', '15m', '2h']}
            sr_data = get_sr_targets(trader.get_max_klines(sym, '1m', max_candles=MAX_CANDLES, verbose=False), rt_price)
            
            print(f"\n🏆 SORTED BY LOWEST 1m RSI. BEST PICK: {sym} (RSI: {new_feats.get('rsi', 'N/A')})")
            format_sr_output(
                symbol=sym, sr=sr_data, current_price=rt_price, cmo_val=cmo_val, 
                vratio=vratio, bull_ratio=bull_ratio, ml_prob=prob, tf_volumes=tf_volumes,
                rejection=rejection, entry_exh=entry_exh, circuit=circuit, 
                forecast_price=rt_price * (1 + (forecast_mult * sr_data.get('avg_range', 0.5)/100)), 
                new_feats=new_feats, live_bt_stats="Best 1m RSI", 
                sma_gate=sma_gate, reg_channel=reg_channel, fg_wave=fg_wave,
                adv_metrics=adv_metrics
            )
            print("\n🛑 BOT STOPPED: Best MTF dip identified via 1m RSI sort.")
            break # EXIT THE INFINITE LOOP

if __name__ == "__main__":
    main()