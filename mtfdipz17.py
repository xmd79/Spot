import gc
import os
import sys
import json
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
from scipy import stats

# Real ML Imports
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ WARNING: scikit-learn not installed. ML Forecasting will be disabled.")
    print("Run: pip install scikit-learn scipy")

MAX_CANDLES = 500  # Strictly 500 data values

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

def ema_stack_gate(close: np.ndarray) -> Dict:
    """
    THE single MTF/1m filter (per user directive): a symbol only passes if
        close < EMA9 < EMA20   (price is dipping below the fast EMA stack)
        AND EMA50 < EMA200     (still in a broader downtrend/pullback structure)
    No other condition is required to pass this gate. Replaces the old SMA-stack
    gate + linear-regression-channel dip check + volume/orderbook confirmation as
    the pass/fail criterion at every stage (15m, 5m, 1m).
    """
    n = len(close)
    if n < 202:
        return {'pass': False, 'ema9': None, 'ema20': None, 'ema50': None, 'ema200': None, 'detail': 'insufficient bars for EMA200'}
    ema9, ema20, ema50, ema200 = ta.EMA(close, timeperiod=9), ta.EMA(close, timeperiod=20), ta.EMA(close, timeperiod=50), ta.EMA(close, timeperiod=200)
    c, e9, e20, e50, e200 = float(close[-1]), float(ema9[-1]), float(ema20[-1]), float(ema50[-1]), float(ema200[-1])
    if any(np.isnan(x) for x in (e9, e20, e50, e200)):
        return {'pass': False, 'ema9': None, 'ema20': None, 'ema50': None, 'ema200': None, 'detail': 'ema warmup'}
    ok = (c < e9) and (e9 < e20) and (e50 < e200)
    return {'pass': bool(ok), 'ema9': e9, 'ema20': e20, 'ema50': e50, 'ema200': e200, 'close': c,
            'detail': f"close {c:.6g} {'<' if c < e9 else '>='} EMA9 {e9:.6g} {'<' if e9 < e20 else '>='} EMA20 {e20:.6g}  |  EMA50 {e50:.6g} {'<' if e50 < e200 else '>='} EMA200 {e200:.6g}"}

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

# NOTE: fear_greed_wave() was removed per accuracy audit. It fit a sine curve to raw
# price and attached emotional labels ('EXTREME FEAR'/'GREED') to the result -- the
# underlying math didn't measure sentiment, it was a decorative relabeling of price
# min/max. The one genuine structural signal it produced (argmin more recent than
# argmax) is already enforced directly by regression_channel_dip_gate()'s
# argmin_i > argmax_i check, so no real gating power was lost by removing it.

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

# ==========================================
# PRICE-VOLUME CONFLUENCE + VOLUME IMPULSE
# + ORDER BOOK PRESSURE ENGINE
# ==========================================
def price_volume_state_engine(close: np.ndarray, volume: np.ndarray, lookback: int = 14) -> Dict:
    """
    Classifies price/volume relationship using the Price-Volume Confirmation Matrix:
        Vol UP   + Price UP   -> Bull Continuation      (same direction)
        Vol DOWN + Price DOWN -> Bullish Accumulation    (same direction)
        Vol UP   + Price DOWN -> Bear Continuation       (opposite direction)
        Vol DOWN + Price UP   -> Distribution            (opposite direction)

    Directional bias rule (as requested):
        Same-direction states (both slopes agree, either both rising or both falling)
            -> INCOMING UP  (energy and motion aligned; pump precursor)
        Opposite-direction states (one slope up, the other down)
            -> INCOMING DOWN (motion without force, or force without motion; dump precursor)

    Also derives volume acceleration/deceleration by comparing the volume slope of the
    first half of the lookback window vs the second half (2nd derivative proxy).
    """
    n = len(close)
    if n < lookback + 5 or len(volume) < lookback + 5:
        return {'valid': False}

    price_seg = close[-lookback:].astype('float64')
    vol_seg = volume[-lookback:].astype('float64')
    x = np.arange(lookback, dtype='float64')

    price_slope, _ = np.polyfit(x, price_seg, 1)
    vol_slope, _ = np.polyfit(x, vol_seg, 1)

    price_mean = float(np.mean(price_seg)) if np.mean(price_seg) != 0 else 1.0
    vol_mean = float(np.mean(vol_seg)) if np.mean(vol_seg) != 0 else 1.0
    price_slope_pct = (price_slope / price_mean) * 100.0
    vol_slope_pct = (vol_slope / vol_mean) * 100.0

    price_up = price_slope_pct > 0
    vol_up = vol_slope_pct > 0
    same_direction = (price_up == vol_up)

    if price_up and vol_up:
        state, interpretation = "BULL_CONTINUATION", "Buyers active, supporting higher prices"
    elif (not price_up) and (not vol_up):
        state, interpretation = "BULLISH_ACCUMULATION", "Selling pressure drying up, bears losing strength"
    elif (not price_up) and vol_up:
        state, interpretation = "BEAR_CONTINUATION", "Strong selling pressure, sellers dominate"
    else:
        state, interpretation = "DISTRIBUTION", "Price rising without participation, weak rally"

    bias = "INCOMING UP" if same_direction else "INCOMING DOWN"

    half = lookback // 2
    vol_early, vol_late = vol_seg[:half], vol_seg[half:]
    x_early, x_late = np.arange(len(vol_early), dtype='float64'), np.arange(len(vol_late), dtype='float64')
    early_slope = float(np.polyfit(x_early, vol_early, 1)[0]) if len(vol_early) >= 2 else 0.0
    late_slope = float(np.polyfit(x_late, vol_late, 1)[0]) if len(vol_late) >= 2 else 0.0
    vol_acceleration = late_slope - early_slope
    accel_state = "ACCELERATING" if vol_acceleration > 0 else "DECELERATING"

    return {
        'valid': True, 'state': state, 'interpretation': interpretation, 'bias': bias,
        'same_direction': same_direction, 'price_slope_pct': price_slope_pct, 'vol_slope_pct': vol_slope_pct,
        'vol_acceleration': vol_acceleration, 'accel_state': accel_state,
        'detail': f"{state} | {bias} | PriceSlope={price_slope_pct:+.3f}%/bar VolSlope={vol_slope_pct:+.3f}%/bar Vol{accel_state}"
    }

def volume_spike_detector(volume: np.ndarray, window: int = 20, pulse_z: float = 1.5, impulse_z: float = 3.0) -> Dict:
    """
    Rolling z-score volume spike classifier (structurally derived from the volume's own
    recent distribution, not an invented fixed threshold):
        PULSE   -> moderate deviation above rolling mean (early build-up)
        IMPULSE -> extreme deviation above rolling mean (spike confirmation)
    """
    n = len(volume)
    if n < window + 2:
        return {'valid': False}
    hist = volume[-(window + 1):-1].astype('float64')
    current = float(volume[-1])
    mean = float(np.mean(hist))
    std = float(np.std(hist))
    z = (current - mean) / std if std > 0 else 0.0

    if z >= impulse_z: level = "IMPULSE"
    elif z >= pulse_z: level = "PULSE"
    else: level = "NONE"

    return {'valid': True, 'z_score': z, 'level': level, 'current_vol': current, 'avg_vol': mean,
            'spike_confirmed': level in ("PULSE", "IMPULSE"),
            'detail': f"Vol z-score={z:.2f} -> {level} (cur={current:.2f} vs avg={mean:.2f})"}

def orderbook_pressure_force(trader: Trader, symbol: str, depth_limit: int = 100, price_band_pct: float = 0.5) -> Dict:
    """
    Live 1m order book (bid/ask) depth imbalance within a % price band around mid-price.
    This is the 'pressure force' confirmation: real resting buy/sell volume backing the move,
    independent of the candle-derived price/volume slopes above.
    """
    trader.rate_limiter.acquire()
    try:
        ob = trader.client.get_order_book(symbol=symbol, limit=depth_limit)
        bids, asks = ob.get('bids', []), ob.get('asks', [])
        if not bids or not asks: return {'valid': False, 'detail': 'empty book'}

        best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
        mid = (best_bid + best_ask) / 2.0
        if mid <= 0: return {'valid': False, 'detail': 'invalid mid'}
        band = mid * (price_band_pct / 100.0)

        bid_vol = sum(float(p) * float(q) for p, q in bids if mid - float(p) <= band)
        ask_vol = sum(float(p) * float(q) for p, q in asks if float(p) - mid <= band)
        total = bid_vol + ask_vol
        if total <= 0: return {'valid': False, 'detail': 'no depth in band'}

        imbalance = (bid_vol - ask_vol) / total  # -1..+1, positive = buy-side pressure
        spread_pct = (best_ask - best_bid) / best_bid * 100.0

        if imbalance > 0.25: pressure_state = "STRONG BUY PRESSURE"
        elif imbalance > 0.08: pressure_state = "BUY PRESSURE"
        elif imbalance < -0.25: pressure_state = "STRONG SELL PRESSURE"
        elif imbalance < -0.08: pressure_state = "SELL PRESSURE"
        else: pressure_state = "BALANCED"

        return {'valid': True, 'bid_vol': bid_vol, 'ask_vol': ask_vol, 'imbalance': imbalance,
                'pressure_state': pressure_state, 'spread_pct': spread_pct, 'mid_price': mid,
                'detail': f"{pressure_state} (Δ={imbalance:+.2f}, bid${bid_vol:,.0f} vs ask${ask_vol:,.0f}, spread={spread_pct:.3f}%)"}
    except Exception as e:
        return {'valid': False, 'detail': f"Error: {e}"}

def hybrid_pv_orderbook_gate(close: np.ndarray, volume: np.ndarray, trader: Trader, symbol: str, lookback: int = 14) -> Dict:
    """
    Full 'pump incoming' confirmation gate. All four structural legs must independently
    align bullish -- no fallback path, no invented probability blend:
        1) Price-Volume confluence bias == INCOMING UP (same-direction state)
        2) Volume is ACCELERATING (2nd derivative positive)
        3) Volume spike confirmed (PULSE or IMPULSE via rolling z-score)
        4) Live order book pressure is on the BUY side (imbalance > 0.08)
    """
    pv = price_volume_state_engine(close, volume, lookback=lookback)
    spike = volume_spike_detector(volume, window=max(20, lookback), pulse_z=1.5, impulse_z=3.0)
    ob = orderbook_pressure_force(trader, symbol, depth_limit=100, price_band_pct=0.5)

    if not (pv.get('valid') and spike.get('valid') and ob.get('valid')):
        return {'valid': False, 'pass': False, 'pv': pv, 'spike': spike, 'orderbook': ob,
                'detail': 'insufficient data for one or more legs'}

    bullish_bias = pv['bias'] == 'INCOMING UP'
    accelerating = pv['accel_state'] == 'ACCELERATING'
    spike_confirmed = spike['spike_confirmed']
    ob_bullish = ob['imbalance'] > 0.08

    gate_pass = bool(bullish_bias and accelerating and spike_confirmed and ob_bullish)

    detail = (f"PV:{pv['state']}({pv['bias']}) | {pv['accel_state']} | "
              f"Spike:{spike['level']}(z={spike['z_score']:.2f}) | OB:{ob['pressure_state']}(Δ={ob['imbalance']:+.2f})")

    return {'valid': True, 'pass': gate_pass, 'pv': pv, 'spike': spike, 'orderbook': ob, 'detail': detail}

def price_volume_matrix_gate(close: np.ndarray, volume: np.ndarray, lookback: int = 20) -> Dict:
    """
    Standalone textbook 'General Rules in Volume Analysis' filter -- the plain 4-cell
    matrix, independent of the fuller pump-incoming engine above:
        Volume Increasing + Price Rising  -> Bullish
        Volume Decreasing + Price Falling -> Bullish
        Volume Increasing + Price Falling -> Bearish
        Volume Decreasing + Price Rising  -> Bearish
    Kept as its own gate (separate from hybrid_pv_orderbook_gate, which additionally
    requires acceleration + a volume spike + live order-book confirmation) so a dip
    candidate must satisfy BOTH the plain matrix reading AND the fuller confluence engine,
    not just one or the other.
    """
    n = len(close)
    if n < lookback + 2 or len(volume) < lookback + 2:
        return {'valid': False, 'pass': False, 'detail': 'insufficient bars'}

    x = np.arange(lookback, dtype='float64')
    price_slope = float(np.polyfit(x, close[-lookback:].astype('float64'), 1)[0])
    vol_slope = float(np.polyfit(x, volume[-lookback:].astype('float64'), 1)[0])
    price_rising = price_slope > 0
    vol_increasing = vol_slope > 0

    if vol_increasing and price_rising: verdict = "BULLISH"
    elif (not vol_increasing) and (not price_rising): verdict = "BULLISH"
    elif vol_increasing and (not price_rising): verdict = "BEARISH"
    else: verdict = "BEARISH"

    gate_pass = verdict == "BULLISH"
    vol_label = "Increasing" if vol_increasing else "Decreasing"
    price_label = "Rising" if price_rising else "Falling"
    return {'valid': True, 'pass': gate_pass, 'verdict': verdict, 'price_slope': price_slope, 'vol_slope': vol_slope,
            'detail': f"Volume {vol_label} / Price {price_label} -> {verdict}"}

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

# NOTE: dynamic_360_cycle_forecast() was removed per accuracy audit. FFT dominant-cycle
# detection on ~500 bars of crypto data has very few effective cycle repetitions to
# estimate a period from, so the phase/direction output is close to random noise dressed
# up as a "360° cyclic circuit" -- it added false confidence without real predictive edge.
# Price targets now come exclusively from the ML walk-forward forecast (when valid) or
# the structural S/R fib grid (get_sr_targets), both of which are backtested/verifiable.

# ==========================================
# STRUCTURAL RANGE ENGINE (STRICTLY 500 BARS)
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
    
    lookbacks_data = []
    n = len(raw_klines)
    
    if n >= 100:
        ext = get_structural_extremes(closes, highs, lows, n)
        grid = build_fib_grid(ext, current_price)
        tol = max(ext['range_size'] * 0.025, avg_range / 100 * current_price * 1.5)
        for g in grid: g.update(volume_profile_at_level(g['price'], raw_klines, tol))
        lookbacks_data.append({'lookback': n, 'extremes': ext, 'grid': grid, 'targets_up': [g for g in grid if g['direction']=='UP'], 'targets_down': [g for g in grid if g['direction']=='DOWN']})
        
    return {'lookbacks': lookbacks_data, 'vol_bias': vol_bias, 'avg_range': avg_range}

# ==========================================
# REAL ML PRICE FORECAST ENGINE
# ==========================================
def engineer_ml_features(c: np.ndarray, h: np.ndarray, l: np.ndarray, v: np.ndarray, o: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    n = len(c); features = {}
    for p in [1, 2, 3, 5, 8, 13, 21]:
        prev = np.roll(c, p); features[f'ret_{p}'] = np.where(prev > 0, (c - prev) / prev, 0)
    for p in [1, 5, 10]:
        prev = np.roll(c, p); features[f'logret_{p}'] = np.where(prev > 0, np.log(c / prev), 0)
    for p in [7, 14, 21]:
        rsi = ta.RSI(c, timeperiod=p); features[f'rsi_{p}'] = np.nan_to_num(rsi, nan=50.0); features[f'rsi_{p}_dev'] = features[f'rsi_{p}'] - 50
    for p in [14, 21]:
        k, d = ta.STOCH(h, l, c, fastk_period=p, slowk_period=3, slowd_period=3)
        features[f'stoch_k_{p}'] = np.nan_to_num(k, nan=50.0); features[f'stoch_d_{p}'] = np.nan_to_num(d, nan=50.0)
        features[f'stoch_kd_diff_{p}'] = features[f'stoch_k_{p}'] - features[f'stoch_d_{p}']
    for p in [20]:
        upper, mid, lower = ta.BBANDS(c, timeperiod=p, nbdevup=2, nbdevdn=2)
        safe_mid = np.where(mid > 0, mid, 1e-10); bbrange = upper - lower; safe_bbrange = np.where(bbrange > 0, bbrange, 1e-10)
        bb_width = np.where(mid > 0, (upper - lower) / safe_mid, 0); bb_pos = np.where(bbrange > 0, (c - lower) / safe_bbrange, 0.5)
        features[f'bb_width_{p}'] = bb_width; features[f'bb_pos_{p}'] = bb_pos; features[f'bb_pctb_{p}'] = bb_pos - 0.5
    macd, signal, hist = ta.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    features['macd'] = np.nan_to_num(macd, nan=0.0); features['macd_signal'] = np.nan_to_num(signal, nan=0.0)
    features['macd_hist'] = np.nan_to_num(hist, nan=0.0); features['macd_norm'] = np.where(c > 0, features['macd'] / c * 100, 0)
    for p in [14]:
        atr = ta.ATR(h, l, c, timeperiod=p); features[f'atr_{p}'] = np.nan_to_num(atr, nan=0.0); features[f'atr_norm_{p}'] = np.where(c > 0, features[f'atr_{p}'] / c, 0)
    for p in [10, 20, 50, 100]:
        sma = ta.SMA(c, timeperiod=p); dist = np.where(sma > 0, (c - sma) / sma * 100, 0); features[f'sma_dist_{p}'] = dist
        rolling_std = np.array([np.std(c[max(0,i-p):i+1]) for i in range(n)])
        safe_std = np.where(rolling_std > 0, rolling_std, 1e-10)
        features[f'sma_zscore_{p}'] = np.where(rolling_std > 0, (c - np.nan_to_num(sma, nan=c)) / safe_std, 0)
    for p in [9, 21]:
        ema = ta.EMA(c, timeperiod=p); features[f'ema_dist_{p}'] = np.where(ema > 0, (c - ema) / ema * 100, 0)
    for p in [10, 20, 50]:
        v_sma = ta.SMA(v, timeperiod=p); safe_v_sma = np.where(v_sma > 0, v_sma, 1e-10); features[f'vol_ratio_{p}'] = np.where(v_sma > 0, v / safe_v_sma, 1.0)
    vol_sma_short = ta.SMA(v, timeperiod=5); vol_sma_long = ta.SMA(v, timeperiod=20)
    safe_vol_sma_long = np.where(vol_sma_long > 0, vol_sma_long, 1e-10)
    features['vol_trend'] = np.where(vol_sma_long > 0, vol_sma_short / safe_vol_sma_long, 1.0)
    obv = ta.OBV(c, v); obv_sma = ta.SMA(obv, timeperiod=20); features['obv_norm'] = np.where(obv_sma != 0, (obv - obv_sma) / (np.abs(obv_sma) + 1), 0)
    ret_1 = np.concatenate([[0], np.diff(c) / np.roll(c, 1)[1:]])
    for p in [10, 20, 50]:
        rolling_std = np.array([np.std(ret_1[max(0,i-p):i+1]) for i in range(n)]); rolling_mean = np.array([np.mean(ret_1[max(0,i-p):i+1]) for i in range(n)])
        features[f'ret_std_{p}'] = rolling_std; features[f'ret_mean_{p}'] = rolling_mean
        safe_std = np.where(rolling_std > 0, rolling_std, 1e-10)
        features[f'sharpe_{p}'] = np.where(rolling_std > 0, rolling_mean / safe_std, 0)
    for p in [20, 50]:
        skew = np.array([stats.skew(ret_1[max(0,i-p):i+1]) if i >= p and np.std(ret_1[max(0,i-p):i+1]) > 1e-10 else 0 for i in range(n)]); features[f'ret_skew_{p}'] = skew
    for p in [20, 50]:
        rolling_high = np.array([np.max(h[max(0,i-p+1):i+1]) for i in range(n)]); rolling_low = np.array([np.min(l[max(0,i-p+1):i+1]) for i in range(n)])
        rng = rolling_high - rolling_low; safe_rng = np.where(rng > 0, rng, 1e-10); safe_c = np.where(c > 0, c, 1e-10); features[f'range_pos_{p}'] = np.where(rng > 0, (c - rolling_low) / safe_rng, 0.5); features[f'range_size_{p}'] = np.where(c > 0, rng / safe_c * 100, 0)
    for p in [5, 10, 20]:
        prev = np.roll(c, p); features[f'momentum_{p}'] = np.where(prev > 0, (c / prev - 1) * 100, 0)
    body = np.abs(c - o); total_range = h - l
    safe_range = np.where(total_range > 0, total_range, 1e-10)
    features['body_ratio'] = np.where(total_range > 0, body / safe_range, 0.5)
    features['upper_wick'] = np.where(total_range > 0, (h - np.maximum(c, o)) / safe_range, 0)
    features['lower_wick'] = np.where(total_range > 0, (np.minimum(c, o) - l) / safe_range, 0)
    features['is_bullish'] = (c >= o).astype(float)
    
    feature_names = sorted(features.keys()); X = np.column_stack([features[name] for name in feature_names])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    for j in range(X.shape[1]):
        std = np.std(X[:, j])
        if std > 0: mean = np.mean(X[:, j]); X[:, j] = np.clip(X[:, j], mean - 5*std, mean + 5*std)
    return X, feature_names

def create_ml_targets(c: np.ndarray, h: np.ndarray, l: np.ndarray, n_ahead: int = 10) -> Dict[str, np.ndarray]:
    n = len(c); mfe = np.full(n, np.nan); ctc = np.full(n, np.nan); realistic = np.full(n, np.nan)
    for i in range(n - n_ahead):
        future_highs = h[i+1:i+n_ahead+1]; mfe[i] = (np.max(future_highs) - c[i]) / c[i] * 100; ctc[i] = (c[i + n_ahead] - c[i]) / c[i] * 100
        best_exit = 0
        for j in range(1, n_ahead + 1):
            gain = (h[i+j] - c[i]) / c[i] * 100
            if gain > 0.5: best_exit = gain; break
            best_exit = (c[i+j] - c[i]) / c[i] * 100
        realistic[i] = best_exit
    combined = np.full(n, np.nan); valid = ~np.isnan(mfe) & ~np.isnan(ctc); combined[valid] = 0.6 * mfe[valid] + 0.4 * np.maximum(0, ctc[valid])
    return {'mfe': mfe, 'ctc': ctc, 'realistic': realistic, 'combined': combined}

def walk_forward_backtest(X: np.ndarray, y: np.ndarray, model_fn, min_train: int = 200, step: int = 1) -> Dict:
    """
    Dense walk-forward validator. `step=1` re-trains and tests at EVERY valid bar
    (previously defaulted to 5-10, which silently skipped 80-90% of possible entry
    points and let false signals hide between tested bars). Slower but honest.
    """
    predictions = []; actuals = []; train_times = []; i = min_train
    while i < len(X):
        if np.isnan(y[i]): i += step; continue
        X_train, y_train, X_test = X[i-min_train:i], y[i-min_train:i], X[i:i+1]
        if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)): i += step; continue
        try:
            t0 = time.time(); model = model_fn(); model.fit(X_train, y_train); pred = model.predict(X_test)[0]; train_times.append(time.time() - t0)
            if not np.isnan(pred) and not np.isinf(pred) and abs(pred) < 50: predictions.append(max(0, pred)); actuals.append(max(0, y[i]))
        except: pass
        i += step
    if len(predictions) < 10: return {'valid': False, 'n_samples': len(predictions), 'error': 'Insufficient predictions'}
    predictions, actuals = np.array(predictions), np.array(actuals)

    # DEGENERACY GUARD -- invalidate outright rather than downweight. Root cause of the
    # 98-99% "hit rate" + R²=0.000 + ConstantInputWarning combo seen in production: the
    # realized-outcome array (`actuals`) was near-constant across the tested window --
    # almost no real bounce ever happened -- so "predict no bounce every time" trivially
    # scores ~99% without any real skill, and pearsonr on a constant array throws the
    # ConstantInputWarning. A meaningless number is worse than no number, so this backtest
    # is marked invalid and excluded from model selection / display entirely.
    actual_std = float(np.std(actuals))
    actual_up_count = int(np.sum(actuals > 0.3))
    actual_up_rate = actual_up_count / len(actuals)
    if actual_std < 1e-6:
        return {'valid': False, 'n_samples': len(predictions),
                'error': f'Degenerate window: near-zero variance in realized outcomes across {len(actuals)} samples (symbol/period had almost no real price movement) -- any hit rate here would be a trivial "always predict no move" artifact'}
    if actual_up_count < 5:
        return {'valid': False, 'n_samples': len(predictions),
                'error': f'Degenerate window: only {actual_up_count}/{len(actuals)} samples ({actual_up_rate*100:.1f}%) show a real bounce (>0.3%) -- too few positive examples to trust any hit rate'}

    errors = predictions - actuals
    mae, rmse, median_ae = np.mean(np.abs(errors)), np.sqrt(np.mean(errors ** 2)), np.median(np.abs(errors))
    # HONEST HIT RATE: fraction of walk-forward test points where the model correctly
    # called direction (predicted bounce > threshold vs actual bounce > threshold).
    # This is shown to the user directly -- no blending with other metrics.
    hit_rate = float(np.mean((predictions > 0.3) == (actuals > 0.3)))
    pred_up_count = np.sum(predictions > 0.3); precision = float(np.sum((predictions > 0.3) & (actuals > 0.3)) / pred_up_count) if pred_up_count > 0 else 0.0
    recall = float(np.sum((predictions > 0.3) & (actuals > 0.3)) / actual_up_count) if actual_up_count > 0 else 0.0

    # pearsonr on a constant `predictions` array also throws ConstantInputWarning even when
    # `actuals` has real variance (e.g. a model that collapsed to a single output value) --
    # guard unconditionally rather than only for len<=5.
    pred_std = float(np.std(predictions))
    corr, pval = (stats.pearsonr(predictions, actuals) if (pred_std >= 1e-9 and len(predictions) > 5) else (0.0, 1.0))

    bias = np.mean(predictions) - np.mean(actuals); relative_bias = (bias / (np.mean(actuals) + 1e-10)) * 100
    ss_res, ss_tot = np.sum(errors ** 2), np.sum((actuals - np.mean(actuals)) ** 2); r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    # This blended score is used ONLY internally to pick which of the 3-4 candidate
    # models to deploy for the live prediction. It is never surfaced to the user as
    # a probability or "confidence" -- the user-facing number is `hit_rate` above.
    dir_score, corr_score, bias_score, r2_score = np.clip((hit_rate - 0.50) * 5, 0, 1), np.clip(corr, 0, 1) * (1 if pval < 0.05 else 0.5), np.clip(1 - abs(relative_bias) / 50, 0, 1), np.clip(r_squared, 0, 1)
    model_selection_score = float(np.clip(dir_score * 0.30 + corr_score * 0.25 + bias_score * 0.15 + r2_score * 0.30, 0, 1))
    return {'valid': True, 'n_samples': len(predictions), 'mae': mae, 'rmse': rmse, 'median_ae': median_ae,
            'directional_accuracy': hit_rate, 'hit_rate': hit_rate, 'precision': precision, 'recall': recall,
            'actual_up_count': actual_up_count, 'actual_up_rate': actual_up_rate,
            'correlation': corr, 'p_value': pval, 'r_squared': r_squared, 'bias': bias, 'relative_bias_pct': relative_bias,
            'mean_predicted': np.mean(predictions), 'mean_actual': np.mean(actuals),
            'model_selection_score': model_selection_score,
            'avg_train_time_ms': np.mean(train_times) * 1000 if train_times else 0,
            'avg_gain_1.0': np.mean(actuals[predictions > 1.0]) if np.sum(predictions > 1.0) > 5 else None,
            'win_rate_1.0': np.mean(actuals[predictions > 1.0] > 0) if np.sum(predictions > 1.0) > 5 else None}

def make_gradient_boosting(): return GradientBoostingRegressor(n_estimators=80, max_depth=4, learning_rate=0.05, min_samples_leaf=20, subsample=0.8, max_features='sqrt', random_state=42)
def make_random_forest(): return RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=15, max_features='sqrt', random_state=42, n_jobs=1)
def make_ridge(): return Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])

def ml_forecast_price(c: np.ndarray, h: np.ndarray, l: np.ndarray, v: np.ndarray, o: np.ndarray, n_ahead: int = 10, fast_mode: bool = False) -> Dict:
    if not SKLEARN_AVAILABLE: return {'valid': False, 'error': 'scikit-learn not installed'}
    n = len(c); min_data = 150 if fast_mode else 350 
    if n < min_data: return {'valid': False, 'error': f'Need {min_data}+ bars, got {n}'}
    t_start = time.time()
    X, feature_names = engineer_ml_features(c, h, l, v, o)
    targets = create_ml_targets(c, h, l, n_ahead=n_ahead); y = targets['combined']
    valid_mask = ~np.isnan(y); X_valid, y_valid = X[valid_mask], y[valid_mask]
    if len(X_valid) < min_data: return {'valid': False, 'error': f'Need {min_data} valid samples, got {len(X_valid)}'}
    # Dense walk-forward: step=1 (every bar) in accurate mode, step=2 in fast mode.
    # Previous defaults of step=5/10 skipped 80-90% of possible entries from validation.
    step, min_train = (2 if fast_mode else 1), (100 if fast_mode else 200)
    models_to_test = [('GradientBoosting', make_gradient_boosting), ('RandomForest', make_random_forest), ('Ridge', make_ridge)]
    if not fast_mode: models_to_test.append(('ExtraTrees', lambda: ExtraTreesRegressor(n_estimators=80, max_depth=5, min_samples_leaf=20, max_features='sqrt', random_state=42, n_jobs=1)))
    backtest_results = {}
    for name, model_fn in models_to_test:
        bt = walk_forward_backtest(X_valid, y_valid, model_fn, min_train=min_train, step=step)
        backtest_results[name] = bt
        if bt.get('valid'): print(f"    {name:20s} | Hit Rate: {bt['hit_rate']*100:5.1f}% (n={bt['n_samples']:4d}) | R²: {bt['r_squared']:5.3f} | MAE: {bt['mae']:.3f}%")
        else: print(f"    {name:20s} | ❌ {bt.get('error', 'failed')}")
    valid_models = [(name, bt) for name, bt in backtest_results.items() if bt.get('valid')]
    if not valid_models: return {'valid': False, 'error': 'All models failed backtest'}
    # Model selection uses the internal blended score (never shown to the user as "confidence").
    best_name, best_bt = max(valid_models, key=lambda x: x[1]['model_selection_score'])
    final_model = dict(models_to_test)[best_name](); final_model.fit(X_valid, y_valid); raw_prediction = final_model.predict(X_valid[-1:])[0]
    top_features = []
    if hasattr(final_model, 'feature_importances_'):
        importance = dict(zip(feature_names, final_model.feature_importances_)); top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    current_price = float(c[-1])
    hit_rate = best_bt.get('hit_rate', 0.0)
    n_samples_bt = best_bt.get('n_samples', 0)

    # ==========================================
    # ANCHORED TARGETING LOGIC (2.5% MINIMUM ENFORCED)
    # ==========================================
    # If ML predicts flat or down, invalidate completely
    if raw_prediction <= 0.01:
        return {
            'valid': False, 'error': f'ML validates NO bounce (raw pred: {raw_prediction:.3f}%)',
            'current_price': current_price, 'best_model': best_name, 'hit_rate': 0.0, 'n_samples_backtest': n_samples_bt,
            'backtest': best_bt, 'all_backtests': backtest_results, 'n_samples': len(X_valid),
            'n_features': len(feature_names), 'computation_time_ms': (time.time() - t_start) * 1000
        }

    # Calculate 100-bar ATR percentage to anchor a realistic target
    atr_period = min(100, len(c) - 1)
    atr_array = ta.ATR(h, l, c, timeperiod=atr_period)
    atr_pct = float(np.nan_to_num(atr_array[-1], nan=0.0)) / current_price * 100

    # Standard dip bounce targets 1.5x to 2x the ATR. Enforce strict 2.5% minimum floor.
    base_target = max(2.5, atr_pct * 2.0)

    # Scale target by the HONEST walk-forward hit rate (not a blended confidence score),
    # and require a minimum sample count before trusting a high hit rate at all --
    # a 70% hit rate on 12 samples is statistical noise, not edge.
    reliable = n_samples_bt >= 30
    if reliable and hit_rate > 0.60:
        final_prediction = max(2.5, base_target * 1.2)
    elif reliable and hit_rate > 0.50:
        final_prediction = max(2.5, base_target)
    else:
        final_prediction = 2.5  # Absolute floor: low hit rate or insufficient samples

    conservative_prediction = 2.5  # Absolute floor for conservative
    optimistic_prediction = final_prediction * 1.5

    forecast_price = current_price * (1 + final_prediction / 100)
    conservative_price = current_price * (1 + conservative_prediction / 100)
    optimistic_price = current_price * (1 + optimistic_prediction / 100)

    return {
        'valid': True, 'current_price': current_price, 'forecast_price': forecast_price,
        'conservative_price': conservative_price, 'optimistic_price': optimistic_price,
        'forecast_gain_pct': final_prediction, 'conservative_gain_pct': conservative_prediction,
        'optimistic_gain_pct': optimistic_prediction, 'n_ahead': n_ahead,
        'best_model': best_name, 'hit_rate': hit_rate, 'n_samples_backtest': n_samples_bt, 'reliable': reliable,
        'backtest': best_bt, 'all_backtests': backtest_results, 'top_features': top_features, 'n_features': len(feature_names),
        'n_samples': len(X_valid),
        'computation_time_ms': (time.time() - t_start) * 1000, 'bias_correction': 0.0
    }

def print_ml_forecast(forecast: Dict, symbol: str):
    if not forecast.get('valid'): print(f"\n  ❌ ML Forecast failed: {forecast.get('error', 'unknown')}"); return
    W = 74; bt = forecast['backtest']
    print("\n" + "=" * W); print(f"  🧠  REAL ML PRICE FORECAST  —  {symbol}"); print("=" * W)
    print(f"  📊 Best Model: {forecast['best_model']} | Features: {forecast['n_features']} | Samples: {forecast['n_samples']}")
    print(f"  ⏱ Computation: {forecast['computation_time_ms']:.0f}ms")
    print("\n  " + "─" * W); print("  📈 WALK-FORWARD BACKTEST (Dense, step=1 — every bar tested)"); print("  " + "─" * W)
    print(f"    Samples Tested: {bt['n_samples']}\n    Historical Hit Rate: {bt['hit_rate']*100:.1f}%\n    R-Squared: {bt['r_squared']:.4f}\n    Correlation: {bt['correlation']:.4f} (p={bt['p_value']:.4f})")
    print("\n  " + "─" * W); print("  🤖 MODEL COMPARISON"); print("  " + "─" * W)
    print(f"    {'Model':<20} {'Hit Rate':>9} {'N':>6} {'R²':>7}"); print("    " + "─" * 50)
    for name, abt in forecast['all_backtests'].items():
        if abt.get('valid'):
            marker = "►" if name == forecast['best_model'] else " "; print(f"  {marker} {name:<18} {abt['hit_rate']*100:7.1f}% {abt['n_samples']:6d} {abt['r_squared']:6.3f}")
    if forecast['top_features']:
        print("\n  " + "─" * W); print("  🔬 TOP 10 FEATURE IMPORTANCE"); print("  " + "─" * W)
        max_imp = forecast['top_features'][0][1]
        for fname, imp in forecast['top_features']:
            bar_len = int(imp / max_imp * 25) if max_imp > 0 else 0; bar = "█" * bar_len + "░" * (25 - bar_len)
            print(f"    {fname:<25} {bar} {imp:.4f}")
    print("\n  " + "═" * W)
    hr, n_bt, reliable = forecast['hit_rate'], forecast['n_samples_backtest'], forecast.get('reliable', False)
    if not reliable: hr_tag = "🔴 UNRELIABLE (n<30)"
    elif hr > 0.60: hr_tag = "🟢 STRONG"
    elif hr > 0.50: hr_tag = "🟡 MARGINAL"
    else: hr_tag = "🔴 WEAK"
    print(f"  📍 CURRENT PRICE:    {forecast['current_price']:.10f}"); print(f"  🎯 FORECAST PRICE:   {forecast['forecast_price']:.10f}  (+{forecast['forecast_gain_pct']:.3f}%)")
    print(f"  📉 CONSERVATIVE:     {forecast['conservative_price']:.10f}  (+{forecast['conservative_gain_pct']:.3f}%)"); print(f"  📈 OPTIMISTIC:       {forecast['optimistic_price']:.10f}  (+{forecast['optimistic_gain_pct']:.3f}%)")
    print(f"  ⏰ TIMEFRAME:        Anchored to 100-bar ATR (Min 2.5% enforced)")
    print(f"  📊 HISTORICAL HIT RATE ON SIMILAR SETUPS: {hr*100:.1f}% (n={n_bt} walk-forward samples) [{hr_tag}]"); print("  " + "═" * W)

# ==========================================
# CONCURRENT FILTER FUNCTIONS
# ==========================================
def check_tf_dip(trader, symbol, interval):
    raw = trader.get_max_klines(symbol, interval, max_candles=MAX_CANDLES)
    if not raw or len(raw) < 210: return (symbol, False, False)
    close = np.array([float(k[4]) for k in raw], dtype='float64')
    gate = ema_stack_gate(close)
    return (symbol, bool(gate['pass']), gate['pass'])

def check_1m_final(trader, symbol) -> Dict:
    """
    Returns a plain dict (not a positional tuple) so fields are addressed by name --
    this removes an entire class of "wrong index unpacked" bugs and makes the pipeline
    easy to extend without touching every call site.
    """
    empty_adv = {'trend_ctx': {'tradeability': 'NA'}, 'dump': {'valid': False, 'drop_pct': 0}, 'tradeability': {'tradeable': False, 'detail': 'No data'}, 'divs': {}, 'position_size': {}, 'pv_orderbook': {'valid': False}, 'pv_matrix': {'valid': False}}
    empty = {'symbol': symbol, 'is_strong': False, 'bull_ratio': 0.0, 'rejection': {}, 'entry_exh': {}, 'new_feats': {}, 'ema_gate': {}, 'reg_channel': {}, 'realtime_price': 0.0, 'advanced_metrics': empty_adv}

    # All volume and momentum metrics in this function (RSI, Stoch, divergences, dump
    # maturity, price-volume matrix, order-book pressure) are derived exclusively from
    # 1-minute klines fetched here -- no higher-timeframe data is mixed in.
    klines = trader.get_max_klines(symbol, '1m', max_candles=MAX_CANDLES)
    if not klines or len(klines) < 210: return empty
    c = np.array([float(k[4]) for k in klines], dtype='float64'); h = np.array([float(k[2]) for k in klines], dtype='float64')
    l = np.array([float(k[3]) for k in klines], dtype='float64'); v = np.array([float(k[5]) for k in klines], dtype='float64')
    o = np.array([float(k[1]) for k in klines], dtype='float64')

    trend_ctx = add_trend_context(c)
    if trend_ctx['tradeability'] == 'AVOID':
        avoid = dict(empty)
        avoid['advanced_metrics'] = {'trend_ctx': trend_ctx, 'dump': {'valid': False, 'drop_pct': 0}, 'tradeability': {'tradeable': False, 'detail': 'Strong downtrend - AVOIDED'}, 'divs': {}, 'position_size': {}, 'pv_orderbook': {'valid': False}, 'pv_matrix': {'valid': False}}
        return avoid

    dump_met = calculate_dump_metrics(c)
    idx_min, idx_max = np.argmin(l), np.argmax(h); argmin_gt_argmax = 1.0 if idx_min > idx_max else 0.0
    rsi = float(ta.RSI(c, timeperiod=14)[-1]); stoch_k, _ = ta.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowd_period=3); stoch_k = float(stoch_k[-1])
    rsi_vals = ta.RSI(c, timeperiod=14); divs = detect_divergences_v2(h, l, rsi_vals)
    dbl_bottom = 0.0; p_lows_idx = argrelextrema(l[-30:], np.less, order=3)[0]
    if len(p_lows_idx) >= 2:
        low1, low2 = l[-30:][p_lows_idx[-2]], l[-30:][p_lows_idx[-1]]
        if abs(low1 - low2) <= ((np.mean(h[-30:]) - np.mean(l[-30:])) * 0.02) and c[-1] > max(low1, low2): dbl_bottom = 1.0
    fib_zone = 0.0; rng = np.max(h[-40:]) - np.min(l[-40:])
    if rng > 0 and np.max(h[-40:]) - (rng * 0.786) <= c[-1] <= np.max(h[-40:]) - (rng * 0.618): fib_zone = 1.0
    deltas = np.zeros(len(klines))
    for i in range(len(klines)):
        rng_k = h[i] - l[i]
        if rng_k > 0: deltas[i] = v[i] * (abs(c[i] - o[i]) / rng_k) * (1 if c[i] > o[i] else -1)
    cum_delta = np.cumsum(deltas); pl_idx = np.argmin(l[-20:]); delta_div = 1.0 if (l[-1] < l[pl_idx] and cum_delta[-1] > cum_delta[pl_idx]) else 0.0
    sl_20 = np.min(l[-21:-1]); sweep_rec = 1.0 if (l[-1] < sl_20 and c[-1] > sl_20) else 0.0
    new_feats = {'argmin': argmin_gt_argmax, 'rsi': rsi, 'stoch': stoch_k, 'reg_div': divs['reg_bull_div'], 'hid_div': divs['hid_bull_div'], 'dbl_bot': dbl_bottom, 'fib': fib_zone, 'delta_div': delta_div, 'sweep': sweep_rec}
    is_rej, bull_ratio = has_bullish_rejection_volume(klines, window=10)

    ema_gate = ema_stack_gate(c)  # THE only pass/fail filter: close < EMA9 < EMA20, EMA50 < EMA200
    reg_channel = regression_channel_dip_gate(c, band_mult=2.0)  # kept for diagnostics/display only, not gating
    pv_matrix_gate = price_volume_matrix_gate(c, v, lookback=20)          # 1m volume metric, diagnostics only
    pv_ob_gate = hybrid_pv_orderbook_gate(c, v, trader, symbol, lookback=14)  # 1m volume/orderbook metric, diagnostics only

    realtime_price = trader.get_realtime_price(symbol)
    if not realtime_price or realtime_price <= 0: realtime_price = float(c[-1])

    # Per directive: the EMA stack (close < EMA9 < EMA20, EMA50 < EMA200) is the ONLY
    # filter that determines pass/fail here. Regression z-score, the price/volume matrix,
    # and the pump-incoming order-book gate above are still computed and shown in the
    # signal output for context, but no longer gate is_strong.
    mandatory_1m = bool(ema_gate.get('pass', False))
    is_strong = mandatory_1m

    rejection = detect_rejection_patterns(klines, lookback=15)
    entry_exh = detect_entry_exhaustion(klines, realtime_price, zone_pct=0.015)
    stop_price = min(l[-20:]) * 0.999 if len(l) >= 20 else realtime_price * 0.985
    pos_size = calculate_position_size(realtime_price, stop_price) if is_strong else {}
    advanced_metrics = {'trend_ctx': trend_ctx, 'dump': dump_met, 'divs': divs, 'tradeability': {'tradeable': False, 'detail': 'Waiting for ML'}, 'position_size': pos_size, 'pv_orderbook': pv_ob_gate, 'pv_matrix': pv_matrix_gate, 'stop_price': stop_price}

    return {'symbol': symbol, 'is_strong': is_strong, 'bull_ratio': bull_ratio, 'rejection': rejection,
            'entry_exh': entry_exh, 'new_feats': new_feats, 'ema_gate': ema_gate, 'reg_channel': reg_channel,
            'realtime_price': realtime_price, 'advanced_metrics': advanced_metrics}

def calculate_tf_confluence(htf_confirmed: int, htf_total: int, mandatory_1m_pass: bool) -> Dict:
    """
    Honest timeframe confluence score. htf_confirmed/htf_total reflects how many of the
    higher-timeframe cascade stages (15m, 5m) the symbol survived to reach this point
    (by construction this is always htf_total/htf_total for anything reaching the 1m stage),
    plus whether the 1m mandatory gate (EMA stack: close < EMA9 < EMA20, EMA50 < EMA200)
    also confirmed. A 2/3 result means the symbol looked like a dip on higher timeframes
    but did NOT get full 1m confirmation -- worth flagging, not hiding.
    """
    total = htf_total + 1
    aligned = htf_confirmed + (1 if mandatory_1m_pass else 0)
    return {'aligned': aligned, 'total': total, 'ratio': aligned / total, 'label': f"{aligned}/{total} timeframes aligned"}

def determine_signal_tier(is_strong: bool, rejection_score: int, hit_rate: float, hit_rate_reliable: bool, pv_ob_pass: bool) -> str:
    """
    Signal strength tiering per the accuracy audit -- not every gate-pass is equally
    trustworthy. STRONG requires full 1m confirmation, multiple rejection patterns,
    a statistically-backed ML hit rate, AND live order-book confirmation all agreeing.
    """
    if is_strong and rejection_score >= 2 and hit_rate_reliable and hit_rate > 0.55 and pv_ob_pass:
        return "STRONG"
    elif is_strong or (rejection_score >= 1 and hit_rate_reliable and hit_rate > 0.50):
        return "MODERATE"
    else:
        return "WEAK"

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
                tracker.update(passed=ok); print(tracker.get_stats(), end="", flush=True)
            except: tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20); return passed

def run_1m_filter(trader, symbols, max_workers=15):
    results = []; tracker = ProgressTracker(len(symbols), "1m analysis")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(check_1m_final, trader, s): s for s in symbols}
        for f in as_completed(futures):
            try:
                res = f.result(); results.append(res)
                tracker.update(passed=res['is_strong']); print(tracker.get_stats(), end="", flush=True)
            except: tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20); return results

# ==========================================
# ACCUMULATION SCANNER
# Volume/Price Matrix (price_volume_matrix_gate) is the ONLY pass/fail filter.
# Everything else here -- volume slope, acceleration, spikes, OBV, relative volume,
# order-book pressure, order-block retest -- are RANKING features, not gates. They
# never disqualify an asset; they only score how strong a bullish-matrix read is.
# ==========================================
def detect_bullish_order_block(raw_klines: list, lookback: int = 60, impulse_mult: float = 1.5) -> Dict:
    """
    Lightweight order-block detector: the most recent down-close candle immediately
    followed by a strong up-impulse (next candle's close clears that candle's high by
    at least `impulse_mult`x the recent average candle range) leaves behind an
    'order block' -- the down-candle's [low, high] range, read as an institutional
    accumulation zone. Price returning into that zone now is a bullish retest.
    """
    if not raw_klines or len(raw_klines) < lookback + 5:
        return {'valid': False, 'found': False}
    window = raw_klines[-lookback:]
    o = np.array([float(k[1]) for k in window]); h = np.array([float(k[2]) for k in window])
    l = np.array([float(k[3]) for k in window]); c = np.array([float(k[4]) for k in window])
    avg_range = float(np.mean(h - l)) or 1e-12

    ob_zone = None
    for i in range(len(window) - 2, 0, -1):
        if c[i] >= o[i]: continue  # need a down-close candle
        if (c[i + 1] - h[i]) > avg_range * impulse_mult:
            ob_zone = {'high': float(h[i]), 'low': float(l[i]), 'bars_ago': len(window) - 1 - i}
            break
    if ob_zone is None:
        return {'valid': True, 'found': False, 'detail': 'No recent bullish order block found'}

    current_price = float(c[-1])
    retest_now = bool(ob_zone['low'] <= current_price <= ob_zone['high'] * 1.005)
    return {'valid': True, 'found': True, 'high': ob_zone['high'], 'low': ob_zone['low'],
            'bars_ago': ob_zone['bars_ago'], 'retest_now': retest_now,
            'detail': f"Bullish OB [{ob_zone['low']:.8f}-{ob_zone['high']:.8f}] formed {ob_zone['bars_ago']}b ago{' -- RETESTING NOW' if retest_now else ''}"}

def analyze_accumulation(trader: Trader, symbol: str, interval: str = '5m', lookback_bars: int = 100) -> Dict:
    """
    Per-asset accumulation analysis. price_volume_matrix_gate() is the ONLY hard filter --
    an asset must read BULLISH on the plain Volume/Price matrix to pass at all. Everything
    below is scoring only, feeding a 0-100 'accumulation_score' used purely for ranking.
    """
    raw = trader.get_max_klines(symbol, interval, max_candles=max(lookback_bars + 60, 200))
    if not raw or len(raw) < lookback_bars + 20:
        return {'symbol': symbol, 'valid': False, 'pass': False, 'detail': 'insufficient data'}

    c = np.array([float(k[4]) for k in raw], dtype='float64'); h = np.array([float(k[2]) for k in raw], dtype='float64')
    l = np.array([float(k[3]) for k in raw], dtype='float64'); v = np.array([float(k[5]) for k in raw], dtype='float64')

    # ---- THE ONLY HARD FILTER ----
    matrix = price_volume_matrix_gate(c, v, lookback=20)
    if not matrix.get('valid') or not matrix.get('pass'):
        return {'symbol': symbol, 'valid': True, 'pass': False, 'matrix': matrix, 'detail': 'Filtered out: Volume/Price matrix not BULLISH'}

    # ---- SCORING FEATURES (never gates) ----
    pv_state = price_volume_state_engine(c, v, lookback=14)
    spike = volume_spike_detector(v, window=20, pulse_z=1.5, impulse_z=3.0)
    ob_pressure = orderbook_pressure_force(trader, symbol, depth_limit=100, price_band_pct=0.5)
    order_block = detect_bullish_order_block(raw, lookback=60)

    vol_slopes = {}
    for lb in (10, 20, 50):
        if len(v) >= lb:
            vs = float(np.polyfit(np.arange(lb, dtype='float64'), v[-lb:], 1)[0])
            vmean = float(np.mean(v[-lb:])) or 1.0
            vol_slopes[f'{lb}bar_pct_per_bar'] = vs / vmean * 100.0
    vol_direction = "INCREASING" if vol_slopes.get('20bar_pct_per_bar', 0) > 0 else "DECREASING"

    obv = ta.OBV(c, v)
    obv_valid = len(obv) >= 20 and not np.any(np.isnan(obv[-20:]))
    obv_slope = float(np.polyfit(np.arange(20, dtype='float64'), obv[-20:], 1)[0]) if obv_valid else 0.0
    obv_trend = "RISING" if obv_slope > 0 else "FALLING"

    vol_sma20 = float(np.mean(v[-20:])) if len(v) >= 20 else float(np.mean(v))
    rel_volume = float(v[-1] / vol_sma20) if vol_sma20 > 0 else 1.0

    bb_upper, bb_mid, bb_lower = ta.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)
    bb_width_now = float((bb_upper[-1] - bb_lower[-1]) / bb_mid[-1] * 100) if (not np.isnan(bb_mid[-1]) and bb_mid[-1] > 0) else 0.0
    hist_len = min(60, len(bb_mid) - 1)
    safe_mid = np.where(bb_mid[-hist_len:] != 0, bb_mid[-hist_len:], 1e-12)
    bb_width_hist = (bb_upper[-hist_len:] - bb_lower[-hist_len:]) / safe_mid * 100.0
    coiling_percentile = float(np.mean(bb_width_now < bb_width_hist)) if hist_len > 0 else 0.5  # low = unusually tight range right now

    # ---- ACCUMULATION SCORE (0-100, ranking ONLY -- not a filter) ----
    score = 0.0
    score += 20.0 if pv_state.get('valid') and pv_state.get('bias') == 'INCOMING UP' else 0.0
    score += 15.0 if pv_state.get('valid') and pv_state.get('accel_state') == 'ACCELERATING' else 0.0
    score += {'IMPULSE': 20.0, 'PULSE': 10.0, 'NONE': 0.0}.get(spike.get('level', 'NONE'), 0.0) if spike.get('valid') else 0.0
    score += 10.0 if obv_trend == 'RISING' else 0.0
    score += float(np.clip((rel_volume - 1.0) * 15.0, 0.0, 15.0))
    score += 10.0 if (ob_pressure.get('valid') and ob_pressure.get('imbalance', 0.0) > 0.08) else 0.0
    score += 10.0 if (order_block.get('found') and order_block.get('retest_now')) else 0.0
    score += float(np.clip((1.0 - coiling_percentile) * 5.0, 0.0, 5.0))  # tight coiling range bonus
    score = float(np.clip(score, 0.0, 100.0))

    return {
        'symbol': symbol, 'valid': True, 'pass': True, 'price': float(c[-1]), 'interval': interval,
        'matrix': matrix, 'pv_state': pv_state, 'spike': spike, 'orderbook': ob_pressure, 'order_block': order_block,
        'vol_slopes': vol_slopes, 'vol_direction': vol_direction, 'obv_trend': obv_trend, 'obv_slope': obv_slope,
        'rel_volume': rel_volume, 'bb_width_pct': bb_width_now, 'coiling_percentile': coiling_percentile,
        'accumulation_score': score,
    }

def scan_accumulation_universe(trader: Trader, symbols: List[str], interval: str = '5m', max_workers: int = 25) -> List[Dict]:
    """Multithreaded scan of every symbol; returns only symbols that passed the Volume/Price
    matrix filter, sorted best-to-worst by accumulation_score."""
    results = []
    tracker = ProgressTracker(len(symbols), f"Accumulation scan ({interval})")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_accumulation, trader, s, interval): s for s in symbols}
        for f in as_completed(futures):
            try:
                res = f.result()
                passed = bool(res.get('valid') and res.get('pass'))
                if passed: results.append(res)
                tracker.update(passed=passed); print(tracker.get_stats(), end="", flush=True)
            except Exception:
                tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20)
    results.sort(key=lambda r: r['accumulation_score'], reverse=True)
    return results

def print_accumulation_table(results: List[Dict], top_n: int = 25) -> None:
    W = 118
    print("\n" + "=" * W)
    print("  📊  ACCUMULATION TABLE  —  Filter: Volume/Price Matrix BULLISH only  |  Ranked by accumulation score")
    print("=" * W)
    print(f"  {'#':>3} {'Symbol':<14} {'Score':>6} {'Price':>16} {'PV Bias':<14} {'VolAccel':<12} {'Spike':<8} {'RelVol':>7} {'OBV':<8} {'OrderBlock':<10} {'Order Book Pressure':<22}")
    print("  " + "-" * (W - 2))
    for i, r in enumerate(results[:top_n], 1):
        ob_retest = "RETEST" if r['order_block'].get('retest_now') else ("found" if r['order_block'].get('found') else "-")
        ob_pressure_label = r['orderbook'].get('pressure_state', 'N/A') if r['orderbook'].get('valid') else 'N/A'
        print(f"  {i:>3} {r['symbol']:<14} {r['accumulation_score']:>6.1f} {r['price']:>16.8f} "
              f"{r['pv_state'].get('bias', 'N/A'):<14} {r['pv_state'].get('accel_state', 'N/A'):<12} "
              f"{r['spike'].get('level', 'N/A'):<8} {r['rel_volume']:>6.2f}x {r['obv_trend']:<8} {ob_retest:<10} {ob_pressure_label:<22}")
    print("=" * W)
    if results:
        best = results[0]
        print(f"\n  🎯 BEST ACCUMULATION CANDIDATE: {best['symbol']}  (score {best['accumulation_score']:.1f}/100)")
        print(f"     {best['matrix'].get('detail', '')}")
        print(f"     {best['pv_state'].get('detail', '')}")
        print(f"     {best['spike'].get('detail', '')}")
        if best['orderbook'].get('valid'): print(f"     Order Book: {best['orderbook'].get('detail', '')}")
        if best['order_block'].get('found'): print(f"     Order Block: {best['order_block'].get('detail', '')}")
        print(f"     ⚠️  This is a ranking of textbook accumulation signals, not a guarantee -- nothing here")
        print(f"        predicts a spike with certainty; treat it as a shortlist to research further.")

def main_accumulation():
    """
    Standalone accumulation scanner mode. Unlike the MTF dip scanner in main(), this loop
    never stops at a single pick -- it re-scans the whole USDC universe on an interval and
    prints a live-updating leaderboard. Run with: python mtfdipz11.py accumulation
    """
    trader = Trader("credentials.txt")
    print("Fetching USDC pairs...")
    symbols = trader.get_usdc_pairs()
    INTERVAL = '5m'
    LOOP_INTERVAL = 15
    print(f"\n🚀 Starting ACCUMULATION scanner ({INTERVAL} candles).")
    print(f"   ONLY hard filter: Volume/Price Matrix must read BULLISH. Everything else (volume")
    print(f"   acceleration, spikes, OBV, relative volume, order-book pressure, order-block")
    print(f"   retest) is a ranking feature, not a gate.")
    print(f"   Scanning {len(symbols)} pairs, {LOOP_INTERVAL}s between scans. Ctrl+C to stop.")
    time.sleep(2)
    scan_count = 0
    while True:
        scan_count += 1
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{'='*60}\n  🔄 ACCUMULATION SCAN #{scan_count} | {local_timestamp()}\n{'='*60}")
        results = scan_accumulation_universe(trader, symbols, interval=INTERVAL, max_workers=25)
        if not results:
            print(f"\n⏳ No symbol passed the Volume/Price Matrix filter this scan. Sleeping {LOOP_INTERVAL}s...")
        else:
            print_accumulation_table(results, top_n=25)
            print(f"\n⏳ Next scan in {LOOP_INTERVAL}s... (Ctrl+C to stop)")
        time.sleep(LOOP_INTERVAL)

# ==========================================
# SIGNAL LOG (REAL OUT-OF-SAMPLE TRACKING)
# ==========================================
SIGNAL_LOG_PATH = str(Path(__file__).resolve().parent / "signal_log.jsonl")
OUTCOME_WINDOW_HOURS = 24

def log_signal(signal: Dict) -> None:
    """Appends the fired signal to a durable log so its real outcome can be checked later.
    This is what actually earns an honest win-rate number over time, instead of the
    invented pre-trade 'confidence' the old version printed."""
    entry = {
        'symbol': signal['symbol'], 'timestamp': signal['timestamp'], 'tier': signal['tier'],
        'signal_price': signal['entry_price'], 'target_1_price': signal['target_1_price'],
        'target_2_price': signal['target_2_price'], 'stop_price': signal['stop_price'],
        'ml_hit_rate_at_signal': signal.get('ml_hit_rate'), 'ml_n_samples_at_signal': signal.get('ml_n_samples'),
        'max_gain_24h': None, 'max_drawdown_24h': None, 'hit_target_1': None, 'hit_stop': None, 'resolved': False
    }
    try:
        with open(SIGNAL_LOG_PATH, 'a') as f: f.write(json.dumps(entry) + "\n")
    except Exception as e: print(f"  ⚠️ Could not write signal log: {e}")

def _read_signal_log() -> List[Dict]:
    if not os.path.exists(SIGNAL_LOG_PATH): return []
    out = []
    with open(SIGNAL_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: out.append(json.loads(line))
            except Exception: continue
    return out

def update_signal_outcomes(trader: Trader, hours: int = OUTCOME_WINDOW_HOURS) -> int:
    """
    Backfills real outcomes for previously-fired signals once `hours` have elapsed,
    by pulling actual 1m klines for the window after the signal fired. This is the
    'ADD REAL OUT-OF-SAMPLE TRACKING' fix -- signals now get judged against what
    actually happened, not a pre-trade probability estimate. Call this once at the
    start of each run; it self-heals the log incrementally over time.
    """
    entries = _read_signal_log()
    if not entries: return 0
    now = datetime.now(); resolved_count = 0; changed = False
    for entry in entries:
        if entry.get('resolved'): continue
        try: sig_time = datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S')
        except Exception: continue
        age_hours = (now - sig_time).total_seconds() / 3600.0
        if age_hours < hours: continue
        start_ms = int(sig_time.timestamp() * 1000); end_ms = start_ms + int(hours * 3600 * 1000)
        raw = trader.get_klines(entry['symbol'], '1m', limit=1000, return_raw=True, start_time=start_ms, end_time=end_ms)
        if not raw: continue
        highs = [float(k[2]) for k in raw]; lows = [float(k[3]) for k in raw]
        entry_price = entry['signal_price']
        if not entry_price or entry_price <= 0: continue
        max_gain = (max(highs) - entry_price) / entry_price * 100
        max_dd = (entry_price - min(lows)) / entry_price * 100
        entry['max_gain_24h'] = max_gain; entry['max_drawdown_24h'] = max_dd
        t1 = entry.get('target_1_price') or float('inf'); sp = entry.get('stop_price') or 0
        entry['hit_target_1'] = bool(max(highs) >= t1); entry['hit_stop'] = bool(min(lows) <= sp)
        entry['resolved'] = True; changed = True; resolved_count += 1
    if changed:
        try:
            with open(SIGNAL_LOG_PATH, 'w') as f:
                for e in entries: f.write(json.dumps(e) + "\n")
        except Exception as e: print(f"  ⚠️ Could not update signal log: {e}")
    return resolved_count

def get_historical_win_rate(min_n: int = 5) -> Dict:
    """Reads RESOLVED (real outcome, not predicted) log entries and reports the realized
    hit rate of past signals against their own stated target_1. This is what should be
    trusted over time -- it only grows as the bot actually runs and signals mature."""
    entries = [e for e in _read_signal_log() if e.get('resolved')]
    if len(entries) < min_n: return {'valid': False, 'n': len(entries)}
    wins = sum(1 for e in entries if e.get('hit_target_1'))
    avg_gain = np.mean([e['max_gain_24h'] for e in entries if e.get('max_gain_24h') is not None])
    avg_dd = np.mean([e['max_drawdown_24h'] for e in entries if e.get('max_drawdown_24h') is not None])
    return {'valid': True, 'n': len(entries), 'win_rate': wins / len(entries), 'avg_max_gain_24h': float(avg_gain), 'avg_max_drawdown_24h': float(avg_dd)}

# ==========================================
# SIGNAL BUILDER (HONEST, ACTIONABLE OUTPUT)
# ==========================================
def build_signal(symbol: str, result: Dict, sr_data: Dict, ml_forecast: Dict, tf_confluence: Dict) -> Dict:
    """
    Assembles the clean, actionable signal dict recommended by the accuracy audit --
    tiered, honestly labeled, and stripped of decorative metrics (cycle/fear-greed).
    """
    rt_price = result['realtime_price']
    rejection = result['rejection']
    adv = result['advanced_metrics']
    pv_ob = adv.get('pv_orderbook', {})
    pv_matrix = adv.get('pv_matrix', {})
    stop_price = adv.get('stop_price', rt_price * 0.985)

    if ml_forecast.get('valid'):
        target_1 = ml_forecast['conservative_price']; target_2 = ml_forecast['forecast_price']
        ml_hit_rate = ml_forecast.get('hit_rate', 0.0); ml_n = ml_forecast.get('n_samples_backtest', 0)
        ml_reliable = ml_forecast.get('reliable', False)
    else:
        # Fallback target: nearest structural resistance from the fib grid, not a cyclic guess.
        grid = sr_data['lookbacks'][0]['grid'] if sr_data.get('lookbacks') else []
        up_targets = sorted([g for g in grid if g.get('direction') == 'UP'], key=lambda g: g['price'])
        target_1 = up_targets[0]['price'] if up_targets else rt_price * 1.025
        target_2 = up_targets[1]['price'] if len(up_targets) > 1 else rt_price * 1.04
        ml_hit_rate, ml_n, ml_reliable = 0.0, 0, False

    rejection_score = rejection.get('rejection_score', 0)
    tier = determine_signal_tier(result['is_strong'], rejection_score, ml_hit_rate, ml_reliable, pv_ob.get('pass', False))

    hist = get_historical_win_rate()
    hist_label = f"{hist['win_rate']*100:.0f}% (n={hist['n']} resolved signals)" if hist.get('valid') else "insufficient resolved history yet"

    return {
        'symbol': symbol, 'timestamp': local_timestamp()[:19], 'tier': tier,
        'entry_price': rt_price, 'stop_price': stop_price,
        'target_1_price': target_1, 'target_2_price': target_2,
        'target_1_gain_pct': (target_1 - rt_price) / rt_price * 100 if rt_price else 0.0,
        'target_2_gain_pct': (target_2 - rt_price) / rt_price * 100 if rt_price else 0.0,
        'stop_loss_pct': (rt_price - stop_price) / rt_price * 100 if rt_price else 0.0,
        'volume_state': f"{pv_ob.get('pv', {}).get('state', 'N/A')} + {pv_ob.get('pv', {}).get('accel_state', 'N/A')}" if pv_ob.get('valid') else 'N/A',
        'volume_price_matrix': pv_matrix.get('detail', 'N/A') if pv_matrix.get('valid') else 'N/A',
        'orderbook_pressure': pv_ob.get('orderbook', {}).get('pressure_state', 'N/A') if pv_ob.get('valid') else 'N/A',
        'rejection_patterns': list(rejection.get('talib_hits', {}).keys()) + (['pin_bar'] if rejection.get('pin_bar') else []) + (['tweezer_bottom'] if rejection.get('tweezer_bottom') else []) + (['wyckoff_spring'] if rejection.get('wyckoff_spring') else []),
        'rejection_score': rejection_score,
        'tf_confluence': tf_confluence['label'],
        'ml_hit_rate': ml_hit_rate, 'ml_n_samples': ml_n, 'ml_reliable': ml_reliable,
        'ml_valid': ml_forecast.get('valid', False),
        'historical_win_rate': hist_label,
        'gate_summary': {
            'ema_stack': result['ema_gate'].get('pass', False),
            'regression_zscore': result['reg_channel'].get('pass', False),
            'volume_price_matrix': pv_matrix.get('pass', False),
            'pump_incoming_ob': pv_ob.get('pass', False),
            'is_strong_1m_confirmed': result['is_strong'],
        }
    }

def print_signal(signal: Dict, verbose_diagnostics: bool = True, sr_data: Dict = None, ml_forecast: Dict = None, adv_metrics: Dict = None, tf_volumes: Dict = None) -> None:
    W = 74
    tier_icon = {'STRONG': '🟢🟢🟢', 'MODERATE': '🟡🟡', 'WEAK': '🔴'}.get(signal['tier'], '⚪')
    print("\n" + "=" * W)
    print(f"  ★  DIP SIGNAL — {signal['symbol']}  |  TIER: {signal['tier']} {tier_icon}")
    print(f"  🕒  {signal['timestamp']}")
    print("=" * W)
    print(f"  Entry Zone     : {signal['entry_price']:.10f}")
    print(f"  Stop           : {signal['stop_price']:.10f}  ({signal['stop_loss_pct']:.2f}% risk)")
    print(f"  Target 1 (cons): {signal['target_1_price']:.10f}  (+{signal['target_1_gain_pct']:.2f}%)")
    print(f"  Target 2 (opt) : {signal['target_2_price']:.10f}  (+{signal['target_2_gain_pct']:.2f}%)")
    print("-" * W)
    print(f"  Volume/Price Matrix: {signal['volume_price_matrix']}")
    print(f"  Volume State   : {signal['volume_state']}")
    print(f"  Order Book     : {signal['orderbook_pressure']}")
    print(f"  Rejection Score: {signal['rejection_score']}  {', '.join(signal['rejection_patterns']) if signal['rejection_patterns'] else '(none confirmed)'}")
    print(f"  TF Confluence  : {signal['tf_confluence']}")
    print("-" * W)
    if signal['ml_valid']:
        rel_tag = "" if signal['ml_reliable'] else "  ⚠️ LOW SAMPLE COUNT — treat as unreliable"
        print(f"  ML Hit Rate    : {signal['ml_hit_rate']*100:.1f}% (n={signal['ml_n_samples']} walk-forward samples){rel_tag}")
    else:
        print(f"  ML Hit Rate    : N/A (ML forecast invalid — target from structural S/R grid)")
    print(f"  Realized Track Record (this bot's own resolved signals): {signal['historical_win_rate']}")
    print("-" * W)
    gs = signal['gate_summary']
    print(f"  Gates: EMA Stack[{'✓' if gs['ema_stack'] else '✗'}]  RegZScore[{'✓' if gs['regression_zscore'] else '✗'}]  "
          f"VolPriceMatrix[{'✓' if gs['volume_price_matrix'] else '✗'}]  PumpIncoming[{'✓' if gs['pump_incoming_ob'] else '✗'}]  "
          f"1mConfirmed[{'✓' if gs['is_strong_1m_confirmed'] else '✗'}]")
    print("=" * W)

    if verbose_diagnostics and sr_data is not None:
        print("\n  " + "─" * (W - 2)); print("  DETAILED DIAGNOSTICS (secondary — for review, not decision-making)"); print("  " + "─" * (W - 2))
        adv_metrics = adv_metrics or {}
        t_ctx = adv_metrics.get('trend_ctx', {})
        print(f"  Macro Trend: {t_ctx.get('tradeability', 'N/A')} (score {t_ctx.get('tradeability_score', 0):.1f})")
        dump = adv_metrics.get('dump', {})
        if dump.get('valid'): print(f"  Dump Maturity: {dump['dump_phase']} | Extended: {dump['extendedness']} (z={dump['z_score_drop']:.1f})")
        if tf_volumes:
            print("  Volume Breakdown by TF:")
            for tf, vd in tf_volumes.items(): print(f"    {tf:>4s}: Bull {vd['bull_pct']:.1f}%")
        for lb_data in sr_data.get('lookbacks', []):
            ext = lb_data['extremes']
            print(f"  Structural Range ({lb_data['lookback']} bars): High {ext['high']:.8f} ({ext['high_age']}b ago) | Low {ext['low']:.8f} ({ext['low_age']}b ago)")
        if ml_forecast and ml_forecast.get('valid'):
            print_ml_forecast(ml_forecast, signal['symbol'])

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    if not SKLEARN_AVAILABLE:
        print("❌ FATAL ERROR: scikit-learn is required for this version of the bot.")
        print("Please run: pip install scikit-learn scipy")
        return

    trader = Trader("credentials.txt")

    print("Checking for signals ready to be scored against real outcomes...")
    n_resolved = update_signal_outcomes(trader, hours=OUTCOME_WINDOW_HOURS)
    if n_resolved: print(f"  ✅ Resolved {n_resolved} past signal(s) against real 24h price action.")
    hist = get_historical_win_rate()
    if hist.get('valid'):
        print(f"  📊 Realized track record so far: {hist['win_rate']*100:.1f}% hit target_1 (n={hist['n']})")
    else:
        print(f"  📊 Not enough resolved signals yet for a track record ({hist.get('n', 0)} so far).")

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

        # MTF selector uses ONLY 15m, 5m, and 1m timeframes -- the 2h stage has been
        # removed. The single filter at every stage is the EMA stack:
        #   close < EMA9 < EMA20  AND  EMA50 < EMA200
        print("\n--- STAGE 1: 15m / 5m TIMEFRAME FUNNELS (EMA-stack filter) ---")
        m15_pass = run_tf_filter(trader, symbols, '15m', max_workers=20)
        if not m15_pass:
            print(f"\n⏳ No 15m dips found. Sleeping for {LOOP_INTERVAL}s..."); time.sleep(LOOP_INTERVAL); continue
        m5_pass = run_tf_filter(trader, m15_pass, '5m', max_workers=20)
        if not m5_pass:
            print(f"\n⏳ No symbols survived the 5m funnel. Sleeping for {LOOP_INTERVAL}s..."); time.sleep(LOOP_INTERVAL); continue

        print("\n" + "="*60)
        print(f"  ✅ STAGE 1 COMPLETE: Found {len(m5_pass)} 5m dip candidates!")
        print(f"  📋 5m Survivors: {', '.join(m5_pass)}")
        print("="*60)

        if len(m5_pass) == 1:
            print("\n🎯 Exactly ONE 5m candidate found. Running full 1m gate stack...")
            sym = m5_pass[0]
            result = run_1m_filter(trader, [sym], max_workers=1)[0]
            if not result['is_strong']:
                ema_pass = result['ema_gate'].get('pass', False)
                print(f"\n❌ {sym} did NOT clear the 1m EMA-stack filter -- NOT presented as a confirmed MTF dip.")
                print(f"   Gate: EMA Stack[{'✓' if ema_pass else '✗'}]  ({result['ema_gate'].get('detail', 'n/a')})")
                print(f"\n⏳ Continuing scan... sleeping {LOOP_INTERVAL}s"); time.sleep(LOOP_INTERVAL); continue
        else:
            print("\n--- STAGE 2: RUNNING 1-MINUTE EMA-STACK FILTER ON 5m CANDIDATES ---")
            results_1m = run_1m_filter(trader, m5_pass, max_workers=15)
            # ONLY candidates that pass the 1m EMA-stack filter (close < EMA9 < EMA20,
            # EMA50 < EMA200) are eligible to be called an MTF dip. Symbols that merely
            # survived the 15m/5m cascade but failed 1m confirmation are reported, never
            # presented as a signal.
            confirmed = [r for r in results_1m if r['is_strong']]
            unconfirmed = [r for r in results_1m if not r['is_strong']]
            if unconfirmed:
                print(f"\n  ⚪ {len(unconfirmed)}/{len(results_1m)} 5m candidate(s) reached 1m but failed EMA-stack confirmation (not shown as signals): {', '.join(r['symbol'] for r in unconfirmed)}")
            if not confirmed:
                print(f"\n❌ No candidate cleared the 1m EMA-stack filter this scan (0/{len(results_1m)} confirmed).")
                print(f"⏳ Continuing scan... sleeping {LOOP_INTERVAL}s"); time.sleep(LOOP_INTERVAL); continue
            confirmed.sort(key=lambda r: r['new_feats'].get('rsi', 100))
            result = confirmed[0]
            sym = result['symbol']
            print(f"\n🏆 Picked {sym}: fully 1m-confirmed, lowest RSI among {len(confirmed)} confirmed candidate(s) ({result['new_feats'].get('rsi', 'N/A'):.1f}).")

        rt_price = result['realtime_price']
        # Volume/momentum metrics traced ONLY via the 1m timeframe (no 5m/15m/2h volume mixed in).
        tf_volumes = {'1m': get_volume_breakdown(trader, sym, '1m', limit=50)}
        sr_data = get_sr_targets(trader.get_max_klines(sym, '1m', max_candles=MAX_CANDLES, verbose=False), rt_price)
        tf_confluence = calculate_tf_confluence(htf_confirmed=2, htf_total=2, mandatory_1m_pass=result['is_strong'])

        print(f"\n🧠 Running Real ML Anchored Forecast on {sym} (dense walk-forward, this may take a bit)...")
        raw_ml = trader.get_max_klines(sym, '1m', max_candles=MAX_CANDLES, verbose=False)
        ml_forecast = {'valid': False}
        if raw_ml and len(raw_ml) >= 150:
            o_ml = np.array([float(k[1]) for k in raw_ml], dtype='float64')
            h_ml = np.array([float(k[2]) for k in raw_ml], dtype='float64')
            l_ml = np.array([float(k[3]) for k in raw_ml], dtype='float64')
            c_ml = np.array([float(k[4]) for k in raw_ml], dtype='float64')
            v_ml = np.array([float(k[5]) for k in raw_ml], dtype='float64')
            ml_forecast = ml_forecast_price(c_ml, h_ml, l_ml, v_ml, o_ml, n_ahead=10, fast_mode=True)
            if ml_forecast.get('valid'):
                result['advanced_metrics']['tradeability'] = check_tradeability(trader, sym, ml_forecast['forecast_gain_pct'])

        signal = build_signal(sym, result, sr_data, ml_forecast, tf_confluence)
        print_signal(signal, verbose_diagnostics=True, sr_data=sr_data, ml_forecast=ml_forecast, adv_metrics=result['advanced_metrics'], tf_volumes=tf_volumes)
        log_signal(signal)
        print(f"\n🛑 BOT STOPPED: Confirmed MTF dip signal logged to {SIGNAL_LOG_PATH} for future outcome tracking.")
        break

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("accumulation", "accum", "--accumulation"):
        main_accumulation()
    else:
        main()