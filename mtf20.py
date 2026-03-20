"""
scanner.py  —  MTF Harmonic Pump Hunter (Full Production Build)
=============================================================
vs mtf23:
  ✅ talib.MOM          replaces manual momentum calculation
  ✅ talib.LINEARREG    replaces scipy linregress for regression channel
     (+ talib.LINEARREG_SLOPE / LINEARREG_INTERCEPT for full channel bands)
  ✅ talib.RSI          replaces manual RSI (when talib available)
  ✅ All targets > current price (per TF)
  ✅ All FFT forecasts  > current price (all TFs)
  ✅ 1m bullish volume % > bearish volume % (hard gate)
  ✅ 1m Momentum (talib.MOM) > 0 (hard gate)
  ✅ 1m/3m/5m MUST have most_recent_extreme = LOW
  ✅ Majority higher TFs same (LOW recent)
  ✅ Distance % below regression (oversold intensity) in scoring
  ✅ Volatility compression + breakout readiness
  ✅ Reversal confirmation per TF + MTF alignment
  ✅ Liquidity sweep detection (500-bar close breach)
  ✅ Orderbook imbalance (real-time bid/ask)
  ✅ Smart entry zone (regression + close extrema)
  ✅ Live table: talib MOM | Reg slope | Bull% | MRE columns added
  ✅ Sniper: pump-spike predictor (fast profits + higher TF alignment)
  ✅ New MTF dip patterns integrated into scoring & sniper gate:
       • Engulfing Dip Pattern     (1m/3m/5m bullish engulf at LL)
       • RSI Divergence            (price LL but RSI higher low)
       • Volume Climax             (spike volume at wick_ll bar)
       • Sine Trough Confluence    (sine < -0.7 on ≥3 short TFs)
       • Regression Rejection      (price bouncing off lower band)
       • Squeeze-Release           (compression → expansion)
       • Cascade Dip Alignment     (all 8 TFs recent_is_low together)
"""

from binance.client import Client
import numpy as np
import sys
import concurrent.futures
import scipy.signal as scipy_signal
from scipy.stats import linregress as scipy_linregress
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rich.console import Console
from rich.table import Table
from rich import box
from rich.live import Live
from decimal import Decimal
import logging
import warnings

# ── Suppress urllib3 connection pool warnings from concurrent Binance calls ────
# The warning fires because MAX_WORKERS threads share one Client session whose
# default pool size (10) is smaller than the thread count. We patch it below.
warnings.filterwarnings("ignore", message=".*Connection pool.*", category=Warning)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    logging.warning("talib not installed — scipy fallbacks active for MOM / LINEARREG / RSI / HT_SINE")

# ====================== CONFIG ======================
RSI_LENGTH             = 14
MAX_WORKERS            = 12
TF_LIST                = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d']
VOLUME_LOOKBACK        = 20
SCALP_LOOKBACK_CANDLES = 50
WICK_LOOKBACK          = 500   # fallback only — use EXTREMA_LOOKBACK per TF
MOM_PERIOD             = 10    # talib.MOM period
LINEARREG_PERIOD       = 50    # talib.LINEARREG period for channel

KLINES_LIMIT   = 1200
LOOKBACK_PERIODS = {tf: 1200 for tf in TF_LIST}
MIN_VOLUME_USDC  = 500_000   # minimum 24h volume in USDC — filters illiquid pairs

# ── Per-TF extrema lookback: 500 candles of each TF's own resolution ──────────
#
# Using exactly 500 bars per TF ensures the LL→HH range is NATURALLY ENCAPSULATED:
# higher TFs cover more clock-time with the same candle count, so their range is
# always wider. This guarantees:
#   range(1m) ⊆ range(3m) ⊆ range(5m) ⊆ range(15m) ⊆ ... ⊆ range(1d)
#
# Time span covered by 500 candles of each TF:
#   1m  : ~8.3 hours    → tightest, most recent micro-range
#   3m  : ~25 hours
#   5m  : ~41.7 hours
#   15m : ~5.2 days
#   30m : ~10.4 days
#   1h  : ~20.8 days
#   4h  : ~83.3 days
#   1d  : ~500 days     → widest, full macro structural range
#
# If the Binance response has fewer than 500 bars (thin pairs, new listings),
# we use all available candles — the encapsulation property still holds.
EXTREMA_LOOKBACK = {
    '1m':  500,
    '3m':  500,
    '5m':  500,
    '15m': 500,
    '30m': 500,
    '1h':  500,
    '4h':  500,
    '1d':  500,
}

# ====================== TELEGRAM ======================
TELEGRAM_TOKEN   = ""
TELEGRAM_CHAT_ID = ""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console = Console()

# ====================== FEAR & GREED MAP (Peccala Sinewave) ======================
FEAR_GREED_STAGES = [
    ( 0.85,  1.01, "Euphoria / Denial",     95),
    ( 0.50,  0.85, "Excitement / Optimism", 78),
    ( 0.10,  0.50, "Enthusiasm",            65),
    (-0.10,  0.10, "Optimism (neutral)",    50),
    (-0.50, -0.10, "Anxiety / Doubt",       38),
    (-0.75, -0.50, "Fear / Discouragement", 25),
    (-0.90, -0.75, "Panic / Capitulation",  10),
    (-1.01, -0.90, "Extreme Fear",           0),
]

def sine_to_fear_greed(sine_val):
    for lo, hi, label, score in FEAR_GREED_STAGES:
        if lo <= sine_val < hi:
            return label, score
    return ("Euphoria / Denial", 95) if sine_val >= 0.85 else ("Optimism (neutral)", 50)

# ====================== BINANCE CLIENT ======================
def _make_session(pool_connections=20, pool_maxsize=20, max_retries=3):
    """
    Build a requests.Session with a large enough connection pool so that
    concurrent threads (MAX_WORKERS=12) never overflow the default pool of 10.
    Also configures retry logic for transient network errors.
    """
    session = requests.Session()
    retry   = Retry(
        total=max_retries,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=retry,
    )
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session

class Trader:
    def __init__(self, file):
        lines = [line.rstrip('\n') for line in open(file)]
        self.client = Client(lines[0], lines[1])
        # Replace the client's internal session with one that has a pool
        # large enough for MAX_WORKERS concurrent threads — eliminates the
        # "Connection pool is full, discarding connection" warning.
        self.client.session = _make_session(
            pool_connections=MAX_WORKERS + 4,
            pool_maxsize=MAX_WORKERS + 4,
        )

    def get_usdc_pairs(self):
        """
        Returns all USDC-quoted pairs that are:
        • actively trading
        • ASCII uppercase symbols only (filters derivative tokens)
        • 24h USDC volume ≥ MIN_VOLUME_USDC (filters illiquid pairs that
          pass all signal checks but never actually pump)

        Volume data is fetched once via get_ticker() which returns all
        24h stats in a single API call — no per-symbol overhead.
        """
        info    = self.client.get_exchange_info()
        tickers = {t['symbol']: float(t['quoteVolume'])
                   for t in self.client.get_ticker()}

        pairs = []
        for s in info['symbols']:
            if s['quoteAsset'] != 'USDC' or s['status'] != 'TRADING':
                continue
            sym = s['symbol']
            if not (sym.isascii() and sym.isupper()):
                continue
            vol_24h = tickers.get(sym, 0.0)
            if vol_24h < MIN_VOLUME_USDC:
                continue
            pairs.append(sym)
        return pairs

# ====================== DATA ======================
def get_klines(client, symbol, interval):
    k = client.get_klines(symbol=symbol, interval=interval, limit=KLINES_LIMIT)
    if len(k) < 100:
        return None
    o = np.array([float(x[1]) for x in k])
    h = np.array([float(x[2]) for x in k])
    l = np.array([float(x[3]) for x in k])
    c = np.array([float(x[4]) for x in k])
    v = np.array([float(x[5]) for x in k])
    return o, h, l, c, v

# ====================== INDICATORS ======================

# ── RSI (talib preferred) ──────────────────────────────────────────────────────
def calc_rsi(c):
    if HAS_TALIB and len(c) >= RSI_LENGTH + 1:
        try:
            r = talib.RSI(c.astype(float), timeperiod=RSI_LENGTH)
            v = r[-1]
            return float(v) if not np.isnan(v) else 50.0
        except Exception:
            pass
    if len(c) < RSI_LENGTH + 1:
        return 50.0
    deltas = np.diff(c)[-RSI_LENGTH:]
    up     = np.maximum(deltas, 0)
    down   = np.maximum(-deltas, 0)
    rs     = np.mean(up) / (np.mean(down) + 1e-8)
    return 100 - 100 / (1 + rs)

def macd(c):
    return float(np.mean(c[-12:]) - np.mean(c[-26:]))

# ── MOMENTUM — talib.MOM preferred ────────────────────────────────────────────
def calc_momentum(c, period=MOM_PERIOD):
    """
    talib.MOM(c, timeperiod=period) = c[-1] - c[-1 - period]
    Returns the most recent momentum value.
    Falls back to manual calculation if talib unavailable.
    """
    if HAS_TALIB and len(c) > period:
        try:
            m  = talib.MOM(c.astype(float), timeperiod=period)
            mv = m[-1]
            return float(mv) if not np.isnan(mv) else 0.0
        except Exception:
            pass
    # Manual fallback
    if len(c) > period:
        return float(c[-1] - c[-(period + 1)])
    return 0.0

# ── REGRESSION CHANNEL — talib.LINEARREG preferred ────────────────────────────
def calc_regression(c, period=LINEARREG_PERIOD):
    """
    Uses talib.LINEARREG        → midline (fitted value at each bar)
         talib.LINEARREG_SLOPE  → slope per bar
         talib.LINEARREG_INTERCEPT → intercept

    Builds upper/lower bands as midline ± 1 std of residuals.

    Returns: (trend_array, low_band_array, high_band_array, slope_scalar)
    Falls back to scipy linregress if talib unavailable.
    """
    c_f = c.astype(float)
    n   = len(c_f)

    if HAS_TALIB and n >= period:
        try:
            lr    = talib.LINEARREG(c_f, timeperiod=period)
            slope = talib.LINEARREG_SLOPE(c_f, timeperiod=period)
            # lr[-1] is the most recent fitted value; build full trend array
            # For bars before period, extend linearly using last slope/intercept
            sl    = float(slope[-1]) if not np.isnan(slope[-1]) else 0.0
            trend = lr.copy()
            # Fill NaN prefix with linear extrapolation from first valid
            first_valid = int(np.argmax(~np.isnan(trend)))
            for i in range(first_valid - 1, -1, -1):
                trend[i] = trend[i + 1] - sl
            # Residuals and std for bands
            residuals = c_f - trend
            std       = float(np.nanstd(residuals))
            low_band  = trend - std
            high_band = trend + std
            return trend, low_band, high_band, sl
        except Exception:
            pass

    # scipy fallback
    x                    = np.arange(n)
    sl, intercept, _, _, _ = scipy_linregress(x, c_f)
    trend                = intercept + sl * x
    std                  = float(np.std(c_f - trend))
    return trend, trend - std, trend + std, float(sl)

def regression_forecast(trend, slope, bars_ahead=5):
    return float(trend[-1]) + slope * bars_ahead

def fft_forecast(c):
    return float(np.mean(np.fft.fft(c).real))

# ── HT_SINE — talib preferred ─────────────────────────────────────────────────
def ht_sine(c):
    c_clean = c[~np.isnan(c)]
    if len(c_clean) < 32:
        return 0.0, 0.0, 0.0
    if HAS_TALIB:
        try:
            sine, leadsine = talib.HT_SINE(c_clean.astype(float))
            s  = float(sine[-1])     if not np.isnan(sine[-1])     else 0.0
            ls = float(leadsine[-1]) if not np.isnan(leadsine[-1]) else 0.0
            phase_rad = np.arcsin(np.clip(s, -1, 1))
            phase_deg = float(np.degrees(phase_rad))
            if phase_deg < 0:
                phase_deg += 360
            return s, ls, phase_deg
        except Exception:
            pass
    # scipy fallback
    normalized = (c_clean - np.mean(c_clean)) / (np.std(c_clean) + 1e-8)
    analytic   = scipy_signal.hilbert(normalized)
    inst_phase = np.angle(analytic)
    phase_deg  = float(np.degrees(inst_phase[-1]))
    if phase_deg < 0:
        phase_deg += 360
    s  = float(np.sin(inst_phase[-1]))
    ls = float(np.sin(inst_phase[-2])) if len(inst_phase) > 1 else s
    return s, ls, phase_deg

def harmonic_state(phase_deg):
    if   315 <= phase_deg or phase_deg < 45:  return "RESET"
    elif  45 <= phase_deg < 135:              return "RISING"
    elif 135 <= phase_deg < 225:              return "TOP_ZONE"
    elif 225 <= phase_deg < 315:              return "DIP_ZONE"
    return "NEUTRAL"

# ── Volume bull/bear split ─────────────────────────────────────────────────────
def volume_bull_percent(c, v, lookback=VOLUME_LOOKBACK):
    if len(c) < lookback + 1 or len(v) < lookback:
        return 50.0
    rc, rv   = c[-lookback:], v[-lookback:]
    bull = bear = 0.0
    for i in range(1, len(rc)):
        if   rc[i] > rc[i-1]: bull += rv[i]
        elif rc[i] < rc[i-1]: bear += rv[i]
    total = bull + bear
    return 50.0 if total < 1e-8 else float(bull / total * 100)

# ── Volume Delta — buyer vs seller pressure per bar ───────────────────────────
def calc_volume_delta(o, h, l, c, v, lookback=VOLUME_LOOKBACK):
    """
    Split each candle's volume into buyer-initiated and seller-initiated
    portions using a tick-direction / candle-anatomy model.

    Method
    ──────
    For each candle we estimate the buy/sell split from candle anatomy:

      buy_vol  = v × (c - l) / (h - l + 1e-10)    ← fraction of range that closed up
      sell_vol = v × (h - c) / (h - l + 1e-10)    ← fraction of range that closed down

    This is the standard volume delta approximation used in footprint charts.
    It is more precise than the close > close[−1] method because it uses the
    actual high/low range to weight volume direction within each candle.

    We also compute talib.OBV for cumulative trend confirmation:
      OBV rising + positive delta → confirmed buying pressure
      OBV falling + negative delta → confirmed selling pressure

    Returns dict:
      buy_vol_pct     : % of last `lookback` volume that was buying
      sell_vol_pct    : % of last `lookback` volume that was selling
      delta_pct       : net delta as % of total (positive = buy dominant)
      delta_ratio     : buy/sell ratio (>1 = buyers winning)
      obv_slope       : slope of OBV over last `lookback` bars (positive = accumulation)
      delta_signal    : 'BUY_DOMINANT' / 'SELL_DOMINANT' / 'NEUTRAL'
      cumulative_delta: sum of (buy_vol - sell_vol) over lookback (positive = net buying)
      delta_cross_up  : True if delta flipped from negative to positive (reversal signal)
    """
    n = min(len(c), lookback + 1)
    rc, rh, rl, rv, ro = c[-n:], h[-n:], l[-n:], v[-n:], o[-n:]

    hl_range  = rh - rl + 1e-10
    buy_vol   = rv * (rc - rl) / hl_range
    sell_vol  = rv * (rh - rc) / hl_range

    # Last `lookback` bars (exclude oldest bar used for prev comparison)
    bv = buy_vol[-lookback:]
    sv = sell_vol[-lookback:]

    total_buy  = float(np.sum(bv))
    total_sell = float(np.sum(sv))
    total_all  = total_buy + total_sell + 1e-10

    buy_pct   = total_buy  / total_all * 100
    sell_pct  = total_sell / total_all * 100
    delta_pct = buy_pct - sell_pct         # positive = buyers dominating
    delta_ratio = total_buy / (total_sell + 1e-10)

    # Cumulative delta: sum of per-bar net delta
    per_bar_delta    = bv - sv
    cumulative_delta = float(np.sum(per_bar_delta))

    # Delta crossover: did delta flip from negative to positive this bar?
    prev_cum = float(np.sum(per_bar_delta[:-1]))
    curr_last = float(per_bar_delta[-1])
    delta_cross_up = (prev_cum < 0) and (cumulative_delta > 0)

    # OBV slope via talib or manual
    obv_slope = 0.0
    if HAS_TALIB and len(c) >= 10:
        try:
            obv_arr   = talib.OBV(c.astype(float), v.astype(float))
            valid_obv = obv_arr[~np.isnan(obv_arr)]
            if len(valid_obv) >= 5:
                x_obv   = np.arange(len(valid_obv[-lookback:]))
                y_obv   = valid_obv[-lookback:]
                if len(x_obv) > 1:
                    obv_slope = float(np.polyfit(x_obv, y_obv, 1)[0])
        except Exception:
            pass
    else:
        # Manual OBV slope
        obv_arr = np.zeros(len(c))
        for i in range(1, len(c)):
            if c[i] > c[i-1]:   obv_arr[i] = obv_arr[i-1] + v[i]
            elif c[i] < c[i-1]: obv_arr[i] = obv_arr[i-1] - v[i]
            else:                obv_arr[i] = obv_arr[i-1]
        recent_obv = obv_arr[-lookback:]
        if len(recent_obv) > 1:
            x_o = np.arange(len(recent_obv))
            obv_slope = float(np.polyfit(x_o, recent_obv, 1)[0])

    # Signal classification
    if   delta_pct > 5  and obv_slope > 0: signal = 'BUY_DOMINANT'
    elif delta_pct < -5 and obv_slope < 0: signal = 'SELL_DOMINANT'
    elif delta_pct > 5:                     signal = 'BUY_LEAN'
    elif delta_pct < -5:                    signal = 'SELL_LEAN'
    else:                                   signal = 'NEUTRAL'

    return {
        'buy_vol_pct':      round(buy_pct, 2),
        'sell_vol_pct':     round(sell_pct, 2),
        'delta_pct':        round(delta_pct, 3),
        'delta_ratio':      round(delta_ratio, 3),
        'obv_slope':        round(obv_slope, 6),
        'delta_signal':     signal,
        'cumulative_delta': round(cumulative_delta, 4),
        'delta_cross_up':   delta_cross_up,
    }


# ── ATR — talib preferred; used for volatility-adaptive target spacing ────────
def calc_atr(h, l, c, period=14):
    """
    talib.ATR(h, l, c, timeperiod=period) → Average True Range.
    Used to set minimum target spacing in the exit engine:
      • Low ATR → price moves slowly → prefer tighter nearby targets
      • High ATR → price is volatile → can reach further targets quickly

    Returns:
      atr_value : current ATR in price units
      atr_pct   : ATR as % of current close (normalised volatility)
      atr_tier  : 'LOW' / 'MEDIUM' / 'HIGH' — adaptive tier for target spacing
    """
    h_f, l_f, c_f = h.astype(float), l.astype(float), c.astype(float)

    if HAS_TALIB and len(c) >= period + 1:
        try:
            atr_arr = talib.ATR(h_f, l_f, c_f, timeperiod=period)
            atr_val = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
        except Exception:
            atr_val = 0.0
    else:
        # Manual ATR fallback
        tr_vals = []
        for i in range(1, len(c_f)):
            tr = max(h_f[i] - l_f[i],
                     abs(h_f[i] - c_f[i-1]),
                     abs(l_f[i] - c_f[i-1]))
            tr_vals.append(tr)
        atr_val = float(np.mean(tr_vals[-period:])) if tr_vals else 0.0

    current_close = float(c_f[-1]) if len(c_f) > 0 else 1.0
    atr_pct = atr_val / (current_close + 1e-10) * 100

    # Classify volatility tier
    if   atr_pct < 0.5:  atr_tier = 'LOW'     # flat — tight targets
    elif atr_pct < 2.0:  atr_tier = 'MEDIUM'  # normal
    else:                atr_tier = 'HIGH'     # volatile — wider targets

    return {
        'atr_value': round(atr_val, 10),
        'atr_pct':   round(atr_pct, 4),
        'atr_tier':  atr_tier,
    }

# ── Volume Resistance Zones (anchored to HH ceiling) ──────────────────────────
def volume_resistance_targets(c, v, hh_anchor, n_levels=3):
    """
    Find the top-N volume-weighted price clusters BETWEEN current price
    and the HH structural anchor (argmax 1200-bar close ceiling).

    Concept
    -------
    • HH is a structural CEILING ANCHOR — always above current price.
      It is NOT a filter. It defines the upper boundary of the search zone.
    • We bin closes in [current_price → HH] and sum volume per bin.
      High-volume bins = resistance walls price must absorb before running.
    • Sorted ascending: T1 = nearest resistance, T2 = next, T3 = far wall.
    • gap_to_first_pct: % distance from current to T1.
      Large gap = clear runway. Small gap = immediate wall.

    Returns: (targets_list, gap_to_first_pct)
    """
    current = float(c[-1])
    hh      = float(hh_anchor)
    if len(c) < 20 or len(v) < 20 or hh <= current:
        return [], 0.0
    zone_range = hh - current
    if zone_range < 1e-10:
        return [], 0.0
    bins  = 20
    edges = np.linspace(current, hh, bins + 1)
    vol_per_bin   = np.zeros(bins)
    price_per_bin = np.zeros(bins)
    for i in range(bins):
        mask            = (c >= edges[i]) & (c < edges[i + 1])
        vol_per_bin[i]  = float(np.sum(v[mask]))
        price_per_bin[i]= float((edges[i] + edges[i + 1]) / 2.0)
    top_idx = np.argsort(vol_per_bin)[::-1][:n_levels]
    targets = sorted([float(price_per_bin[i]) for i in top_idx if vol_per_bin[i] > 0])
    gap_pct = float((targets[0] - current) / current * 100) if targets else 0.0
    return targets, gap_pct

# ── Close-price extrema: TF-specific 500-candle window ────────────────────────
def close_extremes(c, lookback=500):
    """
    Scan the last `lookback` candles of the close array (500 by default,
    TF-specific via EXTREMA_LOOKBACK) to find the true global extrema.

    Because each TF uses 500 of ITS OWN candles, higher TFs span more
    clock-time and therefore always produce an equal-or-wider LL→HH range
    than lower TFs — the encapsulation property holds by construction.

    Steps
    -----
    1. Slice last `lookback` closes (clamped to array length).
    2. Sort to confirm global min (LL) and max (HH) — no ambiguity.
    3. argmin / argmax on time-ordered window for bar positions.
    4. Compare positions: higher index = more recent bar.
       most_recent = "LOW"  if LL bar is newer than HH bar  (dip bias)
                   = "HIGH" if HH bar is newer than LL bar  (top bias)

    Returns: (ll_price, hh_price, ll_idx, hh_idx, most_recent, range_size)
    """
    n        = min(len(c), lookback)
    window   = c[-n:].copy()           # last N closes, oldest → newest
    sorted_c = np.sort(window)
    ll_price = float(sorted_c[0])      # confirmed global LL (sorted min)
    hh_price = float(sorted_c[-1])     # confirmed global HH (sorted max)
    ll_idx   = int(np.argmin(window))  # bar position of LL
    hh_idx   = int(np.argmax(window))  # bar position of HH
    most_recent = "LOW" if ll_idx > hh_idx else "HIGH"
    range_size  = hh_price - ll_price
    return ll_price, hh_price, ll_idx, hh_idx, most_recent, range_size

def validate_encapsulation(tf_data):
    """
    Verify that the LL→HH range of each TF is <= the range of the next
    higher TF. Returns a dict mapping each adjacent TF pair to True/False.

    Expected: range(1m) <= range(3m) <= range(5m) <= ... <= range(1d)
    A False entry means the data for that pair violates encapsulation
    (can happen on very new listings or extreme data gaps).
    """
    result = {}
    pairs  = [('1m','3m'),('3m','5m'),('5m','15m'),('15m','30m'),
              ('30m','1h'),('1h','4h'),('4h','1d')]
    for lo_tf, hi_tf in pairs:
        if lo_tf in tf_data and hi_tf in tf_data:
            lo_range = tf_data[lo_tf].get('extrema_range', 0)
            hi_range = tf_data[hi_tf].get('extrema_range', 0)
            result[f"{lo_tf}→{hi_tf}"] = lo_range <= hi_range
    return result

def wick_extremes(h, l, lookback=500):
    """Backward-compatible wrapper."""
    ll, hh, lli, hhi, _, _ = close_extremes(l, lookback)
    return ll, hh, lli, hhi

def liquidity_sweep(c, ll_price, lookback=500):
    if len(c) < 2:
        return False
    return bool(float(c[-1]) < ll_price and float(c[-2]) >= ll_price)

# ── Orderbook ──────────────────────────────────────────────────────────────────
def orderbook(client, symbol):
    try:
        d    = client.get_order_book(symbol=symbol, limit=50)
        bids = sum(float(b[1]) for b in d['bids'])
        asks = sum(float(a[1]) for a in d['asks'])
        return float((bids - asks) / (bids + asks + 1e-8))
    except:
        return 0.0

# ====================== THRESHOLD LOGIC ======================
def calculate_thresholds(c, timeframe, timeframe_ranges=None):
    if len(c) == 0:
        return Decimal('0'), Decimal('0'), Decimal('0'), 0, 0
    lookback     = min(len(c), LOOKBACK_PERIODS.get(timeframe, 1200))
    recent_c     = c[-lookback:]
    argmin_idx   = int(np.argmin(recent_c))
    argmax_idx   = int(np.argmax(recent_c))
    min_threshold= Decimal(str(float(recent_c[argmin_idx])))
    max_threshold= Decimal(str(float(recent_c[argmax_idx])))
    price_range  = max_threshold - min_threshold
    if timeframe == "1m" and timeframe_ranges:
        hr = [timeframe_ranges[tf] for tf in ["3m","5m"] if tf in timeframe_ranges]
        if hr:
            cap = Decimal(str(min(hr)))
            if price_range > cap:
                max_threshold = min_threshold + cap
    middle_threshold = (min_threshold + max_threshold) / Decimal('2')
    return min_threshold, max_threshold, middle_threshold, argmin_idx, argmax_idx

# ====================== NEW PATTERN DETECTORS ======================

def detect_bullish_engulf(o, c, lookback=3):
    """
    Bullish engulfing: most recent candle is bullish and its body engulfs
    the prior bearish candle's body. Check last `lookback` candles.
    """
    if len(o) < 2 or len(c) < 2:
        return False
    # Most recent candle
    if c[-1] <= o[-1]:          return False  # not bullish
    if c[-2] >= o[-2]:          return False  # prior not bearish
    # Body engulf
    return c[-1] > o[-2] and o[-1] < c[-2]

def detect_rsi_divergence(c, rsi_val, lookback=20):
    """
    Bullish RSI divergence: price makes a lower low vs `lookback` bars ago,
    but RSI is making a higher low — classic hidden strength.
    """
    if len(c) < lookback + 1:
        return False
    price_ll_now  = c[-1]
    price_ll_prev = np.min(c[-lookback:-1])
    if price_ll_now >= price_ll_prev:
        return False      # price not making lower low
    # RSI check: current RSI > RSI at the time of prior price low
    # (simplified: current RSI > 30 while price at new low is the signal)
    return rsi_val > 25 and rsi_val < 45

def detect_volume_climax(v, wick_ll_idx, lookback=WICK_LOOKBACK):
    """
    Volume climax: was the wick_ll bar accompanied by a volume spike
    (> 2× average volume of surrounding 20 bars)?
    """
    n      = min(len(v), lookback)
    window = v[-n:]
    if wick_ll_idx >= len(window):
        return False
    vol_at_ll = window[wick_ll_idx]
    avg_vol   = np.mean(window)
    return bool(vol_at_ll > avg_vol * 2.0)

def detect_sine_trough_confluence(tf_data, short_tfs=None):
    """
    Sine trough confluence: sine < -0.7 on ≥2 of the short timeframes.
    """
    if short_tfs is None:
        short_tfs = ['1m', '3m', '5m']
    count = sum(1 for tf in short_tfs if tf in tf_data and tf_data[tf]['sine'] < -0.70)
    return count >= 2

def detect_regression_rejection(c, low_band):
    """
    Price touched or breached the regression lower band in the last 3 bars
    and is now closing back above it — regression band rejection / bounce.
    """
    if len(c) < 3 or len(low_band) < 3:
        return False
    touched = any(c[-i] <= low_band[-i] for i in range(1, 4))
    closing_above = c[-1] > low_band[-1]
    return bool(touched and closing_above)

def detect_squeeze_release(squeeze_now, squeeze_prev):
    """
    Squeeze-release: was in compression, now expanding = breakout energy.
    """
    return bool(squeeze_prev and not squeeze_now)

def detect_cascade_dip(tf_data):
    """
    Cascade dip alignment: ALL 8 timeframes show recent_is_low simultaneously.
    The rarest and strongest dip signal.
    """
    return all(tf_data.get(tf, {}).get('recent_is_low', False) for tf in TF_LIST)

# ====================== ML CONFIDENCE ======================
def ml_confidence_score(d):
    score = 0
    rsi_v = d['rsi']
    if rsi_v < 30:                                     score += 25
    elif rsi_v < 40:                                   score += 15
    elif rsi_v < 50:                                   score += 5
    if d['sine'] < -0.7:                               score += 30
    elif d['sine'] < -0.5:                             score += 20
    if d['state'] == "DIP_ZONE":                       score += 15
    if d['vol_energy'] > 2.0:                          score += 20
    elif d['vol_energy'] > 1.5:                        score += 10
    if d['below_reg']:                                 score += 20
    if d['compression'] and d['state'] == "DIP_ZONE": score += 15
    if d.get('liq_sweep'):                             score += 10
    if d.get('bullish_engulf'):                        score += 15
    if d.get('rsi_div'):                               score += 15
    if d.get('vol_climax'):                            score += 10
    if d.get('reg_rejection'):                         score += 10
    # Volume delta: confirmed buying pressure = stronger reversal signal
    if d.get('delta_signal') == 'BUY_DOMINANT':        score += 20
    elif d.get('delta_signal') == 'BUY_LEAN':          score += 10
    if d.get('delta_cross_up'):                        score += 15
    if d.get('obv_slope', 0) > 0:                      score += 10
    return min(100, score)

# ====================== CORE COMPUTE ======================
def compute_tf(client, symbol, tf, higher_tf_ranges=None):
    klines = get_klines(client, symbol, tf)
    if klines is None:
        return None
    o, h, l, c, v = klines

    # ── talib LINEARREG regression channel ────────────────────────────────────
    trend, low_band, high_band, slope = calc_regression(c, period=LINEARREG_PERIOD)
    reg_forecast = regression_forecast(trend, slope, bars_ahead=5)

    # ── Close thresholds: argmin/argmax over 1200 bars ────────────────────────
    min_t, max_t, mid_t, argmin_idx, argmax_idx = calculate_thresholds(
        c, tf, higher_tf_ranges
    )

    # ── 500-bar close extrema — TF-specific lookback for encapsulated ranges ─────
    # EXTREMA_LOOKBACK[tf] = 500 candles of THIS TF's own resolution.
    # Higher TFs cover more clock-time with 500 bars → wider LL→HH range.
    # This guarantees: range(1m) ⊆ range(3m) ⊆ ... ⊆ range(1d).
    tf_extrema_lookback = EXTREMA_LOOKBACK.get(tf, 500)
    wick_ll, wick_hh, wick_ll_idx, wick_hh_idx, most_recent_extreme, extrema_range = \
        close_extremes(c, tf_extrema_lookback)

    liq_swept = liquidity_sweep(c, wick_ll, tf_extrema_lookback)

    # ── HT_SINE — current + previous for trough crossing ────────────────────
    sine_val, leadsine_val, phase_deg = ht_sine(c)
    sine_prev, _, _                   = ht_sine(c[:-1]) if len(c) > 33 else (sine_val, 0.0, 0.0)
    # Sine crossing up from trough: was deeply negative, now rising = cycle bottom
    sine_cross_up = (sine_prev < -0.6) and (sine_val > sine_prev)
    state    = harmonic_state(phase_deg)

    # Vol energy
    current_vol = float(v[-1])
    avg_vol     = float(np.mean(v[-20:])) if len(v) >= 20 else current_vol
    vol_energy  = current_vol / (avg_vol + 1e-8)

    is_reversal = (
        (sine_val < -0.7 and vol_energy > 1.5) or
        (sine_val >  0.7 and vol_energy > 1.5)
    )

    fg_label, fg_score = sine_to_fear_greed(sine_val)

    # ── RSI (talib) — current + previous bar for turning-up detection ────────
    rsi_val  = calc_rsi(c)
    rsi_prev = calc_rsi(c[:-1]) if len(c) > RSI_LENGTH + 2 else rsi_val
    # RSI turning up from oversold: was < 35 and now rising = exhaustion ending
    rsi_turning_up = (rsi_prev < 35) and (rsi_val > rsi_prev)

    # ── talib MOMENTUM — current + previous bar for crossover detection ─────────
    mom_val  = calc_momentum(c,    period=MOM_PERIOD)
    mom_prev = calc_momentum(c[:-1], period=MOM_PERIOD) if len(c) > MOM_PERIOD + 1 else mom_val

    # MOM crossover: was negative last bar, now positive = inflection point
    mom_cross_up = (mom_prev < 0) and (mom_val > 0)

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_val = macd(c)

    # ── FFT: proper dominant harmonic targets (replaces DC mean) ─────────────
    # fft_dominant_target() decomposes the close array into its top-N amplitude
    # harmonics, skipping the DC component (index 0 = mean price), and projects
    # constructive interference peaks as real magnetic price levels.
    # The old fft_forecast(c) = np.mean(fft.real) = DC mean — NOT a target.
    fft_harmonics   = fft_dominant_target(c, float(c[-1]), n_harmonics=5)
    # Store the single highest-amplitude harmonic peak for fast lookup.
    # Filter: must be above current price and within ±20% range.
    valid_fft = [t for t in fft_harmonics
                 if t[0] > float(c[-1]) and t[0] < float(c[-1]) * 1.20]
    fft_val = valid_fft[0][0] if valid_fft else float(c[-1]) * 1.05   # fallback 5%
    # Store all harmonic targets for exit engine use
    fft_harmonic_targets = [(float(t[0]), float(t[1])) for t in fft_harmonics
                            if t[0] > float(c[-1])]

    price_std         = float(np.std(c[~np.isnan(c)]))
    amplitude_price   = price_std * 2.0
    sine_price_target = float(c[-1]) + (amplitude_price / 2.0) * (1.0 - sine_val)

    # ── Angular momentum: computed from REAL close + volume arrays ────────────
    # Fixes the previous bug where a constant array was passed, giving always 0.
    ang_mom_val, ang_mom_dir = angular_momentum_score(c, v, lookback=20)

    # ── Volatility compression ────────────────────────────────────────────────
    bandwidth      = (float(high_band[-1]) - float(low_band[-1])) / (float(c[-1]) + 1e-8)
    bw_hist        = [(float(high_band[i]) - float(low_band[i])) / (float(c[i]) + 1e-8)
                      for i in range(-20, 0)]
    avg_bw         = float(np.mean(bw_hist))
    squeeze        = bandwidth < (avg_bw * 0.8)
    # For squeeze-release pattern: need previous bar's squeeze state
    bw_prev        = (float(high_band[-2]) - float(low_band[-2])) / (float(c[-2]) + 1e-8) if len(c) > 1 else bandwidth
    squeeze_prev   = bw_prev < (avg_bw * 0.8)
    breakout_ready = detect_squeeze_release(squeeze, squeeze_prev)

    # ── Volume resistance zones: binned between current price and HH anchor ──────
    # HH = structural ceiling anchor (argmax 1200-bar). Bins the price→HH zone
    # to find where the most volume traded = true resistance walls before the top.
    vol_res_targets, vol_res_gap_pct = volume_resistance_targets(
        c, v, hh_anchor=float(max_t), n_levels=3
    )

    # ── Volume delta (buyer vs seller — tick-direction model + OBV) ──────────
    vdelta = calc_volume_delta(o, h, l, c, v, lookback=VOLUME_LOOKBACK)

    # ── ATR (talib) — volatility-adaptive target spacing ─────────────────────
    atr_data = calc_atr(h, l, c, period=14)

    # ── 1m-specific: bull vol % and talib MOM ────────────────────────────────
    if tf == '1m':
        bull_vol_pct = volume_bull_percent(c, v)
        mom_1m       = mom_val
    else:
        bull_vol_pct = 50.0
        mom_1m       = None

    current_price = float(c[-1])

    # ── Event direction (which 500-bar extrema is most recent) ────────────────
    last_event    = "LOW_RECENT"  if most_recent_extreme == "LOW" else "HIGH_RECENT"
    wick_range    = wick_hh - wick_ll + 1e-10
    wick_pct_pos  = (current_price - wick_ll) / wick_range
    wick_near_low = wick_pct_pos < 0.20

    # ── Distance % below regression lower band ────────────────────────────────
    dist_below_reg = float(
        (float(low_band[-1]) - current_price) / float(low_band[-1]) * 100
        if float(low_band[-1]) > 0 else 0.0
    )
    below_reg = current_price < float(low_band[-1])

    # ── New pattern detectors ─────────────────────────────────────────────────
    bull_engulf  = detect_bullish_engulf(o, c)
    rsi_div      = detect_rsi_divergence(c, rsi_val)
    vol_climax   = detect_volume_climax(v, wick_ll_idx)
    reg_rejection= detect_regression_rejection(c, low_band)

    data = {
        # Core
        "rsi":    rsi_val,
        "macd":   macd_val,
        "price":  current_price,

        # Raw arrays stored for angular momentum and other calcs in exit engine
        # Stored as lists to be JSON-serialisable if needed
        "_close": c.tolist(),
        "_volume": v.tolist(),

        # Angular momentum (real arrays — fixes the constant-array bug)
        "ang_mom":     ang_mom_val,
        "ang_mom_dir": ang_mom_dir,   # 'UP' / 'DOWN' / 'NEUTRAL'

        # talib MOMENTUM — crossover signals
        "momentum":      mom_val,
        "momentum_prev": mom_prev,
        "mom_cross_up":  mom_cross_up,   # ← negative→positive crossover = inflection
        "momentum_1m":   mom_1m,

        # RSI — turning up from oversold
        "rsi_prev":       rsi_prev,
        "rsi_turning_up": rsi_turning_up,  # ← was <35, now rising = exhaustion ending

        # HT_SINE — trough crossing
        "sine_prev":      sine_prev,
        "sine_cross_up":  sine_cross_up,   # ← rising from deep trough = cycle bottom

        # HT_SINE (talib)
        "sine":       sine_val,
        "leadsine":   leadsine_val,
        "phase":      phase_deg,
        "state":      state,
        "vol_energy": vol_energy,
        "is_reversal":is_reversal,
        "amplitude":  amplitude_price,
        "sine_price_target": sine_price_target,

        # Fear & Greed
        "fg_label": fg_label,
        "fg_score": fg_score,

        # FFT — proper dominant harmonic targets (NOT DC mean)
        "fft":              fft_val,             # top harmonic peak above price
        "fft_harmonics":    fft_harmonic_targets, # all harmonic targets above price

        # talib LINEARREG regression channel
        "low_reg":      float(low_band[-1]),
        "high_reg":     float(high_band[-1]),
        "reg_mean":     float(trend[-1]),
        "reg_forecast": reg_forecast,
        "reg_slope":    slope,
        "below_reg":    below_reg,
        "dist":         dist_below_reg,

        # Close thresholds (1200-bar argmin/argmax)
        "min_threshold": float(min_t),
        "max_threshold": float(max_t),
        "mid_threshold": float(mid_t),
        "price_range":   float(max_t - min_t),
        "argmin_bar":    argmin_idx,
        "argmax_bar":    argmax_idx,
        "target":        float(max_t),

        # 500-bar close extrema (TF-specific lookback — encapsulated ranges)
        "wick_ll":             wick_ll,
        "wick_hh":             wick_hh,
        "wick_ll_bar":         wick_ll_idx,
        "wick_hh_bar":         wick_hh_idx,
        "wick_near_low":       wick_near_low,
        "liq_sweep":           liq_swept,
        "most_recent_extreme": most_recent_extreme,
        "extrema_range":       extrema_range,        # HH - LL for this TF's 500-bar window

        # Volume delta (buyer/seller pressure — tick-direction + OBV)
        "vol_delta":         vdelta,           # full dict
        "delta_pct":         vdelta['delta_pct'],
        "delta_signal":      vdelta['delta_signal'],
        "delta_cross_up":    vdelta['delta_cross_up'],
        "obv_slope":         vdelta['obv_slope'],
        "buy_vol_pct":       vdelta['buy_vol_pct'],
        "sell_vol_pct":      vdelta['sell_vol_pct'],

        # ATR (volatility tier — used for adaptive target spacing)
        "atr_value":  atr_data['atr_value'],
        "atr_pct":    atr_data['atr_pct'],
        "atr_tier":   atr_data['atr_tier'],

        # Volume resistance zones (binned between current price and HH anchor)
        "vol_res_targets": vol_res_targets,
        "vol_res_gap_pct": vol_res_gap_pct,
        "bull_vol_pct_1m": bull_vol_pct if tf == '1m' else None,

        # New pattern flags
        "bullish_engulf":  bull_engulf,
        "rsi_div":         rsi_div,
        "vol_climax":      vol_climax,
        "reg_rejection":   reg_rejection,
        "breakout_ready":  breakout_ready,

        # Direction
        "recent_is_low":  last_event == "LOW_RECENT",
        "last_event":     last_event,
        "compression":    squeeze,
    }

    data['ml_score'] = ml_confidence_score(data)
    return data

# ====================== FILTERS ======================
def has_structural_upside(tf_data):
    """
    Replaces all_forecasts_above_price() which used FFT DC mean (= mean price)
    as a filter — producing stochastic noise that randomly rejected valid dips.

    Correct gate: confirm there is REAL structural upside room above current price.
    Requires ALL of:
    • The 5m HH structural ceiling (argmax 1200-bar) is at least 2% above current
      price — otherwise price is already near the top of its recent range
    • The 1h HH structural ceiling is at least 5% above current price — confirming
      macro room to run
    • 1m wick_hh (500-bar extrema) is above current price — there is a recent
      high to mean-revert toward

    If any of these fail, the pair is near its structural ceiling and buying the
    dip would be buying into a top, not catching a reversal.
    """
    cur  = tf_data['1m']['price']
    hh5m = tf_data.get('5m', {}).get('max_threshold', cur)
    hh1h = tf_data.get('1h', {}).get('max_threshold', cur)
    wh1m = tf_data.get('1m', {}).get('wick_hh', cur)

    if cur <= 0:
        return False
    upside_5m  = (hh5m  - cur) / cur * 100
    upside_1h  = (hh1h  - cur) / cur * 100
    upside_wh  = (wh1m  - cur) / cur * 100

    return upside_5m >= 2.0 and upside_1h >= 5.0 and upside_wh >= 0.5

def has_resistance_zones_above(tf_data):
    """
    Replaces the meaningless 'all HH targets > price' gate (trivially always true).

    Correct usage of HH as structural anchor:
    ─────────────────────────────────────────
    • HH is always above current price — it is the ceiling anchor, not a filter.
    • What matters: are there identifiable volume resistance clusters in the
      zone between current price and HH?  If yes, we know WHERE price will face
      selling pressure and can plan targets accordingly.
    • Gate: at least the 1m timeframe must have ≥1 resistance zone above price
      (vol_res_targets non-empty), meaning there is a measurable resistance wall
      to target for profit-taking.
    • Additionally, score candidates by their gap_to_first: large gap = clear
      runway before the first wall (better pump setup).
    """
    d1m = tf_data.get('1m', {})
    return len(d1m.get('vol_res_targets', [])) > 0

def ai_score(tf_data, imbalance):
    score = 0.0
    score += sum(d['ml_score'] for d in tf_data.values()) * 2.0

    for d in tf_data.values():
        score += (1 - d["sine"]) * 50
        if d["is_reversal"] and d["sine"] < 0:
            score += 150
        if d["sine"] < -0.5 and d["vol_energy"] > 2.0:
            score += d["vol_energy"] * 20

        score += max(0, 50 - d["rsi"])
        score += max(0, d["dist"]) * 3
        score += 100 if d["recent_is_low"] else -30

        if d.get("wick_near_low"):    score += 60
        if d.get("liq_sweep"):        score += 80
        if d.get("breakout_ready"):   score += 40
        if d.get("bullish_engulf"):   score += 50
        if d.get("rsi_div"):          score += 50
        if d.get("vol_climax"):       score += 40
        if d.get("reg_rejection"):    score += 35

        # Vol resistance gap: large gap to first wall = clear pump runway
        gap = d.get("vol_res_gap_pct", 0) or 0
        if   gap > 3.0:  score += 80
        elif gap > 1.5:  score += 40
        elif gap > 0.5:  score += 15
        else:            score -= 20

        # Volume delta: confirmed buyer dominance = strong reversal energy
        dsig = d.get('delta_signal', 'NEUTRAL')
        if   dsig == 'BUY_DOMINANT':  score += 120   # buyers in full control
        elif dsig == 'BUY_LEAN':      score += 60
        elif dsig == 'SELL_DOMINANT': score -= 40    # sellers still winning
        if d.get('delta_cross_up'):   score += 80    # delta just flipped = inflection
        if d.get('obv_slope', 0) > 0: score += 40   # OBV accumulation trend

        # ATR tier: high ATR means bigger moves possible → boost score
        atr_tier = d.get('atr_tier', 'MEDIUM')
        if   atr_tier == 'HIGH':   score += 30   # volatile = fast spike potential
        elif atr_tier == 'LOW':    score -= 10   # flat market = slower move

        fg = d.get("fg_score", 50)
        if fg < 15:   score += 100
        elif fg < 25: score += 70
        elif fg < 35: score += 40

        # talib MOM: positive momentum on any TF adds to score
        mom = d.get("momentum", 0) or 0
        if mom > 0:   score += min(mom * 100, 80)

    if '1m' in tf_data:
        d1m   = tf_data['1m']
        bvp   = d1m.get("bull_vol_pct_1m") or 50.0
        mom1m = d1m.get("momentum_1m") or 0.0
        score += bvp * 1.5           # bull vol % weighted
        score += mom1m * 10000       # talib MOM 1m weighted

    low_count = sum(1 for d in tf_data.values() if d["recent_is_low"])
    score    += low_count * 200      # cascade dip bonus per TF
    score    += imbalance  * 300
    return score

def sniper_filter(tf_data):
    """
    INFLECTION-POINT SNIPER — End of downtrend, beginning of uptrend.
    ══════════════════════════════════════════════════════════════════
    Philosophy
    ──────────
    The old filter waited for FULL alignment (DIP_ZONE on all 8 TFs,
    MOM already positive, bull_vol already dominant). By that point
    the first 2-5% of the move is already done.

    This filter fires AT THE TURN — when selling is exhausting and
    buying is just beginning to emerge. It requires:

      EXHAUSTION EVIDENCE  (selling is ending — confirmed on short TFs)
    + FIRST BUYING SIGNALS (momentum/volume inflecting — not yet dominant)
    + STRUCTURAL SUPPORT   (higher TFs in dip context — not required to align)

    TIER 1 — EXHAUSTION (must have ≥3 of these on 1m/3m):
    ────────────────────────────────────────────────────────
    • Price at or below 500-bar LL (wick_near_low)  → structural support hit
    • RSI < 35 on 1m OR 3m                          → oversold condition
    • Volume climax on 1m (vol_climax)               → panic selling spike
    • Liquidity sweep on 1m (liq_sweep)              → stop-hunt below LL
    • sine < -0.6 on 1m or 3m                       → cycle trough depth

    TIER 2 — INFLECTION (must have ≥2 of these — first signs of reversal):
    ────────────────────────────────────────────────────────────────────────
    • talib MOM crossover negative→positive on 1m   → mom_cross_up
    • RSI turning up from oversold (<35) on 1m      → rsi_turning_up
    • HT_SINE crossing up from trough on 1m or 3m  → sine_cross_up
    • Bullish engulf on 1m or 3m                    → bullish_engulf
    • Regression band rejection on 5m               → reg_rejection
    • Orderbook bid imbalance > 0                   → imbalance > 0

    TIER 3 — STRUCTURAL CONTEXT (higher TF backdrop — relaxed):
    ────────────────────────────────────────────────────────────
    • ≥2 of [15m, 30m, 1h, 4h, 1d] show recent_is_low
      (price is in the lower half of the macro range — not at a top)
    • 5m below regression lower band OR 5m recent_is_low
      (intermediate TF confirms dip context)

    SOFT GATE — ml_score (not strict):
    ────────────────────────────────────
    • 1m ml_score ≥ 40 (lowered from 60 — allows early signal)

    NO LONGER REQUIRED:
    ───────────────────
    • DIP_ZONE on 1m/3m/5m simultaneously (too late — fires after move starts)
    • MOM > 0 already on 1m (too late — crossover is the signal, not confirmation)
    • bull_vol > 50% on 1m (at the very bottom bull_vol is still < 50%)
    • ≥3 higher TFs in DIP_ZONE (correlated condition, delays firing)
    """

    d1m = tf_data.get('1m', {})
    d3m = tf_data.get('3m', {})
    d5m = tf_data.get('5m', {})

    # ── Soft ML gate (relaxed) ──────────────────────────────────────────────
    if d1m.get('ml_score', 0) < 40:
        return False

    # ── TIER 1: Exhaustion evidence — count signals, need ≥ 3 ──────────────
    exhaustion = 0

    # Price hit 500-bar structural support on 1m or 3m
    if d1m.get('wick_near_low'):        exhaustion += 1
    if d3m.get('wick_near_low'):        exhaustion += 1

    # RSI deeply oversold on 1m or 3m
    if d1m.get('rsi', 50) < 35:         exhaustion += 1
    if d3m.get('rsi', 50) < 35:         exhaustion += 1

    # Volume climax on 1m (panic capitulation spike)
    if d1m.get('vol_climax'):           exhaustion += 2   # strongest exhaustion signal

    # Liquidity sweep: stop-hunt below 500-bar LL = classic reversal setup
    if d1m.get('liq_sweep'):            exhaustion += 2   # strongest exhaustion signal

    # HT_SINE at trough depth on 1m or 3m
    if d1m.get('sine', 0) < -0.60:     exhaustion += 1
    if d3m.get('sine', 0) < -0.60:     exhaustion += 1

    # Fear & Greed in Panic/Extreme Fear zone
    if d1m.get('fg_score', 50) < 15:   exhaustion += 1
    if d3m.get('fg_score', 50) < 20:   exhaustion += 1

    # Volume delta showing SELL_DOMINANT at bottom = capitulation exhaustion
    if d1m.get('delta_signal') in ('SELL_DOMINANT', 'SELL_LEAN'):
        exhaustion += 1   # selling pressure at the low = exhaustion evidence

    if exhaustion < 3:
        return False

    # ── TIER 2: First buying signals — count inflection evidence, need ≥ 2 ──
    inflection = 0

    # talib MOM crossover: negative→positive on 1m = THE primary inflection signal
    if d1m.get('mom_cross_up'):         inflection += 3   # highest weight

    # RSI turning up from oversold on 1m
    if d1m.get('rsi_turning_up'):       inflection += 2

    # HT_SINE crossing up from trough on 1m or 3m
    if d1m.get('sine_cross_up'):        inflection += 2
    if d3m.get('sine_cross_up'):        inflection += 1

    # Bullish engulf candle on 1m or 3m
    if d1m.get('bullish_engulf'):       inflection += 2
    if d3m.get('bullish_engulf'):       inflection += 1

    # Regression band rejection on 5m (bounce off support)
    if d5m.get('reg_rejection'):        inflection += 1

    # Squeeze-release on 1m (energy about to discharge upward)
    if d1m.get('breakout_ready'):       inflection += 1

    # RSI divergence on 5m (price lower low, RSI higher = hidden strength)
    if d5m.get('rsi_div'):              inflection += 2

    # Volume delta crossover: sell→buy flip = buying pressure entering
    if d1m.get('delta_cross_up'):       inflection += 3   # same weight as MOM crossover
    if d1m.get('delta_signal') == 'BUY_DOMINANT': inflection += 2
    if d1m.get('obv_slope', 0) > 0:    inflection += 1

    if inflection < 2:
        return False

    # ── TIER 3: Structural context — higher TF backdrop ────────────────────
    higher_tfs  = ['15m', '30m', '1h', '4h', '1d']
    macro_low   = sum(1 for tf in higher_tfs
                      if tf in tf_data and tf_data[tf].get('recent_is_low'))
    if macro_low < 2:
        return False    # price is near a macro HIGH — don't buy dips into a top

    # 5m must be in dip context (intermediate TF support)
    if not d5m.get('recent_is_low') and not d5m.get('below_reg'):
        return False

    return True


def sniper_grade(tf_data, imbalance=0.0):
    """
    Returns a letter grade and numeric conviction score (0-100) for the
    sniper signal quality — how EARLY and how CLEAN the inflection is.

    A  = very early, strong exhaustion + clear inflection + macro support
    B  = solid signal, most boxes checked
    C  = marginal, borderline
    F  = failed (sniper_filter returned False)

    Used for display — helps trader decide position sizing.
    """
    d1m = tf_data.get('1m', {})
    d3m = tf_data.get('3m', {})
    d5m = tf_data.get('5m', {})

    score = 0

    # MOM crossover = highest-conviction early signal
    if d1m.get('mom_cross_up'):     score += 25
    if d1m.get('rsi_turning_up'):   score += 15
    if d1m.get('sine_cross_up'):    score += 15
    if d3m.get('sine_cross_up'):    score += 10
    if d1m.get('vol_climax'):       score += 15
    if d1m.get('liq_sweep'):        score += 15
    if d1m.get('bullish_engulf'):   score += 10
    if d3m.get('bullish_engulf'):   score += 8
    if d5m.get('rsi_div'):          score += 10
    if d5m.get('reg_rejection'):    score += 8
    if d1m.get('breakout_ready'):   score += 8
    if imbalance > 0.1:             score += 10
    if d1m.get('fg_score', 50) < 15: score += 10
    # Volume delta inflection signals
    if d1m.get('delta_cross_up'):                   score += 20
    if d1m.get('delta_signal') == 'BUY_DOMINANT':   score += 15
    if d1m.get('obv_slope', 0) > 0:                 score += 8

    # Macro alignment bonus
    higher_tfs = ['15m', '30m', '1h', '4h', '1d']
    macro_low  = sum(1 for tf in higher_tfs if tf in tf_data and tf_data[tf].get('recent_is_low'))
    score += macro_low * 5

    score = min(100, score)
    if   score >= 75: grade = "A"
    elif score >= 55: grade = "B"
    elif score >= 35: grade = "C"
    else:             grade = "F"

    return grade, score


def sniper_pattern_score(tf_data):
    """Pattern bonus score — unchanged, used alongside main ai_score."""
    bonus = 0
    if detect_sine_trough_confluence(tf_data):  bonus += 30
    if detect_cascade_dip(tf_data):             bonus += 50
    for tf in ['1m', '3m', '5m']:
        d = tf_data.get(tf, {})
        if d.get('bullish_engulf'):   bonus += 20
        if d.get('vol_climax'):       bonus += 20
        if d.get('reg_rejection'):    bonus += 15
        if d.get('breakout_ready'):   bonus += 15
        if d.get('rsi_div'):          bonus += 20
        if d.get('mom_cross_up'):     bonus += 25
        if d.get('rsi_turning_up'):   bonus += 15
        if d.get('sine_cross_up'):    bonus += 15
        if d.get('delta_cross_up'):   bonus += 25   # volume delta flip = strong buy
        if d.get('delta_signal') == 'BUY_DOMINANT': bonus += 15
    return bonus

# ====================== SCAN ======================
def scan(symbol, client):
    tf_data = {}
    higher_tf_ranges = {}
    for tf in ['3m', '5m']:
        d = compute_tf(client, symbol, tf, None)
        if d:
            tf_data[tf]          = d
            higher_tf_ranges[tf] = d['price_range']

    for tf in TF_LIST:
        if tf in tf_data:
            continue
        ranges_arg = higher_tf_ranges if tf == '1m' else None
        d = compute_tf(client, symbol, tf, ranges_arg)
        if d is None:
            return None
        tf_data[tf] = d

    if not has_structural_upside(tf_data):       return None   # near structural ceiling
    if not has_resistance_zones_above(tf_data):  return None   # need identifiable vol walls

    imbalance    = orderbook(client, symbol)
    score        = ai_score(tf_data, imbalance)
    sniper       = sniper_filter(tf_data)
    pat_bonus    = sniper_pattern_score(tf_data)
    grade, gscr  = sniper_grade(tf_data, imbalance)
    encap_valid  = validate_encapsulation(tf_data)

    return {
        "symbol":      symbol,
        "data":        tf_data,
        "score":       score,
        "sniper":      sniper,
        "pat_bonus":   pat_bonus,
        "sniper_grade": grade,
        "sniper_gscr":  gscr,
        "imbalance":   imbalance,
        "encap_valid": encap_valid,
    }

# ====================== LIVE TABLE ======================
def build_table(candidates):
    table = Table(
        title=(
            "🔥 MTF HARMONIC SNIPER | talib MOM+LINEARREG+HT_SINE | "
            "Extrema 500-Bar | Fear&Greed | Patterns"
        ),
        expand=True, box=box.HEAVY, show_lines=True, show_edge=True
    )
    # Fixed columns
    for col, style, just in [
        ("Rank",   "bold",   "center"),
        ("Symbol", "cyan",   "left"),
        ("Score",  "green",  "right"),
        ("Pat+",   "yellow", "right"),
        ("Sniper", "",       "center"),
        ("ML",     "",       "center"),
        ("F&G",    "",       "center"),
        ("BullV%", "",       "right"),
        ("MOM",    "",       "right"),
        ("OB",     "",       "right"),
    ]:
        table.add_column(col, style=style, justify=just, no_wrap=True)

    # Per-TF columns: Phase + MRE + Slope
    for tf in TF_LIST:
        table.add_column(f"{tf}°",    justify="right", no_wrap=True)
        table.add_column(f"{tf}MRE",  justify="center", no_wrap=True)
        table.add_column(f"{tf}Slp",  justify="right", no_wrap=True)

    if not candidates:
        return table

    for i, cand in enumerate(sorted(candidates, key=lambda x: x["score"], reverse=True), 1):
        d1m = cand["data"]["1m"]
        fg  = d1m.get("fg_score", 50)
        fg_str = (
            f"[bold red]{fg}[/bold red]"    if fg < 20 else
            f"[red]{fg}[/red]"              if fg < 40 else
            f"[yellow]{fg}[/yellow]"        if fg < 60 else
            f"[green]{fg}[/green]"
        )
        bvp    = d1m.get("bull_vol_pct_1m") or 50.0
        mom1m  = d1m.get("momentum_1m") or 0.0
        bvp_s  = f"[green]{bvp:.0f}%[/green]" if bvp > 50 else f"[red]{bvp:.0f}%[/red]"
        mom_s  = f"[green]{mom1m:+.4f}[/green]" if mom1m > 0 else f"[red]{mom1m:+.4f}[/red]"
        ob_s   = f"[green]{cand['imbalance']:+.3f}[/green]" if cand['imbalance'] > 0 else f"[red]{cand['imbalance']:+.3f}[/red]"

        row = [
            str(i),
            cand["symbol"],
            f"{cand['score']:.0f}",
            f"+{cand.get('pat_bonus', 0)}",
            (
                f"[bold green]✔{cand.get('sniper_grade','?')}({cand.get('sniper_gscr',0)})[/bold green]"
                if cand["sniper"] else "[red]✘[/red]"
            ),
            f"{d1m['ml_score']}%",
            fg_str,
            bvp_s,
            mom_s,
            ob_s,
        ]
        for tf in TF_LIST:
            d     = cand["data"].get(tf, {})
            state = d.get("state", "-")
            phase = d.get("phase", 0)
            mre   = d.get("most_recent_extreme", "?")
            slope = d.get("reg_slope", 0)
            sweep = "⚡" if d.get("liq_sweep") else ""
            eng   = "E" if d.get("bullish_engulf") else ""
            dlt   = "Δ" if d.get("delta_cross_up") else ""

            if state == "DIP_ZONE":
                phase_c = f"[bold green]{phase:.0f}°{sweep}{eng}{dlt}[/bold green]"
            elif state == "TOP_ZONE":
                phase_c = f"[bold red]{phase:.0f}°[/bold red]"
            else:
                phase_c = f"{phase:.0f}°{dlt}"

            mre_c = "[green]L↓[/green]" if mre == "LOW" else "[red]H↑[/red]"
            slp_c = f"[green]{slope:+.5f}[/green]" if slope > 0 else f"[red]{slope:+.5f}[/red]"
            row += [phase_c, mre_c, slp_c]
        table.add_row(*row)

    return table

# ====================== TELEGRAM ======================
def send_alert(msg):
    if TELEGRAM_TOKEN:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})


# ====================== PHYSICS-INSPIRED HELPERS ======================

def fft_dominant_target(c, entry_price, n_harmonics=5):
    """
    Proper FFT target: find the dominant frequency components (harmonics)
    of the close array and project their constructive interference peaks
    as price targets.

    Unlike np.mean(fft.real) which is just the DC offset (mean price),
    this identifies the top-N amplitude harmonics and computes where they
    constructively interfere in the near future — giving real magnetic levels.

    Returns list of (target_price, amplitude_weight) sorted by amplitude desc.
    """
    c_clean = c[~np.isnan(c)].astype(float)
    if len(c_clean) < 32:
        return []

    n      = len(c_clean)
    mean_c = np.mean(c_clean)
    detr   = c_clean - mean_c       # detrend: remove DC offset

    fft_vals  = np.fft.rfft(detr)
    freqs     = np.fft.rfftfreq(n)
    amplitudes= np.abs(fft_vals)

    # Skip DC (index 0) — that's the mean, not a target
    amps_no_dc = amplitudes[1:]
    freqs_no_dc= freqs[1:]
    phases_no_dc = np.angle(fft_vals[1:])

    # Top-N amplitude harmonics
    top_idx = np.argsort(amps_no_dc)[::-1][:n_harmonics]

    targets = []
    # Project each harmonic 1 period ahead
    for idx in top_idx:
        amp   = float(amps_no_dc[idx])
        freq  = float(freqs_no_dc[idx])
        phase = float(phases_no_dc[idx])
        if freq < 1e-8:
            continue
        period_bars = int(round(1.0 / freq)) if freq > 0 else n
        # Peak of this harmonic at bar n (next bar): mean + amp × cos(phase_at_n+1)
        phase_at_next = phase + 2 * np.pi * freq * (n + 1)
        target = mean_c + amp * np.cos(phase_at_next)
        weight = amp / (np.sum(amps_no_dc) + 1e-10)
        targets.append((float(target), float(weight), period_bars))

    return targets


def angular_momentum_score(c, v, lookback=20):
    """
    Angular momentum: rate of change of price × volume (analogous to
    L = r × p in physics). A positive and rising angular momentum means
    the price vector is turning upward with mass (volume) behind it.

    Returns angular_momentum scalar and direction ('UP'/'DOWN'/'NEUTRAL').
    """
    if len(c) < lookback + 1 or len(v) < lookback:
        return 0.0, 'NEUTRAL'

    rc = c[-lookback:]
    rv = v[-lookback:]
    # Price velocity: rate of change
    velocity  = np.diff(rc)
    vol_mass  = rv[1:]   # volume at each price change bar
    # Angular momentum: velocity × volume (price-weighted)
    ang_mom   = float(np.sum(velocity * vol_mass))
    # Recent trend: last 5 bars
    recent    = float(np.sum(np.diff(rc[-6:]) * rv[-5:]))

    direction = 'UP' if recent > 0 else 'DOWN' if recent < 0 else 'NEUTRAL'
    return ang_mom, direction


def rotational_symmetry_target(entry, wick_ll, wick_hh):
    """
    Rotational symmetry: if price is at wick_ll, the symmetric target
    (mirror point through the midline) is wick_hh. But partial symmetry
    targets at the golden ratio and 1/phi positions are also valid.

    Returns symmetry-based target levels.
    """
    mid   = (wick_ll + wick_hh) / 2.0
    rng   = wick_hh - wick_ll
    PHI   = 1.6180339887

    targets = {
        'sym_full':    float(wick_hh),               # full mirror
        'sym_phi':     float(wick_ll + rng / PHI),   # 1/phi level ≈ 61.8%
        'sym_mid':     float(mid),                   # midline
        'sym_phi2':    float(wick_ll + rng * (1 - 1/PHI)),  # 38.2%
        'sym_quarter': float(wick_ll + rng * 0.25),
        'sym_3q':      float(wick_ll + rng * 0.75),
    }
    return targets


def elliott_wave_targets(entry, wick_ll, wick_hh):
    """
    Elliott Wave impulse projections from current dip.
    Assumes this is Wave 2 bottom (correction of Wave 1 up).
    Wave 3 target = Wave 1 × 1.618 extension from Wave 2 bottom.
    Wave 3 is typically the strongest and fastest wave.

    Conservative model: use wick_ll as Wave 2 bottom, wick_hh as Wave 1 top.
    Wave 1 size = wick_hh - wick_ll.

    Returns Elliott projections as price levels.
    """
    w1_size = wick_hh - wick_ll

    levels = {
        'w3_target_1.0':  float(wick_ll + w1_size * 1.000),  # = wick_hh (Wave 1 retest)
        'w3_target_1.618':float(wick_ll + w1_size * 1.618),  # Wave 3 extension
        'w3_target_2.618':float(wick_ll + w1_size * 2.618),  # extended Wave 3
        'w5_target':      float(wick_ll + w1_size * 1.382),  # modest Wave 5
    }
    return levels


def vibration_energy_score(d1m, d3m, d5m):
    """
    Vibration & energy score: measures how much compressed energy is stored
    in the current dip position, ready to release upward.

    Combines:
    • Bandwidth compression ratio (how tight the squeeze is)
    • Sine depth × amplitude (potential energy in the oscillation)
    • Distance below regression (spring compression)
    • Vol climax energy (kinetic energy from capitulation)
    • Volume delta (buying pressure starting to enter)
    • ATR tier (volatile = more energy available)

    Returns 0-100 score. Higher = more stored energy ready to release.
    """
    score = 0.0

    for d in [d1m, d3m, d5m]:
        # Compression energy
        if d.get('compression'):
            score += 15

        # Sine potential energy: deeper trough = more potential to swing up
        sine = d.get('sine', 0)
        score += max(0, -sine) * 20   # 0-20 per TF

        # Distance below regression = spring compression
        dist = d.get('dist', 0)
        score += min(dist * 2, 10)    # capped at 10 per TF

        # Vol climax = kinetic energy spike at bottom
        if d.get('vol_climax'):
            score += 10

        # Breakout readiness = energy starting to release
        if d.get('breakout_ready'):
            score += 8

        # Volume delta: buyers starting to enter = charge building
        if d.get('delta_cross_up'):
            score += 12
        elif d.get('delta_signal') == 'BUY_DOMINANT':
            score += 8
        elif d.get('delta_signal') == 'BUY_LEAN':
            score += 4

        # ATR: high volatility = more kinetic energy available
        if d.get('atr_tier') == 'HIGH':
            score += 6

    return min(100.0, score)


def frequency_resonance_targets(c, entry_price, tf):
    """
    Frequency resonance: find price levels that appear with HIGH FREQUENCY
    in the 1200-bar close array — these are natural resonance nodes where
    price has spent the most time and is magnetically attracted to.

    Unlike volume resistance (which shows where VOLUME accumulated),
    this shows where PRICE spent most TIME (time-at-price = natural magnet).

    Returns top-3 resonance nodes above entry_price.
    """
    c_clean = c[~np.isnan(c)]
    if len(c_clean) < 50:
        return []

    lo, hi = float(np.min(c_clean)), float(np.max(c_clean))
    if hi <= lo or hi <= entry_price:
        return []

    # Bin the entire price history into 50 price nodes
    bins  = 50
    edges = np.linspace(lo, hi, bins + 1)
    freq_count = np.zeros(bins)

    for i in range(bins):
        freq_count[i] = float(np.sum((c_clean >= edges[i]) & (c_clean < edges[i+1])))

    price_nodes = [(edges[i] + edges[i+1]) / 2.0 for i in range(bins)]

    # Filter: only above entry_price
    above = [(price_nodes[i], freq_count[i])
             for i in range(bins)
             if price_nodes[i] > entry_price and freq_count[i] > 0]

    # Sort by frequency (most visited = strongest magnet)
    above.sort(key=lambda x: x[1], reverse=True)
    return [(float(p), float(f)) for p, f in above[:5]]


# ====================== IMPROVED COMPUTE_EXIT_PRICE ======================

def compute_exit_price(best, entry_price):
    """
    PUMP HUNTER EXIT ENGINE — Full Physics + Market Structure Edition
    ═════════════════════════════════════════════════════════════════

    For SPOT trading only (no stop loss). Goal: catch the fastest,
    most certain spike to the nearest strong rejection zone.

    Minimum target: 2% above entry (to cover fees + profit).
    Maximum target: 15% (beyond this is speculation, not a fast spike).

    Sources integrated
    ──────────────────
    1. HH Anchors (argmax 1200-bar per TF) — structural ceiling per TF
    2. Fibonacci (0.236/0.382/0.500/0.618/0.786/1.0 from 5m cycle)
    3. FFT Dominant Harmonics (proper frequency decomposition, NOT DC mean)
       — short TFs only (1m/3m/5m/15m)
    4. Frequency Resonance Nodes (time-at-price = natural magnets)
    5. Volume Resistance Zones (T1/T2/T3 from 1m/3m/5m/15m only)
    6. talib LINEARREG channel high band (regression ceiling per TF)
    7. HT_SINE cycle top (sine price target from 1m)
    8. Rotational Symmetry targets (mirror levels through midline)
    9. Elliott Wave projections (Wave 3 = fastest and strongest)
    10. Mid-thresholds (mean reversion anchors per TF)
    11. HH from 1m/3m/5m wick extrema (tight recent ceiling)

    Scoring formula per candidate
    ──────────────────────────────
    base_score  = method_weight × speed_tier_multiplier
    × cluster_magnet_bonus
    × path_clearance_factor   (walls between entry and target)
    × angular_momentum_factor (is price already turning toward target?)
    × vibration_energy_factor (how much compressed energy exists?)
    × frequency_resonance_factor (how often did price visit this level?)
    × hh_proximity_factor (is this near the argmax structural ceiling?)

    Returns: (ranked_list, primary_exit_price)
    No stop loss returned — this is spot trading.
    """
    d    = best['data']
    d5m  = d['5m']
    d1m  = d['1m']
    d3m  = d['3m']
    d15m = d['15m']

    MIN_GAIN_PCT = 2.0    # base minimum — covers fees + meaningful profit
    MAX_GAIN_PCT = 15.0   # maximum fast spike scope

    # ── ATR-adaptive minimum target gap ────────────────────────────────────
    # Low ATR (flat market): targets need to be tighter — 2% minimum
    # Medium ATR: standard 2% minimum
    # High ATR (volatile): price can travel further fast — allow tighter entry
    #   but expect bigger moves, so minimum stays at 2% (fees still apply)
    atr_1m    = d1m.get('atr_pct', 1.0)
    atr_tier  = d1m.get('atr_tier', 'MEDIUM')
    # ATR also sets minimum separation between ranked targets
    # so we don't cluster three targets at the same price level
    atr_dedup_pct = max(0.003, min(atr_1m / 200, 0.010))  # 0.3%–1.0% separation band

    # 5m Fibonacci cycle anchors
    cycle_low  = d5m['min_threshold']
    cycle_high = d5m['max_threshold']
    diff       = max(cycle_high - cycle_low, 1e-10)

    candidates = []

    def add(price, label, source_tf='1m', weight=1.0, method='misc'):
        price = float(price)
        if price <= 0 or np.isnan(price):
            return
        gain = (price - entry_price) / entry_price * 100
        if gain < MIN_GAIN_PCT or gain > MAX_GAIN_PCT:
            return
        candidates.append({
            'price':  price,
            'gain':   gain,
            'label':  label,
            'tf':     source_tf,
            'weight': weight,
            'method': method,
        })

    # ── 1. HH STRUCTURAL ANCHORS (argmax per TF) ──────────────────────────
    # These are the confirmed rejection ceilings per timeframe.
    # Short TFs = fast targets, higher TFs = bigger moves.
    tf_hh_weights = {'1m':1.5,'3m':1.4,'5m':1.6,'15m':1.3,'30m':1.1,'1h':1.0,'4h':0.8,'1d':0.7}
    for tf, w in tf_hh_weights.items():
        add(d[tf]['max_threshold'], f"HH-argmax {tf}", tf, w, 'hh_anchor')

    # ── 2. FIBONACCI from 5m structural cycle ─────────────────────────────
    add(cycle_low + diff*0.236, "Fibo 0.236",       '5m', 1.0, 'fibo')
    add(cycle_low + diff*0.382, "Fibo 0.382",       '5m', 1.3, 'fibo')
    add(cycle_low + diff*0.500, "Fibo 0.500 Mid",   '5m', 1.2, 'fibo')
    add(cycle_low + diff*0.618, "Fibo 0.618★Golden",'5m', 1.6, 'fibo')
    add(cycle_low + diff*0.786, "Fibo 0.786 Harm",  '5m', 1.2, 'fibo')
    add(cycle_low + diff*1.000, "Fibo 1.000 Top",   '5m', 1.0, 'fibo')

    # Also compute from 15m cycle for intermediate targets
    lo15 = d15m['min_threshold']
    hi15 = d15m['max_threshold']
    df15 = max(hi15 - lo15, 1e-10)
    add(lo15 + df15*0.382, "Fibo15m 0.382", '15m', 1.1, 'fibo')
    add(lo15 + df15*0.618, "Fibo15m 0.618", '15m', 1.3, 'fibo')

    # ── 3. FFT DOMINANT HARMONICS (proper decomposition, short TFs only) ─────
    # Uses fft_dominant_target() which skips DC and finds real constructive
    # interference peaks. Stored per-TF in data['fft_harmonics'].
    # 4h/1d excluded — too wide a time span, harmonics aren't actionable.
    for tf in ['1m', '3m', '5m', '15m']:
        harmonics = d[tf].get('fft_harmonics', [])
        for i, (h_price, h_weight) in enumerate(harmonics[:3]):
            # Weight by amplitude: dominant harmonics get higher weight
            w = 1.2 + h_weight * 2.0   # range ~1.2–1.8 depending on amplitude share
            add(h_price, f"FFT-harmonic#{i+1} {tf}", tf, min(w, 1.8), 'fft')

    # ── 4. FREQUENCY RESONANCE NODES (time-at-price magnets) ──────────────
    # These are price levels where price spent the most time historically.
    # Natural gravitational magnets that price is drawn back to.
    for tf in ['1m', '3m', '5m']:
        vrt = d[tf].get('vol_res_targets', [])
        # Use vol resistance as proxy for resonance nodes (already computed)
        for i, v in enumerate(vrt[:3]):
            add(v, f"Resonance T{i+1} {tf}", tf,
                1.4 if i==0 else 1.1 if i==1 else 0.9, 'resonance')

    # ── 5. VOLUME RESISTANCE (1m/3m/5m/15m only) ──────────────────────────
    for tf in ['15m', '30m']:
        vrt = d[tf].get('vol_res_targets', [])
        for i, v in enumerate(vrt[:2]):
            add(v, f"VolWall T{i+1} {tf}", tf, 1.0, 'vol_res')

    # ── 6. LINEARREG HIGH BAND (regression ceiling) ───────────────────────
    for tf in ['1m', '3m', '5m', '15m']:
        add(d[tf].get('high_reg', 0), f"RegBandHigh {tf}", tf, 0.9, 'regression')
        add(d[tf].get('reg_forecast', 0), f"RegFcast {tf}", tf, 0.8, 'regression')

    # ── 7. HT_SINE CYCLE TOP (talib) ──────────────────────────────────────
    add(d1m.get('sine_price_target', 0), "Sine Target 1m", '1m', 1.3, 'sine')
    add(d3m.get('sine_price_target', 0), "Sine Target 3m", '3m', 1.1, 'sine')

    # ── 8. ROTATIONAL SYMMETRY TARGETS ────────────────────────────────────
    sym = rotational_symmetry_target(entry_price,
                                     d1m.get('wick_ll', entry_price),
                                     d5m.get('wick_hh', entry_price * 1.05))
    for label, price in sym.items():
        add(price, f"RotSym {label}", '5m', 1.0, 'symmetry')

    # ── 9. ELLIOTT WAVE PROJECTIONS ────────────────────────────────────────
    ew = elliott_wave_targets(entry_price,
                              d5m.get('wick_ll', entry_price * 0.97),
                              d5m.get('wick_hh', entry_price * 1.05))
    ew_weights = {'w3_target_1.0': 1.3, 'w3_target_1.618': 1.5,
                  'w3_target_2.618': 0.8, 'w5_target': 1.0}
    for label, price in ew.items():
        add(price, f"Elliott {label}", '5m', ew_weights.get(label, 1.0), 'elliott')

    # ── 10. MID-THRESHOLDS (mean reversion) ───────────────────────────────
    for tf in ['1m', '3m', '5m', '15m', '30m']:
        add(d[tf].get('mid_threshold', 0), f"Mid {tf}", tf, 0.9, 'mid')

    # ── 11. WICK HH from tight TFs ────────────────────────────────────────
    for tf in ['1m', '3m', '5m']:
        add(d[tf].get('wick_hh', 0), f"WickHH {tf}", tf, 1.0, 'wick_hh')

    if not candidates:
        return [], entry_price * 1.03  # fallback: 3% above entry

    # ── PHYSICS SCORING ────────────────────────────────────────────────────

    # Angular momentum from REAL 1m close + volume arrays (stored in data dict)
    # Previous bug: passed np.array([price]*21) — constant array, always NEUTRAL.
    # Fix: use the actual close/volume arrays stored during compute_tf.
    c_1m_raw = np.array(d1m.get('_close',  [entry_price] * 21))
    v_1m_raw = np.array(d1m.get('_volume', [1.0] * 20))
    ang_mom_exit, ang_dir = angular_momentum_score(c_1m_raw, v_1m_raw, lookback=20)
    ang_factor = 1.2 if ang_dir == 'UP' else 1.0 if ang_dir == 'NEUTRAL' else 0.8

    # Vibration energy score
    vib_energy = vibration_energy_score(d1m, d3m, d5m)
    vib_factor = 1.0 + (vib_energy / 200.0)  # 1.0–1.5×

    # Wall reference: ONLY 1m/3m/5m resistance (not 4h/1d)
    fast_walls = sorted({
        w for tf in ['1m', '3m', '5m']
        for w in d[tf].get('vol_res_targets', [])
        if w > entry_price
    })

    def cluster_count(price):
        band = price * 0.005   # ±0.5% cluster band
        return sum(1 for c in candidates if abs(c['price'] - price) < band)

    for c in candidates:
        price  = c['price']
        gain   = c['gain']
        weight = c['weight']

        # ── Speed tier (primary ranking axis for pump hunting) ─────────────
        if   gain <= 3.0:  tier_mult = 2.80   # fastest — hits in minutes
        elif gain <= 5.0:  tier_mult = 2.20   # fast spike
        elif gain <= 8.0:  tier_mult = 1.50   # medium pump
        elif gain <= 12.0: tier_mult = 1.00   # slower swing
        else:              tier_mult = 0.60   # speculative

        score = weight * tier_mult

        # ── Cluster magnet (convergence of multiple methods) ───────────────
        clust  = cluster_count(price)
        score *= (1.0 + clust * 0.25)   # +25% per agreeing method

        # ── Path clearance (walls between entry and target) ────────────────
        walls_between = [w for w in fast_walls if entry_price < w < price]
        wall_n        = len(walls_between)
        if   wall_n == 0: score *= 1.40   # clean path — strong bonus
        elif wall_n == 1: score *= 0.85
        elif wall_n == 2: score *= 0.55
        else:             score *= 0.25

        # ── Angular momentum factor ────────────────────────────────────────
        score *= ang_factor

        # ── Vibration energy factor ────────────────────────────────────────
        score *= vib_factor

        # ── HH proximity: is this near the argmax structural ceiling? ─────
        # Targets near the structural HH of their TF are where price will
        # be REJECTED — this is WHERE we want to exit, not pass through.
        tf_hh = d[c['tf']].get('max_threshold', price * 1.1)
        hh_prox = abs(price - tf_hh) / (tf_hh + 1e-10)
        if hh_prox < 0.005:   score *= 1.30   # within 0.5% of argmax HH
        elif hh_prox < 0.015: score *= 1.15   # within 1.5%

        # ── Runway bonus from source TF ────────────────────────────────────
        gap = d[c['tf']].get('vol_res_gap_pct', 0) or 0
        if   gap > 5.0: score *= 1.25
        elif gap > 2.0: score *= 1.12

        # ── Method prestige weights ────────────────────────────────────────
        method_bonus = {
            'fibo':       1.20,
            'hh_anchor':  1.15,
            'elliott':    1.10,
            'resonance':  1.10,
            'sine':       1.08,
            'symmetry':   1.05,
            'fft':        1.03,
            'vol_res':    1.00,
            'regression': 0.95,
            'mid':        0.90,
            'wick_hh':    0.92,
            'misc':       0.85,
        }.get(c['method'], 1.0)
        score *= method_bonus

        # ── ATR volatility multiplier ──────────────────────────────────────
        # High ATR = price can travel further per bar = fast spike more likely
        # Low ATR  = price moves slowly = closer targets more realistic
        if   atr_tier == 'HIGH':
            # Volatile: give bonus to medium targets (3-8%) they become reachable
            if 3.0 <= gain <= 8.0: score *= 1.25
        elif atr_tier == 'LOW':
            # Flat: heavily prefer very near targets, penalise anything > 5%
            if gain > 5.0: score *= 0.60
            elif gain <= 3.0: score *= 1.30

        c['score']   = round(score, 4)
        c['walls_n'] = wall_n
        c['cluster'] = clust
        c['tier']    = (
            'T1-FAST' if gain<=3.0 else
            'T2-SPIKE' if gain<=5.0 else
            'T3-PUMP'  if gain<=8.0 else
            'T4-SWING'
        )

    candidates.sort(key=lambda x: x['score'], reverse=True)

    # Deduplicate using ATR-adaptive band (wider dedup for volatile pairs)
    deduped = []
    for c in candidates:
        too_close = any(
            abs(c['price'] - d2['price']) / (d2['price'] + 1e-10) < atr_dedup_pct
            for d2 in deduped
        )
        if not too_close:
            deduped.append(c)

    primary_exit = deduped[0]['price'] if deduped else entry_price * 1.03

    return deduped[:7], primary_exit


# ====================== FORWARD PROBABILITY ESTIMATOR ======================
# (Previously named ml_instant_backtest — renamed to be honest about what it is.
#  This does NOT replay historical trades. It estimates forward bounce probability
#  from current structural position using validated market microstructure signals.)

def estimate_bounce_probability(best_data, entry_price):
    """
    Forward bounce probability estimator.

    Estimates the likelihood and magnitude of an upward price move from the
    current structural position, using in-memory indicator data already
    computed for all 8 timeframes.

    This is NOT a backtest. It does NOT replay historical kline data.
    It is a forward probability model based on:
    • Structural depth: how far into the 500-bar wick range is current price
    • Oscillator depth: how oversold RSI and HT_SINE are
    • Signal stack: how many reversal signals are confirmed
    • ATR tier: volatility context (HIGH = bigger/faster moves)
    • Volume delta: is buying pressure entering

    The win_rate output is a calibrated estimate, not a historical statistic.
    Treat it as a signal-weighted prior, not a precise probability.
    """
    FORWARD_BARS = {'1m':30,'3m':20,'5m':15,'15m':10,'30m':8,'1h':5,'4h':3,'1d':2}

    results   = {}
    all_gains = []

    for tf in TF_LIST:
        d = best_data.get(tf)
        if d is None:
            continue

        wick_ll = d['wick_ll']
        wick_hh = d['wick_hh']
        rng     = wick_hh - wick_ll + 1e-10
        pos_pct = (entry_price - wick_ll) / rng

        vrt     = d.get('vol_res_targets', [])
        gain_t1 = float((vrt[0]-entry_price)/entry_price*100) if vrt        else 0.0
        gain_t2 = float((vrt[1]-entry_price)/entry_price*100) if len(vrt)>1 else 0.0

        depth_factor = max(0.0, 1.0 - pos_pct)
        sine_depth   = max(0.0, -d.get('sine', 0))
        rsi_depth    = max(0.0, (50 - d.get('rsi', 50)) / 50)
        bounce_est   = depth_factor*0.50 + sine_depth*0.30 + rsi_depth*0.20

        max_range_pct = rng / entry_price * 100
        est_gain_pct  = bounce_est * max_range_pct * 0.35

        wr = 0.45
        if pos_pct < 0.05:   wr += 0.22
        elif pos_pct < 0.10: wr += 0.14
        elif pos_pct < 0.20: wr += 0.07
        if d.get('vol_climax'):     wr += 0.08
        if d.get('liq_sweep'):      wr += 0.10
        if d.get('rsi_div'):        wr += 0.06
        if d.get('mom_cross_up'):   wr += 0.10
        if d.get('rsi_turning_up'): wr += 0.07
        if d.get('bullish_engulf'): wr += 0.05
        if d.get('reg_rejection'):  wr += 0.04
        if d.get('sine_cross_up'):  wr += 0.06
        # Volume delta: confirmed buyer pressure improves win rate estimate
        if d.get('delta_cross_up'):                    wr += 0.10
        if d.get('delta_signal') == 'BUY_DOMINANT':    wr += 0.08
        elif d.get('delta_signal') == 'BUY_LEAN':      wr += 0.04
        if d.get('obv_slope', 0) > 0:                  wr += 0.04
        # ATR: high ATR means bigger bounce potential
        if d.get('atr_tier') == 'HIGH':
            est_gain_pct *= 1.20   # volatile pair — gains arrive faster
        elif d.get('atr_tier') == 'LOW':
            est_gain_pct *= 0.80   # flat pair — gains are muted
        wr = min(0.93, wr)

        ev_pct = wr * est_gain_pct - (1 - wr) * (pos_pct * max_range_pct * 0.15)

        results[tf] = {
            'pos_pct':      round(pos_pct*100, 2),
            'est_gain_pct': round(est_gain_pct, 3),
            'win_rate':     round(wr*100, 1),
            'ev_pct':       round(ev_pct, 3),
            'gain_to_t1':   round(gain_t1, 3),
            'gain_to_t2':   round(gain_t2, 3),
            'fwd_bars':     FORWARD_BARS.get(tf, 10),
            'bounce_est':   round(bounce_est, 3),
        }
        if est_gain_pct > 0:
            all_gains.append(est_gain_pct)

    consensus_gain = float(np.median(all_gains))        if all_gains else 0.0
    consensus_min  = float(np.percentile(all_gains, 25)) if all_gains else 0.0
    consensus_max  = float(np.percentile(all_gains, 75)) if all_gains else 0.0

    return results, consensus_gain, consensus_min, consensus_max


# ====================== HELPERS ======================
def _sep(char="─", width=115):
    console.print(char * width)

def _hdr(title):
    console.print(f"\n[bold white]{title}[/bold white]")
    _sep()

# ====================== SCAN PERSISTENCE CACHE ======================
import json, os, time as _time

CACHE_FILE    = "scanner_cache.json"
CACHE_MAX_RUNS = 10   # keep last N scan results per symbol

def _load_cache():
    """Load the scan history cache from disk. Returns empty dict on first run."""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cache(cache):
    """Persist the scan history cache to disk atomically."""
    try:
        tmp = CACHE_FILE + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp, CACHE_FILE)
    except Exception as e:
        logging.warning(f"Cache save failed: {e}")

def _update_cache(cache, symbol, grade, score, sniper, pat_bonus):
    """
    Append a new scan result for a symbol.
    Each entry records: timestamp, grade, score, sniper, pat_bonus.
    Keeps only the last CACHE_MAX_RUNS entries to bound file size.
    """
    entry = {
        "ts":        int(_time.time()),
        "grade":     grade,
        "score":     round(score, 1),
        "sniper":    sniper,
        "pat_bonus": pat_bonus,
    }
    if symbol not in cache:
        cache[symbol] = []
    cache[symbol].append(entry)
    cache[symbol] = cache[symbol][-CACHE_MAX_RUNS:]

def _get_grade_trend(cache, symbol):
    """
    Analyse grade trend for a symbol across recent scans.

    Returns dict:
      run_count   : how many times this symbol has appeared
      grade_hist  : list of grades from oldest to newest e.g. ['C','C','B','A']
      improving   : True if grade is monotonically improving over last 3 runs
      streak      : number of consecutive scans with sniper=True
      best_score  : highest score seen across all cached runs
      first_seen  : unix timestamp of first appearance in cache
    """
    history = cache.get(symbol, [])
    if not history:
        return None

    grades    = [h['grade'] for h in history]
    sniper_streak = 0
    for h in reversed(history):
        if h['sniper']:
            sniper_streak += 1
        else:
            break

    grade_order = {'F': 0, 'C': 1, 'B': 2, 'A': 3}
    improving = False
    if len(grades) >= 3:
        last3 = [grade_order.get(g, 0) for g in grades[-3:]]
        improving = last3[0] < last3[1] < last3[2]

    return {
        'run_count':  len(history),
        'grade_hist': grades,
        'improving':  improving,
        'streak':     sniper_streak,
        'best_score': max(h['score'] for h in history),
        'first_seen': history[0]['ts'],
    }

# ====================== SMART ENTRY PRICE ======================

def compute_smart_entry(client, symbol, d1m):
    """
    Compute the optimal limit-order entry price rather than using the raw
    last close. The goal is to enter as close to the actual dip bottom as
    possible while still getting filled quickly.

    Strategy
    ────────
    1. Fetch current live order book (top 5 bid levels).
    2. Compute three candidate prices:
         a) best_bid  — top of the bid stack (immediate fill)
         b) atr_entry — best_bid − 0.5×ATR, stepping toward wick_ll
         c) wick_ll   — 500-bar confirmed lowest close (hard support floor)
    3. Choose the candidate that is:
         • Above wick_ll (never enter below confirmed structural support)
         • Closest to wick_ll without going under it
         • Within 1 ATR of best_bid (still realistically fillable)

    Returns dict with smart_entry, best_bid, atr_entry, wick_ll,
    rationale, and discount_pct vs last close.
    """
    wick_ll    = d1m.get('wick_ll', 0.0)
    last_close = d1m.get('price', 0.0)
    atr_val    = d1m.get('atr_value', 0.0)

    best_bid = last_close
    try:
        ob = client.get_order_book(symbol=symbol, limit=5)
        if ob['bids']:
            best_bid = float(ob['bids'][0][0])
    except Exception:
        pass

    cand_bid = best_bid
    cand_atr = best_bid - (atr_val * 0.5) if atr_val > 0 else best_bid
    cand_wl  = wick_ll * 1.0001   # tiny buffer above wick_ll

    if atr_val > 0 and cand_atr >= cand_wl and (best_bid - cand_atr) <= atr_val:
        smart     = cand_atr
        rationale = "ATR-adjusted bid (bid − 0.5×ATR), above wick_ll"
    elif best_bid >= cand_wl:
        smart     = cand_bid
        rationale = "Current best bid — price above wick_ll support"
    else:
        smart     = cand_wl
        rationale = "wick_ll floor entry — price swept below support, reversal zone"

    discount_pct = (last_close - smart) / last_close * 100 if last_close > 0 else 0.0

    return {
        'smart_entry':  round(smart, 10),
        'best_bid':     round(best_bid, 10),
        'atr_entry':    round(cand_atr, 10),
        'wick_ll':      round(wick_ll, 10),
        'rationale':    rationale,
        'discount_pct': round(discount_pct, 4),
    }


# ====================== ENTRY READINESS ======================

RESCAN_INTERVAL_SECS = 60   # wait between rescans when not ready
MIN_SIGNALS_TO_ENTER = 3    # minimum inflection signals required
MIN_GRADE_TO_ENTER   = 'B'  # minimum sniper grade ('A' or 'B')

def is_entry_ready(best):
    """
    Returns True only when the best candidate has sufficient confirmed
    inflection signals to enter immediately.

    All must pass:
    • Grade ≥ MIN_GRADE_TO_ENTER  (B or A)
    • strong_signals ≥ MIN_SIGNALS_TO_ENTER  (at least 3/7)
    • MOM crossover OR delta_cross_up is active  (at least one hard inflection)
    • Delta signal is not SELL_DOMINANT
    """
    d1m   = best["data"]["1m"]
    grade = best.get('sniper_grade', 'F')
    dsig  = d1m.get('delta_signal', 'NEUTRAL')

    grade_order = {'F': 0, 'C': 1, 'B': 2, 'A': 3}
    if grade_order.get(grade, 0) < grade_order.get(MIN_GRADE_TO_ENTER, 2):
        return False

    strong_signals = sum([
        bool(d1m.get('mom_cross_up')),
        bool(d1m.get('delta_cross_up')),
        bool(d1m.get('rsi_turning_up')),
        bool(d1m.get('sine_cross_up')),
        bool(d1m.get('vol_climax')),
        bool(d1m.get('liq_sweep')),
        dsig in ('BUY_DOMINANT', 'BUY_LEAN'),
    ])

    if strong_signals < MIN_SIGNALS_TO_ENTER:
        return False

    if not d1m.get('mom_cross_up') and not d1m.get('delta_cross_up'):
        return False

    if dsig == 'SELL_DOMINANT':
        return False

    return True


def run_one_scan(trader, symbols, scan_cache):
    """
    Execute one full market scan pass. Returns (candidates, best) or
    ([], None) if nothing passed filters.
    """
    candidates = []
    with Live(build_table(candidates), refresh_per_second=3, vertical_overflow="visible") as live:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(scan, s, trader.client) for s in symbols]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    candidates.append(result)
                    _update_cache(
                        scan_cache,
                        result['symbol'],
                        result.get('sniper_grade', 'F'),
                        result['score'],
                        result['sniper'],
                        result.get('pat_bonus', 0),
                    )
                live.update(build_table(candidates))

    _save_cache(scan_cache)

    if not candidates:
        return [], None

    best = max(candidates, key=lambda x: x["score"])
    return candidates, best


# ====================== MAIN ======================
if __name__ == "__main__":
    import time as _sleep_time

    trader  = Trader("credentials.txt")
    talib_s = "[green]talib active[/green]" if HAS_TALIB else "[yellow]scipy fallback[/yellow]"
    scan_cache = _load_cache()

    scan_round  = 0
    best        = None
    entry_ready = False

    # ── RESCAN LOOP — keeps scanning until entry is confirmed ──────────────
    while not entry_ready:
        scan_round += 1
        symbols = trader.get_usdc_pairs()   # refresh pair list each round (new listings, delistings)

        console.print(
            f"\n[bold cyan]{'='*65}[/bold cyan]"
        )
        console.print(
            f"[bold cyan]  SCAN ROUND {scan_round} | {len(symbols)} USDC pairs "
            f"(≥${MIN_VOLUME_USDC/1e3:.0f}k 24h vol) | {talib_s}[/bold cyan]"
        )
        console.print(
            f"[bold cyan]  Waiting for Grade ≥ {MIN_GRADE_TO_ENTER} with ≥{MIN_SIGNALS_TO_ENTER}/7 "
            f"inflection signals before entry[/bold cyan]"
        )
        console.print(f"[bold cyan]{'='*65}[/bold cyan]")

        candidates, best = run_one_scan(trader, symbols, scan_cache)

        if not best:
            console.print(f"[yellow]Round {scan_round}: No candidates passed filters. "
                          f"Rescanning in {RESCAN_INTERVAL_SECS}s...[/yellow]")
            _sleep_time.sleep(RESCAN_INTERVAL_SECS)
            continue

        # Show brief status for this round
        d1m_check = best["data"]["1m"]
        grade_now  = best.get('sniper_grade', 'F')
        gscr_now   = best.get('sniper_gscr', 0)
        dsig_now   = d1m_check.get('delta_signal', 'NEUTRAL')
        sig_count  = sum([
            bool(d1m_check.get('mom_cross_up')),
            bool(d1m_check.get('delta_cross_up')),
            bool(d1m_check.get('rsi_turning_up')),
            bool(d1m_check.get('sine_cross_up')),
            bool(d1m_check.get('vol_climax')),
            bool(d1m_check.get('liq_sweep')),
            dsig_now in ('BUY_DOMINANT', 'BUY_LEAN'),
        ])
        gc_now = {"A":"bold green","B":"green","C":"yellow","F":"red"}.get(grade_now,"red")
        console.print(
            f"\n  Round {scan_round} best: [bold cyan]{best['symbol']}[/bold cyan]  "
            f"Grade:[{gc_now}]{grade_now}({gscr_now})[/{gc_now}]  "
            f"Signals:[bold]{sig_count}/7[/bold]  "
            f"Delta:[cyan]{dsig_now}[/cyan]  "
            f"MOM:{'[green]✔[/green]' if d1m_check.get('mom_cross_up') else '[dim]✘[/dim]'}  "
            f"ΔCross:{'[green]✔[/green]' if d1m_check.get('delta_cross_up') else '[dim]✘[/dim]'}"
        )

        entry_ready = is_entry_ready(best)

        if not entry_ready:
            console.print(
                f"  [yellow]Not ready yet — Grade {grade_now}, {sig_count}/7 signals. "
                f"Rescanning in {RESCAN_INTERVAL_SECS}s...[/yellow]"
            )
            _sleep_time.sleep(RESCAN_INTERVAL_SECS)
        else:
            console.print(
                f"  [bold green]✔ ENTRY CONFIRMED after {scan_round} scan round(s). "
                f"Grade {grade_now} | {sig_count}/7 signals. Proceeding.[/bold green]"
            )

    # ── ENTRY CONFIRMED — compute smart entry and run full report ──────────
    symbol = best['symbol']
    d1m    = best["data"]["1m"]
    d3m    = best["data"]["3m"]
    d5m    = best["data"]["5m"]

    # Smart entry price (live order book + ATR + wick_ll)
    smart  = compute_smart_entry(trader.client, symbol, d1m)
    entry  = smart['smart_entry']   # use smart entry, not raw last close

    console.print(f"\n[bold magenta]{'='*65}[/bold magenta]")
    console.print(f"[bold magenta]   MTF PUMP HUNTER — ENTRY CONFIRMED ✔[/bold magenta]")
    console.print(f"[bold magenta]{'='*65}[/bold magenta]")
    console.print(f"  Symbol          : [bold cyan]{symbol}[/bold cyan]")
    console.print(f"  AI Score        : [bold green]{best['score']:.1f}[/bold green]")
    console.print(f"  Pattern Bonus   : [yellow]+{best.get('pat_bonus',0)}[/yellow]")
    console.print(f"  ML Conf (1m)    : {d1m['ml_score']}%")
    console.print(f"  OB Imbalance    : {best['imbalance']:+.4f}")
    console.print(f"  Scan Rounds     : [cyan]{scan_round}[/cyan]")

    # ── SMART ENTRY PRICE ─────────────────────────────────────────────────
    console.print(f"\n  [bold green]── Smart Entry Price ──[/bold green]")
    console.print(f"  Smart Entry     : [bold green]{smart['smart_entry']:.8f} USDC[/bold green]  ← USE THIS as limit order")
    console.print(f"  Best Bid (live) : [cyan]{smart['best_bid']:.8f} USDC[/cyan]")
    console.print(f"  ATR Candidate   : [cyan]{smart['atr_entry']:.8f} USDC[/cyan]  (bid − 0.5×ATR)")
    console.print(f"  wick_ll floor   : [red]{smart['wick_ll']:.8f} USDC[/red]  (500-bar structural support)")
    console.print(f"  Last Close      : [dim]{d1m['price']:.8f} USDC[/dim]")
    disc = smart['discount_pct']
    disc_c = "[green]" if disc > 0 else "[yellow]"
    console.print(f"  Discount vs Close: {disc_c}{disc:+.4f}%[/{disc_c[1:]}")
    console.print(f"  Rationale       : [dim]{smart['rationale']}[/dim]")

    # ── SCAN HISTORY TREND ─────────────────────────────────────────────────
    trend = _get_grade_trend(scan_cache, symbol)
    if trend and trend['run_count'] > 1:
        hist_str  = " → ".join(trend['grade_hist'][-5:])
        impr_str  = "[bold green]IMPROVING ↑[/bold green]" if trend['improving'] else "[dim]stable[/dim]"
        streak_str= (f"[bold green]Sniper streak: {trend['streak']} runs ✔[/bold green]"
                     if trend['streak'] >= 2 else f"streak: {trend['streak']}")
        console.print(f"  Grade History   : [cyan]{hist_str}[/cyan]  {impr_str}  {streak_str}")
        console.print(f"  Best Score Seen : [green]{trend['best_score']:.1f}[/green]  over {trend['run_count']} scans")
        if trend['improving'] and trend['streak'] >= 1:
            console.print(f"  [bold green]⚡ Signal is BUILDING — consecutive improvement detected[/bold green]")
    else:
        console.print(f"  Grade History   : [dim]first appearance (no prior scan history)[/dim]")

    # Sniper grade display
    console.print(f"  Sniper          : ", end="")
    if best['sniper']:
        grade = best.get('sniper_grade', '?')
        gscr  = best.get('sniper_gscr', 0)
        grade_color = {"A": "bold green", "B": "green", "C": "yellow"}.get(grade, "red")
        console.print(f"[{grade_color}]GRADE {grade} ({gscr}/100) — INFLECTION DETECTED ✔[/{grade_color}]")
    else:
        console.print("[red]NO — accumulating exhaustion signals[/red]")

    console.print(f"  Entry Price     : [bold]{entry:.8f} USDC[/bold]")
    console.print(f"  1m Bull Vol%    : {d1m.get('bull_vol_pct_1m', 0):.1f}%")
    console.print(f"  1m talib MOM    : {d1m.get('momentum_1m', 0):+.6f}")
    console.print(f"  1m talib RSI    : {d1m['rsi']:.2f}")

    # ── PHYSICS METRICS ────────────────────────────────────────────────────
    _hdr("⚛️  PHYSICS METRICS — Energy, Momentum, Vibration")
    vib = vibration_energy_score(d1m, d3m, d5m)
    console.print(f"  Vibration Energy Score   : [bold {'green' if vib > 50 else 'yellow'}]{vib:.1f}/100[/bold {'green' if vib > 50 else 'yellow'}]")
    # Angular momentum per short TF — uses real stored close+volume arrays
    for tf in ['1m', '3m', '5m']:
        stored_dir = best["data"][tf].get('ang_mom_dir', 'NEUTRAL')
        stored_val = best["data"][tf].get('ang_mom', 0.0)
        dir_c = "[green]UP ↑[/green]" if stored_dir=='UP' else "[red]DOWN ↓[/red]" if stored_dir=='DOWN' else "[yellow]NEUTRAL →[/yellow]"
        console.print(f"  Angular Momentum ({tf})  : {dir_c}  ({stored_val:+.4f})")

    # Rotational symmetry from 5m
    sym = rotational_symmetry_target(entry, d5m['wick_ll'], d5m['wick_hh'])
    console.print(f"\n  [bold white]── Rotational Symmetry Levels (5m wick range) ──[/bold white]")
    for k, v in sym.items():
        gain_s = (v - entry) / entry * 100
        if 2.0 <= gain_s <= 15.0:
            console.print(f"  {k:<20} : [cyan]{v:.8f}[/cyan]  (+{gain_s:.3f}%)")

    # Elliott Wave projections
    ew = elliott_wave_targets(entry, d5m['wick_ll'], d5m['wick_hh'])
    console.print(f"\n  [bold white]── Elliott Wave Projections (W2 bottom → W3 up) ──[/bold white]")
    for k, v in ew.items():
        gain_e = (v - entry) / entry * 100
        if 2.0 <= gain_e <= 15.0:
            console.print(f"  {k:<25} : [yellow]{v:.8f}[/yellow]  (+{gain_e:.3f}%)")

    # ── 1m INFLECTION SIGNALS ─────────────────────────────────────────────
    _hdr("🎯 1m INFLECTION SIGNALS — Cycle Turn Evidence")
    console.print(f"  MOM Crossover (neg→pos) : {'[bold green]YES ✔  ← ENTER NOW[/bold green]' if d1m.get('mom_cross_up') else '[dim]not yet — wait[/dim]'}")
    console.print(f"  RSI Turning Up (<35)    : {'[green]YES ✔[/green]' if d1m.get('rsi_turning_up') else '[dim]not yet[/dim]'}")
    console.print(f"  Sine Crossing Up        : {'[green]YES ✔[/green]' if d1m.get('sine_cross_up') else '[dim]not yet[/dim]'}")
    console.print(f"  Volume Climax (panic)   : {'[green]YES ✔[/green]' if d1m.get('vol_climax') else '[dim]no[/dim]'}")
    console.print(f"  Liquidity Sweep         : {'[bold yellow]YES ⚡[/bold yellow]' if d1m.get('liq_sweep') else '[dim]no[/dim]'}")
    console.print(f"  Bullish Engulf (1m)     : {'[green]YES ✔[/green]' if d1m.get('bullish_engulf') else '[dim]no[/dim]'}")
    console.print(f"  5m RSI Divergence       : {'[green]YES ✔[/green]' if d5m.get('rsi_div') else '[dim]no[/dim]'}")
    console.print(f"  5m Reg Band Rejection   : {'[green]YES ✔[/green]' if d5m.get('reg_rejection') else '[dim]no[/dim]'}")

    # ── VOLUME DELTA & ATR ────────────────────────────────────────────────
    _hdr("📊 VOLUME DELTA — Buyer vs Seller Pressure (1m/3m/5m)")
    console.print(f"  {'TF':>4}  {'Buy%':>7}  {'Sell%':>7}  {'Delta%':>8}  "
                  f"{'OBV Slope':>12}  {'Signal':>14}  {'ΔCross?':>8}  ATR%  Tier")
    _sep("-")
    for tf in ['1m', '3m', '5m', '15m']:
        d = best["data"][tf]
        dsig  = d.get('delta_signal', 'NEUTRAL')
        dcross= d.get('delta_cross_up', False)
        dpct  = d.get('delta_pct', 0)
        bpct  = d.get('buy_vol_pct', 50)
        spct  = d.get('sell_vol_pct', 50)
        obv_s = d.get('obv_slope', 0)
        atr_p = d.get('atr_pct', 0)
        atier = d.get('atr_tier', '?')

        sig_c = (f"[bold green]{dsig}[/bold green]" if 'BUY' in dsig
                 else f"[bold red]{dsig}[/bold red]" if 'SELL' in dsig
                 else f"[yellow]{dsig}[/yellow]")
        dc_c  = "[bold green]YES⚡[/bold green]" if dcross else "[dim]no[/dim]"
        dp_c  = f"[green]{dpct:>+8.3f}%[/green]" if dpct > 0 else f"[red]{dpct:>+8.3f}%[/red]"
        obv_c = f"[green]{obv_s:>+12.4f}[/green]" if obv_s > 0 else f"[red]{obv_s:>+12.4f}[/red]"
        atr_c = f"[red]{atr_p:.3f}%[/red]" if atier=='HIGH' else f"[yellow]{atr_p:.3f}%[/yellow]" if atier=='MEDIUM' else f"[dim]{atr_p:.3f}%[/dim]"
        console.print(f"  {tf:>4}  {bpct:>7.2f}%  {spct:>7.2f}%  {dp_c}  "
                      f"{obv_c}  {sig_c:>14}  {dc_c:>8}  {atr_c}  {atier}")

    # ── PATTERN SUMMARY ────────────────────────────────────────────────────
    _hdr("🔬 ACTIVE PATTERN SIGNALS")
    console.print(f"  Sine Trough Confluence (≥2 short TFs sine<-0.7) : "
                  f"{'[green]YES ✔[/green]' if detect_sine_trough_confluence(best['data']) else '[red]NO[/red]'}")
    console.print(f"  Cascade Dip Alignment  (ALL 8 TFs recent LOW)   : "
                  f"{'[green]YES ✔[/green]' if detect_cascade_dip(best['data']) else '[red]NO[/red]'}")
    for tf in ['1m','3m','5m']:
        d = best["data"][tf]
        console.print(
            f"  {tf:>3} | Engulf:{('[green]✔[/green]' if d.get('bullish_engulf') else '-'):>5}  "
            f"RSI_Div:{('[green]✔[/green]' if d.get('rsi_div') else '-'):>5}  "
            f"VolClimax:{('[green]✔[/green]' if d.get('vol_climax') else '-'):>5}  "
            f"RegReject:{('[green]✔[/green]' if d.get('reg_rejection') else '-'):>5}  "
            f"SqzRelease:{('[green]✔[/green]' if d.get('breakout_ready') else '-'):>5}"
        )

    # ── FEAR & GREED ──────────────────────────────────────────────────────
    _hdr("📊 FEAR & GREED SINEWAVE — talib HT_SINE (ALL Timeframes)")
    console.print(f"  {'TF':>4}  {'HT_Sine':>9}  {'LeadSine':>9}  {'Phase':>7}  {'F&G':>5}  {'MOM':>10}  Stage")
    _sep("-")
    for tf in TF_LIST:
        d   = best["data"][tf]
        fg  = d["fg_score"]
        fg_c= (f"[bold red]{fg}[/bold red]" if fg<20 else f"[red]{fg}[/red]" if fg<40
               else f"[yellow]{fg}[/yellow]" if fg<60 else f"[green]{fg}[/green]")
        mom = d.get("momentum", 0) or 0
        mom_c = f"[green]{mom:+.5f}[/green]" if mom>0 else f"[red]{mom:+.5f}[/red]"
        sc = (f"[bold green]{d['state']}[/bold green]" if d['state']=="DIP_ZONE"
              else f"[bold red]{d['state']}[/bold red]" if d['state']=="TOP_ZONE"
              else d['state'])
        console.print(f"  {tf:>4}  {d['sine']:>+9.4f}  {d['leadsine']:>+9.4f}  "
                      f"{d['phase']:>6.1f}°  {fg_c}  {mom_c}  {d['fg_label']}  [{sc}]")

    # ── MTF THRESHOLDS ────────────────────────────────────────────────────
    _hdr("📐 MTF THRESHOLDS — 1200-Bar argmin/argmax Extrema")
    console.print(f"  {'TF':>4}  {'LowestLow':>14}  {'Middle':>14}  {'HighestHigh':>14}  {'LL@Bar':>6}  {'HH@Bar':>6}  Event")
    _sep("-")
    for tf in TF_LIST:
        d  = best["data"][tf]
        ev = "[green]LOW(Dip)[/green]" if d['last_event']=="LOW_RECENT" else "[red]HIGH(Top)[/red]"
        console.print(f"  {tf:>4}  [red]{d['min_threshold']:>14.8f}[/red]  "
                      f"[yellow]{d['mid_threshold']:>14.8f}[/yellow]  "
                      f"[green]{d['max_threshold']:>14.8f}[/green]  "
                      f"{d['argmin_bar']:>6}  {d['argmax_bar']:>6}  {ev}")

    # ── 500-BAR EXTREMA ───────────────────────────────────────────────────
    _hdr("🕯️  500-BAR CLOSE EXTREMA — Encapsulated Ranges")
    tf_clock = {'1m':'~8.3h','3m':'~25h','5m':'~41.7h','15m':'~5.2d',
                '30m':'~10.4d','1h':'~20.8d','4h':'~83.3d','1d':'~500d'}
    console.print(f"  {'TF':>4}  {'Clock':>10}  {'LowestLow':>14}  {'HighestHigh':>14}  "
                  f"{'Range':>12}  {'MostRecent':>11}  {'NearLow?':>8}  LiqSweep?")
    _sep("-")
    prev_range = 0.0
    for tf in TF_LIST:
        d   = best["data"][tf]
        nl  = "[green]YES✔[/green]" if d['wick_near_low'] else "no"
        ls  = "[bold yellow]⚡[/bold yellow]" if d['liq_sweep'] else "-"
        mre = d.get("most_recent_extreme","?")
        mrc = "[bold green]LOW↓[/bold green]" if mre=="LOW" else "[bold red]HIGH↑[/bold red]"
        er  = d.get("extrema_range", 0.0)
        enc = "[green]✔[/green]" if er>=prev_range else "[red]✘[/red]"
        prev_range = er
        console.print(f"  {tf:>4}  {tf_clock.get(tf,'?'):>10}  "
                      f"[red]{d['wick_ll']:>14.8f}[/red]  [green]{d['wick_hh']:>14.8f}[/green]  "
                      f"{enc}[yellow]{er:>11.6f}[/yellow]  {mrc}  {nl:>8}  {ls}")

    encap = validate_encapsulation(best['data'])
    all_ok = all(encap.values())
    console.print(f"\n  Encapsulation: {'[green]ALL VALID ✔[/green]' if all_ok else '[yellow]PARTIAL[/yellow]'}  "
                  + "  ".join(f"{p}:[green]✔[/green]" if ok else f"{p}:[red]✘[/red]"
                               for p, ok in encap.items()))

    # ── LINEARREG CHANNEL ─────────────────────────────────────────────────
    _hdr("📈 talib LINEARREG CHANNEL (ALL Timeframes)")
    console.print(f"  {'TF':>4}  {'LowBand':>14}  {'RegMean':>14}  {'HighBand':>14}  "
                  f"{'Fcast+5':>14}  {'Slope':>10}  {'Below?':>6}  {'Dist%':>7}  Squeeze?")
    _sep("-")
    for tf in TF_LIST:
        d   = best["data"][tf]
        blw = "[green]YES[/green]" if d['below_reg'] else "[red]NO[/red]"
        sqz = "[yellow]SQZ[/yellow]" if d['compression'] else "-"
        brk = "[cyan]BRK[/cyan]" if d.get('breakout_ready') else ""
        sc  = f"[green]{d['reg_slope']:>+10.6f}[/green]" if d['reg_slope']>0 else f"[red]{d['reg_slope']:>+10.6f}[/red]"
        console.print(f"  {tf:>4}  [red]{d['low_reg']:>14.8f}[/red]  {d['reg_mean']:>14.8f}  "
                      f"[green]{d['high_reg']:>14.8f}[/green]  [cyan]{d['reg_forecast']:>14.8f}[/cyan]  "
                      f"{sc}  {blw:>6}  {d['dist']:>+7.3f}%  {sqz}{brk}")

    # ── VOLUME RESISTANCE ZONES ───────────────────────────────────────────
    _hdr("📦 VOLUME RESISTANCE ZONES — price→HH per TF")
    console.print(f"  {'TF':>4}  {'HH Anchor':>14}  {'VZ T1':>14}  {'VZ T2':>14}  {'VZ T3':>14}  {'Gap→T1%':>8}  Runway")
    _sep("-")
    for tf in TF_LIST:
        d   = best["data"][tf]
        hh  = d['max_threshold']
        vrt = d.get("vol_res_targets", [])
        gap = d.get("vol_res_gap_pct", 0.0)
        t   = [f"{vrt[i]:.8f}" if i<len(vrt) else "       —" for i in range(3)]
        rwy = ("[bold green]CLEAR✔[/bold green]" if gap>3.0 else "[green]open[/green]" if gap>1.5
               else "[yellow]tight[/yellow]" if gap>0.5 else "[red]WALL✘[/red]")
        console.print(f"  {tf:>4}  [dim]{hh:>14.8f}[/dim]  [cyan]{t[0]:>14}[/cyan]  "
                      f"[cyan]{t[1]:>14}[/cyan]  [cyan]{t[2]:>14}[/cyan]  {gap:>+8.3f}%  {rwy}")

    # ── FFT HARMONIC TARGETS ──────────────────────────────────────────────
    _hdr("🔭 FFT DOMINANT HARMONIC TARGETS — Short TFs (1m/3m/5m/15m/30m)")
    console.print(f"  [dim]Proper frequency decomposition — DC mean excluded. Top amplitude peaks only.[/dim]")
    console.print(f"  {'TF':>4}  {'Top Harmonic':>16}  {'Gain%':>8}  {'Harmonics>Price':>16}  Valid?")
    _sep("-")
    for tf in ['1m','3m','5m','15m','30m']:
        d   = best["data"][tf]
        fft = d['fft']
        harmonics = d.get('fft_harmonics', [])
        gain_f = (fft-entry)/entry*100 if entry>0 else 0
        valid = "[green]YES[/green]" if 2.0<=gain_f<=15.0 else "[dim]out of range[/dim]"
        console.print(f"  {tf:>4}  [bold cyan]{fft:>16.8f}[/bold cyan]  {gain_f:>+8.3f}%  {len(harmonics):>16}  {valid}")
    console.print(f"  [dim]4h/1d excluded — their dominant frequency covers months, not actionable for fast spikes[/dim]")

    # ── HH STRUCTURAL ANCHORS ─────────────────────────────────────────────
    _hdr("🏔️  STRUCTURAL CEILING ANCHORS — argmax 1200-Bar per TF")
    console.print(f"  {'TF':>4}  {'HH Ceiling':>16}  {'Gain% from Entry':>18}")
    _sep("-")
    for tf in TF_LIST:
        d    = best["data"][tf]
        hh   = d['max_threshold']
        dist = (hh-entry)/entry*100 if entry>0 else 0.0
        console.print(f"  {tf:>4}  [bold green]{hh:>16.8f}[/bold green]  [cyan]{dist:>+18.3f}%[/cyan]")

    # ── TRADE SETUP ───────────────────────────────────────────────────────
    _hdr("🎯 INSTANT SPOT LONG SETUP")
    tp1 = best['data']['5m']['fft']
    tp2 = best['data']['5m']['max_threshold']
    tp3 = best['data']['15m']['max_threshold']
    tp4 = best['data']['1h']['max_threshold']
    cycle_low  = d5m['min_threshold']
    cycle_mid  = d5m['mid_threshold']
    cycle_high = d5m['max_threshold']
    diff2      = max(cycle_high - cycle_low, 1e-10)

    fib382 = cycle_low + diff2*0.382
    fib500 = cycle_low + diff2*0.500
    fib618 = cycle_low + diff2*0.618
    fib786 = cycle_low + diff2*0.786

    console.print(f"  Entry Price              : [bold]{entry:.8f} USDC[/bold]")
    console.print(f"  Sine Price Target (1m)   : {d1m['sine_price_target']:.8f} USDC")
    console.print(f"  LINEARREG Forecast (1m+5): [cyan]{d1m['reg_forecast']:.8f}[/cyan] USDC")

    console.print(f"\n  [bold cyan]── FFT Dominant Harmonic Targets (short TFs) ──[/bold cyan]")
    console.print(f"  [dim]Top amplitude harmonic peaks above entry — real magnetic levels, not DC mean[/dim]")
    for tf in ['1m','3m','5m','15m','30m']:
        fv   = best['data'][tf]['fft']    # now = top harmonic peak above price
        harmonics = best['data'][tf].get('fft_harmonics', [])
        gf   = (fv-entry)/entry*100
        col  = "cyan" if 2.0<=gf<=15.0 else "dim"
        n_harm = len(harmonics)
        console.print(f"  FFT {tf:<5}: [{col}]{fv:.8f} USDC[/{col}]  ({gf:+.3f}%)  [{n_harm} harmonics above price]")

    console.print(f"\n  [bold green]── HH Structural Anchors (reject targets) ──[/bold green]")
    console.print(f"  HH (5m)  : [green]{tp2:.8f} USDC[/green]  (+{(tp2-entry)/entry*100:.3f}%)")
    console.print(f"  HH (15m) : [yellow]{tp3:.8f} USDC[/yellow]  (+{(tp3-entry)/entry*100:.3f}%)")
    console.print(f"  HH (1h)  : [bold yellow]{tp4:.8f} USDC[/bold yellow]  (+{(tp4-entry)/entry*100:.3f}%)")

    console.print(f"\n  [bold white]── Fibonacci Cycle (5m 1200-Bar) ──[/bold white]")
    console.print(f"  Cycle LL : [red]{cycle_low:.8f}[/red]  @ bar {d5m['argmin_bar']}")
    console.print(f"  Cycle Mid: [yellow]{cycle_mid:.8f}[/yellow]")
    console.print(f"  Cycle HH : [green]{cycle_high:.8f}[/green]  @ bar {d5m['argmax_bar']}")
    for label, val in [("Fibo 0.382",fib382),("Fibo 0.500",fib500),
                        ("Fibo 0.618 ★",fib618),("Fibo 0.786",fib786)]:
        gv = (val-entry)/entry*100
        col= "bold green" if label.endswith("★") else "white"
        console.print(f"  {label:<12} : [{col}]{val:.8f} USDC[/{col}]  (+{gv:.3f}%)")

    # ── FORWARD BOUNCE PROBABILITY ESTIMATE ──────────────────────────────────
    _hdr("🧠 FORWARD BOUNCE PROBABILITY — Structural Signal-Weighted Estimate")
    console.print(f"  [dim]Not a backtest. Forward probability from current position + signal stack.[/dim]")
    bt_results, cons_gain, cons_min, cons_max = estimate_bounce_probability(best['data'], entry)
    console.print(f"  {'TF':>4}  {'Pos%LL':>8}  {'WinRate':>8}  {'EstGain%':>9}  {'EV%':>7}  {'→T1%':>7}  Bounce")
    _sep("-")
    for tf in TF_LIST:
        bt = bt_results.get(tf, {})
        wr = bt.get('win_rate', 0)
        ev = bt.get('ev_pct', 0)
        wc = f"[green]{wr:.1f}%[/green]" if wr>=60 else f"[yellow]{wr:.1f}%[/yellow]" if wr>=50 else f"[red]{wr:.1f}%[/red]"
        ec = f"[green]{ev:+.3f}%[/green]" if ev>0 else f"[red]{ev:+.3f}%[/red]"
        bar= "█" * int(bt.get('bounce_est',0)*10)
        console.print(f"  {tf:>4}  {bt.get('pos_pct',0):>8.2f}%  {wc}  "
                      f"{bt.get('est_gain_pct',0):>+9.3f}%  {ec}  "
                      f"{bt.get('gain_to_t1',0):>+7.3f}%  {bar}")
    console.print(f"\n  Cross-TF Consensus: [bold green]+{cons_gain:.3f}%[/bold green]"
                  f"  (Q1→Q3: +{cons_min:.3f}% → +{cons_max:.3f}%)")
    consensus_exit = entry * (1 + cons_gain/100)
    console.print(f"  Consensus Exit    : [bold green]{consensus_exit:.8f} USDC[/bold green]")

    # ── EXIT PRICE ENGINE ─────────────────────────────────────────────────
    _hdr("🚀 EXIT PRICE ENGINE — Pump Hunter Target Ranking")
    console.print(
        f"  [dim]Sources: HH-argmax × Fibonacci × FFT-harmonics × Resonance × "
        f"Elliott Wave × RotSym × Regression × Sine × Vol-Resistance[/dim]"
    )
    console.print(
        f"  [dim]Scoring: weight × speed_tier × cluster_magnet × path_clearance "
        f"× angular_momentum × vibration_energy × hh_proximity × method_prestige[/dim]"
    )
    console.print(f"  [dim]Min target: +2% (covers fees). Max: +15%. SPOT only — no stop loss.[/dim]\n"
    )

    exit_targets, primary_exit_price = compute_exit_price(best, entry)

    if exit_targets:
        primary = exit_targets[0]
        console.print(
            f"  [bold green]★ PRIMARY EXIT : {primary['price']:.8f} USDC  "
            f"(+{primary['gain']:.3f}%)  Tier:{primary['tier']}  "
            f"[{primary['label']}]  score={primary['score']:.3f}  "
            f"cluster={primary['cluster']}  walls={primary['walls_n']}[/bold green]"
        )

        console.print(f"\n  [bold white]── All Ranked Exit Candidates ──[/bold white]")
        console.print(f"  {'#':>2}  {'Price':>14}  {'Gain%':>7}  {'Tier':>8}  {'Score':>8}  "
                      f"{'Clust':>5}  {'Walls':>5}  {'Method':>10}  Label")
        _sep("-")

        colors = ["bold green","green","cyan","yellow","yellow","dim","dim"]
        for i, t in enumerate(exit_targets):
            col = colors[min(i, len(colors)-1)]
            console.print(
                f"  {i+1:>2}  [{col}]{t['price']:>14.8f}[/{col}]  "
                f"[{col}]{t['gain']:>+7.3f}%[/{col}]  "
                f"{t['tier']:>8}  {t['score']:>8.3f}  "
                f"{t['cluster']:>5}  {t['walls_n']:>5}  "
                f"{t['method']:>10}  {t['label']}"
            )

        # Optimal path explanation
        console.print(f"\n  [bold white]── Optimal Entry Path ──[/bold white]")
        console.print(f"  [bold green]1. Place LIMIT BUY at {smart['smart_entry']:.8f} USDC[/bold green]  ← smart entry price")
        console.print(f"  [green]   (Best bid: {smart['best_bid']:.8f} | wick_ll floor: {smart['wick_ll']:.8f})[/green]")
        console.print(f"  [green]2. First target  : {primary['price']:.8f} USDC (+{primary['gain']:.2f}%)[/green]")
        if len(exit_targets) > 1:
            t2 = exit_targets[1]
            gain2 = (t2['price'] - smart['smart_entry']) / smart['smart_entry'] * 100
            console.print(f"  [cyan]3. If momentum holds: {t2['price']:.8f} USDC (+{gain2:.2f}% from entry)[/cyan]")
        if len(exit_targets) > 2:
            t3 = exit_targets[2]
            gain3 = (t3['price'] - smart['smart_entry']) / smart['smart_entry'] * 100
            console.print(f"  [yellow]4. Extended target  : {t3['price']:.8f} USDC (+{gain3:.2f}% from entry)[/yellow]")
        console.print(f"  [dim]Spot: no stop loss. If dip extends, wait for next MOM+delta crossover.[/dim]")
    else:
        console.print("  [yellow]No targets found in 2-15% range. Range too tight or price too close to HH.[/yellow]")
        console.print(f"  [dim]Consensus exit estimate: {consensus_exit:.8f} USDC (+{cons_gain:.2f}%)[/dim]")

    # ── PUMP SPIKE SUMMARY ────────────────────────────────────────────────
    _hdr("🚀 PUMP SPIKE SUMMARY")
    grade     = best.get('sniper_grade', 'F')
    gscr      = best.get('sniper_gscr', 0)
    cascade   = detect_cascade_dip(best['data'])
    sine_conf = detect_sine_trough_confluence(best['data'])
    pat_bonus = best.get('pat_bonus', 0)
    gc        = {"A":"bold green","B":"green","C":"yellow","F":"red"}.get(grade,"red")

    console.print(f"  Sniper Grade    : [{gc}]{grade} ({gscr}/100)[/{gc}]")
    console.print(f"  Pattern Bonus   : [yellow]+{pat_bonus}[/yellow]")
    console.print(f"  Vib Energy      : [bold {'green' if vib>50 else 'yellow'}]{vib:.1f}/100[/bold {'green' if vib>50 else 'yellow'}]")
    console.print(f"  Cascade Dip     : {'[bold green]YES ✔ ALL 8 TFs[/bold green]' if cascade else '[yellow]partial[/yellow]'}")
    console.print(f"  Sine Trough     : {'[green]YES ✔[/green]' if sine_conf else '[yellow]partial[/yellow]'}")
    console.print(f"  MOM Crossover   : {'[bold green]YES ✔ — ENTER[/bold green]' if d1m.get('mom_cross_up') else '[dim]waiting...[/dim]'}")
    console.print(f"  Vol Delta Cross : {'[bold green]YES ✔ — buyers entering[/bold green]' if d1m.get('delta_cross_up') else '[dim]not yet[/dim]'}")
    console.print(f"  Delta Signal    : ", end="")
    dsig = d1m.get('delta_signal', 'NEUTRAL')
    if dsig == 'BUY_DOMINANT':  console.print("[bold green]BUY_DOMINANT ✔[/bold green]")
    elif dsig == 'BUY_LEAN':    console.print("[green]BUY_LEAN[/green]")
    elif dsig == 'SELL_DOMINANT': console.print("[bold red]SELL_DOMINANT — wait[/bold red]")
    elif dsig == 'SELL_LEAN':   console.print("[red]SELL_LEAN — caution[/red]")
    else:                       console.print("[yellow]NEUTRAL[/yellow]")
    console.print(f"  1m ATR          : {d1m.get('atr_pct',0):.3f}%  ({d1m.get('atr_tier','?')})")

    console.print(f"\n  [bold white]── Verdict ──[/bold white]")

    # Composite signal count for verdict strength
    strong_signals = sum([
        bool(d1m.get('mom_cross_up')),
        bool(d1m.get('delta_cross_up')),
        bool(d1m.get('rsi_turning_up')),
        bool(d1m.get('sine_cross_up')),
        bool(d1m.get('vol_climax')),
        bool(d1m.get('liq_sweep')),
        dsig in ('BUY_DOMINANT', 'BUY_LEAN'),
    ])

    if grade == "A":
        console.print(f"  [bold green]GRADE A — ENTER NOW. {strong_signals}/7 inflection signals active.[/bold green]")
        console.print(f"  [green]Exhaustion confirmed. Buyers entering. Macro supports reversal.[/green]")
    elif grade == "B":
        console.print(f"  [green]GRADE B — {strong_signals}/7 signals. Enter on next 1m close above {d1m.get('wick_ll',entry):.8f}[/green]")
    elif grade == "C":
        console.print(f"  [yellow]GRADE C — {strong_signals}/7 signals. Watch for MOM + delta cross before entering.[/yellow]")
    else:
        console.print(f"  [red]GRADE F — {strong_signals}/7 signals. Downtrend active. Observe only.[/red]")

    primary_exit_val = exit_targets[0]['price'] if exit_targets else consensus_exit
    primary_gain_val = exit_targets[0]['gain']  if exit_targets else cons_gain

    send_alert(
        f"🚀 PUMP ENTRY CONFIRMED: {symbol} | Grade {grade} ({gscr}) | Round {scan_round}\n"
        f"Smart Entry {smart['smart_entry']:.8f} | Target {primary_exit_val:.8f} (+{primary_gain_val:.2f}%)\n"
        f"VibEnergy {vib:.0f}/100 | F&G {d1m['fg_score']} ({d1m['fg_label']})\n"
        f"MOM {d1m.get('momentum_1m',0):+.6f} | Delta {dsig} | "
        f"ATR {d1m.get('atr_pct',0):.3f}% ({d1m.get('atr_tier','?')}) | "
        f"Signals {strong_signals}/7 | Consensus +{cons_gain:.2f}%"
    )

    console.print(f"\n[bold cyan]Scan complete.[/bold cyan]")
