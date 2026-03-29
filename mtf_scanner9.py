from binance.client import Client
import numpy as np
import talib as ta
import sys, gc, time, threading, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    from scipy.signal import argrelextrema as _argrelextrema
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ── sklearn ML imports ────────────────────────────────────
try:
    from sklearn.linear_model    import Ridge, Lasso, BayesianRidge, LinearRegression
    from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm             import SVR
    from sklearn.preprocessing   import StandardScaler, PolynomialFeatures
    from sklearn.pipeline        import Pipeline
    from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print('  [WARN] scikit-learn not found — ML module disabled. pip install scikit-learn')

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CREDENTIALS_FILE = 'credentials.txt'
MAX_WORKERS      = 12
LOOP_SLEEP       = 5
FFT_CANDLES      = 600    # min 3× max expected period; max detectable = 300 bars

# ─────────────────────────────────────────────
#  TIME GEOMETRY CONSTANTS  (φ · e · π)
#
#  Each constant governs a different symmetry domain:
#
#  φ  (golden ratio ≈ 1.618)
#     Domain : STRUCTURE — how price organizes itself
#     Symmetry: self-similar / fractal
#     x → x + 1/x  (scale-invariant proportion)
#     Discrete: Fibonacci steps;  Continuous: logarithmic spiral
#     ↳ Price structure, Fibonacci levels, branching
#
#  e  (Euler ≈ 2.718)
#     Domain : EVOLUTION — how momentum decays / grows over time
#     Symmetry: dynamic / flow  (d/dt eˣ = eˣ — only self-invariant fn)
#     Only function unchanged by differentiation → perfect model of
#     continuous compounding and exponential amplitude envelopes
#     ↳ Oscillation envelope, momentum decay rate, compounding
#
#  π  (pi ≈ 3.14159)
#     Domain : CYCLES — how price repeats and rotates
#     Symmetry: rotational  (e^(iθ) → rotation)
#     Every closed oscillation, wave, and market cycle
#     ↳ Dominant period, phase angle, cycle timing
#
#  BRIDGE 1 — e ↔ π  (e^(iπ) + 1 = 0):
#     Exponential growth becomes rotation in the complex plane.
#     Growth → circle.  Leads to: complex exponential model
#     y(t) = A · e^(α + iω)t  where α=growth, ω=frequency=2π/T
#
#  BRIDGE 2 — φ ↔ e  (φⁿ ≈ e^(n·ln φ)):
#     Fibonacci recursion becomes smooth exponential.
#     Discrete Fibonacci → continuous golden exponential.
#
#  BRIDGE 3 — φ ↔ π  (phyllotaxis, 137.5° golden angle):
#     Structure meets rotation → logarithmic spirals.
#
#  ALL THREE TOGETHER → GOLDEN SPIRAL:
#     r(θ) = A · e^(b·θ)   where  b = ln(φ) / (π/2)
#     After each  π/2 turn: radius ×φ
#     After each  π   turn: radius ×φ²  ≈ ×2.618
#     After each  2π  turn: radius ×φ⁴  ≈ ×6.854
#     This spiral is self-similar, grows by e, closes by π.
#
#  TRADING MEANING:
#     φ = WHERE price will bounce to (structure / Fibonacci targets)
#     e = HOW FAST the bounce envelope grows (momentum decay rate)
#     π = WHEN the bounce happens (cycle phase / timing)
# ─────────────────────────────────────────────
PHI       = (1.0 + np.sqrt(5.0)) / 2.0     # ≈ 1.6180339887
PHI_INV   = 1.0 / PHI                       # ≈ 0.6180339887  (= φ − 1)
PHI2      = PHI ** 2                        # ≈ 2.6180339887  (= φ + 1)
PHI_SQRT  = np.sqrt(PHI)                    # ≈ 1.2720196495
GOLDEN_B  = np.log(PHI) / (np.pi / 2.0)    # golden spiral rate ≈ 0.30635
PHI_ANGLE = 360.0 / PHI2                    # golden angle ≈ 137.5077°
E         = np.e                            # ≈ 2.7182818285
TAU       = 2.0 * np.pi                     # full circle ≈ 6.2831853

# ─────────────────────────────────────────────
#  SINUSOIDAL CIRCUIT CONSTANTS
#  360° full-cycle mapping between extremas:
#    0°   = argmin  (trough / SUPPORT DIP)
#    90°  = Q1→Q2 crossing  (midline, rising)
#    180° = argmax  (peak   / RESISTANCE TOP)
#    270° = Q3→Q4 crossing  (midline, falling)
#    360° = next argmin  (cycle repeats)
#
#  Quadrant labels:
#    Q1  0°– 90°  Emergence from trough  / Capitulation near trough
#    Q2  90°–180° Expansion toward peak  / Accumulation near support
#    Q3  180°–270° Distribution past peak / Decline from peak
#    Q4  270°–360° Exhaustion near next   / Collapse near next trough
#
#  UP   cycle (argmin most recent):  phase travels 0°→180°
#  DOWN cycle (argmax most recent):  phase travels 180°→360°
#
#  Quadrature point = φ away from each extremum bar (90° in harmonic sense)
#  Phase lead rule:  V(t) ≈ P(t + Δφ)   — volume leads price
# ─────────────────────────────────────────────
CIRCUIT_QUADS = {
    'Q1': (  0.0,  90.0),
    'Q2': ( 90.0, 180.0),
    'Q3': (180.0, 270.0),
    'Q4': (270.0, 360.0),
}
# Descriptors for UP and DOWN cycles per quadrant
_CQ_LABEL_UP = {
    'Q1': 'Emergence  — leaving trough, early rise',
    'Q2': 'Expansion  — strong rally toward peak',
    'Q3': 'Distribution — approaching resistance top',
    'Q4': 'Exhaustion — near peak, REVERSAL imminent',
}
_CQ_LABEL_DN = {
    'Q1': 'Capitulation — near trough, REVERSAL imminent',
    'Q2': 'Accumulation — approaching support dip',
    'Q3': 'Decline      — strong sell-off from peak',
    'Q4': 'Collapse     — leaving peak, early drop',
}
# Absorption / exhaustion thresholds (from order-flow logic)
_ABS_LOG_THRESH  = 5.0    # log1p(vol/range) above which absorption is confirmed
_EXHS_LOG_THRESH = 1.5    # log1p(late_vol/late_move) ratio for exhaustion flag

# Fibonacci retracement / extension ratios derived from φ
FIB_RATIOS = [
    (0.236, '23.6% retrace'),
    (0.382, '38.2% retrace (φ⁻²)'),
    (0.500, '50.0% retrace'),
    (0.618, '61.8% retrace (φ⁻¹)'),
    (0.786, '78.6% retrace (√φ⁻¹)'),
    (1.000, '100% retrace'),
    (1.272, '127.2% ext  (√φ)'),
    (1.618, '161.8% ext  (φ)'),
    (2.000, '200% ext'),
    (2.618, '261.8% ext  (φ²)'),
]

# ─────────────────────────────────────────────
#  CLIENT - ONLY USDC SPOT TRADING PAIRS (FIXED)
# ─────────────────────────────────────────────
class Trader:
    def __init__(self, file):
        lines = [l.rstrip('\n') for l in open(file)]
        self.client = Client(lines[0], lines[1])

    # ── Binance product/yield base assets that are NOT real spot coins ──────────
    # These tokens appear as TRADING on exchange info but are Binance-internal
    # yield, collateral, or pegged products with no conventional spot klines.
    _EXCLUDED_BASES = {
        'BFUSD', 'FDUSD', 'TUSD', 'USDP', 'USDS', 'DAI', 'FRAX',
        'LUSD', 'USTC', 'EURS', 'EURT', 'AEUR',
        'BBTC', 'BETH', 'BBNB', 'LDBNB', 'WBETH',
        'LDETH', 'LDBTC', 'LDUSDT', 'LDBUSD',
    }

    def get_usdc_pairs(self):
        """
        Returns only genuine USDC spot trading pairs.
        A symbol must pass ALL gates to be included:

          Gate 1  quoteAsset == USDC
          Gate 2  status == TRADING
          Gate 3  isSpotTradingAllowed explicitly True (default False)
          Gate 4  SPOT in permissions / permissionSets
          Gate 5  baseAsset NOT in _EXCLUDED_BASES (blocks BFUSD, BBTC, BETH…)
          Gate 6  live ticker price NOT within 0.5% of $1.00
                  (catches any stablecoin-vs-USDC pair not in the static list)
        """
        info  = self.client.get_exchange_info()
        raw   = []

        for s in info['symbols']:
            if s['quoteAsset'] != 'USDC':                      continue
            if s['status']     != 'TRADING':                   continue
            if s['symbol'].endswith('USD'):                    continue
            if not s.get('isSpotTradingAllowed', False):       continue

            perms     = s.get('permissions', [])
            perm_sets = s.get('permissionSets', [])
            flat_sets = [p for sub in perm_sets for p in sub]
            if 'SPOT' not in perms and 'SPOT' not in flat_sets:
                continue

            base = s['baseAsset']
            if base in self._EXCLUDED_BASES:                   continue

            raw.append((s['symbol'], base))

        if not raw:
            return [], {}

        # Gate 6: live price sanity — skip anything pegged near $1.00
        try:
            tickers   = self.client.get_all_tickers()
            price_map = {t['symbol']: float(t['price']) for t in tickers}
        except Exception:
            price_map = {}

        pairs     = []
        label_map = {}
        for sym, base in raw:
            price = price_map.get(sym)
            if price is not None and 0.995 <= price <= 1.005:
                continue   # stablecoin-vs-USDC, skip
            pairs.append(sym)
            label_map[sym] = base

        return pairs, label_map


trader = Trader(CREDENTIALS_FILE)

# ─────────────────────────────────────────────
#  FILTER FUNCTIONS — 500-PERIOD REGRESSION CHANNEL
#  Midline = linear regression of last 500 closes
#  Upper   = midline + 1 std-dev of residuals
#  Lower   = midline − 1 std-dev of residuals
#  Pass    = last close BELOW the lower band → confirmed dip
# ─────────────────────────────────────────────

def _channel_pass(klines):
    """
    500-period regression channel filter using TA-Lib LINEARREG.
      midline = ta.LINEARREG(close, timeperiod=500)
      upper   = midline + 1 std-dev of residuals
      lower   = midline − 1 std-dev of residuals
    Pass = current close is BELOW the lower band → confirmed dip.
    Returns (passed, close_list).
    """
    close = [float(e[4]) for e in klines]
    if not close:
        return False, []
    x       = np.array(close, dtype=np.float64)
    period  = min(500, len(x))
    midline = ta.LINEARREG(x, timeperiod=period)
    # use only the valid (non-NaN) tail
    valid   = ~np.isnan(midline)
    if not np.any(valid):
        return False, close
    x_v     = x[valid]
    m_v     = midline[valid]
    std     = np.std(x_v - m_v)
    lower   = m_v - std
    return float(x_v[-1]) < float(lower[-1]), close

def filter1(pair, out, lock):
    klines = trader.client.get_klines(symbol=pair, interval='2h')
    passed, _ = _channel_pass(klines)
    if passed:
        with lock: out.append(pair)

def filter1b(pair, out, lock):
    klines = trader.client.get_klines(symbol=pair, interval='30m')
    passed, _ = _channel_pass(klines)
    if passed:
        with lock: out.append(pair)

def filter2(pair, out, lock):
    klines = trader.client.get_klines(symbol=pair, interval='15m')
    passed, _ = _channel_pass(klines)
    if passed:
        with lock: out.append(pair)

def filter3(pair, out, lock):
    klines = trader.client.get_klines(symbol=pair, interval='5m')
    passed, _ = _channel_pass(klines)
    if passed:
        with lock: out.append(pair)


# ─────────────────────────────────────────────
#  REAL ORDER-FLOW + ABSORPTION + EXHAUSTION
#  (NEW — added exactly as requested)
# ─────────────────────────────────────────────

def get_real_volume_flow(trader, pair, limit=1000):
    """
    Returns:
        buy_vol, sell_vol,
        delta, delta_ratio,
        absorption_score,
        exhaustion_score
    """
    try:
        trades = trader.client.get_aggregate_trades(symbol=pair, limit=limit)
    except Exception:
        return None

    if not trades:
        return None

    buy_vol = 0.0
    sell_vol = 0.0

    prices = []
    qtys   = []

    for t in trades:
        qty = float(t['q'])
        price = float(t['p'])

        prices.append(price)
        qtys.append(qty)

        if t['m']:   # seller aggressor
            sell_vol += qty
        else:        # buyer aggressor
            buy_vol += qty

    total = buy_vol + sell_vol
    if total == 0:
        return None

    delta = buy_vol - sell_vol
    delta_ratio = delta / total

    # ─────────────────────────────────────────────
    # 🟡 ABSORPTION DETECTION
    # high volume, low price movement
    # ─────────────────────────────────────────────
    price_range = max(prices) - min(prices) + 1e-12
    total_volume = sum(qtys)

    absorption_score = total_volume / price_range
    # normalize (log scale safer)
    absorption_score = np.log1p(absorption_score)

    # ─────────────────────────────────────────────
    # 🔴 EXHAUSTION DETECTION
    # volume spike + weak continuation
    # ─────────────────────────────────────────────
    prices_arr = np.array(prices)
    qtys_arr   = np.array(qtys)

    # split into early vs late trades
    mid = len(prices_arr) // 2

    early_move = abs(prices_arr[mid] - prices_arr[0]) + 1e-12
    late_move  = abs(prices_arr[-1] - prices_arr[mid]) + 1e-12

    early_vol = np.sum(qtys_arr[:mid]) + 1e-12
    late_vol  = np.sum(qtys_arr[mid:]) + 1e-12

    # exhaustion: volume increases but move decreases
    vol_ratio  = late_vol / early_vol
    move_ratio = late_move / early_move

    exhaustion_score = vol_ratio / (move_ratio + 1e-12)
    exhaustion_score = np.log1p(exhaustion_score)

    return {
        'buy_vol': buy_vol,
        'sell_vol': sell_vol,
        'delta': delta,
        'delta_ratio': delta_ratio,
        'absorption': absorption_score,
        'exhaustion': exhaustion_score
    }


# ─────────────────────────────────────────────
#  1m DIP CONFIRMATION
#  Two conditions must BOTH be true:
#
#  1. VOLUME: bull_vol% > bear_vol% from total 1m volume
#     (last 500 candles)
#  2. EXTREMA: most recent extreme in last 500 1m closes
#     is the MINIMA, not the maxima.
#     np.argmin index > np.argmax index
#
#  CMO(14) is computed here ONLY for ranking purposes.
#
#  NEW — φ·e·π DIP GEOMETRY SCORE added here:
#     Scores the structural quality of the dip using all three
#     constants (φ-deviations, e-decay R², π-phase proximity).
# ─────────────────────────────────────────────

EXTREMA_LOOKBACK  = 1000   # 1m candles for extrema + volume check

def check_dip_conditions(pair):
    """
    Returns (passed, detail_dict).

    detail_dict contains:
      bull_pct, bear_pct, argmin_idx, argmax_idx,
      raw_cmo, price, cond_vol, cond_ext,
      geometry_score   — φ·e·π dip quality (0-100)
      geometry_detail  — breakdown dict
      close_arr        — numpy array of closes (for downstream geo analysis)
      low_arr          — numpy array of lows
      high_arr         — numpy array of highs
      swing_low        — min of low_arr
      swing_high       — max of high_arr
      NEW: delta_ratio, absorption_score, exhaustion_score
    """
    try:
        klines = trader.client.get_klines(
            symbol=pair, interval='1m', limit=EXTREMA_LOOKBACK
        )
    except Exception:
        return False, {}
    if not klines:
        return False, {}

    close = np.array([float(k[4]) for k in klines], dtype=np.float64)
    open_ = np.array([float(k[1]) for k in klines], dtype=np.float64)
    low_  = np.array([float(k[3]) for k in klines], dtype=np.float64)
    high_ = np.array([float(k[2]) for k in klines], dtype=np.float64)
    vol   = np.array([float(k[5]) for k in klines], dtype=np.float64)

    # ─────────────────────────────────────────────
    # ✅ REAL ORDER FLOW + ABSORPTION + EXHAUSTION
    # (Binance aggTrades → true buyer/seller aggressor volume)
    # ─────────────────────────────────────────────
    flow = get_real_volume_flow(trader, pair)

    if flow:
        buy_vol  = flow['buy_vol']
        sell_vol = flow['sell_vol']
        total_vol_real = buy_vol + sell_vol

        bull_ratio = buy_vol / total_vol_real
        bear_ratio = sell_vol / total_vol_real

        # 🔥 enhanced condition (real buyer pressure OR absorption OR strong delta)
        cond_vol = (
            bull_ratio > 0.5
            or flow['absorption'] > 5.0   # strong absorption
            or flow['delta_ratio'] > 0.1  # real buyer pressure
        )

        absorption_score = flow['absorption']
        exhaustion_score = flow['exhaustion']
        delta_ratio      = flow['delta_ratio']

    else:
        # fallback to old method if API fails
        bull_mask  = close >= open_
        bull_vol   = float(vol[bull_mask].sum())
        total_vol  = float(vol.sum())
        if total_vol == 0:
            return False, {}
        bull_ratio = bull_vol / total_vol
        bear_ratio = 1.0 - bull_ratio
        cond_vol   = bull_ratio > 0.5

        absorption_score = 0.0
        exhaustion_score = 0.0
        delta_ratio      = 0.0

    # ── condition 2: most recent extreme is the minima ──────
    # Use actual wick arrays across ALL 500 bars:
    #   argmin → low_ (true lowest wick, not lowest close)
    #   argmax → high_ (true highest wick, not highest close)
    # cond_ext passes when the absolute deepest low is MORE RECENT
    # than the absolute highest high → price sitting at the
    # floor of the 500-bar window, not just the lowest close.
    argmin_idx = int(np.argmin(low_))    # bar with true lowest wick  ← extrema of all 500
    argmax_idx = int(np.argmax(high_))   # bar with true highest wick ← extrema of all 500
    cond_ext   = argmin_idx > argmax_idx

    # ── raw CMO — ranking only, not a gate ──────────────────
    cmo_arr = ta.CMO(close, timeperiod=14)
    raw_cmo = float(cmo_arr[-1]) if not np.isnan(cmo_arr[-1]) else None

    # ── φ·e·π dip geometry score (NEW) ──────────────────────
    geo_score, geo_detail = _phi_e_pi_dip_score(close, float(close[-1]))

    detail = {
        'bull_pct':        round(bull_ratio * 100.0, 1),
        'bear_pct':        round(bear_ratio * 100.0, 1),
        'argmin_idx':      argmin_idx,
        'argmax_idx':      argmax_idx,
        'raw_cmo':         round(raw_cmo, 2) if raw_cmo is not None else None,
        'price':           round(float(close[-1]), 8),
        'cond_vol':        cond_vol,
        'cond_ext':        cond_ext,
        'geometry_score':  geo_score,
        'geometry_detail': geo_detail,
        # arrays for downstream time geometry analysis
        'close_arr':       close,
        'low_arr':         low_,
        'high_arr':        high_,
        'swing_low':       float(np.min(low_)),
        'swing_high':      float(np.max(high_)),
        # NEW real-order-flow fields
        'delta_ratio':      round(delta_ratio, 4),
        'absorption_score': round(absorption_score, 4),
        'exhaustion_score': round(exhaustion_score, 4),
    }
    passed = cond_vol and cond_ext
    return passed, detail


def momentum(pair, sel_pairs, sel_cmo, sel_detail, lock):
    passed, detail = check_dip_conditions(pair)
    if not detail:
        return
    with lock:
        sel_detail[pair] = detail
        if passed:
            sel_pairs.append(pair)
            sel_cmo.append(detail['raw_cmo'] if detail['raw_cmo'] is not None else 0.0)


# ─────────────────────────────────────────────
#  TIME GEOMETRY FUNCTIONS
#  φ · e · π  — three-constant framework
# ─────────────────────────────────────────────

def _detrend(arr):
    """Remove linear trend from array. Returns (detrended, trend_poly_coeffs)."""
    t = np.arange(len(arr))
    p = np.polyfit(t, arr, 1)
    return arr - np.poly1d(p)(t), p


# ── φ component: extension levels ─────────────────────────

def phi_extension_levels(swing_low, swing_high, current_price):
    """
    Compute φ-based Fibonacci bounce targets above current_price.

    After a dip:
      range = swing_high - swing_low
      bounce target at ratio r = swing_low + r × range

    The golden key ratios (0.618, 1.618, 2.618) are direct powers
    of φ, giving self-similar structure to bounce magnitudes.

    Returns list of (ratio, label, level) above current_price.
    """
    rng = swing_high - swing_low
    if rng <= 0 or swing_low <= 0:
        return []
    levels = []
    for ratio, label in FIB_RATIOS:
        level = swing_low + ratio * rng
        if level > current_price * 1.001:
            levels.append((ratio, label, round(float(level), 8)))
    return sorted(levels, key=lambda x: x[2])


# ── π component: Hilbert Transform cycle analysis ─────────

def hilbert_cycle_analysis(close_arr):
    """
    Uses talib's Hilbert Transform to extract the π (cycle) component.

    HT_DCPERIOD  → dominant cycle period  (natural oscillation length)
    HT_DCPHASE   → current cycle phase 0-360° (where we are in the cycle)
    HT_SINE      → sine of cycle phase (near -1 = cycle trough = BUY)
    HT_PHASOR    → in-phase + quadrature → instantaneous amplitude
    HT_TRENDMODE → 0=cycling, 1=trending

    Phase convention (talib):
      Phase = 0°   → start of cycle
      Phase ≈ 90°  → cycle peak
      Phase ≈ 270° → cycle trough  ← BUY ZONE
      Sine near -1 + LeadSine > Sine → approaching bottom = best entry

    Returns dict or None on failure.
    """
    try:
        arr = np.asarray(close_arr, dtype=np.float64)
        if len(arr) < 32:
            return None

        ht_period     = ta.HT_DCPERIOD(arr)
        ht_phase      = ta.HT_DCPHASE(arr)
        sine, lead    = ta.HT_SINE(arr)
        inph, quad    = ta.HT_PHASOR(arr)
        trend_mode    = ta.HT_TRENDMODE(arr)

        def last(x):
            v = x[~np.isnan(x)]
            return float(v[-1]) if len(v) > 0 else None

        lp    = last(ht_period)
        lph   = last(ht_phase)
        ls    = last(sine)
        ll    = last(lead)
        li    = last(inph)
        lq    = last(quad)
        lt    = last(trend_mode)

        # instantaneous amplitude = |analytic signal|
        amplitude = float(np.sqrt(li**2 + lq**2)) if (li and lq) else None

        # bars to next trough (phase 270° → trough)
        bars_to_trough = None
        if lph is not None and lp is not None:
            trough_phase   = 270.0
            phase_to_go    = (trough_phase - lph) % 360.0
            bars_to_trough = round(phase_to_go / 360.0 * lp, 1)

        # BUY ZONE: sine < 0 and sine > lead_sine
        # (cycle rising from its own trough — the π confirmation)
        in_buy_zone = (ls is not None and ll is not None
                       and ls < 0 and ls > ll)

        # phase label
        if lph is not None:
            if   lph < 45 or lph >= 315: phase_label = 'start/trough'
            elif lph < 135:              phase_label = 'rising'
            elif lph < 225:              phase_label = 'peak'
            else:                        phase_label = 'falling→trough'
        else:
            phase_label = 'unknown'

        return {
            'ht_period':      round(lp,  1) if lp  else None,
            'ht_phase_deg':   round(lph, 1) if lph else None,
            'ht_sine':        round(ls,  4) if ls is not None else None,
            'ht_lead_sine':   round(ll,  4) if ll is not None else None,
            'ht_amplitude':   round(amplitude, 10) if amplitude else None,
            'ht_trend_mode':  int(lt) if lt is not None else None,
            'bars_to_trough': bars_to_trough,
            'in_buy_zone':    in_buy_zone,
            'phase_label':    phase_label,
        }
    except Exception:
        return None


# ── e component: complex exponential fit ──────────────────

def complex_exp_forecast(close_arr, current_price, bars_forward=None):
    """
    Fits the complex exponential model: y(t) = A · e^(α·t) · cos(ω·t + φ₀)

    This is the BRIDGE 1 (e ↔ π) model: exponential amplitude envelope
    combined with rotational oscillation, exactly as described by:
      e^((α + iω)t)   (real=growth/decay, imaginary=oscillation)

    Method:
      1. Detrend price series (remove linear drift)
      2. Use HT_PHASOR to get in-phase (I) and quadrature (Q) components
         → analytic signal = I + jQ
         → envelope A(t) = sqrt(I² + Q²)   [the 'e' component]
         → instantaneous phase φ(t) = atan2(Q, I)  [the 'π' component]
      3. Fit log(A) vs t linearly → slope = α (exp growth/decay rate)
      4. Unwrap phase, compute ω = mean(dφ/dt)
      5. Project: A_fwd = A₀ · e^(α·t_fwd) · cos(φ_now + ω·Δt)

    Parameters:
      bars_forward : projection horizon in bars
                     default = dominant HT period (one full cycle ahead)

    Returns dict or None on failure.
    """
    try:
        arr = np.asarray(close_arr, dtype=np.float64)
        n   = len(arr)
        if n < 32:
            return None

        detrended, trend_coeffs = _detrend(arr)

        # ── analytic signal via HT_PHASOR ──────────────────────
        inph, quad = ta.HT_PHASOR(arr)
        valid      = ~(np.isnan(inph) | np.isnan(quad))
        if np.sum(valid) < 20:
            return None

        envelope  = np.sqrt(inph[valid]**2 + quad[valid]**2)
        t_valid   = np.arange(n)[valid].astype(np.float64)
        log_env   = np.log(np.maximum(envelope, 1e-20))

        # ── fit log(A) = log(A₀) + α·t  [e component] ─────────
        coeffs  = np.polyfit(t_valid, log_env, 1)
        alpha   = float(coeffs[0])
        A0      = float(np.exp(coeffs[1]))

        # R² of the exponential amplitude fit
        pred    = np.poly1d(coeffs)(t_valid)
        ss_res  = float(np.sum((log_env - pred)**2))
        ss_tot  = float(np.sum((log_env - log_env.mean())**2))
        r2      = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        # ── instantaneous phase, angular frequency  [π component] ──
        ht_phase_arr = ta.HT_DCPHASE(arr)
        ht_period_arr= ta.HT_DCPERIOD(arr)
        valid_ph     = ~np.isnan(ht_phase_arr)
        if np.sum(valid_ph) < 10:
            return None

        # get dominant period for default bars_forward
        valid_pe = ~np.isnan(ht_period_arr)
        ht_period_now = float(ht_period_arr[valid_pe][-1]) if np.sum(valid_pe) > 0 else 20.0
        if bars_forward is None:
            bars_forward = max(5, int(round(ht_period_now / 2)))

        phase_arr = np.deg2rad(ht_phase_arr[valid_ph])
        phase_uw  = np.unwrap(phase_arr)
        omega     = float(np.mean(np.diff(phase_uw))) if len(phase_uw) > 1 else 0.0

        period_est = abs(TAU / omega) if abs(omega) > 1e-10 else None

        # current amplitude and phase
        t_now     = float(n - 1)
        A_now     = A0 * np.exp(alpha * t_now)
        phase_now = float(phase_uw[-1]) if len(phase_uw) > 0 else 0.0

        # project forward
        t_fwd      = t_now + bars_forward
        A_fwd      = A0 * np.exp(alpha * t_fwd)
        phase_fwd  = phase_now + omega * bars_forward
        trend_fwd  = float(np.poly1d(trend_coeffs)(t_fwd))
        forecast   = trend_fwd + A_fwd * np.cos(phase_fwd)

        # exponential momentum label
        if   alpha >  1e-6: e_label = 'growing oscillation (e^+ α) — expansion'
        elif alpha < -1e-6: e_label = 'decaying oscillation (e^- α) — compression'
        else:               e_label = 'neutral amplitude'

        return {
            'alpha':         round(alpha,    6),
            'omega_rad':     round(omega,    6),
            'period_est':    round(period_est, 1) if period_est else None,
            'A0':            round(A0,        8),
            'A_now':         round(A_now,     8),
            'bars_forward':  bars_forward,
            'forecast':      round(float(forecast), 8),
            'fit_r2':        round(max(0.0, r2), 4),
            'e_label':       e_label,
        }
    except Exception:
        return None


# ── φ+π component: golden spiral targets ─────────────────

def golden_spiral_targets(current_price, ht_phase_deg, ht_amplitude,
                          swing_low=None, swing_high=None):
    """
    Projects price along the GOLDEN SPIRAL: r(θ) = A · e^(b·θ)
    where  b = ln(φ) / (π/2) = GOLDEN_B

    Amplitude scaling (critical fix):
      HT phasor amplitude is a micro-signal (often 0.02–0.3% of price) —
      too small to produce meaningful price targets.
      Instead, use the swing range as the amplitude base:
        A_base = (swing_high - swing_low) / 2   ← half the 1000-bar range
      If swing not available, fall back to ht_amplitude but scaled up
      to at least 0.5% of current price.

    price_target(r) = current_price + (r - r0) × scale_factor
    where scale_factor = A_base / max(r0, 1e-20)
    This ensures quarter-turn = meaningful % move (not micro pips).
    """
    if ht_phase_deg is None:
        return None

    theta0 = np.deg2rad(ht_phase_deg)

    # ── amplitude base: swing range preferred over HT phasor ─────────
    if swing_low is not None and swing_high is not None and swing_high > swing_low:
        A_base = (swing_high - swing_low) / 2.0
    elif ht_amplitude and ht_amplitude > 0:
        # scale HT amplitude up to at least 0.5% of price
        A_base = max(float(ht_amplitude), current_price * 0.005)
    else:
        A_base = current_price * 0.01   # 1% default

    # normalize: C so that r(theta0) = A_base
    C = A_base / (np.exp(GOLDEN_B * theta0) + 1e-20)

    def spiral_r(delta_theta):
        theta = theta0 + delta_theta
        return float(C * np.exp(GOLDEN_B * theta))

    r0     = spiral_r(0)
    r_q1   = spiral_r(np.pi / 2)
    r_half = spiral_r(np.pi)
    r_3q   = spiral_r(3 * np.pi / 2)
    r_full = spiral_r(TAU)

    # price move = amplitude change × direction
    def price_target(r):
        return round(current_price + (r - r0), 8)

    return {
        'current_angle':  round(np.rad2deg(theta0) % 360.0, 1),
        'A_base':         round(A_base, 8),
        'gs_q1_turn':     price_target(r_q1),
        'gs_half_turn':   price_target(r_half),
        'gs_3q_turn':     price_target(r_3q),
        'gs_full_turn':   price_target(r_full),
        'phi_mult_q1':    round(r_q1  / r0, 4) if r0 > 1e-20 else None,
        'phi_mult_half':  round(r_half / r0, 4) if r0 > 1e-20 else None,
    }


# ── unified φ·e·π dip quality score ──────────────────────

def _phi_e_pi_dip_score(close_arr, current_price):
    """
    Score the dip quality using all three constants (0–100 total).

    φ SCORE (0–40): true Z-score with probabilistic depth
      Z = (lower_band − current_price) / σ
        lower_band = LINEARREG midline − σ(residuals)
      p_value  = norm.cdf(Z) — probability price is this far below channel
      score    = min(40, Z × 10 × φ)
      p_value gives quantitative rarity of the dip (e.g. 0.95 = top-5% dip).

    FAKE DIP GUARD — curvature check (reduces score up to −20):
      Real bottoms show concave-up price structure (2nd derivative > 0).
      Computed via ta.LINEARREG_SLOPE applied twice:
        slope1  = LINEARREG_SLOPE(close,  period=10)   ← 1st derivative
        slope2  = LINEARREG_SLOPE(slope1, period=10)   ← 2nd derivative (curvature)
      If curvature < 0 (still accelerating down) → structural fake → penalty.
      curvature_ok = slope2[-1] > 0
      penalty  = 0 if curvature_ok else min(20, abs(slope2[-1] / σ) × 10)

    e SCORE (0–30): exponential momentum character
      Does recent price decay follow an exponential shape?
      Fit last N bars to log-linear model (y = A·e^(α·t))
      Higher R² of exponential fit AND negative slope (decay)
      → clean momentum → better reversal setup.

    π SCORE (0–30): cycle phase timing
      Is price near the cycle trough (π/2 to 3π/2 phase range)?
      Uses Hilbert phase: trough ≈ 270°.
      Closeness to trough angle → higher score.

    Returns (total_score, detail_dict).
    """
    try:
        from scipy.stats import norm as _norm
    except ImportError:
        _norm = None

    try:
        arr = np.asarray(close_arr, dtype=np.float64)
        n   = len(arr)
        if n < 20:
            return 0.0, {}

        # ── φ score: TRUE Z-SCORE below regression channel lower band ──
        # Use TA-Lib LINEARREG for the midline (consistent with _channel_pass).
        # σ = std of residuals (channel width).
        # Z = (lower_band − price) / σ  → how many σ's below the floor.
        # p_value = norm.cdf(z) where z > 0 means left tail
        period   = min(500, n)
        midline  = ta.LINEARREG(arr, timeperiod=period)
        valid_ml = ~np.isnan(midline)
        if not np.any(valid_ml):
            return 0.0, {}
        x_v       = arr[valid_ml]
        m_v       = midline[valid_ml]
        sigma     = float(np.std(x_v - m_v)) + 1e-20
        trend_now = float(m_v[-1])
        lower_band = trend_now - sigma
        z_score    = (lower_band - current_price) / sigma   # signed; >0 = below band
        phi_devs   = max(0.0, z_score)
        phi_score  = min(40.0, phi_devs * 10.0 * PHI)
        # probabilistic rarity: norm.cdf(z) where z > 0 means left tail
        p_value = float(_norm.cdf(z_score)) if _norm is not None else None

        # ── FAKE DIP GUARD: curvature (2nd derivative via LINEARREG_SLOPE) ──
        # slope1 = 1st derivative of price (momentum direction)
        # slope2 = slope of slope1 = 2nd derivative (curvature / acceleration)
        # Genuine dip bottom: curvature turns positive (deceleration of selling).
        # Fake dip:           curvature still negative (price still accelerating down).
        curv_period   = min(10, n // 2)
        slope1        = ta.LINEARREG_SLOPE(arr,    timeperiod=curv_period)
        slope2        = ta.LINEARREG_SLOPE(slope1, timeperiod=curv_period)
        valid_s2      = ~np.isnan(slope2)
        curvature_now = float(slope2[valid_s2][-1]) if np.any(valid_s2) else 0.0
        curvature_ok  = curvature_now > 0.0
        # penalty scales with how negative curvature is, normalised by σ
        curv_penalty  = 0.0 if curvature_ok else min(20.0, abs(curvature_now / sigma) * 10.0)
        is_fake_dip   = (not curvature_ok) and (curv_penalty >= 10.0)

        # ── e score: exponential amplitude fit ─────────────────
        # use last 30 bars for recent momentum character
        look = min(30, n)
        last_n   = arr[-look:]
        t_n      = np.arange(look, dtype=np.float64)
        log_last = np.log(np.maximum(last_n, 1e-20))
        coeffs_e = np.polyfit(t_n, log_last, 1)
        pred_e   = np.poly1d(coeffs_e)(t_n)
        ss_res   = float(np.sum((log_last - pred_e)**2))
        ss_tot   = float(np.sum((log_last - log_last.mean())**2))
        r2_exp   = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        decay_r  = float(coeffs_e[0])
        # score: full 30 pts only if R² > 0.8 AND slope is negative (decay)
        e_score  = r2_exp * (30.0 if decay_r < 0 else 12.0)

        # ── π score: Hilbert phase proximity to trough ─────────
        ht_ph_arr    = ta.HT_DCPHASE(arr)
        valid_ph     = ~np.isnan(ht_ph_arr)
        pi_score     = 0.0
        ht_phase_now = None
        if np.sum(valid_ph) > 0:
            ht_phase_now = float(ht_ph_arr[valid_ph][-1])
            # trough at 270°; compute angular distance
            dist = abs(ht_phase_now - 270.0)
            dist = min(dist, 360.0 - dist)   # take shorter arc
            pi_score = max(0.0, 30.0 * (1.0 - dist / 180.0))

        raw_total = phi_score + e_score + pi_score
        total     = round(max(0.0, raw_total - curv_penalty), 1)
        detail = {
            # φ — Z-score depth
            'phi_devs':       round(phi_devs,       3),
            'z_score':        round(z_score,         3),
            'p_value':        round(p_value,         4) if p_value is not None else None,
            'phi_score':      round(phi_score,       1),
            # curvature / fake-dip
            'curvature':      round(curvature_now,   6),
            'curvature_ok':   curvature_ok,
            'curv_penalty':   round(curv_penalty,    1),
            'is_fake_dip':    is_fake_dip,
            # e
            'e_r2':           round(r2_exp,          3),
            'e_decay_rate':   round(decay_r,         6),
            'e_score':        round(e_score,         1),
            # π
            'ht_phase_now':   round(ht_phase_now,    1) if ht_phase_now is not None else None,
            'pi_score':       round(pi_score,        1),
            'total':          total,
        }
        return total, detail

    except Exception:
        return 0.0, {}


# ── MTF harmony score ──────────────────────────────────────

def mtf_harmony_score(stf_results, htf_results):
    """
    Measures harmonic alignment of cycles across all timeframes.

    Three independent checks, each scoring one constant's contribution:

    φ HARMONY (0–30): period φ-ratios
      Are consecutive TF dominant periods in φ-ratio to each other?
      e.g. 1m period × φ ≈ 3m period → self-similar cycle nesting.
      Check ratios: φ, φ², 2, 3, φ/2  (±20% tolerance).
      Score = (φ-resonant pairs / total pairs) × 30

    e HARMONY (0–30): forecast convergence
      Do all TF forecasts agree on a similar price target?
      Coefficient of variation (CV = σ/μ) of all forecasts.
      Low CV → tight consensus → e-envelope aligns across TFs.
      Score = (1 − min(1, CV × 20)) × 30

    π HARMONY (0–20): directional sync
      Are all TFs pointing upside (positive upside_pct)?
      All TFs in sync → rotational alignment across scales.
      Score = (n_up / n_total) × 20

    Bonus (0–20): STF ↔ HTF phase alignment
      Is the short-term cycle phase aligned with the HTF cycle?
      (both near trough simultaneously → strongest signal)
      Currently derived from cascade_stop absence and direction.

    Total raw / 100 → 0-100 score.
    Returns (score_0_100, detail_dict).
    """
    all_r = (stf_results or []) + (htf_results or [])
    if len(all_r) < 2:
        return 0.0, {}

    periods   = [r['dominant_period'] for r in all_r]
    forecasts = [r['forecast']        for r in all_r]
    upsides   = [r['upside_pct']      for r in all_r]

    # ── φ harmony: period ratio check ──────────────────────────
    phi_pairs   = 0
    total_pairs = 0
    phi_ratio_targets = [PHI, PHI2, 2.0, 3.0, PHI * 0.5, PHI_INV]
    for i in range(len(periods) - 1):
        p1, p2 = periods[i], periods[i + 1]
        if p1 > 0 and p2 > 0:
            ratio = max(p1, p2) / min(p1, p2)
            for target in phi_ratio_targets:
                if abs(ratio - target) / target < 0.20:
                    phi_pairs += 1
                    break
            total_pairs += 1
    phi_h = (phi_pairs / total_pairs * 30.0) if total_pairs > 0 else 0.0

    # ── e harmony: forecast convergence ────────────────────────
    fc_arr  = np.array(forecasts, dtype=np.float64)
    fc_mean = float(np.mean(fc_arr))
    fc_std  = float(np.std(fc_arr))
    fc_cv   = fc_std / fc_mean if fc_mean > 1e-20 else 1.0
    e_h     = max(0.0, 30.0 * (1.0 - min(1.0, fc_cv * 20.0)))

    # ── π harmony: directional sync ────────────────────────────
    n_up  = sum(1 for u in upsides if u > 0)
    pi_h  = (n_up / len(upsides)) * 20.0

    # ── bonus: cascade did NOT stop early (more TFs in agreement) ──
    no_stop_bonus = 0.0
    if htf_results:
        n_stopped = sum(1 for r in htf_results if r.get('cascade_stop'))
        if n_stopped == 0 and len(htf_results) >= 3:
            no_stop_bonus = 20.0   # all HTF TFs aligned without wall
        elif n_stopped == 0:
            no_stop_bonus = 10.0

    raw     = phi_h + e_h + pi_h + no_stop_bonus
    score   = round(min(100.0, raw), 1)

    detail = {
        'phi_pairs':      phi_pairs,
        'total_pairs':    total_pairs,
        'phi_harmony':    round(phi_h,  1),
        'fc_cv_pct':      round(fc_cv * 100.0, 2),
        'e_harmony':      round(e_h,    1),
        'n_up':           n_up,
        'n_total':        len(upsides),
        'pi_harmony':     round(pi_h,   1),
        'no_stop_bonus':  round(no_stop_bonus, 1),
        'harmony':        score,
    }
    return score, detail


# ─────────────────────────────────────────────
#  CONCURRENT STAGE RUNNER
# ─────────────────────────────────────────────

def run_stage(fn, symbols, label):
    out   = []
    lock  = threading.Lock()
    total = len(symbols)
    done  = [0]

    def worker(sym):
        fn(sym, out, lock)
        with lock:
            done[0] += 1
            pct = int(done[0] / total * 100)
            bar = '█' * (pct // 4) + '░' * (25 - pct // 4)
            print(f'\r  {label}  [{bar}] {pct:3d}%  {done[0]}/{total}',
                  end='', flush=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        fs = [pool.submit(worker, s) for s in symbols]
        for f in as_completed(fs): pass
    print()
    return out


# ─────────────────────────────────────────────
#  FFT + VOLUME-RESISTANCE ANALYSIS
#  (original logic preserved; HT data added to result)
# ─────────────────────────────────────────────

def fft_analysis(close_list, volume_list, high_list, current_price, tf_label,
                 sanity_cap_pct=25.0):
    """
    Per-TF FFT analysis with volume-profile resistance.
    NOW ALSO includes:
      - Hilbert cycle analysis (ht_data key in result)
      - Complex exponential fit (cexp key in result)
    These are additive — they do not alter the core FFT/resistance logic.
    """
    n = min(FFT_CANDLES, len(close_list))
    if n < 32:
        return None

    close  = np.array(close_list[-n:], dtype=np.float64)
    volume = np.array(volume_list[-n:], dtype=np.float64)
    high   = np.array(high_list[-n:],  dtype=np.float64)

    tf_price_ref = float(np.median(close))

    # ── 1. detrend ────────────────────────────────────────────
    detrended, trend_coeffs = _detrend(close)

    # ── 2. FFT ───────────────────────────────────────────────
    spectrum = np.fft.rfft(detrended)
    freqs    = np.fft.rfftfreq(n)
    power    = np.abs(spectrum)
    power[0] = 0

    min_period   = 4
    valid_mask   = (freqs > 0) & (freqs <= 1.0 / min_period)
    if not np.any(valid_mask):
        return None
    masked_power              = power.copy()
    masked_power[~valid_mask] = 0

    dom_idx         = int(np.argmax(masked_power))
    dom_freq        = freqs[dom_idx]
    dominant_period = int(round(1.0 / dom_freq)) if dom_freq > 0 else n
    dominant_period = min(dominant_period, n // 2)

    # ── 3. reconstruct: dominant + 3 harmonics ───────────────
    top_indices              = np.argsort(masked_power)[-4:]
    clean_spec               = np.zeros_like(spectrum)
    clean_spec[top_indices]  = spectrum[top_indices]
    reconstructed            = np.fft.irfft(clean_spec, n=n)

    # ── 4. phase-aware projection ─────────────────────────────
    trend_at_end   = float(np.poly1d(trend_coeffs)(n - 1))
    trend_slope    = float(trend_coeffs[0])
    trend_forward  = trend_at_end + trend_slope * dominant_period

    osc_amplitude  = float(np.max(reconstructed) - np.min(reconstructed)) / 2.0
    osc_now        = float(reconstructed[-1])
    osc_mean       = float(np.mean(reconstructed))

    if osc_now < osc_mean:
        osc_contribution = osc_amplitude + abs(osc_now - osc_mean)
    else:
        osc_contribution = osc_amplitude * 0.5

    fft_target = trend_forward + osc_contribution
    fft_target = max(fft_target, current_price * 1.0001)

    # ── 5. volume-profile resistance ─────────────────────────
    BIN_PCT  = 0.003
    bin_size = tf_price_ref * BIN_PCT
    bins     = {}
    for h, v in zip(high, volume):
        if h > current_price * 1.001:
            b        = round(h / bin_size) * bin_size
            bins[b]  = bins.get(b, 0.0) + float(v)

    res_target = None
    res_volume = 0.0
    if bins:
        vol_threshold = float(np.percentile(list(bins.values()), 70))
        candidates    = {
            k: v for k, v in bins.items()
            if v >= vol_threshold and k > current_price
        }
        if candidates:
            res_target = float(min(candidates.keys()))
            res_volume = float(candidates[res_target])

    # ── 6. blend ──────────────────────────────────────────────
    if res_target and res_target > current_price:
        if fft_target > res_target:
            forecast = res_target * 0.65 + fft_target * 0.35
        else:
            forecast = fft_target * 0.60 + res_target * 0.40
    else:
        forecast = fft_target

    forecast   = min(forecast, current_price * (1.0 + sanity_cap_pct / 100.0))
    forecast   = max(forecast, current_price * 1.0001)
    upside_pct = (forecast - current_price) / current_price * 100.0

    # ── NEW: Hilbert cycle data for this TF ───────────────────
    ht_data  = hilbert_cycle_analysis(close)

    # ── NEW: complex exponential fit for this TF ─────────────
    cexp_data = complex_exp_forecast(close, current_price)

    # ── NEW: Absorption / Exhaustion scoring (cause-effect law)
    #  Absorption: high volume, low price progress → smart money absorbing
    #  Exhaustion:  late vol > early vol but late move < early move → fuel gone
    _abs_score  = 0.0
    _exhs_score = 0.0
    _abs_flag   = False
    _exhs_flag  = False
    try:
        if len(volume) >= 20:
            # absorption: use last dominant_period bars
            look_abs = min(dominant_period, len(close) // 2, len(close) - 1)
            look_abs = max(look_abs, 4)
            c_seg    = close[-look_abs:]
            v_seg    = volume[-look_abs:]
            p_range  = max(c_seg) - min(c_seg) + 1e-12
            v_total  = float(v_seg.sum())
            _abs_score  = float(np.log1p(v_total / p_range))
            _abs_flag   = _abs_score > _ABS_LOG_THRESH

            # exhaustion: split the dominant-period window in two halves
            mid       = look_abs // 2
            early_m   = abs(float(c_seg[mid] - c_seg[0])) + 1e-12
            late_m    = abs(float(c_seg[-1] - c_seg[mid])) + 1e-12
            early_v   = float(v_seg[:mid].sum()) + 1e-12
            late_v    = float(v_seg[mid:].sum()) + 1e-12
            exhs_raw  = (late_v / early_v) / (late_m / early_m)
            _exhs_score = float(np.log1p(exhs_raw))
            _exhs_flag  = _exhs_score > _EXHS_LOG_THRESH
    except Exception:
        pass

    # ── NEW: FFT sinusoidal phase angle at dominant frequency ─
    _fft_phase_deg = None
    try:
        dom_coeff      = spectrum[dom_idx]
        _fft_phase_deg = float(np.degrees(np.angle(dom_coeff))) % 360.0
    except Exception:
        pass

    # ── NEW: sinusoid fit quality (R² of dominant reconstruction)
    _sinusoid_r2 = None
    try:
        dom_only          = np.zeros_like(spectrum)
        dom_only[dom_idx] = spectrum[dom_idx]
        dom_rec           = np.fft.irfft(dom_only, n=n)
        ss_res  = float(np.sum((detrended - dom_rec)**2))
        ss_tot  = float(np.sum((detrended - detrended.mean())**2))
        _sinusoid_r2 = round(max(0.0, 1.0 - ss_res / ss_tot), 4) if ss_tot > 1e-12 else 0.0
    except Exception:
        pass

    return {
        'tf':              tf_label,
        'dominant_period': dominant_period,
        'osc_amplitude':   round(osc_amplitude, 8),
        'fft_target':      round(fft_target,    8),
        'res_target':      round(res_target,    8) if res_target else None,
        'res_volume':      round(res_volume,    2),
        'forecast':        round(forecast,      8),
        'upside_pct':      round(upside_pct,    4),
        'cascade_stop':    False,
        'ht_data':         ht_data,
        'cexp_data':       cexp_data,
        # ── cause-effect order-flow signals ──────────────────────
        'absorption_score': round(_abs_score,  4),
        'exhaustion_score': round(_exhs_score, 4),
        'absorption_flag':  _abs_flag,
        'exhaustion_flag':  _exhs_flag,
        # ── sinusoidal circuit data ───────────────────────────────
        'fft_phase_deg':   round(_fft_phase_deg, 2) if _fft_phase_deg is not None else None,
        'sinusoid_r2':     _sinusoid_r2,
        'dom_idx':         int(dom_idx),
        'dom_freq':        float(dom_freq),
    }


def _run_fft_for_tfs(pair, current_price, tf_list, sanity_cap_pct=25.0):
    tf_results = []
    for label, interval in tf_list:
        try:
            klines = trader.client.get_klines(
                symbol=pair, interval=interval, limit=FFT_CANDLES + 20
            )
        except Exception:
            continue
        if len(klines) < 32:
            continue
        close  = [float(k[4]) for k in klines]
        volume = [float(k[5]) for k in klines]
        high   = [float(k[2]) for k in klines]
        result = fft_analysis(close, volume, high, current_price,
                              label, sanity_cap_pct=sanity_cap_pct)
        if result:
            tf_results.append(result)

    if not tf_results:
        return [], None

    forecasts = np.array([r['forecast'] for r in tf_results])
    weights   = np.array([
        r['res_volume'] if r['res_volume'] > 0 else 1.0
        for r in tf_results
    ], dtype=np.float64)

    best_forecast = float(np.average(forecasts, weights=weights))
    best_upside   = (best_forecast - current_price) / current_price * 100.0
    spread        = float(np.std(forecasts) / best_forecast * 100) \
                    if best_forecast > 0 else 0.0
    confidence    = round(max(0.0, min(100.0, 100.0 - spread * 8)), 1)

    best_overall = {
        'current':    current_price,
        'forecast':   round(best_forecast, 8),
        'upside_pct': round(best_upside,   4),
        'confidence': confidence,
        'spread_pct': round(spread,        4),
    }
    return tf_results, best_overall


def full_fft_report(pair, current_price):
    """
    Short-term  : 1m, 3m, 5m         — cap +25% each TF
    Higher-TF   : 15m, 30m, 1h, 2h   — cap +60% each TF, cascade stop logic
    Returns (stf_results, stf_best, htf_results, htf_best).
    """
    stf_tfs = [('1m', '1m'), ('3m', '3m'), ('5m', '5m')]
    htf_tfs = [('15m', '15m'), ('30m', '30m'), ('1h', '1h'), ('2h', '2h')]

    stf_results, stf_best = _run_fft_for_tfs(
        pair, current_price, stf_tfs, sanity_cap_pct=25.0
    )

    htf_results = []
    for label, interval in htf_tfs:
        try:
            klines = trader.client.get_klines(
                symbol=pair, interval=interval, limit=FFT_CANDLES + 20
            )
        except Exception:
            continue
        if len(klines) < 32:
            continue

        close  = [float(k[4]) for k in klines]
        volume = [float(k[5]) for k in klines]
        high   = [float(k[2]) for k in klines]

        result = fft_analysis(close, volume, high, current_price,
                              label, sanity_cap_pct=60.0)
        if not result:
            continue

        result['cascade_stop'] = False
        htf_results.append(result)

        if (result['res_target'] is not None
                and result['res_target'] > current_price * 1.015
                and result['fft_target'] >= result['res_target']):
            result['cascade_stop'] = True
            break

    if htf_results:
        forecasts = np.array([r['forecast'] for r in htf_results])
        weights   = np.array([
            r['res_volume'] if r['res_volume'] > 0 else 1.0
            for r in htf_results
        ], dtype=np.float64)
        htf_best_forecast = float(np.average(forecasts, weights=weights))
        htf_best_upside   = (htf_best_forecast - current_price) / current_price * 100.0
        spread            = float(np.std(forecasts) / htf_best_forecast * 100) \
                            if htf_best_forecast > 0 else 0.0
        confidence        = round(max(0.0, min(100.0, 100.0 - spread * 5)), 1)
        stopped_tf        = next(
            (r['tf'] for r in htf_results if r.get('cascade_stop')), 'none'
        )
        htf_best = {
            'current':    current_price,
            'forecast':   round(htf_best_forecast, 8),
            'upside_pct': round(htf_best_upside,   4),
            'confidence': confidence,
            'spread_pct': round(spread,            4),
            'tfs_used':   len(htf_results),
            'stopped_at': stopped_tf,
        }
    else:
        htf_best = None

    return stf_results, stf_best, htf_results, htf_best


# ─────────────────────────────────────────────
#  SPIKE SCORE  (for stage tables)
# ─────────────────────────────────────────────

def spike_score_and_cmo(pair):
    """
    Score 0-100:
      40 pts — depth below 5m trendline (compressed = coiled)
      40 pts — CMO 1m oversold depth
      20 pts — bullish volume ratio 1m
    Also returns cmo_1m value and current price.
    """
    try:
        k5 = trader.client.get_klines(symbol=pair, interval='5m')
        c5 = [float(e[4]) for e in k5]
        if not c5:
            return 0.0, None, None
        x5    = c5
        y5    = range(len(x5))
        fit5  = np.poly1d(np.polyfit(y5, x5, 1))(y5)
        dev   = max(0.0, (fit5[-1] - x5[-1]) / fit5[-1] * 100.0)
        t_pts = min(40.0, dev * 400.0)

        k1  = trader.client.get_klines(symbol=pair, interval='1m')
        c1  = [float(e[4]) for e in k1]
        if not c1:
            return t_pts, None, None
        price   = c1[-1]
        cmo_arr = ta.CMO(np.asarray(c1, dtype=np.float64), timeperiod=14)
        cmo_val = float(cmo_arr[-1]) if not np.isnan(cmo_arr[-1]) else 0.0
        c_pts   = min(40.0, max(0.0, -cmo_val / 100.0 * 40.0))

        bull  = sum(float(k[5]) for k in k1 if float(k[4]) >= float(k[1]))
        tot   = sum(float(k[5]) for k in k1)
        v_pts = (bull / tot * 20.0) if tot > 0 else 0.0

        score = round(t_pts + c_pts + v_pts, 1)
        return score, round(cmo_val, 2), round(price, 8)
    except Exception:
        return 0.0, None, None


# ─────────────────────────────────────────────
#  STAGE TABLE PRINTER
# ─────────────────────────────────────────────

def print_stage_table(pairs, label_map, stage_label, show_cmo=False):
    if not pairs:
        print(f'  (no pairs passed {stage_label})\n')
        return

    print(f'\n  ┌─ {stage_label} — {len(pairs)} pairs ─────────────────────────────────────────┐')

    data   = {}
    d_lock = threading.Lock()

    def compute(p):
        sc, cmo, pr = spike_score_and_cmo(p)
        with d_lock:
            data[p] = (sc, cmo, pr)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        fs = [pool.submit(compute, p) for p in pairs]
        for f in as_completed(fs): pass

    ranked = sorted(pairs, key=lambda p: data.get(p, (0,))[0], reverse=True)

    hdr = f'  │  {"#":>3}  {"Ticker":<10}  {"Price":>13}  {"Score/100":>9}  {"CMO 1m":>8}  │'
    sep = '  │' + '─' * (len(hdr) - 4) + '│'
    print(sep)
    print(hdr)
    print(sep)

    for i, p in enumerate(ranked, 1):
        sc, cmo, pr = data.get(p, (0.0, None, None))
        lbl   = label_map.get(p, p.replace('USDC', ''))
        pr_s  = (f'{pr:.6f}' if pr and pr < 1 else f'{pr:.4f}') if pr else '—'
        cmo_s = f'{cmo:.2f}' if cmo is not None else '—'
        print(f'  │  {i:>3}  {lbl:<10}  {pr_s:>13}  {sc:>9.1f}  {cmo_s:>8}  │')

    print(sep + '\n')


# ─────────────────────────────────────────────
#  FFT REPORT PRINTER
# ─────────────────────────────────────────────

def _print_tf_block(r):
    """Print one timeframe FFT block."""
    has_res  = r['res_target'] is not None
    stop_tag = '  ◄ CASCADE STOP (resistance reached)' \
               if r.get('cascade_stop') else ''
    # HT quick summary in block
    ht = r.get('ht_data') or {}
    ht_line = ''
    if ht.get('ht_period'):
        bz  = ' ✔ BUY ZONE' if ht.get('in_buy_zone') else ''
        ht_line = (f'  │  HT cycle        : period={ht["ht_period"]}b  '
                   f'phase={ht.get("ht_phase_deg","?")}°  '
                   f'→{ht.get("bars_to_trough","?")}b to trough{bz}')

    # Absorption / exhaustion flags (cause-effect law)
    abs_flag  = r.get('absorption_flag',  False)
    exhs_flag = r.get('exhaustion_flag',  False)
    abs_s     = r.get('absorption_score', 0.0)
    exhs_s    = r.get('exhaustion_score', 0.0)
    fft_ph    = r.get('fft_phase_deg')
    sin_r2    = r.get('sinusoid_r2')

    print(f'  ┌─ [{r["tf"]}] {"─"*52}┐')
    print(f'  │  Dominant cycle  : {r["dominant_period"]} bars')
    print(f'  │  Oscillation amp : {r["osc_amplitude"]}')
    if fft_ph is not None:
        r2_str = f'  sinusoid R²={sin_r2}' if sin_r2 is not None else ''
        print(f'  │  FFT phase angle : {fft_ph}°{r2_str}')
    if ht_line:
        print(ht_line)
    # ── cause-effect signals ──
    abs_tag  = ' 🟡 ABSORPTION ACTIVE'  if abs_flag  else ''
    exhs_tag = ' 🔴 EXHAUSTION ACTIVE'  if exhs_flag else ''
    if abs_flag or exhs_flag:
        print(f'  │  ── Order-flow signals (cause→effect) ───────────────────')
        print(f'  │  Absorption scr  : {abs_s:.4f}{abs_tag}')
        print(f'  │  Exhaustion scr  : {exhs_s:.4f}{exhs_tag}')
        if abs_flag and not exhs_flag:
            print(f'  │  → Effort ≠ result: smart money absorbing, reversal near')
        if exhs_flag:
            print(f'  │  → Vol fuel spent: one side running out, phase-flip risk')
    print(f'  │  FFT projection  : {r["fft_target"]}')
    if has_res:
        print(f'  │  Vol resistance  : {r["res_target"]}  '
              f'(vol weight {r["res_volume"]:.0f}){stop_tag}')
    else:
        print(f'  │  Vol resistance  : none found above entry')
    print(f'  │  ── Forecast ────────────────────────────────────────────')
    print(f'  │  Price target    : {r["forecast"]}')
    print(f'  │  Upside          : +{r["upside_pct"]} %')
    blend = '60% vol-res + 40% FFT' if has_res else '100% FFT (no resistance)'
    print(f'  │  Blend method    : {blend}')
    print(f'  └{"─"*60}┘')
    print()


def print_fft_report(pair, label_map, stf_results, stf_best,
                     htf_results, htf_best):
    lbl = label_map.get(pair, pair.replace('USDC', ''))
    w   = 62

    print(f'\n  {"═"*w}')
    print(f'  ◈  FFT SPIKE FORECAST  ·  {lbl}  ({pair})')
    print(f'  {"═"*w}')
    print(f'  Entry price : {stf_best["current"] if stf_best else "—"}')
    print()

    if stf_results:
        print(f'  ▸ SHORT-TERM TARGETS  (1m · 3m · 5m)')
        print()
        for r in stf_results:
            _print_tf_block(r)

        print(f'  {"═"*w}')
        print(f'  ★  BEST SHORT-TERM FORECAST  (1m/3m/5m consensus)')
        print(f'  {"─"*w}')
        print(f'  Consensus target : {stf_best["forecast"]}')
        print(f'  Upside           : +{stf_best["upside_pct"]} %')
        print(f'  Confidence       : {stf_best["confidence"]} %'
              f'  (TF spread {stf_best["spread_pct"]} %)')
        print(f'  Method           : volume-weighted avg · 1m/3m/5m')
        print(f'  {"═"*w}')
        print()

    if htf_results:
        stopped = htf_results[-1].get('cascade_stop', False)
        stop_tf = htf_results[-1]['tf']
        tfs_run = ' · '.join(r['tf'] for r in htf_results)
        print(f'  ▸ HIGHER-TIMEFRAME TARGETS  ({tfs_run})')
        if stopped:
            print(f'    Cascade stopped at {stop_tf} — '
                  f'resistance wall reached, no projection beyond')
        print()
        for r in htf_results:
            _print_tf_block(r)

        if htf_best:
            print(f'  {"═"*w}')
            print(f'  ★  BEST HIGHER-TIMEFRAME FORECAST  ({tfs_run})')
            print(f'  {"─"*w}')
            print(f'  Consensus target : {htf_best["forecast"]}')
            print(f'  Upside           : +{htf_best["upside_pct"]} %')
            print(f'  Confidence       : {htf_best["confidence"]} %'
                  f'  (TF spread {htf_best["spread_pct"]} %)')
            print(f'  TFs used         : {htf_best["tfs_used"]}  '
                  f'(stopped at {htf_best["stopped_at"]})')
            print(f'  Method           : volume-weighted avg · HTF cascade')
            print(f'  {"═"*w}')
            print()
    else:
        print(f'  HTF forecast: insufficient data.\n')


# ─────────────────────────────────────────────
#  TIME GEOMETRY REPORT PRINTER  (φ · e · π)
# ─────────────────────────────────────────────

def run_time_geometry(pair, label_map, current_price, sel_detail,
                      stf_results, htf_results):
    """
    Run and print the full φ·e·π time geometry report.

    Uses the 1m close/low/high arrays already stored in sel_detail
    (from check_dip_conditions — no extra fetch needed).

    Sections:
      1. φ·e·π dip geometry score (from check_dip_conditions)
      2. Hilbert cycle analysis (π component)
      3. Complex exponential fit (e·π bridge)
      4. Golden spiral targets (φ·π bridge)
      5. φ Fibonacci extension levels (φ component)
      6. MTF harmony score (all three constants across TFs)
    """
    lbl = label_map.get(pair, pair.replace('USDC', ''))
    w   = 62

    d          = sel_detail.get(pair, {})
    close_arr  = d.get('close_arr')
    swing_low  = d.get('swing_low')
    swing_high = d.get('swing_high')
    geo_d      = d.get('geometry_detail', {})

    if close_arr is None or len(close_arr) < 32:
        print(f'  Time geometry: insufficient 1m data.\n')
        return

    # ── run analyses ─────────────────────────────────────────
    ht   = hilbert_cycle_analysis(close_arr)
    cexp = complex_exp_forecast(close_arr, current_price)
    phi_levels = phi_extension_levels(swing_low, swing_high, current_price) \
                 if (swing_low and swing_high) else []
    gs   = golden_spiral_targets(
        current_price,
        ht.get('ht_phase_deg') if ht else None,
        ht.get('ht_amplitude') if ht else None,
        swing_low=swing_low, swing_high=swing_high,
    ) if ht else None
    harm_score, harm_d = mtf_harmony_score(stf_results, htf_results)

    print(f'\n  {"═"*w}')
    print(f'  ◈  TIME GEOMETRY REPORT  ·  φ · e · π  ·  {lbl}')
    print(f'  {"═"*w}')
    print(f'  φ = structure/proportion   e = evolution/time   π = cycles/rotation')
    print(f'  Together → GOLDEN SPIRAL: r(θ) = A·e^(b·θ),  b = ln(φ)/(π/2)')
    print()

    # ── section 1: φ·e·π dip geometry score ─────────────────
    print(f'  ┌─ φ·e·π DIP GEOMETRY SCORE  (1m, {EXTREMA_LOOKBACK} bars) {"─"*18}┐')
    if geo_d:
        print(f'  │  φ  (structural depth)  : {geo_d.get("phi_devs","?")} φ-deviations below trendline')
        print(f'  │    → score              : {geo_d.get("phi_score","?"):>5} / 40')
        print(f'  │  e  (momentum decay)    : α R²={geo_d.get("e_r2","?")}  decay_rate={geo_d.get("e_decay_rate","?")}')
        print(f'  │    → score              : {geo_d.get("e_score","?"):>5} / 30')
        print(f'  │  π  (cycle phase)       : HT phase={geo_d.get("ht_phase_now","?")}°  (270°=trough)')
        print(f'  │    → score              : {geo_d.get("pi_score","?"):>5} / 30')
        tot = geo_d.get('total', 0)
        bar = '█' * int(tot // 5) + '░' * (20 - int(tot // 5))
        print(f'  │  ─────────────────────────────────────────────────────────')
        print(f'  │  GEOMETRY SCORE         : {tot:>5.1f} / 100   [{bar}]')
    else:
        print(f'  │  (geometry data unavailable)')
    print(f'  └{"─"*60}┘')
    print()

    # ── section 2: Hilbert Transform (π component) ───────────
    print(f'  ┌─ π COMPONENT — Hilbert Transform Cycle (1m) {"─"*14}┐')
    if ht:
        bz_tag = '  ← BUY ZONE ✔' if ht.get('in_buy_zone') else ''
        tm_tag = 'CYCLING' if ht.get('ht_trend_mode') == 0 else \
                 ('TRENDING' if ht.get('ht_trend_mode') == 1 else '?')
        print(f'  │  Dominant HT period : {ht.get("ht_period","?")} bars')
        print(f'  │  Current phase      : {ht.get("ht_phase_deg","?")}°  '
              f'({ht.get("phase_label","?")})')
        print(f'  │  Sine / Lead sine   : {ht.get("ht_sine","?")} / '
              f'{ht.get("ht_lead_sine","?")}{bz_tag}')
        print(f'  │  HT amplitude       : {ht.get("ht_amplitude","?")}')
        print(f'  │  Market mode        : {tm_tag}')
        print(f'  │  Bars to trough     : {ht.get("bars_to_trough","?")}  '
              f'(phase 270° = lowest cycle point)')
    else:
        print(f'  │  (Hilbert Transform data unavailable)')
    print(f'  └{"─"*60}┘')
    print()

    # ── section 3: complex exponential fit (e·π bridge) ──────
    CEXP_MIN_R2 = 0.30   # below this R² the fit is unreliable — warn, don't forecast
    print(f'  ┌─ e·π BRIDGE — Complex Exponential Fit  y=A·e^(α+iω)t {"─"*3}┐')
    if cexp:
        r2_val = cexp.get('fit_r2', 0.0)
        r2_ok  = r2_val >= CEXP_MIN_R2
        print(f'  │  α  (exp growth rate)  : {cexp.get("alpha","?")}  '
              f'← {cexp.get("e_label","")}')
        print(f'  │  ω  (angular freq)     : {cexp.get("omega_rad","?")} rad/bar')
        print(f'  │  Period  (2π/ω)        : {cexp.get("period_est","?")} bars')
        r2_warn = '' if r2_ok else f'  ⚠ UNRELIABLE (min {CEXP_MIN_R2})'
        print(f'  │  Envelope fit  R²      : {r2_val}{r2_warn}')
        print(f'  │  Bars projected        : {cexp.get("bars_forward","?")}')
        if r2_ok and cexp.get('forecast'):
            up = (cexp['forecast'] - current_price) / current_price * 100.0
            print(f'  │  Complex forecast      : {cexp["forecast"]}  ({up:+.2f}%)')
        else:
            print(f'  │  Complex forecast      : suppressed (R²={r2_val} < {CEXP_MIN_R2})')
    else:
        print(f'  │  (complex exponential data unavailable)')
    print(f'  └{"─"*60}┘')
    print()

    # ── section 4: golden spiral targets (φ·π bridge) ────────
    print(f'  ┌─ φ·π BRIDGE — Golden Spiral Targets  r(θ)=A·e^(b·θ) {"─"*4}┐')
    if gs:
        print(f'  │  Amplitude base (swing/2): {gs.get("A_base","?")}  '
              f'({gs.get("A_base",0)/current_price*100:.2f}% of price)')
        print(f'  │  Current spiral angle    : {gs.get("current_angle","?")}°')
        for label_s, key, phi_mult_key in [
            (f'+π/2  (×φ  =×{PHI:.3f})',  'gs_q1_turn',   'phi_mult_q1'),
            (f'+π    (×φ² =×{PHI2:.3f})',  'gs_half_turn', 'phi_mult_half'),
            (f'+3π/2 (×φ³ =×{PHI**3:.3f})','gs_3q_turn',  None),
            (f'+2π   (×φ⁴ =×{PHI**4:.3f})','gs_full_turn', None),
        ]:
            v = gs.get(key)
            if v:
                pct = (v - current_price) / current_price * 100.0
                mult = f'  (×{gs[phi_mult_key]} radius)' if phi_mult_key and gs.get(phi_mult_key) else ''
                print(f'  │  {label_s:<26}: {v}  ({pct:+.2f}%){mult}')
        print(f'  │  GOLDEN_B = ln(φ)/(π/2) = {GOLDEN_B:.5f}')
    else:
        print(f'  │  (golden spiral data unavailable — need HT phase)')
    print(f'  └{"─"*60}┘')
    print()

    # ── section 5: φ Fibonacci extension levels ──────────────
    print(f'  ┌─ φ COMPONENT — Fibonacci Extension Levels (from swing) {"─"*3}┐')
    if phi_levels:
        print(f'  │  Swing low  : {swing_low:.8f}    '
              f'Swing high : {swing_high:.8f}')
        print(f'  │  Range      : {swing_high - swing_low:.8f}')
        print(f'  │  Entry      : {current_price:.8f}')
        print(f'  │  {"─"*54}')
        for ratio, label_txt, level in phi_levels[:7]:   # show top 7
            pct = (level - current_price) / current_price * 100.0
            marker = ' ← φ' if abs(ratio - PHI_INV) < 0.01 or abs(ratio - PHI) < 0.01 \
                     else (' ← φ²' if abs(ratio - PHI2) < 0.01 else '')
            print(f'  │  {label_txt:<28}: {level:.8f}  '
                  f'(+{pct:.2f}%){marker}')
    else:
        print(f'  │  (insufficient swing data for Fibonacci levels)')
    print(f'  └{"─"*60}┘')
    print()

    # ── section 6: MTF harmony score ─────────────────────────
    print(f'  ┌─ MTF HARMONY SCORE  (φ·e·π alignment across all TFs) {"─"*3}┐')
    if harm_d:
        n_tfs = len(stf_results or []) + len(htf_results or [])
        print(f'  │  TFs analysed         : {n_tfs}')
        print(f'  │  φ harmony (periods)  : {harm_d.get("phi_pairs","?")}/{harm_d.get("total_pairs","?")} '
              f'period pairs in φ-ratio  → {harm_d.get("phi_harmony","?")} pts')
        print(f'  │  e harmony (consensus): CV={harm_d.get("fc_cv_pct","?")}%  '
              f'→ {harm_d.get("e_harmony","?")} pts')
        print(f'  │  π harmony (direction): {harm_d.get("n_up","?")}/{harm_d.get("n_total","?")} TFs upside  '
              f'→ {harm_d.get("pi_harmony","?")} pts')
        print(f'  │  No-stop bonus        : {harm_d.get("no_stop_bonus","?")} pts')
        hs = harm_d.get('harmony', 0)
        bar = '█' * int(hs // 5) + '░' * (20 - int(hs // 5))
        print(f'  │  ─────────────────────────────────────────────────────────')
        print(f'  │  HARMONY SCORE        : {hs:>5.1f} / 100   [{bar}]')
        if   hs >= 80: harmony_label = 'STRONG — all constants aligned'
        elif hs >= 60: harmony_label = 'MODERATE — partial alignment'
        elif hs >= 40: harmony_label = 'WEAK — limited alignment'
        else:          harmony_label = 'DISCORD — constants not aligned'
        print(f'  │  Interpretation       : {harmony_label}')
    else:
        print(f'  │  (insufficient TF data for harmony score)')
    print(f'  └{"─"*60}┘')
    print()

    print(f'  {"═"*w}\n')


# ═════════════════════════════════════════════════════════════
#  MULTI-TF ARGMIN/ARGMAX CONFIRMATION FILTER
#  Passes only if ALL of 1m, 3m, 5m independently confirm:
#    argmin(low  wicks, last 500 bars) > argmax(high wicks, last 500 bars)
#  i.e. the bar with the LOWEST WICK (true trough) is more recent
#  than the bar with the HIGHEST WICK (true peak) on every short TF.
#  Uses low[] for argmin and high[] for argmax — not close prices.
#  Also computes per-TF price thresholds (true min/mid/max from wicks).
# ═════════════════════════════════════════════════════════════

MULTI_TF_LOOKBACK = 1000
_MULTI_TF_LIST    = [('1m', '1m'), ('3m', '3m'), ('5m', '5m')]

def multi_tf_argmin_check(pair):
    """
    Returns (passed, threshold_dict).
    threshold_dict: {tf_label: {min_price, max_price, mid_price, argmin, argmax, close_arr}}

    argmin = index of the true LOWEST LOW (wick) across all MULTI_TF_LOOKBACK bars
    argmax = index of the true HIGHEST HIGH (wick) across all MULTI_TF_LOOKBACK bars

    passed = True only when argmin > argmax on 1m AND 3m AND 5m:
      → the absolute lowest wick in the window is MORE RECENT
        than the absolute highest wick on every short TF.
    """
    thresholds = {}
    all_passed = True
    for label, interval in _MULTI_TF_LIST:
        try:
            klines = trader.client.get_klines(
                symbol=pair, interval=interval, limit=MULTI_TF_LOOKBACK
            )
        except Exception:
            return False, {}
        if len(klines) < 50:
            return False, {}
        close = np.array([float(k[4]) for k in klines], dtype=np.float64)
        low_  = np.array([float(k[3]) for k in klines], dtype=np.float64)  # true low wicks
        high_ = np.array([float(k[2]) for k in klines], dtype=np.float64)  # true high wicks

        # True extremes across ALL bars in the window — not closes, actual wicks
        amin_idx = int(np.argmin(low_))    # bar with the lowest  wick (deepest trough)
        amax_idx = int(np.argmax(high_))   # bar with the highest wick (tallest peak)
        tf_pass  = amin_idx > amax_idx     # deepest low is MORE RECENT than highest high
        if not tf_pass:
            all_passed = False
        thresholds[label] = {
            'argmin':    amin_idx,
            'argmax':    amax_idx,
            'min_price': round(float(np.min(low_)),  8),   # true lowest  low
            'max_price': round(float(np.max(high_)), 8),   # true highest high
            'mid_price': round(float((np.min(low_) + np.max(high_)) / 2.0), 8),
            'close_arr': close,
            'low_arr':   low_,
            'high_arr':  high_,
            'passed':    tf_pass,
        }
    return all_passed, thresholds


def run_multi_tf_argmin_stage(pairs):
    """Parallel multi-TF argmin confirmation. Returns (passed_list, threshold_map)."""
    out       = []
    thr_map   = {}
    lock      = threading.Lock()
    total     = len(pairs)
    done      = [0]

    def worker(p):
        passed, td = multi_tf_argmin_check(p)
        with lock:
            thr_map[p] = td
            if passed:
                out.append(p)
            done[0] += 1
            pct = int(done[0] / total * 100)
            bar = '█' * (pct // 4) + '░' * (25 - pct // 4)
            print(f'\r  mtf-argmin  [{bar}] {pct:3d}%  {done[0]}/{total}',
                  end='', flush=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        fs = [pool.submit(worker, p) for p in pairs]
        for f in as_completed(fs): pass
    print()
    return out, thr_map


def print_multi_tf_threshold_table(pairs, label_map, thr_map):
    """Print per-TF argmin/argmax + price threshold table."""
    if not pairs:
        return
    w = 80
    print(f'\n  ┌─ MULTI-TF ARGMIN CONFIRMATION  (1m · 3m · 5m, last {MULTI_TF_LOOKBACK} bars) {"─"*6}┐')
    hdr = (f'  │  {"Ticker":<10}  {"TF":>3}  '
           f'{"ArgMin":>7}  {"ArgMax":>7}  '
           f'{"Min Price":>14}  {"Mid Price":>14}  {"Max Price":>14}  {"OK":>4}  │')
    sep = '  │' + '─' * (len(hdr) - 4) + '│'
    print(sep); print(hdr); print(sep)
    for p in pairs:
        td  = thr_map.get(p, {})
        lbl = label_map.get(p, p.replace('USDC', ''))
        if not td:
            print(f'  │  {lbl:<10}  {"—":>3}  {"—":>7}  {"—":>7}  {"—":>14}  {"—":>14}  {"—":>14}  {"—":>4}  │')
            continue
        first = True
        for tf_label in ('1m', '3m', '5m'):
            t  = td.get(tf_label, {})
            if not t:
                continue
            ok  = '✔' if t.get('passed') else '✗'
            pr_fmt = lambda v: f'{v:.6f}' if v and v < 1 else f'{v:.4f}'
            lbl_col = lbl if first else ''
            print(f'  │  {lbl_col:<10}  {tf_label:>3}  '
                  f'{t["argmin"]:>7}  {t["argmax"]:>7}  '
                  f'{pr_fmt(t["min_price"]):>14}  '
                  f'{pr_fmt(t["mid_price"]):>14}  '
                  f'{pr_fmt(t["max_price"]):>14}  {ok:>4}  │')
            first = False
    print(sep + '\n')


# ═════════════════════════════════════════════════════════════
#  ML COMPOUND FORECAST ENGINE
#  Algorithms:
#   1.  Linear Regression Channel  (last 500 bars, ±σ bands)
#   2.  Polynomial Regression       (degree 3)
#   3.  Ridge Regression            (regularized linear)
#   4.  Lasso Regression            (sparse)
#   5.  Bayesian Ridge              (probabilistic linear)
#   6.  Gaussian Process            (uncertainty-aware, RBF+Matern)
#   7.  Random Forest               (300 trees, OOB score)
#   8.  Gradient Boosting           (300 estimators)
#   9.  SVR                         (RBF kernel)
#  10.  Random Walk Monte Carlo     (10 000 paths)
#
#  Feature matrix (built from all available data):
#   Time index, log-returns (1/2/3/5/10/20 bars), rolling
#   mean/std (10/20/50 bars), RSI-14, CMO-14, MOM-10,
#   ATR-14 (normalised), Bollinger-band position,
#   φ-deviations, HT phase (sin+cos), HT period (norm),
#   FFT dominant period (norm), e-decay alpha,
#   volume bull/bear ratio, argmin/argmax distance ratio,
#   Fibonacci distance from swing low/high,
#   golden-spiral angle (sin+cos).
#
#  Instant backtesting:
#   Split last N bars: 80% train, 20% test (walk-forward).
#   Reports MAE, RMSE, directional accuracy per model.
#   Ensemble = median of all model point forecasts
#   weighted by (1 / MAE) on the test window.
#
#  Volume S/R re-computed fresh on 1m 500-bar data.
# ═════════════════════════════════════════════════════════════

ML_LOOKBACK   = 1000   # bars for ML training window
ML_TEST_RATIO = 0.20   # 20% held out for backtesting
ML_WALKS      = 10_000 # Monte-Carlo random walk paths
ML_HORIZON    = 30     # bars to project forward (1m bars)


# ── feature engineering ───────────────────────────────────

def _safe_talib(fn, *arrays, **kw):
    """
    Wrapper for talib functions with variable number of input arrays.
    Single-array:  _safe_talib(ta.RSI, close, timeperiod=14)
    Multi-array:   _safe_talib(ta.ATR, high, low, close, timeperiod=14)
    Returns zeros (same length as first array) on any error.
    """
    try:
        res = fn(*arrays, **kw)
        return np.where(np.isnan(res), 0.0, res)
    except Exception:
        return np.zeros(len(arrays[0]))


def build_feature_matrix(close, volume, high, low,
                          geo_detail=None, ht_data_arr=None,
                          phi_devs=0.0, e_alpha=0.0,
                          fft_period=20, ht_period=20,
                          swing_low=None, swing_high=None):
    """
    Build a 2-D feature matrix  X  of shape (n_samples, n_features)
    and target vector  y  (next-bar close, shifted by 1).

    Every feature is a scalar time-series aligned to the same bar index.
    NaN-padding at the start is zeroed.

    Features:
      [0]  t_norm          — time index  / n  (0→1)
      [1]  log_ret1        — log return lag 1
      [2]  log_ret2        — log return lag 2
      [3]  log_ret3        — log return lag 3
      [4]  log_ret5        — log return lag 5
      [5]  log_ret10       — log return lag 10
      [6]  log_ret20       — log return lag 20
      [7]  roll_mean10     — rolling mean 10
      [8]  roll_std10      — rolling std  10
      [9]  roll_mean20     — rolling mean 20
      [10] roll_std20      — rolling std  20
      [11] roll_mean50     — rolling mean 50
      [12] roll_std50      — rolling std  50
      [13] rsi14           — RSI-14
      [14] cmo14           — CMO-14
      [15] mom10           — MOM-10
      [16] atr14_norm      — ATR-14 / close (%)
      [17] bb_pos          — Bollinger position  (close - lower)/(upper-lower)
      [18] phi_dev         — φ-deviations below trendline
      [19] ht_phase_sin    — sin(HT_DCPHASE)
      [20] ht_phase_cos    — cos(HT_DCPHASE)
      [21] ht_period_norm  — HT dominant period / n
      [22] fft_period_norm — FFT dominant period / n
      [23] e_alpha         — exponential decay rate
      [24] bull_vol_ratio  — bull volume / total volume (rolling 20)
      [25] argmin_dist     — (n-1-argmin) / n  (recency of low)
      [26] argmax_dist     — (n-1-argmax) / n  (recency of high)
      [27] fib_dist_low    — (close - swing_low) / range
      [28] fib_dist_high   — (swing_high - close) / range
      [29] gs_angle_sin    — sin(golden_spiral_angle)
      [30] gs_angle_cos    — cos(golden_spiral_angle)
    """
    n   = len(close)
    arr = np.asarray(close, dtype=np.float64)
    vol = np.asarray(volume, dtype=np.float64)
    hi  = np.asarray(high,   dtype=np.float64)
    lo  = np.asarray(low,    dtype=np.float64)

    def safe_roll_mean(a, w):
        out = np.full(n, 0.0)
        for i in range(w - 1, n):
            out[i] = np.mean(a[i - w + 1: i + 1])
        return out

    def safe_roll_std(a, w):
        out = np.full(n, 1e-10)
        for i in range(w - 1, n):
            out[i] = np.std(a[i - w + 1: i + 1]) + 1e-10
        return out

    def log_ret_lag(lag):
        out = np.zeros(n)
        for i in range(lag, n):
            if arr[i - lag] > 1e-20:
                out[i] = np.log(arr[i] / arr[i - lag])
        return out

    # time index
    t_norm = np.arange(n, dtype=np.float64) / max(n - 1, 1)

    # log returns
    lr1  = log_ret_lag(1);  lr2  = log_ret_lag(2);  lr3  = log_ret_lag(3)
    lr5  = log_ret_lag(5);  lr10 = log_ret_lag(10); lr20 = log_ret_lag(20)

    # rolling stats
    rm10 = safe_roll_mean(arr, 10);  rs10 = safe_roll_std(arr, 10)
    rm20 = safe_roll_mean(arr, 20);  rs20 = safe_roll_std(arr, 20)
    rm50 = safe_roll_mean(arr, 50);  rs50 = safe_roll_std(arr, 50)

    # talib indicators
    rsi14   = _safe_talib(ta.RSI,  arr, timeperiod=14) / 100.0
    cmo14   = _safe_talib(ta.CMO,  arr, timeperiod=14) / 100.0
    mom10   = _safe_talib(ta.MOM,  arr, timeperiod=10)
    mom10_n = mom10 / (arr + 1e-20)                               # normalise

    atr14_r = _safe_talib(ta.ATR, hi, lo, arr, timeperiod=14)
    atr14_n = atr14_r / (arr + 1e-20)

    # Bollinger position
    upper_b = rm20 + 2.0 * rs20
    lower_b = rm20 - 2.0 * rs20
    bb_pos  = (arr - lower_b) / (upper_b - lower_b + 1e-20)

    # φ deviation series — use rolling regression
    phi_dev_ts = np.zeros(n)
    win = min(50, n)
    for i in range(win, n):
        seg  = arr[i - win: i + 1]
        t_   = np.arange(len(seg), dtype=np.float64)
        p_   = np.polyfit(t_, seg, 1)
        tr_  = np.poly1d(p_)(t_)
        sg   = float(np.std(seg)) + 1e-20
        phi_dev_ts[i] = max(0.0, (tr_[-1] - seg[-1]) / sg)

    # HT phase series (time-series, not single point)
    ht_ph_ts = _safe_talib(ta.HT_DCPHASE, arr)
    ht_pe_ts = _safe_talib(ta.HT_DCPERIOD, arr)
    ht_ph_sin = np.sin(np.deg2rad(ht_ph_ts))
    ht_ph_cos = np.cos(np.deg2rad(ht_ph_ts))
    ht_pe_n   = ht_pe_ts / max(n, 1)

    # FFT period (scalar) broadcast to series
    fft_per_n = np.full(n, fft_period / max(n, 1))

    # e-alpha (scalar) broadcast
    e_alpha_ts = np.full(n, float(e_alpha))

    # bull volume ratio rolling 20
    open_arr  = np.asarray([0.0] * n)    # placeholder; use close as proxy if open unavailable
    bull_mask = arr >= np.roll(arr, 1)   # close >= prev close as proxy
    bull_mask[0] = False
    bull_vol_r = np.zeros(n)
    for i in range(20, n):
        bv = vol[i - 20: i + 1][bull_mask[i - 20: i + 1]].sum()
        tv = vol[i - 20: i + 1].sum() + 1e-20
        bull_vol_r[i] = bv / tv

    # argmin / argmax recency series
    # Use low wicks for argmin (true deepest price reached)
    # Use high wicks for argmax (true highest price reached)
    # This gives the correct recency signal: how recently was the
    # ACTUAL lowest low (not lowest close) compared to actual highest high.
    argmin_d = np.zeros(n)
    argmax_d = np.zeros(n)
    look_ml  = min(100, n)
    for i in range(look_ml, n):
        seg_lo = lo[i - look_ml: i + 1]   # low wick window
        seg_hi = hi[i - look_ml: i + 1]   # high wick window
        argmin_d[i] = (look_ml - int(np.argmin(seg_lo))) / look_ml  # recency of lowest wick
        argmax_d[i] = (look_ml - int(np.argmax(seg_hi))) / look_ml  # recency of highest wick

    # Fibonacci distances
    sl = swing_low  if swing_low  else float(np.min(arr))
    sh = swing_high if swing_high else float(np.max(arr))
    rng = (sh - sl) + 1e-20
    fib_low  = (arr - sl) / rng
    fib_high = (sh - arr) / rng

    # golden spiral angle series
    gs_theta  = np.deg2rad(ht_ph_ts) * GOLDEN_B
    gs_sin    = np.sin(gs_theta)
    gs_cos    = np.cos(gs_theta)

    # stack
    X = np.column_stack([
        t_norm, lr1, lr2, lr3, lr5, lr10, lr20,
        rm10, rs10, rm20, rs20, rm50, rs50,
        rsi14, cmo14, mom10_n, atr14_n, bb_pos,
        phi_dev_ts, ht_ph_sin, ht_ph_cos, ht_pe_n,
        fft_per_n, e_alpha_ts,
        bull_vol_r, argmin_d, argmax_d,
        fib_low, fib_high,
        gs_sin, gs_cos,
    ])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# ── volume S/R anchored to argmin / argmax extrema ───────

def compute_volume_sr(close, volume, high, low, open_,
                      current_price, pair=None, lookback=100, bin_pct=0.003):
    """
    Real Volume S/R — unconstrained scan over last `lookback` 1m bars.

    ── WHAT THIS FINDS ─────────────────────────────────────────────────
    The most significant, most consistent, and growing bullish/bearish
    volume clusters — located ANYWHERE in the full price range.
    No "below entry only" / "above entry only" restriction.

    ── THREE SCORING DIMENSIONS ────────────────────────────────────────
    For every price bin we compute three metrics and combine them:

      1. RAW VOLUME  — total bull or bear volume accumulated in that bin.
         Identifies where the most capital changed hands.

      2. CONSISTENCY — how many of the last `lookback` bars touched that
         bin.  Bins hit repeatedly are structurally significant levels.
         consistency = n_bars_in_bin / total_bars  (0→1)

      3. MOMENTUM (vol growth) — split the window in half.
         vol_growth = vol_second_half / (vol_first_half + ε) − 1
         >0 = volume is increasing at that price → area building.
         <0 = volume fading → level weakening.

      composite_score = raw_vol × (1 + consistency) × max(1, 1 + vol_growth)

    The top-3 bins by composite_score are the real S/R levels.
    No predefined zone filter.  Zone label is ADDED AFTER ranking.

    ── ORDERBOOK WALLS ─────────────────────────────────────────────────
    If `pair` is provided, fetches live orderbook (depth=50).
    Finds the largest bid wall (support) and ask wall (resistance).
    These are reported alongside OHLCV levels but do NOT override them.

    ── VOLUME PROFILE (full window) ────────────────────────────────────
    Total volume (all bars, typical price (H+L+C)/3):
      POC = Point of Control — bin with most total volume
      VAH / VAL = Value Area High/Low (70% of volume around POC)

    ── MARKET PROFILE (TPO) ────────────────────────────────────────────
    Bar count per typical-price bin:
      mPOC / mVAH / mVAL = most time spent / 70% time cluster

    ── BIAS (full window) ──────────────────────────────────────────────
    bull_pct vs bear_pct from ALL bars in window.
    predominance = 'BULLISH' (≥55% bull) | 'BEARISH' (≥55% bear) | 'NEUTRAL'

    ── RETURNS ─────────────────────────────────────────────────────────
    (support_levels, resistance_levels, profile)

    Each level tuple (9 fields):
      [0] price          — bin center price
      [1] raw_vol        — total side volume at bin
      [2] n_bars         — bar count at bin (consistency)
      [3] sr_floor       — bottom edge of 70%-vol cluster
      [4] sr_ceiling     — top    edge of 70%-vol cluster
      [5] range_pct      — (ceiling-floor)/price × 100
      [6] zone_label     — where it landed vs argmin/argmax/entry
      [7] dist_pct       — signed % distance from current_price
      [8] vol_pct        — % of that side's total volume
      [9] composite      — composite score (vol × consistency × growth)
      [10] vol_growth    — volume growth ratio (recent half vs older half)
      [11] consistency   — n_bars / total_bars

    profile dict keys:
      vol_poc, vol_vah, vol_val, mkt_poc, mkt_vah, mkt_val,
      bull_vol, bear_vol, total_vol, bull_pct, bear_pct, predominance,
      vol_bins [(price,vol)…], mkt_bins [(price,count)…],
      absolute_min, absolute_max,
      ob_bid_wall  — (price, qty) largest bid wall or None
      ob_ask_wall  — (price, qty) largest ask wall or None
    """
    close  = np.asarray(close,  dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    high_  = np.asarray(high,   dtype=np.float64)
    low_   = np.asarray(low,    dtype=np.float64)
    open__ = np.asarray(open_,  dtype=np.float64)
    n_full = len(close)

    if n_full < 5:
        return [], [], {}

    # ── use last `lookback` bars for S/R scoring ──────────────
    lb     = min(lookback, n_full)
    half   = lb // 2
    c      = close[-lb:];   v  = volume[-lb:]
    h      = high_[-lb:];   lo = low_[-lb:]
    op     = open__[-lb:]
    n      = lb

    # ── full window for bias + profile ────────────────────────
    c_full = close;  v_full = volume
    h_full = high_;  l_full = low_
    o_full = open__

    cp           = float(current_price)
    absolute_min = float(np.min(low_))
    absolute_max = float(np.max(high_))

    full_range = absolute_max - absolute_min + 1e-20
    bs = max(full_range * bin_pct, 1e-12)

    def _bin(price):
        return round(round(price / bs) * bs, 10)

    argmin_bin = _bin(absolute_min)
    argmax_bin = _bin(absolute_max)
    entry_bin  = _bin(cp)

    # ── zone classifiers (no restriction — pure labelling) ────
    def _zone(price):
        b = _bin(price)
        if   b <  argmin_bin: pos = f'BELOW_argmin(<{absolute_min:.5g})'
        elif b == argmin_bin: pos = f'AT_argmin(={absolute_min:.5g})'
        elif b <  entry_bin:  pos = f'between_argmin_entry'
        elif b == entry_bin:  pos = f'AT_entry(≈{cp:.5g})'
        elif b <  argmax_bin: pos = f'between_entry_argmax'
        elif b == argmax_bin: pos = f'AT_argmax(={absolute_max:.5g})'
        else:                 pos = f'ABOVE_argmax(>{absolute_max:.5g})'
        return pos

    # ── scan lookback window — bull and bear bins ─────────────
    # first half and second half for growth detection
    bull_h1 = {}; bull_h2 = {}; bull_cnt = {}
    bear_h1 = {}; bear_h2 = {}; bear_cnt = {}

    for i in range(n):
        cl_i = float(c[i]); op_i = float(op[i]); v_i = float(v[i])
        cb   = _bin(cl_i)
        is_bull = cl_i >= op_i
        in_h2   = i >= half
        if is_bull:
            bull_cnt[cb]  = bull_cnt.get(cb, 0) + 1
            if in_h2:  bull_h2[cb] = bull_h2.get(cb, 0.0) + v_i
            else:      bull_h1[cb] = bull_h1.get(cb, 0.0) + v_i
        else:
            bear_cnt[cb]  = bear_cnt.get(cb, 0) + 1
            if in_h2:  bear_h2[cb] = bear_h2.get(cb, 0.0) + v_i
            else:      bear_h1[cb] = bear_h1.get(cb, 0.0) + v_i

    # merge into total bins
    all_bins_b = set(bull_h1) | set(bull_h2)
    all_bins_r = set(bear_h1) | set(bear_h2)
    bull_bins  = {b: bull_h1.get(b, 0.0) + bull_h2.get(b, 0.0) for b in all_bins_b}
    bear_bins  = {b: bear_h1.get(b, 0.0) + bear_h2.get(b, 0.0) for b in all_bins_r}

    total_bull_lb = sum(bull_bins.values()) + 1e-20
    total_bear_lb = sum(bear_bins.values()) + 1e-20

    # ── full-window bias + profile ────────────────────────────
    vol_bins_fp = {}; mkt_bins_fp = {}
    total_bull_fw = 0.0; total_bear_fw = 0.0

    for i in range(n_full):
        cl_i = float(c_full[i]); op_i = float(o_full[i]); v_i = float(v_full[i])
        tp_i = (float(h_full[i]) + float(l_full[i]) + cl_i) / 3.0
        tb   = _bin(tp_i)
        vol_bins_fp[tb] = vol_bins_fp.get(tb, 0.0) + v_i
        mkt_bins_fp[tb] = mkt_bins_fp.get(tb, 0)   + 1
        if cl_i >= op_i: total_bull_fw += v_i
        else:            total_bear_fw += v_i

    total_vol_fw = total_bull_fw + total_bear_fw + 1e-20
    bull_pct_fw  = total_bull_fw / total_vol_fw * 100.0
    bear_pct_fw  = total_bear_fw / total_vol_fw * 100.0
    if   bull_pct_fw >= 55.0: predominance = 'BULLISH'
    elif bear_pct_fw >= 55.0: predominance = 'BEARISH'
    else:                      predominance = 'NEUTRAL'

    # ── POC / VAH / VAL ───────────────────────────────────────
    def _poc_va(bins):
        if not bins: return None, None, None
        poc  = max(bins, key=bins.get)
        tv   = sum(bins.values()); tgt = tv * 0.70
        cl   = {poc: bins[poc]}; cv = bins[poc]
        ap   = sorted(bins.keys())
        if poc not in ap: return round(poc,8), round(poc,8), round(poc,8)
        li = ap.index(poc); hi = li
        while cv < tgt:
            cl_ = li > 0; ch = hi < len(ap)-1
            if not cl_ and not ch: break
            lv = bins.get(ap[li-1], 0.0) if cl_ else 0.0
            hv = bins.get(ap[hi+1], 0.0) if ch  else 0.0
            if lv >= hv and cl_:
                li -= 1; cv += lv; cl[ap[li]] = lv
            elif ch:
                hi += 1; cv += hv; cl[ap[hi]] = hv
            else: break
        return (round(poc,8),
                round(max(cl.keys())+bs*0.5, 8),
                round(min(cl.keys())-bs*0.5, 8))

    vol_poc, vol_vah, vol_val = _poc_va(vol_bins_fp)
    mkt_poc, mkt_vah, mkt_val = _poc_va(mkt_bins_fp)

    # ── composite scoring + level building ───────────────────
    def _build_levels(bins, cnt, h1, h2, side_total, strongest_n=3):
        if not bins: return []
        all_p  = sorted(bins.keys())
        all_vol = sum(bins.values())

        scored = []
        for price, raw_vol in bins.items():
            n_bars      = cnt.get(price, 0)
            consistency = n_bars / n
            v1 = h1.get(price, 0.0); v2 = h2.get(price, 0.0)
            vol_growth  = (v2 / (v1 + 1e-20)) - 1.0   # >0 = growing
            composite   = raw_vol * (1.0 + consistency) * max(1.0, 1.0 + vol_growth)
            scored.append((price, raw_vol, n_bars, consistency, vol_growth, composite))

        # rank by composite score descending
        scored.sort(key=lambda x: x[5], reverse=True)
        top = scored[:strongest_n]

        levels = []
        for price, raw_vol, n_bars, consistency, vol_growth, composite in top:
            vol_pct  = round(raw_vol / (side_total + 1e-20) * 100.0, 1)
            dist_pct = round((price - cp) / (cp + 1e-20) * 100.0, 3)

            # SR cluster range: walk outward from peak until 70% of side vol
            if len(bins) == 1:
                sr_fl = round(price - bs*0.5, 8); sr_ce = round(price + bs*0.5, 8)
            else:
                tgt = all_vol * 0.70
                cl  = {price: raw_vol}; cv = raw_vol
                idx = all_p.index(price) if price in all_p else 0
                li = idx; hi = idx
                while cv < tgt:
                    can_l = li > 0; can_h = hi < len(all_p)-1
                    if not can_l and not can_h: break
                    lv = bins.get(all_p[li-1],0.0) if can_l else 0.0
                    hv = bins.get(all_p[hi+1],0.0) if can_h else 0.0
                    if lv >= hv and can_l:
                        li -= 1; cv += lv; cl[all_p[li]] = lv
                    elif can_h:
                        hi += 1; cv += hv; cl[all_p[hi]] = hv
                    else: break
                sr_fl = round(min(cl.keys())-bs*0.5, 8)
                sr_ce = round(max(cl.keys())+bs*0.5, 8)

            range_pct = round((sr_ce - sr_fl) / (price + 1e-20) * 100.0, 3)
            levels.append((
                round(price,      8),   # [0] price
                round(raw_vol,    2),   # [1] raw vol
                n_bars,                 # [2] bar count
                sr_fl,                  # [3] sr_floor
                sr_ce,                  # [4] sr_ceiling
                range_pct,              # [5] range %
                _zone(price),           # [6] zone label
                dist_pct,               # [7] dist % from entry
                vol_pct,                # [8] % of side total
                round(composite, 2),    # [9] composite score
                round(vol_growth, 4),   # [10] volume growth
                round(consistency, 4),  # [11] consistency
            ))
        return levels

    support_levels    = _build_levels(bull_bins, bull_cnt, bull_h1, bull_h2, total_bull_lb)
    resistance_levels = _build_levels(bear_bins, bear_cnt, bear_h1, bear_h2, total_bear_lb)

    # ── guaranteed fallback ───────────────────────────────────
    if not support_levels:
        support_levels = [(round(absolute_min,8),0.0,0,
            round(absolute_min-bs*0.5,8), round(absolute_min+bs*0.5,8),
            round(bs/(absolute_min+1e-20)*100,3), _zone(absolute_min),
            round((absolute_min-cp)/(cp+1e-20)*100,3), 0.0, 0.0, 0.0, 0.0)]
    if not resistance_levels:
        resistance_levels = [(round(absolute_max,8),0.0,0,
            round(absolute_max-bs*0.5,8), round(absolute_max+bs*0.5,8),
            round(bs/(absolute_max+1e-20)*100,3), _zone(absolute_max),
            round((absolute_max-cp)/(cp+1e-20)*100,3), 0.0, 0.0, 0.0, 0.0)]

    # ── orderbook walls ───────────────────────────────────────
    ob_bid_wall = None; ob_ask_wall = None
    if pair:
        try:
            ob = trader.client.get_order_book(symbol=pair, limit=50)
            # bids: [[price, qty], …] — largest single qty = wall
            bids = [(float(b[0]), float(b[1])) for b in ob.get('bids', [])]
            asks = [(float(a[0]), float(a[1])) for a in ob.get('asks', [])]
            if bids:
                ob_bid_wall = max(bids, key=lambda x: x[1])
            if asks:
                ob_ask_wall = max(asks, key=lambda x: x[1])
        except Exception:
            pass

    profile = {
        # volume profile
        'vol_poc':      vol_poc,
        'vol_vah':      vol_vah,
        'vol_val':      vol_val,
        # market profile
        'mkt_poc':      mkt_poc,
        'mkt_vah':      mkt_vah,
        'mkt_val':      mkt_val,
        # bias (full window)
        'bull_vol':     round(total_bull_fw, 2),
        'bear_vol':     round(total_bear_fw, 2),
        'total_vol':    round(total_vol_fw,  2),
        'bull_pct':     round(bull_pct_fw,   2),
        'bear_pct':     round(bear_pct_fw,   2),
        'predominance': predominance,
        # charts
        'vol_bins':     sorted(vol_bins_fp.items()),
        'mkt_bins':     sorted(mkt_bins_fp.items()),
        # range info
        'absolute_min': absolute_min,
        'absolute_max': absolute_max,
        # orderbook
        'ob_bid_wall':  ob_bid_wall,
        'ob_ask_wall':  ob_ask_wall,
        # scan window
        'lookback':     lb,
    }

    return support_levels, resistance_levels, profile



# ── random walk Monte Carlo ───────────────────────────────

def random_walk_mc(close, horizon=ML_HORIZON, n_paths=ML_WALKS,
                   use_full_lookback=True, dip_drift_boost=True):
    """
    Monte Carlo random walk using empirical log-return distribution.

    Key improvements vs naive MC:
    1. Samples from the FULL ML_LOOKBACK window (not just last 100 bars),
       so a recent bearish streak doesn't poison the whole distribution.
    2. Mean-reversion drift boost: because we only call this on confirmed
       dip candidates, we inject a small positive drift equal to half
       the absolute mean log-return, encouraging recovery paths.
       This is conservative — it doesn't guarantee up, just doesn't
       assume the dip trend continues forever.
    3. Multi-horizon: returns projections at 3 horizons:
       short (horizon bars), medium (horizon*4), long (horizon*16).

    Returns dict or None.
    """
    arr  = np.asarray(close, dtype=np.float64)
    look = min(ML_LOOKBACK, len(arr)) if use_full_lookback else min(100, len(arr))
    log_rets = np.diff(np.log(arr[-look:] + 1e-20))
    if len(log_rets) < 10:
        return None

    # ── drift: use abs(mean) as floor so dip recovery is represented ──
    raw_drift = float(np.mean(log_rets))
    vol       = float(np.std(log_rets))
    if dip_drift_boost:
        # mean-reversion: half the absolute volatility as positive drift
        # This reflects the dip-recovery setup without over-inflating targets
        drift = max(raw_drift, vol * 0.15)
    else:
        drift = raw_drift

    rng_gen   = np.random.default_rng(seed=None)   # fresh seed each call
    last_price = float(arr[-1])

    def run_paths(h):
        # draw from full distribution + apply drift correction
        draws = rng_gen.choice(log_rets, size=(n_paths, h), replace=True)
        draws = draws + drift   # shift distribution toward recovery
        cum   = np.exp(draws.sum(axis=1)) * last_price
        return cum

    fp_s = run_paths(horizon)
    fp_m = run_paths(horizon * 4)
    fp_l = run_paths(horizon * 16)

    def pcts(fp):
        return {
            'p5':      round(float(np.percentile(fp,  5)), 8),
            'p25':     round(float(np.percentile(fp, 25)), 8),
            'p50':     round(float(np.percentile(fp, 50)), 8),
            'p75':     round(float(np.percentile(fp, 75)), 8),
            'p95':     round(float(np.percentile(fp, 95)), 8),
            'mean':    round(float(np.mean(fp)),           8),
            'std':     round(float(np.std(fp)),            8),
            'prob_up': round(float(np.mean(fp > last_price)), 4),
        }

    return {
        'short':      pcts(fp_s),   # horizon bars
        'medium':     pcts(fp_m),   # horizon*4 bars
        'long':       pcts(fp_l),   # horizon*16 bars
        # flat aliases for backward-compat with existing prints
        'p5':      pcts(fp_s)['p5'],
        'p25':     pcts(fp_s)['p25'],
        'p50':     pcts(fp_s)['p50'],
        'p75':     pcts(fp_s)['p75'],
        'p95':     pcts(fp_s)['p95'],
        'mean':    pcts(fp_s)['mean'],
        'std':     pcts(fp_s)['std'],
        'prob_up': pcts(fp_s)['prob_up'],
        'raw_drift':     round(raw_drift, 8),
        'drift_applied': round(drift,     8),
        'vol_per_bar':   round(vol,       8),
        'horizon_s':     horizon,
        'horizon_m':     horizon * 4,
        'horizon_l':     horizon * 16,
    }


# ── regression channel ────────────────────────────────────

def regression_channel(close, n_sigma=2.0):
    """
    Linear regression channel on the full 'close' array.
    Returns dict: slope, intercept, upper_band, mid_line, lower_band at last bar,
    also projected mid-line at bar n (next bar outside the window).
    """
    arr = np.asarray(close, dtype=np.float64)
    n   = len(arr)
    t   = np.arange(n, dtype=np.float64)
    p   = np.polyfit(t, arr, 1)
    fit = np.poly1d(p)(t)
    res = arr - fit
    sig = float(np.std(res))

    last_fit    = float(np.poly1d(p)(n - 1))
    next_fit    = float(np.poly1d(p)(n))
    upper_last  = last_fit + n_sigma * sig
    lower_last  = last_fit - n_sigma * sig
    upper_next  = next_fit + n_sigma * sig
    lower_next  = next_fit - n_sigma * sig

    # current bar position inside channel
    curr      = float(arr[-1])
    band_rng  = (upper_last - lower_last) + 1e-20
    position  = (curr - lower_last) / band_rng   # 0=lower, 1=upper, <0=below, >1=above

    return {
        'slope':        round(float(p[0]), 10),
        'intercept':    round(float(p[1]), 8),
        'upper_band':   round(upper_last,  8),
        'mid_line':     round(last_fit,    8),
        'lower_band':   round(lower_last,  8),
        'next_upper':   round(upper_next,  8),
        'next_mid':     round(next_fit,    8),
        'next_lower':   round(lower_next,  8),
        'channel_pos':  round(position,    4),
        'residual_std': round(sig,         8),
    }


# ── ML training + backtesting + forecast ─────────────────

def _define_models():
    """Return dict of {name: sklearn_estimator} for the ensemble."""
    kernel = Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1.0)
    return {
        'Ridge':     Ridge(alpha=1.0),
        'Lasso':     Lasso(alpha=0.001, max_iter=5000),
        'BayesRidge':BayesianRidge(),
        'PolyReg3':  Pipeline([('poly', PolynomialFeatures(degree=3, include_bias=False)),
                               ('scaler', StandardScaler()),
                               ('ridge', Ridge(alpha=10.0))]),
        'RandForest':RandomForestRegressor(n_estimators=300, max_depth=8,
                                           min_samples_leaf=5, n_jobs=-1,
                                           random_state=42, oob_score=True),
        'GradBoost': GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                               learning_rate=0.05, subsample=0.8,
                                               random_state=42),
        'SVR_rbf':   Pipeline([('scaler', StandardScaler()),
                               ('svr',   SVR(kernel='rbf', C=100, epsilon=0.001,
                                             gamma='scale'))]),
        'GaussProc': GaussianProcessRegressor(kernel=kernel,
                                              n_restarts_optimizer=3,
                                              normalize_y=True,
                                              random_state=42),
    }


def _build_Xy(X, close, horizon=1):
    """
    Supervised: X[i] → y[i+horizon] (predict 'horizon' bars ahead).
    Returns (X_ml, y_ml) trimmed to aligned length.
    """
    n   = len(close)
    y_f = np.asarray(close, dtype=np.float64)
    X_s = X[: n - horizon]
    y_s = y_f[horizon:]
    return X_s, y_s


def train_and_backtest(X, close, horizon=ML_HORIZON, test_ratio=ML_TEST_RATIO):
    """
    Walk-forward backtest:
      - Split: first (1 - test_ratio) rows = train, rest = test
      - Train each model on train
      - Predict on test
      - Compute MAE, RMSE, dir_acc (directional accuracy)
      - Use test-set MAE to weight ensemble

    Returns (models_fitted, backtest_metrics, X_train, X_test, y_train, y_test).
    """
    X_ml, y_ml = _build_Xy(X, close, horizon=horizon)
    n          = len(X_ml)
    split      = max(20, int(n * (1.0 - test_ratio)))

    X_train, X_test = X_ml[:split], X_ml[split:]
    y_train, y_test = y_ml[:split], y_ml[split:]

    if len(X_train) < 15 or len(X_test) < 5:
        return None, None, X_train, X_test, y_train, y_test

    models   = _define_models()
    fitted   = {}
    metrics  = {}

    for name, mdl in models.items():
        try:
            mdl.fit(X_train, y_train)
            preds = mdl.predict(X_test)

            mae   = float(mean_absolute_error(y_test, preds))
            rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
            r2    = float(r2_score(y_test, preds))

            # directional accuracy: % of test bars where direction is correct
            actual_dir = np.sign(np.diff(y_test))
            pred_dir   = np.sign(np.diff(preds))
            if len(actual_dir) > 0:
                dir_acc = float(np.mean(actual_dir == pred_dir))
            else:
                dir_acc = 0.0

            fitted[name]  = mdl
            metrics[name] = {
                'mae':     round(mae,     8),
                'rmse':    round(rmse,    8),
                'r2':      round(r2,      4),
                'dir_acc': round(dir_acc, 4),
            }
        except Exception as ex:
            metrics[name] = {'mae': 1e9, 'rmse': 1e9, 'r2': -999.0, 'dir_acc': 0.0}

    return fitted, metrics, X_train, X_test, y_train, y_test


def ensemble_forecast(fitted_models, metrics, X_last):
    """
    Point forecast for the NEXT bar (using the last available feature row).
    Weight = 1 / MAE (lower MAE → higher weight).
    Returns (weighted_ensemble, individual_forecasts_dict).
    """
    forecasts = {}
    weights   = {}

    for name, mdl in fitted_models.items():
        try:
            pred = float(mdl.predict(X_last.reshape(1, -1))[0])
            mae  = metrics[name]['mae']
            forecasts[name] = pred
            weights[name]   = 1.0 / (mae + 1e-20)
        except Exception:
            pass

    if not forecasts:
        return None, {}

    total_w   = sum(weights.values())
    ensemble  = sum(forecasts[n] * weights[n] for n in forecasts) / (total_w + 1e-20)
    return float(ensemble), forecasts


# ── main ML entry point ───────────────────────────────────

def ml_compound_forecast(pair, current_price, sel_detail,
                          stf_results, htf_results, thr_map=None):
    """
    Full ML compound analysis.
    Fetches 1m, 3m, 5m klines independently for feature building.
    Uses 1m as the primary series for ML training and projection.
    Returns comprehensive result dict or None on failure.
    """
    if not ML_AVAILABLE:
        return None

    # ── fetch all TF data for feature enrichment ─────────────
    tf_data = {}
    for label, interval in [('1m', '1m'), ('3m', '3m'), ('5m', '5m')]:
        try:
            klines = trader.client.get_klines(
                symbol=pair, interval=interval, limit=ML_LOOKBACK + 50
            )
        except Exception:
            continue
        if len(klines) < 50:
            continue
        tf_data[label] = {
            'close':  np.array([float(k[4]) for k in klines], dtype=np.float64),
            'volume': np.array([float(k[5]) for k in klines], dtype=np.float64),
            'high':   np.array([float(k[2]) for k in klines], dtype=np.float64),
            'low':    np.array([float(k[3]) for k in klines], dtype=np.float64),
            'open':   np.array([float(k[1]) for k in klines], dtype=np.float64),
        }

    if '1m' not in tf_data:
        return None

    # primary 1m arrays
    close  = tf_data['1m']['close']
    volume = tf_data['1m']['volume']
    high   = tf_data['1m']['high']
    low    = tf_data['1m']['low']
    open_  = tf_data['1m']['open']
    n      = len(close)

    # ── gather scalars from existing analyses ─────────────────
    d         = sel_detail.get(pair, {})
    geo_d     = d.get('geometry_detail', {})
    phi_devs  = geo_d.get('phi_devs', 0.0)
    e_alpha   = geo_d.get('e_decay_rate', 0.0)
    swing_low  = d.get('swing_low')
    swing_high = d.get('swing_high')

    # FFT dominant period from STF results
    fft_period = 20
    if stf_results:
        fft_period = stf_results[0].get('dominant_period', 20)

    # HT period from 1m (via talib)
    ht_pe_arr  = ta.HT_DCPERIOD(close)
    valid_pe   = ~np.isnan(ht_pe_arr)
    ht_period  = float(ht_pe_arr[valid_pe][-1]) if np.any(valid_pe) else 20.0

    # ── regression channel ────────────────────────────────────
    reg_ch = regression_channel(close[-ML_LOOKBACK:])

    # ── volume S/R ────────────────────────────────────────────
    # Anchored to true argmin/argmax extrema of the 500-bar window.
    # Support  = peak BULLISH volume in the lowest-low zone.
    # Resistance = peak BEARISH volume in the highest-high zone.
    sup_levels, res_levels, vol_profile = compute_volume_sr(
        close, volume, high, low, open_,
        current_price, pair=pair, lookback=100
    )

    # ── argmin / argmax on 1m, 3m, 5m ────────────────────────
    # argmin = index of the true lowest LOW wick  (not lowest close)
    # argmax = index of the true highest HIGH wick (not highest close)
    # min/max/mid prices use the actual wick extremes across all bars.
    argmin_data = {}
    for label in ('1m', '3m', '5m'):
        td = tf_data.get(label)
        if td is None:
            continue
        lo  = td['low']
        hi  = td['high']
        win = min(ML_LOOKBACK, len(lo))
        lo_w = lo[-win:]   # true low wicks — all 500 bars
        hi_w = hi[-win:]   # true high wicks — all 500 bars
        argmin_data[label] = {
            'argmin':    int(np.argmin(lo_w)),               # bar of deepest trough wick
            'argmax':    int(np.argmax(hi_w)),               # bar of highest peak wick
            'min_price': round(float(np.min(lo_w)), 8),      # true lowest  price (wick)
            'max_price': round(float(np.max(hi_w)), 8),      # true highest price (wick)
            'mid_price': round(float((np.min(lo_w) + np.max(hi_w)) / 2.0), 8),
            'n_bars':    win,
        }

    # ── build feature matrix (1m primary) ────────────────────
    X = build_feature_matrix(
        close, volume, high, low,
        geo_detail=geo_d, phi_devs=phi_devs,
        e_alpha=e_alpha, fft_period=fft_period,
        ht_period=ht_period,
        swing_low=swing_low, swing_high=swing_high,
    )

    # ── train + backtest ──────────────────────────────────────
    fitted, metrics, X_train, X_test, y_train, y_test = \
        train_and_backtest(X, close, horizon=ML_HORIZON, test_ratio=ML_TEST_RATIO)

    if fitted is None:
        return None

    # ── ensemble point forecast (X at last bar) ───────────────
    X_last         = X[-1]
    ens_forecast, indiv = ensemble_forecast(fitted, metrics, X_last)

    # ── random walk Monte Carlo ───────────────────────────────
    mc = random_walk_mc(close, horizon=ML_HORIZON, n_paths=ML_WALKS)

    # ── Fibonacci targets from argmin thresholds ──────────────
    # Use 1m swing for primary Fib levels
    sl  = swing_low  if swing_low  else float(np.min(low[-ML_LOOKBACK:]))
    sh  = swing_high if swing_high else float(np.max(high[-ML_LOOKBACK:]))
    phi_targets = phi_extension_levels(sl, sh, current_price)

    # ── collect ALL forecast sources with tier labels ─────────
    #
    # TIER DESIGN (revised):
    #
    #   Every TF has its own argmax area: the price where the highest HIGH
    #   occurred in the 1000-bar window.  That IS the recovery target for
    #   that TF — price was there recently, buyers want to get back there.
    #   argmax targets are the PRIMARY anchors; everything else is secondary.
    #
    #   STF  — 1m/3m/5m argmax prices + FFT STF + MC short (≥30min horizon)
    #   MTF  — 15m/30m argmax prices  + FFT MTF  + MC medium (~2h horizon)
    #   HTF  — 1h/2h argmax prices    + FFT HTF  + MC long   + Fib ext + RegCh upper
    #
    #   ML model point forecasts:
    #     Only included if dir_acc >= MIN_DIRAC (45%).
    #     Models below this are predicting direction wrong more than half
    #     the time — their price target is meaningless noise.
    #     Weight = dir_acc² / MAE  (accuracy-squared penalises weak models)
    #
    #   Fibonacci levels: classified by distance into tiers.
    #
    #   tier_stf  = weighted avg of STF sources
    #   tier_mtf  = weighted avg of MTF sources
    #   tier_htf  = weighted avg of HTF sources  ← deepest reliable target

    MIN_DIRAC = 0.45   # minimum directional accuracy for ML models to contribute

    stf_fc  = []   # (label, price, weight)
    mtf_fc  = []
    htf_fc  = []

    stf_tf_labels = {'1m', '3m', '5m'}
    mtf_tf_labels = {'15m', '30m'}
    htf_tf_labels = {'1h', '2h', '4h'}

    # ── PRIMARY: argmax recovery targets per TF ───────────────
    # For each TF fetch the max_price from argmin_data (already computed).
    # Weight = inverse of how old the argmax is (recency) × range magnitude.
    # A recent high = strong magnet; an old high = weaker but still valid.
    for tf_label, tf_dict in (argmin_data or {}).items():
        mp   = tf_dict.get('max_price')
        amin = tf_dict.get('argmin', 0)
        amax = tf_dict.get('argmax', 0)
        nb   = tf_dict.get('n_bars', 1)
        if mp is None or mp <= current_price:
            continue
        # recency weight: argmax closer to recent bars = higher weight
        # amax=0 means the high was at bar 0 (oldest) → lower weight
        # amax=nb-1 means the high was very recent → higher weight (but on dip
        # scan argmax is always OLD — still valid as recovery target)
        recency = max(1, amax) / nb   # 0→1, higher = more recent
        range_w = (mp - current_price) / current_price  # bigger move = higher anchor weight
        w       = (recency + 0.3) * range_w * 10.0     # +0.3 floor so old highs still count

        lbl = f'ArgMax-{tf_label}'
        if   tf_label in stf_tf_labels: stf_fc.append((lbl, mp, w))
        elif tf_label in mtf_tf_labels: mtf_fc.append((lbl, mp, w))
        else:                            htf_fc.append((lbl, mp, w))

    # ── ML models: only above dir_acc threshold ───────────────
    for nm, fc in (indiv or {}).items():
        if fc <= current_price:
            continue
        m_info = metrics.get(nm, {})
        da     = m_info.get('dir_acc', 0.0)
        mae    = m_info.get('mae', 1.0) + 1e-20
        if da < MIN_DIRAC:
            continue                          # too inaccurate directionally
        w = (da ** 2) / mae                   # accuracy² / MAE
        stf_fc.append((nm, fc, w))

    # ML ensemble (only if at least one model passes dir_acc gate)
    passing_models = [nm for nm, m in (metrics or {}).items()
                      if m.get('dir_acc', 0.0) >= MIN_DIRAC]
    if ens_forecast and ens_forecast > current_price and passing_models:
        best_da  = max(metrics[nm]['dir_acc'] for nm in passing_models)
        best_mae = min(metrics[nm]['mae']     for nm in passing_models) + 1e-20
        w        = (best_da ** 2) / best_mae * 2.0
        stf_fc.append(('ML Ensemble', ens_forecast, w))

    # ── MC short → STF, medium → MTF, long → HTF ─────────────
    if mc:
        p75_s = mc['short']['p75'];  p95_s = mc['short']['p95']
        p50_m = mc['medium']['p50']; p75_m = mc['medium']['p75']
        p75_l = mc['long']['p75'];   p95_l = mc['long']['p95']
        vol_w = 1.0 / (mc['vol_per_bar'] + 1e-20)
        if p75_s > current_price:  stf_fc.append(('MC-short-p75', p75_s, vol_w))
        if p95_s > current_price:  stf_fc.append(('MC-short-p95', p95_s, vol_w * 0.5))
        if p50_m > current_price:  mtf_fc.append(('MC-med-p50',   p50_m, vol_w))
        if p75_m > current_price:  mtf_fc.append(('MC-med-p75',   p75_m, vol_w))
        if p75_l > current_price:  htf_fc.append(('MC-long-p75',  p75_l, vol_w))
        if p95_l > current_price:  htf_fc.append(('MC-long-p95',  p95_l, vol_w))

    # ── FFT results → tiers by TF ─────────────────────────────
    for r in (stf_results or []):
        fc = r.get('forecast', 0)
        if fc > current_price:
            w   = r.get('res_volume', 1.0) or 1.0
            lbl = f'FFT-{r["tf"]}'
            if r['tf'] in stf_tf_labels: stf_fc.append((lbl, fc, w))
            else:                         mtf_fc.append((lbl, fc, w))
    for r in (htf_results or []):
        fc = r.get('forecast', 0)
        if fc > current_price:
            w   = r.get('res_volume', 1.0) or 1.0
            lbl = f'FFT-{r["tf"]}'
            if r['tf'] in mtf_tf_labels: mtf_fc.append((lbl, fc, w))
            else:                         htf_fc.append((lbl, fc, w))

    # ── Fibonacci φ targets → tiers by distance ───────────────
    for ratio, lbl_t, level in (phi_targets or []):
        if level <= current_price:
            continue
        dist_pct = (level - current_price) / current_price * 100.0
        w = PHI if (abs(ratio - PHI_INV) < 0.01 or abs(ratio - PHI) < 0.01) else 1.0
        if   dist_pct <= 5.0:    stf_fc.append((f'Fib-{lbl_t[:8]}', level, w))
        elif dist_pct <= 15.0:   mtf_fc.append((f'Fib-{lbl_t[:8]}', level, w))
        else:                     htf_fc.append((f'Fib-{lbl_t[:8]}', level, w))

    # ── Regression channel: upper band → HTF ─────────────────
    if reg_ch:
        ub = reg_ch['next_upper']
        if ub > current_price:
            htf_fc.append(('RegCh-Upper', ub, 1.0))

    def weighted_avg(lst):
        """Weighted average of (label, price, weight) list. Returns None if empty."""
        if not lst:
            return None
        prices  = np.array([x[1] for x in lst], dtype=np.float64)
        weights = np.array([x[2] for x in lst], dtype=np.float64)
        weights = np.maximum(weights, 1e-20)
        return float(np.average(prices, weights=weights))

    tier_stf = weighted_avg(stf_fc)
    tier_mtf = weighted_avg(mtf_fc)
    tier_htf = weighted_avg(htf_fc)

    # all_forecasts list for the synthesis print (sorted ascending)
    all_forecasts = sorted(
        [(lbl, fc) for lbl, fc, _ in (stf_fc + mtf_fc + htf_fc)],
        key=lambda x: x[1]
    )

    hard_cap       = res_levels[0][0]   if res_levels else None
    hard_cap_floor = res_levels[0][3]   if res_levels else None   # sr_floor of resistance
    hard_cap_ceil  = res_levels[0][4]   if res_levels else None   # sr_ceiling of resistance

    # best_target = HTF tier (deepest reliable target), fall back to MTF then STF
    best_target = tier_htf or tier_mtf or tier_stf
    if best_target and hard_cap and best_target > hard_cap * 1.05:
        best_target = hard_cap   # only cap if meaningfully above wall

    # ── nearest support as stop reference ────────────────────────
    soft_stop       = sup_levels[0][0] if sup_levels else None
    soft_stop_floor = sup_levels[0][3] if sup_levels else None   # sr_floor (hard floor)
    soft_stop_ceil  = sup_levels[0][4] if sup_levels else None   # sr_ceiling (top of support zone)

    # mid threshold from 1m argmin data
    mid_1m = argmin_data.get('1m', {}).get('mid_price')

    return {
        'pair':            pair,
        'current_price':   current_price,
        'reg_channel':     reg_ch,
        'sup_levels':      sup_levels,
        'res_levels':      res_levels,
        'vol_profile':     vol_profile,
        'argmin_data':     argmin_data,
        'fitted_models':   fitted,
        'metrics':         metrics,
        'ens_forecast':    ens_forecast,
        'indiv_forecasts': indiv,
        'mc':              mc,
        'phi_targets':     phi_targets,
        'all_forecasts':   all_forecasts,
        'stf_fc':          stf_fc,
        'mtf_fc':          mtf_fc,
        'htf_fc':          htf_fc,
        'tier_stf':        round(tier_stf, 8) if tier_stf else None,
        'tier_mtf':        round(tier_mtf, 8) if tier_mtf else None,
        'tier_htf':        round(tier_htf, 8) if tier_htf else None,
        'best_target':     round(best_target,    8) if best_target    else None,
        'hard_cap':        round(hard_cap,       8) if hard_cap       else None,
        'hard_cap_floor':  round(hard_cap_floor, 8) if hard_cap_floor else None,
        'hard_cap_ceil':   round(hard_cap_ceil,  8) if hard_cap_ceil  else None,
        'soft_stop':       round(soft_stop,      8) if soft_stop      else None,
        'soft_stop_floor': round(soft_stop_floor,8) if soft_stop_floor else None,
        'soft_stop_ceil':  round(soft_stop_ceil, 8) if soft_stop_ceil  else None,
        'mid_threshold':   mid_1m,
        'ml_horizon_bars': ML_HORIZON,
    }


# ── ML report printer ─────────────────────────────────────

def print_ml_report(ml_result, label_map):
    """Print the full ML compound forecast report."""
    if ml_result is None:
        print('  ML: analysis unavailable.\n')
        return

    pair  = ml_result['pair']
    lbl   = label_map.get(pair, pair.replace('USDC', ''))
    cp    = ml_result['current_price']
    w     = 66

    def pf(v, dp=8):
        """format price"""
        if v is None: return '—'
        return f'{v:.6f}' if v < 1 else f'{v:.4f}'

    def pp(v, base=None):
        """format price + pct change vs base"""
        if v is None: return '—'
        s = pf(v)
        if base:
            pct = (v - base) / base * 100.0
            s  += f'  ({pct:+.2f}%)'
        return s

    print(f'\n  {"═"*w}')
    print(f'  ◈  ML COMPOUND FORECAST  ·  {lbl}  ({pair})')
    print(f'  {"═"*w}')
    print(f'  Entry price   : {pf(cp)}'
          f'   |  Horizon: {ml_result["ml_horizon_bars"]} bars (1m)')
    print()

    # ── 1. Regression Channel ─────────────────────────────────
    rc = ml_result.get('reg_channel') or {}
    print(f'  ┌─ 1. LINEAR REGRESSION CHANNEL  (last {ML_LOOKBACK} bars) {"─"*10}┐')
    if rc:
        cp_label = 'above mid' if cp > rc.get('mid_line', cp) else \
                   ('below mid' if cp < rc.get('mid_line', cp) else 'at mid')
        band_pos = rc.get('channel_pos', 0.0)
        pos_lbl  = ('ABOVE UPPER — extended'  if band_pos > 1.0 else
                    'BELOW LOWER — compressed ← dip zone' if band_pos < 0.0 else
                    f'pos={band_pos:.2f} (0=lower, 1=upper)')
        print(f'  │  Slope            : {rc["slope"]:.10f}  (per bar)')
        print(f'  │  Upper band (±2σ) : {pf(rc["upper_band"])}')
        print(f'  │  Mid line         : {pf(rc["mid_line"])}   ← {cp_label}')
        print(f'  │  Lower band (±2σ) : {pf(rc["lower_band"])}')
        print(f'  │  Residual σ       : {rc["residual_std"]}')
        print(f'  │  Channel position : {pos_lbl}')
        print(f'  │  ── Next-bar projection ──────────────────────────────────')
        print(f'  │  Next upper       : {pp(rc["next_upper"], cp)}')
        print(f'  │  Next mid  ← ML  : {pp(rc["next_mid"],   cp)}')
        print(f'  │  Next lower       : {pp(rc["next_lower"], cp)}')
    print(f'  └{"─"*w}┘')
    print()

    # ── 2. Multi-TF Argmin/Argmax thresholds ─────────────────
    am = ml_result.get('argmin_data', {})
    print(f'  ┌─ 2. ARGMIN / ARGMAX PRICE THRESHOLDS  (last {ML_LOOKBACK} bars) {"─"*4}┐')
    hdr2 = f'  │  {"TF":>3}  {"ArgMin":>7}  {"ArgMax":>7}  {"Min Price":>14}  {"Mid Price":>14}  {"Max Price":>14}  {"dist":>6}  │'
    sep2 = '  │' + '─' * (len(hdr2) - 4) + '│'
    print(sep2); print(hdr2); print(sep2)
    for tf_lbl in ('1m', '3m', '5m'):
        t = am.get(tf_lbl, {})
        if not t:
            continue
        amin = t['argmin']; amax = t['argmax']
        dist = amin - amax   # positive = argmin more recent than argmax
        tick = '✔ ' if dist > 0 else '✗ '
        print(f'  │  {tf_lbl:>3}  {amin:>7}  {amax:>7}  '
              f'{pf(t["min_price"]):>14}  '
              f'{pf(t["mid_price"]):>14}  '
              f'{pf(t["max_price"]):>14}  '
              f'{tick}{dist:>4}  │')
    # current price vs thresholds
    if am.get('1m'):
        mn = am['1m']['min_price']; mx = am['1m']['max_price']
        mid = am['1m']['mid_price']
        pct_from_min = (cp - mn) / (mx - mn + 1e-20) * 100.0
        print(sep2)
        print(f'  │  Current {pf(cp)} is {pct_from_min:.1f}% from min to max'
              f'   (mid={pf(mid)})  │')
    print(f'  └{"─"*w}┘')
    print()

    # ── 3. Volume & Market Profile + S/R ─────────────────────
    sups = ml_result.get('sup_levels', [])
    ress = ml_result.get('res_levels', [])
    vp   = ml_result.get('vol_profile', {})
    lb   = vp.get('lookback', 100)

    # ── ASCII dual-profile chart ──────────────────────────────
    def _ascii_vp(vol_bins_list, mkt_bins_list, poc, vah, val,
                  mpoc, mvah, mval, entry, sups_p, ress_p, width=18):
        if not vol_bins_list:
            return []
        prices  = sorted(set([b[0] for b in vol_bins_list] +
                              [b[0] for b in mkt_bins_list]), reverse=True)
        vd = dict(vol_bins_list); md = dict(mkt_bins_list)
        max_v  = max(vd.values()) if vd else 1.0
        max_m  = max(md.values()) if md else 1.0
        pr_rng = abs(prices[0] - prices[-1]) if len(prices) > 1 else 1.0
        tol    = pr_rng * 0.025
        step   = max(1, len(prices) // 28)
        lines  = []
        for pr in prices[::step]:
            v_bar = int(vd.get(pr, 0.0) / max_v * width)
            m_bar = int(md.get(pr, 0.0) / max_m * width)
            mk = []
            if poc  is not None and abs(pr-poc)  < tol: mk.append('P')
            if mpoc is not None and abs(pr-mpoc) < tol: mk.append('p')
            if entry is not None and abs(pr-entry)< tol: mk.append('E')
            if vah  is not None and abs(pr-vah)  < tol: mk.append('▲')
            if val  is not None and abs(pr-val)  < tol: mk.append('▼')
            if any(abs(pr-sp) < tol for sp in sups_p): mk.append('S')
            if any(abs(pr-rp) < tol for rp in ress_p): mk.append('R')
            mk_s  = ''.join(mk)[:3].ljust(3)
            v_str = '█' * v_bar + '░' * (width - v_bar)
            m_str = '█' * m_bar + '░' * (width - m_bar)
            lines.append(f'  │  {pf(pr):>12} {mk_s} {v_str} {m_str}  │')
        return lines

    # ── S/R row formatter (uses new tuple fields) ─────────────
    def _sr_row(item, side):
        price  = item[0];  raw_v = item[1];  n_b   = item[2]
        sr_fl  = item[3];  sr_ce = item[4];  rng_p = item[5]
        zone   = item[6] if len(item) > 6 else '—'
        dist   = item[7] if len(item) > 7 else (price - cp) / (cp+1e-20) * 100.0
        vpct   = item[8] if len(item) > 8 else 0.0
        comp   = item[9] if len(item) > 9 else 0.0
        grow   = item[10] if len(item) > 10 else 0.0
        cons   = item[11] if len(item) > 11 else 0.0
        sign   = 'above' if dist > 0 else 'below'
        grow_s = f'▲{grow:+.1%}' if grow > 0 else f'▼{grow:+.1%}'
        tag    = 'bull_vol' if side == 'sup' else 'bear_vol'
        return (
            f'  │    {pf(price)}  {tag}={raw_v:>14.0f}  bars={n_b:>4}'
            f'  dist={abs(dist):.2f}%{sign}  {vpct:.1f}%side'
            f'  score={comp:.0f}  growth={grow_s}  consist={cons:.0%}'
            f'  zone={zone}'
            f'  [{pf(sr_fl)}–{pf(sr_ce)} ±{rng_p}%]'
        )

    # ── computed scalars ──────────────────────────────────────
    best_sup_p   = sups[0][0] if sups else None
    best_res_p   = ress[0][0] if ress else None
    sr_range_pct = round((best_res_p - best_sup_p) / (cp+1e-20) * 100.0, 2) \
                   if (best_sup_p and best_res_p) else None
    sups_prices  = [s[0] for s in sups]
    ress_prices  = [r[0] for r in ress]

    bull_pct_vp = vp.get('bull_pct', 0.0)
    bear_pct_vp = vp.get('bear_pct', 0.0)
    predom      = vp.get('predominance', '—')
    bias_len    = 30
    bull_bar    = int(bull_pct_vp / 100.0 * bias_len)
    bear_bar    = bias_len - bull_bar
    bias_str    = '▲' * bull_bar + '▼' * bear_bar
    bias_icon   = '🟢' if predom == 'BULLISH' else ('🔴' if predom == 'BEARISH' else '⚪')

    # orderbook walls
    ob_bid = vp.get('ob_bid_wall')
    ob_ask = vp.get('ob_ask_wall')

    print(f'  ┌─ 3. VOLUME & MARKET PROFILE + REAL S/R  (last {lb} 1m bars) {"─"*10}┐')
    print(f'  │  Bull/bear vol: ALL bars scanned, NO side filter, scored by:           │')
    print(f'  │    raw vol × (1+consistency) × max(1, 1+growth)  → composite score    │')
    print(f'  │  Markers: P=vol POC  p=mkt POC  E=entry  ▲/▼=VAH/VAL  S=sup  R=res   │')
    print(f'  │  {"─"*62}│')

    # bias
    print(f'  │  1m BIAS  {bias_icon} {predom:<8}'
          f'  bull={bull_pct_vp:.1f}%  bear={bear_pct_vp:.1f}%  [{bias_str}]  │')
    print(f'  │  Total={vp.get("total_vol",0):.0f}'
          f'  bull={vp.get("bull_vol",0):.0f}'
          f'  bear={vp.get("bear_vol",0):.0f}                              │')

    # orderbook walls
    if ob_bid or ob_ask:
        bid_s = f'bid wall @ {pf(ob_bid[0])}  qty={ob_bid[1]:.2f}' if ob_bid else 'no bid wall'
        ask_s = f'ask wall @ {pf(ob_ask[0])}  qty={ob_ask[1]:.2f}' if ob_ask else 'no ask wall'
        print(f'  │  ORDERBOOK: {bid_s}  │  {ask_s}  │')

    print(f'  │  {"─"*62}│')
    print(f'  │  Vol  POC={pf(vp.get("vol_poc"))}  VAH={pf(vp.get("vol_vah"))}  VAL={pf(vp.get("vol_val"))}    │')
    print(f'  │  Mkt  POC={pf(vp.get("mkt_poc"))}  VAH={pf(vp.get("mkt_vah"))}  VAL={pf(vp.get("mkt_val"))}    │')
    print(f'  │  Entry={pf(cp)}  argmin={pf(vp.get("absolute_min"))}  argmax={pf(vp.get("absolute_max"))}  │')
    print(f'  │  {"─"*62}│')
    print(f'  │  {"Price":>12} Mk  {"─VOL PROFILE─":^18} {"─MKT PROFILE─":^18}  │')
    print(f'  │  {"─"*62}│')

    for ln in _ascii_vp(
        vp.get('vol_bins', []), vp.get('mkt_bins', []),
        vp.get('vol_poc'), vp.get('vol_vah'), vp.get('vol_val'),
        vp.get('mkt_poc'), vp.get('mkt_vah'), vp.get('mkt_val'),
        cp, sups_prices, ress_prices
    ):
        print(ln)

    print(f'  │  {"─"*62}│')
    print(f'  │  SUPPORT LEVELS  (most significant bullish vol — anywhere, ranked)     │')
    print(f'  │  score = raw_vol × (1+consistency) × max(1, 1+growth)                 │')
    if sups:
        for item in sups[:3]:
            print(_sr_row(item, 'sup'))
    else:
        print(f'  │    (none found in last {lb} bars)')

    print(f'  │  {"─"*62}│')
    print(f'  │  RESISTANCE LEVELS  (most significant bearish vol — anywhere, ranked)  │')
    if ress:
        for item in ress[:3]:
            print(_sr_row(item, 'res'))
    else:
        print(f'  │    (none found in last {lb} bars)')

    print(f'  │  {"─"*62}│')
    if sr_range_pct is not None:
        print(f'  │  S→R spread: {pf(best_sup_p)} → {pf(best_res_p)}'
              f'  = {sr_range_pct:+.2f}% of entry price               │')
    if ob_bid and ob_ask:
        ob_spread = round((ob_ask[0] - ob_bid[0]) / (cp+1e-20) * 100.0, 3)
        print(f'  │  OB wall spread: {pf(ob_bid[0])} bid → {pf(ob_ask[0])} ask'
              f'  = {ob_spread:+.3f}%                    │')
    print(f'  └{"─"*w}┘')
    print()


    # ── 4. Backtest metrics ───────────────────────────────────
    mets = ml_result.get('metrics', {})
    print(f'  ┌─ 4. INSTANT BACKTEST  (80% train / 20% test, horizon={ML_HORIZON}b) {"─"*2}┐')
    hdr4 = f'  │  {"Model":<12}  {"MAE":>12}  {"RMSE":>12}  {"R²":>7}  {"DirAcc":>8}  │'
    sep4 = '  │' + '─' * (len(hdr4) - 4) + '│'
    print(sep4); print(hdr4); print(sep4)
    # sort by MAE ascending
    sorted_mets = sorted(mets.items(), key=lambda x: x[1]['mae'])
    for nm, m in sorted_mets:
        if m['mae'] > 1e8: continue
        bar_da  = '█' * int(m['dir_acc'] * 10)
        print(f'  │  {nm:<12}  {m["mae"]:>12.8f}  {m["rmse"]:>12.8f}  '
              f'{m["r2"]:>7.4f}  {m["dir_acc"]:>7.1%}  │')
    print(sep4)
    # best model
    best_name = sorted_mets[0][0] if sorted_mets else '—'
    best_dacc  = sorted_mets[0][1]['dir_acc'] if sorted_mets else 0.0
    print(f'  │  Best model: {best_name:<10}   Directional accuracy: {best_dacc:.1%}       │')
    print(f'  └{"─"*w}┘')
    print()

    # ── 5. Individual model forecasts ─────────────────────────
    indiv = ml_result.get('indiv_forecasts', {}) or {}
    print(f'  ┌─ 5. INDIVIDUAL MODEL POINT FORECASTS  (+{ML_HORIZON} bars ahead) {"─"*4}┐')
    hdr5 = f'  │  {"Model":<12}  {"Forecast":>14}  {"Δ from entry":>14}  {"Δ%":>8}  │'
    sep5 = '  │' + '─' * (len(hdr5) - 4) + '│'
    print(sep5); print(hdr5); print(sep5)
    # sort by forecast ascending
    sorted_indiv = sorted(indiv.items(), key=lambda x: x[1])
    for nm, fc in sorted_indiv:
        delta   = fc - cp
        delta_p = delta / cp * 100.0
        tag     = ' ▲' if delta > 0 else ' ▼'
        print(f'  │  {nm:<12}  {pf(fc):>14}  {delta:>+14.8f}  {delta_p:>+7.2f}%{tag}  │')
    # ensemble
    ens = ml_result.get('ens_forecast')
    if ens:
        delta   = ens - cp
        delta_p = delta / cp * 100.0
        print(sep5)
        print(f'  │  {"★ Ensemble":<12}  {pf(ens):>14}  {delta:>+14.8f}  {delta_p:>+7.2f}%  │')
    print(f'  └{"─"*w}┘')
    print()

    # ── 6. Random Walk Monte Carlo ────────────────────────────
    mc = ml_result.get('mc') or {}
    print(f'  ┌─ 6. RANDOM WALK MONTE CARLO  '
          f'({ML_WALKS:,} paths · 3 horizons) {"─"*14}┐')
    if mc:
        prob_up = mc.get('prob_up', 0.0)
        bar_mc  = '█' * int(prob_up * 20) + '░' * (20 - int(prob_up * 20))
        print(f'  │  Drift applied    : {mc.get("drift_applied",0):+.8f}/bar'
              f'  (raw={mc.get("raw_drift",0):+.8f}  vol={mc.get("vol_per_bar",0):.8f})')
        print(f'  │  P(short UP)      : {prob_up:.1%}  [{bar_mc}]')
        print(f'  │')
        hdr_mc = f'  │  {"Horizon":<18}  {"p5":>12}  {"p25":>12}  {"p50 (median)":>14}  {"p75":>12}  {"p95":>12}  │'
        sep_mc = '  │' + '─' * (len(hdr_mc) - 4) + '│'
        print(sep_mc)
        print(hdr_mc)
        print(sep_mc)
        for tier_lbl, key in [
            (f'Short  ({mc.get("horizon_s","?")}b ≈{mc.get("horizon_s",0)}m)', 'short'),
            (f'Medium ({mc.get("horizon_m","?")}b ≈{mc.get("horizon_m",0)//60}h)', 'medium'),
            (f'Long   ({mc.get("horizon_l","?")}b ≈{mc.get("horizon_l",0)//60}h)', 'long'),
        ]:
            t = mc.get(key) or {}
            if not t:
                continue
            pu = t.get('prob_up', 0.0)
            print(f'  │  {tier_lbl:<18}  '
                  f'{pp(t.get("p5"),  cp):>12}  '
                  f'{pp(t.get("p25"), cp):>12}  '
                  f'{pp(t.get("p50"), cp):>14}  '
                  f'{pp(t.get("p75"), cp):>12}  '
                  f'{pp(t.get("p95"), cp):>12}  │')
        print(sep_mc)
    else:
        print(f'  │  (Monte Carlo unavailable)')
    print(f'  └{"─"*w}┘')
    print()

    # ── 7. Fibonacci φ targets ────────────────────────────────
    phi_t = ml_result.get('phi_targets', [])
    print(f'  ┌─ 7. φ FIBONACCI TARGETS  (from {ML_LOOKBACK}-bar swing) {"─"*18}┐')
    if phi_t:
        for ratio, lbl_txt, level in phi_t[:6]:
            dist_p = (level - cp) / cp * 100.0
            mark   = ' ← KEY φ' if abs(ratio - PHI_INV) < 0.01 or abs(ratio - PHI) < 0.01 else ''
            print(f'  │  {lbl_txt:<28}  {pf(level)}  ({dist_p:+.2f}%){mark}')
    else:
        print(f'  │  (insufficient swing data)')
    print(f'  └{"─"*w}┘')
    print()

    # ── 8. All forecast synthesis ─────────────────────────────
    all_fc = ml_result.get('all_forecasts', [])
    print(f'  ┌─ 8. FORECAST SYNTHESIS  (all sources combined) {"─"*18}┐')
    if all_fc:
        vals = np.array([v for _, v in all_fc])
        # show tier breakdown
        stf_items = [(l, v) for l, v, _ in ml_result.get('stf_fc', [])]
        mtf_items = [(l, v) for l, v, _ in ml_result.get('mtf_fc', [])]
        htf_items = [(l, v) for l, v, _ in ml_result.get('htf_fc', [])]
        passing_da = [nm for nm, m in (mets or {}).items()
                      if m.get('dir_acc', 0.0) >= 0.45]
        excl_da    = [nm for nm, m in (mets or {}).items()
                      if m.get('dir_acc', 0.0) < 0.45]
        print(f'  │  Total sources : {len(all_fc)}  '
              f'(STF={len(stf_items)}  MTF={len(mtf_items)}  HTF={len(htf_items)})')
        print(f'  │  ML dir_acc≥45%: {passing_da or ["none"]}')
        if excl_da:
            print(f'  │  ML excluded   : {excl_da}  (dir_acc < 45%)')
        print(f'  │  Range  :  {pf(float(vals.min()))}  →  {pf(float(vals.max()))}')
        print(f'  │  Median :  {pf(float(np.median(vals)))}')
        print(f'  │  Mean   :  {pf(float(np.mean(vals)))}')
        print()
        # group by tier
        for tier_lbl, items in [('── STF (1m/3m/5m argmax + FFT STF + MC short)', stf_items),
                                  ('── MTF (15m/30m argmax + FFT MTF + MC med)',    mtf_items),
                                  ('── HTF (1h/2h argmax + FFT HTF + Fib + MC long)', htf_items)]:
            if not items:
                continue
            print(f'  │  {tier_lbl}')
            for src, fc in sorted(items, key=lambda x: x[1]):
                d_p = (fc - cp) / cp * 100.0
                tag = ' ◄ ARGMAX' if src.startswith('ArgMax') else ''
                print(f'  │    {src:<22}  {pf(fc):>14}  ({d_p:>+6.2f}%){tag}')
            print(f'  │')
    print(f'  └{"─"*w}┘')
    print()

    # ── FINAL SUMMARY: ENTRY / EXIT / STOP ───────────────────
    bt     = ml_result.get('best_target')
    t_stf  = ml_result.get('tier_stf')
    t_mtf  = ml_result.get('tier_mtf')
    t_htf  = ml_result.get('tier_htf')
    hcap   = ml_result.get('hard_cap')
    hcap_f = ml_result.get('hard_cap_floor')
    hcap_c = ml_result.get('hard_cap_ceil')
    stop   = ml_result.get('soft_stop')
    stop_f = ml_result.get('soft_stop_floor')
    stop_c = ml_result.get('soft_stop_ceil')
    mid_t  = ml_result.get('mid_threshold')

    print(f'  {"═"*w}')
    print(f'  ★★★  ML COMPOUND FINAL DECISION  ·  {lbl}')
    print(f'  {"═"*w}')
    print(f'  ▶  ENTRY              : {pf(cp)}  (current 1m close)')
    print()

    if mid_t:
        dist_mid = (cp - mid_t) / cp * 100.0 if cp > mid_t else (mid_t - cp) / cp * 100.0
        side     = 'BELOW' if cp < mid_t else 'ABOVE'
        zone     = 'dip zone ← good entry' if cp < mid_t else 'elevated vs 500-bar range'
        print(f'  ▶  MID THRESHOLD      : {pf(mid_t)}'
              f'  ({side} by {dist_mid:.2f}%) [{zone}]')
    print()

    # tiered targets
    for tier_lbl, tv, desc in [
        ('STF TARGET  (~30m)',  t_stf, 'ML models + FFT 1m/3m/5m + MC short'),
        ('MTF TARGET  (~2h)',   t_mtf, 'FFT 15m/30m + MC medium + φ near Fib'),
        ('HTF TARGET  (swing)', t_htf, 'FFT 1h/2h + MC long p75/p95 + φ ext + RegCh upper'),
    ]:
        if tv:
            up_pct = (tv - cp) / cp * 100.0
            arrow  = '▲' if tv > cp else '▼'
            print(f'  ▶  {tier_lbl:<22}: {pf(tv)}'
                  f'  ({arrow}{up_pct:+.2f}%)  [{desc}]')
        else:
            print(f'  ▶  {tier_lbl:<22}: —  (insufficient data)')
    print()

    if hcap:
        cap_pct  = (hcap - cp) / cp * 100.0
        rng_str  = f'  SR=[{pf(hcap_f)}–{pf(hcap_c)}]' if hcap_f else ''
        print(f'  ▶  HARD CAP (vol wall): {pf(hcap)}'
              f'  (+{cap_pct:.2f}%)  ← nearest resistance{rng_str}')
    else:
        print(f'  ▶  HARD CAP           : —  (no vol resistance found)')
    print()

    if stop:
        stop_pct = (cp - stop) / cp * 100.0
        rng_str  = f'  SR=[{pf(stop_f)}–{pf(stop_c)}]' if stop_f else ''
        print(f'  ▶  SOFT STOP LOSS     : {pf(stop)}'
              f'  (-{stop_pct:.2f}% from entry)  [vol support floor{rng_str}]')
    else:
        print(f'  ▶  SOFT STOP LOSS     : —  (no volume support found below)')
    print()

    # risk/reward vs HTF target (the meaningful one)
    target_rr = t_htf or t_mtf or t_stf
    if target_rr and stop:
        reward = target_rr - cp
        risk   = cp        - stop
        rr     = reward / risk if risk > 0 else 0.0
        grade  = ('EXCELLENT ★★★' if rr >= 3.0 else
                  'GOOD ★★'       if rr >= 2.0 else
                  'OK ★'          if rr >= 1.5 else 'POOR')
        print(f'  ▶  RISK / REWARD      : {rr:.2f}  ({grade})')
        print(f'     Reward +{(reward/cp*100):.2f}%  |  Risk -{(risk/cp*100):.2f}%')
    print()

    # best model directional call
    if mets:
        best_nm  = sorted(mets.items(), key=lambda x: x[1]['mae'])[0][0]
        best_fc  = indiv.get(best_nm)
        if best_fc:
            direction = '▲ UPSIDE' if best_fc > cp else '▼ DOWNSIDE'
            print(f'  ▶  BEST MODEL         : {best_nm} → {direction}  '
                  f'target={pf(best_fc)}  '
                  f'(MAE={mets[best_nm]["mae"]:.8f}  dirAcc={mets[best_nm]["dir_acc"]:.1%})')
    # MC call (use medium horizon for more meaningful signal)
    if mc:
        mc_m    = mc.get('medium', {})
        prob_up = mc_m.get('prob_up', 0.0) if mc_m else mc.get('prob_up', 0.0)
        mc_call = '▲ BULLISH' if prob_up > 0.55 else \
                  ('▼ BEARISH' if prob_up < 0.45 else '→ NEUTRAL')
        p50_m   = mc_m.get('p50', 0) if mc_m else mc.get('p50', 0)
        print(f'  ▶  MONTE CARLO        : {mc_call}  '
              f'({prob_up:.1%} paths above entry at medium horizon)'
              f'  p50={pf(p50_m)}')
    print()
    print(f'  {"═"*w}\n')


# ═════════════════════════════════════════════════════════════
#  φ-REVERSAL FORECAST BLOCK
#
#  Fuses the three φ·e·π layers with argmin/argmax extrema
#  detection to produce a standalone reversal call for the
#  MTF winner:
#
#  Layer 1 — φ DECAY BANDS  (negative exponential powers)
#    PHI^-1 … PHI^-7  = [0.618, 0.382, 0.236, 0.146, 0.090, 0.056, 0.034]
#    and extensions PHI^+1=1.618, PHI^+2=2.618
#    Scored by how close current price sits to any band.
#
#  Layer 2 — GOLDEN TRIANGLE TARGETS
#    Isosceles apex=36°, base-angles=72°, leg/base = φ
#    apex height  = bar_range × sin(72°) / (2×sin(36°))
#    gnomon leg   = bar_range × φ²  (sub-triangle recursion)
#    T1 = current ± apex_height   (primary)
#    T2 = current ± gnomon        (extended, φ² move)
#
#  Layer 3 — GOLDEN SPIRAL TIMING
#    r(θ) = A·e^(b·θ)  where b = ln(φ)/(π/2) = GOLDEN_B
#    After each π/2 turn radius ×φ → discrete windows:
#    anchor + round(φ^k) for k=1…7 = [2,3,4,7,11,18,29] bars
#    spiral_ok = current bar within ±order bars of any window
#
#  EXTREMA DETECTION
#    argrelextrema(low_arr,  np.less,    order=N) → swing lows
#    argrelextrema(high_arr, np.greater, order=N) → swing highs
#    Direction: argmin_idx > argmax_idx → BUY setup
#               argmax_idx > argmin_idx → SELL setup
#
#  CONFIDENCE  (three-layer composite)
#    0.50 × φ_band_proximity  (how close price is to a φ level)
#    0.30 × spiral_timing     (is bar inside a spiral window?)
#    0.20 × extremum_score    (how close the detected extremum
#                               itself sits to a φ band)
#
#  OUTPUT
#    trend          : 'UP' | 'DOWN' | 'NEUTRAL'
#    forecast_price : median of all φ-aligned target levels
#    target_T1      : golden-triangle primary target
#    target_T2      : golden-triangle φ² gnomon target
#    phi_band_label : nearest φ band name
#    phi_band_level : exact φ level price
#    confidence     : composite 0–1
#    direction      : 'BUY' | 'SELL'
# ═════════════════════════════════════════════════════════════

PHI_NEG_POWERS  = np.array([PHI ** -n for n in range(1, 8)])

# ═════════════════════════════════════════════════════════════
#  SINUSOIDAL CIRCUIT ENGINE — 360° CYCLE QUADRANT SYSTEM
#
#  MATHEMATICS:
#    Price model:  P(t) = A · sin(2π/T · t + φ₀) + m·t + b
#    where T  = dominant FFT period (bars)
#          A  = amplitude = (swing_high − swing_low) / 2
#          φ₀ = phase offset (fitted to argmin/argmax positions)
#          m  = linear trend slope
#
#  360° CIRCUIT MAP (one full sinusoidal period):
#    0°   = argmin  → SUPPORT DIP    (sin = −1, trough)
#    90°  = rising zero crossing     (sin =  0, Q1→Q2 boundary)
#    180° = argmax  → RESISTANCE TOP (sin = +1, peak)
#    270° = falling zero crossing    (sin =  0, Q3→Q4 boundary)
#    360° = next argmin              (sin = −1, next trough)
#
#  QUADRANTS (universal — same angles always):
#    Q1   0°– 90°   Emergence from trough   / Capitulation near trough
#    Q2  90°–180°   Expansion toward peak   / Accumulation near support
#    Q3 180°–270°   Distribution past peak  / Decline from peak
#    Q4 270°–360°   Exhaustion near trough  / Collapse from peak
#
#  CYCLE DIRECTION:
#    UP   (argmin most recent): current circuit angle in [0°, 180°)
#    DOWN (argmax most recent): current circuit angle in [180°, 360°)
#
#  FFT ENFORCEMENT RULES:
#    Rule 1  Dominant period T from spectral peak arg_max(|FFT|²)
#    Rule 2  Phase φ₀ anchored to global argmin position:
#              φ₀ = −2π/T × argmin_bar + 3π/2   (sin=−1 at trough)
#    Rule 3  Circuit angle θ_now = (2π/T × current_bar + φ₀) mod 2π → °
#    Rule 4  Quadrature bars from anchor: Δt = T/4 (90° offset)
#    Rule 5  Next reversal bar: anchor + T/2 (half period)
#    Rule 6  FFT amplitude from reconstructed waveform peak-to-trough
#    Rule 7  Volume cause-effect: absorption/exhaustion flag per quadrant
#              Absorption at Q3/Q4 UP = smart money selling into rally
#              Exhaustion  at Q1/Q2 DN = selling pressure collapsing
#    Rule 8  Reversal confidence boost:
#              +0.20 if absorption confirmed in distribution quadrant
#              +0.20 if exhaustion confirmed near extremum
#              +0.15 if FFT sinusoid R² > 0.5 (clean cycle)
#              +0.15 if current angle within ±15° of reversal point
#
#  FORECAST TARGET:
#    UP  cycle: target = swing_high (resistance top)
#               + φ-extension if vol exhaustion confirmed
#    DOWN cycle: target = swing_low (support dip)
#               + φ-retracement if absorption confirmed
#    Refined by: FFT half-period projection + golden spiral timing
# ═════════════════════════════════════════════════════════════

def _fit_sinusoid_to_price(close_arr, dominant_period):
    """
    Fit P(t) = A·sin(2π/T·t + φ₀) + m·t + b to close_arr.
    Returns (A, phi0, trend_slope, trend_intercept, r2, fitted_arr).
    Uses least-squares via the FFT coefficient at the dominant frequency.
    """
    n     = len(close_arr)
    arr   = np.asarray(close_arr, dtype=np.float64)
    t     = np.arange(n, dtype=np.float64)

    # detrend
    p        = np.polyfit(t, arr, 1)
    trend    = np.poly1d(p)(t)
    detr     = arr - trend

    # Use FFT coefficient at dominant frequency for exact phase
    T   = float(dominant_period)
    k   = int(round(n / T))                     # bin index
    k   = max(1, min(k, len(np.fft.rfft(detr)) - 1))
    sp  = np.fft.rfft(detr)
    A   = 2.0 * abs(sp[k]) / n
    phi0 = float(np.angle(sp[k]))               # radians

    fitted = A * np.sin(TAU / T * t + phi0) + trend
    ss_res = float(np.sum((arr - fitted)**2))
    ss_tot = float(np.sum((arr - arr.mean())**2))
    r2     = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return float(A), float(phi0), float(p[0]), float(p[1]), round(r2, 4), fitted


def _circuit_angle_from_fft(bar_idx, dominant_period, argmin_bar):
    """
    Compute the 360° circuit angle at a given bar index.

    Convention:
      sin = −1  (trough)  →   0° / 360°   [argmin position]
      sin =  0  (rising)  →  90°           [Q1/Q2 boundary]
      sin = +1  (peak)    → 180°           [argmax position]
      sin =  0  (falling) → 270°           [Q3/Q4 boundary]

    Achieved by mapping the standard sine phase so that
    sin(θ_circuit) = −1  when  bar_idx = argmin_bar.

    Standard sin = −1 at θ = −π/2 = 3π/2, so:
      θ_raw(t) = 2π/T × t
      θ_circuit(t) = θ_raw(t) − θ_raw(argmin) − π/2   mod 2π  →  °
    """
    T   = float(dominant_period)
    raw = TAU / T * (bar_idx - argmin_bar)          # phase relative to anchor
    # shift so that anchor → 0° (sin = −1 means our 0° = trough)
    # raw=0 at argmin → we want circuit=0° there
    # But sin is −1 at 3π/2 standard, so we need +π/2 shift so 0 maps to our trough
    theta = (raw) % TAU
    return float(np.degrees(theta))


def _assign_quadrant(angle_deg):
    """Return 'Q1', 'Q2', 'Q3', or 'Q4' from 0°–360° circuit angle."""
    a = angle_deg % 360.0
    if   a <  90.0: return 'Q1'
    elif a < 180.0: return 'Q2'
    elif a < 270.0: return 'Q3'
    else:           return 'Q4'


def _sinusoid_price_at_bar(bar_idx, A, phi0, slope, intercept, dominant_period):
    """
    Project sinusoidal price at any bar using the fitted model:
      P(t) = A·sin(2π/T·t + φ₀) + slope·t + intercept
    """
    t     = float(bar_idx)
    T     = float(dominant_period)
    trend = slope * t + intercept
    osc   = A * np.sin(TAU / T * t + phi0)
    return float(trend + osc)


def _quadrature_bars(anchor_bar, dominant_period, cycle_dir='UP',
                     A=None, phi0=None, slope=None, intercept=None,
                     swing_low=None, swing_high=None):
    """
    Return the four quadrature bar positions in a full 360° circuit,
    with sinusoidal price projections and cycle-aware labels.

    For UP cycle (anchor = argmin = trough):
      Q1_start = trough        (0°   — ENTRY / support)
      Q2_start = anchor + T/4  (90°  — rising midline)
      Q3_start = anchor + T/2  (180° — ★ RESISTANCE TOP / reversal target)
      Q4_start = anchor + 3T/4 (270° — post-reversal decline begins)
      next_trough = anchor + T (360° — next trough, new cycle)

    For DOWN cycle (anchor = argmax = peak):
      Q3_start = peak           (180° — ENTRY / resistance)
      Q4_start = anchor + T/4  (270° — early collapse)
      Q1_start = anchor + T/2  (0°   — ★ SUPPORT DIP / reversal target)
      Q2_start = anchor + 3T/4 (90°  — post-reversal accumulation)
      next_peak = anchor + T   (180° — next peak, new cycle)
    """
    T  = float(dominant_period)
    has_fit = (A is not None and phi0 is not None
               and slope is not None and intercept is not None)

    def _proj(bar):
        if not has_fit:
            return None
        raw = _sinusoid_price_at_bar(bar, A, phi0, slope, intercept, T)
        # clamp to plausible range if swing anchors are available
        if swing_low is not None and swing_high is not None:
            raw = max(swing_low * 0.99, min(swing_high * 1.01, raw))
        return round(float(raw), 8)

    if cycle_dir == 'UP':
        b_q1  = int(anchor_bar)
        b_q2  = int(anchor_bar + T / 4.0)
        b_q3  = int(anchor_bar + T / 2.0)
        b_q4  = int(anchor_bar + 3.0 * T / 4.0)
        b_end = int(anchor_bar + T)
        return {
            'cycle_dir':    cycle_dir,
            'Q1_start':     b_q1,   'Q1_price': _proj(b_q1),   'Q1_label': 'TROUGH  (entry / support)',
            'Q2_start':     b_q2,   'Q2_price': _proj(b_q2),   'Q2_label': 'rising midline (90°)',
            'Q3_start':     b_q3,   'Q3_price': _proj(b_q3),   'Q3_label': '★ RESISTANCE TOP — reversal target',
            'Q4_start':     b_q4,   'Q4_price': _proj(b_q4),   'Q4_label': 'post-reversal decline (270°)',
            'next_trough':  b_end,  'next_trough_price': _proj(b_end), 'next_trough_label': 'next TROUGH / cycle restart',
        }
    else:  # DOWN
        b_q3  = int(anchor_bar)
        b_q4  = int(anchor_bar + T / 4.0)
        b_q1  = int(anchor_bar + T / 2.0)
        b_q2  = int(anchor_bar + 3.0 * T / 4.0)
        b_end = int(anchor_bar + T)
        return {
            'cycle_dir':   cycle_dir,
            'Q3_start':    b_q3,   'Q3_price': _proj(b_q3),   'Q3_label': 'PEAK  (entry / resistance)',
            'Q4_start':    b_q4,   'Q4_price': _proj(b_q4),   'Q4_label': 'early collapse (270°)',
            'Q1_start':    b_q1,   'Q1_price': _proj(b_q1),   'Q1_label': '★ SUPPORT DIP — reversal target',
            'Q2_start':    b_q2,   'Q2_price': _proj(b_q2),   'Q2_label': 'post-reversal accumulation (90°)',
            'next_peak':   b_end,  'next_peak_price': _proj(b_end),  'next_peak_label': 'next PEAK / cycle restart',
        }



# ── MTF Sinusoidal Circuit: per-timeframe runner + ML ensemble ────────────────
_MTF_CIRCUIT_TFS = [
    ('1m',  60 * 16,  1.0),   # (interval, limit, weight)  1m  — primary
    ('3m',  60 *  8,  0.9),   # 3m
    ('5m',  60 *  5,  0.85),  # 5m
    ('15m', 60 *  3,  0.75),  # 15m
    ('30m', 60 *  2,  0.65),  # 30m
    ('2h',  60 *  1,  0.50),  # 2h — HTF context, lower weight
]


def _run_circuit_on_tf(pair, tf, limit, current_price):
    """
    Fetch klines for `tf`, fit sinusoidal circuit, return a compact
    result dict (same schema as sinusoidal_circuit_engine but without
    the external stf/htf_results dependency).
    Never raises.
    """
    r = {
        'tf': tf, 'pair': pair, 'current_price': current_price,
        'cycle_dir': None, 'current_angle_deg': None,
        'current_quadrant': None, 'quadrant_label': '—',
        'dominant_period': None, 'sinusoid_r2': None,
        'reversal_type': '—', 'reversal_target': None,
        'reversal_pct': None, 'reversal_target_fft': None,
        'reversal_target_swing': None,
        'phi_target_ext': None, 'bars_to_reversal': None,
        'swing_low': None, 'swing_high': None,
        'fft_amplitude': None, 'quadrature_bars': {},
        'error': None,
    }
    try:
        klines    = trader.client.get_klines(symbol=pair, interval=tf, limit=limit)
        close_raw = np.array([float(k[4]) for k in klines], dtype=np.float64)
        low_raw   = np.array([float(k[3]) for k in klines], dtype=np.float64)
        high_raw  = np.array([float(k[2]) for k in klines], dtype=np.float64)

        # If the initial fetch came back short, retry with the maximum allowed
        # limit so we use every available bar on the exchange for this TF.
        if len(close_raw) < 64:
            klines    = trader.client.get_klines(symbol=pair, interval=tf, limit=1000)
            close_raw = np.array([float(k[4]) for k in klines], dtype=np.float64)
            low_raw   = np.array([float(k[3]) for k in klines], dtype=np.float64)
            high_raw  = np.array([float(k[2]) for k in klines], dtype=np.float64)

        # Strip zero and NaN values — keep only rows where all three arrays
        # are finite and strictly positive (i.e. real traded candles).
        _valid = (
            np.isfinite(close_raw) & (close_raw > 0) &
            np.isfinite(low_raw)   & (low_raw   > 0) &
            np.isfinite(high_raw)  & (high_raw  > 0)
        )
        close_arr = close_raw[_valid]
        low_arr   = low_raw[_valid]
        high_arr  = high_raw[_valid]

        n = len(close_arr)
        # Hard floor: need at least 16 bars for a meaningful FFT + sinusoid fit
        if n < 16:
            r['error'] = 'insufficient data'; return r

        swing_low  = float(np.min(low_arr))
        swing_high = float(np.max(high_arr))
        r['swing_low']  = round(swing_low,  8)
        r['swing_high'] = round(swing_high, 8)

        global_amin = int(np.argmin(low_arr))
        global_amax = int(np.argmax(high_arr))
        r['argmin_bar'] = global_amin
        r['argmax_bar'] = global_amax

        cycle_dir = 'UP' if global_amin > global_amax else 'DOWN'
        r['cycle_dir'] = cycle_dir
        anchor_bar  = global_amin if cycle_dir == 'UP' else global_amax
        current_bar = n - 1

        # FFT dominant period
        _dtr, _ = _detrend(close_arr)
        _sp     = np.fft.rfft(_dtr)
        _fr     = np.fft.rfftfreq(n)
        _pw     = np.abs(_sp); _pw[0] = 0
        _vm     = (_fr > 0) & (_fr <= 0.25)
        if np.any(_vm):
            _pw2 = _pw.copy(); _pw2[~_vm] = 0
            _dom = int(np.argmax(_pw2))
            _df  = float(_fr[_dom])
            dominant_period = max(8, int(round(1.0 / _df))) if _df > 0 else 20
        else:
            dominant_period = 20
        dominant_period = min(dominant_period, n // 2)
        r['dominant_period'] = dominant_period

        A_fit, phi0, slope, intercept, r2_fit, _ = \
            _fit_sinusoid_to_price(close_arr, dominant_period)
        r['sinusoid_r2']   = round(r2_fit,        4)
        r['fft_amplitude'] = round(float(A_fit),  8)

        # Circuit angle
        angle_deg = _circuit_angle_from_fft(current_bar, dominant_period, anchor_bar)
        if cycle_dir == 'DOWN':
            down_angle = _circuit_angle_from_fft(current_bar, dominant_period, global_amax)
            angle_deg  = (down_angle + 180.0) % 360.0
        r['current_angle_deg'] = round(angle_deg, 1)
        quadrant = _assign_quadrant(angle_deg)
        r['current_quadrant'] = quadrant
        r['quadrant_label']   = (_CQ_LABEL_UP if cycle_dir == 'UP' else _CQ_LABEL_DN).get(quadrant, '—')

        # Quadrature bars + prices
        r['quadrature_bars'] = _quadrature_bars(
            anchor_bar, dominant_period, cycle_dir=cycle_dir,
            A=A_fit, phi0=phi0, slope=slope, intercept=intercept,
            swing_low=swing_low, swing_high=swing_high
        )

        # Consistent reversal target
        T      = float(dominant_period)
        rev_bar = int(round(anchor_bar + T / 2.0))
        fft_proj  = _sinusoid_price_at_bar(rev_bar, A_fit, phi0, slope, intercept, T)
        r['reversal_target_fft'] = round(float(fft_proj), 8)

        if cycle_dir == 'UP':
            r['reversal_type']   = 'RESISTANCE TOP'
            emp_price  = float(high_arr[global_amax])
            raw_target = 0.40 * swing_high + 0.40 * fft_proj + 0.20 * emp_price
            raw_target = max(current_price, min(swing_high * 1.05, raw_target))
            r['reversal_target_swing'] = round(swing_high, 8)
            span = swing_high - swing_low
            r['phi_target_ext'] = round(swing_high + PHI_INV * span, 8)
        else:
            r['reversal_type']   = 'SUPPORT DIP'
            emp_price  = float(low_arr[global_amin])
            raw_target = 0.40 * swing_low + 0.40 * fft_proj + 0.20 * emp_price
            raw_target = min(current_price, max(swing_low * 0.95, raw_target))
            r['reversal_target_swing'] = round(swing_low, 8)
            span = swing_high - swing_low
            r['phi_target_ext'] = round(swing_low - PHI_INV * span, 8)

        r['reversal_target']  = round(float(raw_target), 8)
        r['reversal_pct']     = round((raw_target - current_price) / (current_price + 1e-20) * 100.0, 4)
        r['bars_to_reversal'] = max(0, rev_bar - current_bar)
        r['reversal_bar_est'] = rev_bar

    except Exception as ex:
        r['error'] = f'{type(ex).__name__}: {ex}'
    return r


def sinusoidal_circuit_mtf(pair, current_price, sel_detail,
                             stf_results=None, htf_results=None):
    """
    Run sinusoidal circuit on every MTF_CIRCUIT_TFS timeframe in parallel,
    then aggregate targets via weighted ML ensemble.

    Returns:
      tf_results  : list of per-TF result dicts
      mtf_summary : aggregated/ML summary dict
    """
    tf_results = []
    with ThreadPoolExecutor(max_workers=len(_MTF_CIRCUIT_TFS)) as ex:
        futs = {
            ex.submit(_run_circuit_on_tf, pair, tf, lim, current_price): (tf, w)
            for tf, lim, w in _MTF_CIRCUIT_TFS
        }
        for fut in as_completed(futs):
            tf, w = futs[fut]
            try:
                res = fut.result()
                res['weight'] = w
                tf_results.append(res)
            except Exception:
                pass

    # sort by TF order
    tf_order = {tf: i for i, (tf, _, _) in enumerate(_MTF_CIRCUIT_TFS)}
    tf_results.sort(key=lambda r: tf_order.get(r.get('tf'), 99))

    # ── ML ensemble: weighted aggregation of reversal targets ────────
    # Only use TFs with valid targets + R² > 0.3 for reliability
    valid = [r for r in tf_results
             if r.get('reversal_target') is not None
             and (r.get('sinusoid_r2') or 0) > 0.2
             and r.get('error') is None]

    mtf_summary = {
        'n_valid_tfs':        len(valid),
        'mtf_target':         None,
        'mtf_target_pct':     None,
        'mtf_phi_ext':        None,
        'mtf_confidence':     0.0,
        'dominant_cycle_dir': None,
        'dominant_quadrant':  None,
        'target_range_low':   None,
        'target_range_high':  None,
        'best_r2_tf':         None,
    }

    if valid:
        # weighted mean target (R² × TF-weight)
        weights  = np.array([r['sinusoid_r2'] * r['weight'] for r in valid], dtype=np.float64)
        targets  = np.array([r['reversal_target'] for r in valid], dtype=np.float64)
        phi_exts = np.array([r['phi_target_ext'] for r in valid
                             if r.get('phi_target_ext') is not None], dtype=np.float64)

        w_sum    = weights.sum()
        if w_sum > 0:
            mtf_target = float(np.average(targets, weights=weights))
        else:
            mtf_target = float(np.median(targets))

        mtf_summary['mtf_target']     = round(mtf_target,  8)
        mtf_summary['mtf_target_pct'] = round(
            (mtf_target - current_price) / (current_price + 1e-20) * 100.0, 4)
        mtf_summary['target_range_low']  = round(float(np.min(targets)),  8)
        mtf_summary['target_range_high'] = round(float(np.max(targets)),  8)

        if len(phi_exts) > 0:
            mtf_summary['mtf_phi_ext'] = round(float(np.average(phi_exts,
                weights=weights[:len(phi_exts)])), 8)

        # dominant cycle direction (majority vote weighted by R²)
        up_w   = sum(r['sinusoid_r2'] * r['weight'] for r in valid if r.get('cycle_dir') == 'UP')
        dn_w   = sum(r['sinusoid_r2'] * r['weight'] for r in valid if r.get('cycle_dir') == 'DOWN')
        mtf_summary['dominant_cycle_dir'] = 'UP' if up_w >= dn_w else 'DOWN'

        # dominant quadrant (mode)
        from collections import Counter
        q_votes = Counter(r.get('current_quadrant') for r in valid if r.get('current_quadrant'))
        mtf_summary['dominant_quadrant'] = q_votes.most_common(1)[0][0] if q_votes else None

        # overall MTF confidence: mean R² of valid TFs × agreement fraction
        mean_r2   = float(np.mean([r['sinusoid_r2'] for r in valid]))
        agree_frac = max(up_w, dn_w) / (up_w + dn_w + 1e-12)
        mtf_summary['mtf_confidence'] = round(min(1.0, mean_r2 * agree_frac * 1.5), 4)

        # best TF by R²
        best_r2_tf = max(valid, key=lambda r: r.get('sinusoid_r2', 0))
        mtf_summary['best_r2_tf'] = best_r2_tf.get('tf')

    return tf_results, mtf_summary


def sinusoidal_circuit_engine(pair, current_price, sel_detail,
                               stf_results=None, htf_results=None):

    """
    360° Sinusoidal Circuit Engine.

    Builds a complete circuit model between argmin/argmax extremas using FFT,
    maps the current price to a circuit angle + quadrant, identifies cycle
    direction, and forecasts the next reversal target (SUPPORT DIP or
    RESISTANCE TOP) with volume cause-effect confirmation.

    Parameters
    ----------
    pair          : symbol (e.g. 'BTCUSDC')
    current_price : float — current 1m close
    sel_detail    : dict from check_dip_conditions (contains close/low/high arrays)
    stf_results   : list of FFT result dicts from full_fft_report (STF)
    htf_results   : list of FFT result dicts from full_fft_report (HTF)

    Returns
    -------
    dict — comprehensive circuit state, never raises
    """
    result = {
        'pair':               pair,
        'current_price':      current_price,
        # cycle info
        'cycle_dir':          'NEUTRAL',
        'cycle_label':        '—',
        'current_angle_deg':  None,
        'current_quadrant':   None,
        'quadrant_label':     '—',
        # circuit geometry
        'dominant_period':    None,
        'fft_amplitude':      None,
        'sinusoid_r2':        None,
        'phi0_rad':           None,
        # quadrature anchors
        'argmin_bar':         None,
        'argmax_bar':         None,
        'swing_low':          None,
        'swing_high':         None,
        'quadrature_bars':    {},
        # reversal forecast
        'reversal_type':      '—',
        'reversal_target':    None,
        'reversal_bar_est':   None,
        'bars_to_reversal':   None,
        'reversal_pct':       None,
        # volume cause-effect
        'absorption_flag':    False,
        'exhaustion_flag':    False,
        'absorption_score':   0.0,
        'exhaustion_score':   0.0,
        'vol_rule_label':     '—',
        # confidence
        'circuit_confidence': 0.0,
        'confidence_detail':  {},
        # φ-extension / retracement on target
        'phi_target_ext':     None,
        'phi_target_label':   '—',
        # error
        'error':              None,
    }

    try:
        # ── 1. Retrieve price arrays ─────────────────────────────
        d         = sel_detail.get(pair, {})
        close_arr = d.get('close_arr')
        low_arr   = d.get('low_arr')
        high_arr  = d.get('high_arr')

        if close_arr is None or len(close_arr) < 64:
            try:
                klines    = trader.client.get_klines(
                    symbol=pair, interval='1m', limit=EXTREMA_LOOKBACK
                )
                close_arr = np.array([float(k[4]) for k in klines], dtype=np.float64)
                low_arr   = np.array([float(k[3]) for k in klines], dtype=np.float64)
                high_arr  = np.array([float(k[2]) for k in klines], dtype=np.float64)
            except Exception as ex:
                result['error'] = str(ex); return result

        close_arr = np.asarray(close_arr, dtype=np.float64)
        low_arr   = np.asarray(low_arr   if low_arr   is not None else close_arr, dtype=np.float64)
        high_arr  = np.asarray(high_arr  if high_arr  is not None else close_arr, dtype=np.float64)
        n         = len(close_arr)

        # ── 2. Global swing anchors ──────────────────────────────
        swing_low  = float(d.get('swing_low')  or np.min(low_arr))
        swing_high = float(d.get('swing_high') or np.max(high_arr))
        global_amin = int(np.argmin(low_arr))
        global_amax = int(np.argmax(high_arr))

        result['argmin_bar']  = global_amin
        result['argmax_bar']  = global_amax
        result['swing_low']   = round(swing_low,  8)
        result['swing_high']  = round(swing_high, 8)

        # ── 3. Cycle direction ───────────────────────────────────
        if global_amin > global_amax:
            cycle_dir = 'UP'       # most recent extreme is the LOW → rising
        elif global_amax > global_amin:
            cycle_dir = 'DOWN'     # most recent extreme is the HIGH → falling
        else:
            cycle_dir = 'UP'       # tie → default bullish (only dip winners here)
        result['cycle_dir'] = cycle_dir

        # ── 4. FFT dominant period ───────────────────────────────
        # Use the best available STF period; fall back to computing it here
        dominant_period = None
        fft_amplitude   = None
        sinusoid_r2     = None
        fft_phase_deg   = None

        # Collect from passed FFT results first (most reliable, multi-TF)
        all_fft = (stf_results or []) + (htf_results or [])
        if all_fft:
            # weight by sinusoid_r2 if available, otherwise uniform
            periods = [r['dominant_period'] for r in all_fft if r.get('dominant_period')]
            r2s     = [r.get('sinusoid_r2') or 0.5 for r in all_fft if r.get('dominant_period')]
            if periods:
                # weighted median of periods
                dominant_period = int(round(np.average(periods, weights=r2s)))
                # best amplitude from STF
                amps = [r.get('osc_amplitude', 0) for r in (stf_results or [])]
                if amps:
                    fft_amplitude = float(np.mean(amps))
                # best sinusoid R² from STF
                r2s_stf = [r.get('sinusoid_r2') for r in (stf_results or [])
                           if r.get('sinusoid_r2') is not None]
                if r2s_stf:
                    sinusoid_r2 = float(np.mean(r2s_stf))
                # FFT phase from STF 1m if available
                for r in (stf_results or []):
                    if r.get('tf') == '1m' and r.get('fft_phase_deg') is not None:
                        fft_phase_deg = r['fft_phase_deg']
                        break

        if dominant_period is None or dominant_period < 4:
            # compute directly from 1m close
            _dtr, _ = _detrend(close_arr)
            _sp     = np.fft.rfft(_dtr)
            _fr     = np.fft.rfftfreq(n)
            _pw     = np.abs(_sp)
            _pw[0]  = 0
            _vm     = (_fr > 0) & (_fr <= 0.25)
            if np.any(_vm):
                _pw2         = _pw.copy(); _pw2[~_vm] = 0
                _dom         = int(np.argmax(_pw2))
                _df          = float(_fr[_dom])
                dominant_period = max(8, int(round(1.0 / _df))) if _df > 0 else 20
            else:
                dominant_period = 20

        dominant_period = min(dominant_period, n // 2)
        result['dominant_period'] = dominant_period

        # ── 5. Fit sinusoid to 1m close ──────────────────────────
        A_fit, phi0, slope, intercept, r2_fit, fitted_arr = \
            _fit_sinusoid_to_price(close_arr, dominant_period)

        if fft_amplitude is None:
            fft_amplitude = A_fit
        if sinusoid_r2 is None:
            sinusoid_r2 = r2_fit

        result['fft_amplitude'] = round(float(fft_amplitude), 8)
        result['sinusoid_r2']   = round(float(sinusoid_r2),   4)
        result['phi0_rad']      = round(float(phi0),          6)

        # ── 6. Circuit angle for current bar (bar n-1) ───────────
        # Anchor = argmin_bar (sin = −1 = 0° in our circuit)
        anchor_bar  = global_amin if cycle_dir == 'UP' else global_amax
        current_bar = n - 1
        angle_deg   = _circuit_angle_from_fft(current_bar, dominant_period, anchor_bar)

        # For DOWN cycle: flip so that argmax is 0° in the down-circuit
        # (mirror the UP convention: peak = 0°, trough = 180°)
        # But the user wants Q4→Q3→Q2→Q1 for DOWN, which matches angle
        # continuing from 180° to 360°.  So for DOWN cycle, the effective
        # down-cycle angle is how far we've gone past argmax:
        if cycle_dir == 'DOWN':
            # angle from argmax (not argmin)
            down_angle = _circuit_angle_from_fft(current_bar, dominant_period, global_amax)
            # In DOWN cycle Q4 is closest to argmax (just started falling)
            # map: 0° from argmax → 270°–360° circuit space (Q4 start)
            angle_deg = (down_angle + 180.0) % 360.0

        result['current_angle_deg'] = round(angle_deg, 1)

        # ── 7. Quadrant assignment ───────────────────────────────
        quadrant = _assign_quadrant(angle_deg)
        result['current_quadrant'] = quadrant

        if cycle_dir == 'UP':
            result['quadrant_label'] = _CQ_LABEL_UP.get(quadrant, '—')
            result['cycle_label']    = 'UP cycle  (argmin most recent → SUPPORT DIP confirmed)'
        else:
            result['quadrant_label'] = _CQ_LABEL_DN.get(quadrant, '—')
            result['cycle_label']    = 'DOWN cycle (argmax most recent → RESISTANCE TOP confirmed)'

        # ── 8. Quadrature bars (with sinusoid price projections) ─────
        result['quadrature_bars'] = _quadrature_bars(
            anchor_bar, dominant_period, cycle_dir=cycle_dir,
            A=A_fit, phi0=phi0, slope=slope, intercept=intercept,
            swing_low=swing_low, swing_high=swing_high
        )

        # ── 9. Reversal target + type ────────────────────────────
        # Consistent target: weighted blend of
        #   (a) historical swing extremum  — structural reference
        #   (b) FFT sinusoid projection at reversal bar — dynamic model
        #   (c) actual argmax/argmin price — empirical anchor
        # Weight: 0.40 × swing  +  0.40 × fft_proj  +  0.20 × argmin/max price
        T   = float(dominant_period)
        half_T = T / 2.0

        if cycle_dir == 'UP':
            result['reversal_type']   = 'RESISTANCE TOP'
            rev_bar    = int(round(anchor_bar + half_T))
            # FFT projection at reversal bar
            fft_proj   = _sinusoid_price_at_bar(rev_bar, A_fit, phi0, slope, intercept, T)
            # empirical: price at argmax bar
            emp_price  = float(high_arr[global_amax])
            # consistent blend
            raw_target = (0.40 * swing_high + 0.40 * fft_proj + 0.20 * emp_price)
            # clamp: never below current_price or above swing_high * 1.05
            raw_target = max(current_price, min(swing_high * 1.05, raw_target))
        else:
            result['reversal_type']   = 'SUPPORT DIP'
            rev_bar    = int(round(anchor_bar + half_T))
            fft_proj   = _sinusoid_price_at_bar(rev_bar, A_fit, phi0, slope, intercept, T)
            emp_price  = float(low_arr[global_amin])
            raw_target = (0.40 * swing_low + 0.40 * fft_proj + 0.20 * emp_price)
            raw_target = min(current_price, max(swing_low * 0.95, raw_target))

        result['reversal_target_fft']   = round(float(fft_proj),   8)
        result['reversal_target_swing'] = round(float(swing_high if cycle_dir == 'UP' else swing_low), 8)
        result['reversal_target_emp']   = round(float(emp_price),   8)

        # bars remaining until reversal
        bars_to_rev = max(0, rev_bar - current_bar)
        result['reversal_bar_est'] = rev_bar
        result['bars_to_reversal'] = bars_to_rev

        # φ-extension on reversal target (structural level beyond the raw extremum)
        span = swing_high - swing_low
        if cycle_dir == 'UP':
            phi_ext = swing_high + PHI_INV * span      # 61.8% extension above top
            phi_lbl = f'+61.8% ext above swing_high → {phi_ext:.8g}'
        else:
            phi_ext = swing_low  - PHI_INV * span      # 61.8% extension below bottom
            phi_lbl = f'−61.8% ext below swing_low  → {phi_ext:.8g}'

        result['reversal_target']  = round(raw_target,  8)
        result['phi_target_ext']   = round(phi_ext,     8)
        result['phi_target_label'] = phi_lbl
        result['reversal_pct']     = round(
            (raw_target - current_price) / (current_price + 1e-20) * 100.0, 4
        )

        # ── 10. Volume cause-effect: absorption & exhaustion ─────
        # Gather from best STF 1m FFT result
        abs_flag  = False; abs_score  = 0.0
        exhs_flag = False; exhs_score = 0.0

        for r in (stf_results or []):
            if r.get('tf') in ('1m', '3m'):
                abs_flag   = abs_flag  or r.get('absorption_flag',  False)
                exhs_flag  = exhs_flag or r.get('exhaustion_flag',  False)
                abs_score  = max(abs_score,  r.get('absorption_score', 0.0))
                exhs_score = max(exhs_score, r.get('exhaustion_score', 0.0))

        # Also check from sel_detail flow data
        det_flow = d.get('delta_ratio', 0.0)
        if det_flow:
            if abs(float(det_flow)) < 0.05 and abs_score < 3.0:
                # near-zero delta with some volume → absorption-like
                abs_score = max(abs_score, 3.0)

        result['absorption_flag']  = abs_flag
        result['exhaustion_flag']  = exhs_flag
        result['absorption_score'] = round(abs_score,  4)
        result['exhaustion_score'] = round(exhs_score, 4)

        # Volume rule interpretation per quadrant + cycle
        if cycle_dir == 'UP':
            if quadrant in ('Q3', 'Q4') and abs_flag:
                result['vol_rule_label'] = (
                    '🟡 ABSORPTION in distribution zone — smart money selling into rally'
                )
            elif quadrant in ('Q3', 'Q4') and exhs_flag:
                result['vol_rule_label'] = (
                    '🔴 EXHAUSTION near resistance — buyer fuel spent, SELL reversal risk'
                )
            elif quadrant in ('Q1', 'Q2') and abs_flag:
                result['vol_rule_label'] = (
                    '🟢 ABSORPTION at support — smart money buying, BUY continuation'
                )
            else:
                result['vol_rule_label'] = '⚪ Neutral — no dominant order-flow signal'
        else:
            if quadrant in ('Q1', 'Q2') and abs_flag:
                result['vol_rule_label'] = (
                    '🟢 ABSORPTION at support dip — smart money buying, BUY reversal near'
                )
            elif quadrant in ('Q1', 'Q2') and exhs_flag:
                result['vol_rule_label'] = (
                    '🔴 EXHAUSTION near support — seller fuel spent, BUY reversal risk'
                )
            elif quadrant in ('Q3', 'Q4') and exhs_flag:
                result['vol_rule_label'] = (
                    '🟡 EXHAUSTION from peak — selling pressure waning, watch for floor'
                )
            else:
                result['vol_rule_label'] = '⚪ Neutral — no dominant order-flow signal'

        # ── 11. Circuit confidence composite ────────────────────
        conf = 0.0
        conf_d = {}

        # a. Sinusoid fit quality (R²)
        r2_contrib = float(sinusoid_r2) * 0.25
        conf      += r2_contrib
        conf_d['sinusoid_r2']   = round(r2_contrib, 4)

        # b. Proximity to nearest quadrant boundary or extremum (±15° = max)
        boundary_angles = [0.0, 90.0, 180.0, 270.0, 360.0]
        min_dist = min(abs(angle_deg - b) for b in boundary_angles)
        min_dist = min(min_dist, abs(angle_deg - 360.0 - min(boundary_angles, key=lambda x: abs(angle_deg-x))))
        prox_score = max(0.0, 1.0 - min_dist / 90.0) * 0.20
        conf      += prox_score
        conf_d['phase_proximity'] = round(prox_score, 4)

        # c. Absorption/exhaustion in the right quadrant
        vol_bonus = 0.0
        if cycle_dir == 'UP' and quadrant in ('Q3', 'Q4'):
            if abs_flag:  vol_bonus += 0.20
            if exhs_flag: vol_bonus += 0.20
        elif cycle_dir == 'DOWN' and quadrant in ('Q1', 'Q2'):
            if abs_flag:  vol_bonus += 0.20
            if exhs_flag: vol_bonus += 0.20
        conf   += min(0.30, vol_bonus)
        conf_d['vol_signal'] = round(min(0.30, vol_bonus), 4)

        # d. Spiral timing alignment
        anchor   = global_amin if cycle_dir == 'UP' else global_amax
        wins     = _spiral_windows(anchor, n=7)
        spi_ok   = _in_spiral_window(current_bar, wins, tol=7)
        spi_c    = 0.15 if spi_ok else 0.0
        conf    += spi_c
        conf_d['spiral_timing'] = round(spi_c, 4)

        # e. Bars-to-reversal: boost if very close (≤10 bars)
        close_bonus = 0.10 if bars_to_rev <= 10 else (0.05 if bars_to_rev <= 20 else 0.0)
        conf       += close_bonus
        conf_d['proximity_to_rev'] = round(close_bonus, 4)

        result['circuit_confidence'] = round(min(1.0, conf), 4)
        result['confidence_detail']  = conf_d

    except Exception as ex:
        result['error'] = f'{type(ex).__name__}: {ex}'

    return result


def print_circuit_block(circ, label_map, mtf_tf_results=None, mtf_summary=None):
    """
    Print the full 360° sinusoidal circuit block.
    Called after print_phi_reversal_block() in the main loop.
    Includes: cycle quadrature with prices, per-TF circuit table, MTF ML targets.
    """
    if not circ:
        return

    pair = circ.get('pair', '?')
    lbl  = label_map.get(pair, pair.replace('USDC', ''))
    cp   = circ.get('current_price') or 0.0
    w    = 70

    def pf(v):
        if v is None: return '—'
        return f'{v:.6f}' if abs(v) < 1 else f'{v:.4f}'

    def pp(v, ref=None):
        if v is None: return '—'
        base = ref if ref is not None else cp
        pct  = (v - base) / (base + 1e-20) * 100.0
        arr  = '▲' if pct > 0 else '▼'
        return f'{pf(v)}  ({arr}{abs(pct):.2f}%)'

    err       = circ.get('error')
    cycle_dir = circ.get('cycle_dir', 'NEUTRAL')
    quadrant  = circ.get('current_quadrant', '?')
    angle     = circ.get('current_angle_deg')
    conf      = circ.get('circuit_confidence', 0.0)
    conf_bar  = '█' * int(conf * 20) + '░' * (20 - int(conf * 20))

    rev_type  = circ.get('reversal_type', '—')
    rev_tgt   = circ.get('reversal_target')
    rev_tgt_fft   = circ.get('reversal_target_fft')
    rev_tgt_swing = circ.get('reversal_target_swing')
    rev_tgt_emp   = circ.get('reversal_target_emp')
    rev_pct   = circ.get('reversal_pct')
    rev_bar   = circ.get('reversal_bar_est')
    bars_rem  = circ.get('bars_to_reversal')
    T         = circ.get('dominant_period')
    r2        = circ.get('sinusoid_r2')
    amp       = circ.get('fft_amplitude')
    qbars     = circ.get('quadrature_bars', {})
    abs_flag  = circ.get('absorption_flag',  False)
    exhs_flag = circ.get('exhaustion_flag',  False)
    vol_lbl   = circ.get('vol_rule_label', '—')
    phi_ext   = circ.get('phi_target_ext')
    phi_lbl   = circ.get('phi_target_label', '—')
    amin_bar  = circ.get('argmin_bar')
    amax_bar  = circ.get('argmax_bar')
    slo       = circ.get('swing_low')
    shi       = circ.get('swing_high')
    conf_d    = circ.get('confidence_detail', {})

    # cycle direction icon
    if   cycle_dir == 'UP':      dir_icon = '▲ UP   — rising from SUPPORT DIP'
    elif cycle_dir == 'DOWN':    dir_icon = '▼ DOWN — falling toward SUPPORT DIP'
    else:                        dir_icon = '→ NEUTRAL'

    # quadrant display in the arc diagram
    def _arc(cur_q, cdir):
        """ASCII sinusoidal arc showing current quadrant."""
        labels = ['Q1', 'Q2', 'Q3', 'Q4']
        parts  = []
        for q in labels:
            tag = f'[{q}▶]' if q == cur_q else f' {q} '
            parts.append(tag)
        arc = '─'.join(parts)
        lo = '0°=argmin' if cdir == 'UP' else '180°=argmax'
        hi = '180°=argmax' if cdir == 'UP' else '360°→argmin'
        return f'  │  {lo} ── {arc} ── {hi}'

    print(f'\n  {"═"*w}')
    print(f'  ◈  360° SINUSOIDAL CIRCUIT  ·  {lbl}  ({pair})')
    print(f'  {"═"*w}')
    if err:
        print(f'  [WARN] {err}')

    # ── Section 1: Cycle identification ─────────────────────────────
    print(f'  ┌─ CYCLE IDENTIFICATION {"─"*46}┐')
    print(f'  │  Entry price        : {pf(cp)}')
    print(f'  │  Swing low (argmin) : {pf(slo)}   @ bar {amin_bar}')
    print(f'  │  Swing high (argmax): {pf(shi)}   @ bar {amax_bar}')
    print(f'  │  Cycle direction    : {dir_icon}')
    print(f'  │  {circ.get("cycle_label","—")}')
    print(f'  └{"─"*w}┘')
    print()

    # ── Section 2: 360° circuit arc diagram ─────────────────────────
    print(f'  ┌─ 360° CIRCUIT ARC  (P(t) = A·sin(2π/T·t + φ₀) + trend) {"─"*8}┐')
    print(_arc(quadrant, cycle_dir))
    print(f'  │')
    ang_str = f'{angle:.1f}°' if angle is not None else '?'
    print(f'  │  Current angle      : {ang_str}   [{quadrant}]')
    print(f'  │  Quadrant meaning   : {circ.get("quadrant_label","—")}')
    print(f'  │')
    print(f'  │  QUADRANT LEGEND (both UP and DOWN cycles):')
    print(f'  │    Q1   0°– 90°  Emergence from trough  /  Capitulation near trough')
    print(f'  │    Q2  90°–180°  Expansion toward peak  /  Accumulation near support')
    print(f'  │    Q3 180°–270°  Distribution past peak /  Decline from peak')
    print(f'  │    Q4 270°–360°  Exhaustion near trough /  Collapse from peak')
    print(f'  │')
    print(f'  │  UP cycle  Q sequence : Q1 → Q2 → Q3[REVERSAL TOP] → Q4 → next trough')
    print(f'  │  DOWN cycle Q sequence: Q3 → Q4 → Q1[REVERSAL DIP] → Q2 → next peak')
    print(f'  └{"─"*w}┘')
    print()

    # ── Section 3: Sinusoidal fit + FFT parameters ───────────────────
    print(f'  ┌─ FFT SINUSOIDAL FIT  (P = A·sin(2π/T·t + φ₀) + trend) {"─"*10}┐')
    print(f'  │  Dominant period T  : {T} bars   (~{T} minutes on 1m)')
    t_h = round(T / 60.0, 1) if T else '?'
    print(f'  │  Period in hours    : {t_h} h')
    print(f'  │  Amplitude A        : {pf(amp)}   '
          f'({round(float(amp)/float(cp)*100,3) if amp and cp else "?"}% of price)')
    print(f'  │  Sinusoid fit R²    : {r2}   '
          f'({"clean cycle ✔" if (r2 or 0) > 0.5 else "noisy — interpret with caution"})')
    print(f'  │')

    # ── Quadrature with prices and cycle-aware labeling ──────────────
    q_seg = round(T/4.0, 1) if T else '?'
    print(f'  │  QUADRATURE BARS  (T/4 = {q_seg} bars each segment)  [{cycle_dir} CYCLE]')
    if qbars:
        cdir_q = qbars.get('cycle_dir', cycle_dir)
        if cdir_q == 'UP':
            _q_order = [
                ('Q1_start',    'Q1_price',         'Q1_label',          '  0°  — TROUGH / entry'),
                ('Q2_start',    'Q2_price',         'Q2_label',          ' 90°  — rising midline'),
                ('Q3_start',    'Q3_price',         'Q3_label',          '180°  — ★ REVERSAL TOP'),
                ('Q4_start',    'Q4_price',         'Q4_label',          '270°  — post-top decline'),
                ('next_trough', 'next_trough_price','next_trough_label', '360°  — next trough / new cycle'),
            ]
        else:
            _q_order = [
                ('Q3_start', 'Q3_price', 'Q3_label', '180°  — PEAK / entry'),
                ('Q4_start', 'Q4_price', 'Q4_label', '270°  — early collapse'),
                ('Q1_start', 'Q1_price', 'Q1_label', '360°  — ★ REVERSAL DIP'),
                ('Q2_start', 'Q2_price', 'Q2_label', ' 90°  — post-dip accumulation'),
                ('next_peak', 'next_peak_price', 'next_peak_label', '180°  — next peak / new cycle'),
            ]
        for bk, pk, lk, ang_hint in _q_order:
            bar_v  = qbars.get(bk)
            pr_v   = qbars.get(pk)
            lbl_v  = qbars.get(lk, '')
            cur_mk = ' ◄ NOW' if (bar_v is not None
                                  and angle is not None
                                  and _assign_quadrant(angle) == bk.split('_')[0]
                                  and bk.startswith('Q')) else ''
            pr_s   = pf(pr_v) if pr_v else '—'
            print(f'  │    {ang_hint}  bar {bar_v:<6}  price {pr_s}{cur_mk}')
    print(f'  └{"─"*w}┘')
    print()

    # ── Section 4: Reversal forecast (consistent targets) ────────────
    rev_arr = '▲' if (rev_pct or 0) > 0 else '▼'
    print(f'  ┌─ REVERSAL FORECAST  (next circuit extremum) {"─"*22}┐')
    print(f'  │  Reversal type      : {rev_type}')
    # consistent target breakdown
    if rev_tgt is not None:
        rpct_s = f'{rev_arr}{rev_pct:+.2f}%' if rev_pct is not None else ''
        print(f'  │  ── CONSISTENT TARGET (blended) ──────────────────────────')
        print(f'  │  Reversal target    : {pf(rev_tgt)}  ({rpct_s})')
        print(f'  │    ├ Swing extremum : {pf(rev_tgt_swing)}'
              f'  ({round((rev_tgt_swing - cp)/(cp+1e-20)*100,2):+.2f}%)'
              if rev_tgt_swing else '')
        print(f'  │    ├ FFT projection : {pf(rev_tgt_fft)}'
              f'  ({round((rev_tgt_fft - cp)/(cp+1e-20)*100,2):+.2f}%)'
              if rev_tgt_fft else '')
        print(f'  │    └ Empirical bar  : {pf(rev_tgt_emp)}'
              f'  ({round((rev_tgt_emp - cp)/(cp+1e-20)*100,2):+.2f}%)'
              if rev_tgt_emp else '')
    print(f'  │  At bar est.        : {rev_bar}  ({bars_rem} bars remaining)')
    print(f'  │  φ-extension target : {phi_lbl}')
    print(f'  │    [{pp(phi_ext)}]')
    print(f'  └{"─"*w}┘')
    print()

    # ── Section 5: Volume cause-effect signals ────────────────────────
    print(f'  ┌─ VOLUME CAUSE-EFFECT  (V(t) ≈ P(t+Δφ) — volume leads price) {"─"*4}┐')
    print(f'  │  {vol_lbl}')
    print(f'  │')
    abs_bar  = '█' * min(20, int(circ.get("absorption_score",0)*2)) + '░' * max(0, 20-int(circ.get("absorption_score",0)*2))
    exhs_bar = '█' * min(20, int(circ.get("exhaustion_score",0)*5)) + '░' * max(0, 20-int(circ.get("exhaustion_score",0)*5))
    print(f'  │  Absorption score   : {circ.get("absorption_score",0.0):.4f}  [{abs_bar}]'
          f'  {"🟡 CONFIRMED" if abs_flag else ""}')
    print(f'  │  Exhaustion score   : {circ.get("exhaustion_score",0.0):.4f}  [{exhs_bar}]'
          f'  {"🔴 CONFIRMED" if exhs_flag else ""}')
    print(f'  │')
    print(f'  │  CAUSE-EFFECT LAW (market cycle):')
    print(f'  │    Volume = CAUSE   (effort, intent)')
    print(f'  │    Price  = EFFECT  (result, follows after Δφ lag)')
    print(f'  │    Effort ≠ Result  → reversal inevitable')
    print(f'  │    Volume leads by: V(t) ≈ P(t + dominant_period/4)')
    print(f'  └{"─"*w}┘')
    print()

    # ── Section 6: Circuit confidence ────────────────────────────────
    print(f'  ┌─ CIRCUIT CONFIDENCE SCORE {"─"*42}┐')
    print(f'  │  Sinusoid R² contrib  : {conf_d.get("sinusoid_r2",0.0):.4f}  (weight 0.25)')
    print(f'  │  Phase proximity      : {conf_d.get("phase_proximity",0.0):.4f}  (weight 0.20)')
    print(f'  │  Volume signal        : {conf_d.get("vol_signal",0.0):.4f}  (weight 0.30)')
    print(f'  │  Spiral timing        : {conf_d.get("spiral_timing",0.0):.4f}  (weight 0.15)')
    print(f'  │  Reversal proximity   : {conf_d.get("proximity_to_rev",0.0):.4f}  (weight 0.10)')
    print(f'  │  ──────────────────────────────────────────────────────────────')
    print(f'  │  CIRCUIT CONFIDENCE   : {conf:.4f}  [{conf_bar}]')
    if   conf >= 0.75: cgrade = 'STRONG CIRCUIT SIGNAL ★★★ — high probability reversal'
    elif conf >= 0.55: cgrade = 'MODERATE SIGNAL ★★ — reasonable circuit setup'
    elif conf >= 0.35: cgrade = 'WEAK SIGNAL ★ — incomplete circuit alignment'
    else:              cgrade = 'LOW CONFIDENCE — circuit not aligned yet'
    print(f'  │  Grade                : {cgrade}')
    print(f'  └{"─"*w}┘')
    print()

    # ── Section 7: Per-timeframe circuit table ────────────────────────
    if mtf_tf_results:
        print(f'  ┌─ MTF CIRCUIT TABLE  (all timeframes) {"─"*30}┐')
        hdr = (f'  │  {"TF":<5}  {"Dir":<5}  {"Q":<3}  {"Angle":>7}  '
               f'{"R²":>6}  {"T-bars":>7}  {"Target":>12}  '
               f'{"Δ%":>7}  {"φ-Ext":>12}  {"BarsLeft":>8}')
        print(hdr)
        print(f'  │  {"─"*90}')
        for r in mtf_tf_results:
            tf_s   = r.get('tf', '—')
            cdir_s = r.get('cycle_dir') or '—'
            q_s    = r.get('current_quadrant') or '—'
            ang_s  = f'{r.get("current_angle_deg",0.0):.1f}°' if r.get('current_angle_deg') is not None else '—'
            r2_s   = f'{r.get("sinusoid_r2",0.0):.3f}' if r.get('sinusoid_r2') is not None else '—'
            T_s    = str(r.get('dominant_period') or '—')
            tgt_s  = pf(r.get('reversal_target'))
            pct_s  = (f'{r.get("reversal_pct",0.0):+.2f}%'
                      if r.get('reversal_pct') is not None else '—')
            ext_s  = pf(r.get('phi_target_ext'))
            bleft  = str(r.get('bars_to_reversal') or '—')
            err_s  = f'  [!{r.get("error","")[:20]}]' if r.get('error') else ''
            # mark current cycle direction
            dir_mk = '▲' if cdir_s == 'UP' else ('▼' if cdir_s == 'DOWN' else '→')
            rev_mk = '★' if q_s in ('Q3','Q4') and cdir_s == 'UP' else \
                     ('★' if q_s in ('Q1','Q2') and cdir_s == 'DOWN' else ' ')
            print(f'  │  {tf_s:<5}  {dir_mk}{cdir_s:<4}  {rev_mk}{q_s:<2}  {ang_s:>7}  '
                  f'{r2_s:>6}  {T_s:>7}  {tgt_s:>12}  '
                  f'{pct_s:>7}  {ext_s:>12}  {bleft:>8}{err_s}')
        print(f'  └{"─"*w}┘')
        print()

    # ── Section 8: MTF ML aggregate targets ──────────────────────────
    if mtf_summary and mtf_summary.get('mtf_target') is not None:
        ms     = mtf_summary
        mtgt   = ms['mtf_target']
        mtpct  = ms.get('mtf_target_pct', 0.0)
        mext   = ms.get('mtf_phi_ext')
        mconf  = ms.get('mtf_confidence', 0.0)
        mdir   = ms.get('dominant_cycle_dir', '—')
        mq     = ms.get('dominant_quadrant', '—')
        nv     = ms.get('n_valid_tfs', 0)
        rlo    = ms.get('target_range_low')
        rhi    = ms.get('target_range_high')
        best_r2_tf = ms.get('best_r2_tf', '—')
        mconf_bar  = '█' * int(mconf * 20) + '░' * (20 - int(mconf * 20))
        marr   = '▲' if (mtpct or 0) > 0 else '▼'

        if   mconf >= 0.65: mgrade = 'HIGH MTF AGREEMENT ★★★'
        elif mconf >= 0.45: mgrade = 'MODERATE MTF AGREEMENT ★★'
        elif mconf >= 0.25: mgrade = 'WEAK MTF AGREEMENT ★'
        else:               mgrade = 'LOW MTF CONFIDENCE'

        print(f'  ┌─ MTF ML AGGREGATE TARGET  ({nv} valid TFs) {"─"*25}┐')
        print(f'  │  Dominant direction : {mdir}  (dominant quadrant: {mq})')
        print(f'  │  Best R² timeframe  : {best_r2_tf}')
        print(f'  │  ──────────────────────────────────────────────────────────────')
        print(f'  │  ★ MTF TARGET        : {pf(mtgt)}  ({marr}{mtpct:+.2f}%)')
        print(f'  │    Target range     : {pf(rlo)}  →  {pf(rhi)}')
        if mext:
            mext_pct = (mext - cp) / (cp + 1e-20) * 100.0
            print(f'  │    φ-Ext MTF target : {pf(mext)}  ({marr}{mext_pct:+.2f}%)')
        print(f'  │  MTF Confidence     : {mconf:.4f}  [{mconf_bar}]  {mgrade}')
        print(f'  └{"─"*w}┘')
        print()

    # ── FINAL SUMMARY ───────────────────────────────────────────────
    # Use MTF ML target if available, else fall back to 1m circuit target
    summary_tgt     = (mtf_summary.get('mtf_target') if mtf_summary else None) or rev_tgt
    summary_tgt_pct = (mtf_summary.get('mtf_target_pct') if mtf_summary else None) or rev_pct
    summary_ext     = (mtf_summary.get('mtf_phi_ext') if mtf_summary else None) or phi_ext
    s_arr = '▲' if (summary_tgt_pct or 0) > 0 else '▼'

    print(f'  {"═"*w}')
    print(f'  ★★★  360° CIRCUIT DECISION  ·  {lbl}')
    print(f'  {"═"*w}')
    print(f'  ▶  ENTRY              : {pf(cp)}')
    print(f'  ▶  CYCLE DIRECTION    : {cycle_dir}  ({quadrant}  at {ang_str})')
    print(f'  ▶  REVERSAL TYPE      : {rev_type}')
    print(f'  ▶  REVERSAL TARGET    : {pf(rev_tgt)}'
          + (f'  ({rev_arr}{rev_pct:+.2f}%)' if rev_pct is not None else ''))
    print(f'  ▶  MTF ML TARGET      : {pf(summary_tgt)}'
          + (f'  ({s_arr}{summary_tgt_pct:+.2f}%)' if summary_tgt_pct is not None else ''))
    print(f'  ▶  φ-EXT TARGET (MTF) : {pf(summary_ext)}')
    print(f'  ▶  BARS TO REVERSAL   : ~{bars_rem}  (bar {rev_bar} est)')
    print(f'  ▶  VOLUME SIGNAL      : {vol_lbl}')
    print(f'  ▶  CONFIDENCE         : {conf:.4f}  [{cgrade}]')
    if mtf_summary:
        print(f'  ▶  MTF CONFIDENCE     : {mtf_summary.get("mtf_confidence",0.0):.4f}')
    print()
    print(f'  {"═"*w}\n')# [0.6180, 0.3820, 0.2361, 0.1459, 0.0902, 0.0557, 0.0344]
PHI_NEG_LABELS  = [f'φ⁻{n} ({PHI**-n:.4f})' for n in range(1, 8)]
PHI_EXT_POWERS  = [(PHI,  'φ¹  (1.618 ext)'),
                   (PHI2, 'φ²  (2.618 ext)')]
# Golden-triangle geometry
_GT_BASE_ANG    = 72.0
_GT_APEX_ANG    = 36.0
_GT_HEIGHT_MULT = np.sin(np.radians(_GT_BASE_ANG)) / (
                      2.0 * np.sin(np.radians(_GT_APEX_ANG)))  # ≈ 0.9511


def _phi_bands_all(swing_low, swing_high, direction='up'):
    """
    Return list of (label, ratio, price) for ALL φ bands
    (retracements φ⁻¹…φ⁻⁷ and extensions φ¹ φ²).
    """
    span = swing_high - swing_low
    if span <= 0 or swing_low <= 0:
        return []
    bands = []
    for n, label in enumerate(PHI_NEG_LABELS, start=1):
        ratio = PHI ** -n
        if direction == 'up':
            price = swing_low + ratio * span
        else:
            price = swing_high - ratio * span
        bands.append((label, ratio, round(float(price), 8)))
    for ratio, label in PHI_EXT_POWERS:
        if direction == 'up':
            price = swing_low + ratio * span
        else:
            price = swing_high - ratio * span
        bands.append((label, ratio, round(float(price), 8)))
    return bands


def _nearest_phi_band(current, bands):
    """Return (label, ratio, level, dist_pct) of closest φ band."""
    if not bands:
        return None, None, None, 999.0
    best = min(bands, key=lambda b: abs(b[2] - current))
    dist = abs(best[2] - current) / (current + 1e-20) * 100.0
    return best[0], best[1], best[2], dist


def _golden_triangle_targets(pivot, bar_range, direction='up'):
    """
    T1 = pivot ± apex_height  (primary)
    T2 = pivot ± φ² × bar_range  (gnomon / extended)
    T1_ret = pivot ∓ apex_height  (retracement stop reference)
    """
    apex_h = bar_range * _GT_HEIGHT_MULT
    gnomon = bar_range * PHI2
    sign   = 1.0 if direction == 'up' else -1.0
    return {
        'T1_primary':  round(pivot + sign * apex_h,  8),
        'T2_gnomon':   round(pivot + sign * gnomon,   8),
        'T1_retrace':  round(pivot - sign * apex_h,  8),
    }


def _spiral_windows(anchor_bar, n=7):
    """Bar indices where golden spiral completes a π/2 quarter-turn from anchor."""
    return np.array([anchor_bar + round(PHI ** k) for k in range(1, n + 1)],
                    dtype=int)


def _in_spiral_window(bar, windows, tol=1):
    return bool(np.any(np.abs(windows.astype(int) - int(bar)) <= tol))


def phi_reversal_forecast(pair, current_price, sel_detail,
                           order=7, phi_band_tol_pct=1.2, min_confidence=0.25):
    """
    Full φ·e·π reversal forecast for the MTF winner.

    Uses 1m data already in sel_detail when available; re-fetches if needed.
    Returns a comprehensive result dict — never raises.
    """
    result = {
        'pair':           pair,
        'current_price':  current_price,
        'trend':          'NEUTRAL',
        'direction':      '—',
        'forecast_price': None,
        'target_T1':      None,
        'target_T2':      None,
        'phi_band_label': '—',
        'phi_band_level': None,
        'phi_score':      0.0,
        'spiral_ok':      False,
        'confidence':     0.0,
        'argmin_bar':     None,
        'argmax_bar':     None,
        'swing_low':      None,
        'swing_high':     None,
        'phi_bands_above': [],
        'phi_bands_below': [],
        'all_signals':    [],
        'n_extrema':      0,
        'error':          None,
    }

    try:
        d         = sel_detail.get(pair, {})
        close_arr = d.get('close_arr')
        low_arr   = d.get('low_arr')
        high_arr  = d.get('high_arr')

        # ── fetch 1m data if not already in sel_detail ───────────
        if close_arr is None or len(close_arr) < 50:
            try:
                klines = trader.client.get_klines(
                    symbol=pair, interval='1m', limit=EXTREMA_LOOKBACK
                )
                close_arr = np.array([float(k[4]) for k in klines], dtype=np.float64)
                low_arr   = np.array([float(k[3]) for k in klines], dtype=np.float64)
                high_arr  = np.array([float(k[2]) for k in klines], dtype=np.float64)
            except Exception as ex:
                result['error'] = str(ex); return result

        close_arr = np.asarray(close_arr, dtype=np.float64)
        low_arr   = np.asarray(low_arr  if low_arr  is not None else close_arr, dtype=np.float64)
        high_arr  = np.asarray(high_arr if high_arr is not None else close_arr, dtype=np.float64)
        n         = len(close_arr)

        # ── swing anchors ────────────────────────────────────────
        swing_low  = d.get('swing_low')  or float(np.min(low_arr))
        swing_high = d.get('swing_high') or float(np.max(high_arr))
        result['swing_low']  = round(swing_low,  8)
        result['swing_high'] = round(swing_high, 8)

        # ── EXTREMA DETECTION (argmin/argmax + argrelextrema) ────
        # Global argmin/argmax (same logic as multi_tf_argmin_check)
        global_amin = int(np.argmin(low_arr))
        global_amax = int(np.argmax(high_arr))
        result['argmin_bar'] = global_amin
        result['argmax_bar'] = global_amax

        # Direction: most recent extreme determines bias
        if global_amin > global_amax:
            base_dir = 'up'    # deepest low is MORE recent → mean-reversion BUY
        elif global_amax > global_amin:
            base_dir = 'down'  # highest high is MORE recent → mean-reversion SELL
        else:
            base_dir = 'up'   # tie → default bullish (we only run on confirmed dips)

        # Local extrema via argrelextrema for richer signal set
        signals = []
        if SCIPY_AVAILABLE and n >= order * 3:
            local_lows  = _argrelextrema(low_arr,   np.less,    order=order)[0]
            local_highs = _argrelextrema(high_arr,  np.greater, order=order)[0]
            result['n_extrema'] = len(local_lows) + len(local_highs)

            # Build (bar, price, kind) pairs from local extrema
            extrema = (
                [(int(i), float(low_arr[i]),  'low')  for i in local_lows]  +
                [(int(i), float(high_arr[i]), 'high') for i in local_highs]
            )
            extrema.sort(key=lambda x: x[0])

            # Score each consecutive pair as a potential reversal
            for i in range(1, len(extrema)):
                prev_bar, prev_px, prev_kind = extrema[i - 1]
                curr_bar, curr_px, curr_kind = extrema[i]
                lo = min(prev_px, curr_px)
                hi = max(prev_px, curr_px)
                if hi <= lo:
                    continue

                # --- signal direction ---
                if curr_kind == 'low':
                    sig_dir  = 'BUY'
                    phi_dir  = 'up'
                    pivot    = curr_px
                else:
                    sig_dir  = 'SELL'
                    phi_dir  = 'down'
                    pivot    = curr_px

                # --- Layer 1: φ decay bands ---
                bands = _phi_bands_all(lo, hi, direction=phi_dir)
                lbl, ratio, level, dist_pct = _nearest_phi_band(curr_px, bands)
                if dist_pct > phi_band_tol_pct or lbl is None:
                    continue

                phi_score = max(0.0, 1.0 - dist_pct / phi_band_tol_pct)

                # --- Layer 2: Golden triangle ---
                bar_range = hi - lo
                gt        = _golden_triangle_targets(pivot, bar_range, phi_dir)

                # --- Layer 3: Golden spiral timing ---
                windows   = _spiral_windows(prev_bar, n=7)
                spiral_ok = _in_spiral_window(curr_bar, windows, tol=order)

                # --- Composite confidence ---
                band_score   = max(0.0, 1.0 - dist_pct / 1.0)
                spiral_score = 1.0 if spiral_ok else 0.0
                confidence   = round(
                    0.50 * band_score + 0.30 * spiral_score + 0.20 * phi_score, 4
                )
                if confidence < min_confidence:
                    continue

                signals.append({
                    'bar':            curr_bar,
                    'price':          curr_px,
                    'direction':      sig_dir,
                    'phi_band_label': lbl,
                    'phi_band_level': level,
                    'phi_score':      round(phi_score, 4),
                    'spiral_ok':      spiral_ok,
                    'T1_primary':     gt['T1_primary'],
                    'T2_gnomon':      gt['T2_gnomon'],
                    'T1_retrace':     gt['T1_retrace'],
                    'confidence':     confidence,
                })
        else:
            result['n_extrema'] = 0

        result['all_signals'] = sorted(signals, key=lambda s: s['bar'])

        # ── Determine trend direction from global argmin/argmax ──
        # (consistent with the main mtf argmin>argmax logic)
        result['direction'] = 'BUY' if base_dir == 'up' else 'SELL'
        result['trend']     = 'UP'  if base_dir == 'up' else 'DOWN'

        # ── φ bands from global swing for the full price range ───
        all_bands   = _phi_bands_all(swing_low, swing_high, direction=base_dir)
        above_bands = sorted(
            [(lbl2, lvl) for lbl2, _, lvl in all_bands if lvl > current_price * 1.001],
            key=lambda x: x[1]
        )
        below_bands = sorted(
            [(lbl2, lvl) for lbl2, _, lvl in all_bands if lvl < current_price * 0.999],
            key=lambda x: x[1], reverse=True
        )
        result['phi_bands_above'] = above_bands[:4]
        result['phi_bands_below'] = below_bands[:4]

        # ── Nearest φ band to current price ─────────────────────
        lbl_now, _, lvl_now, dist_now = _nearest_phi_band(current_price, all_bands)
        result['phi_band_label'] = lbl_now or '—'
        result['phi_band_level'] = lvl_now
        result['phi_score']      = round(max(0.0, 1.0 - dist_now / phi_band_tol_pct), 4)

        # ── Golden triangle from global swing ────────────────────
        bar_range_global = swing_high - swing_low
        gt_global = _golden_triangle_targets(current_price, bar_range_global, base_dir)
        result['target_T1'] = gt_global['T1_primary']
        result['target_T2'] = gt_global['T2_gnomon']

        # ── Spiral check for current bar ─────────────────────────
        anchor_bar = global_amin if base_dir == 'up' else global_amax
        windows_now = _spiral_windows(anchor_bar, n=7)
        result['spiral_ok'] = _in_spiral_window(n - 1, windows_now, tol=order)

        # ── Composite confidence on current price ────────────────
        band_s = max(0.0, 1.0 - dist_now / 1.0)
        spi_s  = 1.0 if result['spiral_ok'] else 0.0
        result['confidence'] = round(0.50 * band_s + 0.30 * spi_s +
                                     0.20 * result['phi_score'], 4)

        # ── Forecast price: median of nearest 3 φ bands above ────
        # (for BUY) or below (for SELL) the current price
        if base_dir == 'up' and above_bands:
            targets = [lvl for _, lvl in above_bands[:3]]
        elif base_dir == 'down' and below_bands:
            targets = [lvl for _, lvl in below_bands[:3]]
        else:
            targets = []

        if targets:
            result['forecast_price'] = round(float(np.median(targets)), 8)
        else:
            # fall back to T1 if no φ bands found
            result['forecast_price'] = result['target_T1']

    except Exception as ex:
        result['error'] = f'{type(ex).__name__}: {ex}'

    return result


def print_phi_reversal_block(rev, label_map):
    """
    Print the φ-Reversal forecast block for the MTF winner.
    Appears after print_ml_report() in the main loop.
    """
    if not rev:
        return

    pair = rev.get('pair', '?')
    lbl  = label_map.get(pair, pair.replace('USDC', ''))
    cp   = rev.get('current_price') or 0.0
    w    = 66

    def pf(v):
        if v is None: return '—'
        return f'{v:.6f}' if abs(v) < 1 else f'{v:.4f}'

    def pp(v):
        if v is None: return '—'
        pct = (v - cp) / (cp + 1e-20) * 100.0
        arrow = '▲' if pct > 0 else '▼'
        return f'{pf(v)}  ({arrow}{pct:+.2f}%)'

    trend     = rev.get('trend', 'NEUTRAL')
    direction = rev.get('direction', '—')
    conf      = rev.get('confidence', 0.0)
    spiral_ok = rev.get('spiral_ok', False)
    phi_score = rev.get('phi_score', 0.0)
    n_ext     = rev.get('n_extrema', 0)
    err       = rev.get('error')

    # trend label + icon
    if   trend == 'UP':      trend_icon = '▲ UP  (BUY reversal setup)'
    elif trend == 'DOWN':    trend_icon = '▼ DOWN (SELL reversal setup)'
    else:                    trend_icon = '→ NEUTRAL'

    # confidence bar
    conf_bar = '█' * int(conf * 20) + '░' * (20 - int(conf * 20))

    print(f'\n  {"═"*w}')
    print(f'  ◈  φ-REVERSAL FORECAST  ·  {lbl}  ({pair})')
    print(f'  {"═"*w}')

    if err:
        print(f'  [WARN] {err}')

    # ── Header: trend / direction / confidence ────────────────────
    print(f'  ┌─ TREND & REVERSAL DIRECTION {"─"*36}┐')
    print(f'  │  Entry price    : {pf(cp)}')
    print(f'  │  Swing low      : {pf(rev.get("swing_low"))}   '
          f'argmin@bar {rev.get("argmin_bar","—")}')
    print(f'  │  Swing high     : {pf(rev.get("swing_high"))}   '
          f'argmax@bar {rev.get("argmax_bar","—")}')
    print(f'  │  TREND          : {trend_icon}')
    print(f'  │  Extrema found  : {n_ext} local swing points')
    print(f'  └{"─"*w}┘')
    print()

    # ── φ decay bands ─────────────────────────────────────────────
    print(f'  ┌─ φ DECAY BANDS  (negative exponential powers of φ) {"─"*12}┐')
    print(f'  │  φ⁻¹…φ⁻⁷ = {np.round(PHI_NEG_POWERS, 4).tolist()}')
    print(f'  │  Nearest band   : {rev.get("phi_band_label","—")}')
    print(f'  │  Band level     : {pf(rev.get("phi_band_level"))}')
    print(f'  │  φ score        : {phi_score:.4f}  (1.0 = exactly on band)')
    if rev.get('phi_bands_above'):
        print(f'  │  ── φ levels ABOVE entry (targets) ───────────────────────')
        for band_lbl, band_lvl in rev['phi_bands_above']:
            print(f'  │    {band_lbl:<32}  {pp(band_lvl)}')
    if rev.get('phi_bands_below'):
        print(f'  │  ── φ levels BELOW entry (support / stop ref) ─────────────')
        for band_lbl, band_lvl in rev['phi_bands_below']:
            print(f'  │    {band_lbl:<32}  {pp(band_lvl)}')
    print(f'  └{"─"*w}┘')
    print()

    # ── Golden Triangle targets ───────────────────────────────────
    print(f'  ┌─ GOLDEN TRIANGLE TARGETS  (apex=36°  base=72°  leg/base=φ) {"─"*4}┐')
    print(f'  │  T1 primary  (apex height)      : {pp(rev.get("target_T1"))}')
    print(f'  │  T2 gnomon   (φ² × bar range)   : {pp(rev.get("target_T2"))}')
    print(f'  └{"─"*w}┘')
    print()

    # ── Golden Spiral timing ──────────────────────────────────────
    spi_str = '✔  YES — price bar is inside a φ-spiral timing window' \
              if spiral_ok else '✗  No   — bar outside spiral windows'
    print(f'  ┌─ GOLDEN SPIRAL TIMING  (r = A·e^(b·θ), b = ln(φ)/(π/2)) {"─"*4}┐')
    print(f'  │  Spiral windows from argmin/argmax anchor:')
    anchor_bar = rev.get('argmin_bar') if trend == 'UP' else rev.get('argmax_bar')
    if anchor_bar is not None:
        wins = _spiral_windows(int(anchor_bar), n=7)
        wins_str = ', '.join(str(w2) for w2 in wins)
        print(f'  │    anchor bar {anchor_bar} → windows at [{wins_str}]')
    print(f'  │  Current in window : {spi_str}')
    print(f'  └{"─"*w}┘')
    print()

    # ── Individual reversal signals from argrelextrema ────────────
    sigs = rev.get('all_signals', [])
    if sigs:
        print(f'  ┌─ φ-REVERSAL SIGNALS  ({len(sigs)} signals, conf≥{0.25}) {"─"*24}┐')
        hdr_s = (f'  │  {"Bar":>5}  {"Price":>12}  {"Dir":>4}  '
                 f'{"φ Band":<24}  {"T1 target":>12}  {"T2 target":>12}  {"Conf":>6}  │')
        sep_s = '  │' + '─' * (len(hdr_s) - 4) + '│'
        print(sep_s); print(hdr_s); print(sep_s)
        # show top-5 by confidence
        top5 = sorted(sigs, key=lambda s: s['confidence'], reverse=True)[:5]
        for s in top5:
            sp_tag = '✔' if s['spiral_ok'] else '·'
            print(f'  │  {s["bar"]:>5}  {pf(s["price"]):>12}  {s["direction"]:>4}  '
                  f'{s["phi_band_label"][:24]:<24}  '
                  f'{pf(s["T1_primary"]):>12}  '
                  f'{pf(s["T2_gnomon"]):>12}  '
                  f'{s["confidence"]:>5.3f}{sp_tag}  │')
        print(sep_s)
        print(f'  └{"─"*w}┘')
        print()

    # ── FINAL REVERSAL SUMMARY ────────────────────────────────────
    fc_p  = rev.get('forecast_price')
    t1_p  = rev.get('target_T1')
    t2_p  = rev.get('target_T2')

    if fc_p:
        fc_pct = (fc_p - cp) / (cp + 1e-20) * 100.0
        fc_arr = '▲' if fc_pct > 0 else '▼'
    else:
        fc_pct = 0.0; fc_arr = ''

    print(f'  {"═"*w}')
    print(f'  ★★★  φ-REVERSAL DECISION  ·  {lbl}')
    print(f'  {"═"*w}')
    print(f'  ▶  ENTRY               : {pf(cp)}')
    print(f'  ▶  TREND               : {trend_icon}')
    print()
    print(f'  ▶  φ FORECAST TARGET   : {pf(fc_p)}  '
          f'({fc_arr}{fc_pct:+.2f}%)  [median of nearest φ bands]')
    print(f'  ▶  T1  (triangle)      : {pp(t1_p)}  [apex·height = bar_range × {_GT_HEIGHT_MULT:.4f}]')
    print(f'  ▶  T2  (φ² gnomon)    : {pp(t2_p)}  [bar_range × φ² = {PHI2:.4f}]')
    print()
    print(f'  ▶  CONFIDENCE          : {conf:.4f}  [{conf_bar}]')
    print(f'     φ band proximity    : {rev.get("phi_score",0.0):.4f} × 0.50')
    print(f'     spiral timing       : {"1.00" if spiral_ok else "0.00"} × 0.30')
    print(f'     extremum score      : (from argrelextrema signals)')
    print()

    # grade
    if   conf >= 0.75: grade = 'STRONG REVERSAL SETUP ★★★'
    elif conf >= 0.55: grade = 'MODERATE SETUP ★★'
    elif conf >= 0.35: grade = 'WEAK SETUP ★'
    else:              grade = 'LOW CONFIDENCE — wait for better alignment'
    print(f'  ▶  GRADE               : {grade}')
    print()
    print(f'  {"═"*w}\n')


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

print(f'\n  MTF Dip Scanner + FFT Forecast + φ·e·π Time Geometry')
print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'  All USDC pairs  |  {MAX_WORKERS} threads  |  retry {LOOP_SLEEP}s')
print(f'  φ={PHI:.4f}  e={E:.4f}  b={GOLDEN_B:.5f}  φ∠={PHI_ANGLE:.2f}°\n')

trading_pairs, label_map = trader.get_usdc_pairs()
print(f'  {len(trading_pairs)} USDC pairs loaded\n')

iteration = 0

while True:
    iteration += 1
    print(f'  ══ Iteration {iteration}  ·  {datetime.now().strftime("%H:%M:%S")} ══\n')

    # ── 2h ──────────────────────────────────────────────────
    fp1 = run_stage(filter1, trading_pairs, '2h ')
    print(f'  2h  → {len(fp1)} passed')
    print_stage_table(fp1, label_map, '2h filter', show_cmo=True)

    if not fp1:
        gc.collect()
        print(f'  Nothing passed 2h. Retry in {LOOP_SLEEP}s\n')
        time.sleep(LOOP_SLEEP); continue

    # ── 30m ─────────────────────────────────────────────────
    fp1b = run_stage(filter1b, fp1, '30m')
    print(f'  30m → {len(fp1b)} passed')
    print_stage_table(fp1b, label_map, '30m filter', show_cmo=True)

    if not fp1b:
        del fp1, fp1b; gc.collect()
        print(f'  Nothing passed 30m. Retry in {LOOP_SLEEP}s\n')
        time.sleep(LOOP_SLEEP); continue

    # ── 15m ─────────────────────────────────────────────────
    fp2 = run_stage(filter2, fp1b, '15m')
    print(f'  15m → {len(fp2)} passed')
    print_stage_table(fp2, label_map, '15m filter', show_cmo=True)

    if not fp2:
        del fp1, fp1b, fp2; gc.collect()
        print(f'  Nothing passed 15m. Retry in {LOOP_SLEEP}s\n')
        time.sleep(LOOP_SLEEP); continue

    # ── 5m ──────────────────────────────────────────────────
    fp3 = run_stage(filter3, fp2, '5m ')
    print(f'  5m  → {len(fp3)} passed')
    print_stage_table(fp3, label_map, '5m filter', show_cmo=True)

    if not fp3:
        del fp1, fp1b, fp2, fp3; gc.collect()
        print(f'  Nothing passed 5m. Retry in {LOOP_SLEEP}s\n')
        time.sleep(LOOP_SLEEP); continue

    # ── multi-TF argmin confirmation (1m AND 3m AND 5m) ──────
    #  Passes only if argmin > argmax on ALL three TFs.
    #  This ensures the most recent price extreme is the LOW
    #  (not the high) across every short timeframe simultaneously.
    print(f'  Running multi-TF argmin check on {len(fp3)} pairs...')
    fp4, thr_map = run_multi_tf_argmin_stage(fp3)
    print(f'  multi-TF argmin → {len(fp4)} passed (1m∧3m∧5m argmin>argmax)')
    print_multi_tf_threshold_table(fp4, label_map, thr_map)

    if not fp4:
        del fp1, fp1b, fp2, fp3, fp4, thr_map; gc.collect()
        print(f'  Nothing passed multi-TF argmin. Retry in {LOOP_SLEEP}s\n')
        time.sleep(LOOP_SLEEP); continue

    # ── 1m dip confirmation gate ─────────────────────────────
    sel_pairs  = []
    sel_cmo    = []
    sel_detail = {}
    sel_lock   = threading.Lock()
    total_1m   = len(fp4)
    done_1m    = [0]

    def _mom(sym):
        momentum(sym, sel_pairs, sel_cmo, sel_detail, sel_lock)
        with sel_lock:
            done_1m[0] += 1
            pct = int(done_1m[0] / total_1m * 100)
            bar = '█' * (pct // 4) + '░' * (25 - pct // 4)
            print(f'\r  1m  [{bar}] {pct:3d}%  {done_1m[0]}/{total_1m}',
                  end='', flush=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        fs = [pool.submit(_mom, s) for s in fp4]
        for f in as_completed(fs): pass
    print()

    # ── full diagnostic ──────────────────────────────────────
    print(f'\n  [1m dip diagnostic]  {len(fp4)} candidates'
          f'  (conditions: bull%>bear%  AND  argmin>argmax)\n')
    diag_header = (f'  {"Ticker":<12}  {"Price":>10}  '
                   f'{"Bull%":>6}  {"Bear%":>6}  '
                   f'{"ArgMin":>7}  {"ArgMax":>7}  '
                   f'{"CMO":>7}  {"Geo":>5}  '
                   f'{"Z":>6}  {"p-val":>6}  {"Curv":>5}  {"Fake?":>5}  {"Result"}')
    print(diag_header)
    print('  ' + '─' * (len(diag_header) - 2 + 20))
    for p in fp4:
        d    = sel_detail.get(p, {})
        lbl  = label_map.get(p, p.replace('USDC', ''))
        pr_v = d.get('price')
        pr_s = (f'{pr_v:.6f}' if pr_v and pr_v < 1 else f'{pr_v:.4f}') \
               if pr_v else '—'
        bull     = d.get('bull_pct')
        bear     = d.get('bear_pct')
        amin     = d.get('argmin_idx')
        amax     = d.get('argmax_idx')
        raw_cmo  = d.get('raw_cmo')
        cond_vol = d.get('cond_vol')
        cond_ext = d.get('cond_ext')
        geo_s    = d.get('geometry_score', 0.0)

        # φ Z-score + p-value + curvature fields (from geometry_detail sub-dict)
        geo_det   = d.get('geometry_detail', {})
        z_val     = geo_det.get('z_score')
        p_val     = geo_det.get('p_value')
        curv_val  = geo_det.get('curvature')
        fake_flag = geo_det.get('is_fake_dip', False)

        if not d:
            result = 'no data'
        elif cond_vol and cond_ext:
            result = f'PASS ✔  (CMO={raw_cmo}  bull>{bear:.1f}%  min@{amin}>max@{amax})'
        else:
            reasons = []
            if not cond_vol:
                reasons.append(f'bear vol dominant ({bear:.1f}%>{bull:.1f}%)')
            if not cond_ext:
                reasons.append(f'maxima more recent (argmax={amax}>argmin={amin})')
            result = 'fail  ' + '  |  '.join(reasons)

        if fake_flag:
            result += '  ⚠ FAKE DIP (curv↓)'

        bull_s  = f'{bull:.1f}'      if bull      is not None else '—'
        bear_s  = f'{bear:.1f}'      if bear      is not None else '—'
        amin_s  = f'{amin}'          if amin      is not None else '—'
        amax_s  = f'{amax}'          if amax      is not None else '—'
        cmo_s   = f'{raw_cmo:.1f}'   if raw_cmo   is not None else '—'
        geo_str = f'{geo_s:.0f}'     if geo_s     else '—'
        z_s     = f'{z_val:+.2f}'    if z_val     is not None else '—'
        p_s     = f'{p_val:.3f}'     if p_val     is not None else '—'
        curv_s  = f'{curv_val:+.4f}' if curv_val  is not None else '—'
        fake_s  = '⚠ YES' if fake_flag else 'no'
        print(f'  {lbl:<12}  {pr_s:>10}  '
              f'{bull_s:>6}  {bear_s:>6}  '
              f'{amin_s:>7}  {amax_s:>7}  '
              f'{cmo_s:>7}  {geo_str:>5}  '
              f'{z_s:>6}  {p_s:>6}  {curv_s:>5}  {fake_s:>5}  {result}')
    print()

    print(f'  1m  → {len(sel_pairs)} passed (bull%>bear% AND argmin>argmax)')
    print_stage_table(sel_pairs, label_map,
                      '1m confirmed dips', show_cmo=True)

    # ── selection: rank by composite score (CMO × geometry) ──
    if len(sel_pairs) > 1:
        lbls = [label_map.get(p, p) for p in sel_pairs]
        print(f'  {len(sel_pairs)} mtf dips found: {lbls}')
        print(f'  Ranking by: raw CMO (most negative = deepest oversold)')
        print(f'  Secondary:  φ·e·π geometry score (structural quality)')

        # composite rank: normalize CMO (most negative = best) and geo score
        scored = []
        for i, p in enumerate(sel_pairs):
            cmo_v = sel_cmo[i] or 0.0
            geo_v = sel_detail[p].get('geometry_score', 0.0)
            # lower CMO = better; higher geo = better
            # composite = -cmo_normalized + geo_normalized
            scored.append((p, cmo_v, geo_v))

        # rank by CMO primarily (most negative), then by geo
        scored.sort(key=lambda x: (x[1], -x[2]))
        best_symbol = scored[0][0]
        best_d      = sel_detail[best_symbol]
        best_geo    = best_d.get('geometry_detail', {})
        best_z      = best_geo.get('z_score');    z_s  = f'{best_z:+.2f}' if best_z  is not None else '—'
        best_p      = best_geo.get('p_value');    p_s  = f'{best_p:.3f}'  if best_p  is not None else '—'
        best_fake   = best_geo.get('is_fake_dip', False)
        fake_warn   = '  ⚠ FAKE DIP (curv↓)' if best_fake else ''
        print(f'  Best → {label_map.get(best_symbol, best_symbol)}'
              f'  CMO={best_d["raw_cmo"]}'
              f'  bull={best_d["bull_pct"]}%'
              f'  geo={best_d["geometry_score"]:.0f}/100'
              f'  Z={z_s}  p={p_s}'
              f'  argmin@{best_d["argmin_idx"]}>argmax@{best_d["argmax_idx"]}'
              f'{fake_warn}\n')

    elif len(sel_pairs) == 1:
        best_symbol = sel_pairs[0]
        best_d      = sel_detail[best_symbol]
        best_geo    = best_d.get('geometry_detail', {})
        best_z      = best_geo.get('z_score');    z_s  = f'{best_z:+.2f}' if best_z  is not None else '—'
        best_p      = best_geo.get('p_value');    p_s  = f'{best_p:.3f}'  if best_p  is not None else '—'
        best_fake   = best_geo.get('is_fake_dip', False)
        fake_warn   = '  ⚠ FAKE DIP (curv↓)' if best_fake else ''
        print(f'  1 mtf dip found: {label_map.get(best_symbol, best_symbol)}'
              f'  CMO={best_d["raw_cmo"]}'
              f'  bull={best_d["bull_pct"]}%'
              f'  geo={best_d["geometry_score"]:.0f}/100'
              f'  Z={z_s}  p={p_s}'
              f'  argmin@{best_d["argmin_idx"]}>argmax@{best_d["argmax_idx"]}'
              f'{fake_warn}\n')

    else:
        print(f'  No MTF dips confirmed (bull%>bear% AND argmin>argmax — none passed).')
        del fp1, fp1b, fp2, fp3, fp4, thr_map, sel_pairs, sel_cmo, sel_detail
        gc.collect()
        print(f'  GC done. Retry in {LOOP_SLEEP}s\n')
        print(f'  {"·" * 62}\n')
        time.sleep(LOOP_SLEEP); continue

    # ── FFT forecast on winner ────────────────────────────────
    current_price = sel_detail[best_symbol].get('price')
    if current_price:
        lbl = label_map.get(best_symbol, best_symbol)
        print(f'  Running FFT + Time Geometry forecast on {lbl}...')
        stf_results, stf_best, htf_results, htf_best = \
            full_fft_report(best_symbol, current_price)

        if stf_results or htf_results:
            # standard FFT report
            print_fft_report(best_symbol, label_map,
                             stf_results, stf_best,
                             htf_results, htf_best)

            # φ·e·π time geometry report
            run_time_geometry(
                best_symbol, label_map, current_price, sel_detail,
                stf_results, htf_results
            )

            # ML compound forecast
            print(f'  Running ML compound forecast on {lbl}...')
            ml_result = ml_compound_forecast(
                best_symbol, current_price, sel_detail,
                stf_results, htf_results, thr_map=thr_map
            )
            print_ml_report(ml_result, label_map)

            # ── φ-Reversal forecast (argmin/argmax + φ decay bands
            #    + golden triangle + golden spiral timing) ─────────
            print(f'  Running φ-Reversal forecast on {lbl}...')
            phi_rev = phi_reversal_forecast(
                best_symbol, current_price, sel_detail,
                order=7, phi_band_tol_pct=1.2, min_confidence=0.25
            )
            print_phi_reversal_block(phi_rev, label_map)

            # ── 360° Sinusoidal Circuit Engine ───────────────────
            #  Builds the full circuit model between argmin/argmax,
            #  maps current price to circuit angle + quadrant, and
            #  forecasts the incoming reversal target (support dip
            #  or resistance top) with volume cause-effect rules.
            #  Also runs across ALL timeframes and produces MTF ML targets.
            print(f'  Running 360° Sinusoidal Circuit Engine on {lbl}...')
            circ = sinusoidal_circuit_engine(
                best_symbol, current_price, sel_detail,
                stf_results=stf_results, htf_results=htf_results
            )
            # ── MTF multi-timeframe circuit ───────────────────────
            print(f'  Running MTF Sinusoidal Circuit across all timeframes...')
            mtf_tf_results, mtf_summary = sinusoidal_circuit_mtf(
                best_symbol, current_price, sel_detail,
                stf_results=stf_results, htf_results=htf_results
            )
            print_circuit_block(circ, label_map,
                                 mtf_tf_results=mtf_tf_results,
                                 mtf_summary=mtf_summary)
        else:
            print('  FFT: insufficient data for forecast.\n')
    else:
        print('  Could not fetch current price for FFT.\n')

    del fp1, fp1b, fp2, fp3, fp4, thr_map, sel_pairs, sel_cmo, sel_detail
    gc.collect()
    sys.exit(0)
