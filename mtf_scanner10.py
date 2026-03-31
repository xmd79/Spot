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
FFT_CANDLES      = 600

# ─────────────────────────────────────────────
#  TIME GEOMETRY CONSTANTS  (φ · e · π)
# ─────────────────────────────────────────────
PHI       = (1.0 + np.sqrt(5.0)) / 2.0
PHI_INV   = 1.0 / PHI
PHI2      = PHI ** 2
PHI_SQRT  = np.sqrt(PHI)
GOLDEN_B  = np.log(PHI) / (np.pi / 2.0)
PHI_ANGLE = 360.0 / PHI2
E         = np.e
TAU       = 2.0 * np.pi

CIRCUIT_QUADS = {
    'Q1': (  0.0,  90.0),
    'Q2': ( 90.0, 180.0),
    'Q3': (180.0, 270.0),
    'Q4': (270.0, 360.0),
}
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
_ABS_LOG_THRESH  = 5.0
_EXHS_LOG_THRESH = 1.5

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
#  CLIENT
# ─────────────────────────────────────────────
class Trader:
    def __init__(self, file):
        lines = [l.rstrip('\n') for l in open(file)]
        self.client = Client(lines[0], lines[1])

    _EXCLUDED_BASES = {
        'BFUSD', 'FDUSD', 'TUSD', 'USDP', 'USDS', 'DAI', 'FRAX',
        'LUSD', 'USTC', 'EURS', 'EURT', 'AEUR',
        'BBTC', 'BETH', 'BBNB', 'LDBNB', 'WBETH',
        'LDETH', 'LDBTC', 'LDUSDT', 'LDBUSD',
    }

    def validate_usdc_pair(self, symbol):
        """
        Validate that a symbol is a genuine USDC spot trading pair.
        Returns (is_valid, base_asset, current_price) or (False, None, None).
        """
        try:
            info = self.client.get_exchange_info()
        except Exception as ex:
            print(f'  [ERROR] Cannot fetch exchange info: {ex}')
            return False, None, None

        for s in info['symbols']:
            if s['symbol'] != symbol:
                continue
            if s['quoteAsset'] != 'USDC':
                print(f'  [INVALID] {symbol} — quote asset is {s["quoteAsset"]}, not USDC')
                return False, None, None
            if s['status'] != 'TRADING':
                print(f'  [INVALID] {symbol} — status is {s["status"]}, not TRADING')
                return False, None, None
            if not s.get('isSpotTradingAllowed', False):
                print(f'  [INVALID] {symbol} — spot trading not allowed')
                return False, None, None

            perms     = s.get('permissions', [])
            perm_sets = s.get('permissionSets', [])
            flat_sets = [p for sub in perm_sets for p in sub]
            if 'SPOT' not in perms and 'SPOT' not in flat_sets:
                print(f'  [INVALID] {symbol} — not in SPOT permission set')
                return False, None, None

            base = s['baseAsset']
            if base in self._EXCLUDED_BASES:
                print(f'  [INVALID] {symbol} — {base} is an excluded/synthetic asset')
                return False, None, None

            # Fetch live price
            try:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                price  = float(ticker['price'])
            except Exception:
                price = None

            if price is not None and 0.995 <= price <= 1.005:
                print(f'  [INVALID] {symbol} — looks like a stablecoin pair (price ≈ $1)')
                return False, None, None

            return True, base, price

        print(f'  [INVALID] {symbol} — symbol not found on Binance')
        return False, None, None

    def list_usdc_pairs(self, search=''):
        """Return list of valid USDC spot pairs, optionally filtered by search string."""
        try:
            info  = self.client.get_exchange_info()
            tickers = {t['symbol']: float(t['price'])
                       for t in self.client.get_all_tickers()}
        except Exception:
            return []

        pairs = []
        for s in info['symbols']:
            if s['quoteAsset'] != 'USDC':      continue
            if s['status']     != 'TRADING':   continue
            if not s.get('isSpotTradingAllowed', False): continue
            perms     = s.get('permissions', [])
            perm_sets = s.get('permissionSets', [])
            flat_sets = [p for sub in perm_sets for p in sub]
            if 'SPOT' not in perms and 'SPOT' not in flat_sets: continue
            base = s['baseAsset']
            if base in self._EXCLUDED_BASES: continue
            sym = s['symbol']
            price = tickers.get(sym)
            if price is not None and 0.995 <= price <= 1.005: continue
            if search.upper() in sym:
                pairs.append(sym)
        return sorted(pairs)


trader = Trader(CREDENTIALS_FILE)


# ─────────────────────────────────────────────
#  CORE ANALYSIS FUNCTIONS (unchanged from original)
# ─────────────────────────────────────────────

def _channel_pass(klines):
    close = [float(e[4]) for e in klines]
    if not close:
        return False, []
    x       = np.array(close, dtype=np.float64)
    period  = min(500, len(x))
    midline = ta.LINEARREG(x, timeperiod=period)
    valid   = ~np.isnan(midline)
    if not np.any(valid):
        return False, close
    x_v     = x[valid]
    m_v     = midline[valid]
    std     = np.std(x_v - m_v)
    lower   = m_v - std
    return float(x_v[-1]) < float(lower[-1]), close


def get_real_volume_flow(trader, pair, limit=1000):
    try:
        trades = trader.client.get_aggregate_trades(symbol=pair, limit=limit)
    except Exception:
        return None
    if not trades:
        return None
    buy_vol = 0.0; sell_vol = 0.0
    prices = []; qtys = []
    for t in trades:
        qty = float(t['q']); price = float(t['p'])
        prices.append(price); qtys.append(qty)
        if t['m']:   sell_vol += qty
        else:        buy_vol  += qty
    total = buy_vol + sell_vol
    if total == 0: return None
    delta = buy_vol - sell_vol
    delta_ratio = delta / total
    price_range = max(prices) - min(prices) + 1e-12
    total_volume = sum(qtys)
    absorption_score = np.log1p(total_volume / price_range)
    prices_arr = np.array(prices); qtys_arr = np.array(qtys)
    mid = len(prices_arr) // 2
    early_move = abs(prices_arr[mid] - prices_arr[0]) + 1e-12
    late_move  = abs(prices_arr[-1] - prices_arr[mid]) + 1e-12
    early_vol  = np.sum(qtys_arr[:mid]) + 1e-12
    late_vol   = np.sum(qtys_arr[mid:]) + 1e-12
    vol_ratio  = late_vol / early_vol
    move_ratio = late_move / early_move
    exhaustion_score = np.log1p(vol_ratio / (move_ratio + 1e-12))
    return {
        'buy_vol': buy_vol, 'sell_vol': sell_vol,
        'delta': delta, 'delta_ratio': delta_ratio,
        'absorption': absorption_score, 'exhaustion': exhaustion_score
    }


EXTREMA_LOOKBACK = 1000

def check_dip_conditions(pair):
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

    flow = get_real_volume_flow(trader, pair)
    if flow:
        buy_vol  = flow['buy_vol']; sell_vol = flow['sell_vol']
        total_vol_real = buy_vol + sell_vol
        bull_ratio = buy_vol / total_vol_real
        bear_ratio = sell_vol / total_vol_real
        cond_vol = (
            bull_ratio > 0.5 or
            flow['absorption'] > 5.0 or
            flow['delta_ratio'] > 0.1
        )
        absorption_score = flow['absorption']
        exhaustion_score = flow['exhaustion']
        delta_ratio      = flow['delta_ratio']
    else:
        bull_mask  = close >= open_
        bull_vol   = float(vol[bull_mask].sum())
        total_vol  = float(vol.sum())
        if total_vol == 0: return False, {}
        bull_ratio = bull_vol / total_vol
        bear_ratio = 1.0 - bull_ratio
        cond_vol   = bull_ratio > 0.5
        absorption_score = 0.0; exhaustion_score = 0.0; delta_ratio = 0.0

    argmin_idx = int(np.argmin(low_))
    argmax_idx = int(np.argmax(high_))
    cond_ext   = argmin_idx > argmax_idx
    cmo_arr    = ta.CMO(close, timeperiod=14)
    raw_cmo    = float(cmo_arr[-1]) if not np.isnan(cmo_arr[-1]) else None
    geo_score, geo_detail = _phi_e_pi_dip_score(close, float(close[-1]))

    detail = {
        'bull_pct':         round(bull_ratio * 100.0, 1),
        'bear_pct':         round(bear_ratio * 100.0, 1),
        'argmin_idx':       argmin_idx,
        'argmax_idx':       argmax_idx,
        'raw_cmo':          round(raw_cmo, 2) if raw_cmo is not None else None,
        'price':            round(float(close[-1]), 8),
        'cond_vol':         cond_vol,
        'cond_ext':         cond_ext,
        'geometry_score':   geo_score,
        'geometry_detail':  geo_detail,
        'close_arr':        close,
        'low_arr':          low_,
        'high_arr':         high_,
        'swing_low':        float(np.min(low_)),
        'swing_high':       float(np.max(high_)),
        'delta_ratio':      round(delta_ratio, 4),
        'absorption_score': round(absorption_score, 4),
        'exhaustion_score': round(exhaustion_score, 4),
    }
    passed = cond_vol and cond_ext
    return passed, detail


def _detrend(arr):
    t = np.arange(len(arr))
    p = np.polyfit(t, arr, 1)
    return arr - np.poly1d(p)(t), p


def phi_extension_levels(swing_low, swing_high, current_price):
    rng = swing_high - swing_low
    if rng <= 0 or swing_low <= 0: return []
    levels = []
    for ratio, label in FIB_RATIOS:
        level = swing_low + ratio * rng
        if level > current_price * 1.001:
            levels.append((ratio, label, round(float(level), 8)))
    return sorted(levels, key=lambda x: x[2])


def hilbert_cycle_analysis(close_arr):
    try:
        arr = np.asarray(close_arr, dtype=np.float64)
        if len(arr) < 32: return None
        ht_period  = ta.HT_DCPERIOD(arr)
        ht_phase   = ta.HT_DCPHASE(arr)
        sine, lead = ta.HT_SINE(arr)
        inph, quad = ta.HT_PHASOR(arr)
        trend_mode = ta.HT_TRENDMODE(arr)
        def last(x):
            v = x[~np.isnan(x)]
            return float(v[-1]) if len(v) > 0 else None
        lp = last(ht_period); lph = last(ht_phase)
        ls = last(sine);      ll  = last(lead)
        li = last(inph);      lq  = last(quad)
        lt = last(trend_mode)
        amplitude = float(np.sqrt(li**2 + lq**2)) if (li and lq) else None
        bars_to_trough = None
        if lph is not None and lp is not None:
            trough_phase   = 270.0
            phase_to_go    = (trough_phase - lph) % 360.0
            bars_to_trough = round(phase_to_go / 360.0 * lp, 1)
        in_buy_zone = (ls is not None and ll is not None and ls < 0 and ls > ll)
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


def complex_exp_forecast(close_arr, current_price, bars_forward=None):
    try:
        arr = np.asarray(close_arr, dtype=np.float64)
        n   = len(arr)
        if n < 32: return None
        detrended, trend_coeffs = _detrend(arr)
        inph, quad = ta.HT_PHASOR(arr)
        valid      = ~(np.isnan(inph) | np.isnan(quad))
        if np.sum(valid) < 20: return None
        envelope  = np.sqrt(inph[valid]**2 + quad[valid]**2)
        t_valid   = np.arange(n)[valid].astype(np.float64)
        log_env   = np.log(np.maximum(envelope, 1e-20))
        coeffs    = np.polyfit(t_valid, log_env, 1)
        alpha     = float(coeffs[0]); A0 = float(np.exp(coeffs[1]))
        pred      = np.poly1d(coeffs)(t_valid)
        ss_res    = float(np.sum((log_env - pred)**2))
        ss_tot    = float(np.sum((log_env - log_env.mean())**2))
        r2        = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        ht_phase_arr  = ta.HT_DCPHASE(arr)
        ht_period_arr = ta.HT_DCPERIOD(arr)
        valid_ph  = ~np.isnan(ht_phase_arr)
        if np.sum(valid_ph) < 10: return None
        valid_pe  = ~np.isnan(ht_period_arr)
        ht_period_now = float(ht_period_arr[valid_pe][-1]) if np.sum(valid_pe) > 0 else 20.0
        if bars_forward is None:
            bars_forward = max(5, int(round(ht_period_now / 2)))
        phase_arr = np.deg2rad(ht_phase_arr[valid_ph])
        phase_uw  = np.unwrap(phase_arr)
        omega     = float(np.mean(np.diff(phase_uw))) if len(phase_uw) > 1 else 0.0
        period_est = abs(TAU / omega) if abs(omega) > 1e-10 else None
        t_now     = float(n - 1)
        A_now     = A0 * np.exp(alpha * t_now)
        phase_now = float(phase_uw[-1]) if len(phase_uw) > 0 else 0.0
        t_fwd     = t_now + bars_forward
        A_fwd     = A0 * np.exp(alpha * t_fwd)
        phase_fwd = phase_now + omega * bars_forward
        trend_fwd = float(np.poly1d(trend_coeffs)(t_fwd))
        forecast  = trend_fwd + A_fwd * np.cos(phase_fwd)
        if   alpha >  1e-6: e_label = 'growing oscillation (e^+ α) — expansion'
        elif alpha < -1e-6: e_label = 'decaying oscillation (e^- α) — compression'
        else:               e_label = 'neutral amplitude'
        return {
            'alpha': round(alpha, 6), 'omega_rad': round(omega, 6),
            'period_est': round(period_est, 1) if period_est else None,
            'A0': round(A0, 8), 'A_now': round(A_now, 8),
            'bars_forward': bars_forward,
            'forecast': round(float(forecast), 8),
            'fit_r2': round(max(0.0, r2), 4), 'e_label': e_label,
        }
    except Exception:
        return None


def golden_spiral_targets(current_price, ht_phase_deg, ht_amplitude,
                           swing_low=None, swing_high=None):
    if ht_phase_deg is None: return None
    theta0 = np.deg2rad(ht_phase_deg)
    if swing_low is not None and swing_high is not None and swing_high > swing_low:
        A_base = (swing_high - swing_low) / 2.0
    elif ht_amplitude and ht_amplitude > 0:
        A_base = max(float(ht_amplitude), current_price * 0.005)
    else:
        A_base = current_price * 0.01
    C = A_base / (np.exp(GOLDEN_B * theta0) + 1e-20)
    def spiral_r(delta_theta):
        theta = theta0 + delta_theta
        return float(C * np.exp(GOLDEN_B * theta))
    r0     = spiral_r(0); r_q1   = spiral_r(np.pi / 2)
    r_half = spiral_r(np.pi); r_3q = spiral_r(3 * np.pi / 2)
    r_full = spiral_r(TAU)
    def price_target(r):
        return round(current_price + (r - r0), 8)
    return {
        'current_angle': round(np.rad2deg(theta0) % 360.0, 1),
        'A_base':        round(A_base, 8),
        'gs_q1_turn':    price_target(r_q1),
        'gs_half_turn':  price_target(r_half),
        'gs_3q_turn':    price_target(r_3q),
        'gs_full_turn':  price_target(r_full),
        'phi_mult_q1':   round(r_q1  / r0, 4) if r0 > 1e-20 else None,
        'phi_mult_half': round(r_half / r0, 4) if r0 > 1e-20 else None,
    }


def _phi_e_pi_dip_score(close_arr, current_price):
    try:
        from scipy.stats import norm as _norm
    except ImportError:
        _norm = None
    try:
        arr = np.asarray(close_arr, dtype=np.float64)
        n   = len(arr)
        if n < 20: return 0.0, {}
        period   = min(500, n)
        midline  = ta.LINEARREG(arr, timeperiod=period)
        valid_ml = ~np.isnan(midline)
        if not np.any(valid_ml): return 0.0, {}
        x_v       = arr[valid_ml]; m_v = midline[valid_ml]
        sigma     = float(np.std(x_v - m_v)) + 1e-20
        trend_now = float(m_v[-1]); lower_band = trend_now - sigma
        z_score   = (lower_band - current_price) / sigma
        phi_devs  = max(0.0, z_score)
        phi_score = min(40.0, phi_devs * 10.0 * PHI)
        p_value   = float(_norm.cdf(z_score)) if _norm is not None else None
        curv_period  = min(10, n // 2)
        slope1       = ta.LINEARREG_SLOPE(arr,    timeperiod=curv_period)
        slope2       = ta.LINEARREG_SLOPE(slope1, timeperiod=curv_period)
        valid_s2     = ~np.isnan(slope2)
        curvature_now = float(slope2[valid_s2][-1]) if np.any(valid_s2) else 0.0
        curvature_ok  = curvature_now > 0.0
        curv_penalty  = 0.0 if curvature_ok else min(20.0, abs(curvature_now / sigma) * 10.0)
        is_fake_dip   = (not curvature_ok) and (curv_penalty >= 10.0)
        look   = min(30, n); last_n = arr[-look:]
        t_n    = np.arange(look, dtype=np.float64)
        log_last  = np.log(np.maximum(last_n, 1e-20))
        coeffs_e  = np.polyfit(t_n, log_last, 1)
        pred_e    = np.poly1d(coeffs_e)(t_n)
        ss_res    = float(np.sum((log_last - pred_e)**2))
        ss_tot    = float(np.sum((log_last - log_last.mean())**2))
        r2_exp    = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        decay_r   = float(coeffs_e[0])
        e_score   = r2_exp * (30.0 if decay_r < 0 else 12.0)
        ht_ph_arr = ta.HT_DCPHASE(arr)
        valid_ph  = ~np.isnan(ht_ph_arr)
        pi_score  = 0.0; ht_phase_now = None
        if np.sum(valid_ph) > 0:
            ht_phase_now = float(ht_ph_arr[valid_ph][-1])
            dist = abs(ht_phase_now - 270.0)
            dist = min(dist, 360.0 - dist)
            pi_score = max(0.0, 30.0 * (1.0 - dist / 180.0))
        raw_total = phi_score + e_score + pi_score
        total     = round(max(0.0, raw_total - curv_penalty), 1)
        detail = {
            'phi_devs': round(phi_devs, 3), 'z_score': round(z_score, 3),
            'p_value':  round(p_value, 4) if p_value is not None else None,
            'phi_score': round(phi_score, 1),
            'curvature': round(curvature_now, 6), 'curvature_ok': curvature_ok,
            'curv_penalty': round(curv_penalty, 1), 'is_fake_dip': is_fake_dip,
            'e_r2': round(r2_exp, 3), 'e_decay_rate': round(decay_r, 6),
            'e_score': round(e_score, 1),
            'ht_phase_now': round(ht_phase_now, 1) if ht_phase_now is not None else None,
            'pi_score': round(pi_score, 1), 'total': total,
        }
        return total, detail
    except Exception:
        return 0.0, {}


def mtf_harmony_score(stf_results, htf_results):
    all_r = (stf_results or []) + (htf_results or [])
    if len(all_r) < 2: return 0.0, {}
    periods   = [r['dominant_period'] for r in all_r]
    forecasts = [r['forecast']        for r in all_r]
    upsides   = [r['upside_pct']      for r in all_r]
    phi_pairs = 0; total_pairs = 0
    phi_ratio_targets = [PHI, PHI2, 2.0, 3.0, PHI * 0.5, PHI_INV]
    for i in range(len(periods) - 1):
        p1, p2 = periods[i], periods[i + 1]
        if p1 > 0 and p2 > 0:
            ratio = max(p1, p2) / min(p1, p2)
            for target in phi_ratio_targets:
                if abs(ratio - target) / target < 0.20:
                    phi_pairs += 1; break
            total_pairs += 1
    phi_h = (phi_pairs / total_pairs * 30.0) if total_pairs > 0 else 0.0
    fc_arr = np.array(forecasts, dtype=np.float64)
    fc_mean = float(np.mean(fc_arr)); fc_std = float(np.std(fc_arr))
    fc_cv   = fc_std / fc_mean if fc_mean > 1e-20 else 1.0
    e_h     = max(0.0, 30.0 * (1.0 - min(1.0, fc_cv * 20.0)))
    n_up    = sum(1 for u in upsides if u > 0)
    pi_h    = (n_up / len(upsides)) * 20.0
    no_stop_bonus = 0.0
    if htf_results:
        n_stopped = sum(1 for r in htf_results if r.get('cascade_stop'))
        if n_stopped == 0 and len(htf_results) >= 3: no_stop_bonus = 20.0
        elif n_stopped == 0:                          no_stop_bonus = 10.0
    raw   = phi_h + e_h + pi_h + no_stop_bonus
    score = round(min(100.0, raw), 1)
    detail = {
        'phi_pairs': phi_pairs, 'total_pairs': total_pairs,
        'phi_harmony': round(phi_h, 1), 'fc_cv_pct': round(fc_cv * 100.0, 2),
        'e_harmony': round(e_h, 1), 'n_up': n_up, 'n_total': len(upsides),
        'pi_harmony': round(pi_h, 1), 'no_stop_bonus': round(no_stop_bonus, 1),
        'harmony': score,
    }
    return score, detail


def fft_analysis(close_list, volume_list, high_list, current_price, tf_label,
                 sanity_cap_pct=25.0):
    n = min(FFT_CANDLES, len(close_list))
    if n < 32: return None
    close  = np.array(close_list[-n:], dtype=np.float64)
    volume = np.array(volume_list[-n:], dtype=np.float64)
    high   = np.array(high_list[-n:],  dtype=np.float64)
    tf_price_ref = float(np.median(close))
    detrended, trend_coeffs = _detrend(close)
    spectrum = np.fft.rfft(detrended); freqs = np.fft.rfftfreq(n)
    power    = np.abs(spectrum); power[0] = 0
    min_period   = 4
    valid_mask   = (freqs > 0) & (freqs <= 1.0 / min_period)
    if not np.any(valid_mask): return None
    masked_power              = power.copy()
    masked_power[~valid_mask] = 0
    dom_idx         = int(np.argmax(masked_power))
    dom_freq        = freqs[dom_idx]
    dominant_period = int(round(1.0 / dom_freq)) if dom_freq > 0 else n
    dominant_period = min(dominant_period, n // 2)
    top_indices              = np.argsort(masked_power)[-4:]
    clean_spec               = np.zeros_like(spectrum)
    clean_spec[top_indices]  = spectrum[top_indices]
    reconstructed            = np.fft.irfft(clean_spec, n=n)
    trend_at_end   = float(np.poly1d(trend_coeffs)(n - 1))
    trend_slope    = float(trend_coeffs[0])
    trend_forward  = trend_at_end + trend_slope * dominant_period
    osc_amplitude  = float(np.max(reconstructed) - np.min(reconstructed)) / 2.0
    osc_now        = float(reconstructed[-1]); osc_mean = float(np.mean(reconstructed))
    if osc_now < osc_mean:
        osc_contribution = osc_amplitude + abs(osc_now - osc_mean)
    else:
        osc_contribution = osc_amplitude * 0.5
    fft_target = trend_forward + osc_contribution
    fft_target = max(fft_target, current_price * 1.0001)
    BIN_PCT  = 0.003; bin_size = tf_price_ref * BIN_PCT; bins = {}
    for h, v in zip(high, volume):
        if h > current_price * 1.001:
            b = round(h / bin_size) * bin_size
            bins[b] = bins.get(b, 0.0) + float(v)
    res_target = None; res_volume = 0.0
    if bins:
        vol_threshold = float(np.percentile(list(bins.values()), 70))
        candidates    = {k: v for k, v in bins.items() if v >= vol_threshold and k > current_price}
        if candidates:
            res_target = float(min(candidates.keys()))
            res_volume = float(candidates[res_target])
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
    ht_data    = hilbert_cycle_analysis(close)
    cexp_data  = complex_exp_forecast(close, current_price)
    _abs_score  = 0.0; _exhs_score = 0.0; _abs_flag = False; _exhs_flag = False
    try:
        if len(volume) >= 20:
            look_abs = min(dominant_period, len(close) // 2, len(close) - 1)
            look_abs = max(look_abs, 4)
            c_seg    = close[-look_abs:]; v_seg = volume[-look_abs:]
            p_range  = max(c_seg) - min(c_seg) + 1e-12
            v_total  = float(v_seg.sum())
            _abs_score  = float(np.log1p(v_total / p_range))
            _abs_flag   = _abs_score > _ABS_LOG_THRESH
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
    _fft_phase_deg = None
    try:
        dom_coeff      = spectrum[dom_idx]
        _fft_phase_deg = float(np.degrees(np.angle(dom_coeff))) % 360.0
    except Exception:
        pass
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
        'tf': tf_label, 'dominant_period': dominant_period,
        'osc_amplitude': round(osc_amplitude, 8), 'fft_target': round(fft_target, 8),
        'res_target': round(res_target, 8) if res_target else None,
        'res_volume': round(res_volume, 2),
        'forecast': round(forecast, 8), 'upside_pct': round(upside_pct, 4),
        'cascade_stop': False,
        'ht_data': ht_data, 'cexp_data': cexp_data,
        'absorption_score': round(_abs_score, 4), 'exhaustion_score': round(_exhs_score, 4),
        'absorption_flag': _abs_flag, 'exhaustion_flag': _exhs_flag,
        'fft_phase_deg': round(_fft_phase_deg, 2) if _fft_phase_deg is not None else None,
        'sinusoid_r2': _sinusoid_r2, 'dom_idx': int(dom_idx), 'dom_freq': float(dom_freq),
    }


def _run_fft_for_tfs(pair, current_price, tf_list, sanity_cap_pct=25.0):
    tf_results = []
    for label, interval in tf_list:
        try:
            klines = trader.client.get_klines(symbol=pair, interval=interval, limit=FFT_CANDLES + 20)
        except Exception: continue
        if len(klines) < 32: continue
        close  = [float(k[4]) for k in klines]
        volume = [float(k[5]) for k in klines]
        high   = [float(k[2]) for k in klines]
        result = fft_analysis(close, volume, high, current_price, label, sanity_cap_pct=sanity_cap_pct)
        if result: tf_results.append(result)
    if not tf_results: return [], None
    forecasts = np.array([r['forecast'] for r in tf_results])
    weights   = np.array([r['res_volume'] if r['res_volume'] > 0 else 1.0 for r in tf_results], dtype=np.float64)
    best_forecast = float(np.average(forecasts, weights=weights))
    best_upside   = (best_forecast - current_price) / current_price * 100.0
    spread        = float(np.std(forecasts) / best_forecast * 100) if best_forecast > 0 else 0.0
    confidence    = round(max(0.0, min(100.0, 100.0 - spread * 8)), 1)
    best_overall  = {
        'current': current_price, 'forecast': round(best_forecast, 8),
        'upside_pct': round(best_upside, 4), 'confidence': confidence,
        'spread_pct': round(spread, 4),
    }
    return tf_results, best_overall


def full_fft_report(pair, current_price):
    stf_tfs = [('1m', '1m'), ('3m', '3m'), ('5m', '5m')]
    htf_tfs = [('15m', '15m'), ('30m', '30m'), ('1h', '1h'), ('2h', '2h')]
    stf_results, stf_best = _run_fft_for_tfs(pair, current_price, stf_tfs, sanity_cap_pct=25.0)
    htf_results = []
    for label, interval in htf_tfs:
        try:
            klines = trader.client.get_klines(symbol=pair, interval=interval, limit=FFT_CANDLES + 20)
        except Exception: continue
        if len(klines) < 32: continue
        close  = [float(k[4]) for k in klines]
        volume = [float(k[5]) for k in klines]
        high   = [float(k[2]) for k in klines]
        result = fft_analysis(close, volume, high, current_price, label, sanity_cap_pct=60.0)
        if not result: continue
        result['cascade_stop'] = False
        htf_results.append(result)
        if (result['res_target'] is not None and
                result['res_target'] > current_price * 1.015 and
                result['fft_target'] >= result['res_target']):
            result['cascade_stop'] = True; break
    if htf_results:
        forecasts = np.array([r['forecast'] for r in htf_results])
        weights   = np.array([r['res_volume'] if r['res_volume'] > 0 else 1.0 for r in htf_results], dtype=np.float64)
        htf_best_forecast = float(np.average(forecasts, weights=weights))
        htf_best_upside   = (htf_best_forecast - current_price) / current_price * 100.0
        spread            = float(np.std(forecasts) / htf_best_forecast * 100) if htf_best_forecast > 0 else 0.0
        confidence        = round(max(0.0, min(100.0, 100.0 - spread * 5)), 1)
        stopped_tf        = next((r['tf'] for r in htf_results if r.get('cascade_stop')), 'none')
        htf_best = {
            'current': current_price, 'forecast': round(htf_best_forecast, 8),
            'upside_pct': round(htf_best_upside, 4), 'confidence': confidence,
            'spread_pct': round(spread, 4), 'tfs_used': len(htf_results), 'stopped_at': stopped_tf,
        }
    else:
        htf_best = None
    return stf_results, stf_best, htf_results, htf_best


def _print_tf_block(r):
    has_res  = r['res_target'] is not None
    stop_tag = '  ◄ CASCADE STOP (resistance reached)' if r.get('cascade_stop') else ''
    ht = r.get('ht_data') or {}
    ht_line = ''
    if ht.get('ht_period'):
        bz  = ' ✔ BUY ZONE' if ht.get('in_buy_zone') else ''
        ht_line = (f'  │  HT cycle        : period={ht["ht_period"]}b  '
                   f'phase={ht.get("ht_phase_deg","?")}°  '
                   f'→{ht.get("bars_to_trough","?")}b to trough{bz}')
    abs_flag  = r.get('absorption_flag',  False); exhs_flag = r.get('exhaustion_flag', False)
    abs_s     = r.get('absorption_score', 0.0);   exhs_s    = r.get('exhaustion_score', 0.0)
    fft_ph    = r.get('fft_phase_deg');            sin_r2    = r.get('sinusoid_r2')
    print(f'  ┌─ [{r["tf"]}] {"─"*52}┐')
    print(f'  │  Dominant cycle  : {r["dominant_period"]} bars')
    print(f'  │  Oscillation amp : {r["osc_amplitude"]}')
    if fft_ph is not None:
        r2_str = f'  sinusoid R²={sin_r2}' if sin_r2 is not None else ''
        print(f'  │  FFT phase angle : {fft_ph}°{r2_str}')
    if ht_line: print(ht_line)
    if abs_flag or exhs_flag:
        print(f'  │  ── Order-flow signals (cause→effect) ───────────────────')
        print(f'  │  Absorption scr  : {abs_s:.4f}{"  🟡 ABSORPTION ACTIVE" if abs_flag else ""}')
        print(f'  │  Exhaustion scr  : {exhs_s:.4f}{"  🔴 EXHAUSTION ACTIVE" if exhs_flag else ""}')
        if abs_flag and not exhs_flag:
            print(f'  │  → Effort ≠ result: smart money absorbing, reversal near')
        if exhs_flag:
            print(f'  │  → Vol fuel spent: one side running out, phase-flip risk')
    print(f'  │  FFT projection  : {r["fft_target"]}')
    if has_res:
        print(f'  │  Vol resistance  : {r["res_target"]}  (vol weight {r["res_volume"]:.0f}){stop_tag}')
    else:
        print(f'  │  Vol resistance  : none found above entry')
    print(f'  │  ── Forecast ────────────────────────────────────────────')
    print(f'  │  Price target    : {r["forecast"]}')
    print(f'  │  Upside          : +{r["upside_pct"]} %')
    blend = '60% vol-res + 40% FFT' if has_res else '100% FFT (no resistance)'
    print(f'  │  Blend method    : {blend}')
    print(f'  └{"─"*60}┘\n')


def print_fft_report(pair, label_map, stf_results, stf_best, htf_results, htf_best):
    lbl = label_map.get(pair, pair.replace('USDC', ''))
    w   = 62
    print(f'\n  {"═"*w}')
    print(f'  ◈  FFT SPIKE FORECAST  ·  {lbl}  ({pair})')
    print(f'  {"═"*w}')
    print(f'  Entry price : {stf_best["current"] if stf_best else "—"}\n')
    if stf_results:
        print(f'  ▸ SHORT-TERM TARGETS  (1m · 3m · 5m)\n')
        for r in stf_results: _print_tf_block(r)
        print(f'  {"═"*w}')
        print(f'  ★  BEST SHORT-TERM FORECAST  (1m/3m/5m consensus)')
        print(f'  {"─"*w}')
        print(f'  Consensus target : {stf_best["forecast"]}')
        print(f'  Upside           : +{stf_best["upside_pct"]} %')
        print(f'  Confidence       : {stf_best["confidence"]} %  (TF spread {stf_best["spread_pct"]} %)')
        print(f'  Method           : volume-weighted avg · 1m/3m/5m')
        print(f'  {"═"*w}\n')
    if htf_results:
        stopped = htf_results[-1].get('cascade_stop', False)
        stop_tf = htf_results[-1]['tf']
        tfs_run = ' · '.join(r['tf'] for r in htf_results)
        print(f'  ▸ HIGHER-TIMEFRAME TARGETS  ({tfs_run})')
        if stopped:
            print(f'    Cascade stopped at {stop_tf} — resistance wall reached\n')
        else:
            print()
        for r in htf_results: _print_tf_block(r)
        if htf_best:
            print(f'  {"═"*w}')
            print(f'  ★  BEST HIGHER-TIMEFRAME FORECAST  ({tfs_run})')
            print(f'  {"─"*w}')
            print(f'  Consensus target : {htf_best["forecast"]}')
            print(f'  Upside           : +{htf_best["upside_pct"]} %')
            print(f'  Confidence       : {htf_best["confidence"]} %  (TF spread {htf_best["spread_pct"]} %)')
            print(f'  TFs used         : {htf_best["tfs_used"]}  (stopped at {htf_best["stopped_at"]})')
            print(f'  Method           : volume-weighted avg · HTF cascade')
            print(f'  {"═"*w}\n')
    else:
        print(f'  HTF forecast: insufficient data.\n')


def run_time_geometry(pair, label_map, current_price, sel_detail, stf_results, htf_results):
    lbl = label_map.get(pair, pair.replace('USDC', ''))
    w   = 62
    d          = sel_detail.get(pair, {})
    close_arr  = d.get('close_arr')
    swing_low  = d.get('swing_low')
    swing_high = d.get('swing_high')
    geo_d      = d.get('geometry_detail', {})
    if close_arr is None or len(close_arr) < 32:
        print(f'  Time geometry: insufficient 1m data.\n'); return
    ht   = hilbert_cycle_analysis(close_arr)
    cexp = complex_exp_forecast(close_arr, current_price)
    phi_levels = phi_extension_levels(swing_low, swing_high, current_price) if (swing_low and swing_high) else []
    gs   = golden_spiral_targets(current_price, ht.get('ht_phase_deg') if ht else None,
                                  ht.get('ht_amplitude') if ht else None,
                                  swing_low=swing_low, swing_high=swing_high) if ht else None
    harm_score, harm_d = mtf_harmony_score(stf_results, htf_results)
    print(f'\n  {"═"*w}')
    print(f'  ◈  TIME GEOMETRY REPORT  ·  φ · e · π  ·  {lbl}')
    print(f'  {"═"*w}')
    print(f'  φ = structure/proportion   e = evolution/time   π = cycles/rotation')
    print(f'  Together → GOLDEN SPIRAL: r(θ) = A·e^(b·θ),  b = ln(φ)/(π/2)\n')
    # Dip geometry score
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
    print(f'  └{"─"*60}┘\n')
    # HT cycle
    print(f'  ┌─ π COMPONENT — Hilbert Transform Cycle (1m) {"─"*14}┐')
    if ht:
        bz_tag = '  ← BUY ZONE ✔' if ht.get('in_buy_zone') else ''
        tm_tag = 'CYCLING' if ht.get('ht_trend_mode') == 0 else ('TRENDING' if ht.get('ht_trend_mode') == 1 else '?')
        print(f'  │  Dominant HT period : {ht.get("ht_period","?")} bars')
        print(f'  │  Current phase      : {ht.get("ht_phase_deg","?")}°  ({ht.get("phase_label","?")})')
        print(f'  │  Sine / Lead sine   : {ht.get("ht_sine","?")} / {ht.get("ht_lead_sine","?")}{bz_tag}')
        print(f'  │  HT amplitude       : {ht.get("ht_amplitude","?")}')
        print(f'  │  Market mode        : {tm_tag}')
        print(f'  │  Bars to trough     : {ht.get("bars_to_trough","?")}  (phase 270° = lowest cycle point)')
    print(f'  └{"─"*60}┘\n')
    # Complex exponential
    CEXP_MIN_R2 = 0.30
    print(f'  ┌─ e·π BRIDGE — Complex Exponential Fit  y=A·e^(α+iω)t {"─"*3}┐')
    if cexp:
        r2_val = cexp.get('fit_r2', 0.0); r2_ok = r2_val >= CEXP_MIN_R2
        print(f'  │  α  (exp growth rate)  : {cexp.get("alpha","?")}  ← {cexp.get("e_label","")}')
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
    print(f'  └{"─"*60}┘\n')
    # Golden spiral
    print(f'  ┌─ φ·π BRIDGE — Golden Spiral Targets  r(θ)=A·e^(b·θ) {"─"*4}┐')
    if gs:
        print(f'  │  Amplitude base (swing/2): {gs.get("A_base","?")}  ({gs.get("A_base",0)/current_price*100:.2f}% of price)')
        print(f'  │  Current spiral angle    : {gs.get("current_angle","?")}°')
        for label_s, key, phi_mult_key in [
            (f'+π/2  (×φ  =×{PHI:.3f})',   'gs_q1_turn',  'phi_mult_q1'),
            (f'+π    (×φ² =×{PHI2:.3f})',   'gs_half_turn','phi_mult_half'),
            (f'+3π/2 (×φ³ =×{PHI**3:.3f})','gs_3q_turn',  None),
            (f'+2π   (×φ⁴ =×{PHI**4:.3f})','gs_full_turn', None),
        ]:
            v = gs.get(key)
            if v:
                pct  = (v - current_price) / current_price * 100.0
                mult = f'  (×{gs[phi_mult_key]} radius)' if phi_mult_key and gs.get(phi_mult_key) else ''
                print(f'  │  {label_s:<26}: {v}  ({pct:+.2f}%){mult}')
        print(f'  │  GOLDEN_B = ln(φ)/(π/2) = {GOLDEN_B:.5f}')
    print(f'  └{"─"*60}┘\n')
    # Fibonacci
    print(f'  ┌─ φ COMPONENT — Fibonacci Extension Levels (from swing) {"─"*3}┐')
    if phi_levels:
        print(f'  │  Swing low  : {swing_low:.8f}    Swing high : {swing_high:.8f}')
        print(f'  │  Range      : {swing_high - swing_low:.8f}')
        print(f'  │  Entry      : {current_price:.8f}')
        print(f'  │  {"─"*54}')
        for ratio, label_txt, level in phi_levels[:7]:
            pct    = (level - current_price) / current_price * 100.0
            marker = ' ← φ' if abs(ratio - PHI_INV) < 0.01 or abs(ratio - PHI) < 0.01 else (' ← φ²' if abs(ratio - PHI2) < 0.01 else '')
            print(f'  │  {label_txt:<28}: {level:.8f}  (+{pct:.2f}%){marker}')
    print(f'  └{"─"*60}┘\n')
    # MTF harmony
    print(f'  ┌─ MTF HARMONY SCORE  (φ·e·π alignment across all TFs) {"─"*3}┐')
    if harm_d:
        n_tfs = len(stf_results or []) + len(htf_results or [])
        print(f'  │  TFs analysed         : {n_tfs}')
        print(f'  │  φ harmony (periods)  : {harm_d.get("phi_pairs","?")}/{harm_d.get("total_pairs","?")} period pairs in φ-ratio  → {harm_d.get("phi_harmony","?")} pts')
        print(f'  │  e harmony (consensus): CV={harm_d.get("fc_cv_pct","?")}%  → {harm_d.get("e_harmony","?")} pts')
        print(f'  │  π harmony (direction): {harm_d.get("n_up","?")}/{harm_d.get("n_total","?")} TFs upside  → {harm_d.get("pi_harmony","?")} pts')
        print(f'  │  No-stop bonus        : {harm_d.get("no_stop_bonus","?")} pts')
        hs  = harm_d.get('harmony', 0)
        bar = '█' * int(hs // 5) + '░' * (20 - int(hs // 5))
        print(f'  │  ─────────────────────────────────────────────────────────')
        print(f'  │  HARMONY SCORE        : {hs:>5.1f} / 100   [{bar}]')
        if   hs >= 80: harmony_label = 'STRONG — all constants aligned'
        elif hs >= 60: harmony_label = 'MODERATE — partial alignment'
        elif hs >= 40: harmony_label = 'WEAK — limited alignment'
        else:          harmony_label = 'DISCORD — constants not aligned'
        print(f'  │  Interpretation       : {harmony_label}')
    print(f'  └{"─"*60}┘\n')
    print(f'  {"═"*w}\n')


# ── ML functions (all identical to original) ──────────────

ML_LOOKBACK   = 1000
ML_TEST_RATIO = 0.20
ML_WALKS      = 10_000
ML_HORIZON    = 30

def _safe_talib(fn, *arrays, **kw):
    try:
        res = fn(*arrays, **kw)
        return np.where(np.isnan(res), 0.0, res)
    except Exception:
        return np.zeros(len(arrays[0]))

def build_feature_matrix(close, volume, high, low, geo_detail=None,
                          ht_data_arr=None, phi_devs=0.0, e_alpha=0.0,
                          fft_period=20, ht_period=20, swing_low=None, swing_high=None):
    n   = len(close); arr = np.asarray(close, dtype=np.float64)
    vol = np.asarray(volume, dtype=np.float64)
    hi  = np.asarray(high,   dtype=np.float64)
    lo  = np.asarray(low,    dtype=np.float64)
    def safe_roll_mean(a, w):
        out = np.full(n, 0.0)
        for i in range(w - 1, n): out[i] = np.mean(a[i - w + 1: i + 1])
        return out
    def safe_roll_std(a, w):
        out = np.full(n, 1e-10)
        for i in range(w - 1, n): out[i] = np.std(a[i - w + 1: i + 1]) + 1e-10
        return out
    def log_ret_lag(lag):
        out = np.zeros(n)
        for i in range(lag, n):
            if arr[i - lag] > 1e-20: out[i] = np.log(arr[i] / arr[i - lag])
        return out
    t_norm = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    lr1  = log_ret_lag(1);  lr2  = log_ret_lag(2);  lr3  = log_ret_lag(3)
    lr5  = log_ret_lag(5);  lr10 = log_ret_lag(10); lr20 = log_ret_lag(20)
    rm10 = safe_roll_mean(arr,10); rs10 = safe_roll_std(arr,10)
    rm20 = safe_roll_mean(arr,20); rs20 = safe_roll_std(arr,20)
    rm50 = safe_roll_mean(arr,50); rs50 = safe_roll_std(arr,50)
    rsi14   = _safe_talib(ta.RSI, arr, timeperiod=14) / 100.0
    cmo14   = _safe_talib(ta.CMO, arr, timeperiod=14) / 100.0
    mom10   = _safe_talib(ta.MOM, arr, timeperiod=10)
    mom10_n = mom10 / (arr + 1e-20)
    atr14_r = _safe_talib(ta.ATR, hi, lo, arr, timeperiod=14)
    atr14_n = atr14_r / (arr + 1e-20)
    upper_b = rm20 + 2.0 * rs20; lower_b = rm20 - 2.0 * rs20
    bb_pos  = (arr - lower_b) / (upper_b - lower_b + 1e-20)
    phi_dev_ts = np.zeros(n); win = min(50, n)
    for i in range(win, n):
        seg = arr[i - win: i + 1]; t_ = np.arange(len(seg), dtype=np.float64)
        p_  = np.polyfit(t_, seg, 1); tr_ = np.poly1d(p_)(t_)
        sg  = float(np.std(seg)) + 1e-20
        phi_dev_ts[i] = max(0.0, (tr_[-1] - seg[-1]) / sg)
    ht_ph_ts  = _safe_talib(ta.HT_DCPHASE,  arr)
    ht_pe_ts  = _safe_talib(ta.HT_DCPERIOD, arr)
    ht_ph_sin = np.sin(np.deg2rad(ht_ph_ts))
    ht_ph_cos = np.cos(np.deg2rad(ht_ph_ts))
    ht_pe_n   = ht_pe_ts / max(n, 1)
    fft_per_n = np.full(n, fft_period / max(n, 1))
    e_alpha_ts = np.full(n, float(e_alpha))
    bull_mask  = arr >= np.roll(arr, 1); bull_mask[0] = False
    bull_vol_r = np.zeros(n)
    for i in range(20, n):
        bv = vol[i-20:i+1][bull_mask[i-20:i+1]].sum()
        tv = vol[i-20:i+1].sum() + 1e-20
        bull_vol_r[i] = bv / tv
    argmin_d = np.zeros(n); argmax_d = np.zeros(n); look_ml = min(100, n)
    for i in range(look_ml, n):
        seg_lo = lo[i-look_ml:i+1]; seg_hi = hi[i-look_ml:i+1]
        argmin_d[i] = (look_ml - int(np.argmin(seg_lo))) / look_ml
        argmax_d[i] = (look_ml - int(np.argmax(seg_hi))) / look_ml
    sl = swing_low  if swing_low  else float(np.min(arr))
    sh = swing_high if swing_high else float(np.max(arr))
    rng = (sh - sl) + 1e-20
    fib_low  = (arr - sl) / rng; fib_high = (sh - arr) / rng
    gs_theta = np.deg2rad(ht_ph_ts) * GOLDEN_B
    gs_sin   = np.sin(gs_theta);  gs_cos = np.cos(gs_theta)
    X = np.column_stack([
        t_norm, lr1, lr2, lr3, lr5, lr10, lr20,
        rm10, rs10, rm20, rs20, rm50, rs50,
        rsi14, cmo14, mom10_n, atr14_n, bb_pos,
        phi_dev_ts, ht_ph_sin, ht_ph_cos, ht_pe_n,
        fft_per_n, e_alpha_ts,
        bull_vol_r, argmin_d, argmax_d,
        fib_low, fib_high, gs_sin, gs_cos,
    ])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def compute_volume_sr(close, volume, high, low, open_,
                       current_price, pair=None, lookback=100, bin_pct=0.003):
    close  = np.asarray(close,  dtype=np.float64); volume = np.asarray(volume, dtype=np.float64)
    high_  = np.asarray(high,   dtype=np.float64); low_   = np.asarray(low,    dtype=np.float64)
    open__ = np.asarray(open_,  dtype=np.float64); n_full = len(close)
    if n_full < 5: return [], [], {}
    lb = min(lookback, n_full); half = lb // 2
    c = close[-lb:]; v = volume[-lb:]; h = high_[-lb:]
    lo = low_[-lb:]; op = open__[-lb:]; n = lb
    c_full = close; v_full = volume; h_full = high_; l_full = low_; o_full = open__
    cp = float(current_price)
    absolute_min = float(np.min(low_)); absolute_max = float(np.max(high_))
    full_range = absolute_max - absolute_min + 1e-20
    bs = max(full_range * bin_pct, 1e-12)
    def _bin(price): return round(round(price / bs) * bs, 10)
    argmin_bin = _bin(absolute_min); argmax_bin = _bin(absolute_max); entry_bin = _bin(cp)
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
    bull_h1={}; bull_h2={}; bull_cnt={}; bear_h1={}; bear_h2={}; bear_cnt={}
    for i in range(n):
        cl_i=float(c[i]); op_i=float(op[i]); v_i=float(v[i]); cb=_bin(cl_i)
        is_bull=cl_i>=op_i; in_h2=i>=half
        if is_bull:
            bull_cnt[cb]=bull_cnt.get(cb,0)+1
            if in_h2: bull_h2[cb]=bull_h2.get(cb,0.0)+v_i
            else:     bull_h1[cb]=bull_h1.get(cb,0.0)+v_i
        else:
            bear_cnt[cb]=bear_cnt.get(cb,0)+1
            if in_h2: bear_h2[cb]=bear_h2.get(cb,0.0)+v_i
            else:     bear_h1[cb]=bear_h1.get(cb,0.0)+v_i
    all_bins_b=set(bull_h1)|set(bull_h2); all_bins_r=set(bear_h1)|set(bear_h2)
    bull_bins={b: bull_h1.get(b,0.0)+bull_h2.get(b,0.0) for b in all_bins_b}
    bear_bins={b: bear_h1.get(b,0.0)+bear_h2.get(b,0.0) for b in all_bins_r}
    total_bull_lb=sum(bull_bins.values())+1e-20; total_bear_lb=sum(bear_bins.values())+1e-20
    vol_bins_fp={}; mkt_bins_fp={}; total_bull_fw=0.0; total_bear_fw=0.0
    for i in range(n_full):
        cl_i=float(c_full[i]); op_i=float(o_full[i]); v_i=float(v_full[i])
        tp_i=(float(h_full[i])+float(l_full[i])+cl_i)/3.0; tb=_bin(tp_i)
        vol_bins_fp[tb]=vol_bins_fp.get(tb,0.0)+v_i; mkt_bins_fp[tb]=mkt_bins_fp.get(tb,0)+1
        if cl_i>=op_i: total_bull_fw+=v_i
        else:          total_bear_fw+=v_i
    total_vol_fw=total_bull_fw+total_bear_fw+1e-20
    bull_pct_fw=total_bull_fw/total_vol_fw*100.0; bear_pct_fw=total_bear_fw/total_vol_fw*100.0
    if   bull_pct_fw>=55.0: predominance='BULLISH'
    elif bear_pct_fw>=55.0: predominance='BEARISH'
    else:                   predominance='NEUTRAL'
    def _poc_va(bins):
        if not bins: return None,None,None
        poc=max(bins,key=bins.get); tv=sum(bins.values()); tgt=tv*0.70
        cl={poc:bins[poc]}; cv=bins[poc]; ap=sorted(bins.keys())
        if poc not in ap: return round(poc,8),round(poc,8),round(poc,8)
        li=ap.index(poc); hi_i=li
        while cv<tgt:
            cl_=li>0; ch=hi_i<len(ap)-1
            if not cl_ and not ch: break
            lv=bins.get(ap[li-1],0.0) if cl_ else 0.0; hv=bins.get(ap[hi_i+1],0.0) if ch else 0.0
            if lv>=hv and cl_: li-=1; cv+=lv; cl[ap[li]]=lv
            elif ch: hi_i+=1; cv+=hv; cl[ap[hi_i]]=hv
            else: break
        return round(poc,8),round(max(cl.keys())+bs*0.5,8),round(min(cl.keys())-bs*0.5,8)
    vol_poc,vol_vah,vol_val=_poc_va(vol_bins_fp); mkt_poc,mkt_vah,mkt_val=_poc_va(mkt_bins_fp)
    def _build_levels(bins, cnt, h1, h2, side_total, strongest_n=3):
        if not bins: return []
        all_p=sorted(bins.keys()); all_vol=sum(bins.values()); scored=[]
        for price, raw_vol in bins.items():
            n_bars=cnt.get(price,0); consistency=n_bars/n
            v1=h1.get(price,0.0); v2=h2.get(price,0.0)
            vol_growth=(v2/(v1+1e-20))-1.0
            composite=raw_vol*(1.0+consistency)*max(1.0,1.0+vol_growth)
            scored.append((price,raw_vol,n_bars,consistency,vol_growth,composite))
        scored.sort(key=lambda x:x[5],reverse=True); top=scored[:strongest_n]; levels=[]
        for price,raw_vol,n_bars,consistency,vol_growth,composite in top:
            vol_pct=round(raw_vol/(side_total+1e-20)*100.0,1)
            dist_pct=round((price-cp)/(cp+1e-20)*100.0,3)
            if len(bins)==1:
                sr_fl=round(price-bs*0.5,8); sr_ce=round(price+bs*0.5,8)
            else:
                tgt=all_vol*0.70; cl={price:raw_vol}; cv=raw_vol
                idx=all_p.index(price) if price in all_p else 0; li=idx; hi_i=idx
                while cv<tgt:
                    can_l=li>0; can_h=hi_i<len(all_p)-1
                    if not can_l and not can_h: break
                    lv=bins.get(all_p[li-1],0.0) if can_l else 0.0
                    hv=bins.get(all_p[hi_i+1],0.0) if can_h else 0.0
                    if lv>=hv and can_l: li-=1; cv+=lv; cl[all_p[li]]=lv
                    elif can_h: hi_i+=1; cv+=hv; cl[all_p[hi_i]]=hv
                    else: break
                sr_fl=round(min(cl.keys())-bs*0.5,8); sr_ce=round(max(cl.keys())+bs*0.5,8)
            range_pct=round((sr_ce-sr_fl)/(price+1e-20)*100.0,3)
            levels.append((round(price,8),round(raw_vol,2),n_bars,sr_fl,sr_ce,range_pct,
                           _zone(price),dist_pct,vol_pct,round(composite,2),round(vol_growth,4),round(consistency,4)))
        return levels
    support_levels    = _build_levels(bull_bins,bull_cnt,bull_h1,bull_h2,total_bull_lb)
    resistance_levels = _build_levels(bear_bins,bear_cnt,bear_h1,bear_h2,total_bear_lb)
    if not support_levels:
        support_levels=[(round(absolute_min,8),0.0,0,round(absolute_min-bs*0.5,8),round(absolute_min+bs*0.5,8),
            round(bs/(absolute_min+1e-20)*100,3),_zone(absolute_min),round((absolute_min-cp)/(cp+1e-20)*100,3),0.0,0.0,0.0,0.0)]
    if not resistance_levels:
        resistance_levels=[(round(absolute_max,8),0.0,0,round(absolute_max-bs*0.5,8),round(absolute_max+bs*0.5,8),
            round(bs/(absolute_max+1e-20)*100,3),_zone(absolute_max),round((absolute_max-cp)/(cp+1e-20)*100,3),0.0,0.0,0.0,0.0)]
    ob_bid_wall=None; ob_ask_wall=None
    if pair:
        try:
            ob=trader.client.get_order_book(symbol=pair,limit=50)
            bids=[(float(b[0]),float(b[1])) for b in ob.get('bids',[])]
            asks=[(float(a[0]),float(a[1])) for a in ob.get('asks',[])]
            if bids: ob_bid_wall=max(bids,key=lambda x:x[1])
            if asks: ob_ask_wall=max(asks,key=lambda x:x[1])
        except Exception: pass
    profile={
        'vol_poc':vol_poc,'vol_vah':vol_vah,'vol_val':vol_val,
        'mkt_poc':mkt_poc,'mkt_vah':mkt_vah,'mkt_val':mkt_val,
        'bull_vol':round(total_bull_fw,2),'bear_vol':round(total_bear_fw,2),
        'total_vol':round(total_vol_fw,2),'bull_pct':round(bull_pct_fw,2),
        'bear_pct':round(bear_pct_fw,2),'predominance':predominance,
        'vol_bins':sorted(vol_bins_fp.items()),'mkt_bins':sorted(mkt_bins_fp.items()),
        'absolute_min':absolute_min,'absolute_max':absolute_max,
        'ob_bid_wall':ob_bid_wall,'ob_ask_wall':ob_ask_wall,'lookback':lb,
    }
    return support_levels, resistance_levels, profile

def random_walk_mc(close, horizon=ML_HORIZON, n_paths=ML_WALKS,
                   use_full_lookback=True, dip_drift_boost=True):
    arr  = np.asarray(close, dtype=np.float64)
    look = min(ML_LOOKBACK, len(arr)) if use_full_lookback else min(100, len(arr))
    log_rets = np.diff(np.log(arr[-look:] + 1e-20))
    if len(log_rets) < 10: return None
    raw_drift = float(np.mean(log_rets)); vol = float(np.std(log_rets))
    drift = max(raw_drift, vol * 0.15) if dip_drift_boost else raw_drift
    rng_gen    = np.random.default_rng(seed=None)
    last_price = float(arr[-1])
    def run_paths(h):
        draws = rng_gen.choice(log_rets, size=(n_paths, h), replace=True) + drift
        return np.exp(draws.sum(axis=1)) * last_price
    fp_s=run_paths(horizon); fp_m=run_paths(horizon*4); fp_l=run_paths(horizon*16)
    def pcts(fp):
        return {'p5':round(float(np.percentile(fp,5)),8),'p25':round(float(np.percentile(fp,25)),8),
                'p50':round(float(np.percentile(fp,50)),8),'p75':round(float(np.percentile(fp,75)),8),
                'p95':round(float(np.percentile(fp,95)),8),'mean':round(float(np.mean(fp)),8),
                'std':round(float(np.std(fp)),8),'prob_up':round(float(np.mean(fp>last_price)),4)}
    return {'short':pcts(fp_s),'medium':pcts(fp_m),'long':pcts(fp_l),
            **{k:pcts(fp_s)[k] for k in ('p5','p25','p50','p75','p95','mean','std','prob_up')},
            'raw_drift':round(raw_drift,8),'drift_applied':round(drift,8),
            'vol_per_bar':round(vol,8),'horizon_s':horizon,'horizon_m':horizon*4,'horizon_l':horizon*16}

def regression_channel(close, n_sigma=2.0):
    arr=np.asarray(close,dtype=np.float64); n=len(arr); t=np.arange(n,dtype=np.float64)
    p=np.polyfit(t,arr,1); fit=np.poly1d(p)(t); res=arr-fit; sig=float(np.std(res))
    last_fit=float(np.poly1d(p)(n-1)); next_fit=float(np.poly1d(p)(n))
    upper_last=last_fit+n_sigma*sig; lower_last=last_fit-n_sigma*sig
    upper_next=next_fit+n_sigma*sig; lower_next=next_fit-n_sigma*sig
    curr=float(arr[-1]); band_rng=(upper_last-lower_last)+1e-20
    position=(curr-lower_last)/band_rng
    return {'slope':round(float(p[0]),10),'intercept':round(float(p[1]),8),
            'upper_band':round(upper_last,8),'mid_line':round(last_fit,8),
            'lower_band':round(lower_last,8),'next_upper':round(upper_next,8),
            'next_mid':round(next_fit,8),'next_lower':round(lower_next,8),
            'channel_pos':round(position,4),'residual_std':round(sig,8)}

def _define_models():
    kernel=Matern(length_scale=1.0,nu=1.5)+WhiteKernel(noise_level=1.0)
    return {
        'Ridge':     Ridge(alpha=1.0),
        'Lasso':     Lasso(alpha=0.001,max_iter=5000),
        'BayesRidge':BayesianRidge(),
        'PolyReg3':  Pipeline([('poly',PolynomialFeatures(degree=3,include_bias=False)),
                               ('scaler',StandardScaler()),('ridge',Ridge(alpha=10.0))]),
        'RandForest':RandomForestRegressor(n_estimators=300,max_depth=8,min_samples_leaf=5,
                                           n_jobs=-1,random_state=42,oob_score=True),
        'GradBoost': GradientBoostingRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,
                                               subsample=0.8,random_state=42),
        'SVR_rbf':   Pipeline([('scaler',StandardScaler()),
                               ('svr',SVR(kernel='rbf',C=100,epsilon=0.001,gamma='scale'))]),
        'GaussProc': GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=3,
                                              normalize_y=True,random_state=42),
    }

def _build_Xy(X, close, horizon=1):
    n=len(close); y_f=np.asarray(close,dtype=np.float64)
    return X[:n-horizon], y_f[horizon:]

def train_and_backtest(X, close, horizon=ML_HORIZON, test_ratio=ML_TEST_RATIO):
    X_ml,y_ml=_build_Xy(X,close,horizon=horizon); n=len(X_ml)
    split=max(20,int(n*(1.0-test_ratio)))
    X_train,X_test=X_ml[:split],X_ml[split:]
    y_train,y_test=y_ml[:split],y_ml[split:]
    if len(X_train)<15 or len(X_test)<5:
        return None,None,X_train,X_test,y_train,y_test
    models=_define_models(); fitted={}; metrics={}
    for name,mdl in models.items():
        try:
            mdl.fit(X_train,y_train); preds=mdl.predict(X_test)
            mae=float(mean_absolute_error(y_test,preds))
            rmse=float(np.sqrt(mean_squared_error(y_test,preds)))
            r2=float(r2_score(y_test,preds))
            actual_dir=np.sign(np.diff(y_test)); pred_dir=np.sign(np.diff(preds))
            dir_acc=float(np.mean(actual_dir==pred_dir)) if len(actual_dir)>0 else 0.0
            fitted[name]=mdl
            metrics[name]={'mae':round(mae,8),'rmse':round(rmse,8),'r2':round(r2,4),'dir_acc':round(dir_acc,4)}
        except Exception:
            metrics[name]={'mae':1e9,'rmse':1e9,'r2':-999.0,'dir_acc':0.0}
    return fitted,metrics,X_train,X_test,y_train,y_test

def ensemble_forecast(fitted_models, metrics, X_last):
    forecasts={}; weights={}
    for name,mdl in fitted_models.items():
        try:
            pred=float(mdl.predict(X_last.reshape(1,-1))[0])
            mae=metrics[name]['mae']
            forecasts[name]=pred; weights[name]=1.0/(mae+1e-20)
        except Exception: pass
    if not forecasts: return None,{}
    total_w=sum(weights.values())
    ensemble=sum(forecasts[n]*weights[n] for n in forecasts)/(total_w+1e-20)
    return float(ensemble),forecasts

def ml_compound_forecast(pair, current_price, sel_detail, stf_results, htf_results, thr_map=None):
    if not ML_AVAILABLE: return None
    tf_data={}
    for label,interval in [('1m','1m'),('3m','3m'),('5m','5m')]:
        try:
            klines=trader.client.get_klines(symbol=pair,interval=interval,limit=ML_LOOKBACK+50)
        except Exception: continue
        if len(klines)<50: continue
        tf_data[label]={
            'close': np.array([float(k[4]) for k in klines],dtype=np.float64),
            'volume':np.array([float(k[5]) for k in klines],dtype=np.float64),
            'high':  np.array([float(k[2]) for k in klines],dtype=np.float64),
            'low':   np.array([float(k[3]) for k in klines],dtype=np.float64),
            'open':  np.array([float(k[1]) for k in klines],dtype=np.float64),
        }
    if '1m' not in tf_data: return None
    close=tf_data['1m']['close']; volume=tf_data['1m']['volume']
    high=tf_data['1m']['high'];   low=tf_data['1m']['low']; open_=tf_data['1m']['open']
    n=len(close)
    d=sel_detail.get(pair,{}); geo_d=d.get('geometry_detail',{})
    phi_devs=geo_d.get('phi_devs',0.0); e_alpha=geo_d.get('e_decay_rate',0.0)
    swing_low=d.get('swing_low'); swing_high=d.get('swing_high')
    fft_period=stf_results[0].get('dominant_period',20) if stf_results else 20
    ht_pe_arr=ta.HT_DCPERIOD(close); valid_pe=~np.isnan(ht_pe_arr)
    ht_period=float(ht_pe_arr[valid_pe][-1]) if np.any(valid_pe) else 20.0
    reg_ch=regression_channel(close[-ML_LOOKBACK:])
    sup_levels,res_levels,vol_profile=compute_volume_sr(
        close,volume,high,low,open_,current_price,pair=pair,lookback=100)
    argmin_data={}
    for label in ('1m','3m','5m'):
        td=tf_data.get(label)
        if td is None: continue
        lo_w=td['low'][-min(ML_LOOKBACK,len(td['low'])):]
        hi_w=td['high'][-min(ML_LOOKBACK,len(td['high'])):]
        argmin_data[label]={
            'argmin':int(np.argmin(lo_w)),'argmax':int(np.argmax(hi_w)),
            'min_price':round(float(np.min(lo_w)),8),'max_price':round(float(np.max(hi_w)),8),
            'mid_price':round(float((np.min(lo_w)+np.max(hi_w))/2.0),8),'n_bars':len(lo_w),
        }
    X=build_feature_matrix(close,volume,high,low,geo_detail=geo_d,phi_devs=phi_devs,
                            e_alpha=e_alpha,fft_period=fft_period,ht_period=ht_period,
                            swing_low=swing_low,swing_high=swing_high)
    fitted,metrics,X_train,X_test,y_train,y_test=train_and_backtest(X,close,horizon=ML_HORIZON,test_ratio=ML_TEST_RATIO)
    if fitted is None: return None
    X_last=X[-1]; ens_forecast,indiv=ensemble_forecast(fitted,metrics,X_last)
    mc=random_walk_mc(close,horizon=ML_HORIZON,n_paths=ML_WALKS)
    sl=swing_low if swing_low else float(np.min(low[-ML_LOOKBACK:]))
    sh=swing_high if swing_high else float(np.max(high[-ML_LOOKBACK:]))
    phi_targets=phi_extension_levels(sl,sh,current_price)
    MIN_DIRAC=0.45; stf_fc=[]; mtf_fc=[]; htf_fc=[]
    stf_tf_labels={'1m','3m','5m'}; mtf_tf_labels={'15m','30m'}; htf_tf_labels={'1h','2h','4h'}
    for tf_label,tf_dict in (argmin_data or {}).items():
        mp=tf_dict.get('max_price'); amin=tf_dict.get('argmin',0); amax=tf_dict.get('argmax',0); nb=tf_dict.get('n_bars',1)
        if mp is None or mp<=current_price: continue
        recency=max(1,amax)/nb; range_w=(mp-current_price)/current_price; w=(recency+0.3)*range_w*10.0
        lbl_=f'ArgMax-{tf_label}'
        if   tf_label in stf_tf_labels: stf_fc.append((lbl_,mp,w))
        elif tf_label in mtf_tf_labels: mtf_fc.append((lbl_,mp,w))
        else:                            htf_fc.append((lbl_,mp,w))
    for nm,fc in (indiv or {}).items():
        if fc<=current_price: continue
        m_info=metrics.get(nm,{}); da=m_info.get('dir_acc',0.0); mae=m_info.get('mae',1.0)+1e-20
        if da<MIN_DIRAC: continue
        w=(da**2)/mae; stf_fc.append((nm,fc,w))
    passing_models=[nm for nm,m in (metrics or {}).items() if m.get('dir_acc',0.0)>=MIN_DIRAC]
    if ens_forecast and ens_forecast>current_price and passing_models:
        best_da=max(metrics[nm]['dir_acc'] for nm in passing_models)
        best_mae=min(metrics[nm]['mae'] for nm in passing_models)+1e-20
        w=(best_da**2)/best_mae*2.0; stf_fc.append(('ML Ensemble',ens_forecast,w))
    if mc:
        p75_s=mc['short']['p75']; p95_s=mc['short']['p95']
        p50_m=mc['medium']['p50']; p75_m=mc['medium']['p75']
        p75_l=mc['long']['p75']; p95_l=mc['long']['p95']
        vol_w=1.0/(mc['vol_per_bar']+1e-20)
        if p75_s>current_price: stf_fc.append(('MC-short-p75',p75_s,vol_w))
        if p95_s>current_price: stf_fc.append(('MC-short-p95',p95_s,vol_w*0.5))
        if p50_m>current_price: mtf_fc.append(('MC-med-p50',p50_m,vol_w))
        if p75_m>current_price: mtf_fc.append(('MC-med-p75',p75_m,vol_w))
        if p75_l>current_price: htf_fc.append(('MC-long-p75',p75_l,vol_w))
        if p95_l>current_price: htf_fc.append(('MC-long-p95',p95_l,vol_w))
    for r in (stf_results or []):
        fc=r.get('forecast',0)
        if fc>current_price:
            w=r.get('res_volume',1.0) or 1.0; lbl_=f'FFT-{r["tf"]}'
            if r['tf'] in stf_tf_labels: stf_fc.append((lbl_,fc,w))
            else: mtf_fc.append((lbl_,fc,w))
    for r in (htf_results or []):
        fc=r.get('forecast',0)
        if fc>current_price:
            w=r.get('res_volume',1.0) or 1.0; lbl_=f'FFT-{r["tf"]}'
            if r['tf'] in mtf_tf_labels: mtf_fc.append((lbl_,fc,w))
            else: htf_fc.append((lbl_,fc,w))
    for ratio,lbl_t,level in (phi_targets or []):
        if level<=current_price: continue
        dist_pct=(level-current_price)/current_price*100.0
        w=PHI if (abs(ratio-PHI_INV)<0.01 or abs(ratio-PHI)<0.01) else 1.0
        if   dist_pct<=5.0:  stf_fc.append((f'Fib-{lbl_t[:8]}',level,w))
        elif dist_pct<=15.0: mtf_fc.append((f'Fib-{lbl_t[:8]}',level,w))
        else:                 htf_fc.append((f'Fib-{lbl_t[:8]}',level,w))
    if reg_ch:
        ub=reg_ch['next_upper']
        if ub>current_price: htf_fc.append(('RegCh-Upper',ub,1.0))
    def weighted_avg(lst):
        if not lst: return None
        prices=np.array([x[1] for x in lst],dtype=np.float64)
        weights=np.maximum(np.array([x[2] for x in lst],dtype=np.float64),1e-20)
        return float(np.average(prices,weights=weights))
    tier_stf=weighted_avg(stf_fc); tier_mtf=weighted_avg(mtf_fc); tier_htf=weighted_avg(htf_fc)
    all_forecasts=sorted([(lbl_,fc) for lbl_,fc,_ in (stf_fc+mtf_fc+htf_fc)],key=lambda x:x[1])
    hard_cap=res_levels[0][0] if res_levels else None
    hard_cap_floor=res_levels[0][3] if res_levels else None
    hard_cap_ceil=res_levels[0][4] if res_levels else None
    best_target=tier_htf or tier_mtf or tier_stf
    if best_target and hard_cap and best_target>hard_cap*1.05: best_target=hard_cap
    soft_stop=sup_levels[0][0] if sup_levels else None
    soft_stop_floor=sup_levels[0][3] if sup_levels else None
    soft_stop_ceil=sup_levels[0][4] if sup_levels else None
    mid_1m=argmin_data.get('1m',{}).get('mid_price')
    return {
        'pair':pair,'current_price':current_price,'reg_channel':reg_ch,
        'sup_levels':sup_levels,'res_levels':res_levels,'vol_profile':vol_profile,
        'argmin_data':argmin_data,'fitted_models':fitted,'metrics':metrics,
        'ens_forecast':ens_forecast,'indiv_forecasts':indiv,'mc':mc,
        'phi_targets':phi_targets,'all_forecasts':all_forecasts,
        'stf_fc':stf_fc,'mtf_fc':mtf_fc,'htf_fc':htf_fc,
        'tier_stf':round(tier_stf,8) if tier_stf else None,
        'tier_mtf':round(tier_mtf,8) if tier_mtf else None,
        'tier_htf':round(tier_htf,8) if tier_htf else None,
        'best_target':round(best_target,8) if best_target else None,
        'hard_cap':round(hard_cap,8) if hard_cap else None,
        'hard_cap_floor':round(hard_cap_floor,8) if hard_cap_floor else None,
        'hard_cap_ceil':round(hard_cap_ceil,8) if hard_cap_ceil else None,
        'soft_stop':round(soft_stop,8) if soft_stop else None,
        'soft_stop_floor':round(soft_stop_floor,8) if soft_stop_floor else None,
        'soft_stop_ceil':round(soft_stop_ceil,8) if soft_stop_ceil else None,
        'mid_threshold':mid_1m,'ml_horizon_bars':ML_HORIZON,
    }


def print_ml_report(ml_result, label_map):
    if ml_result is None: print('  ML: analysis unavailable.\n'); return
    pair=ml_result['pair']; lbl=label_map.get(pair,pair.replace('USDC','')); cp=ml_result['current_price']; w=66
    def pf(v,dp=8):
        if v is None: return '—'
        return f'{v:.6f}' if v<1 else f'{v:.4f}'
    def pp(v,base=None):
        if v is None: return '—'
        s=pf(v)
        if base:
            pct=(v-base)/base*100.0; s+=f'  ({pct:+.2f}%)'
        return s
    print(f'\n  {"═"*w}'); print(f'  ◈  ML COMPOUND FORECAST  ·  {lbl}  ({pair})')
    print(f'  {"═"*w}'); print(f'  Entry price   : {pf(cp)}   |  Horizon: {ml_result["ml_horizon_bars"]} bars (1m)\n')
    rc=ml_result.get('reg_channel') or {}
    print(f'  ┌─ 1. LINEAR REGRESSION CHANNEL  (last {ML_LOOKBACK} bars) {"─"*10}┐')
    if rc:
        cp_label='above mid' if cp>rc.get('mid_line',cp) else ('below mid' if cp<rc.get('mid_line',cp) else 'at mid')
        band_pos=rc.get('channel_pos',0.0)
        pos_lbl=('ABOVE UPPER — extended' if band_pos>1.0 else 'BELOW LOWER — compressed ← dip zone' if band_pos<0.0 else f'pos={band_pos:.2f} (0=lower, 1=upper)')
        print(f'  │  Slope            : {rc["slope"]:.10f}  (per bar)')
        print(f'  │  Upper band (±2σ) : {pf(rc["upper_band"])}')
        print(f'  │  Mid line         : {pf(rc["mid_line"])}   ← {cp_label}')
        print(f'  │  Lower band (±2σ) : {pf(rc["lower_band"])}')
        print(f'  │  Residual σ       : {rc["residual_std"]}')
        print(f'  │  Channel position : {pos_lbl}')
        print(f'  │  ── Next-bar projection ──────────────────────────────────')
        print(f'  │  Next upper       : {pp(rc["next_upper"],cp)}')
        print(f'  │  Next mid  ← ML  : {pp(rc["next_mid"],cp)}')
        print(f'  │  Next lower       : {pp(rc["next_lower"],cp)}')
    print(f'  └{"─"*w}┘\n')
    am=ml_result.get('argmin_data',{})
    print(f'  ┌─ 2. ARGMIN / ARGMAX PRICE THRESHOLDS  (last {ML_LOOKBACK} bars) {"─"*4}┐')
    hdr2=f'  │  {"TF":>3}  {"ArgMin":>7}  {"ArgMax":>7}  {"Min Price":>14}  {"Mid Price":>14}  {"Max Price":>14}  {"dist":>6}  │'
    sep2='  │'+'─'*(len(hdr2)-4)+'│'
    print(sep2); print(hdr2); print(sep2)
    for tf_lbl in ('1m','3m','5m'):
        t=am.get(tf_lbl,{})
        if not t: continue
        amin=t['argmin']; amax=t['argmax']; dist=amin-amax; tick='✔ ' if dist>0 else '✗ '
        print(f'  │  {tf_lbl:>3}  {amin:>7}  {amax:>7}  {pf(t["min_price"]):>14}  {pf(t["mid_price"]):>14}  {pf(t["max_price"]):>14}  {tick}{dist:>4}  │')
    if am.get('1m'):
        mn=am['1m']['min_price']; mx=am['1m']['max_price']; mid=am['1m']['mid_price']
        pct_from_min=(cp-mn)/(mx-mn+1e-20)*100.0
        print(sep2); print(f'  │  Current {pf(cp)} is {pct_from_min:.1f}% from min to max   (mid={pf(mid)})  │')
    print(f'  └{"─"*w}┘\n')
    mets=ml_result.get('metrics',{})
    print(f'  ┌─ 4. INSTANT BACKTEST  (80% train / 20% test, horizon={ML_HORIZON}b) {"─"*2}┐')
    hdr4=f'  │  {"Model":<12}  {"MAE":>12}  {"RMSE":>12}  {"R²":>7}  {"DirAcc":>8}  │'
    sep4='  │'+'─'*(len(hdr4)-4)+'│'
    print(sep4); print(hdr4); print(sep4)
    sorted_mets=sorted(mets.items(),key=lambda x:x[1]['mae'])
    for nm,m in sorted_mets:
        if m['mae']>1e8: continue
        print(f'  │  {nm:<12}  {m["mae"]:>12.8f}  {m["rmse"]:>12.8f}  {m["r2"]:>7.4f}  {m["dir_acc"]:>7.1%}  │')
    print(sep4)
    best_name=sorted_mets[0][0] if sorted_mets else '—'; best_dacc=sorted_mets[0][1]['dir_acc'] if sorted_mets else 0.0
    print(f'  │  Best model: {best_name:<10}   Directional accuracy: {best_dacc:.1%}       │')
    print(f'  └{"─"*w}┘\n')
    indiv=ml_result.get('indiv_forecasts',{}) or {}
    print(f'  ┌─ 5. INDIVIDUAL MODEL POINT FORECASTS  (+{ML_HORIZON} bars ahead) {"─"*4}┐')
    hdr5=f'  │  {"Model":<12}  {"Forecast":>14}  {"Δ from entry":>14}  {"Δ%":>8}  │'
    sep5='  │'+'─'*(len(hdr5)-4)+'│'
    print(sep5); print(hdr5); print(sep5)
    for nm,fc in sorted(indiv.items(),key=lambda x:x[1]):
        delta=fc-cp; delta_p=delta/cp*100.0; tag=' ▲' if delta>0 else ' ▼'
        print(f'  │  {nm:<12}  {pf(fc):>14}  {delta:>+14.8f}  {delta_p:>+7.2f}%{tag}  │')
    ens=ml_result.get('ens_forecast')
    if ens:
        delta=ens-cp; delta_p=delta/cp*100.0; print(sep5)
        print(f'  │  {"★ Ensemble":<12}  {pf(ens):>14}  {delta:>+14.8f}  {delta_p:>+7.2f}%  │')
    print(f'  └{"─"*w}┘\n')
    mc=ml_result.get('mc') or {}
    print(f'  ┌─ 6. RANDOM WALK MONTE CARLO  ({ML_WALKS:,} paths · 3 horizons) {"─"*14}┐')
    if mc:
        prob_up=mc.get('prob_up',0.0); bar_mc='█'*int(prob_up*20)+'░'*(20-int(prob_up*20))
        print(f'  │  Drift applied    : {mc.get("drift_applied",0):+.8f}/bar  (raw={mc.get("raw_drift",0):+.8f}  vol={mc.get("vol_per_bar",0):.8f})')
        print(f'  │  P(short UP)      : {prob_up:.1%}  [{bar_mc}]')
        hdr_mc=f'  │  {"Horizon":<18}  {"p5":>12}  {"p25":>12}  {"p50 (median)":>14}  {"p75":>12}  {"p95":>12}  │'
        sep_mc='  │'+'─'*(len(hdr_mc)-4)+'│'
        print(sep_mc); print(hdr_mc); print(sep_mc)
        for tier_lbl,key in [(f'Short  ({mc.get("horizon_s","?")}b)','short'),(f'Medium ({mc.get("horizon_m","?")}b)','medium'),(f'Long   ({mc.get("horizon_l","?")}b)','long')]:
            t_=mc.get(key) or {}
            if not t_: continue
            print(f'  │  {tier_lbl:<18}  {pp(t_.get("p5"),cp):>12}  {pp(t_.get("p25"),cp):>12}  {pp(t_.get("p50"),cp):>14}  {pp(t_.get("p75"),cp):>12}  {pp(t_.get("p95"),cp):>12}  │')
        print(sep_mc)
    print(f'  └{"─"*w}┘\n')
    bt=ml_result.get('best_target'); t_stf=ml_result.get('tier_stf'); t_mtf=ml_result.get('tier_mtf')
    t_htf=ml_result.get('tier_htf'); hcap=ml_result.get('hard_cap'); stop=ml_result.get('soft_stop')
    mid_t=ml_result.get('mid_threshold')
    print(f'  {"═"*w}'); print(f'  ★★★  ML COMPOUND FINAL DECISION  ·  {lbl}'); print(f'  {"═"*w}')
    print(f'  ▶  ENTRY              : {pf(cp)}  (current 1m close)\n')
    if mid_t:
        dist_mid=(cp-mid_t)/cp*100.0 if cp>mid_t else (mid_t-cp)/cp*100.0
        side='BELOW' if cp<mid_t else 'ABOVE'
        zone='dip zone ← good entry' if cp<mid_t else 'elevated vs 500-bar range'
        print(f'  ▶  MID THRESHOLD      : {pf(mid_t)}  ({side} by {dist_mid:.2f}%) [{zone}]\n')
    for tier_lbl,tv,desc in [('STF TARGET  (~30m)',t_stf,'ML models + FFT 1m/3m/5m + MC short'),
                               ('MTF TARGET  (~2h)', t_mtf,'FFT 15m/30m + MC medium + φ near Fib'),
                               ('HTF TARGET  (swing)',t_htf,'FFT 1h/2h + MC long p75/p95 + φ ext + RegCh upper')]:
        if tv:
            up_pct=(tv-cp)/cp*100.0; arrow='▲' if tv>cp else '▼'
            print(f'  ▶  {tier_lbl:<22}: {pf(tv)}  ({arrow}{up_pct:+.2f}%)  [{desc}]')
        else:
            print(f'  ▶  {tier_lbl:<22}: —  (insufficient data)')
    print()
    if hcap:
        cap_pct=(hcap-cp)/cp*100.0
        print(f'  ▶  HARD CAP (vol wall): {pf(hcap)}  (+{cap_pct:.2f}%)  ← nearest resistance')
    if stop:
        stop_pct=(cp-stop)/cp*100.0
        print(f'  ▶  SOFT STOP LOSS     : {pf(stop)}  (-{stop_pct:.2f}% from entry)  [vol support floor]')
    target_rr=t_htf or t_mtf or t_stf
    if target_rr and stop:
        reward=target_rr-cp; risk=cp-stop; rr=reward/risk if risk>0 else 0.0
        grade=('EXCELLENT ★★★' if rr>=3.0 else 'GOOD ★★' if rr>=2.0 else 'OK ★' if rr>=1.5 else 'POOR')
        print(f'\n  ▶  RISK / REWARD      : {rr:.2f}  ({grade})')
        print(f'     Reward +{(reward/cp*100):.2f}%  |  Risk -{(risk/cp*100):.2f}%')
    print(f'\n  {"═"*w}\n')


# ── φ-Reversal and Circuit functions ─────────────────────

PHI_NEG_POWERS  = np.array([PHI ** -n for n in range(1, 8)])
PHI_NEG_LABELS  = [f'φ⁻{n} ({PHI**-n:.4f})' for n in range(1, 8)]
PHI_EXT_POWERS  = [(PHI, 'φ¹  (1.618 ext)'), (PHI2, 'φ²  (2.618 ext)')]
_GT_BASE_ANG    = 72.0; _GT_APEX_ANG = 36.0
_GT_HEIGHT_MULT = np.sin(np.radians(_GT_BASE_ANG)) / (2.0 * np.sin(np.radians(_GT_APEX_ANG)))

def _phi_bands_all(swing_low, swing_high, direction='up'):
    span=swing_high-swing_low
    if span<=0 or swing_low<=0: return []
    bands=[]
    for n,label in enumerate(PHI_NEG_LABELS,start=1):
        ratio=PHI**-n
        price=swing_low+ratio*span if direction=='up' else swing_high-ratio*span
        bands.append((label,ratio,round(float(price),8)))
    for ratio,label in PHI_EXT_POWERS:
        price=swing_low+ratio*span if direction=='up' else swing_high-ratio*span
        bands.append((label,ratio,round(float(price),8)))
    return bands

def _nearest_phi_band(current,bands):
    if not bands: return None,None,None,999.0
    best=min(bands,key=lambda b:abs(b[2]-current))
    dist=abs(best[2]-current)/(current+1e-20)*100.0
    return best[0],best[1],best[2],dist

def _golden_triangle_targets(pivot,bar_range,direction='up'):
    apex_h=bar_range*_GT_HEIGHT_MULT; gnomon=bar_range*PHI2; sign=1.0 if direction=='up' else -1.0
    return {'T1_primary':round(pivot+sign*apex_h,8),'T2_gnomon':round(pivot+sign*gnomon,8),'T1_retrace':round(pivot-sign*apex_h,8)}

def _spiral_windows(anchor_bar,n=7):
    return np.array([anchor_bar+round(PHI**k) for k in range(1,n+1)],dtype=int)

def _in_spiral_window(bar,windows,tol=1):
    return bool(np.any(np.abs(windows.astype(int)-int(bar))<=tol))

def phi_reversal_forecast(pair, current_price, sel_detail, order=7, phi_band_tol_pct=1.2, min_confidence=0.25):
    result={
        'pair':pair,'current_price':current_price,'trend':'NEUTRAL','direction':'—',
        'forecast_price':None,'target_T1':None,'target_T2':None,
        'phi_band_label':'—','phi_band_level':None,'phi_score':0.0,
        'spiral_ok':False,'confidence':0.0,'argmin_bar':None,'argmax_bar':None,
        'swing_low':None,'swing_high':None,'phi_bands_above':[],'phi_bands_below':[],'all_signals':[],'n_extrema':0,'error':None,
    }
    try:
        d=sel_detail.get(pair,{})
        close_arr=d.get('close_arr'); low_arr=d.get('low_arr'); high_arr=d.get('high_arr')
        if close_arr is None or len(close_arr)<50:
            try:
                klines=trader.client.get_klines(symbol=pair,interval='1m',limit=EXTREMA_LOOKBACK)
                close_arr=np.array([float(k[4]) for k in klines],dtype=np.float64)
                low_arr=np.array([float(k[3]) for k in klines],dtype=np.float64)
                high_arr=np.array([float(k[2]) for k in klines],dtype=np.float64)
            except Exception as ex: result['error']=str(ex); return result
        close_arr=np.asarray(close_arr,dtype=np.float64)
        low_arr=np.asarray(low_arr if low_arr is not None else close_arr,dtype=np.float64)
        high_arr=np.asarray(high_arr if high_arr is not None else close_arr,dtype=np.float64)
        n=len(close_arr)
        swing_low=d.get('swing_low') or float(np.min(low_arr))
        swing_high=d.get('swing_high') or float(np.max(high_arr))
        result['swing_low']=round(swing_low,8); result['swing_high']=round(swing_high,8)
        global_amin=int(np.argmin(low_arr)); global_amax=int(np.argmax(high_arr))
        result['argmin_bar']=global_amin; result['argmax_bar']=global_amax
        base_dir='up' if global_amin>global_amax else ('down' if global_amax>global_amin else 'up')
        signals=[]
        if SCIPY_AVAILABLE and n>=order*3:
            local_lows=_argrelextrema(low_arr,np.less,order=order)[0]
            local_highs=_argrelextrema(high_arr,np.greater,order=order)[0]
            result['n_extrema']=len(local_lows)+len(local_highs)
            extrema=([(int(i),float(low_arr[i]),'low') for i in local_lows]+[(int(i),float(high_arr[i]),'high') for i in local_highs])
            extrema.sort(key=lambda x:x[0])
            for i in range(1,len(extrema)):
                prev_bar,prev_px,prev_kind=extrema[i-1]; curr_bar,curr_px,curr_kind=extrema[i]
                lo_=min(prev_px,curr_px); hi_=max(prev_px,curr_px)
                if hi_<=lo_: continue
                sig_dir='BUY' if curr_kind=='low' else 'SELL'
                phi_dir='up' if curr_kind=='low' else 'down'
                pivot=curr_px
                bands=_phi_bands_all(lo_,hi_,direction=phi_dir)
                lbl_,ratio,level,dist_pct=_nearest_phi_band(curr_px,bands)
                if dist_pct>phi_band_tol_pct or lbl_ is None: continue
                phi_score=max(0.0,1.0-dist_pct/phi_band_tol_pct)
                bar_range_=hi_-lo_; gt=_golden_triangle_targets(pivot,bar_range_,phi_dir)
                windows=_spiral_windows(prev_bar,n=7); spiral_ok=_in_spiral_window(curr_bar,windows,tol=order)
                band_score=max(0.0,1.0-dist_pct/1.0); spiral_score=1.0 if spiral_ok else 0.0
                confidence=round(0.50*band_score+0.30*spiral_score+0.20*phi_score,4)
                if confidence<min_confidence: continue
                signals.append({'bar':curr_bar,'price':curr_px,'direction':sig_dir,'phi_band_label':lbl_,'phi_band_level':level,'phi_score':round(phi_score,4),'spiral_ok':spiral_ok,'T1_primary':gt['T1_primary'],'T2_gnomon':gt['T2_gnomon'],'T1_retrace':gt['T1_retrace'],'confidence':confidence})
        result['all_signals']=sorted(signals,key=lambda s:s['bar'])
        result['direction']='BUY' if base_dir=='up' else 'SELL'
        result['trend']='UP' if base_dir=='up' else 'DOWN'
        all_bands=_phi_bands_all(swing_low,swing_high,direction=base_dir)
        above_bands=sorted([(lbl2,lvl) for lbl2,_,lvl in all_bands if lvl>current_price*1.001],key=lambda x:x[1])
        below_bands=sorted([(lbl2,lvl) for lbl2,_,lvl in all_bands if lvl<current_price*0.999],key=lambda x:x[1],reverse=True)
        result['phi_bands_above']=above_bands[:4]; result['phi_bands_below']=below_bands[:4]
        lbl_now,_,lvl_now,dist_now=_nearest_phi_band(current_price,all_bands)
        result['phi_band_label']=lbl_now or '—'; result['phi_band_level']=lvl_now
        result['phi_score']=round(max(0.0,1.0-dist_now/phi_band_tol_pct),4)
        bar_range_global=swing_high-swing_low; gt_global=_golden_triangle_targets(current_price,bar_range_global,base_dir)
        result['target_T1']=gt_global['T1_primary']; result['target_T2']=gt_global['T2_gnomon']
        anchor_bar=global_amin if base_dir=='up' else global_amax
        windows_now=_spiral_windows(anchor_bar,n=7); result['spiral_ok']=_in_spiral_window(n-1,windows_now,tol=order)
        band_s=max(0.0,1.0-dist_now/1.0); spi_s=1.0 if result['spiral_ok'] else 0.0
        result['confidence']=round(0.50*band_s+0.30*spi_s+0.20*result['phi_score'],4)
        if base_dir=='up' and above_bands: targets=[lvl for _,lvl in above_bands[:3]]
        elif base_dir=='down' and below_bands: targets=[lvl for _,lvl in below_bands[:3]]
        else: targets=[]
        result['forecast_price']=round(float(np.median(targets)),8) if targets else result['target_T1']
    except Exception as ex: result['error']=f'{type(ex).__name__}: {ex}'
    return result

def print_phi_reversal_block(rev, label_map):
    if not rev: return
    pair=rev.get('pair','?'); lbl=label_map.get(pair,pair.replace('USDC','')); cp=rev.get('current_price') or 0.0; w=66
    def pf(v):
        if v is None: return '—'
        return f'{v:.6f}' if abs(v)<1 else f'{v:.4f}'
    def pp(v):
        if v is None: return '—'
        pct=(v-cp)/(cp+1e-20)*100.0; arrow='▲' if pct>0 else '▼'
        return f'{pf(v)}  ({arrow}{pct:+.2f}%)'
    trend=rev.get('trend','NEUTRAL'); conf=rev.get('confidence',0.0)
    spiral_ok=rev.get('spiral_ok',False); phi_score=rev.get('phi_score',0.0); n_ext=rev.get('n_extrema',0)
    trend_icon=('▲ UP  (BUY reversal setup)' if trend=='UP' else ('▼ DOWN (SELL reversal setup)' if trend=='DOWN' else '→ NEUTRAL'))
    conf_bar='█'*int(conf*20)+'░'*(20-int(conf*20))
    print(f'\n  {"═"*w}'); print(f'  ◈  φ-REVERSAL FORECAST  ·  {lbl}  ({pair})'); print(f'  {"═"*w}')
    if rev.get('error'): print(f'  [WARN] {rev["error"]}')
    print(f'  ┌─ TREND & REVERSAL DIRECTION {"─"*36}┐')
    print(f'  │  Entry price    : {pf(cp)}')
    print(f'  │  Swing low      : {pf(rev.get("swing_low"))}   argmin@bar {rev.get("argmin_bar","—")}')
    print(f'  │  Swing high     : {pf(rev.get("swing_high"))}   argmax@bar {rev.get("argmax_bar","—")}')
    print(f'  │  TREND          : {trend_icon}')
    print(f'  │  Extrema found  : {n_ext} local swing points')
    print(f'  └{"─"*w}┘\n')
    print(f'  ┌─ φ DECAY BANDS  (negative exponential powers of φ) {"─"*12}┐')
    print(f'  │  φ⁻¹…φ⁻⁷ = {np.round(PHI_NEG_POWERS,4).tolist()}')
    print(f'  │  Nearest band   : {rev.get("phi_band_label","—")}')
    print(f'  │  Band level     : {pf(rev.get("phi_band_level"))}')
    print(f'  │  φ score        : {phi_score:.4f}  (1.0 = exactly on band)')
    if rev.get('phi_bands_above'):
        print(f'  │  ── φ levels ABOVE entry (targets) ───────────────────────')
        for band_lbl,band_lvl in rev['phi_bands_above']: print(f'  │    {band_lbl:<32}  {pp(band_lvl)}')
    if rev.get('phi_bands_below'):
        print(f'  │  ── φ levels BELOW entry (support / stop ref) ─────────────')
        for band_lbl,band_lvl in rev['phi_bands_below']: print(f'  │    {band_lbl:<32}  {pp(band_lvl)}')
    print(f'  └{"─"*w}┘\n')
    print(f'  ┌─ GOLDEN TRIANGLE TARGETS  (apex=36°  base=72°  leg/base=φ) {"─"*4}┐')
    print(f'  │  T1 primary  (apex height)      : {pp(rev.get("target_T1"))}')
    print(f'  │  T2 gnomon   (φ² × bar range)   : {pp(rev.get("target_T2"))}')
    print(f'  └{"─"*w}┘\n')
    spi_str='✔  YES — price bar is inside a φ-spiral timing window' if spiral_ok else '✗  No   — bar outside spiral windows'
    print(f'  ┌─ GOLDEN SPIRAL TIMING  (r = A·e^(b·θ), b = ln(φ)/(π/2)) {"─"*4}┐')
    anchor_bar=rev.get('argmin_bar') if trend=='UP' else rev.get('argmax_bar')
    if anchor_bar is not None:
        wins=_spiral_windows(int(anchor_bar),n=7); wins_str=', '.join(str(w2) for w2 in wins)
        print(f'  │    anchor bar {anchor_bar} → windows at [{wins_str}]')
    print(f'  │  Current in window : {spi_str}')
    print(f'  └{"─"*w}┘\n')
    sigs=rev.get('all_signals',[])
    if sigs:
        print(f'  ┌─ φ-REVERSAL SIGNALS  ({len(sigs)} signals, conf≥{0.25}) {"─"*24}┐')
        hdr_s=f'  │  {"Bar":>5}  {"Price":>12}  {"Dir":>4}  {"φ Band":<24}  {"T1 target":>12}  {"T2 target":>12}  {"Conf":>6}  │'
        sep_s='  │'+'─'*(len(hdr_s)-4)+'│'
        print(sep_s); print(hdr_s); print(sep_s)
        for s in sorted(sigs,key=lambda s:s['confidence'],reverse=True)[:5]:
            sp_tag='✔' if s['spiral_ok'] else '·'
            print(f'  │  {s["bar"]:>5}  {pf(s["price"]):>12}  {s["direction"]:>4}  {s["phi_band_label"][:24]:<24}  {pf(s["T1_primary"]):>12}  {pf(s["T2_gnomon"]):>12}  {s["confidence"]:>5.3f}{sp_tag}  │')
        print(sep_s); print(f'  └{"─"*w}┘\n')
    fc_p=rev.get('forecast_price'); t1_p=rev.get('target_T1'); t2_p=rev.get('target_T2')
    if fc_p:
        fc_pct=(fc_p-cp)/(cp+1e-20)*100.0; fc_arr='▲' if fc_pct>0 else '▼'
    else:
        fc_pct=0.0; fc_arr=''
    print(f'  {"═"*w}'); print(f'  ★★★  φ-REVERSAL DECISION  ·  {lbl}'); print(f'  {"═"*w}')
    print(f'  ▶  ENTRY               : {pf(cp)}'); print(f'  ▶  TREND               : {trend_icon}\n')
    print(f'  ▶  φ FORECAST TARGET   : {pf(fc_p)}  ({fc_arr}{fc_pct:+.2f}%)  [median of nearest φ bands]')
    print(f'  ▶  T1  (triangle)      : {pp(t1_p)}  [apex·height = bar_range × {_GT_HEIGHT_MULT:.4f}]')
    print(f'  ▶  T2  (φ² gnomon)    : {pp(t2_p)}  [bar_range × φ² = {PHI2:.4f}]\n')
    print(f'  ▶  CONFIDENCE          : {conf:.4f}  [{conf_bar}]')
    print(f'     φ band proximity    : {rev.get("phi_score",0.0):.4f} × 0.50')
    print(f'     spiral timing       : {"1.00" if spiral_ok else "0.00"} × 0.30\n')
    grade=('STRONG REVERSAL SETUP ★★★' if conf>=0.75 else 'MODERATE SETUP ★★' if conf>=0.55 else 'WEAK SETUP ★' if conf>=0.35 else 'LOW CONFIDENCE — wait for better alignment')
    print(f'  ▶  GRADE               : {grade}\n'); print(f'  {"═"*w}\n')


def _fit_sinusoid_to_price(close_arr, dominant_period):
    n=len(close_arr); arr=np.asarray(close_arr,dtype=np.float64); t=np.arange(n,dtype=np.float64)
    p=np.polyfit(t,arr,1); trend=np.poly1d(p)(t); detr=arr-trend
    T=float(dominant_period); k=int(round(n/T)); k=max(1,min(k,len(np.fft.rfft(detr))-1))
    sp=np.fft.rfft(detr); A=2.0*abs(sp[k])/n; phi0=float(np.angle(sp[k]))
    fitted=A*np.sin(TAU/T*t+phi0)+trend
    ss_res=float(np.sum((arr-fitted)**2)); ss_tot=float(np.sum((arr-arr.mean())**2))
    r2=max(0.0,1.0-ss_res/ss_tot) if ss_tot>1e-12 else 0.0
    return float(A),float(phi0),float(p[0]),float(p[1]),round(r2,4),fitted

def _circuit_angle_from_fft(bar_idx, dominant_period, argmin_bar):
    T=float(dominant_period); raw=TAU/T*(bar_idx-argmin_bar)
    theta=raw%TAU; return float(np.degrees(theta))

def _assign_quadrant(angle_deg):
    a=angle_deg%360.0
    if   a<90.0:  return 'Q1'
    elif a<180.0: return 'Q2'
    elif a<270.0: return 'Q3'
    else:         return 'Q4'

def _sinusoid_price_at_bar(bar_idx, A, phi0, slope, intercept, dominant_period):
    t=float(bar_idx); T=float(dominant_period)
    return float(slope*t+intercept+A*np.sin(TAU/T*t+phi0))

def _quadrature_bars(anchor_bar, dominant_period, cycle_dir='UP', A=None, phi0=None, slope=None, intercept=None, swing_low=None, swing_high=None):
    T=float(dominant_period); has_fit=(A is not None and phi0 is not None and slope is not None and intercept is not None)
    def _proj(bar):
        if not has_fit: return None
        raw=_sinusoid_price_at_bar(bar,A,phi0,slope,intercept,T)
        if swing_low is not None and swing_high is not None: raw=max(swing_low*0.99,min(swing_high*1.01,raw))
        return round(float(raw),8)
    if cycle_dir=='UP':
        b_q1=int(anchor_bar); b_q2=int(anchor_bar+T/4.0); b_q3=int(anchor_bar+T/2.0)
        b_q4=int(anchor_bar+3.0*T/4.0); b_end=int(anchor_bar+T)
        return {'cycle_dir':cycle_dir,'Q1_start':b_q1,'Q1_price':_proj(b_q1),'Q1_label':'TROUGH  (entry / support)',
                'Q2_start':b_q2,'Q2_price':_proj(b_q2),'Q2_label':'rising midline (90°)',
                'Q3_start':b_q3,'Q3_price':_proj(b_q3),'Q3_label':'★ RESISTANCE TOP — reversal target',
                'Q4_start':b_q4,'Q4_price':_proj(b_q4),'Q4_label':'post-reversal decline (270°)',
                'next_trough':b_end,'next_trough_price':_proj(b_end),'next_trough_label':'next TROUGH / cycle restart'}
    else:
        b_q3=int(anchor_bar); b_q4=int(anchor_bar+T/4.0); b_q1=int(anchor_bar+T/2.0)
        b_q2=int(anchor_bar+3.0*T/4.0); b_end=int(anchor_bar+T)
        return {'cycle_dir':cycle_dir,'Q3_start':b_q3,'Q3_price':_proj(b_q3),'Q3_label':'PEAK  (entry / resistance)',
                'Q4_start':b_q4,'Q4_price':_proj(b_q4),'Q4_label':'early collapse (270°)',
                'Q1_start':b_q1,'Q1_price':_proj(b_q1),'Q1_label':'★ SUPPORT DIP — reversal target',
                'Q2_start':b_q2,'Q2_price':_proj(b_q2),'Q2_label':'post-reversal accumulation (90°)',
                'next_peak':b_end,'next_peak_price':_proj(b_end),'next_peak_label':'next PEAK / cycle restart'}

_MTF_CIRCUIT_TFS=[('1m',60*16,1.0),('3m',60*8,0.9),('5m',60*5,0.85),('15m',60*3,0.75),('30m',60*2,0.65),('2h',60*1,0.50)]

def _run_circuit_on_tf(pair, tf, limit, current_price):
    r={'tf':tf,'pair':pair,'current_price':current_price,'cycle_dir':None,'current_angle_deg':None,'current_quadrant':None,'quadrant_label':'—','dominant_period':None,'sinusoid_r2':None,'reversal_type':'—','reversal_target':None,'reversal_pct':None,'reversal_target_fft':None,'reversal_target_swing':None,'phi_target_ext':None,'bars_to_reversal':None,'swing_low':None,'swing_high':None,'fft_amplitude':None,'quadrature_bars':{},'error':None}
    try:
        klines=trader.client.get_klines(symbol=pair,interval=tf,limit=limit)
        close_raw=np.array([float(k[4]) for k in klines],dtype=np.float64)
        low_raw=np.array([float(k[3]) for k in klines],dtype=np.float64)
        high_raw=np.array([float(k[2]) for k in klines],dtype=np.float64)
        if len(close_raw)<64:
            klines=trader.client.get_klines(symbol=pair,interval=tf,limit=1000)
            close_raw=np.array([float(k[4]) for k in klines],dtype=np.float64)
            low_raw=np.array([float(k[3]) for k in klines],dtype=np.float64)
            high_raw=np.array([float(k[2]) for k in klines],dtype=np.float64)
        _valid=(np.isfinite(close_raw)&(close_raw>0)&np.isfinite(low_raw)&(low_raw>0)&np.isfinite(high_raw)&(high_raw>0))
        close_arr=close_raw[_valid]; low_arr=low_raw[_valid]; high_arr=high_raw[_valid]; n=len(close_arr)
        if n<16: r['error']='insufficient data'; return r
        swing_low=float(np.min(low_arr)); swing_high=float(np.max(high_arr))
        r['swing_low']=round(swing_low,8); r['swing_high']=round(swing_high,8)
        global_amin=int(np.argmin(low_arr)); global_amax=int(np.argmax(high_arr))
        r['argmin_bar']=global_amin; r['argmax_bar']=global_amax
        cycle_dir='UP' if global_amin>global_amax else 'DOWN'
        r['cycle_dir']=cycle_dir; anchor_bar=global_amin if cycle_dir=='UP' else global_amax; current_bar=n-1
        _dtr,_=_detrend(close_arr); _sp=np.fft.rfft(_dtr); _fr=np.fft.rfftfreq(n)
        _pw=np.abs(_sp); _pw[0]=0; _vm=(_fr>0)&(_fr<=0.25)
        if np.any(_vm):
            _pw2=_pw.copy(); _pw2[~_vm]=0; _dom=int(np.argmax(_pw2)); _df=float(_fr[_dom])
            dominant_period=max(8,int(round(1.0/_df))) if _df>0 else 20
        else: dominant_period=20
        dominant_period=min(dominant_period,n//2); r['dominant_period']=dominant_period
        A_fit,phi0,slope,intercept,r2_fit,_=_fit_sinusoid_to_price(close_arr,dominant_period)
        r['sinusoid_r2']=round(r2_fit,4); r['fft_amplitude']=round(float(A_fit),8)
        angle_deg=_circuit_angle_from_fft(current_bar,dominant_period,anchor_bar)
        if cycle_dir=='DOWN':
            down_angle=_circuit_angle_from_fft(current_bar,dominant_period,global_amax)
            angle_deg=(down_angle+180.0)%360.0
        r['current_angle_deg']=round(angle_deg,1); quadrant=_assign_quadrant(angle_deg)
        r['current_quadrant']=quadrant; r['quadrant_label']=(_CQ_LABEL_UP if cycle_dir=='UP' else _CQ_LABEL_DN).get(quadrant,'—')
        r['quadrature_bars']=_quadrature_bars(anchor_bar,dominant_period,cycle_dir=cycle_dir,A=A_fit,phi0=phi0,slope=slope,intercept=intercept,swing_low=swing_low,swing_high=swing_high)
        T=float(dominant_period); rev_bar=int(round(anchor_bar+T/2.0))
        fft_proj=_sinusoid_price_at_bar(rev_bar,A_fit,phi0,slope,intercept,T)
        r['reversal_target_fft']=round(float(fft_proj),8)
        if cycle_dir=='UP':
            r['reversal_type']='RESISTANCE TOP'; emp_price=float(high_arr[global_amax])
            raw_target=0.40*swing_high+0.40*fft_proj+0.20*emp_price
            raw_target=max(current_price,min(swing_high*1.05,raw_target))
            r['reversal_target_swing']=round(swing_high,8); span=swing_high-swing_low
            r['phi_target_ext']=round(swing_high+PHI_INV*span,8)
        else:
            r['reversal_type']='SUPPORT DIP'; emp_price=float(low_arr[global_amin])
            raw_target=0.40*swing_low+0.40*fft_proj+0.20*emp_price
            raw_target=min(current_price,max(swing_low*0.95,raw_target))
            r['reversal_target_swing']=round(swing_low,8); span=swing_high-swing_low
            r['phi_target_ext']=round(swing_low-PHI_INV*span,8)
        r['reversal_target']=round(float(raw_target),8)
        r['reversal_pct']=round((raw_target-current_price)/(current_price+1e-20)*100.0,4)
        r['bars_to_reversal']=max(0,rev_bar-current_bar); r['reversal_bar_est']=rev_bar
    except Exception as ex: r['error']=f'{type(ex).__name__}: {ex}'
    return r

def sinusoidal_circuit_mtf(pair, current_price, sel_detail, stf_results=None, htf_results=None):
    tf_results=[]
    with ThreadPoolExecutor(max_workers=len(_MTF_CIRCUIT_TFS)) as ex:
        futs={ex.submit(_run_circuit_on_tf,pair,tf,lim,current_price):(tf,w) for tf,lim,w in _MTF_CIRCUIT_TFS}
        for fut in as_completed(futs):
            tf,w=futs[fut]
            try:
                res=fut.result(); res['weight']=w; tf_results.append(res)
            except Exception: pass
    tf_order={tf:i for i,(tf,_,_) in enumerate(_MTF_CIRCUIT_TFS)}
    tf_results.sort(key=lambda r:tf_order.get(r.get('tf'),99))
    valid=[r for r in tf_results if r.get('reversal_target') is not None and (r.get('sinusoid_r2') or 0)>0.2 and r.get('error') is None]
    mtf_summary={'n_valid_tfs':len(valid),'mtf_target':None,'mtf_target_pct':None,'mtf_phi_ext':None,'mtf_confidence':0.0,'dominant_cycle_dir':None,'dominant_quadrant':None,'target_range_low':None,'target_range_high':None,'best_r2_tf':None}
    if valid:
        weights=np.array([r['sinusoid_r2']*r['weight'] for r in valid],dtype=np.float64)
        targets=np.array([r['reversal_target'] for r in valid],dtype=np.float64)
        phi_exts=np.array([r['phi_target_ext'] for r in valid if r.get('phi_target_ext') is not None],dtype=np.float64)
        w_sum=weights.sum(); mtf_target=float(np.average(targets,weights=weights)) if w_sum>0 else float(np.median(targets))
        mtf_summary['mtf_target']=round(mtf_target,8)
        mtf_summary['mtf_target_pct']=round((mtf_target-current_price)/(current_price+1e-20)*100.0,4)
        mtf_summary['target_range_low']=round(float(np.min(targets)),8); mtf_summary['target_range_high']=round(float(np.max(targets)),8)
        if len(phi_exts)>0: mtf_summary['mtf_phi_ext']=round(float(np.average(phi_exts,weights=weights[:len(phi_exts)])),8)
        up_w=sum(r['sinusoid_r2']*r['weight'] for r in valid if r.get('cycle_dir')=='UP')
        dn_w=sum(r['sinusoid_r2']*r['weight'] for r in valid if r.get('cycle_dir')=='DOWN')
        mtf_summary['dominant_cycle_dir']='UP' if up_w>=dn_w else 'DOWN'
        from collections import Counter
        q_votes=Counter(r.get('current_quadrant') for r in valid if r.get('current_quadrant'))
        mtf_summary['dominant_quadrant']=q_votes.most_common(1)[0][0] if q_votes else None
        mean_r2=float(np.mean([r['sinusoid_r2'] for r in valid]))
        agree_frac=max(up_w,dn_w)/(up_w+dn_w+1e-12)
        mtf_summary['mtf_confidence']=round(min(1.0,mean_r2*agree_frac*1.5),4)
        best_r2_tf=max(valid,key=lambda r:r.get('sinusoid_r2',0))
        mtf_summary['best_r2_tf']=best_r2_tf.get('tf')
    return tf_results, mtf_summary

def sinusoidal_circuit_engine(pair, current_price, sel_detail, stf_results=None, htf_results=None):
    result={'pair':pair,'current_price':current_price,'cycle_dir':'NEUTRAL','cycle_label':'—','current_angle_deg':None,'current_quadrant':None,'quadrant_label':'—','dominant_period':None,'fft_amplitude':None,'sinusoid_r2':None,'phi0_rad':None,'argmin_bar':None,'argmax_bar':None,'swing_low':None,'swing_high':None,'quadrature_bars':{},'reversal_type':'—','reversal_target':None,'reversal_bar_est':None,'bars_to_reversal':None,'reversal_pct':None,'absorption_flag':False,'exhaustion_flag':False,'absorption_score':0.0,'exhaustion_score':0.0,'vol_rule_label':'—','circuit_confidence':0.0,'confidence_detail':{},'phi_target_ext':None,'phi_target_label':'—','error':None}
    try:
        d=sel_detail.get(pair,{}); close_arr=d.get('close_arr'); low_arr=d.get('low_arr'); high_arr=d.get('high_arr')
        if close_arr is None or len(close_arr)<64:
            try:
                klines=trader.client.get_klines(symbol=pair,interval='1m',limit=EXTREMA_LOOKBACK)
                close_arr=np.array([float(k[4]) for k in klines],dtype=np.float64)
                low_arr=np.array([float(k[3]) for k in klines],dtype=np.float64)
                high_arr=np.array([float(k[2]) for k in klines],dtype=np.float64)
            except Exception as ex: result['error']=str(ex); return result
        close_arr=np.asarray(close_arr,dtype=np.float64)
        low_arr=np.asarray(low_arr if low_arr is not None else close_arr,dtype=np.float64)
        high_arr=np.asarray(high_arr if high_arr is not None else close_arr,dtype=np.float64)
        n=len(close_arr)
        swing_low=float(d.get('swing_low') or np.min(low_arr)); swing_high=float(d.get('swing_high') or np.max(high_arr))
        global_amin=int(np.argmin(low_arr)); global_amax=int(np.argmax(high_arr))
        result['argmin_bar']=global_amin; result['argmax_bar']=global_amax
        result['swing_low']=round(swing_low,8); result['swing_high']=round(swing_high,8)
        cycle_dir='UP' if global_amin>global_amax else ('DOWN' if global_amax>global_amin else 'UP')
        result['cycle_dir']=cycle_dir
        dominant_period=None; fft_amplitude=None; sinusoid_r2=None
        all_fft=(stf_results or [])+(htf_results or [])
        if all_fft:
            periods=[r['dominant_period'] for r in all_fft if r.get('dominant_period')]
            r2s=[r.get('sinusoid_r2') or 0.5 for r in all_fft if r.get('dominant_period')]
            if periods:
                dominant_period=int(round(np.average(periods,weights=r2s)))
                amps=[r.get('osc_amplitude',0) for r in (stf_results or [])]
                if amps: fft_amplitude=float(np.mean(amps))
                r2s_stf=[r.get('sinusoid_r2') for r in (stf_results or []) if r.get('sinusoid_r2') is not None]
                if r2s_stf: sinusoid_r2=float(np.mean(r2s_stf))
        if dominant_period is None or dominant_period<4:
            _dtr,_=_detrend(close_arr); _sp=np.fft.rfft(_dtr); _fr=np.fft.rfftfreq(n)
            _pw=np.abs(_sp); _pw[0]=0; _vm=(_fr>0)&(_fr<=0.25)
            if np.any(_vm):
                _pw2=_pw.copy(); _pw2[~_vm]=0; _dom=int(np.argmax(_pw2)); _df=float(_fr[_dom])
                dominant_period=max(8,int(round(1.0/_df))) if _df>0 else 20
            else: dominant_period=20
        dominant_period=min(dominant_period,n//2); result['dominant_period']=dominant_period
        A_fit,phi0,slope,intercept,r2_fit,fitted_arr=_fit_sinusoid_to_price(close_arr,dominant_period)
        if fft_amplitude is None: fft_amplitude=A_fit
        if sinusoid_r2 is None: sinusoid_r2=r2_fit
        result['fft_amplitude']=round(float(fft_amplitude),8); result['sinusoid_r2']=round(float(sinusoid_r2),4); result['phi0_rad']=round(float(phi0),6)
        anchor_bar=global_amin if cycle_dir=='UP' else global_amax; current_bar=n-1
        angle_deg=_circuit_angle_from_fft(current_bar,dominant_period,anchor_bar)
        if cycle_dir=='DOWN':
            down_angle=_circuit_angle_from_fft(current_bar,dominant_period,global_amax)
            angle_deg=(down_angle+180.0)%360.0
        result['current_angle_deg']=round(angle_deg,1); quadrant=_assign_quadrant(angle_deg); result['current_quadrant']=quadrant
        if cycle_dir=='UP':
            result['quadrant_label']=_CQ_LABEL_UP.get(quadrant,'—'); result['cycle_label']='UP cycle  (argmin most recent → SUPPORT DIP confirmed)'
        else:
            result['quadrant_label']=_CQ_LABEL_DN.get(quadrant,'—'); result['cycle_label']='DOWN cycle (argmax most recent → RESISTANCE TOP confirmed)'
        result['quadrature_bars']=_quadrature_bars(anchor_bar,dominant_period,cycle_dir=cycle_dir,A=A_fit,phi0=phi0,slope=slope,intercept=intercept,swing_low=swing_low,swing_high=swing_high)
        T=float(dominant_period); half_T=T/2.0
        if cycle_dir=='UP':
            result['reversal_type']='RESISTANCE TOP'; rev_bar=int(round(anchor_bar+half_T))
            fft_proj=_sinusoid_price_at_bar(rev_bar,A_fit,phi0,slope,intercept,T)
            emp_price=float(high_arr[global_amax])
            raw_target=max(current_price,min(swing_high*1.05,(0.40*swing_high+0.40*fft_proj+0.20*emp_price)))
        else:
            result['reversal_type']='SUPPORT DIP'; rev_bar=int(round(anchor_bar+half_T))
            fft_proj=_sinusoid_price_at_bar(rev_bar,A_fit,phi0,slope,intercept,T)
            emp_price=float(low_arr[global_amin])
            raw_target=min(current_price,max(swing_low*0.95,(0.40*swing_low+0.40*fft_proj+0.20*emp_price)))
        result['reversal_target_fft']=round(float(fft_proj),8); result['reversal_target_swing']=round(float(swing_high if cycle_dir=='UP' else swing_low),8); result['reversal_target_emp']=round(float(emp_price),8)
        result['reversal_bar_est']=rev_bar; result['bars_to_reversal']=max(0,rev_bar-current_bar); result['reversal_target']=round(raw_target,8)
        span=swing_high-swing_low
        phi_ext=swing_high+PHI_INV*span if cycle_dir=='UP' else swing_low-PHI_INV*span
        phi_lbl=f'+61.8% ext above swing_high → {phi_ext:.8g}' if cycle_dir=='UP' else f'−61.8% ext below swing_low  → {phi_ext:.8g}'
        result['phi_target_ext']=round(phi_ext,8); result['phi_target_label']=phi_lbl
        result['reversal_pct']=round((raw_target-current_price)/(current_price+1e-20)*100.0,4)
        abs_flag=False; abs_score=0.0; exhs_flag=False; exhs_score=0.0
        for r in (stf_results or []):
            if r.get('tf') in ('1m','3m'):
                abs_flag=abs_flag or r.get('absorption_flag',False); exhs_flag=exhs_flag or r.get('exhaustion_flag',False)
                abs_score=max(abs_score,r.get('absorption_score',0.0)); exhs_score=max(exhs_score,r.get('exhaustion_score',0.0))
        det_flow=d.get('delta_ratio',0.0)
        if det_flow and abs(float(det_flow))<0.05 and abs_score<3.0: abs_score=max(abs_score,3.0)
        result['absorption_flag']=abs_flag; result['exhaustion_flag']=exhs_flag; result['absorption_score']=round(abs_score,4); result['exhaustion_score']=round(exhs_score,4)
        if cycle_dir=='UP':
            if quadrant in ('Q3','Q4') and abs_flag: result['vol_rule_label']='🟡 ABSORPTION in distribution zone — smart money selling into rally'
            elif quadrant in ('Q3','Q4') and exhs_flag: result['vol_rule_label']='🔴 EXHAUSTION near resistance — buyer fuel spent, SELL reversal risk'
            elif quadrant in ('Q1','Q2') and abs_flag: result['vol_rule_label']='🟢 ABSORPTION at support — smart money buying, BUY continuation'
            else: result['vol_rule_label']='⚪ Neutral — no dominant order-flow signal'
        else:
            if quadrant in ('Q1','Q2') and abs_flag: result['vol_rule_label']='🟢 ABSORPTION at support dip — smart money buying, BUY reversal near'
            elif quadrant in ('Q1','Q2') and exhs_flag: result['vol_rule_label']='🔴 EXHAUSTION near support — seller fuel spent, BUY reversal risk'
            elif quadrant in ('Q3','Q4') and exhs_flag: result['vol_rule_label']='🟡 EXHAUSTION from peak — selling pressure waning, watch for floor'
            else: result['vol_rule_label']='⚪ Neutral — no dominant order-flow signal'
        conf=0.0; conf_d={}
        r2_contrib=float(sinusoid_r2)*0.25; conf+=r2_contrib; conf_d['sinusoid_r2']=round(r2_contrib,4)
        boundary_angles=[0.0,90.0,180.0,270.0,360.0]; min_dist=min(abs(angle_deg-b) for b in boundary_angles)
        prox_score=max(0.0,1.0-min_dist/90.0)*0.20; conf+=prox_score; conf_d['phase_proximity']=round(prox_score,4)
        vol_bonus=0.0
        if cycle_dir=='UP' and quadrant in ('Q3','Q4'):
            if abs_flag: vol_bonus+=0.20
            if exhs_flag: vol_bonus+=0.20
        elif cycle_dir=='DOWN' and quadrant in ('Q1','Q2'):
            if abs_flag: vol_bonus+=0.20
            if exhs_flag: vol_bonus+=0.20
        conf+=min(0.30,vol_bonus); conf_d['vol_signal']=round(min(0.30,vol_bonus),4)
        wins=_spiral_windows(global_amin if cycle_dir=='UP' else global_amax,n=7)
        spi_ok=_in_spiral_window(current_bar,wins,tol=7); spi_c=0.15 if spi_ok else 0.0
        conf+=spi_c; conf_d['spiral_timing']=round(spi_c,4)
        bars_to_rev=max(0,rev_bar-current_bar)
        close_bonus=0.10 if bars_to_rev<=10 else (0.05 if bars_to_rev<=20 else 0.0)
        conf+=close_bonus; conf_d['proximity_to_rev']=round(close_bonus,4)
        result['circuit_confidence']=round(min(1.0,conf),4); result['confidence_detail']=conf_d
    except Exception as ex: result['error']=f'{type(ex).__name__}: {ex}'
    return result

def print_circuit_block(circ, label_map, mtf_tf_results=None, mtf_summary=None):
    if not circ: return
    pair=circ.get('pair','?'); lbl=label_map.get(pair,pair.replace('USDC','')); cp=circ.get('current_price') or 0.0; w=70
    def pf(v):
        if v is None: return '—'
        return f'{v:.6f}' if abs(v)<1 else f'{v:.4f}'
    def pp(v,ref=None):
        if v is None: return '—'
        base=ref if ref is not None else cp; pct=(v-base)/(base+1e-20)*100.0; arr='▲' if pct>0 else '▼'
        return f'{pf(v)}  ({arr}{abs(pct):.2f}%)'
    err=circ.get('error'); cycle_dir=circ.get('cycle_dir','NEUTRAL'); quadrant=circ.get('current_quadrant','?')
    angle=circ.get('current_angle_deg'); conf=circ.get('circuit_confidence',0.0)
    conf_bar='█'*int(conf*20)+'░'*(20-int(conf*20))
    rev_type=circ.get('reversal_type','—'); rev_tgt=circ.get('reversal_target')
    rev_tgt_fft=circ.get('reversal_target_fft'); rev_tgt_swing=circ.get('reversal_target_swing'); rev_tgt_emp=circ.get('reversal_target_emp')
    rev_pct=circ.get('reversal_pct'); rev_bar=circ.get('reversal_bar_est'); bars_rem=circ.get('bars_to_reversal')
    T=circ.get('dominant_period'); r2=circ.get('sinusoid_r2'); amp=circ.get('fft_amplitude'); qbars=circ.get('quadrature_bars',{})
    abs_flag=circ.get('absorption_flag',False); exhs_flag=circ.get('exhaustion_flag',False); vol_lbl=circ.get('vol_rule_label','—')
    phi_ext=circ.get('phi_target_ext'); phi_lbl=circ.get('phi_target_label','—')
    amin_bar=circ.get('argmin_bar'); amax_bar=circ.get('argmax_bar'); slo=circ.get('swing_low'); shi=circ.get('swing_high'); conf_d=circ.get('confidence_detail',{})
    dir_icon=('▲ UP   — rising from SUPPORT DIP' if cycle_dir=='UP' else ('▼ DOWN — falling toward SUPPORT DIP' if cycle_dir=='DOWN' else '→ NEUTRAL'))
    def _arc(cur_q,cdir):
        labels=['Q1','Q2','Q3','Q4']; parts=['['+q+'▶]' if q==cur_q else ' '+q+' ' for q in labels]
        arc='─'.join(parts); lo='0°=argmin' if cdir=='UP' else '180°=argmax'; hi='180°=argmax' if cdir=='UP' else '360°→argmin'
        return f'  │  {lo} ── {arc} ── {hi}'
    print(f'\n  {"═"*w}'); print(f'  ◈  360° SINUSOIDAL CIRCUIT  ·  {lbl}  ({pair})'); print(f'  {"═"*w}')
    if err: print(f'  [WARN] {err}')
    print(f'  ┌─ CYCLE IDENTIFICATION {"─"*46}┐')
    print(f'  │  Entry price        : {pf(cp)}')
    print(f'  │  Swing low (argmin) : {pf(slo)}   @ bar {amin_bar}')
    print(f'  │  Swing high (argmax): {pf(shi)}   @ bar {amax_bar}')
    print(f'  │  Cycle direction    : {dir_icon}')
    print(f'  │  {circ.get("cycle_label","—")}'); print(f'  └{"─"*w}┘\n')
    print(f'  ┌─ 360° CIRCUIT ARC  (P(t) = A·sin(2π/T·t + φ₀) + trend) {"─"*8}┐')
    print(_arc(quadrant,cycle_dir)); print(f'  │')
    ang_str=f'{angle:.1f}°' if angle is not None else '?'
    print(f'  │  Current angle      : {ang_str}   [{quadrant}]')
    print(f'  │  Quadrant meaning   : {circ.get("quadrant_label","—")}'); print(f'  │')
    print(f'  │  Q1   0°– 90°  Emergence from trough  /  Capitulation near trough')
    print(f'  │  Q2  90°–180°  Expansion toward peak  /  Accumulation near support')
    print(f'  │  Q3 180°–270°  Distribution past peak /  Decline from peak')
    print(f'  │  Q4 270°–360°  Exhaustion near trough /  Collapse from peak')
    print(f'  └{"─"*w}┘\n')
    print(f'  ┌─ FFT SINUSOIDAL FIT  (P = A·sin(2π/T·t + φ₀) + trend) {"─"*10}┐')
    print(f'  │  Dominant period T  : {T} bars')
    t_h=round(T/60.0,1) if T else '?'; print(f'  │  Period in hours    : {t_h} h')
    print(f'  │  Amplitude A        : {pf(amp)}   ({round(float(amp)/float(cp)*100,3) if amp and cp else "?"}% of price)')
    print(f'  │  Sinusoid fit R²    : {r2}   ({"clean cycle ✔" if (r2 or 0)>0.5 else "noisy — interpret with caution"})')
    print(f'  │'); q_seg=round(T/4.0,1) if T else '?'
    print(f'  │  QUADRATURE BARS  (T/4 = {q_seg} bars each segment)  [{cycle_dir} CYCLE]')
    if qbars:
        cdir_q=qbars.get('cycle_dir',cycle_dir)
        _q_order=([('Q1_start','Q1_price','Q1_label','  0°  — TROUGH / entry'),('Q2_start','Q2_price','Q2_label',' 90°  — rising midline'),('Q3_start','Q3_price','Q3_label','180°  — ★ REVERSAL TOP'),('Q4_start','Q4_price','Q4_label','270°  — post-top decline'),('next_trough','next_trough_price','next_trough_label','360°  — next trough / new cycle')] if cdir_q=='UP'
                  else [('Q3_start','Q3_price','Q3_label','180°  — PEAK / entry'),('Q4_start','Q4_price','Q4_label','270°  — early collapse'),('Q1_start','Q1_price','Q1_label','360°  — ★ REVERSAL DIP'),('Q2_start','Q2_price','Q2_label',' 90°  — post-dip accumulation'),('next_peak','next_peak_price','next_peak_label','180°  — next peak / new cycle')])
        for bk,pk,lk,ang_hint in _q_order:
            bar_v=qbars.get(bk); pr_v=qbars.get(pk); pr_s=pf(pr_v) if pr_v else '—'
            print(f'  │    {ang_hint}  bar {bar_v:<6}  price {pr_s}')
    print(f'  └{"─"*w}┘\n')
    rev_arr='▲' if (rev_pct or 0)>0 else '▼'
    print(f'  ┌─ REVERSAL FORECAST  (next circuit extremum) {"─"*22}┐')
    print(f'  │  Reversal type      : {rev_type}')
    if rev_tgt is not None:
        rpct_s=f'{rev_arr}{rev_pct:+.2f}%' if rev_pct is not None else ''
        print(f'  │  ── CONSISTENT TARGET (blended) ──────────────────────────')
        print(f'  │  Reversal target    : {pf(rev_tgt)}  ({rpct_s})')
        if rev_tgt_swing: print(f'  │    ├ Swing extremum : {pf(rev_tgt_swing)}  ({round((rev_tgt_swing-cp)/(cp+1e-20)*100,2):+.2f}%)')
        if rev_tgt_fft:   print(f'  │    ├ FFT projection : {pf(rev_tgt_fft)}  ({round((rev_tgt_fft-cp)/(cp+1e-20)*100,2):+.2f}%)')
        if rev_tgt_emp:   print(f'  │    └ Empirical bar  : {pf(rev_tgt_emp)}  ({round((rev_tgt_emp-cp)/(cp+1e-20)*100,2):+.2f}%)')
    print(f'  │  At bar est.        : {rev_bar}  ({bars_rem} bars remaining)')
    print(f'  │  φ-extension target : {phi_lbl}')
    print(f'  │    [{pp(phi_ext)}]'); print(f'  └{"─"*w}┘\n')
    print(f'  ┌─ VOLUME CAUSE-EFFECT {"─"*46}┐')
    print(f'  │  {vol_lbl}')
    abs_bar='█'*min(20,int(circ.get("absorption_score",0)*2))+'░'*max(0,20-int(circ.get("absorption_score",0)*2))
    exhs_bar='█'*min(20,int(circ.get("exhaustion_score",0)*5))+'░'*max(0,20-int(circ.get("exhaustion_score",0)*5))
    print(f'  │  Absorption score   : {circ.get("absorption_score",0.0):.4f}  [{abs_bar}]  {"🟡 CONFIRMED" if abs_flag else ""}')
    print(f'  │  Exhaustion score   : {circ.get("exhaustion_score",0.0):.4f}  [{exhs_bar}]  {"🔴 CONFIRMED" if exhs_flag else ""}')
    print(f'  └{"─"*w}┘\n')
    print(f'  ┌─ CIRCUIT CONFIDENCE SCORE {"─"*42}┐')
    print(f'  │  Sinusoid R² contrib  : {conf_d.get("sinusoid_r2",0.0):.4f}')
    print(f'  │  Phase proximity      : {conf_d.get("phase_proximity",0.0):.4f}')
    print(f'  │  Volume signal        : {conf_d.get("vol_signal",0.0):.4f}')
    print(f'  │  Spiral timing        : {conf_d.get("spiral_timing",0.0):.4f}')
    print(f'  │  Reversal proximity   : {conf_d.get("proximity_to_rev",0.0):.4f}')
    print(f'  │  CIRCUIT CONFIDENCE   : {conf:.4f}  [{conf_bar}]')
    cgrade=('STRONG CIRCUIT SIGNAL ★★★' if conf>=0.75 else 'MODERATE SIGNAL ★★' if conf>=0.55 else 'WEAK SIGNAL ★' if conf>=0.35 else 'LOW CONFIDENCE')
    print(f'  │  Grade                : {cgrade}'); print(f'  └{"─"*w}┘\n')
    if mtf_tf_results:
        print(f'  ┌─ MTF CIRCUIT TABLE  (all timeframes) {"─"*30}┐')
        hdr=f'  │  {"TF":<5}  {"Dir":<5}  {"Q":<3}  {"Angle":>7}  {"R²":>6}  {"T-bars":>7}  {"Target":>12}  {"Δ%":>7}  {"φ-Ext":>12}  {"BarsLeft":>8}'
        print(hdr); print(f'  │  {"─"*90}')
        for r_ in mtf_tf_results:
            tf_s=r_.get('tf','—'); cdir_s=r_.get('cycle_dir') or '—'; q_s=r_.get('current_quadrant') or '—'
            ang_s=f'{r_.get("current_angle_deg",0.0):.1f}°' if r_.get('current_angle_deg') is not None else '—'
            r2_s=f'{r_.get("sinusoid_r2",0.0):.3f}' if r_.get('sinusoid_r2') is not None else '—'
            T_s=str(r_.get('dominant_period') or '—'); tgt_s=pf(r_.get('reversal_target'))
            pct_s=f'{r_.get("reversal_pct",0.0):+.2f}%' if r_.get('reversal_pct') is not None else '—'
            ext_s=pf(r_.get('phi_target_ext')); bleft=str(r_.get('bars_to_reversal') or '—')
            dir_mk='▲' if cdir_s=='UP' else ('▼' if cdir_s=='DOWN' else '→')
            print(f'  │  {tf_s:<5}  {dir_mk}{cdir_s:<4}  {q_s:<3}  {ang_s:>7}  {r2_s:>6}  {T_s:>7}  {tgt_s:>12}  {pct_s:>7}  {ext_s:>12}  {bleft:>8}')
        print(f'  └{"─"*w}┘\n')
    if mtf_summary and mtf_summary.get('mtf_target') is not None:
        ms=mtf_summary; mtgt=ms['mtf_target']; mtpct=ms.get('mtf_target_pct',0.0)
        mext=ms.get('mtf_phi_ext'); mconf=ms.get('mtf_confidence',0.0); mdir=ms.get('dominant_cycle_dir','—'); mq=ms.get('dominant_quadrant','—')
        nv=ms.get('n_valid_tfs',0); rlo=ms.get('target_range_low'); rhi=ms.get('target_range_high'); best_r2_tf=ms.get('best_r2_tf','—')
        mconf_bar='█'*int(mconf*20)+'░'*(20-int(mconf*20)); marr='▲' if (mtpct or 0)>0 else '▼'
        mgrade=('HIGH MTF AGREEMENT ★★★' if mconf>=0.65 else 'MODERATE MTF AGREEMENT ★★' if mconf>=0.45 else 'WEAK MTF AGREEMENT ★' if mconf>=0.25 else 'LOW MTF CONFIDENCE')
        print(f'  ┌─ MTF ML AGGREGATE TARGET  ({nv} valid TFs) {"─"*25}┐')
        print(f'  │  Dominant direction : {mdir}  (dominant quadrant: {mq})')
        print(f'  │  Best R² timeframe  : {best_r2_tf}')
        print(f'  │  ★ MTF TARGET        : {pf(mtgt)}  ({marr}{mtpct:+.2f}%)')
        print(f'  │    Target range     : {pf(rlo)}  →  {pf(rhi)}')
        if mext:
            mext_pct=(mext-cp)/(cp+1e-20)*100.0; print(f'  │    φ-Ext MTF target : {pf(mext)}  ({marr}{mext_pct:+.2f}%)')
        print(f'  │  MTF Confidence     : {mconf:.4f}  [{mconf_bar}]  {mgrade}'); print(f'  └{"─"*w}┘\n')
    summary_tgt=(mtf_summary.get('mtf_target') if mtf_summary else None) or rev_tgt
    summary_tgt_pct=(mtf_summary.get('mtf_target_pct') if mtf_summary else None) or rev_pct
    summary_ext=(mtf_summary.get('mtf_phi_ext') if mtf_summary else None) or phi_ext
    s_arr='▲' if (summary_tgt_pct or 0)>0 else '▼'
    print(f'  {"═"*w}'); print(f'  ★★★  360° CIRCUIT DECISION  ·  {lbl}'); print(f'  {"═"*w}')
    print(f'  ▶  ENTRY              : {pf(cp)}')
    print(f'  ▶  CYCLE DIRECTION    : {cycle_dir}  ({quadrant}  at {ang_str})')
    print(f'  ▶  REVERSAL TYPE      : {rev_type}')
    print(f'  ▶  REVERSAL TARGET    : {pf(rev_tgt)}'+(f'  ({rev_arr}{rev_pct:+.2f}%)' if rev_pct is not None else ''))
    print(f'  ▶  MTF ML TARGET      : {pf(summary_tgt)}'+(f'  ({s_arr}{summary_tgt_pct:+.2f}%)' if summary_tgt_pct is not None else ''))
    print(f'  ▶  φ-EXT TARGET (MTF) : {pf(summary_ext)}')
    print(f'  ▶  BARS TO REVERSAL   : ~{bars_rem}  (bar {rev_bar} est)')
    print(f'  ▶  VOLUME SIGNAL      : {vol_lbl}')
    print(f'  ▶  CONFIDENCE         : {conf:.4f}  [{cgrade}]')
    if mtf_summary: print(f'  ▶  MTF CONFIDENCE     : {mtf_summary.get("mtf_confidence",0.0):.4f}')
    print(f'\n  {"═"*w}\n')


# ═════════════════════════════════════════════════════════════
#  ASSET ANALYSIS ENTRY POINT
#  Replaces the scanner loop: asks user for a specific asset,
#  validates it, then runs the full analysis suite on it.
# ═════════════════════════════════════════════════════════════

def run_full_analysis(symbol, label_map):
    """Run the complete analysis pipeline on a single symbol."""
    lbl = label_map.get(symbol, symbol.replace('USDC', ''))
    print(f'\n  ════════════════════════════════════════════════════════')
    print(f'  ◈  FULL ANALYSIS  ·  {lbl}  ({symbol})')
    print(f'  ════════════════════════════════════════════════════════')
    print(f'  φ={PHI:.4f}  e={E:.4f}  π=3.14159  b={GOLDEN_B:.5f}')
    print(f'  Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

    # ── Step 1: Fetch full 1m dip detail (always run regardless of pass/fail) ──
    print(f'  [1/6] Fetching 1m order-flow + dip geometry...')
    passed, detail = check_dip_conditions(symbol)

    if not detail:
        print(f'  [ERROR] Could not fetch data for {symbol}. Check the symbol and try again.\n')
        return

    # ── Print raw 1m diagnostic ───────────────────────────────────────────────
    geo_d     = detail.get('geometry_detail', {})
    z_val     = geo_d.get('z_score');    z_s  = f'{z_val:+.2f}' if z_val is not None else '—'
    p_val     = geo_d.get('p_value');    p_s  = f'{p_val:.3f}' if p_val is not None else '—'
    curv_val  = geo_d.get('curvature'); curv_s = f'{curv_val:+.4f}' if curv_val is not None else '—'
    fake_flag = geo_d.get('is_fake_dip', False)

    print(f'\n  ┌─ 1m DIP DIAGNOSTIC ───────────────────────────────────────────────────┐')
    print(f'  │  Symbol          : {symbol}  ({lbl})')
    print(f'  │  Current price   : {detail["price"]}')
    print(f'  │  Bull / Bear vol : {detail["bull_pct"]}% / {detail["bear_pct"]}%  (cond_vol={detail["cond_vol"]})')
    print(f'  │  ArgMin / ArgMax : {detail["argmin_idx"]} / {detail["argmax_idx"]}  (cond_ext={detail["cond_ext"]})')
    print(f'  │  CMO-14          : {detail["raw_cmo"]}')
    print(f'  │  Delta ratio     : {detail["delta_ratio"]}')
    print(f'  │  Absorption scr  : {detail["absorption_score"]}')
    print(f'  │  Exhaustion scr  : {detail["exhaustion_score"]}')
    print(f'  │  Geometry score  : {detail["geometry_score"]:.1f} / 100')
    print(f'  │  Z-score (depth) : {z_s}   p-value: {p_s}')
    print(f'  │  Curvature       : {curv_s}  (positive = concave-up = genuine bottom)')
    if fake_flag:
        print(f'  │  ⚠  FAKE DIP detected — curvature still negative (price accelerating down)')
    print(f'  │')
    dip_result = 'DIP CONFIRMED ✔' if passed else 'DIP NOT CONFIRMED ✗'
    if not passed:
        reasons = []
        if not detail['cond_vol']:
            reasons.append(f'bear vol dominant ({detail["bear_pct"]}% > {detail["bull_pct"]}%)')
        if not detail['cond_ext']:
            reasons.append(f'argmax more recent than argmin ({detail["argmax_idx"]} > {detail["argmin_idx"]})')
        print(f'  │  Result          : {dip_result}  ({" | ".join(reasons)})')
    else:
        print(f'  │  Result          : {dip_result}')
    print(f'  └───────────────────────────────────────────────────────────────────────┘\n')

    # Store detail in dict format expected by downstream functions
    sel_detail = {symbol: detail}
    current_price = detail['price']

    # ── Step 2: FFT + Time Geometry ───────────────────────────────────────────
    print(f'  [2/6] Running FFT + Time Geometry across all timeframes...')
    stf_results, stf_best, htf_results, htf_best = full_fft_report(symbol, current_price)

    if stf_results or htf_results:
        print_fft_report(symbol, label_map, stf_results, stf_best, htf_results, htf_best)
        run_time_geometry(symbol, label_map, current_price, sel_detail, stf_results, htf_results)
    else:
        print(f'  FFT: insufficient data.\n')

    # ── Step 3: ML Compound Forecast ─────────────────────────────────────────
    print(f'  [3/6] Running ML compound forecast (9 models, backtest, MC)...')
    ml_result = ml_compound_forecast(symbol, current_price, sel_detail, stf_results, htf_results)
    print_ml_report(ml_result, label_map)

    # ── Step 4: φ-Reversal Forecast ──────────────────────────────────────────
    print(f'  [4/6] Running φ-Reversal forecast...')
    phi_rev = phi_reversal_forecast(symbol, current_price, sel_detail, order=7, phi_band_tol_pct=1.2, min_confidence=0.25)
    print_phi_reversal_block(phi_rev, label_map)

    # ── Step 5: 360° Sinusoidal Circuit (1m primary) ─────────────────────────
    print(f'  [5/6] Running 360° Sinusoidal Circuit Engine...')
    circ = sinusoidal_circuit_engine(symbol, current_price, sel_detail, stf_results=stf_results, htf_results=htf_results)

    # ── Step 6: MTF Sinusoidal Circuit ───────────────────────────────────────
    print(f'  [6/6] Running MTF Sinusoidal Circuit across all timeframes...')
    mtf_tf_results, mtf_summary = sinusoidal_circuit_mtf(symbol, current_price, sel_detail, stf_results=stf_results, htf_results=htf_results)
    print_circuit_block(circ, label_map, mtf_tf_results=mtf_tf_results, mtf_summary=mtf_summary)

    print(f'  ════════════════════════════════════════════════════════')
    print(f'  ✓  Analysis complete for {lbl} ({symbol})')
    print(f'  ════════════════════════════════════════════════════════\n')


# ═════════════════════════════════════════════════════════════
#  MAIN — Interactive asset prompt
# ═════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f'\n  ╔══════════════════════════════════════════════════════════════╗')
    print(f'  ║  MTF Asset Analyzer  ·  φ · e · π  Time Geometry Suite      ║')
    print(f'  ║  Runs complete analysis on ANY USDC spot pair you specify    ║')
    print(f'  ╚══════════════════════════════════════════════════════════════╝')
    print(f'  φ={PHI:.4f}  e={E:.4f}  b={GOLDEN_B:.5f}  φ∠={PHI_ANGLE:.2f}°')
    print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    print(f'  How to use:')
    print(f'    Enter the BASE asset symbol (e.g. BTC, ETH, SOL)')
    print(f'    or the full pair (e.g. BTCUSDC, ETHUSDC)')
    print(f'    Type "list" to search available pairs')
    print(f'    Type "quit" to exit\n')

    while True:
        try:
            raw = input('  Enter asset symbol → ').strip().upper()
        except (EOFError, KeyboardInterrupt):
            print('\n  Exiting.'); sys.exit(0)

        if not raw:
            continue

        if raw in ('QUIT', 'EXIT', 'Q'):
            print('  Goodbye.'); sys.exit(0)

        # Handle "list" command
        if raw.startswith('LIST'):
            search_term = raw[4:].strip()
            print(f'  Searching for USDC pairs{"matching "+search_term if search_term else ""}...')
            available = trader.list_usdc_pairs(search=search_term)
            if available:
                cols = 6; rows = [available[i:i+cols] for i in range(0, len(available), cols)]
                print(f'  Found {len(available)} pairs:\n')
                for row in rows:
                    print('    ' + '  '.join(f'{s:<14}' for s in row))
                print()
            else:
                print('  No pairs found.\n')
            continue

        # Normalize: if user typed "BTC" make it "BTCUSDC"
        symbol = raw if raw.endswith('USDC') else raw + 'USDC'

        print(f'\n  Validating {symbol}...')
        is_valid, base_asset, live_price = trader.validate_usdc_pair(symbol)

        if not is_valid:
            print(f'  Try "list {raw[:3]}" to search for matching pairs.\n')
            continue

        print(f'  ✓ {symbol}  ({base_asset}/USDC)  live price: {live_price}')

        label_map = {symbol: base_asset}

        # Ask if user wants to run another after analysis
        run_full_analysis(symbol, label_map)

        # Prompt to continue or exit
        try:
            again = input('  Analyze another asset? [Y/n] → ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print('\n  Exiting.'); sys.exit(0)

        if again in ('n', 'no', 'q', 'quit'):
            print('  Goodbye.'); sys.exit(0)
        print()