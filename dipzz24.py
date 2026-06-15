import gc
from binance.client import Client
import numpy as np
import talib as ta
import time
import sys
import pywt
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from scipy import stats


# ==========================================
# PHI / GOLDEN HARMONIC CONSTANTS
# ==========================================

PHI     = (1.0 + 5.0 ** 0.5) / 2.0
PHI_INV = 1.0 / PHI
PHI_SQ  = PHI * PHI

FIB_RATIOS = {
    "F236": 0.236, "F382": 0.382, "F500": 0.500,
    "F618": PHI_INV, "F786": PHI_INV ** 0.5,
}


# ==========================================
# φ-PHASE REVERSAL ENGINE
# ==========================================
#
# Each bar index i is mapped to a golden-ratio phase:
#   phase(i) = (i / φ) mod 1.0   → irrational, never repeats
#   theta(i) = 2π * phase(i)     → angle on unit circle
#
# Swing highs / lows are detected in a local window and their
# phase values are collected.  Three outputs feed downstream:
#
#   1. phi_phase_score    – how tightly reversals cluster around
#                           known φ-harmonic zones (0-100% scale)
#   2. current_phase_zone – "REVERSAL_HIGH", "REVERSAL_LOW",
#                           "NEUTRAL", or "AMBIGUOUS"
#   3. forward_return_bias– sign and magnitude of the expected
#                           20-bar forward return at the current
#                           phase, derived from the observed
#                           phase→return histogram.
#
# VALID PHASE ZONES (empirically derived from the harmonic grid):
#   φ-harmonics land near:  0.0, 1/φ≈0.618, 1/φ²≈0.382,
#                            1/φ³≈0.236, 1-1/φ²≈0.764
#   These are the same Fibonacci ratios already in FIB_RATIOS.

PHI_PHASE_ZONES = np.array([0.0, 0.236, 0.382, 0.500, 0.618, 0.764, 1.0])
PHI_PHASE_TOL   = 0.04   # ±4% of the [0,1] phase axis = ±14.4° on the circle


def compute_phi_phase(bar_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map bar indices to φ-phase [0,1] and circle angle [0, 2π]."""
    phase = (bar_indices / PHI) % 1.0
    theta = 2.0 * np.pi * phase
    return phase, theta


def detect_swing_highs_lows(prices: np.ndarray,
                              window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Local-extrema swing detection with a symmetric window.
    Returns arrays of bar indices for highs and lows.
    """
    n = len(prices)
    highs, lows = [], []
    for i in range(window, n - window):
        segment = prices[i - window: i + window + 1]
        if prices[i] == segment.max():
            highs.append(i)
        if prices[i] == segment.min():
            lows.append(i)
    return np.array(highs, dtype=int), np.array(lows, dtype=int)


def phi_phase_forward_return_bias(phase: np.ndarray,
                                   prices: np.ndarray,
                                   forward: int = 20,
                                   bins: int = 20) -> np.ndarray:
    """
    Compute the average forward return for each of `bins` phase buckets.
    Returns an array of shape (bins,) — positive = bullish zone,
    negative = bearish zone.

    Only uses bars where a forward return can be computed (first n-forward bars).
    """
    n = len(prices) - forward
    if n < bins * 2:
        return np.zeros(bins)
    fwd_ret  = (prices[forward:forward + n] - prices[:n]) / (prices[:n] + 1e-12)
    ph_slice = phase[:n]
    avg_ret  = np.zeros(bins)
    edges    = np.linspace(0.0, 1.0, bins + 1)
    for b in range(bins):
        mask = (ph_slice >= edges[b]) & (ph_slice < edges[b + 1])
        if mask.sum() > 0:
            avg_ret[b] = float(np.mean(fwd_ret[mask]))
    return avg_ret


def phi_clustering_score(indices: np.ndarray,
                          phase: np.ndarray,
                          bins: int = 20) -> float:
    """
    χ²-based clustering score vs uniform random baseline.
    Returns a multiplier: 1.0 = random, >2.0 = meaningful clustering.
    """
    if len(indices) == 0:
        return 1.0
    counts   = np.zeros(bins)
    edges    = np.linspace(0.0, 1.0, bins + 1)
    for i in indices:
        b = min(int(phase[i] * bins), bins - 1)
        counts[b] += 1
    expected = len(indices) / bins
    chi2     = float(np.sum((counts - expected) ** 2 / (expected + 1e-9)))
    return float(np.clip(chi2 / (bins * 2.0), 0.0, 9.9))


def current_phi_phase_zone(current_phase: float,
                            avg_ret_by_bin: np.ndarray,
                            bins: int = 20) -> str:
    """
    Classify the current bar's φ-phase into an actionable zone.
    Uses both harmonic proximity AND empirical return bias.
    """
    # Which bin is the current phase in?
    b        = min(int(current_phase * bins), bins - 1)
    ret_bias = float(avg_ret_by_bin[b]) if len(avg_ret_by_bin) > b else 0.0

    # Proximity to known φ-harmonic levels
    dist_to_zone = float(np.min(np.abs(PHI_PHASE_ZONES - current_phase)))
    near_harmonic = dist_to_zone < PHI_PHASE_TOL

    # Classify
    if near_harmonic and ret_bias > 0.003:
        return "REVERSAL_LOW"     # harmonic + positive forward return → buy zone
    elif near_harmonic and ret_bias < -0.003:
        return "REVERSAL_HIGH"    # harmonic + negative forward return → sell zone
    elif near_harmonic:
        return "HARMONIC"         # harmonic but return neutral
    elif ret_bias > 0.005:
        return "BIAS_BULL"        # empirical bull zone (not a named harmonic)
    elif ret_bias < -0.005:
        return "BIAS_BEAR"
    else:
        return "NEUTRAL"


def phi_phase_analysis(prices: np.ndarray,
                        swing_window: int = 5,
                        fwd_bars: int = 20,
                        bins: int = 20) -> Dict:
    """
    Master φ-phase analysis function.
    Integrates directly with compute_phase_alignment() and
    get_sinusoidal_dip_timing() — call after those.

    Returns a dict ready to be merged into `combined` in check_1m_final()
    and score_15m_candidate().
    """
    n = len(prices)
    if n < 64:
        return {
            "phi_phase":          0.5,
            "phi_phase_zone":     "NEUTRAL",
            "phi_cluster_score":  1.0,
            "phi_fwd_bias":       0.0,
            "phi_near_harmonic":  False,
            "phi_reversal_ready": False,
            "phi_bar_index":      0,
        }

    arr      = np.array(prices, dtype=float)
    indices  = np.arange(n, dtype=float)
    phase, _ = compute_phi_phase(indices)

    highs, lows   = detect_swing_highs_lows(arr, window=swing_window)
    avg_ret       = phi_phase_forward_return_bias(phase, arr, forward=fwd_bars, bins=bins)

    current_phase = float(phase[-1])
    current_bin   = min(int(current_phase * bins), bins - 1)
    fwd_bias      = float(avg_ret[current_bin])

    zone          = current_phi_phase_zone(current_phase, avg_ret, bins)
    cluster       = phi_clustering_score(
                        np.concatenate([highs, lows]) if (len(highs) + len(lows)) > 0
                        else np.array([], dtype=int),
                        phase, bins)
    near_harmonic = float(np.min(np.abs(PHI_PHASE_ZONES - current_phase))) < PHI_PHASE_TOL

    # reversal_ready: we are in a φ-harmonic zone AND the empirical bias is bullish
    reversal_ready = (zone in ("REVERSAL_LOW", "BIAS_BULL") and near_harmonic)

    return {
        "phi_phase":          current_phase,
        "phi_phase_zone":     zone,
        "phi_cluster_score":  cluster,
        "phi_fwd_bias":       fwd_bias,
        "phi_near_harmonic":  near_harmonic,
        "phi_reversal_ready": reversal_ready,
        "phi_bar_index":      n - 1,
        "phi_swing_highs":    len(highs),
        "phi_swing_lows":     len(lows),
        "phi_avg_ret_bins":   avg_ret.tolist(),
    }


# ==========================================
# STATIONARITY + CYCLIC LINE ENGINE
# ==========================================
#
# Before trusting FFT/wavelet phase estimates, test whether the
# price series is actually stationary (mean-reverting).  A random
# walk (I(1)) will produce arbitrary FFT phases that look like
# structure but contain no predictive information.
#
# Two tests are used:
#   ADF (Augmented Dickey-Fuller) approximation — fast OLS version
#       H₀: series has unit root (non-stationary)
#       Reject at p < 0.10 → stationary
#
#   Hurst exponent via R/S rescaled range:
#       H < 0.45 → mean-reverting  (best for cyclic models)
#       H ≈ 0.50 → random walk     (cyclic fits unreliable)
#       H > 0.55 → trending        (detrend first)
#
# Once stationarity is confirmed (or the series is detrended),
# fit a sinusoidal CYCLIC LINE:
#   price ≈ C₀ + C_drift·t + A·sin(ω·t + φ)
#
# This gives:
#   R²              — fit quality (how much variance the cycle explains)
#   dist_from_midline — where price is relative to the fitted cycle
#   at_cyclic_low   — price is below the midline by >0.7 amplitudes
#   price_forecast_low / high — fitted cycle extremes in price units

def adf_stationarity(arr: np.ndarray) -> Dict:
    """
    Augmented Dickey-Fuller approximation + Hurst exponent.
    Pure NumPy/SciPy — no statsmodels dependency.
    """
    n = len(arr)
    if n < 32:
        return {"is_stationary": False, "hurst": 0.5,
                "adf_stat": 0.0, "adf_p": 1.0,
                "mean_reverting": False, "trending": False}

    # --- ADF via simple OLS on first differences vs lagged level ---
    y = np.diff(arr).astype(float)
    x = arr[:-1].astype(float)
    slope, _, _, p_value, _ = stats.linregress(x, y)
    is_stationary = bool(slope < 0 and p_value < 0.10)

    # --- Hurst exponent via R/S rescaled range ---
    lags = [max(4, n // 8), max(8, n // 4), max(16, n // 2)]
    rs_vals = []
    for lag in lags:
        sub     = arr[:lag].astype(float)
        devs    = np.cumsum(sub - np.mean(sub))
        R       = float(np.max(devs) - np.min(devs))
        S       = float(np.std(sub) + 1e-9)
        rs_vals.append(R / S)

    if len(rs_vals) >= 2:
        log_n  = np.log(np.array(lags, dtype=float))
        log_rs = np.log(np.maximum(rs_vals, 1e-9))
        hurst  = float(np.clip(np.polyfit(log_n, log_rs, 1)[0], 0.0, 1.0))
    else:
        hurst = 0.5

    return {
        "is_stationary": is_stationary,
        "hurst":          hurst,
        "adf_stat":       float(slope),
        "adf_p":          float(p_value),
        "mean_reverting": hurst < 0.45,
        "trending":       hurst > 0.55,
    }


def cyclic_line_fit(arr: np.ndarray, cycle_bars: int) -> Dict:
    """
    Fit: price ≈ C₀ + C_drift·t + A·sin(ωt) + B·cos(ωt)
    using ordinary least squares.

    Only meaningful when called after confirming stationarity
    (or after first-differencing trending series).
    """
    n = len(arr)
    empty = {"r_squared": 0.0, "phase_deg": 0.0, "amplitude": 0.0,
             "dist_from_midline": 0.0, "at_cyclic_low": False,
             "fit_quality": "POOR", "cyclic_midline": float(arr[-1]),
             "fitted_low_price": float(arr[-1]),
             "fitted_high_price": float(arr[-1])}
    if n < cycle_bars * 2 or cycle_bars < 4:
        return empty

    t     = np.arange(n, dtype=float)
    omega = 2.0 * np.pi / cycle_bars
    A_mat = np.column_stack([np.ones(n), np.sin(omega * t),
                             np.cos(omega * t), t])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, arr, rcond=None)
    except np.linalg.LinAlgError:
        return empty

    c0, c_sin, c_cos, c_drift = coeffs
    fitted    = A_mat @ coeffs
    ss_res    = float(np.sum((arr - fitted) ** 2))
    ss_tot    = float(np.sum((arr - np.mean(arr)) ** 2) + 1e-9)
    r_sq      = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))
    amplitude = float(np.sqrt(c_sin ** 2 + c_cos ** 2))
    phase_rad = float(np.arctan2(c_cos, c_sin))
    cur_phase = (omega * (n - 1) + phase_rad) % (2.0 * np.pi)
    midline   = float(c0 + c_drift * (n - 1))
    dist      = float((arr[-1] - midline) / (amplitude + 1e-9))

    fit_q = ("EXCELLENT" if r_sq > 0.65 else
             "GOOD"      if r_sq > 0.40 else
             "FAIR"      if r_sq > 0.20 else "POOR")

    return {
        "r_squared":          r_sq,
        "phase_deg":          float(np.degrees(cur_phase)),
        "amplitude":          amplitude,
        "dist_from_midline":  dist,
        "at_cyclic_low":      dist < -0.70,
        "fit_quality":        fit_q,
        "cyclic_midline":     midline,
        "fitted_low_price":   midline - amplitude,
        "fitted_high_price":  midline + amplitude,
    }


def stationarity_gated_cycle(prices: np.ndarray,
                              dominant_cycle_bars: int) -> Dict:
    """
    Combined entry point:
      1. Test stationarity (ADF + Hurst)
      2. Detrend if I(1)-trending (first-difference)
      3. Fit cyclic regression line
      4. Compute reversal_score gated by both stationarity
         quality and cycle fit quality

    Output dict is merged into `combined` in check_1m_final()
    and score_15m_candidate().
    """
    arr  = np.array(prices, dtype=float)
    stat = adf_stationarity(arr)

    if not stat["is_stationary"] and stat["trending"]:
        arr_work  = np.diff(arr)
        detrended = True
    else:
        arr_work  = arr - float(np.mean(arr))
        detrended = False

    cyc_bars = dominant_cycle_bars if dominant_cycle_bars >= 8 else 32
    cycle    = cyclic_line_fit(arr_work, cyc_bars)

    stat_conf  = (1.0  if stat["is_stationary"] else
                  0.60 if stat["mean_reverting"] else 0.30)
    cycle_conf = {"EXCELLENT": 1.0, "GOOD": 0.75,
                  "FAIR": 0.40,     "POOR": 0.10}.get(cycle["fit_quality"], 0.0)
    rev_score  = float(np.clip(
        stat_conf * cycle_conf * (1.0 + float(cycle["at_cyclic_low"])),
        0.0, 1.0))

    # Translate fitted prices back to original scale when detrended
    base = float(arr[-1])
    if detrended:
        fcast_low  = base + cycle["fitted_low_price"]
        fcast_high = base + cycle["fitted_high_price"]
    else:
        fcast_low  = cycle["fitted_low_price"]  + float(np.mean(arr))
        fcast_high = cycle["fitted_high_price"] + float(np.mean(arr))

    return {
        # stationarity
        "stat_is_stationary":  stat["is_stationary"],
        "stat_hurst":          stat["hurst"],
        "stat_mean_reverting": stat["mean_reverting"],
        "stat_trending":       stat["trending"],
        "stat_adf_p":          stat["adf_p"],
        # cyclic line
        "cyc_r_squared":       cycle["r_squared"],
        "cyc_fit_quality":     cycle["fit_quality"],
        "cyc_at_cyclic_low":   cycle["at_cyclic_low"],
        "cyc_dist_midline":    cycle["dist_from_midline"],
        "cyc_amplitude":       cycle["amplitude"],
        "cyc_phase_deg":       cycle["phase_deg"],
        # combined
        "detrended":           detrended,
        "stat_confidence":     stat_conf,
        "reversal_score":      rev_score,
        "price_forecast_low":  fcast_low,
        "price_forecast_high": fcast_high,
    }


# ==========================================
# QUADRATIC PRICE FORECAST ENGINE
# ==========================================
#
# THEORY — area model → price levels:
#
#   Given extrema L (lowest low) and H (highest high) from a tight
#   adaptive window (dom_cycle * 3 bars, min 60):
#
#     n   = (H + L) / 2    midpoint  — centre of the n×n square
#     R   = H - L          range     — side difference in (n-1)(n+1)
#     √R                   Gann geometric step (one "square root" of range)
#     QR  = √(n²-1)/n · R  quadratic residue projection (deep)
#
#   UPSIDE  : H + {0.236R, √R, 0.618R, R, QR}
#   DOWNSIDE: L - {0.236R, √R, 0.618R, R, QR}
#
#   Confidence decays linearly with distance from current price.
#   Each target is weighted by: spike_prob, reversal_score,
#   phi_fwd_bias, freq alignment_score, pos_in_range.
#
#   ADAPTIVE LOOKBACK ("will be hit" logic):
#     window = max(dom_cycle * 3, 60) bars on 1m
#   This is the tightest window containing at least one full dominant
#   cycle, so both extrema are recent enough to be revisited shortly.

def quadratic_price_forecast(
        raw_klines: list,
        current_price: float,
        golden: Dict,
        dom_cycle: int = 0) -> Dict:
    """
    Compute quadratic area-model price targets from a tight adaptive window.

    Parameters
    ----------
    raw_klines    : list of raw 1m klines (at least 60 bars)
    current_price : latest close price
    golden        : combined dict from get_sinusoidal_dip_timing / check_1m_final
    dom_cycle     : dominant cycle bars (0 = auto-detect from FFT)

    Returns
    -------
    Dict with:
        window_bars, L, H, n_mid, R, sqrt_R, QR,
        targets_up   : list of dicts [{label, price, dist_pct, conf, method}]
        targets_down : list of dicts [{label, price, dist_pct, conf, method}]
        best_top     : single highest-confidence upside target
        best_dip     : single highest-confidence downside target
        forecast_bias: 'TOP' | 'DIP' | 'NEUTRAL'
        summary_line : human-readable one-liner
    """
    if not raw_klines or len(raw_klines) < 30:
        return {"window_bars": 0, "L": 0.0, "H": 0.0, "targets_up": [],
                "targets_down": [], "best_top": None, "best_dip": None,
                "forecast_bias": "NEUTRAL", "summary_line": "Insufficient data"}

    # ── 1. Adaptive window selection ──────────────────────────────────
    # Use dom_cycle from freq engine if available; otherwise auto-detect
    if dom_cycle <= 0:
        freq = golden.get("freq") or {}
        dom_cycle = freq.get("dominant_cycle", 0) or 0
    if dom_cycle <= 0:
        # Quick FFT fallback on the closes
        closes_tmp = np.array([float(k[4]) for k in raw_klines], dtype=float)
        if len(closes_tmp) >= 32:
            fft_v  = np.abs(np.fft.rfft(closes_tmp - np.mean(closes_tmp)))
            fft_f  = np.fft.rfftfreq(len(closes_tmp))
            fft_f[fft_f == 0] = 1e-12
            peak   = int(np.argmax(fft_v[1:]) + 1)
            dom_cycle = int(round(1.0 / (fft_f[peak] + 1e-12)))
        dom_cycle = max(dom_cycle, 10)

    window = int(np.clip(dom_cycle * 3, 60, len(raw_klines)))
    kw = raw_klines[-window:]

    highs  = np.array([float(k[2]) for k in kw])
    lows   = np.array([float(k[3]) for k in kw])
    closes = np.array([float(k[4]) for k in kw])

    H = float(highs.max())
    L = float(lows.min())
    H_idx = int(np.argmax(highs))
    L_idx = int(np.argmin(lows))

    R = H - L
    if R <= 0 or L <= 0:
        return {"window_bars": window, "L": L, "H": H, "targets_up": [],
                "targets_down": [], "best_top": None, "best_dip": None,
                "forecast_bias": "NEUTRAL", "summary_line": "Zero-range window"}

    n_mid   = (H + L) / 2.0
    sqrt_R  = float(np.sqrt(R))

    # Quadratic residue: √(n²-1)/n · R  — the "missing corner" projection
    n2_minus1 = max(n_mid ** 2 - 1.0, 0.0)
    QR = float(np.sqrt(n2_minus1) / (n_mid + 1e-12) * R)

    # ── 2. Confidence weights from existing engines ────────────────────
    spike_prob     = float(golden.get("spike_prob",     0.0))
    reversal_score = float(golden.get("reversal_score", 0.0))
    phi_fwd_bias   = float(golden.get("phi_fwd_bias",   0.0))
    phi_ready      = bool(golden.get("phi_reversal_ready", False))
    pos_in_range   = float(golden.get("pos_in_range",   0.5))
    freq_align     = float((golden.get("freq") or {}).get("alignment_score", 0.0))
    wave_bottom    = bool(golden.get("wave_near_bottom", False))
    turning_up     = bool(golden.get("turning_up", False))
    cyc_low        = bool(golden.get("cyc_at_cyclic_low", False))

    # Base confidence: how strong is the reversal setup right now?
    base_conf = float(np.clip(
        0.25 * spike_prob +
        0.20 * reversal_score +
        0.15 * freq_align +
        0.10 * float(np.clip(phi_fwd_bias * 10.0, 0.0, 1.0)) +
        0.10 * float(phi_ready) +
        0.10 * float(wave_bottom) +
        0.05 * float(turning_up) +
        0.05 * float(cyc_low),
        0.05, 1.0
    ))

    # Directional bias from position in range
    # pos_in_range: 0=at L, 1=at H
    # If near L → bias toward TOP target (bounce expected)
    # If near H → bias toward DIP target (rejection expected)
    upside_bias   = float(np.clip(1.0 - pos_in_range, 0.0, 1.0))
    downside_bias = float(np.clip(pos_in_range,        0.0, 1.0))

    # ── 3. Build target levels ──────────────────────────────────────────
    # Distance-decay: confidence drops by 30% for each step outward
    decay = [1.0, 0.82, 0.65, 0.50, 0.38]

    up_specs = [
        ("QU1_Fib236",  H + 0.236 * R,  "Fib 0.236 ext"),
        ("QU2_SqrtR",   H + sqrt_R,      "Gann √R step"),
        ("QU3_Fib618",  H + 0.618 * R,  "Fib 0.618 ext"),
        ("QU4_Measured",H + R,           "Measured move"),
        ("QU5_Quad",    H + QR,          "Quadratic residue"),
    ]
    dn_specs = [
        ("QD1_Fib236",  L - 0.236 * R,  "Fib 0.236 ext"),
        ("QD2_SqrtR",   L - sqrt_R,      "Gann √R step"),
        ("QD3_Fib618",  L - 0.618 * R,  "Fib 0.618 ext"),
        ("QD4_Measured",L - R,           "Measured move"),
        ("QD5_Quad",    L - QR,          "Quadratic residue"),
    ]

    def make_targets(specs, bias):
        out = []
        for i, (label, price, method) in enumerate(specs):
            if price <= 0:
                continue
            dist_pct = (price - current_price) / current_price * 100.0
            conf     = float(np.clip(base_conf * decay[i] * bias, 0.01, 1.0))
            out.append({
                "label":    label,
                "price":    price,
                "dist_pct": dist_pct,
                "conf":     conf,
                "method":   method,
            })
        return out

    targets_up   = make_targets(up_specs,  upside_bias)
    targets_down = make_targets(dn_specs, downside_bias)

    # ── 4. Best targets (highest confidence among positive-dist) ────────
    valid_up = [t for t in targets_up   if t["dist_pct"] > 0.05]
    valid_dn = [t for t in targets_down if t["dist_pct"] < -0.05]

    best_top = max(valid_up, key=lambda t: t["conf"]) if valid_up else None
    best_dip = max(valid_dn, key=lambda t: t["conf"]) if valid_dn else None

    # ── 5. Forecast bias ────────────────────────────────────────────────
    if pos_in_range < 0.25 and (wave_bottom or cyc_low):
        bias_label = "TOP"
    elif pos_in_range > 0.75:
        bias_label = "DIP"
    elif upside_bias > downside_bias + 0.15:
        bias_label = "TOP"
    elif downside_bias > upside_bias + 0.15:
        bias_label = "DIP"
    else:
        bias_label = "NEUTRAL"

    # ── 6. Summary line ─────────────────────────────────────────────────
    if best_top and bias_label == "TOP":
        summary = (f"🎯 TOP target: {best_top['price']:.8f}  "
                   f"({best_top['dist_pct']:+.3f}%)  "
                   f"via {best_top['method']}  "
                   f"conf={best_top['conf']*100:.0f}%")
    elif best_dip and bias_label == "DIP":
        summary = (f"🎯 DIP target: {best_dip['price']:.8f}  "
                   f"({best_dip['dist_pct']:+.3f}%)  "
                   f"via {best_dip['method']}  "
                   f"conf={best_dip['conf']*100:.0f}%")
    elif best_top:
        summary = (f"⬆  Next top: {best_top['price']:.8f}  "
                   f"({best_top['dist_pct']:+.3f}%)")
    elif best_dip:
        summary = (f"⬇  Next dip: {best_dip['price']:.8f}  "
                   f"({best_dip['dist_pct']:+.3f}%)")
    else:
        summary = "No reachable quadratic targets computed."

    return {
        "window_bars":   window,
        "dom_cycle":     dom_cycle,
        "L":             L,
        "H":             H,
        "H_age":         window - H_idx,
        "L_age":         window - L_idx,
        "n_mid":         n_mid,
        "R":             R,
        "sqrt_R":        sqrt_R,
        "QR":            QR,
        "base_conf":     base_conf,
        "pos_in_range":  pos_in_range,
        "targets_up":    targets_up,
        "targets_down":  targets_down,
        "best_top":      best_top,
        "best_dip":      best_dip,
        "forecast_bias": bias_label,
        "summary_line":  summary,
    }


def format_quadratic_block(qf: Dict, W: int = 74):
    """
    Print the quadratic forecast block inside format_sr_output.
    Designed to appear after the cyclic line block and before the
    consolidated trade bias section.
    """
    if not qf or qf.get("window_bars", 0) == 0:
        return
    print("─" * W)
    print("  📐  QUADRATIC AREA-MODEL PRICE FORECAST")
    print("  (extrema → n=(H+L)/2 → (n-1)(n+1)=n²-1 → √R + φ-ext targets)")
    print("─" * W)

    L, H, n   = qf["L"], qf["H"], qf["n_mid"]
    R, sqR    = qf["R"], qf["sqrt_R"]
    QR        = qf["QR"]
    wb        = qf["window_bars"]
    dc        = qf["dom_cycle"]
    bc        = qf["base_conf"]
    bias      = qf["forecast_bias"]
    H_age     = qf.get("H_age", 0)
    L_age     = qf.get("L_age", 0)

    bias_icon = {"TOP": "⬆  TOP", "DIP": "⬇  DIP", "NEUTRAL": "⚪ NEUTRAL"}.get(bias, "⚪")

    print(f"  Window       : {wb} bars  (dom cycle ≈ {dc} bars × 3)")
    print(f"  Lowest Low   : {L:.8f}  ({L_age} bars ago)")
    print(f"  Highest High : {H:.8f}  ({H_age} bars ago)")
    print(f"  Midpoint  n  : {n:.8f}  (= (H+L)/2)")
    print(f"  Range     R  : {R:.8f}")
    print(f"  √R (Gann)    : {sqR:.8f}  (geometric step)")
    print(f"  QR (n²-res)  : {QR:.8f}  (quadratic residue)")
    print(f"  Base Conf    : {bc*100:.1f}%  (weighted from φ/cyclic/freq engines)")
    print(f"  Bias         : {bias_icon}")
    print()

    # Upside targets
    ups = qf.get("targets_up", [])
    if ups:
        print(f"  {'Label':<16} {'Price':>14} {'Dist%':>8} {'Conf':>6}  Method")
        print("  " + "─" * 60)
        for t in ups:
            marker = "►" if t["dist_pct"] > 0 else "·"
            hi_c   = "★" if t.get("conf", 0) >= 0.55 else " "
            print(f"  {marker}{t['label']:<15} {t['price']:>14.8f} "
                  f"{t['dist_pct']:>+7.3f}% {t['conf']*100:>5.0f}%{hi_c} "
                  f"{t['method']}")
    print()

    # Downside targets
    dns = qf.get("targets_down", [])
    if dns:
        print(f"  {'Label':<16} {'Price':>14} {'Dist%':>8} {'Conf':>6}  Method")
        print("  " + "─" * 60)
        for t in dns:
            marker = "◄" if t["dist_pct"] < 0 else "·"
            hi_c   = "★" if t.get("conf", 0) >= 0.55 else " "
            print(f"  {marker}{t['label']:<15} {t['price']:>14.8f} "
                  f"{t['dist_pct']:>+7.3f}% {t['conf']*100:>5.0f}%{hi_c} "
                  f"{t['method']}")
    print()

    # Best targets callout
    best_top = qf.get("best_top")
    best_dip = qf.get("best_dip")
    if best_top:
        print(f"  ⬆  BEST TOP  : {best_top['label']:<16} "
              f"{best_top['price']:.8f}  ({best_top['dist_pct']:+.3f}%)  "
              f"conf={best_top['conf']*100:.0f}%")
    if best_dip:
        print(f"  ⬇  BEST DIP  : {best_dip['label']:<16} "
              f"{best_dip['price']:.8f}  ({best_dip['dist_pct']:+.3f}%)  "
              f"conf={best_dip['conf']*100:.0f}%")
    print(f"\n  🎯  {qf['summary_line']}")
    print()


# ==========================================
# FREQUENCY / CYCLE THEORY (BTC CONTEXT)
# ==========================================
#
# Price is a superposition of nested cycles:
#   Low  freq  (4H–1D)   → macro trend / "bass line"
#   Mid  freq  (15m–1H)  → swing structure
#   High freq  (1m–5m)   → micro-momentum / noise
#
# FFT  : decomposes the ENTIRE window into global sinusoids
#        → best for dominant cycle length detection
#
# Wavelet (DWT) : time-localised decomposition
#        → separates trend / swing / micro at each bar
#        → detects WHERE in time a cycle is turning
#
# Momentum by frequency layer:
#   Trend momentum  = slope of low-freq reconstruction
#   Swing momentum  = rate-of-change in mid-freq band
#   Micro momentum  = sign + magnitude of high-freq band
#
# KEY RULE: higher freq → shorter, faster cycles nested
# inside larger ones.  We want:
#   1.  Low-freq band  in dip  (macro oversold)
#   2.  Mid-freq band  turning up  (swing reversal)
#   3.  High-freq band showing first micro-momentum burst
#
# All three aligning = high-probability reversal entry.


# ==========================================
# WAVELET FREQUENCY DECOMPOSITION ENGINE
# ==========================================

WAVELET   = 'db4'    # Daubechies-4 — good for price action
MAX_LEVEL = 5        # 5 levels → ~32-bar resolution at L5

def wavelet_decompose(prices: np.ndarray, level: int = MAX_LEVEL) -> Dict:
    """
    Multi-resolution analysis via Discrete Wavelet Transform.

    Returns reconstructed approximation (low-freq trend)
    and each detail band (high → low freq as level increases).

    Level mapping for 1m bars (approx):
      D1  → 2-4   bar cycles   (high-freq noise / micro)
      D2  → 4-8   bar cycles
      D3  → 8-16  bar cycles   (mid-freq swing)
      D4  → 16-32 bar cycles
      D5  → 32-64 bar cycles   (approaching trend)
      A5  → 64+   bar cycles   (low-freq macro trend)
    """
    arr = np.array(prices, dtype=float)
    # Pad to next power-of-2 for clean DWT
    n     = len(arr)
    n_pad = int(2 ** np.ceil(np.log2(max(n, 8))))
    padded = np.pad(arr, (0, n_pad - n), mode='edge')

    coeffs = pywt.wavedec(padded, WAVELET, level=level)
    # coeffs[0] = approx (A_level = low-freq)
    # coeffs[1..level] = details D_level..D1 (high to low freq)

    # Reconstruct each band back to original length
    bands = {}
    # Low-freq trend (approx band)
    rec_approx = pywt.waverec(
        [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], WAVELET
    )[:n]
    bands['trend'] = rec_approx   # LOW  freq (~64+ bar macro)

    # Individual detail bands
    for i in range(1, level + 1):
        zero = [np.zeros_like(c) for c in coeffs]
        zero[i] = coeffs[i]
        rec = pywt.waverec(zero, WAVELET)[:n]
        bands[f'D{i}'] = rec      # D1=highest freq, D5=lowest detail

    # Composite bands for practical use
    # "Swing" = D3+D4 (8-32 bar cycles)
    bands['swing'] = bands.get('D3', np.zeros(n)) + bands.get('D4', np.zeros(n))
    # "Micro"  = D1+D2 (2-8 bar cycles)
    bands['micro']  = bands.get('D1', np.zeros(n)) + bands.get('D2', np.zeros(n))

    return bands


def frequency_momentum(bands: Dict, lookback: int = 10) -> Dict:
    """
    Momentum score per frequency layer.

    Logic:
      trend_mom  : slope of low-freq band over last `lookback` bars
      swing_mom  : last value of swing band vs its recent min/max
      micro_mom  : sign of micro band derivative (burst detection)
    """
    def safe_slope(arr):
        if len(arr) < 2: return 0.0
        x = np.arange(len(arr), dtype=float)
        try:
            return float(np.polyfit(x, arr, 1)[0])
        except Exception:
            return 0.0

    trend  = bands.get('trend', np.array([]))
    swing  = bands.get('swing', np.array([]))
    micro  = bands.get('micro', np.array([]))

    # --- Trend momentum (macro direction) ---
    t_window = trend[-lookback:] if len(trend) >= lookback else trend
    trend_slope = safe_slope(t_window)
    # Normalise by std of trend band
    trend_std = float(np.std(trend[-50:]) + 1e-9)
    trend_mom_norm = float(np.clip(trend_slope / trend_std, -3.0, 3.0))

    # --- Swing momentum (are we reversing?) ---
    if len(swing) >= lookback:
        sw = swing[-lookback:]
        sw_min, sw_max = float(np.min(sw)), float(np.max(sw))
        sw_rng = sw_max - sw_min + 1e-9
        swing_pos = (float(swing[-1]) - sw_min) / sw_rng   # 0=bottom 1=top
        swing_slope = safe_slope(sw[-5:])
        swing_turning_up = bool(swing_slope > 0 and swing_pos < 0.35)
    else:
        swing_pos, swing_slope, swing_turning_up = 0.5, 0.0, False

    # --- Micro momentum (first burst signal) ---
    if len(micro) >= 5:
        micro_slope = safe_slope(micro[-5:])
        micro_std = float(np.std(micro[-20:]) + 1e-9)
        micro_mom_norm = float(np.clip(micro_slope / micro_std, -3.0, 3.0))
        micro_positive = bool(micro_mom_norm > 0.3)
    else:
        micro_mom_norm, micro_positive = 0.0, False

    # --- Dominant cycle length (FFT on trend-removed price) ---
    if len(swing) >= 32:
        fft_vals  = np.abs(np.fft.rfft(swing[-min(len(swing), 256):]))
        fft_freqs = np.fft.rfftfreq(len(swing[-min(len(swing), 256):]))
        fft_freqs[fft_freqs == 0] = 1e-12
        peak_idx  = int(np.argmax(fft_vals[1:]) + 1)
        dominant_cycle_bars = int(round(1.0 / (fft_freqs[peak_idx] + 1e-12)))
    else:
        dominant_cycle_bars = 0

    return {
        "trend_mom":          trend_mom_norm,
        "swing_pos":          float(swing_pos),
        "swing_slope":        float(swing_slope),
        "swing_turning_up":   swing_turning_up,
        "micro_mom":          micro_mom_norm,
        "micro_positive":     micro_positive,
        "dominant_cycle_bars": dominant_cycle_bars,
    }


def spectral_cycle_state(prices: np.ndarray, dt: float = 1.0) -> Dict:
    """
    FFT-based cycle position + phase state.
    Uses the top-3 dominant frequency components to reconstruct
    a composite cycle, then measures where in that cycle we are.

    Bass line  (low  freq): dominant trend oscillation
    Mid-band   (mid  freq): swing cycles inside trend
    Hi-freq    (high freq): micro bursts
    """
    n = len(prices)
    if n < 64:
        return {"cycle_phase_pct": 50.0, "cycle_state": "UNKNOWN",
                "bass_phase": 0.0, "mid_phase": 0.0, "hi_phase": 0.0,
                "cycle_score": 0.0}

    arr  = np.array(prices, dtype=float) - np.mean(prices)
    fft  = np.fft.rfft(arr)
    mags = np.abs(fft)
    freqs = np.fft.rfftfreq(n, dt)

    # Split into three frequency bands
    pos_mask = freqs > 0
    lo_mask  = pos_mask & (freqs <= 1/64)   # ≥64 bar periods
    mid_mask = pos_mask & (freqs > 1/64) & (freqs <= 1/16)  # 16–64 bar
    hi_mask  = pos_mask & (freqs > 1/16)    # <16 bar periods

    def band_phase(mask):
        if not np.any(mask): return 0.0
        idx = int(np.argmax(mags * mask))
        return float(np.angle(fft[idx]))

    def band_energy(mask):
        if not np.any(mask): return 0.0
        return float(np.sum(mags[mask] ** 2))

    bass_phase = band_phase(lo_mask)
    mid_phase  = band_phase(mid_mask)
    hi_phase   = band_phase(hi_mask)

    lo_e  = band_energy(lo_mask)
    mid_e = band_energy(mid_mask)
    hi_e  = band_energy(hi_mask)
    tot_e = lo_e + mid_e + hi_e + 1e-9

    # Cycle phase as % through current oscillation (0=trough, 50=peak, 100=trough again)
    # Use the strongest low-freq component
    lo_idxs = np.where(lo_mask)[0]
    if len(lo_idxs) > 0:
        best_lo  = lo_idxs[int(np.argmax(mags[lo_idxs]))]
        period   = int(round(1.0 / (freqs[best_lo] + 1e-12)))
        phase_rad = float(np.angle(fft[best_lo]))
        # Map phase to 0-100% of cycle (0%=trough, 25%=rising, 50%=peak, 75%=falling)
        phase_pct = float((phase_rad / (2 * np.pi) + 0.5) % 1.0) * 100.0
    else:
        period, phase_pct = 0, 50.0

    # Cycle state based on phase and energy distribution
    if phase_pct < 20:
        cycle_state = "TROUGH"         # at the bottom — prime buy zone
    elif phase_pct < 40:
        cycle_state = "RISING_EARLY"   # just turned up
    elif phase_pct < 60:
        cycle_state = "PEAK"           # top of cycle
    elif phase_pct < 80:
        cycle_state = "FALLING"        # coming down
    else:
        cycle_state = "TROUGH_APPROACH" # approaching next bottom

    # Score: high when near trough with mid/hi energy starting to rise
    trough_proximity = max(0.0, 1.0 - abs(phase_pct - 10) / 30.0)
    energy_quality   = min(mid_e / (tot_e + 1e-9) + hi_e / (tot_e + 1e-9), 1.0)
    cycle_score      = float(np.clip(0.6 * trough_proximity + 0.4 * energy_quality, 0.0, 1.0))

    return {
        "cycle_phase_pct":   phase_pct,
        "cycle_state":       cycle_state,
        "cycle_period_bars": period,
        "bass_phase":        bass_phase,
        "mid_phase":         mid_phase,
        "hi_phase":          hi_phase,
        "lo_energy_pct":     lo_e / tot_e * 100,
        "mid_energy_pct":    mid_e / tot_e * 100,
        "hi_energy_pct":     hi_e / tot_e * 100,
        "cycle_score":       cycle_score,
        "trough_proximity":  trough_proximity,
    }


def mtf_frequency_filter(prices_1m: List[float],
                          prices_5m: List[float],
                          prices_15m: List[float],
                          prices_4h: List[float]) -> Dict:
    """
    Multi-timeframe frequency alignment check.

    For a high-quality dip reversal we need ALL of:
      [4H  TREND]  : low-freq band declining (macro dip confirmed)
      [15M SWING]  : swing band turning up  (reversal initiated)
      [5M  MICRO]  : micro band positive    (momentum spark)
      [1M  CYCLE]  : near trough in cycle   (entry timing)
    """
    results = {}

    # 4H — check that low-freq trend is in dip
    if len(prices_4h) >= 32:
        bands_4h = wavelet_decompose(np.array(prices_4h), level=4)
        mom_4h   = frequency_momentum(bands_4h, lookback=20)
        # Dip = trend momentum negative (price declining)
        in_macro_dip = bool(mom_4h['trend_mom'] < -0.1)
        results['4h'] = {**mom_4h, 'in_macro_dip': in_macro_dip}
    else:
        results['4h'] = {'in_macro_dip': False, 'trend_mom': 0.0, 'swing_turning_up': False}

    # 15M — swing reversal
    if len(prices_15m) >= 32:
        bands_15m = wavelet_decompose(np.array(prices_15m), level=4)
        mom_15m   = frequency_momentum(bands_15m, lookback=15)
        results['15m'] = {**mom_15m}
    else:
        results['15m'] = {'swing_turning_up': False, 'micro_positive': False, 'trend_mom': 0.0}

    # 5M — micro momentum spark
    if len(prices_5m) >= 32:
        bands_5m = wavelet_decompose(np.array(prices_5m), level=4)
        mom_5m   = frequency_momentum(bands_5m, lookback=10)
        results['5m'] = {**mom_5m}
    else:
        results['5m'] = {'micro_positive': False, 'swing_turning_up': False, 'trend_mom': 0.0}

    # 1M — cycle phase (are we at trough?)
    if len(prices_1m) >= 64:
        cycle_1m = spectral_cycle_state(np.array(prices_1m))
        bands_1m = wavelet_decompose(np.array(prices_1m), level=5)
        mom_1m   = frequency_momentum(bands_1m, lookback=8)
        near_trough = cycle_1m['cycle_state'] in ('TROUGH', 'TROUGH_APPROACH', 'RISING_EARLY')
        results['1m'] = {**cycle_1m, **mom_1m, 'near_trough': near_trough}
    else:
        results['1m'] = {'near_trough': False, 'cycle_state': 'UNKNOWN', 'cycle_score': 0.0}

    # === ALIGNMENT SCORE ===
    # Each condition contributes to the alignment score
    cond_macro_dip     = bool(results['4h'].get('in_macro_dip', False))
    cond_swing_rev_15  = bool(results['15m'].get('swing_turning_up', False))
    cond_micro_spark   = bool(results['5m'].get('micro_positive', False))
    cond_1m_trough     = bool(results['1m'].get('near_trough', False))
    cond_swing_rev_5   = bool(results['5m'].get('swing_turning_up', False))

    alignment_count = sum([
        cond_macro_dip,
        cond_swing_rev_15,
        cond_micro_spark,
        cond_1m_trough,
        cond_swing_rev_5,
    ])
    alignment_score = alignment_count / 5.0

    # Dominant cycle from 1m
    dom_cycle = results['1m'].get('dominant_cycle_bars', 0)

    return {
        "4h":               results['4h'],
        "15m":              results['15m'],
        "5m":               results['5m'],
        "1m":               results['1m'],
        "cond_macro_dip":   cond_macro_dip,
        "cond_swing_rev":   cond_swing_rev_15 or cond_swing_rev_5,
        "cond_micro_spark": cond_micro_spark,
        "cond_1m_trough":   cond_1m_trough,
        "alignment_count":  alignment_count,
        "alignment_score":  alignment_score,
        "dominant_cycle":   dom_cycle,
        "strong_alignment": alignment_count >= 3,
    }


def print_frequency_block(freq: Dict, W: int = 74):
    """Pretty-print the frequency decomposition analysis."""
    print("─" * W)
    print("  📡  FREQUENCY DECOMPOSITION ENGINE (FFT + Wavelet MTF)")
    print("─" * W)

    def yn(v): return "✅ YES" if v else "❌ NO"
    def fmt(v): return f"{v:+.3f}"

    # 4H
    h4 = freq.get('4h', {})
    print(f"  [4H  TREND ]  macro dip?  {yn(h4.get('in_macro_dip'))}   "
          f"trend_mom={fmt(h4.get('trend_mom', 0))}  "
          f"swing_up={yn(h4.get('swing_turning_up'))}")

    # 15M
    m15 = freq.get('15m', {})
    print(f"  [15M SWING ]  swing rev?  {yn(m15.get('swing_turning_up'))}   "
          f"swing_pos={m15.get('swing_pos', 0.5)*100:.0f}%  "
          f"micro+={yn(m15.get('micro_positive'))}")

    # 5M
    m5 = freq.get('5m', {})
    print(f"  [5M  MICRO ]  micro+?     {yn(m5.get('micro_positive'))}   "
          f"micro_mom={fmt(m5.get('micro_mom', 0))}  "
          f"swing_up={yn(m5.get('swing_turning_up'))}")

    # 1M cycle
    m1 = freq.get('1m', {})
    cs  = m1.get('cycle_state', 'UNKNOWN')
    csc = m1.get('cycle_score', 0.0)
    cph = m1.get('cycle_phase_pct', 50.0)
    dcy = freq.get('dominant_cycle', 0)
    lo_e  = m1.get('lo_energy_pct', 0.0)
    mid_e = m1.get('mid_energy_pct', 0.0)
    hi_e  = m1.get('hi_energy_pct', 0.0)
    print(f"  [1M  CYCLE ]  near trough?{yn(m1.get('near_trough'))}   "
          f"state={cs:<16}  phase={cph:.0f}%  score={csc:.2f}")
    print(f"               Dominant cycle ≈ {dcy} bars")
    print(f"               Energy split  → "
          f"Trend(low):{lo_e:.0f}%  Swing(mid):{mid_e:.0f}%  Micro(hi):{hi_e:.0f}%")

    # Alignment bar
    ac = freq.get('alignment_count', 0)
    alabel = ["⚫ NONE", "🔴 WEAK", "🟡 MODERATE", "🟢 GOOD", "✅ STRONG", "🚀 PERFECT"][min(ac, 5)]
    bar = "█" * ac + "░" * (5 - ac)
    print(f"\n  Alignment: [{bar}] {ac}/5  {alabel}")
    conds = [
        ("Macro 4H dip",      freq.get('cond_macro_dip', False)),
        ("Swing reversal",    freq.get('cond_swing_rev', False)),
        ("Micro spark",       freq.get('cond_micro_spark', False)),
        ("1M cycle trough",   freq.get('cond_1m_trough', False)),
    ]
    for label, val in conds:
        icon = "✅" if val else "⬜"
        print(f"    {icon}  {label}")
    print()


# ==========================================
# GOLDEN HARMONIC ENGINE  (unchanged)
# ==========================================

def golden_signal(t: np.ndarray, omega0: float = 1.0, N: int = 3) -> np.ndarray:
    x = np.zeros_like(t, dtype=float)
    for n in range(-N, N + 1):
        omega = omega0 * (PHI ** n)
        A     = 1.0 / (PHI ** abs(n))
        x    += A * np.sin(omega * t)
    return x

def golden_fft_detect(signal: np.ndarray, dt: float = 1.0,
                       epsilon: float = 0.18) -> Tuple[np.ndarray, np.ndarray, float]:
    n         = len(signal)
    fft_vals  = np.fft.rfft(signal)
    freqs     = np.fft.rfftfreq(n, dt)
    magnitudes = np.abs(fft_vals)
    idx        = np.argsort(magnitudes)[-10:]
    peak_freqs = np.sort(freqs[idx])
    peak_freqs = peak_freqs[peak_freqs > 0]
    if len(peak_freqs) < 2: return peak_freqs, np.array([]), 0.0
    ratios         = peak_freqs[1:] / np.maximum(peak_freqs[:-1], 1e-12)
    golden_targets = np.array([PHI, PHI_SQ, PHI_INV])
    hits = sum(float(np.min(np.abs(r - golden_targets))) < epsilon for r in ratios)
    return peak_freqs, ratios, hits / len(ratios)

def compute_phase_alignment(close_prices: List[float], dt: float = 1.0,
                             omega0: float = None, N: int = 3,
                             epsilon: float = 0.18) -> Dict:
    if len(close_prices) < 64:
        return {"golden_score": 0.0, "energy_state": "INSUFFICIENT",
                "energy_ratio": 1.0, "spike_prob": 0.0,
                "phase_aligned": False, "near_min": False,
                "ratios": [], "pos_in_range": 0.5}
    arr      = np.array(close_prices, dtype=float)
    arr_norm = arr - np.mean(arr)
    if omega0 is None: omega0 = 2.0 * np.pi / len(arr_norm)
    peak_freqs, ratios, golden_score = golden_fft_detect(arr_norm, dt, epsilon)
    energy      = arr_norm ** 2
    mid         = len(energy) // 2
    early_e     = float(np.mean(energy[:mid]))
    recent_e    = float(np.mean(energy[mid:]))
    energy_ratio = recent_e / (early_e + 1e-9)
    if   energy_ratio < 0.40: energy_state = "COMPRESSION"
    elif energy_ratio < 0.75: energy_state = "BUILDING"
    elif energy_ratio < 1.40: energy_state = "EQUILIBRIUM"
    elif energy_ratio < 2.50: energy_state = "EXPANSION"
    else:                     energy_state = "PEAK"
    arr_min, arr_max = float(arr.min()), float(arr.max())
    rng          = arr_max - arr_min
    pos_in_range = (float(arr[-1]) - arr_min) / (rng + 1e-9)
    near_min     = pos_in_range < 0.25
    phase_aligned = (golden_score > 0.30 and energy_state in ("COMPRESSION", "BUILDING"))
    energy_bonus = {"COMPRESSION": 1.0, "BUILDING": 0.75, "EQUILIBRIUM": 0.40,
                    "EXPANSION": 0.20, "PEAK": 0.05}.get(energy_state, 0.0)
    spike_prob = float(np.clip(
        0.45 * golden_score + 0.35 * energy_bonus + 0.20 * float(near_min), 0.0, 1.0))
    return {"golden_score": float(golden_score), "energy_state": energy_state,
            "energy_ratio": float(energy_ratio), "spike_prob": spike_prob,
            "phase_aligned": bool(phase_aligned), "near_min": bool(near_min),
            "ratios": [float(r) for r in ratios], "pos_in_range": float(pos_in_range)}

def golden_fib_proximity(current_price: float, ref_low: float, ref_high: float) -> Dict:
    rng = ref_high - ref_low
    if rng <= 0: return {"nearest": "NONE", "dist_pct": 0.0, "level_price": current_price}
    results = {}
    for label, ratio in FIB_RATIOS.items():
        level_price = ref_low + rng * ratio
        dist_pct    = abs(current_price - level_price) / current_price * 100.0
        results[label] = {"price": level_price, "ratio": ratio, "dist_pct": dist_pct}
    nearest = min(results, key=lambda k: results[k]["dist_pct"])
    return {"nearest": nearest, "dist_pct": results[nearest]["dist_pct"],
            "level_price": results[nearest]["price"], "all_levels": results}


# ==========================================
# RATE LIMITER & TRADER
# ==========================================

class RateLimiter:
    def __init__(self, requests_per_second: float = 15, burst: int = 25):
        self.rate, self.burst, self.tokens = requests_per_second, burst, burst
        self.last_update, self.lock = time.time(), Lock()
    def acquire(self):
        while True:
            with self.lock:
                now     = time.time()
                elapsed = now - self.last_update
                self.tokens     = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_update = now
                if self.tokens >= 1: self.tokens -= 1; return
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
        pairs = [s['symbol'] for s in exchange_info['symbols']
                 if s['quoteAsset'] == 'USDC' and s['status'] == 'TRADING']
        print(f"Found {len(pairs)} USDC trading pairs"); return pairs
    def get_klines(self, symbol: str, interval: str, limit: int = 500,
                   return_raw: bool = False,
                   start_time: int = None, end_time: int = None):
        self.rate_limiter.acquire()
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        if start_time is not None: params['startTime'] = start_time
        if end_time   is not None: params['endTime']   = end_time
        for attempt in range(3):
            try:
                klines = self.client.get_klines(**params)
                return klines if return_raw else [float(k[4]) for k in klines]
            except Exception as e:
                time.sleep(2 ** attempt * 2 if 'rate limit' in str(e).lower() else 0.5)
        return []
    def get_klines_extended(self, symbol: str, interval: str, total: int = 1200):
        MAX = 1000
        if total <= MAX: return self.get_klines(symbol, interval, limit=total, return_raw=True)
        first = self.get_klines(symbol, interval, limit=MAX, return_raw=True)
        if not first: return []
        second = self.get_klines(symbol, interval, limit=total - MAX, return_raw=True,
                                  end_time=int(first[0][0]) - 1)
        return (second + first) if second else first


# ==========================================
# VOLUME BREAKDOWN (per-TF)
# ==========================================

def get_volume_breakdown(trader: Trader, symbol: str, interval: str,
                          limit: int = 100) -> Dict:
    klines = trader.get_klines(symbol, interval, limit=limit, return_raw=True)
    if not klines: return {'bull_pct': 50.0, 'bear_pct': 50.0, 'total': 0}
    bull = sum(float(k[5]) for k in klines if float(k[4]) >= float(k[1]))
    bear = sum(float(k[5]) for k in klines if float(k[4]) <  float(k[1]))
    tot  = bull + bear
    return {'bull_pct': bull / tot * 100 if tot > 0 else 50.0,
            'bear_pct': bear / tot * 100 if tot > 0 else 50.0,
            'bull': bull, 'bear': bear, 'total': tot}


# ==========================================
# CANDIDATE FILTER INDICATORS
# ==========================================

def is_confirmed_dip(close: list, high_tf: bool = False) -> bool:
    if len(close) < 200: return False
    arr = np.array(close, dtype=float)
    sma12, sma27, sma56, sma200 = (ta.SMA(arr, 12), ta.SMA(arr, 27),
                                    ta.SMA(arr, 56), ta.SMA(arr, 200))
    if high_tf:
        sma360 = ta.SMA(arr, 360) if len(arr) >= 360 else sma200
        return (arr[-1] < sma12[-1] and
                sma12[-1] < sma27[-1] < sma56[-1] < sma200[-1] and
                arr[-1] < sma360[-1])
    rsi  = ta.RSI(arr, 14)
    macd, macdsignal, macdhist = ta.MACD(arr)
    oversold       = rsi[-1] < 35
    momentum_shift = (macdhist[-1] > macdhist[-2]) and (macdhist[-1] > -0.5)
    return oversold and momentum_shift and rsi[-1] < 30

def is_below_regression_low(close: List[float], deviation: float = 0.01) -> bool:
    if len(close) < 20: return False
    x           = np.arange(len(close))
    slope, intercept = np.polyfit(x, close, 1)
    trend       = slope * x + intercept
    lower_band  = trend * (1 - deviation)
    return close[-1] < lower_band[-1]

def get_sinusoidal_dip_timing(close_prices: list, lookback: int = 500) -> Dict:
    """
    STRICT UPGRADE: Lowest sine extrema confirmed AND up cycle with pump incoming.
    Analyzes the last 20 bars of the wave macro-trend to avoid 1-bar noise bounces.
    """
    if len(close_prices) < lookback: lookback = len(close_prices)
    arr      = np.array(close_prices[-lookback:], dtype=float)
    arr_norm = arr - np.mean(arr)

    golden = compute_phase_alignment(arr_norm.tolist(), dt=1.0, N=3, epsilon=0.18)
    t      = np.arange(len(arr_norm))
    wave   = golden_signal(t, omega0=2 * np.pi / len(arr_norm), N=2)

    current_phase_pos = ((arr_norm[-1] - np.min(arr_norm)) /
                         (np.max(arr_norm) - np.min(arr_norm) + 1e-9))
    near_bottom = current_phase_pos < 0.20

    lookback_window = 20
    recent_wave     = wave[-lookback_window:]
    wave_abs_min_idx = int(np.argmin(recent_wave))

    just_hit_bottom = (lookback_window - wave_abs_min_idx) <= 5
    moving_upward   = (wave[-1] > np.min(recent_wave)) and (wave[-1] > wave[-3])

    turning_up          = just_hit_bottom and moving_upward
    confirmed_dip_pump  = near_bottom and turning_up

    cycle_length    = len(arr_norm) / 3.0
    bars_to_up      = (int(cycle_length * (0.75 - current_phase_pos))
                       if near_bottom else int(cycle_length * 0.6))

    # --- φ-Phase reversal engine ---
    phi_data = phi_phase_analysis(arr.tolist(), swing_window=5,
                                  fwd_bars=20, bins=20)

    # --- Stationarity-gated cyclic line ---
    dom_cycle = golden.get("dominant_cycle_bars", 0) or 32
    stat_data = stationarity_gated_cycle(arr, dom_cycle)

    return {
        "wave_near_bottom":  confirmed_dip_pump,
        "turning_up":        turning_up,
        "near_bottom_raw":   near_bottom,
        "est_bars_to_pump":  max(5, bars_to_up),
        "phase_pos":         float(current_phase_pos),
        **golden,
        **phi_data,
        **stat_data,
    }

def has_bullish_rejection_volume(raw_klines: list,
                                  window: int = 10) -> Tuple[bool, float]:
    if not raw_klines or len(raw_klines) < window: return False, 0.0
    recent   = raw_klines[-window:]
    bull_vol = bear_vol = 0.0
    for k in recent:
        o, c, v = float(k[1]), float(k[4]), float(k[5])
        if v > 0:
            if   c > o: bull_vol += v
            elif c < o: bear_vol += v
    total = bull_vol + bear_vol
    if total == 0: return False, 0.0
    ratio = bull_vol / total
    return ratio > 0.65, ratio

def calculate_effort_result_metrics(close: List[float],
                                     volumes: List[float],
                                     window: int = 20) -> Dict:
    if len(close) < window + 2: return {"R": 0, "C": 0, "E": 0}
    ca, va = np.array(close[-window:], dtype='float64'), np.array(volumes[-window:], dtype='float64')
    dp, tv, eps = abs(ca[-1] - ca[0]), np.sum(va), 1e-9
    return {"R": tv / (dp + eps), "C": tv / (np.std(ca) + eps),
            "E": tv / ((dp * window) + eps)}

def ml_spike_probability(R, C, E, bull_ratio, cmo, vratio) -> float:
    score = (0.30 * np.log1p(R) + 0.25 * np.log1p(C) +
             0.20 * np.log1p(E) + 0.15 * bull_ratio +
             0.05 * (-cmo / 100.0) + 0.05 * min(vratio / 5.0, 1.0))
    return 1 / (1 + np.exp(-score))


# ==========================================
# STRUCTURAL RANGE ENGINE (multi-lookback)
# ==========================================

def get_structural_extremes(close: np.ndarray, highs: np.ndarray,
                              lows: np.ndarray, lookback: int) -> Dict:
    n, start = len(close), max(0, len(close) - lookback)
    c, h, l  = close[start:], highs[start:], lows[start:]
    sl       = len(c)
    amax_i, amin_i = int(np.argmax(c)), int(np.argmin(c))
    g_high, g_low  = float(c[amax_i]), float(c[amin_i])
    high_age, low_age = sl - amax_i, sl - amin_i
    if   low_age < high_age:
        more_recent, mr_label = "ARGMIN", "🟢 ARGMIN (low is fresher → floor established recently)"
    elif high_age < low_age:
        more_recent, mr_label = "ARGMAX", "🔴 ARGMAX (high is fresher → ceiling established recently)"
    else:
        more_recent, mr_label = "EQUAL",  "⚪ EQUAL (both extremes same age)"
    rng     = g_high - g_low
    rng_pct = (rng / g_low * 100) if g_low > 0 else 0
    pos     = (close[-1] - g_low) / rng if rng > 0 else 0.5
    return {'high': g_high, 'low': g_low,
            'high_bar': float(h[amax_i]), 'low_bar': float(l[amin_i]),
            'high_age': high_age, 'low_age': low_age,
            'more_recent': more_recent, 'mr_label': mr_label,
            'range_size': rng, 'range_pct': rng_pct,
            'position': pos, 'bars_used': sl}

def build_fib_grid(extremes: Dict, current_price: float) -> List[Dict]:
    lo, hi, rng = extremes['low'], extremes['high'], extremes['range_size']
    if rng <= 0: return []
    fibs = [(0.000, "ARGMIN"), (0.236, "F236"), (0.382, "F382"),
            (0.500, "F500"),   (0.618, "F618"), (0.786, "F786"), (1.000, "ARGMAX")]
    grid = []
    for fib, label in fibs:
        price     = lo + rng * fib
        dist      = (price - current_price) / current_price * 100
        direction = 'UP' if price > current_price else ('DOWN' if price < current_price else 'AT')
        grid.append({'price': price, 'fib': fib, 'label': label,
                     'dist_pct': dist, 'direction': direction})
    return grid

def volume_profile_at_level(level_price: float, raw_klines: list,
                              tolerance: float) -> Dict:
    bull_vol = bear_vol = 0.0
    touches  = bull_rej = bear_rej = 0
    vol_seq  = []
    lo, hi   = level_price - tolerance, level_price + tolerance
    for k in raw_klines:
        o, h, l, c, v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
        if l <= hi and h >= lo:
            touches += 1; vol_seq.append(v)
            if v > 0:
                if c >= o: bull_vol += v
                else:      bear_vol += v
            if l < lo and c >= level_price: bull_rej += 1
            elif h > hi and c <= level_price: bear_rej += 1
    total = bull_vol + bear_vol
    bp    = bull_vol / total if total > 0 else 0.5
    exh, exh_detail = 0.0, "N/A"
    if len(vol_seq) >= 6:
        mid        = len(vol_seq) // 2
        first_avg  = np.mean(vol_seq[:mid])
        second_avg = np.mean(vol_seq[mid:])
        if first_avg > 0:
            ratio = second_avg / first_avg
            exh   = max(0.0, min(1.0, 1.0 - ratio))
            if   ratio < 0.5: exh_detail = f"STRONG ({ratio:.0%})"
            elif ratio < 0.8: exh_detail = f"MODERATE ({ratio:.0%})"
            else:             exh_detail = f"Weak ({ratio:.0%})"
    rej_vol_total = 0.0
    for k in raw_klines:
        o, h, l, c, v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
        if l <= hi and h >= lo:
            if (l < lo and c >= level_price) or (h > hi and c <= level_price):
                rej_vol_total += v
    rej_int = rej_vol_total / total if total > 0 else 0.0
    verdict = "SUPPORT" if bp > 0.58 else ("RESISTANCE" if bp < 0.42 else "NEUTRAL")
    return {'bull_pct': bp, 'total_volume': total, 'touches': touches,
            'bull_rej': bull_rej, 'bear_rej': bear_rej,
            'total_rej': bull_rej + bear_rej,
            'exhaustion': exh, 'exhaustion_detail': exh_detail,
            'rej_intensity': rej_int, 'verdict': verdict}


# ==========================================
# REJECTION CLUSTER ENERGY & SPIKE DETECTION
# ==========================================

def compute_rejection_cluster_score(vp: Dict, range_size: float) -> Dict:
    N, RI, V = vp['total_rej'], vp['rej_intensity'], vp['total_volume']
    if range_size <= 0 or V == 0:
        return {"score": 0.0, "raw": 0.0, "N": N, "state": "INVALID"}
    energy     = (N ** 2) * RI * (V / range_size)
    norm_energy = np.log1p(energy)
    state = ("NOISE" if N < 3 else
             ("BUILDING" if N <= 4 else
              ("COMPRESSION" if N <= 6 else "UNSTABLE")))
    return {"score": norm_energy, "raw": energy, "N": N, "state": state}

def detect_spike_trigger(curr_vp: Dict, prev_vp: Dict) -> bool:
    if not prev_vp: return False
    return (curr_vp['total_rej'] <= prev_vp['total_rej'] and
            curr_vp['rej_intensity'] < prev_vp['rej_intensity'] and
            curr_vp['total_volume'] > prev_vp['total_volume'] * 0.8)

def is_valid_spike(cluster: Dict, vp: Dict, vol_bias: float) -> bool:
    return (cluster['state'] in ["COMPRESSION", "UNSTABLE"] and
            cluster['score'] > 1.5 and
            vp['rej_intensity'] > 0.2 and
            0.45 < vol_bias < 0.65)

def detect_cluster_transition(cluster: Dict, prev_cluster: Dict) -> bool:
    return bool(prev_cluster and
                prev_cluster['state'] == "COMPRESSION" and
                cluster['state'] == "UNSTABLE")

def detect_extreme_exhaustion(extreme_price: float, direction: str,
                               raw_klines: list, zone_pct: float = 0.04) -> Dict:
    z_lo = (extreme_price * (1 - zone_pct) if direction == 'high'
            else extreme_price * (1 - zone_pct * 0.5))
    z_hi = (extreme_price * (1 + zone_pct * 0.5) if direction == 'high'
            else extreme_price * (1 + zone_pct))
    zv, zc = [], []
    for k in raw_klines:
        h, l, c, v = float(k[2]), float(k[3]), float(k[4]), float(k[5])
        if l <= z_hi and h >= z_lo: zv.append(v); zc.append(c)
    if len(zv) < 8:
        return {'exhaustion': 0.0, 'detail': 'Insufficient data',
                'pattern': 'NONE', 'approach_vol': 0, 'final_vol': 0}
    split    = int(len(zv) * 0.6)
    app_vol  = np.mean(zv[:split])
    fin_vol  = np.mean(zv[split:])
    reached  = ((max(zc[split:]) >= extreme_price * 0.998) if direction == 'high'
                else (min(zc[split:]) <= extreme_price * 1.002))
    exh, pattern, detail = 0.0, "NONE", ""
    if app_vol > 0:
        ratio = fin_vol / app_vol
        if direction == 'high':
            if   ratio < 0.4 and reached:
                exh, pattern, detail = 0.9, "CLIMAX_EXHAUSTION", f"Vol collapsed to {ratio:.0%} at peak → rejection likely"
            elif ratio < 0.65 and reached:
                exh, pattern, detail = 0.6, "FADE", f"Vol faded to {ratio:.0%} near high"
            elif ratio < 0.85:
                exh, pattern, detail = 0.3, "MILD_FADE", f"Slight fade to {ratio:.0%}"
            else:
                detail = f"No exhaustion (vol at {ratio:.0%})"
        else:
            if   ratio < 0.35 and reached:
                exh, pattern, detail = 0.9, "CAPITULATION", f"Vol died to {ratio:.0%} after low → bounce likely"
            elif ratio < 0.6 and reached:
                exh, pattern, detail = 0.6, "SELLING_EXHAUSTION", f"Selling exhausted at {ratio:.0%}"
            elif ratio < 0.85:
                exh, pattern, detail = 0.3, "MILD_EXHAUSTION", f"Mild exhaustion at {ratio:.0%}"
            else:
                detail = f"No exhaustion (vol at {ratio:.0%})"
    else:
        detail = "No approach volume"
    return {'exhaustion': exh, 'detail': detail, 'pattern': pattern,
            'approach_vol': app_vol, 'final_vol': fin_vol}


def score_level(level: Dict, vp: Dict, extreme_exh: float,
                range_pct: float, is_above: bool) -> Tuple[float, Dict]:
    bp, touches = vp['bull_pct'], vp['touches']
    rej, ri, exh = vp['total_rej'], vp['rej_intensity'], vp['exhaustion']
    pressure   = max(0.0, (0.5 - bp) * 2.0) if is_above else max(0.0, (bp - 0.5) * 2.0)
    rej_bonus  = (vp['bear_rej'] if is_above else vp['bull_rej']) / max(touches, 1)
    touch_sc   = min(touches / 12.0, 1.0) if touches > 0 else 0.0
    rej_sc     = min(rej_bonus * 3.0, 1.0)
    ri_sc      = min(ri * 5.0, 1.0)
    exh_sc     = extreme_exh
    fib        = level['fib']
    ext_prox   = (max(0.0, 1.0 - abs(fib - 1.0) * 2.0) if is_above
                  else max(0.0, 1.0 - abs(fib - 0.0) * 2.0))
    score = (0.25 * pressure + 0.15 * touch_sc + 0.15 * rej_sc +
             0.15 * ri_sc + 0.20 * exh_sc + 0.10 * ext_prox)
    return score, {'pressure': pressure, 'touches': touch_sc,
                   'rejection_candles': rej_sc, 'rejection_intensity': ri_sc,
                   'exhaustion': exh_sc, 'extreme_prox': ext_prox}


def estimate_eta(dist_pct: float, range_pct: float, vol_bias: float) -> str:
    if range_pct <= 0: return "N/A"
    bias  = vol_bias if vol_bias >= 0.5 else (1.0 - vol_bias)
    speed = ((0.7 + 0.6 * bias) if
             ((dist_pct > 0 and vol_bias > 0.5) or (dist_pct < 0 and vol_bias < 0.5))
             else (1.2 + 0.8 * (1.0 - bias)))
    mins  = abs(dist_pct) / max(range_pct, 0.01) * 360 * speed
    if   mins < 5:   return "~1-5 min"
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
                'targets_down': [], 'exh_high': {}, 'exh_low': {}, 'grid': [], 'min_dist': 0}
    grid      = build_fib_grid(ext, current_price)
    tolerance = max(ext['range_size'] * 0.025, avg_range_pct / 100 * current_price * 1.5)
    klines_slice = raw_klines[max(0, len(close) - lookback):]
    exh_high  = detect_extreme_exhaustion(ext['high'], 'high', klines_slice)
    exh_low   = detect_extreme_exhaustion(ext['low'],  'low',  klines_slice)
    min_dist  = max(ext['range_pct'] * 0.05, 0.08)
    targets_up, targets_down, grid_out = [], [], []
    prev_vp, prev_cluster = None, None
    for level in grid:
        vp          = volume_profile_at_level(level['price'], klines_slice, tolerance)
        cluster     = compute_rejection_cluster_score(vp, ext['range_size'])
        trigger     = detect_spike_trigger(vp, prev_vp)
        valid_spike = is_valid_spike(cluster, vp, vol_bias)
        explosion   = detect_cluster_transition(cluster, prev_cluster)
        prev_vp, prev_cluster = vp, cluster
        lev_exh     = (exh_high['exhaustion'] if level['fib'] >= 0.618
                       else (exh_low['exhaustion'] if level['fib'] <= 0.382 else 0.0))
        dist        = abs(level['dist_pct'])
        is_above    = level['direction'] == 'UP'
        is_below_p  = level['direction'] == 'DOWN'
        if dist < min_dist:
            grid_out.append({**level, **vp, 'score': 0, 'status': 'TOO_CLOSE',
                              'cluster_score': cluster['score'],
                              'cluster_state': cluster['state'],
                              'valid_spike': valid_spike,
                              'trigger': trigger, 'explosion': explosion})
            continue
        score, details = score_level(level, vp, lev_exh, ext['range_pct'], is_above)
        score += min(cluster['score'] * 0.15, 0.3)
        eta    = estimate_eta(level['dist_pct'], ext['range_pct'], vol_bias)
        entry  = {'price': level['price'], 'score': score,
                  'dist_pct': level['dist_pct'], 'label': level['label'],
                  'fib': level['fib'], 'verdict': vp['verdict'],
                  'bull_pct': vp['bull_pct'], 'touches': vp['touches'],
                  'rejections': vp['total_rej'], 'rej_intensity': vp['rej_intensity'],
                  'cluster_score': cluster['score'], 'cluster_state': cluster['state'],
                  'cluster_raw': cluster['raw'], 'trigger': trigger,
                  'valid_spike': valid_spike, 'explosion': explosion,
                  'eta': eta, 'details': details}
        grid_out.append({**level, **vp, **entry, 'status': 'ACTIVE'})
        if is_above:   targets_up.append(entry)
        elif is_below_p: targets_down.append(entry)
    targets_up.sort(key=lambda t: t['score'], reverse=True)
    targets_down.sort(key=lambda t: t['score'], reverse=True)
    return {'lookback': lookback, 'extremes': ext,
            'targets_up':   sorted(targets_up[:4],   key=lambda t:  t['dist_pct']),
            'targets_down': sorted(targets_down[:4], key=lambda t: -t['dist_pct']),
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
    closed_vols   = [v for v in volumes[-21:-1] if v > 0]
    vol_bias      = 0.5
    if closed_vols:
        rec    = raw_klines[-21:-1]
        bv     = sum(float(k[5]) for k in rec if float(k[4]) >= float(k[1]) and float(k[5]) > 0)
        bear_v = sum(float(k[5]) for k in rec if float(k[4]) <  float(k[1]) and float(k[5]) > 0)
        tv     = bv + bear_v
        vol_bias = bv / tv if tv > 0 else 0.5
    # ── Adaptive lookbacks anchored to dominant cycle ─────────────────
    # Detect dominant cycle from closes for window calibration
    _dom_cycle = 40  # fallback
    if len(closes) >= 64:
        try:
            _fft_v = np.abs(np.fft.rfft(closes - np.mean(closes)))
            _fft_f = np.fft.rfftfreq(len(closes))
            _fft_f[_fft_f == 0] = 1e-12
            _peak  = int(np.argmax(_fft_v[1:]) + 1)
            _dc    = int(round(1.0 / (_fft_f[_peak] + 1e-12)))
            _dom_cycle = int(np.clip(_dc, 10, 200))
        except Exception:
            pass

    # Three windows: 3x, 6x, 10x dominant cycle — tight, moderate, wide
    # Minimum 60 bars; cap at available data
    _lbs_raw = [
        max(60,  _dom_cycle * 3),
        max(100, _dom_cycle * 6),
        max(150, _dom_cycle * 10),
    ]
    _seen, _lbs = set(), []
    for _lb in _lbs_raw:
        _lb = min(_lb, len(raw_klines))
        if _lb not in _seen and _lb >= 60:
            _seen.add(_lb); _lbs.append(_lb)

    lookbacks = [
        analyze_lookback(raw_klines, closes, highs, lows, current_price,
                         lb, avg_range, vol_bias)
        for lb in _lbs if len(raw_klines) >= lb
    ]
    return {'lookbacks': lookbacks, 'vol_bias': vol_bias,
            'avg_range': avg_range, 'dom_cycle': _dom_cycle}


# ==========================================
# CONCURRENT FILTER FUNCTIONS
# ==========================================

def check_tf_dip(trader, symbol, interval):
    close = trader.get_klines(symbol, interval, limit=500)
    return (symbol, is_confirmed_dip(close, high_tf=True))

def check_5m_regression(trader, symbol):
    close = trader.get_klines(symbol, '5m', limit=100)
    return (symbol, is_below_regression_low(close, deviation=0.01))

def score_15m_candidate(trader, symbol) -> Tuple[str, float, dict]:
    """
    Score a symbol that passed 15m but failed the 5m regression gate.
    Returns (symbol, score, combined_dict) ranked by reversal potential.

    Scoring is built from whatever is already in-memory cheap:
      - 1m CMO oversold depth          (deeper = better)
      - 1m bullish rejection volume     (>65% bull = good)
      - ML spike probability
      - Golden phase alignment / spike_prob
      - Sinusoidal dip timing (wave near bottom + turning up)
      - Wavelet MTF frequency alignment
      - Position in range (lower = closer to floor)
    No 5m regression requirement here — this is the safety net.
    """
    try:
        klines = trader.get_klines(symbol, '1m', limit=500, return_raw=True)
        if not klines or len(klines) < 100:
            return (symbol, 0.0, {})

        close   = [float(k[4]) for k in klines]
        volumes = [float(k[5]) for k in klines]

        cmo     = ta.CMO(np.asarray(close), timeperiod=14)
        cmo_val = float(cmo[-1]) if not np.isnan(cmo[-1]) else 0.0

        closed_vols = [v for v in volumes[:-1] if v > 0]
        vratio      = 0.0
        if closed_vols:
            avg_vol     = np.mean(closed_vols[-50:])
            last_closed = closed_vols[-1]
            vratio      = last_closed / avg_vol if avg_vol > 0 else 0.0

        is_rej, bull_ratio = has_bullish_rejection_volume(klines, window=10)
        metrics = calculate_effort_result_metrics(close, volumes, window=20)
        prob    = ml_spike_probability(metrics["R"], metrics["C"], metrics["E"],
                                       bull_ratio, cmo_val, vratio)

        golden     = compute_phase_alignment(close, dt=1.0, N=3, epsilon=0.18)
        sinusoidal = get_sinusoidal_dip_timing(close, 500)
        combined   = {**golden, **sinusoidal}

        # Wavelet MTF (best effort)
        freq_result = None
        try:
            prices_5m  = trader.get_klines(symbol, '5m',  limit=100)
            prices_15m = trader.get_klines(symbol, '15m', limit=100)
            prices_4h  = trader.get_klines(symbol, '4h',  limit=100)
            if prices_5m and prices_15m and prices_4h:
                freq_result = mtf_frequency_filter(close, prices_5m, prices_15m, prices_4h)
        except Exception:
            freq_result = None
        combined['freq'] = freq_result

        # ── Quadratic price forecast (fallback path) ──────────────────
        klines_tight_fb = klines[-200:] if len(klines) > 200 else klines
        dc_fb = int((freq_result or {}).get("dominant_cycle", 0) or 0)
        combined['quad_forecast'] = quadratic_price_forecast(
            klines_tight_fb, float(klines[-1][4]), combined, dom_cycle=dc_fb)

        # --- Composite reversal score (no regression gate) ---
        gs             = combined.get("spike_prob", 0.0)
        near_bottom    = combined.get("wave_near_bottom", False)
        turning_up     = combined.get("turning_up", False)
        est_bars       = combined.get("est_bars_to_pump", 999)
        pos_in_range   = combined.get("pos_in_range", 0.5)
        freq_align     = (freq_result or {}).get("alignment_score", 0.0)
        cmo_score      = float(np.clip(-cmo_val / 100.0, 0.0, 1.0))
        reversal_score = combined.get("reversal_score", 0.0)
        at_cyclic_low  = combined.get("cyc_at_cyclic_low", False)
        phi_fwd_bias   = float(combined.get("phi_fwd_bias", 0.0))
        phi_ready      = combined.get("phi_reversal_ready", False)

        score = (
            prob          * 0.25 +
            gs            * 0.18 +
            cmo_score     * 0.12 +
            freq_align    * 0.12 +
            reversal_score * 0.12 +
            bull_ratio    * 0.08 +
            phi_fwd_bias * 5.0 * 0.05 +
            (1.0 - pos_in_range) * 0.08
        )
        if near_bottom:   score += 0.12
        if turning_up:    score += 0.08
        if at_cyclic_low: score += 0.07
        if phi_ready:     score += 0.06
        if est_bars < 20: score += 0.05

        combined['_cmo_val']    = cmo_val
        combined['_vratio']     = vratio
        combined['_bull_ratio'] = bull_ratio
        combined['_ml_prob']    = prob

        return (symbol, float(score), combined)

    except Exception:
        return (symbol, 0.0, {})


def run_best_mtf_fallback(trader, symbols: List[str],
                           max_workers: int = 15) -> Tuple:
    """
    When 5m regression gate kills all candidates, score each 15m-passed
    symbol for raw reversal potential and return the best one as a
    result tuple compatible with the normal 1m-filter output format:
      (symbol, cmo_val, vratio, is_strong=False, bull_ratio, ml_prob, combined)
    """
    if not symbols:
        return None

    results = []
    tracker = ProgressTracker(len(symbols), "MTF fallback scoring")
    print(f"\n⚡  5m regression gate failed — scoring {len(symbols)} candidates for best MTF dip...")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(score_15m_candidate, trader, s): s for s in symbols}
        for f in as_completed(futures):
            try:
                res = f.result()
                if res[2]:   # non-empty combined dict
                    results.append(res)
                tracker.update(passed=bool(res[2]))
                print(tracker.get_stats(), end="", flush=True)
            except:
                tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20)

    if not results:
        return None

    results.sort(key=lambda r: r[1], reverse=True)
    best_sym, best_score, best_combined = results[0]

    cmo_val    = best_combined.get('_cmo_val', 0.0)
    vratio     = best_combined.get('_vratio', 0.0)
    bull_ratio = best_combined.get('_bull_ratio', 0.0)
    ml_prob    = best_combined.get('_ml_prob', 0.0)

    # Mark as fallback so the output banner knows
    best_combined['_is_fallback']    = True
    best_combined['_fallback_score'] = best_score
    best_combined['_fallback_rank']  = f"1/{len(results)}"

    return (best_sym, cmo_val, vratio, False, bull_ratio, ml_prob, best_combined)


def score_daily_best_candidate(trader, symbol) -> Tuple[str, float, dict]:
    """
    Score a symbol that is confirmed at a dip on the daily TF.
    Called when max scans are exhausted and no full MTF setup was found.

    Scoring weights daily-TF conviction more heavily than score_15m_candidate:
      - 1D CMO oversold depth        (primary — deeper = more depressed = more bounce fuel)
      - 1D pos_in_range              (lower = closer to floor)
      - spike_prob from golden engine (φ + energy compression)
      - reversal_score (ADF + cyclic line)
      - φ fwd_bias and φ_ready
      - Freq alignment (best-effort MTF)
      - ML prob on 1m
      - wave_near_bottom / turning_up on 1m

    Returns (symbol, score, combined_dict).
    """
    try:
        # ── Daily data for primary score ───────────────────────────────
        klines_1d = trader.get_klines(symbol, '1d', limit=500, return_raw=True)
        if not klines_1d or len(klines_1d) < 60:
            return (symbol, 0.0, {})

        close_1d   = [float(k[4]) for k in klines_1d]
        volumes_1d = [float(k[5]) for k in klines_1d]

        cmo_1d_arr = ta.CMO(np.asarray(close_1d, dtype=float), timeperiod=14)
        cmo_1d     = float(cmo_1d_arr[-1]) if not np.isnan(cmo_1d_arr[-1]) else 0.0

        # daily sinusoidal / golden / φ-phase / cyclic analysis
        sinusoidal_1d = get_sinusoidal_dip_timing(close_1d, min(len(close_1d), 500))

        # position in daily range
        arr_1d     = np.array(close_1d, dtype=float)
        d_min, d_max = float(arr_1d.min()), float(arr_1d.max())
        pos_1d     = (float(arr_1d[-1]) - d_min) / (d_max - d_min + 1e-9)

        # ── 1m data for micro scoring ───────────────────────────────────
        klines_1m = trader.get_klines(symbol, '1m', limit=500, return_raw=True)
        if not klines_1m or len(klines_1m) < 100:
            return (symbol, 0.0, {})

        close_1m   = [float(k[4]) for k in klines_1m]
        volumes_1m = [float(k[5]) for k in klines_1m]

        closed_vols = [v for v in volumes_1m[:-1] if v > 0]
        vratio      = 0.0
        if closed_vols:
            avg_vol = np.mean(closed_vols[-50:])
            vratio  = closed_vols[-1] / avg_vol if avg_vol > 0 else 0.0

        is_rej, bull_ratio = has_bullish_rejection_volume(klines_1m, window=10)
        metrics = calculate_effort_result_metrics(close_1m, volumes_1m, window=20)
        cmo_1m_arr = ta.CMO(np.asarray(close_1m, dtype=float), timeperiod=14)
        cmo_1m     = float(cmo_1m_arr[-1]) if not np.isnan(cmo_1m_arr[-1]) else 0.0
        prob       = ml_spike_probability(metrics["R"], metrics["C"], metrics["E"],
                                          bull_ratio, cmo_1m, vratio)

        golden_1m     = compute_phase_alignment(close_1m, dt=1.0, N=3, epsilon=0.18)
        sinusoidal_1m = get_sinusoidal_dip_timing(close_1m, 500)

        # MTF wavelet (best effort)
        freq_result = None
        try:
            prices_5m  = trader.get_klines(symbol, '5m',  limit=100)
            prices_15m = trader.get_klines(symbol, '15m', limit=100)
            prices_4h  = trader.get_klines(symbol, '4h',  limit=100)
            if prices_5m and prices_15m and prices_4h:
                freq_result = mtf_frequency_filter(close_1m, prices_5m, prices_15m, prices_4h)
        except Exception:
            freq_result = None

        # Merge combined from daily sinusoidal + 1m analysis
        combined = {**golden_1m, **sinusoidal_1m}
        for k, v in sinusoidal_1d.items():
            combined.setdefault(f"_1d_{k}", v)
        combined['freq'] = freq_result

        # Quadratic forecast using 1m klines
        klines_tight = klines_1m[-200:] if len(klines_1m) > 200 else klines_1m
        dc_val = int((freq_result or {}).get("dominant_cycle", 0) or 0)
        combined['quad_forecast'] = quadratic_price_forecast(
            klines_tight, float(klines_1m[-1][4]), combined, dom_cycle=dc_val)

        # ── Composite score: daily-conviction weighted ──────────────────
        gs             = combined.get("spike_prob", 0.0)
        near_bottom    = combined.get("wave_near_bottom", False)
        turning_up     = combined.get("turning_up", False)
        est_bars       = combined.get("est_bars_to_pump", 999)
        reversal_score = combined.get("reversal_score", 0.0)
        at_cyclic_low  = combined.get("cyc_at_cyclic_low", False)
        phi_fwd_bias   = float(combined.get("phi_fwd_bias", 0.0))
        phi_ready      = bool(combined.get("phi_reversal_ready", False))
        freq_align     = float((freq_result or {}).get("alignment_score", 0.0))

        # daily CMO as primary driver: deeper oversold = higher score
        cmo_1d_score   = float(np.clip(-cmo_1d / 100.0, 0.0, 1.0))
        # daily position in range: lower is better
        daily_pos_score = float(np.clip(1.0 - pos_1d, 0.0, 1.0))

        score = (
            cmo_1d_score    * 0.28 +   # daily CMO depth — primary signal
            daily_pos_score * 0.15 +   # how close to 1D floor
            gs              * 0.15 +   # golden spike prob (φ energy)
            prob            * 0.12 +   # ML micro prob
            reversal_score  * 0.10 +   # cyclic reversal confidence
            freq_align      * 0.08 +   # MTF freq alignment
            phi_fwd_bias * 5.0 * 0.04 +
            bull_ratio      * 0.08
        )
        if near_bottom:   score += 0.10
        if turning_up:    score += 0.08
        if at_cyclic_low: score += 0.06
        if phi_ready:     score += 0.05
        if est_bars < 20: score += 0.04
        # Extra bonus: daily + 1m both oversold simultaneously
        if cmo_1d < -50 and cmo_1m < -40:
            score += 0.06

        # Stash raw metrics for output banner
        combined['_cmo_1d']    = cmo_1d
        combined['_cmo_1m']    = cmo_1m
        combined['_vratio']    = vratio
        combined['_bull_ratio'] = bull_ratio
        combined['_ml_prob']   = prob
        combined['_pos_1d']    = pos_1d

        return (symbol, float(score), combined)

    except Exception:
        return (symbol, 0.0, {})


def run_daily_best_fallback(trader, symbols: List[str],
                             max_workers: int = 15) -> Tuple:
    """
    Last-resort fallback: called when max_scans exhausted.
    Scores every USDC pair that passes the 1D dip filter right now,
    picks the one with the best daily dip + spike potential,
    and returns it as a result tuple compatible with the main output path:
      (symbol, cmo_val, vratio, False, bull_ratio, ml_prob, combined)
    """
    if not symbols:
        return None

    # Re-run 1D filter to get current daily dips
    W = 78
    print("\n" + "╔" + "═" * W + "╗")
    print("║" + " " * W + "║")
    print("║" + "  🌅  DAILY BEST-DIP FALLBACK — scanning 1D dips across universe...".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("║" + "  (Max scans reached with no full MTF setup — delivering best daily candidate)".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("╚" + "═" * W + "╝")

    daily_dips = run_tf_filter(trader, symbols, '1d', max_workers=20)
    if not daily_dips:
        print("⚠️  No daily dips found in universe right now.")
        return None

    print(f"\n📊  {len(daily_dips)} daily dips found — scoring for best spike/pump candidate...")

    results = []
    tracker = ProgressTracker(len(daily_dips), "Daily best-dip scoring")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(score_daily_best_candidate, trader, s): s for s in daily_dips}
        for f in as_completed(futures):
            try:
                res = f.result()
                if res[2]:
                    results.append(res)
                tracker.update(passed=bool(res[2]))
                print(tracker.get_stats(), end="", flush=True)
            except Exception:
                tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20)

    if not results:
        return None

    results.sort(key=lambda r: r[1], reverse=True)
    best_sym, best_score, best_combined = results[0]

    cmo_val    = best_combined.get('_cmo_1m', best_combined.get('_cmo_1d', 0.0))
    vratio     = best_combined.get('_vratio', 0.0)
    bull_ratio = best_combined.get('_bull_ratio', 0.0)
    ml_prob    = best_combined.get('_ml_prob', 0.0)

    best_combined['_is_daily_fallback']   = True
    best_combined['_daily_fallback_score'] = best_score
    best_combined['_daily_fallback_rank']  = f"1/{len(results)}"
    best_combined['_total_daily_dips']     = len(daily_dips)

    return (best_sym, cmo_val, vratio, False, bull_ratio, ml_prob, best_combined)


def check_1m_final(trader, symbol):
    """
    Enhanced 1m final check with wavelet frequency decomposition.

    New frequency-layer conditions added on top of original 4:
      cond_5: Wavelet swing reversal on 15m
      cond_6: Micro momentum positive spark (5m)
      cond_7: 1m cycle near trough (spectral)
      cond_8: MTF frequency alignment >= 3/5
    """
    klines = trader.get_klines(symbol, '1m', limit=500, return_raw=True)
    default_dict = {"golden_score": 0.0, "energy_state": "INSUFFICIENT",
                    "spike_prob": 0.0, "phase_aligned": False, "near_min": False,
                    "wave_near_bottom": False, "turning_up": False,
                    "est_bars_to_pump": 0, "phase_pos": 0.5,
                    "freq": None}
    if not klines or len(klines) < 100:
        return (symbol, 0.0, 0.0, False, 0.0, 0.0, default_dict)

    close   = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    cmo     = ta.CMO(np.asarray(close), timeperiod=14)
    cmo_val = float(cmo[-1]) if not np.isnan(cmo[-1]) else 0.0

    closed_vols = [v for v in volumes[:-1] if v > 0]
    vratio      = 0.0
    if closed_vols:
        avg_vol     = np.mean(closed_vols[-50:])
        last_closed = closed_vols[-1]
        vratio      = last_closed / avg_vol if avg_vol > 0 else 0.0

    is_rej, bull_ratio = has_bullish_rejection_volume(klines, window=10)
    metrics = calculate_effort_result_metrics(close, volumes, window=20)
    prob    = ml_spike_probability(metrics["R"], metrics["C"], metrics["E"],
                                   bull_ratio, cmo_val, vratio)

    golden    = compute_phase_alignment(close, dt=1.0, N=3, epsilon=0.18)
    sinusoidal = get_sinusoidal_dip_timing(close, 500)
    combined  = {**golden, **sinusoidal}

    # ----- Frequency decomposition (new) -----
    # Pull extra timeframes for MTF wavelet check
    # (these are lightweight fetches — 100 bars each)
    freq_result = None
    try:
        prices_5m  = trader.get_klines(symbol, '5m',  limit=100)
        prices_15m = trader.get_klines(symbol, '15m', limit=100)
        prices_4h  = trader.get_klines(symbol, '4h',  limit=100)
        if prices_5m and prices_15m and prices_4h:
            freq_result = mtf_frequency_filter(close, prices_5m,
                                               prices_15m, prices_4h)
    except Exception:
        freq_result = None

    combined['freq'] = freq_result

    # ── Quadratic price forecast ──────────────────────────────────────
    # Use a tight klines window (200 bars max) so targets will be hit
    klines_tight = klines[-200:] if len(klines) > 200 else klines
    dom_cycle_now = int((freq_result or {}).get("dominant_cycle", 0) or 0)
    quad_forecast = quadratic_price_forecast(
        klines_tight, current_price, combined, dom_cycle=dom_cycle_now)
    combined['quad_forecast'] = quad_forecast

    # ORIGINAL GATEKEEPERS
    cond_1 = is_confirmed_dip(close, high_tf=False)
    cond_2 = is_below_regression_low(close, deviation=0.01)
    cond_3 = combined.get('wave_near_bottom', False)
    cond_4 = combined.get('turning_up', False)

    # NEW FREQUENCY GATEKEEPERS
    if freq_result:
        cond_5 = bool(freq_result.get('cond_swing_rev', False))        # swing reversing
        cond_6 = bool(freq_result.get('cond_micro_spark', False))      # micro burst
        cond_7 = bool(freq_result.get('cond_1m_trough', False))        # cycle trough
        cond_8 = bool(freq_result.get('strong_alignment', False))      # 3+/5 aligned
    else:
        cond_5 = cond_6 = cond_7 = cond_8 = False

    # PASS RULE:
    #   Original 4 must hold (strict structural dip confirmed)
    #   + at least 2 of the 4 new frequency conditions
    freq_extra = sum([cond_5, cond_6, cond_7, cond_8])
    is_strong  = cond_1 and cond_2 and cond_3 and cond_4 and freq_extra >= 2

    return (symbol, cmo_val, vratio, is_strong, bull_ratio, prob, combined)


class ProgressTracker:
    def __init__(self, total, label):
        self.total      = total
        self.label      = label
        self.completed  = 0
        self.passed     = 0
        self.lock       = Lock()
        self.start_time = time.time()
    def update(self, passed=False):
        with self.lock:
            self.completed += 1
            if passed: self.passed += 1
    def get_stats(self):
        with self.lock:
            e   = time.time() - self.start_time
            r   = self.completed / e if e > 0 else 0
            rem = (self.total - self.completed) / r if r > 0 else 0
            return (f"\r{self.label}: {self.completed}/{self.total} | "
                    f"✓{self.passed} | {r:.1f}/s | ETA: {rem:.0f}s")

def run_tf_filter(trader, symbols, interval, max_workers=20):
    passed, tracker = [], ProgressTracker(len(symbols), f"{interval} filter")
    print(f"Running {interval} filter on {len(symbols)} pairs...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(check_tf_dip, trader, s, interval): s for s in symbols}
        for f in as_completed(futures):
            try:
                sym, ok = f.result()
                if ok: passed.append(sym)
                tracker.update(passed=ok)
                print(tracker.get_stats(), end="", flush=True)
            except: tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20)
    return passed

def run_5m_regression_filter(trader, symbols, max_workers=20):
    passed, tracker = [], ProgressTracker(len(symbols), "5m Reg filter")
    print(f"Running 5m Lowest Regression filter on {len(symbols)} pairs...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(check_5m_regression, trader, s): s for s in symbols}
        for f in as_completed(futures):
            try:
                sym, ok = f.result()
                if ok: passed.append(sym)
                tracker.update(passed=ok)
                print(tracker.get_stats(), end="", flush=True)
            except: tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20)
    return passed

def run_1m_filter(trader, symbols, max_workers=15):
    results, tracker = [], ProgressTracker(len(symbols), "1m filter")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(check_1m_final, trader, s): s for s in symbols}
        for f in as_completed(futures):
            try:
                res = f.result()
                results.append(res)
                tracker.update(passed=res[3])
                print(tracker.get_stats(), end="", flush=True)
            except: tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20)
    return results


# ==========================================
# OUTPUT FORMATTER
# ==========================================

def format_golden_block(golden: Dict, W: int = 74):
    gs     = golden.get("golden_score", 0.0)
    estate = golden.get("energy_state", "N/A")
    sp     = golden.get("spike_prob", 0.0)
    pa     = golden.get("phase_aligned", False)
    nm     = golden.get("near_min", False)
    er     = golden.get("energy_ratio", 1.0)
    pos    = golden.get("pos_in_range", 0.5)
    ratios = golden.get("ratios", [])
    wave_near_bottom = golden.get("wave_near_bottom", False)
    turning_up       = golden.get("turning_up", False)
    est_bars_to_pump = golden.get("est_bars_to_pump", 0)
    phase_pos        = golden.get("phase_pos", 0.5)
    estate_icon = {"COMPRESSION": "🔵", "BUILDING": "🟡", "EQUILIBRIUM": "⚪",
                   "EXPANSION": "🟠", "PEAK": "🔴"}.get(estate, "⚪")
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
    print("─" * W)
    print("  🌊  SINUSOIDAL DIP TIMING (STRICT EXTREMA CHECK)")
    print("─" * W)
    print(f"  Wave Near Bottom : {'✅ YES — Confirmed Lowest Extrema' if wave_near_bottom else '❌ NO'}")
    print(f"  Turning Up       : {'✅ YES — Macro Up-Cycle Initiated' if turning_up else '❌ NO'}")
    print(f"  Est. Bars to Pump: ~{est_bars_to_pump} bars (1m)")
    print(f"  Phase Position   : {phase_pos*100:.1f}%")
    if ratios:
        ratio_str = "  ".join(f"{r:.3f}" for r in ratios[:6])
        phi_hits  = sum(1 for r in ratios
                        if abs(r - PHI) < 0.18 or abs(r - PHI_SQ) < 0.18
                        or abs(r - PHI_INV) < 0.18)
        print(f"  FFT Ratios       : {ratio_str}")
        print(f"  φ-ratio hits     : {phi_hits}/{len(ratios)}  "
              f"(φ≈{PHI:.3f}  φ²≈{PHI_SQ:.3f}  1/φ≈{PHI_INV:.3f})")
    states = ["COMPRESSION", "BUILDING", "EQUILIBRIUM", "EXPANSION", "PEAK"]
    bar    = "  Flow: "
    for s in states:
        bar += f"[{s[:4]}]→" if s == estate else f" {s[:4]} →"
    print(f"{bar[:-1]}\n")

    # ── φ-Phase Reversal Block ──────────────────────────────────────────
    phi_phase   = golden.get("phi_phase", None)
    phi_zone    = golden.get("phi_phase_zone", "NEUTRAL")
    phi_cluster = golden.get("phi_cluster_score", 1.0)
    phi_fwd     = golden.get("phi_fwd_bias", 0.0)
    phi_ready   = golden.get("phi_reversal_ready", False)
    phi_highs   = golden.get("phi_swing_highs", 0)
    phi_lows    = golden.get("phi_swing_lows", 0)
    if phi_phase is not None:
        zone_icon = {"REVERSAL_LOW": "🟢", "REVERSAL_HIGH": "🔴",
                     "HARMONIC": "🔵", "BIAS_BULL": "🟡",
                     "BIAS_BEAR": "🟠", "NEUTRAL": "⚪"}.get(phi_zone, "⚪")
        print("─" * W)
        print("  φ-PHASE REVERSAL MAP  (bar-index → golden-ratio phase)")
        print("─" * W)
        print(f"  Current φ-Phase  : {phi_phase:.4f}  (θ={phi_phase*360:.1f}°)")
        print(f"  Phase Zone       : {zone_icon} {phi_zone}")
        print(f"  Harmonic Prox    : {'✅ Near φ-node' if golden.get('phi_near_harmonic') else '❌ Between nodes'}")
        print(f"  Fwd Return Bias  : {phi_fwd*100:+.3f}%  (20-bar empirical avg for this phase bin)")
        print(f"  Cluster Score    : {phi_cluster:.2f}x  ({'non-random' if phi_cluster > 2 else 'near random baseline'})")
        print(f"  Swing Pts Found  : {phi_highs} highs  /  {phi_lows} lows")
        print(f"  Reversal Ready   : {'✅ YES — φ-harmonic + bullish bias' if phi_ready else '❌ NO'}\n")

    # ── Stationarity + Cyclic Line Block ───────────────────────────────
    stat_ok   = golden.get("stat_is_stationary", None)
    hurst     = golden.get("stat_hurst", None)
    fit_q     = golden.get("cyc_fit_quality", None)
    r_sq      = golden.get("cyc_r_squared", None)
    rev_score = golden.get("reversal_score", None)
    at_cyc_lo = golden.get("cyc_at_cyclic_low", False)
    fcast_lo  = golden.get("price_forecast_low", None)
    fcast_hi  = golden.get("price_forecast_high", None)
    detrended = golden.get("detrended", False)
    dist_mid  = golden.get("cyc_dist_midline", None)
    if stat_ok is not None:
        stat_icon = "✅" if stat_ok else ("🟡" if golden.get("stat_mean_reverting") else "🔴")
        hurst_lbl = ("mean-reverting" if hurst < 0.45 else
                     ("random walk" if hurst < 0.55 else "trending"))
        print("─" * W)
        print("  STATIONARITY + CYCLIC LINE  (ADF + Hurst + sinusoidal OLS fit)")
        print("─" * W)
        print(f"  Stationary       : {stat_icon} {'YES' if stat_ok else 'NO'}  "
              f"(ADF p={golden.get('stat_adf_p', 1.0):.3f})")
        print(f"  Hurst Exponent   : {hurst:.3f}  ({hurst_lbl})")
        if detrended:
            print(f"  Series Work      : ⚠️  First-differenced (I(1) trend removed)")
        print(f"  Cyclic Fit       : {fit_q}  (R²={r_sq:.3f})")
        print(f"  Dist from Midline: {dist_mid:+.3f}σ  "
              f"({'✅ AT CYCLIC LOW' if at_cyc_lo else 'above midline'})")
        if fcast_lo is not None:
            print(f"  Cycle Forecast   : LOW={fcast_lo:.8f}  HIGH={fcast_hi:.8f}")
        print(f"  Reversal Score   : {rev_score:.3f}\n")

    # Frequency block (unchanged)
    freq = golden.get("freq")
    if freq:
        print_frequency_block(freq, W)


def format_sr_output(symbol, sr, current_price, cmo_val, vratio,
                      bull_ratio, ml_prob, tf_volumes, golden: Dict = None):
    vb, avg_r = sr['vol_bias'], sr['avg_range']
    bp_pct    = vb * 100
    bias_lbl  = ("🟢 BULLISH" if vb > 0.55 else
                 ("🔴 BEARISH" if vb < 0.45 else "⚪ NEUTRAL"))
    W         = 74
    print("\n" + "=" * W)
    print(f"  ★  STRUCTURAL RANGE S/R  —  {symbol}  ★")
    print(f"  (argmin/argmax anchored · multi-lookback · volume exhaustion · φ-harmonics · wavelet)")
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
        bull_len = int(vd['bull_pct'] / 100 * 30)
        bar      = "🟢" * bull_len + "🔴" * (bar_len - bull_len)
        print(f"  {tf:>4s}  [{bar}]  Bull: {vd['bull_pct']:.1f}%  Bear: {vd['bear_pct']:.1f}%")
    if golden:
        format_golden_block(golden, W)

    # ── Quadratic Area-Model Forecast ────────────────────────────────
    qf = golden.get("quad_forecast") if golden else None
    if qf:
        format_quadratic_block(qf, W)

    all_signals = []
    if golden:
        if golden.get("phase_aligned"):
            all_signals.append(f"φ Phase Aligned (score={golden['golden_score']*100:.0f}%, state={golden['energy_state']})")
        if golden.get("near_min") and golden.get("energy_state") == "COMPRESSION":
            all_signals.append("φ COMPRESSION at cycle minimum → bounce setup")
        if golden.get("spike_prob", 0) > 0.65:
            all_signals.append(f"φ Golden spike prob ({golden['spike_prob']*100:.0f}%)")
        if golden.get("wave_near_bottom") and golden.get("turning_up"):
            all_signals.append("🌊 LOWEST EXTREMA CONFIRMED & UP CYCLE PUMP INCOMING")
        if golden.get("est_bars_to_pump", 0) < 15:
            all_signals.append(f"🌊 Estimated <{golden['est_bars_to_pump']} bars to pump")

        # φ-Phase reversal signals
        phi_zone  = golden.get("phi_phase_zone", "NEUTRAL")
        phi_ready = golden.get("phi_reversal_ready", False)
        phi_phase = golden.get("phi_phase", None)
        phi_fwd   = golden.get("phi_fwd_bias", 0.0)
        phi_clust = golden.get("phi_cluster_score", 1.0)
        if phi_ready:
            all_signals.append(
                f"φ-PHASE REVERSAL ZONE (phase={phi_phase:.3f}, zone={phi_zone}, "
                f"fwd_bias={phi_fwd*100:+.2f}%)")
        elif phi_zone in ("HARMONIC", "BIAS_BULL") and phi_fwd > 0.002:
            all_signals.append(
                f"φ-phase near harmonic node (zone={phi_zone}, bias={phi_fwd*100:+.2f}%)")
        if phi_clust > 2.5:
            all_signals.append(
                f"φ-phase reversal clustering detected ({phi_clust:.1f}x vs random)")

        # Stationarity + cyclic line signals
        if golden.get("cyc_at_cyclic_low"):
            all_signals.append(
                f"📐 AT CYCLIC LOW (dist={golden.get('cyc_dist_midline', 0):+.2f}σ, "
                f"R²={golden.get('cyc_r_squared', 0):.2f})")
        if golden.get("reversal_score", 0.0) > 0.55:
            all_signals.append(
                f"📐 Cyclic reversal score high ({golden['reversal_score']:.2f}, "
                f"fit={golden.get('cyc_fit_quality', 'N/A')})")
        if golden.get("stat_mean_reverting") and not golden.get("stat_trending"):
            all_signals.append(
                f"📐 Series mean-reverting (Hurst={golden.get('stat_hurst', 0.5):.3f})")

        # Quadratic forecast signals
        qf = golden.get("quad_forecast")
        if qf and qf.get("window_bars", 0) > 0:
            if qf.get("forecast_bias") == "TOP" and qf.get("best_top"):
                bt = qf["best_top"]
                all_signals.append(
                    f"📐 QUAD FORECAST TOP → {bt['price']:.8f}  "
                    f"({bt['dist_pct']:+.3f}%)  [{bt['method']}]  "
                    f"conf={bt['conf']*100:.0f}%")
            elif qf.get("forecast_bias") == "DIP" and qf.get("best_dip"):
                bd = qf["best_dip"]
                all_signals.append(
                    f"📐 QUAD FORECAST DIP  → {bd['price']:.8f}  "
                    f"({bd['dist_pct']:+.3f}%)  [{bd['method']}]  "
                    f"conf={bd['conf']*100:.0f}%")

        # Frequency alignment signals
        freq = golden.get("freq")
        if freq:
            if freq.get("cond_macro_dip"):
                all_signals.append("📡 4H macro dip confirmed (low-freq trend declining)")
            if freq.get("cond_swing_rev"):
                all_signals.append("📡 Swing reversal detected (mid-freq turning up)")
            if freq.get("cond_micro_spark"):
                all_signals.append("📡 Micro momentum spark (high-freq burst initiated)")
            if freq.get("cond_1m_trough"):
                all_signals.append(f"📡 1M cycle at TROUGH (phase={freq['1m'].get('cycle_phase_pct', 0):.0f}%)")
            if freq.get("strong_alignment"):
                all_signals.append(f"📡 STRONG FREQ ALIGNMENT {freq['alignment_count']}/5 layers confluent")

    for lb_data in sr['lookbacks']:
        lb   = lb_data['lookback']
        ext  = lb_data['extremes']
        exh_h = lb_data['exh_high']
        exh_l = lb_data['exh_low']
        rng_pct, pos, min_d = ext['range_pct'], ext['position'], lb_data['min_dist']
        print("\n" + "─" * W)
        print(f"  📐  LOOKBACK: {lb} BARS  ({lb} min)")
        print("─" * W)
        print(f"  Global High     : {ext['high']:.10f}  ({ext['high_age']} bars ago)")
        print(f"  Global Low      : {ext['low']:.10f}  ({ext['low_age']} bars ago)")
        print(f"  True Range      : {rng_pct:.3f}%")
        print(f"  More Recent     : {ext['mr_label']}")
        print(f"  Min Target Dist : {min_d:.3f}%")
        pos_pct = pos * 100
        blen    = 40
        bpos    = int(pos * 40)
        pbar    = "─" * bpos + "▲" + "─" * (blen - bpos - 1)
        pos_txt = ('near LOW' if pos < 0.25 else ('near HIGH' if pos > 0.75 else 'mid-range'))
        print(f"  Position        : [{pbar}]  {pos_pct:.1f}%  ({pos_txt})")
        phi_prox = golden_fib_proximity(current_price, ext['low'], ext['high'])
        print(f"  φ Nearest Level : {phi_prox['nearest']}  @ {phi_prox['level_price']:.10f}"
              f"  (dist {phi_prox['dist_pct']:.3f}%)")
        print(f"\n  🫁  Exhaustion at HIGH: ", end="")
        if   exh_h.get('exhaustion', 0) > 0.5:
            print(f"🔴 {exh_h['pattern']} ({exh_h['exhaustion']:.2f})\n     {exh_h['detail']}")
        elif exh_h.get('exhaustion', 0) > 0.2:
            print(f"🟡 {exh_h['pattern']} ({exh_h['exhaustion']:.2f})\n     {exh_h['detail']}")
        else:
            print(f"⚪ {exh_h.get('pattern', 'NONE')} ({exh_h.get('exhaustion', 0):.2f})\n     {exh_h.get('detail', '')}")
        print(f"  🫁  Exhaustion at LOW : ", end="")
        if exh_l.get('exhaustion', 0) > 0.5:
            print(f"🟢 {exh_l['pattern']} ({exh_l['exhaustion']:.2f})\n     {exh_l['detail']}")
            if pos < 0.5:
                all_signals.append(f"[{lb}] Selling exhaustion at low ({exh_l['pattern']})")
        elif exh_l.get('exhaustion', 0) > 0.2:
            print(f"🟡 {exh_l['pattern']} ({exh_l['exhaustion']:.2f})\n     {exh_l['detail']}")
        else:
            print(f"⚪ {exh_l.get('pattern', 'NONE')} ({exh_l.get('exhaustion', 0):.2f})\n     {exh_l.get('detail', '')}")
        if   ext['more_recent'] == 'ARGMIN':
            all_signals.append(f"[{lb}] ARGMIN more recent → recent floor")
        elif ext['more_recent'] == 'ARGMAX':
            all_signals.append(f"[{lb}] ARGMAX more recent → recent ceiling")
        grid = lb_data['grid']
        if grid:
            print(f"\n  📊  Fibonacci Grid (volume profile + cluster energy + φ levels)")
            print(f"  {'Level':<8} {'Price':>14} {'Dist%':>8} {'Bull%':>6} {'Tch':>4} "
                  f"{'Rej':>4} {'Exh':>5} {'ClSt':<12} {'ClSc':>5} {'Verdict':<10} {'St'}")
            print("  " + "─" * 88)
            for g in grid:
                st  = g.get('status', '?')
                cs  = g.get('cluster_state', '—')
                csc = g.get('cluster_score', 0.0)
                vs  = g.get('valid_spike', False)
                cs_icon = ("💣" if cs == "UNSTABLE" else
                           ("🔥" if cs == "COMPRESSION" else
                            ("⚡" if cs == "BUILDING" else "·")))
                m   = ("·" if st == 'TOO_CLOSE' else
                       ("►" if g.get('direction') == 'UP' else "◄"))
                print(f"  {m}{g['label']:<7} {g['price']:>14.8f} "
                      f"{g['dist_pct']:>+7.3f}% {g['bull_pct']*100:>5.0f}% "
                      f"{g['touches']:>4} {g['total_rej']:>4} {g['exhaustion']:>4.2f} "
                      f"{cs_icon}{cs:<11} {csc:>5.2f} {g['verdict']:<10} "
                      f"{'✅SPIKE' if vs else st}")
        for direction, tgt_list, label_prefix in [
            ("UP",   lb_data['targets_up'],   "📈 RESISTANCE"),
            ("DOWN", lb_data['targets_down'], "📉 SUPPORT")
        ]:
            if tgt_list:
                print(f"\n  {label_prefix} TARGETS ({lb} bars)\n")
                for i, t in enumerate(tgt_list, 1):
                    bar    = "█" * int(t['score'] * 10) + "░" * (10 - int(t['score'] * 10))
                    vi     = ("🔴" if t['verdict'] == "RESISTANCE" else
                              ("🟢" if t['verdict'] == "SUPPORT" else "⚪"))
                    cs     = t.get('cluster_state', '—')
                    csc    = t.get('cluster_score', 0.0)
                    vs     = t.get('valid_spike', False)
                    trig   = t.get('trigger', False)
                    expl   = t.get('explosion', False)
                    spike_tag = ("  💣 EXPLOSION SETUP" if expl else
                                 ("  🔥 VALID SPIKE" if vs else
                                  ("  ⚡ TRIGGER" if trig else "")))
                    print(f"  {label_prefix[0]}{i}  {t['label']:5s}  {t['price']:.10f}  "
                          f"({t['dist_pct']:+.3f}%)  ETA: {t['eta']}{spike_tag}")
                    print(f"       [{bar}] {t['score']:.2f}  {vi} {t['verdict']}  "
                          f"BullVol: {t['bull_pct']*100:.0f}%  Tch: {t['touches']}  "
                          f"Rej: {t['rejections']}  RejInt: {t['rej_intensity']:.2f}")
                    print(f"       ClusterState: {cs:<12}  ClusterScore: {csc:.3f}")
                    d, parts = t.get('details', {}), []
                    if d.get('exhaustion', 0) > 0.2:       parts.append(f"Exh:{d['exhaustion']:.2f}")
                    if d.get('rejection_candles', 0) > 0.2: parts.append(f"Rej:{d['rejection_candles']:.2f}")
                    if d.get('extreme_prox', 0) > 0.3:      parts.append(f"NearExt")
                    if parts: print(f"             + {' | '.join(parts)}")
                    print()
            else:
                print(f"\n  {label_prefix} targets beyond {min_d:.3f}% minimum not found.\n")

    print("=" * W)
    print("  ⚡  CONSOLIDATED TRADE BIAS")
    print("=" * W)
    argmin_count = sum(1 for lb in sr['lookbacks']
                       if lb['extremes']['more_recent'] == 'ARGMIN')
    argmax_count = sum(1 for lb in sr['lookbacks']
                       if lb['extremes']['more_recent'] == 'ARGMAX')
    total_lb = len(sr['lookbacks'])
    print(f"\n  Recency Across Lookbacks:\n"
          f"    ARGMIN more recent : {argmin_count}/{total_lb}\n"
          f"    ARGMAX more recent : {argmax_count}/{total_lb}")
    if argmin_count > argmax_count:
        all_signals.append(f"ARGMIN dominant across lookbacks ({argmin_count}/{total_lb})")
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
                if t.get('explosion'): explosion_found = True
                if t.get('valid_spike'): spike_found   = True
    if explosion_found:
        all_signals.append(f"💣 EXPLOSION SETUP: COMPRESSION→UNSTABLE transition detected")
    elif spike_found:
        all_signals.append(f"🔥 VALID SPIKE zone (cluster state={best_cluster_state}, score={best_cluster_score:.2f})")
    elif best_cluster_state in ("COMPRESSION", "UNSTABLE") and best_cluster_score > 1.0:
        all_signals.append(f"⚡ Cluster energy building ({best_cluster_state}, score={best_cluster_score:.2f})")
    if vb > 0.55:          all_signals.append(f"1m Bullish vol bias ({vb*100:.0f}%)")
    if cmo_val < -50:       all_signals.append(f"CMO oversold ({cmo_val:.0f})")
    if bull_ratio > 0.65:  all_signals.append(f"Bull rejection vol ({bull_ratio*100:.0f}%)")
    if ml_prob > 0.65:     all_signals.append(f"ML spike prob ({ml_prob*100:.0f}%)")
    cluster_prob    = min(best_cluster_score / 3.0, 1.0)
    trigger_bonus   = (1 if explosion_found else (0.5 if spike_found else 0))
    golden_contrib  = golden.get("spike_prob", 0.0) if golden else 0.0
    sinusoidal_contrib = 0.0
    if golden:
        if golden.get("wave_near_bottom") and golden.get("turning_up"):
            sinusoidal_contrib = 0.8
        elif golden.get("wave_near_bottom"):
            sinusoidal_contrib = 0.5
        if golden.get("est_bars_to_pump", 0) < 15:
            sinusoidal_contrib = min(sinusoidal_contrib + 0.2, 1.0)

    # Frequency contribution to final prob
    freq_contrib = 0.0
    freq_result  = golden.get("freq") if golden else None
    if freq_result:
        freq_contrib = float(freq_result.get("alignment_score", 0.0))

    # φ-phase and cyclic contributions
    phi_contrib    = 0.0
    cyclic_contrib = 0.0
    if golden:
        phi_fwd_bias  = float(golden.get("phi_fwd_bias", 0.0))
        phi_ready     = golden.get("phi_reversal_ready", False)
        phi_contrib   = float(np.clip(phi_fwd_bias * 10.0 + (0.3 if phi_ready else 0.0), 0.0, 1.0))
        cyclic_contrib = float(golden.get("reversal_score", 0.0))

    # Rebalanced weights across all 7 components
    enhanced_prob = (
        0.22 * ml_prob +
        0.15 * cluster_prob +
        0.10 * trigger_bonus +
        0.12 * golden_contrib +
        0.15 * sinusoidal_contrib +
        0.08 * freq_contrib +
        0.10 * phi_contrib +
        0.08 * cyclic_contrib
    )
    print(f"\n  ⚡  Enhanced Spike Probability (φ + sinusoidal + wavelet + φ-phase + cyclic):")
    print(f"     ML Prob        : {ml_prob*100:.1f}%")
    print(f"     Cluster Prob   : {cluster_prob*100:.1f}%  "
          f"(best score={best_cluster_score:.2f}, state={best_cluster_state or 'N/A'})")
    print(f"     Trigger Bonus  : {'EXPLOSION' if explosion_found else ('SPIKE' if spike_found else 'none')}")
    if golden:
        print(f"     φ Golden Prob  : {golden_contrib*100:.1f}%  "
              f"(state={golden.get('energy_state','N/A')})")
        print(f"     Sinusoidal Prob: {sinusoidal_contrib*100:.1f}%  "
              f"(lowest extrema & up cycle: {golden.get('wave_near_bottom', False)})")
        print(f"     φ-Phase Contrib: {phi_contrib*100:.1f}%  "
              f"(zone={golden.get('phi_phase_zone','N/A')}, "
              f"bias={golden.get('phi_fwd_bias', 0.0)*100:+.2f}%)")
        print(f"     Cyclic Contrib : {cyclic_contrib*100:.1f}%  "
              f"(fit={golden.get('cyc_fit_quality','N/A')}, "
              f"H={golden.get('stat_hurst', 0.5):.3f})")
    if freq_result:
        print(f"     Freq Alignment : {freq_contrib*100:.1f}%  "
              f"({freq_result['alignment_count']}/5 layers — "
              f"4H/15M/5M/1M — "
              f"{'STRONG' if freq_result['strong_alignment'] else 'PARTIAL'})")
    print(f"     FINAL PROB     : {enhanced_prob*100:.1f}%")

    best_up = best_dn = None
    for lb in reversed(sr['lookbacks']):
        if not best_up and lb['targets_up']:   best_up = lb['targets_up'][0]
        if not best_dn and lb['targets_down']: best_dn = lb['targets_down'][0]
        if best_up and best_dn: break
    if best_up and best_dn:
        rr = abs(best_up['dist_pct']) / max(abs(best_dn['dist_pct']), 0.0001)
        print(f"\n  Best Target : {best_up['label']:5s}  {best_up['price']:.10f}  "
              f"({best_up['dist_pct']:+.3f}%)  ETA: {best_up['eta']}")
        print(f"  Best Stop   : {best_dn['label']:5s}  {best_dn['price']:.10f}  "
              f"({best_dn['dist_pct']:+.3f}%)")
        print(f"  R:R         : {rr:.2f}x")
        if rr >= 1.5: all_signals.append(f"R:R favorable ({rr:.1f}x)")
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
    for s in all_signals: print(f"    ✅  {s}")
    print()
    if   ns >= 4: v = "✅  STRONG LONG  —  Multiple structural confirmations"
    elif ns >= 3: v = "✅  LONG  —  Good structural alignment"
    elif ns >= 2: v = "⏳  PROBABLE LONG  —  Awaiting final confirmation"
    elif ns >= 1: v = "⏳  WEAK SIGNAL  —  Insufficient confirmation"
    else:         v = "⚪  NEUTRAL  —  No clear structural bias"
    print(f"  VERDICT : {v}")
    print("\n" + "=" * W)
    print(f"  CURRENT PRICE     : {current_price:.8f} USDC")
    print(f"  ARGMIN vs ARGMAX  : {argmin_count}/{total_lb} lookbacks show recent floor")
    if golden:
        print(f"  Sinusoidal Timing : Extrema & Up Cycle: "
              f"{'YES' if golden.get('wave_near_bottom') else 'NO'}")
        print(f"  Est. Bars to Pump : ~{golden.get('est_bars_to_pump', 'N/A')} bars (1m)")
    if freq_result:
        dc = freq_result.get('dominant_cycle', 0)
        print(f"  Dominant Cycle    : ~{dc} bars on 1m  "
              f"(≈ {dc} min cycle length)")
        print(f"  Freq Alignment    : {freq_result['alignment_count']}/5 layers aligned")
    if golden:
        phi_z  = golden.get("phi_phase_zone", "NEUTRAL")
        phi_p  = golden.get("phi_phase", None)
        flo    = golden.get("price_forecast_low",  None)
        fhi    = golden.get("price_forecast_high", None)
        hurst  = golden.get("stat_hurst", None)
        fit_q  = golden.get("cyc_fit_quality", None)
        if phi_p is not None:
            print(f"  φ-Phase Zone      : {phi_z}  (phase={phi_p:.4f})")
        if flo is not None:
            print(f"  Cyclic Forecast   : LOW={flo:.8f}  HIGH={fhi:.8f}  "
                  f"(fit={fit_q}, H={hurst:.3f})")
    # Quadratic forecast summary
    qf = golden.get("quad_forecast") if golden else None
    if qf and qf.get("window_bars", 0) > 0:
        bias_q = qf.get("forecast_bias", "NEUTRAL")
        bt_q   = qf.get("best_top")
        bd_q   = qf.get("best_dip")
        dc_q   = qf.get("dom_cycle", 0)
        wb_q   = qf.get("window_bars", 0)
        print(f"  Quad Forecast     : bias={bias_q}  window={wb_q}bars  domCycle≈{dc_q}bars")
        if bt_q:
            print(f"  → TOP target      : {bt_q['price']:.8f}  "
                  f"({bt_q['dist_pct']:+.3f}%)  [{bt_q['method']}]  "
                  f"conf={bt_q['conf']*100:.0f}%")
        if bd_q:
            print(f"  → DIP target      : {bd_q['price']:.8f}  "
                  f"({bd_q['dist_pct']:+.3f}%)  [{bd_q['method']}]  "
                  f"conf={bd_q['conf']*100:.0f}%")
    print(f"  Expected Move     : Strong reversal spike expected (φ-compression + exhaustion + wavelet + cyclic)")
    print("=" * W + "\n")
    return v, ns


# ==========================================
# SCAN HEADER
# ==========================================

def print_scan_header(scan_count: int):
    W         = 78
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "╔" + "═" * W + "╗")
    print("║" + " " * W + "║")
    print("║" + f"  🚀  SCAN #{scan_count} STARTED  —  {timestamp}".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("║" + "  🔍  CASCADE: 1D→4H→2H→15M→5M(REG)→1M(REG+SINE+WAVELET+φ-PHASE+CYCLIC+QUAD)".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("╚" + "═" * W + "╝")


# ==========================================
# MAIN SCAN LOOP
# ==========================================

def main():
    CREDENTIALS_FILE   = "credentials.txt"
    MIN_SIGNALS_REQUIRED = 3
    max_scans          = 10

    print("=" * 78)
    print("  🌊  STRICT MTF CASCADE DIP DETECTOR  (WAVELET + FFT + SINE + φ + φ-PHASE + CYCLIC + QUAD)  🌊")
    print("=" * 78)
    print()

    try:
        trader = Trader(CREDENTIALS_FILE)
    except Exception as e:
        print(f"❌ Failed to initialize trader: {e}")
        return

    scan_count = 0

    # Track the deepest MTF survivors seen across all scans.
    # "Deepest" = passed the most cascade stages (1D+4H+2H > 1D+4H > 1D).
    # When any scan stalls, we score whatever survived and deliver immediately.
    _best_pool: List[str] = []
    _best_pool_label: str = ""

    def _deliver_cascade_best(pool: List[str], label: str) -> bool:
        """
        Score `pool` with score_15m_candidate (full MTF scoring, no regression
        gate required), pick the best by composite score, and print the full
        S/R + golden + freq analysis.  Returns True if a candidate was delivered.
        """
        if not pool:
            return False
        W = 78
        print("\n" + "╔" + "═" * W + "╗")
        print("║" + " " * W + "║")
        print("║" + f"  🎯  CASCADE STALL — DELIVERING BEST {label} CANDIDATE".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("║" + f"  {len(pool)} pair(s) confirmed at {label} dip level.".ljust(W) + "║")
        print("║" + "  Scoring all for reversal potential + spike speed...".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("╚" + "═" * W + "╝")

        results = []
        tracker = ProgressTracker(len(pool), f"{label} scoring")
        with ThreadPoolExecutor(max_workers=15) as ex:
            futures = {ex.submit(score_15m_candidate, trader, s): s for s in pool}
            for f in as_completed(futures):
                try:
                    res = f.result()
                    if res[2]:
                        results.append(res)
                    tracker.update(passed=bool(res[2]))
                    print(tracker.get_stats(), end="", flush=True)
                except Exception:
                    tracker.update()
        print(f"\r{tracker.get_stats()}" + " " * 20)

        if not results:
            return False

        # Sort by composite reversal score (already computed inside score_15m_candidate)
        results.sort(key=lambda r: r[1], reverse=True)

        # Show ranked shortlist
        print(f"\n  📊  Ranked candidates ({label}):")
        for i, (sym, sc, _) in enumerate(results[:5], 1):
            print(f"     #{i}  {sym:<20}  score={sc:.4f}")

        best_sym, best_score, best_combined = results[0]
        cmo_val    = best_combined.get('_cmo_val', 0.0)
        vratio     = best_combined.get('_vratio', 0.0)
        bull_ratio = best_combined.get('_bull_ratio', 0.0)
        ml_prob    = best_combined.get('_ml_prob', 0.0)
        best_combined['_is_cascade_fallback']   = True
        best_combined['_cascade_label']         = label
        best_combined['_cascade_score']         = best_score
        best_combined['_cascade_pool_size']     = len(results)

        print("\n" + "╔" + "═" * W + "╗")
        print("║" + " " * W + "║")
        print("║" + f"  🥇  BEST {label} CANDIDATE SELECTED".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("║" + f"  Asset      : {best_sym}  (rank 1/{len(results)})".ljust(W) + "║")
        print("║" + f"  MTF Score  : {best_score:.4f}".ljust(W) + "║")
        print("║" + f"  1m CMO     : {cmo_val:+.2f}   ML Prob: {ml_prob*100:.1f}%   Bull Rej: {bull_ratio*100:.1f}%".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("║" + f"  ✅  {label} dip confirmed on all checked timeframes.".ljust(W) + "║")
        print("║" + "  ⚠️  Lower TF gates not met — use moderate stops.".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("╚" + "═" * W + "╝")

        klines_1m = trader.get_klines(best_sym, '1m', limit=1200, return_raw=True)
        if not klines_1m:
            print(f"❌ Could not fetch 1m klines for {best_sym}. Exiting...")
            return False

        current_price = float(klines_1m[-1][4])
        sr = get_sr_targets(klines_1m, current_price)
        tf_volumes = {
            '5m':  get_volume_breakdown(trader, best_sym, '5m'),
            '15m': get_volume_breakdown(trader, best_sym, '15m'),
            '4h':  get_volume_breakdown(trader, best_sym, '4h'),
            '1d':  get_volume_breakdown(trader, best_sym, '1d'),
        }
        verdict, signal_count = format_sr_output(
            best_sym, sr, current_price, cmo_val, vratio,
            bull_ratio, ml_prob, tf_volumes, best_combined
        )
        print("\n" + "╔" + "═" * W + "╗")
        print("║" + " " * W + "║")
        print("║" + f"  🎯  {label} BEST-DIP DELIVERED — BOT STOPPING".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("║" + f"  Asset    : {best_sym}".ljust(W) + "║")
        print("║" + f"  Price    : {current_price:.8f} USDC".ljust(W) + "║")
        print("║" + f"  Signals  : {signal_count}  ({label} — lower TFs not required)".ljust(W) + "║")
        print("║" + f"  Verdict  : {verdict}".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("║" + "  🎯  Best reversal+spike potential among confirmed dips.".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("╚" + "═" * W + "╝")
        print(f"\n✅ Bot completed ({label} cascade-best mode). Exiting...")
        return True

    while scan_count < max_scans:
        scan_count += 1
        print_scan_header(scan_count)

        try:
            symbols = trader.get_usdc_pairs()
            if not symbols:
                print("❌ No USDC pairs found. Retrying...")
                time.sleep(10)
                continue

            daily_passed = run_tf_filter(trader, symbols, '1d', max_workers=20)
            if not daily_passed:
                print("❌ No pairs passed 1d dip filter. Retrying...")
                time.sleep(5)
                continue

            four_h_passed = run_tf_filter(trader, daily_passed, '4h', max_workers=20)
            if not four_h_passed:
                # Only 1D dips survived — score and deliver immediately
                print("⚠️  4H gate killed all candidates — delivering best 1D dip now.")
                if _deliver_cascade_best(daily_passed, "1D"):
                    return
                time.sleep(5)
                continue

            two_h_passed = run_tf_filter(trader, four_h_passed, '2h', max_workers=20)
            if not two_h_passed:
                # 1D+4H confirmed — score and deliver
                print("⚠️  2H gate killed all candidates — delivering best 1D+4H dip now.")
                if _deliver_cascade_best(four_h_passed, "1D+4H"):
                    return
                time.sleep(5)
                continue

            fifteen_m_passed = run_tf_filter(trader, two_h_passed, '15m', max_workers=20)
            if not fifteen_m_passed:
                # 1D+4H+2H confirmed — score and deliver
                print("⚠️  15M gate killed all candidates — delivering best 1D+4H+2H dip now.")
                if _deliver_cascade_best(two_h_passed, "1D+4H+2H"):
                    return
                time.sleep(5)
                continue

            five_m_passed = run_5m_regression_filter(trader, fifteen_m_passed, max_workers=20)
            if not five_m_passed:
                # 1D+4H+2H+15M confirmed — best MTF fallback (existing logic preserved)
                print("⚠️  No pairs below 5m lowest regression line — activating MTF fallback...")
                fallback = run_best_mtf_fallback(trader, fifteen_m_passed, max_workers=15)
                if not fallback:
                    # Still nothing — fall back to 1D+4H+2H pool
                    print("⚠️  MTF fallback empty — delivering best 1D+4H+2H dip...")
                    if _deliver_cascade_best(two_h_passed, "1D+4H+2H"):
                        return
                    time.sleep(5)
                    continue

                # Unpack fallback result and route straight to analysis
                f_symbol    = fallback[0]
                f_cmo       = fallback[1]
                f_vratio    = fallback[2]
                f_bull      = fallback[4]
                f_ml        = fallback[5]
                f_combined  = fallback[6]
                f_score     = f_combined.get('_fallback_score', 0.0)
                f_rank      = f_combined.get('_fallback_rank', '?')

                W = 78
                print("\n" + "╔" + "═" * W + "╗")
                print("║" + " " * W + "║")
                print("║" + "  ⚡  FALLBACK MODE — BEST MTF DIP (NO 5M REGRESSION)".ljust(W) + "║")
                print("║" + " " * W + "║")
                print("║" + f"  Candidate : {f_symbol}  (rank {f_rank})".ljust(W) + "║")
                print("║" + f"  MTF Score : {f_score:.4f}".ljust(W) + "║")
                print("║" + f"  CMO       : {f_cmo:+.2f}   ML Prob: {f_ml*100:.1f}%   Bull Rej: {f_bull*100:.1f}%".ljust(W) + "║")
                print("║" + " " * W + "║")
                print("║" + "  ⚠️  5m regression gate NOT met — treat as lower-confidence setup".ljust(W) + "║")
                print("║" + " " * W + "║")
                print("╚" + "═" * W + "╝")

                klines_1m = trader.get_klines(f_symbol, '1m', limit=1200, return_raw=True)
                if klines_1m:
                    current_price = float(klines_1m[-1][4])
                    sr = get_sr_targets(klines_1m, current_price)
                    tf_volumes = {
                        '5m':  get_volume_breakdown(trader, f_symbol, '5m'),
                        '15m': get_volume_breakdown(trader, f_symbol, '15m'),
                        '4h':  get_volume_breakdown(trader, f_symbol, '4h'),
                        '1d':  get_volume_breakdown(trader, f_symbol, '1d'),
                    }
                    verdict, signal_count = format_sr_output(
                        f_symbol, sr, current_price, f_cmo, f_vratio,
                        f_bull, f_ml, tf_volumes, f_combined
                    )
                    W = 78
                    print("\n" + "╔" + "═" * W + "╗")
                    print("║" + " " * W + "║")
                    print("║" + "  ⚡  FALLBACK CANDIDATE DELIVERED — BOT STOPPING".ljust(W) + "║")
                    print("║" + " " * W + "║")
                    print("║" + f"  Asset   : {f_symbol}".ljust(W) + "║")
                    print("║" + f"  Price   : {current_price:.8f} USDC".ljust(W) + "║")
                    print("║" + f"  Signals : {signal_count}  (fallback — 5m regression skipped)".ljust(W) + "║")
                    print("║" + f"  Verdict : {verdict}".ljust(W) + "║")
                    print("║" + " " * W + "║")
                    print("║" + "  ⚠️  Use wider stops — regression gate not confirmed.".ljust(W) + "║")
                    print("║" + " " * W + "║")
                    print("╚" + "═" * W + "╝")
                    print("\n✅ Bot completed (fallback mode). Exiting...")
                    return
                else:
                    print(f"❌ Could not fetch 1m klines for fallback candidate {f_symbol}. Retrying...")
                    time.sleep(3)
                    continue

            results_1m = run_1m_filter(trader, five_m_passed, max_workers=15)
            if not results_1m:
                print("⚠️  1m filter empty — delivering best 1D+4H+2H+15M dip...")
                if _deliver_cascade_best(fifteen_m_passed, "1D+4H+2H+15M"):
                    return
                time.sleep(5)
                continue

            def final_score(r):
                ml             = r[5]
                gs             = r[6].get("spike_prob", 0.0)
                near_bottom    = r[6].get("wave_near_bottom", False)
                turning_up     = r[6].get("turning_up", False)
                est_bars       = r[6].get("est_bars_to_pump", 0)
                freq           = r[6].get("freq") or {}
                freq_alignment = freq.get("alignment_score", 0.0)
                reversal_score  = r[6].get("reversal_score", 0.0)
                at_cyclic_low   = r[6].get("cyc_at_cyclic_low", False)
                phi_ready       = r[6].get("phi_reversal_ready", False)
                phi_fwd_bias    = float(r[6].get("phi_fwd_bias", 0.0))
                phi_zone        = r[6].get("phi_phase_zone", "NEUTRAL")
                score  = (ml * 0.28 + gs * 0.20 + freq_alignment * 0.12
                          + reversal_score * 0.12 + phi_fwd_bias * 5.0 * 0.05)
                if near_bottom:     score += 0.10
                if turning_up:      score += 0.08
                if at_cyclic_low:   score += 0.07
                if phi_ready:       score += 0.06
                if phi_zone in ("REVERSAL_LOW", "BIAS_BULL"): score += 0.04
                if est_bars < 15:   score += 0.05
                score -= r[1] * 0.0015
                return score

            results_1m.sort(key=final_score, reverse=True)

            top_candidate = results_1m[0]
            symbol    = top_candidate[0]
            cmo_val   = top_candidate[1]
            vratio    = top_candidate[2]
            bull_ratio = top_candidate[4]
            ml_prob   = top_candidate[5]
            golden    = top_candidate[6]

            print(f"\n🏆 TOP CANDIDATE: {symbol}")
            print(f"   Final Score: {final_score(top_candidate):.4f}")

            klines_1m = trader.get_klines(symbol, '1m', limit=1200, return_raw=True)

            if klines_1m:
                current_price = float(klines_1m[-1][4])
                sr = get_sr_targets(klines_1m, current_price)

                tf_volumes = {
                    '5m':  get_volume_breakdown(trader, symbol, '5m'),
                    '15m': get_volume_breakdown(trader, symbol, '15m'),
                    '4h':  get_volume_breakdown(trader, symbol, '4h'),
                    '1d':  get_volume_breakdown(trader, symbol, '1d'),
                }

                verdict, signal_count = format_sr_output(
                    symbol, sr, current_price, cmo_val, vratio,
                    bull_ratio, ml_prob, tf_volumes, golden
                )

                if signal_count >= MIN_SIGNALS_REQUIRED:
                    W = 78
                    print("\n" + "╔" + "═" * W + "╗")
                    print("║" + " " * W + "║")
                    print("║" + "  ✅  SETUP FOUND — BOT STOPPING".ljust(W) + "║")
                    print("║" + " " * W + "║")
                    print("║" + f"  Asset: {symbol}".ljust(W) + "║")
                    print("║" + f"  Price: {current_price:.8f} USDC".ljust(W) + "║")
                    print("║" + f"  Signals: {signal_count}".ljust(W) + "║")
                    print("║" + f"  Verdict: {verdict}".ljust(W) + "║")
                    print("║" + " " * W + "║")
                    print("║" + "  🎯  Review the analysis above and make your decision.".ljust(W) + "║")
                    print("║" + " " * W + "║")
                    print("╚" + "═" * W + "╝")
                    print("\n✅ Bot completed successfully. Exiting...")
                    return
                else:
                    print(f"\n⚠️  Found {signal_count} signals (need {MIN_SIGNALS_REQUIRED}). Continuing search...")
                    time.sleep(3)
            else:
                print(f"\n❌ Failed to get detailed analysis for {symbol}")
                time.sleep(3)

            gc.collect()

        except KeyboardInterrupt:
            print("\n\n⚠️  Scan interrupted by user. Exiting...")
            return
        except Exception as e:
            print(f"\n❌ Error in scan: {e}")
            time.sleep(5)

    # ── MAX SCANS EXHAUSTED — deliver best daily-dip candidate ────────
    W = 78
    print("\n" + "╔" + "═" * W + "╗")
    print("║" + " " * W + "║")
    print("║" + "  ⚠️   MAX SCANS REACHED — NO FULL MTF SETUP FOUND".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("║" + f"  Completed {scan_count} scans without finding".ljust(W) + "║")
    print("║" + f"  a setup with {MIN_SIGNALS_REQUIRED}+ signals.".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("║" + "  🌅  Activating DAILY BEST-DIP FALLBACK...".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("╚" + "═" * W + "╝")

    try:
        all_symbols = trader.get_usdc_pairs()
    except Exception as e:
        print(f"❌ Could not fetch symbol list for daily fallback: {e}")
        print("\n❌ Bot completed without finding setup. Exiting...")
        return

    daily_fallback = run_daily_best_fallback(trader, all_symbols, max_workers=15)

    if not daily_fallback:
        print("\n" + "╔" + "═" * W + "╗")
        print("║" + " " * W + "║")
        print("║" + "  ❌  DAILY FALLBACK FOUND NO CANDIDATES".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("║" + "  No symbols are currently at a confirmed 1D dip.".ljust(W) + "║")
        print("║" + "  Market conditions are unfavorable. Try again later.".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("╚" + "═" * W + "╝")
        print("\n❌ Bot completed without finding setup. Exiting...")
        return

    df_symbol  = daily_fallback[0]
    df_cmo     = daily_fallback[1]
    df_vratio  = daily_fallback[2]
    df_bull    = daily_fallback[4]
    df_ml      = daily_fallback[5]
    df_combined = daily_fallback[6]

    df_score       = df_combined.get('_daily_fallback_score', 0.0)
    df_rank        = df_combined.get('_daily_fallback_rank', '?')
    df_total_dips  = df_combined.get('_total_daily_dips', 0)
    df_cmo_1d      = df_combined.get('_cmo_1d', 0.0)
    df_pos_1d      = df_combined.get('_pos_1d', 0.5)

    print("\n" + "╔" + "═" * W + "╗")
    print("║" + " " * W + "║")
    print("║" + "  🌅  DAILY BEST-DIP CANDIDATE SELECTED".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("║" + f"  Candidate    : {df_symbol}  (rank {df_rank} of {df_total_dips} daily dips)".ljust(W) + "║")
    print("║" + f"  Daily Score  : {df_score:.4f}".ljust(W) + "║")
    print("║" + f"  1D CMO       : {df_cmo_1d:+.2f}  (oversold depth on daily TF)".ljust(W) + "║")
    print("║" + f"  1D Pos/Range : {df_pos_1d*100:.1f}%  (0%=at floor, 100%=at ceiling)".ljust(W) + "║")
    print("║" + f"  ML Prob      : {df_ml*100:.1f}%   Bull Rej: {df_bull*100:.1f}%".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("║" + "  ⚠️  Full MTF cascade not satisfied — daily dip only.".ljust(W) + "║")
    print("║" + "  ⚠️  Use wider stops. Best potential among current daily dips.".ljust(W) + "║")
    print("║" + " " * W + "║")
    print("╚" + "═" * W + "╝")

    klines_1m = trader.get_klines(df_symbol, '1m', limit=1200, return_raw=True)
    if klines_1m:
        current_price = float(klines_1m[-1][4])
        sr = get_sr_targets(klines_1m, current_price)
        tf_volumes = {
            '5m':  get_volume_breakdown(trader, df_symbol, '5m'),
            '15m': get_volume_breakdown(trader, df_symbol, '15m'),
            '4h':  get_volume_breakdown(trader, df_symbol, '4h'),
            '1d':  get_volume_breakdown(trader, df_symbol, '1d'),
        }
        verdict, signal_count = format_sr_output(
            df_symbol, sr, current_price, df_cmo, df_vratio,
            df_bull, df_ml, tf_volumes, df_combined
        )
        print("\n" + "╔" + "═" * W + "╗")
        print("║" + " " * W + "║")
        print("║" + "  🌅  DAILY BEST-DIP FALLBACK DELIVERED — BOT STOPPING".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("║" + f"  Asset    : {df_symbol}".ljust(W) + "║")
        print("║" + f"  Price    : {current_price:.8f} USDC".ljust(W) + "║")
        print("║" + f"  Signals  : {signal_count}  (daily-dip fallback — full MTF not confirmed)".ljust(W) + "║")
        print("║" + f"  1D CMO   : {df_cmo_1d:+.2f}   1D Position: {df_pos_1d*100:.1f}% of range".ljust(W) + "║")
        print("║" + f"  Verdict  : {verdict}".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("║" + "  ⚠️  Daily dip confirmed — micro/swing TF alignment not fully met.".ljust(W) + "║")
        print("║" + "  ⚠️  Best potential to pump fast. Use wider stops than normal.".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("╚" + "═" * W + "╝")
        print("\n✅ Bot completed (daily best-dip fallback). Exiting...")
    else:
        print(f"❌ Could not fetch 1m klines for daily fallback candidate {df_symbol}. Exiting...")
        print("\n❌ Bot completed without full analysis. Exiting...")


if __name__ == "__main__":
    main()