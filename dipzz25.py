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


# ══════════════════════════════════════════════════════════════════════════════
# HT_SINE + FFT FORECAST ENGINE  —  dipzz28
# ══════════════════════════════════════════════════════════════════════════════
#

# ══════════════════════════════════════════════════════════════════════════════
# HT_SINE + FFT FORECAST ENGINE  —  dipzz28
# ══════════════════════════════════════════════════════════════════════════════
#
# THREE MANDATORY HARD-GATE CONDITIONS (any failure → immediate reject):
#
#  CONDITION A — LOCAL DIP ABOVE GLOBAL FLOOR
#    The "local dip low" is defined as the lowest close within the most
#    recent dominant-cycle window, confirmed as a swing low (at least one
#    higher close follows it).  This is NOT current_close — it is the
#    actual bottom of the current local dip.
#    Gate:  local_dip_low > global_argmin_price
#    Why:   If the local dip undercuts the global floor, the asset is
#           printing new structural lows.  No floor exists.  Reject.
#
#  CONDITION B — ARGMIN MORE RECENT THAN ARGMAX (global, full window)
#    Gate:  argmin_idx > argmax_idx
#    Why:   The most recent extreme must be a low, not a high.  If the
#           most recent extreme is a high, price peaked after the last low —
#           we are buying into a downswing from peak.  Reject.
#
#  CONDITION C — FFT DOMINANT-CYCLE FORECAST POINTS UP
#    Gate:  fft_forecast_next_bar > current_close
#    Why:   If the top-5 dominant frequencies reconstructed from the full
#           1000-bar 1m series predict the next bar will be lower than the
#           current close, the cycle is still pointed down.  Reject.
#
# HT_SINE ANALYSIS (display + ranking, not a hard gate by itself)
#    ta.HT_SINE applied to the FULL close array (up to 1000 bars).
#    Values extracted at: global argmin bar, global argmax bar, current bar.
#    Midpoint price = (argmin_price + argmax_price) / 2.
#    ht_bull_signal = leadsine > sine AND sine < -0.5 AND cond_A AND cond_B.
#    This is the strongest possible HT_SINE confirmation — used to boost
#    the exhaustion resonance score when true.


def _find_local_dip_low(arr: np.ndarray, cycle_bars: int) -> tuple:
    """
    Find the most recent confirmed local dip low in the last `cycle_bars` bars.

    A confirmed local dip low requires:
      1. It is the minimum close within the lookback window.
      2. At least one bar AFTER it closes HIGHER (recovery has begun).

    If no confirmed swing low found (still falling), slides window back
    up to 3 times.  Falls back to global argmin if all attempts fail.

    Returns (local_dip_price: float, local_dip_idx: int)
    where local_dip_idx is the index in the FULL array.
    """
    n = len(arr)
    lookback = max(cycle_bars, 10)
    for attempt in range(4):
        start = max(0, n - lookback - attempt * (lookback // 2))
        window = arr[start:]
        if len(window) < 4:
            break
        local_min_pos = int(np.argmin(window))
        local_min_val = float(window[local_min_pos])
        # Check recovery: at least one bar after local min is higher
        post = window[local_min_pos + 1:]
        if len(post) > 0 and float(np.max(post)) > local_min_val:
            full_idx = start + local_min_pos
            return local_min_val, full_idx
    # Fallback: global argmin
    g_idx = int(np.argmin(arr))
    return float(arr[g_idx]), g_idx


def compute_ht_sine_extrema(closes: list, cycle_bars: int = 20) -> dict:
    """
    HT_SINE analysis anchored between the global argmin and argmax of the
    full close array.

    Extracts HT_SINE + HT_LEADSINE values at:
      - Global argmin bar  (lowest close = absolute floor of window)
      - Global argmax bar  (highest close = absolute ceiling of window)
      - Current bar        (last close)

    Also finds the most recent LOCAL DIP LOW using _find_local_dip_low
    with the dominant cycle length as the lookback window.

    The price MIDPOINT between global extrema is: (argmin_price + argmax_price) / 2.

    Condition A (price_above_floor) is checked on LOCAL DIP LOW vs GLOBAL FLOOR:
      local_dip_price > argmin_price
      → The most recent local dip did NOT undercut the absolute floor.

    ht_bull_signal = True when ALL FOUR hold simultaneously:
      1. leadsine[-1] > sine[-1]   (lead above sine: cycle is turning up)
      2. sine[-1] < -0.5           (currently in the lower half of the cycle)
      3. local_dip_price > argmin  (local dip above the global floor)
      4. argmin_idx > argmax_idx   (global floor established more recently than ceiling)
    """
    arr = np.asarray(closes, dtype=float)
    n   = len(arr)
    empty = {
        "sine": 0.0, "leadsine": 0.0,
        "sine_at_argmin": 0.0, "leadsine_at_argmin": 0.0,
        "sine_at_argmax": 0.0, "leadsine_at_argmax": 0.0,
        "argmin_price": 0.0, "argmax_price": 0.0,
        "argmin_idx": 0, "argmax_idx": 0,
        "midpoint_price": 0.0, "current_price": 0.0,
        "local_dip_price": 0.0, "local_dip_idx": 0,
        "price_above_floor": False,
        "argmin_more_recent": False,
        "leadsine_above_sine": False,
        "sine_in_lower_half": False,
        "ht_bull_signal": False,
        "ht_trough_proximity": 0.0,
        "n_bars": n,
    }
    if n < 32:
        return empty
    try:
        sine_arr, leadsine_arr = ta.HT_SINE(arr)
    except Exception:
        return empty

    sine_now  = float(sine_arr[-1])     if not np.isnan(sine_arr[-1])     else 0.0
    lead_now  = float(leadsine_arr[-1]) if not np.isnan(leadsine_arr[-1]) else 0.0

    # Global extrema
    argmin_idx   = int(np.argmin(arr))
    argmax_idx   = int(np.argmax(arr))
    argmin_price = float(arr[argmin_idx])
    argmax_price = float(arr[argmax_idx])
    midpoint     = (argmin_price + argmax_price) / 2.0
    current_price = float(arr[-1])

    # Most recent local dip low (lookback = one dominant cycle)
    local_dip_price, local_dip_idx = _find_local_dip_low(arr, cycle_bars)

    # HT_SINE values at global extrema bars
    def _safe(a, i): return float(a[i]) if not np.isnan(a[i]) else 0.0
    sine_at_min = _safe(sine_arr,     argmin_idx)
    lead_at_min = _safe(leadsine_arr, argmin_idx)
    sine_at_max = _safe(sine_arr,     argmax_idx)
    lead_at_max = _safe(leadsine_arr, argmax_idx)

    # Condition A: local dip must be STRICTLY above the global floor
    price_above_floor  = bool(local_dip_price > argmin_price)
    argmin_more_recent = bool(argmin_idx > argmax_idx)
    leadsine_above     = bool(lead_now > sine_now)
    sine_lower_half    = bool(sine_now < -0.5)

    # Trough proximity: 1.0 when sine = -1 (mathematical bottom)
    ht_trough_prox = float(np.clip((1.0 - sine_now) / 2.0, 0.0, 1.0))

    ht_bull = bool(leadsine_above and sine_lower_half and
                   price_above_floor and argmin_more_recent)

    return {
        "sine":                 sine_now,
        "leadsine":             lead_now,
        "sine_at_argmin":       sine_at_min,
        "leadsine_at_argmin":   lead_at_min,
        "sine_at_argmax":       sine_at_max,
        "leadsine_at_argmax":   lead_at_max,
        "argmin_price":         argmin_price,
        "argmax_price":         argmax_price,
        "argmin_idx":           argmin_idx,
        "argmax_idx":           argmax_idx,
        "midpoint_price":       midpoint,
        "current_price":        current_price,
        "local_dip_price":      local_dip_price,
        "local_dip_idx":        local_dip_idx,
        "price_above_floor":    price_above_floor,
        "argmin_more_recent":   argmin_more_recent,
        "leadsine_above_sine":  leadsine_above,
        "sine_in_lower_half":   sine_lower_half,
        "ht_bull_signal":       ht_bull,
        "ht_trough_proximity":  ht_trough_prox,
        "n_bars":               n,
    }


def compute_fft_forecast(closes: list, forecast_bars: int = 5,
                          top_n_freqs: int = 5) -> dict:
    """
    FFT price forecast: detrend → FFT → keep top-N frequencies by magnitude
    → extrapolate each frequency component forward → re-add trend.

    The immediate next-bar forecast price (forecast_next) must be ABOVE the
    current close for the gate to pass (Condition C).

    Returns:
      fft_forecast_above_close  bool   — forecast_next > current_close
      forecast_next             float  — predicted price of next 1m bar
      current_close             float  — last close used
      lowest_low                float  — min(closes) = global floor
      forecast_prices           list   — next `forecast_bars` predicted prices
      dominant_cycles           list   — period lengths of top-N components
      fft_slope_5bar            float  — % slope per bar over forecast window
    """
    arr = np.asarray(closes, dtype=float)
    n   = len(arr)
    empty = {
        "fft_forecast_above_close": False,
        "forecast_next": 0.0, "current_close": 0.0,
        "lowest_low":    0.0, "forecast_prices": [],
        "dominant_cycles": [], "fft_slope_5bar": 0.0,
    }
    if n < 32:
        return empty

    current_close = float(arr[-1])
    lowest_low    = float(np.min(arr))

    # Detrend
    t          = np.arange(n, dtype=float)
    trend_coef = np.polyfit(t, arr, 1)
    trend_line = np.polyval(trend_coef, t)
    detrended  = arr - trend_line

    # FFT — keep only cycles between 4 bars and n/2 bars
    fft_vals   = np.fft.rfft(detrended)
    freqs      = np.fft.rfftfreq(n)
    magnitudes = np.abs(fft_vals)
    valid      = np.where((freqs > 0) & (freqs <= 1.0 / 4))[0]
    if len(valid) == 0:
        return {**empty, "current_close": current_close, "lowest_low": lowest_low}

    top_idxs = valid[np.argsort(magnitudes[valid])[::-1]][:top_n_freqs]
    dominant_cycles = []
    for idx in top_idxs:
        if freqs[idx] > 0:
            dominant_cycles.append(int(round(1.0 / freqs[idx])))

    # Extrapolate each component forward
    t_future = np.arange(n, n + forecast_bars, dtype=float)
    forecast_osc = np.zeros(forecast_bars)
    for idx in top_idxs:
        freq  = float(freqs[idx])
        mag   = float(magnitudes[idx])
        phase = float(np.angle(fft_vals[idx]))
        forecast_osc += (2.0 / n) * mag * np.cos(
            2.0 * np.pi * freq * t_future + phase)

    forecast_prices_arr = forecast_osc + np.polyval(trend_coef, t_future)
    forecast_next  = float(forecast_prices_arr[0])
    forecast_prices = forecast_prices_arr.tolist()

    fft_above = bool(forecast_next > current_close)
    if forecast_bars >= 2:
        slope = float((forecast_prices_arr[-1] - forecast_prices_arr[0]) /
                       (forecast_bars - 1) / (current_close + 1e-12) * 100.0)
    else:
        slope = 0.0

    return {
        "fft_forecast_above_close": fft_above,
        "forecast_next":   forecast_next,
        "current_close":   current_close,
        "lowest_low":      lowest_low,
        "forecast_prices": forecast_prices,
        "dominant_cycles": dominant_cycles,
        "fft_slope_5bar":  slope,
    }


def compute_ht_fft_gate(raw_klines_1m: list) -> dict:
    """
    Master gate.  Uses the FULL 1m klines array (up to 1000 bars).

    THREE CONDITIONS — all must pass:
      A. local_dip_price > global_argmin_price   (local dip above global floor)
      B. argmin_idx > argmax_idx                  (minima more recent than maxima)
      C. fft_forecast_next > current_close        (dominant cycle pointing up)

    local_dip_price = lowest close of the most recent local dip,
    confirmed as a swing low (recovery bar exists after it).
    This is NOT current_close — it's the actual floor of the current local dip.
    """
    empty = {
        "gate_pass": False,
        "cond_A_local_dip_above_floor": False,
        "cond_B_argmin_recent":         False,
        "cond_C_fft_up":                False,
        "ht": {}, "fft": {},
        "resonance_extra": 0.0,
        "gate_reason": "insufficient data",
    }
    if not raw_klines_1m or len(raw_klines_1m) < 64:
        return empty

    closes = [float(k[4]) for k in raw_klines_1m]
    arr    = np.asarray(closes, dtype=float)

    # Estimate dominant cycle for local dip lookback
    try:
        n = len(arr)
        detrended = arr - np.polyval(np.polyfit(np.arange(n), arr, 1), np.arange(n))
        fft_v  = np.abs(np.fft.rfft(detrended))
        fft_f  = np.fft.rfftfreq(n)
        valid  = np.where((fft_f > 0) & (fft_f <= 0.25))[0]
        if len(valid) > 0:
            peak     = valid[int(np.argmax(fft_v[valid]))]
            cycle_bars = int(np.clip(round(1.0 / fft_f[peak]), 6, n // 4))
        else:
            cycle_bars = 20
    except Exception:
        cycle_bars = 20

    ht  = compute_ht_sine_extrema(closes, cycle_bars=cycle_bars)
    fft = compute_fft_forecast(closes, forecast_bars=5, top_n_freqs=5)

    cond_A = bool(ht["price_above_floor"])
    cond_B = bool(ht["argmin_more_recent"])
    cond_C = bool(fft["fft_forecast_above_close"])

    gate_pass = cond_A and cond_B and cond_C

    reasons = []
    if not cond_A:
        reasons.append(
            f"local_dip({ht['local_dip_price']:.8f}) "
            f"<= global_floor({ht['argmin_price']:.8f}) "
            f"— new structural low, no floor")
    if not cond_B:
        reasons.append(
            f"argmax({ht['argmax_idx']}) more recent than "
            f"argmin({ht['argmin_idx']}) — still falling from peak")
    if not cond_C:
        reasons.append(
            f"FFT forecast({fft['forecast_next']:.8f}) "
            f"<= close({fft['current_close']:.8f}) — cycle pointing down")
    gate_reason = "ALL PASS" if gate_pass else " | ".join(reasons)

    # Resonance bonus: how strongly do the conditions pass?
    conds_met   = sum([cond_A, cond_B, cond_C])
    ht_bonus    = 0.15 if ht.get("ht_bull_signal") else 0.0
    slope_bonus = float(np.clip(fft.get("fft_slope_5bar", 0.0) / 2.0, 0.0, 0.15))
    # Margin bonus for condition A: how far above the floor is the local dip?
    if cond_A and ht["argmin_price"] > 0:
        margin_pct   = (ht["local_dip_price"] - ht["argmin_price"]) / ht["argmin_price"] * 100.0
        margin_bonus = float(np.clip(margin_pct / 1.0, 0.0, 0.10))
    else:
        margin_bonus = 0.0

    resonance_extra = float(np.clip(
        conds_met / 3.0 * 0.60 + ht_bonus + slope_bonus + margin_bonus,
        0.0, 1.0
    ))

    return {
        "gate_pass":                    gate_pass,
        "cond_A_local_dip_above_floor": cond_A,
        "cond_B_argmin_recent":         cond_B,
        "cond_C_fft_up":                cond_C,
        "ht":                           ht,
        "fft":                          fft,
        "resonance_extra":              resonance_extra,
        "gate_reason":                  gate_reason,
        "cycle_bars_used":              cycle_bars,
    }


def format_ht_fft_block(gate: dict, W: int = 74):
    """
    Print the full HT_SINE + FFT gate block inside format_sr_output.
    """
    if not gate:
        return
    print("─" * W)
    print("  📡  HT_SINE + FFT FORECAST GATE  (3 MANDATORY HARD CONDITIONS)")
    print("─" * W)

    def yn(v): return "✅ PASS" if v else "❌ FAIL"

    gp  = gate.get("gate_pass", False)
    cA  = gate.get("cond_A_local_dip_above_floor", False)
    cB  = gate.get("cond_B_argmin_recent",         False)
    cC  = gate.get("cond_C_fft_up",                False)
    ht  = gate.get("ht",  {})
    fft = gate.get("fft", {})
    gr  = gate.get("gate_reason", "")
    cyc = gate.get("cycle_bars_used", 0)

    icon = "✅ GATE OPEN — all 3 conditions met" if gp else "❌ GATE CLOSED — asset rejected"
    print(f"\n  {icon}")
    print(f"  Cycle used for local-dip lookback: {cyc} bars")
    print(f"  Reason: {gr}\n")

    # A: local dip above global floor
    ldp = ht.get("local_dip_price", 0.0)
    llf = ht.get("argmin_price",    0.0)
    lid = ht.get("local_dip_idx",   0)
    n   = ht.get("n_bars",          0)
    cp  = ht.get("current_price",   0.0)
    gap = (ldp - llf) / (llf + 1e-12) * 100.0 if llf > 0 else 0.0
    print(f"  [A] Local dip above global floor   : {yn(cA)}")
    print(f"      local_dip_price = {ldp:.8f}  (bar {lid}, {n-lid} bars ago)")
    print(f"      global_floor    = {llf:.8f}  (argmin of full {n}-bar window)")
    print(f"      margin          = {gap:+.4f}%   current_close = {cp:.8f}")

    # B: argmin more recent
    ami = ht.get("argmin_idx", 0)
    axi = ht.get("argmax_idx", 0)
    print(f"  [B] Argmin more recent than argmax : {yn(cB)}")
    print(f"      argmin bar={ami} ({n-ami} bars ago)  "
          f"argmax bar={axi} ({n-axi} bars ago)")

    # C: FFT forecast
    fn  = fft.get("forecast_next",  0.0)
    cc  = fft.get("current_close",  0.0)
    fd  = (fn - cc) / (cc + 1e-12) * 100.0 if cc > 0 else 0.0
    sl  = fft.get("fft_slope_5bar", 0.0)
    dc  = fft.get("dominant_cycles", [])
    print(f"  [C] FFT forecast next bar > close  : {yn(cC)}")
    print(f"      forecast_next = {fn:.8f}  current = {cc:.8f}  delta = {fd:+.5f}%")
    print(f"      5-bar slope = {sl:+.4f}%/bar   dominant cycles = {dc}")

    # HT_SINE extrema table
    print(f"\n  📊  HT_SINE between extrema (full {n}-bar 1m window)")
    aln_p = ht.get("argmin_price",     0.0)
    axn_p = ht.get("argmax_price",     0.0)
    mid_p = ht.get("midpoint_price",   0.0)
    s_min = ht.get("sine_at_argmin",   0.0)
    l_min = ht.get("leadsine_at_argmin", 0.0)
    s_max = ht.get("sine_at_argmax",   0.0)
    l_max = ht.get("leadsine_at_argmax", 0.0)
    s_now = ht.get("sine",             0.0)
    l_now = ht.get("leadsine",         0.0)
    print(f"  {'Point':<14} {'Price':>14} {'HT_SINE':>10} {'LEADSINE':>10}  Note")
    print("  " + "─" * 54)
    print(f"  {'ARGMIN (floor)':<14} {aln_p:>14.8f} {s_min:>10.4f} {l_min:>10.4f}  ← global lowest low")
    print(f"  {'MIDPOINT':<14} {mid_p:>14.8f} {'—':>10} {'—':>10}  ← (low+high)/2")
    print(f"  {'LOCAL DIP':<14} {ldp:>14.8f} {'—':>10} {'—':>10}  ← most recent local dip")
    print(f"  {'ARGMAX (ceil)':<14} {axn_p:>14.8f} {s_max:>10.4f} {l_max:>10.4f}  ← global highest high")
    print(f"  {'CURRENT':<14} {cc:>14.8f} {s_now:>10.4f} {l_now:>10.4f}  ← now")

    # HT_SINE bull signal breakdown
    ht_bs = ht.get("ht_bull_signal",     False)
    la    = ht.get("leadsine_above_sine", False)
    sl_h  = ht.get("sine_in_lower_half",  False)
    paf   = ht.get("price_above_floor",   False)
    amr   = ht.get("argmin_more_recent",  False)
    ht_tp = ht.get("ht_trough_proximity", 0.0)
    print(f"\n  HT_SINE Bull Signal : {'✅ YES — full 4-condition confirmation' if ht_bs else '❌ NO'}")
    print(f"    leadsine > sine    : {'✅' if la   else '❌'}  (cycle turning up from trough)")
    print(f"    sine < -0.5        : {'✅' if sl_h else '❌'}  (in lower half, proximity={ht_tp*100:.0f}%)")
    print(f"    local dip > floor  : {'✅' if paf  else '❌'}  (floor intact)")
    print(f"    argmin more recent : {'✅' if amr  else '❌'}  (low was the most recent extreme)")

    # FFT forecast line
    fp = fft.get("forecast_prices", [])
    if fp:
        print(f"\n  FFT 5-bar price forecast : {' → '.join(f'{p:.6f}' for p in fp[:5])}")

    re  = gate.get("resonance_extra", 0.0)
    bar = "█" * int(re * 20) + "░" * (20 - int(re * 20))
    print(f"  Gate resonance bonus     : [{bar}]  {re*100:.0f}%")
    print()

def _sma_cross_recent(closes: list, fast_p: int, slow_p: int,
                       lookback: int = 3) -> dict:
    """
    Check if fast SMA crossed above slow SMA within the last `lookback` bars.

    Returns:
      crossed   – bool: cross happened within lookback bars
      imminent  – bool: fast is within 0.1% of slow (cross very close)
      fast_now  – current fast SMA value
      slow_now  – current slow SMA value
      gap_pct   – fast-slow gap as % of price (positive = fast above slow)
    """
    arr = np.asarray(closes, dtype=float)
    min_len = max(slow_p + lookback + 2, 30)
    if len(arr) < min_len:
        return {"crossed": False, "imminent": False,
                "fast_now": 0.0, "slow_now": 0.0, "gap_pct": 0.0}
    fast = ta.SMA(arr, timeperiod=fast_p)
    slow = ta.SMA(arr, timeperiod=slow_p)
    # Check recent bars for cross: fast[i-1] <= slow[i-1] and fast[i] > slow[i]
    crossed = False
    for i in range(-lookback, 0):
        if (not np.isnan(fast[i]) and not np.isnan(slow[i]) and
                not np.isnan(fast[i - 1]) and not np.isnan(slow[i - 1])):
            if fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]:
                crossed = True
                break
    fn = float(fast[-1]) if not np.isnan(fast[-1]) else 0.0
    sn = float(slow[-1]) if not np.isnan(slow[-1]) else 0.0
    gap = (fn - sn) / (sn + 1e-12) * 100.0
    return {
        "crossed":   crossed,
        "imminent":  bool(abs(gap) < 0.15 and fn > sn * 0.998),
        "fast_now":  fn,
        "slow_now":  sn,
        "gap_pct":   float(gap),
    }


def _sine_cosine_trough_state(closes: list, cycle_bars: int) -> dict:
    """
    Fit price = C0 + A·sin(ωt + φ) via OLS.
    Compute sin and cos components at the current bar.
    Determine proximity to the ideal mathematical trough.

    The ideal trough: sin_component = 0 (crossing upward), cos_component = +max.
    In terms of the fitted model: when ωt + φ ≡ -π/2 (mod 2π), sin = -1 (absolute min).
    The RECOVERY POINT is when ωt + φ ≡ 0 (mod 2π), sin = 0 rising, cos = 1.

    divergence_angle:
      0°  = exactly at the mathematical recovery point (sin=0 rising, cos=1)
      90° = at the bottom of the sine (sin=-1)
      180°= at the top (sin=+1)

    We want divergence_angle close to 0 (near recovery) OR close to 90
    (at bottom, about to turn) — both are valid entry zones.
    Near 90° = "at the floor",  near 0° = "just turned".
    """
    arr = np.asarray(closes, dtype=float)
    n   = len(arr)
    empty = {"sin_val": 0.0, "cos_val": 0.0, "div_angle_deg": 90.0,
             "at_trough": False, "just_turned": False, "trough_proximity": 0.0,
             "cycle_phase_deg": 0.0, "amplitude": 0.0, "r_squared": 0.0}
    if n < cycle_bars * 2 or cycle_bars < 4:
        return empty
    t     = np.arange(n, dtype=float)
    omega = 2.0 * np.pi / cycle_bars
    # OLS: price = c0 + c_sin·sin(ωt) + c_cos·cos(ωt) + c_drift·t
    A_mat = np.column_stack([np.ones(n),
                              np.sin(omega * t),
                              np.cos(omega * t),
                              t])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, arr, rcond=None)
    except Exception:
        return empty
    c0, c_sin, c_cos, c_drift = coeffs
    amplitude = float(np.sqrt(c_sin ** 2 + c_cos ** 2))
    if amplitude < 1e-10:
        return empty
    # Current phase angle
    phase_offset = float(np.arctan2(c_cos, c_sin))   # OLS phase
    cur_phase    = (omega * (n - 1) + phase_offset)   # unwrapped
    cur_phase_mod = float(cur_phase % (2 * np.pi))
    # Normalised sin/cos at current bar (unit amplitude)
    sin_val = float(np.sin(cur_phase_mod))
    cos_val = float(np.cos(cur_phase_mod))
    # R² of the fit
    fitted = A_mat @ coeffs
    ss_res = float(np.sum((arr - fitted) ** 2))
    ss_tot = float(np.sum((arr - np.mean(arr)) ** 2) + 1e-9)
    r_sq   = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))
    # Divergence angle from the ideal recovery point (sin=0, cos=+1)
    # Using vector angle: ideal = (cos=1, sin=0), current = (cos_val, sin_val)
    div_angle = float(np.degrees(np.arctan2(abs(sin_val), max(cos_val, 0.0))))
    div_angle = float(np.clip(div_angle, 0.0, 180.0))
    # at_trough: phase is in [π*0.75, π*1.25] i.e. near sin = -1
    at_trough  = bool(np.pi * 0.70 <= cur_phase_mod <= np.pi * 1.30)
    # just_turned: phase is in [0, π*0.30] or [2π-0.30π, 2π] i.e. near sin = 0 rising
    just_turned = bool(cur_phase_mod <= np.pi * 0.30 or
                       cur_phase_mod >= np.pi * 1.70)
    # Trough proximity: 1.0 at the bottom (phase=π), decays to 0 at phase=0
    trough_prox = float(np.clip(
        1.0 - abs(cur_phase_mod - np.pi) / np.pi, 0.0, 1.0
    ))
    return {
        "sin_val":           sin_val,
        "cos_val":           cos_val,
        "div_angle_deg":     div_angle,
        "at_trough":         at_trough,
        "just_turned":       just_turned,
        "trough_proximity":  trough_prox,
        "cycle_phase_deg":   float(np.degrees(cur_phase_mod)),
        "amplitude":         amplitude,
        "r_squared":         r_sq,
    }


def _energy_density_per_tf(closes: list, level: int = 5) -> dict:
    """
    Compute wavelet energy density per frequency band.

    Energy density = mean(detail_band²) / len(band)
    For exhaustion confirmation, we check whether energy density is
    DECLINING in recent bars vs the historical average for each band.

    Returns per-band (trend, swing, micro) energy densities and
    the 'all_declining' flag.
    """
    arr = np.asarray(closes, dtype=float)
    n   = len(arr)
    if n < 64:
        return {"trend_ed": 0.0, "swing_ed": 0.0, "micro_ed": 0.0,
                "trend_declining": False, "swing_declining": False,
                "micro_declining": False, "all_declining": False,
                "energy_collapse_score": 0.0}
    bands = wavelet_decompose(arr, level=min(level, 4))
    def band_energy_density(b_arr: np.ndarray, window: int = 20) -> tuple:
        if len(b_arr) < window * 2:
            return 0.0, 0.0, False
        hist_ed = float(np.mean(b_arr[:-window] ** 2) / (len(b_arr) - window + 1e-9))
        rec_ed  = float(np.mean(b_arr[-window:] ** 2) / (window + 1e-9))
        declining = bool(rec_ed < hist_ed * 0.80)   # recent 20%+ below historical
        return hist_ed, rec_ed, declining
    trend_h, trend_r, trend_dec = band_energy_density(bands.get('trend', np.zeros(n)))
    swing_h, swing_r, swing_dec = band_energy_density(bands.get('swing', np.zeros(n)))
    micro_h, micro_r, micro_dec = band_energy_density(bands.get('micro', np.zeros(n)))
    all_dec   = trend_dec and swing_dec and micro_dec
    # Collapse score: how far below historical are ALL bands?
    def ratio_score(h, r): return float(np.clip(1.0 - r / (h + 1e-9), 0.0, 1.0))
    collapse = np.mean([ratio_score(trend_h, trend_r),
                        ratio_score(swing_h, swing_r),
                        ratio_score(micro_h, micro_r)])
    return {
        "trend_ed_hist":    trend_h, "trend_ed_recent":  trend_r,
        "swing_ed_hist":    swing_h, "swing_ed_recent":  swing_r,
        "micro_ed_hist":    micro_h, "micro_ed_recent":  micro_r,
        "trend_declining":  trend_dec,
        "swing_declining":  swing_dec,
        "micro_declining":  micro_dec,
        "all_declining":    all_dec,
        "energy_collapse_score": float(collapse),
    }


def _wyckoff_spring_detected(raw_klines: list, min_bars: int = 5) -> dict:
    """
    Wyckoff spring: last N bars made successively lower lows, BUT
    with DECREASING volume on each new low.
    Sellers are running out of fuel — this is the final capitulation
    before the reversal spike.

    Also detects the classic Wyckoff 'test': after a spring, price
    retests the low on even LOWER volume — strongest spring signature.

    Returns:
      spring      – bool: decreasing-vol new-lows pattern found
      test        – bool: spring + retest both detected
      new_lows    – number of successive new lows found
      vol_decline – average % decline in volume across new-low bars
      strength    – 0-1 composite strength
    """
    if not raw_klines or len(raw_klines) < min_bars + 2:
        return {"spring": False, "test": False, "new_lows": 0,
                "vol_decline": 0.0, "strength": 0.0}
    # Find bars with new lows
    closes = [float(k[4]) for k in raw_klines]
    lows   = [float(k[3]) for k in raw_klines]
    vols   = [float(k[5]) for k in raw_klines]
    # Walk backward from current bar looking for sequence of new-low bars
    new_low_bars = []
    running_low  = float('inf')
    for i in range(len(lows) - 1, max(0, len(lows) - 20), -1):
        if lows[i] < running_low:
            running_low = lows[i]
            new_low_bars.append(i)
        else:
            if len(new_low_bars) >= 2:
                break
    new_low_bars = list(reversed(new_low_bars))
    if len(new_low_bars) < 2:
        return {"spring": False, "test": False, "new_lows": 0,
                "vol_decline": 0.0, "strength": 0.0}
    # Check volumes on those bars are decreasing
    vols_at_lows = [vols[i] for i in new_low_bars]
    vol_declining = all(vols_at_lows[i] > vols_at_lows[i + 1]
                        for i in range(len(vols_at_lows) - 1))
    # Volume decline magnitude
    if len(vols_at_lows) >= 2 and vols_at_lows[0] > 0:
        vol_dec_pct = float((vols_at_lows[0] - vols_at_lows[-1]) /
                             vols_at_lows[0] * 100.0)
    else:
        vol_dec_pct = 0.0
    spring = vol_declining and len(new_low_bars) >= 2
    # Test detection: after the last new low, does price attempt the low again
    # with even lower volume?
    test = False
    if spring and len(raw_klines) > new_low_bars[-1] + 2:
        post_bars  = raw_klines[new_low_bars[-1] + 1:]
        if post_bars:
            retest_low  = min(float(k[3]) for k in post_bars[:5])
            retest_vol  = min(float(k[5]) for k in post_bars[:5])
            orig_low    = lows[new_low_bars[-1]]
            orig_vol    = vols[new_low_bars[-1]]
            test = bool(retest_low <= orig_low * 1.003 and
                        retest_vol < orig_vol * 0.70)
    strength = float(np.clip(
        0.50 * float(spring) +
        0.30 * float(test) +
        0.20 * float(np.clip(vol_dec_pct / 50.0, 0.0, 1.0)),
        0.0, 1.0
    ))
    return {
        "spring":      spring,
        "test":        test,
        "new_lows":    len(new_low_bars),
        "vol_decline": vol_dec_pct,
        "strength":    strength,
    }


def _dominant_cycle_fft(closes: list) -> int:
    """Quick FFT dominant-cycle detection. Returns bars (int)."""
    arr = np.asarray(closes, dtype=float)
    n   = len(arr)
    if n < 32:
        return 20
    detrended = arr - np.polyval(np.polyfit(np.arange(n), arr, 1), np.arange(n))
    fft_v  = np.abs(np.fft.rfft(detrended))
    fft_f  = np.fft.rfftfreq(n)
    fft_f  = np.where(fft_f == 0, 1e-12, fft_f)
    peak   = int(np.argmax(fft_v[1:]) + 1)
    cyc    = int(round(1.0 / fft_f[peak]))
    return int(np.clip(cyc, 6, n // 2))


def _harmonic_anchor_score(trader, symbol: str,
                             current_price: float,
                             tolerance_pct: float = 0.40) -> dict:
    """
    Magnetic resonance between extrema across TFs.

    For each TF (1m, 5m, 15m, 4h):
      1. Detect the dominant cycle
      2. Fit the cyclic regression line
      3. Find the last fitted trough PRICE (cyclic midline - amplitude)

    If the current price is within `tolerance_pct`% of the fitted trough
    price on 2+ TFs, they are "resonant" — the same magnetic price level
    is a turning point across multiple frequencies.

    anchor_count: 0-4 (number of TFs where current price ≈ cyclic trough)
    resonant    : bool (anchor_count >= 2)
    anchor_score: 0-1
    """
    tfs = [('1m', 500), ('5m', 200), ('15m', 200), ('4h', 200)]
    trough_prices = {}
    for interval, limit in tfs:
        try:
            c = trader.get_klines(symbol, interval, limit=limit)
            if not c or len(c) < 40:
                continue
            cyc = _dominant_cycle_fft(c)
            fit = cyclic_line_fit(np.asarray(c, dtype=float), cyc)
            trough_prices[interval] = float(fit.get("fitted_low_price", 0.0))
        except Exception:
            pass
    if not trough_prices:
        return {"anchor_count": 0, "resonant": False,
                "anchor_score": 0.0, "anchor_tfs": []}
    anchor_tfs = []
    for interval, tp in trough_prices.items():
        if tp <= 0:
            continue
        dist_pct = abs(current_price - tp) / (tp + 1e-12) * 100.0
        if dist_pct <= tolerance_pct:
            anchor_tfs.append(interval)
    count = len(anchor_tfs)
    return {
        "anchor_count":  count,
        "resonant":      count >= 2,
        "anchor_score":  float(np.clip(count / 4.0, 0.0, 1.0)),
        "anchor_tfs":    anchor_tfs,
        "trough_prices": trough_prices,
    }


def compute_exhaustion_profile(trader, symbol: str) -> dict:
    """
    Master exhaustion profile.  Runs all seven exhaustion proofs
    concurrently where possible.

    Returns a single dict with:
      resonance_score      : 0-1 composite (PRIMARY sort key)
      exhaustion_confirmed : bool (resonance_score >= 0.68)
      + all sub-signals

    Sort order for best candidate selection:
      1. exhaustion_confirmed = True  (mandatory tier)
      2. resonance_score DESC         (depth of confirmation)
      3. rsi_1m ASC                   (most oversold wins ties)
      4. anchor_count DESC            (most resonant TFs)
    """
    result = {
        "resonance_score":      0.0,
        "exhaustion_confirmed": False,
        # sine/cosine
        "sc_1m": {}, "sc_5m": {}, "sc_15m": {},
        # SMA cross
        "sma_cross_1m": {}, "sma_cross_5m": {}, "sma_cross_15m": {},
        "sma_cross_count": 0,
        # energy density
        "energy_1m": {}, "energy_5m": {},
        # wyckoff
        "wyckoff_1m": {}, "wyckoff_5m": {},
        # harmonic anchor
        "anchor": {},
        # RSI (kept for sort tiebreak)
        "rsi_1m": 50.0,
        # argmin recency (kept from v25)
        "argmin_1m": False, "argmin_5m": False, "argmin_15m": False,
        "argmin_count": 0,
        # bull vol (kept)
        "bull_vol_1m": 50.0, "bull_vol_majority": False,
    }
    current_price = 0.0
    # ── 1m ───────────────────────────────────────────────────────────
    try:
        kl1 = trader.get_klines(symbol, '1m', limit=500, return_raw=True)
        if kl1 and len(kl1) >= 60:
            c1  = [float(k[4]) for k in kl1]
            current_price = c1[-1]
            cyc1 = _dominant_cycle_fft(c1)
            # Sine/cosine trough state
            result["sc_1m"] = _sine_cosine_trough_state(c1, cyc1)
            # SMA cross
            fp1 = max(3, cyc1 // 4)
            sp1 = max(6, cyc1 // 2)
            result["sma_cross_1m"] = _sma_cross_recent(c1, fp1, sp1, lookback=3)
            # Energy density
            result["energy_1m"] = _energy_density_per_tf(c1)
            # Wyckoff spring
            result["wyckoff_1m"] = _wyckoff_spring_detected(kl1, min_bars=3)
            # RSI
            rsi_arr = ta.RSI(np.asarray(c1), timeperiod=14)
            valid   = rsi_arr[~np.isnan(rsi_arr)]
            result["rsi_1m"] = float(valid[-1]) if len(valid) > 0 else 50.0
            # Bull vol
            bull = sum(float(k[5]) for k in kl1 if float(k[4]) >= float(k[1]))
            bear = sum(float(k[5]) for k in kl1 if float(k[4]) <  float(k[1]))
            tot  = bull + bear + 1e-12
            result["bull_vol_1m"]       = float(bull / tot * 100.0)
            result["bull_vol_majority"] = bool(bull > bear)
            # Argmin
            arr1 = np.asarray(c1)
            result["argmin_1m"] = bool(int(np.argmin(arr1)) > int(np.argmax(arr1)))
    except Exception:
        pass
    # ── 5m ───────────────────────────────────────────────────────────
    try:
        kl5 = trader.get_klines(symbol, '5m', limit=200, return_raw=True)
        if kl5 and len(kl5) >= 40:
            c5  = [float(k[4]) for k in kl5]
            cyc5 = _dominant_cycle_fft(c5)
            result["sc_5m"]       = _sine_cosine_trough_state(c5, cyc5)
            fp5 = max(3, cyc5 // 4)
            sp5 = max(6, cyc5 // 2)
            result["sma_cross_5m"] = _sma_cross_recent(c5, fp5, sp5, lookback=3)
            result["energy_5m"]    = _energy_density_per_tf(c5)
            result["wyckoff_5m"]   = _wyckoff_spring_detected(kl5, min_bars=3)
            arr5 = np.asarray(c5)
            result["argmin_5m"] = bool(int(np.argmin(arr5)) > int(np.argmax(arr5)))
    except Exception:
        pass
    # ── 15m ──────────────────────────────────────────────────────────
    try:
        kl15 = trader.get_klines(symbol, '15m', limit=200, return_raw=True)
        if kl15 and len(kl15) >= 40:
            c15  = [float(k[4]) for k in kl15]
            cyc15 = _dominant_cycle_fft(c15)
            result["sc_15m"]        = _sine_cosine_trough_state(c15, cyc15)
            fp15 = max(3, cyc15 // 4)
            sp15 = max(6, cyc15 // 2)
            result["sma_cross_15m"] = _sma_cross_recent(c15, fp15, sp15, lookback=3)
            arr15 = np.asarray(c15)
            result["argmin_15m"] = bool(int(np.argmin(arr15)) > int(np.argmax(arr15)))
    except Exception:
        pass
    # ── Harmonic anchor (uses all TFs internally) ─────────────────────
    try:
        if current_price > 0:
            result["anchor"] = _harmonic_anchor_score(trader, symbol, current_price)
    except Exception:
        result["anchor"] = {"anchor_count": 0, "resonant": False,
                            "anchor_score": 0.0, "anchor_tfs": []}
    # ── Derived aggregates ────────────────────────────────────────────
    ac = sum([result["argmin_1m"], result["argmin_5m"], result["argmin_15m"]])
    result["argmin_count"] = ac
    # SMA cross count across TFs
    sma_count = sum([
        bool(result["sma_cross_1m"].get("crossed") or
             result["sma_cross_1m"].get("imminent")),
        bool(result["sma_cross_5m"].get("crossed") or
             result["sma_cross_5m"].get("imminent")),
        bool(result["sma_cross_15m"].get("crossed") or
             result["sma_cross_15m"].get("imminent")),
    ])
    result["sma_cross_count"] = sma_count
    # ── RESONANCE SCORE (0-1) ─────────────────────────────────────────
    #
    # Component weights — each targeting one dimension of exhaustion:
    #
    #  Sine/cos trough proximity 1m  → 0.18  (are we AT the mathematical trough?)
    #  Sine/cos trough proximity 5m  → 0.12
    #  SMA cross count (0-3 TFs)     → 0.15  (frequency-locked momentum flip)
    #  Energy collapse 1m            → 0.12  (selling impulse spent across freqs)
    #  Energy collapse 5m            → 0.08
    #  Wyckoff spring 1m             → 0.12  (volume exhaustion on new lows)
    #  Harmonic anchor score         → 0.10  (magnetic resonance across TFs)
    #  RSI depth 1m                  → 0.08  (oversold fuel)
    #  Argmin recency (0-3 / 3)      → 0.05  (floor established recently)
    #
    sc1_prox = float(result["sc_1m"].get("trough_proximity", 0.0))
    sc5_prox = float(result["sc_5m"].get("trough_proximity", 0.0))
    sma_frac = sma_count / 3.0
    e1_score = float(result["energy_1m"].get("energy_collapse_score", 0.0))
    e5_score = float(result["energy_5m"].get("energy_collapse_score", 0.0))
    wy1_str  = float(result["wyckoff_1m"].get("strength", 0.0))
    anchor_s = float(result["anchor"].get("anchor_score", 0.0))
    rsi_d    = float(np.clip((70.0 - result["rsi_1m"]) / 70.0, 0.0, 1.0))
    arg_frac = ac / 3.0
    # Bonus flags
    sc1_just_turned = float(result["sc_1m"].get("just_turned", False))
    sc1_at_trough   = float(result["sc_1m"].get("at_trough",   False))
    wy1_test        = float(result["wyckoff_1m"].get("test", False))
    anchor_resonant = float(result["anchor"].get("resonant", False))
    resonance = (
        sc1_prox  * 0.18 +
        sc5_prox  * 0.12 +
        sma_frac  * 0.15 +
        e1_score  * 0.12 +
        e5_score  * 0.08 +
        wy1_str   * 0.12 +
        anchor_s  * 0.10 +
        rsi_d     * 0.08 +
        arg_frac  * 0.05 +
        # Bonus: exact trough state
        sc1_just_turned * 0.06 +
        sc1_at_trough   * 0.03 +
        wy1_test        * 0.04 +
        anchor_resonant * 0.04
    )
    # Cap at 1.0 — bonuses can push above base
    resonance = float(np.clip(resonance, 0.0, 1.0))
    result["resonance_score"]      = resonance
    result["exhaustion_confirmed"] = bool(resonance >= 0.68)
    return result


def pick_best_exhausted_candidate(trader, symbols: List[str],
                                   label: str,
                                   max_workers: int = 12) -> tuple:
    """
    Rank all symbols by exhaustion_profile.resonance_score.

    Sort order:
      1. exhaustion_confirmed = True  (hard tier — must be exhausted)
      2. resonance_score DESC         (depth of confirmation)
      3. rsi_1m ASC                   (most oversold wins ties)
      4. anchor_count DESC            (most resonant TFs)

    Returns (symbol, exhaustion_dict) for the winner, or None.
    """
    if not symbols:
        return None
    W = 78
    print(f"\n  ⚡  Exhaustion-scoring {len(symbols)} {label} candidates...")
    profiles = {}
    tracker  = ProgressTracker(len(symbols), "exhaustion scoring")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(compute_exhaustion_profile, trader, s): s
                   for s in symbols}
        for f in as_completed(futures):
            try:
                sym  = futures[f]
                prof = f.result()
                profiles[sym] = prof
                tracker.update(passed=bool(prof.get("exhaustion_confirmed")))
                print(tracker.get_stats(), end="", flush=True)
            except Exception:
                tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20)
    if not profiles:
        return None
    confirmed = [(s, p) for s, p in profiles.items() if p.get("exhaustion_confirmed")]
    unconfirmed = [(s, p) for s, p in profiles.items() if not p.get("exhaustion_confirmed")]
    def sort_key(pair):
        _, p = pair
        return (
            -p.get("resonance_score", 0.0),
             p.get("rsi_1m", 50.0),
            -p.get("argmin_count", 0),
            -p.get("anchor", {}).get("anchor_count", 0),
        )
    confirmed.sort(key=sort_key)
    unconfirmed.sort(key=sort_key)
    ranked = confirmed + unconfirmed
    n_confirmed = len(confirmed)
    print(f"\n  📊  Exhaustion ranked ({label}) — "
          f"{n_confirmed} confirmed / {len(unconfirmed)} soft:")
    for i, (sym, p) in enumerate(ranked[:8], 1):
        tier = "✅EXHST" if p.get("exhaustion_confirmed") else "·soft "
        rs   = f"res={p.get('resonance_score', 0)*100:.1f}%"
        rsi  = f"RSI={p.get('rsi_1m', 50):.1f}"
        sc1  = f"sinφ={p.get('sc_1m', {}).get('trough_proximity', 0)*100:.0f}%"
        sma  = f"SMAx={p.get('sma_cross_count', 0)}/3"
        wy   = f"wy={p.get('wyckoff_1m', {}).get('strength', 0)*100:.0f}%"
        anc  = f"anc={p.get('anchor', {}).get('anchor_count', 0)}"
        print(f"     #{i:<2} {sym:<20} {tier}  {rs}  {rsi}  {sc1}  {sma}  {wy}  {anc}")
    return ranked[0]


def format_exhaustion_block(exh: dict, W: int = 74):
    """
    Print the full exhaustion profile block in format_sr_output.
    Shows all seven exhaustion dimensions + resonance score verdict.
    """
    if not exh:
        return
    print("─" * W)
    print("  ⚡  CONFIRMED DIP EXHAUSTION ENGINE  (7-DIMENSIONAL PROOF)")
    print("─" * W)
    def yn(v): return "✅ YES" if v else "❌ NO"
    rs   = exh.get("resonance_score", 0.0)
    conf = exh.get("exhaustion_confirmed", False)
    # 1. Sine/cosine trough state
    sc1 = exh.get("sc_1m", {})
    sc5 = exh.get("sc_5m", {})
    print(f"  [1] Sine/Cos Trough  [1m] : "
          f"phase={sc1.get('cycle_phase_deg',0):.0f}°  "
          f"sin={sc1.get('sin_val',0):+.3f}  cos={sc1.get('cos_val',0):+.3f}  "
          f"prox={sc1.get('trough_proximity',0)*100:.0f}%  "
          f"at_trough={yn(sc1.get('at_trough'))}  "
          f"just_turned={yn(sc1.get('just_turned'))}")
    print(f"       Sine/Cos Trough  [5m] : "
          f"phase={sc5.get('cycle_phase_deg',0):.0f}°  "
          f"prox={sc5.get('trough_proximity',0)*100:.0f}%  "
          f"at_trough={yn(sc5.get('at_trough'))}  "
          f"R²={sc5.get('r_squared',0):.2f}")
    # 2. SMA cross
    sx1 = exh.get("sma_cross_1m",  {})
    sx5 = exh.get("sma_cross_5m",  {})
    sx15= exh.get("sma_cross_15m", {})
    sxc = exh.get("sma_cross_count", 0)
    print(f"  [2] SMA Cross [1m]: {yn(sx1.get('crossed') or sx1.get('imminent'))} "
          f"gap={sx1.get('gap_pct',0):+.3f}%  "
          f"[5m]: {yn(sx5.get('crossed') or sx5.get('imminent'))}  "
          f"[15m]: {yn(sx15.get('crossed') or sx15.get('imminent'))}  "
          f"({sxc}/3 TFs)")
    # 3. Energy density
    e1 = exh.get("energy_1m", {})
    e5 = exh.get("energy_5m", {})
    print(f"  [3] Energy Collapse [1m]: "
          f"trend={yn(e1.get('trend_declining'))}  "
          f"swing={yn(e1.get('swing_declining'))}  "
          f"micro={yn(e1.get('micro_declining'))}  "
          f"all={yn(e1.get('all_declining'))}  "
          f"score={e1.get('energy_collapse_score',0)*100:.0f}%")
    print(f"       Energy Collapse [5m]: "
          f"trend={yn(e5.get('trend_declining'))}  "
          f"swing={yn(e5.get('swing_declining'))}  "
          f"micro={yn(e5.get('micro_declining'))}  "
          f"all={yn(e5.get('all_declining'))}  "
          f"score={e5.get('energy_collapse_score',0)*100:.0f}%")
    # 4. Wyckoff spring
    wy1 = exh.get("wyckoff_1m", {})
    wy5 = exh.get("wyckoff_5m", {})
    print(f"  [4] Wyckoff Spring  [1m]: "
          f"spring={yn(wy1.get('spring'))}  "
          f"test={yn(wy1.get('test'))}  "
          f"new_lows={wy1.get('new_lows',0)}  "
          f"vol_decline={wy1.get('vol_decline',0):.1f}%  "
          f"strength={wy1.get('strength',0)*100:.0f}%")
    print(f"       Wyckoff Spring  [5m]: "
          f"spring={yn(wy5.get('spring'))}  "
          f"test={yn(wy5.get('test'))}  "
          f"strength={wy5.get('strength',0)*100:.0f}%")
    # 5. Harmonic anchor
    anc = exh.get("anchor", {})
    atfs = anc.get("anchor_tfs", [])
    print(f"  [5] Harmonic Anchor : "
          f"count={anc.get('anchor_count',0)}/4 TFs  "
          f"resonant={yn(anc.get('resonant'))}  "
          f"tfs={atfs}  "
          f"score={anc.get('anchor_score',0)*100:.0f}%")
    # 6. RSI + argmin + bull vol
    rsi_v = exh.get("rsi_1m", 50.0)
    bv    = exh.get("bull_vol_1m", 50.0)
    ac    = exh.get("argmin_count", 0)
    print(f"  [6] RSI(14) [1m]    : {rsi_v:.2f}  "
          f"{'✅ OVERSOLD' if rsi_v < 30 else ('⚠️ borderline' if rsi_v < 40 else '❌ not oversold')}")
    print(f"  [7] Argmin recency  : {ac}/3 TFs confirm floor  "
          f"| 1m bull vol: {bv:.1f}%  {yn(exh.get('bull_vol_majority'))}")
    # Resonance composite bar
    bar = "█" * int(rs * 20) + "░" * (20 - int(rs * 20))
    verdict = "✅ EXHAUSTION CONFIRMED — dip floor proven across all dimensions" if conf else \
              ("⚠️  NEAR THRESHOLD — strong but not all dimensions confirmed" if rs >= 0.50 else \
               "❌ NOT EXHAUSTED — risk of further decline remains")
    print(f"\n  ⚡  RESONANCE SCORE : [{bar}]  {rs*100:.1f}%")
    print(f"  ⚡  VERDICT         : {verdict}")
    print()

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



def check_1m_final(trader, symbol):
    """
    1m final filter — dipzz28.

    Gate hierarchy (all must pass for is_strong = True):

    LAYER 0  HT_FFT HARD GATE  [immediate reject on any failure]
      Fetches 1000 bars (maximum Binance 1m history).
      Uses dominant-cycle FFT to determine local-dip lookback window.

      A. local_dip_price > global_argmin_price
         The most recent local dip low (confirmed swing low within one
         dominant cycle) must be STRICTLY ABOVE the global floor of the
         full 1000-bar window.  This is NOT current_close — it is the
         actual bottom of the current local dip move.
         If it equals or undercuts the global floor → new structural low
         → no established support → REJECT.

      B. argmin_idx > argmax_idx  (on full 1000-bar window)
         The most recent extreme in the full window must be a LOW, not
         a HIGH.  If the most recent extreme is a high, price is falling
         FROM that high — we are mid-downswing.  REJECT.

      C. fft_forecast_next > current_close
         The top-5 dominant frequencies reconstructed from 1000 bars
         and projected forward must predict the next bar HIGHER than the
         current close.  If the dominant cycle is still pointing down
         → dip is NOT over → REJECT.

    LAYER 1  STRUCTURAL DIP (cond 1-4)
      cond_1: RSI < 30 + MACD hist turning
      cond_2: price below OLS regression lower band
      cond_3: sinusoidal wave at bottom AND turning up
      cond_4: cycle wave confirmed turning up

    LAYER 2  FREQUENCY ALIGNMENT (cond 5-8, need ≥ 2)
      cond_5: wavelet swing reversal on 15m
      cond_6: micro momentum positive on 5m
      cond_7: 1m spectral cycle at trough
      cond_8: MTF frequency alignment ≥ 3/5

    LAYER 3  EXHAUSTION PROFILE
      compute_exhaustion_profile runs the 7-dimensional exhaustion check.
      Resonance score is boosted by HT_FFT gate quality.
      If exhaustion_confirmed AND freq_extra >= 1 → is_strong = True
      (relaxes layer-2 requirement when exhaustion is proven).
    """
    default_dict = {
        "golden_score": 0.0, "energy_state": "INSUFFICIENT",
        "spike_prob": 0.0, "phase_aligned": False, "near_min": False,
        "wave_near_bottom": False, "turning_up": False,
        "est_bars_to_pump": 0, "phase_pos": 0.5,
        "freq": None, "_ht_fft_gate": {}, "_exh": {},
    }

    # ── LAYER 0: HT_FFT HARD GATE ────────────────────────────────────
    klines_full = trader.get_klines(symbol, '1m', limit=1000, return_raw=True)
    if not klines_full or len(klines_full) < 64:
        return (symbol, 0.0, 0.0, False, 0.0, 0.0, default_dict)

    ht_fft_gate = compute_ht_fft_gate(klines_full)
    default_dict["_ht_fft_gate"] = ht_fft_gate

    if not ht_fft_gate["gate_pass"]:
        return (symbol, 0.0, 0.0, False, 0.0, 0.0, default_dict)

    # ── LAYERS 1-3: deeper analysis on last 500 bars ──────────────────
    klines = klines_full[-500:] if len(klines_full) > 500 else klines_full
    if len(klines) < 100:
        return (symbol, 0.0, 0.0, False, 0.0, 0.0, default_dict)

    close         = [float(k[4]) for k in klines]
    volumes       = [float(k[5]) for k in klines]
    current_price = float(close[-1])

    cmo     = ta.CMO(np.asarray(close), timeperiod=14)
    cmo_val = float(cmo[-1]) if not np.isnan(cmo[-1]) else 0.0

    closed_vols = [v for v in volumes[:-1] if v > 0]
    vratio = 0.0
    if closed_vols:
        avg_vol = np.mean(closed_vols[-50:])
        vratio  = closed_vols[-1] / avg_vol if avg_vol > 0 else 0.0

    is_rej, bull_ratio = has_bullish_rejection_volume(klines, window=10)
    metrics = calculate_effort_result_metrics(close, volumes, window=20)
    prob    = ml_spike_probability(metrics["R"], metrics["C"], metrics["E"],
                                   bull_ratio, cmo_val, vratio)

    golden     = compute_phase_alignment(close, dt=1.0, N=3, epsilon=0.18)
    sinusoidal = get_sinusoidal_dip_timing(close, 500)
    combined   = {**golden, **sinusoidal}
    combined["_ht_fft_gate"] = ht_fft_gate

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

    klines_tight  = klines[-200:] if len(klines) > 200 else klines
    dom_cycle_now = int((freq_result or {}).get("dominant_cycle", 0) or 0)
    quad_forecast = quadratic_price_forecast(
        klines_tight, current_price, combined, dom_cycle=dom_cycle_now)
    combined['quad_forecast'] = quad_forecast

    # LAYER 1
    cond_1 = is_confirmed_dip(close, high_tf=False)
    cond_2 = is_below_regression_low(close, deviation=0.01)
    cond_3 = combined.get('wave_near_bottom', False)
    cond_4 = combined.get('turning_up',       False)

    # LAYER 2
    if freq_result:
        cond_5 = bool(freq_result.get('cond_swing_rev',   False))
        cond_6 = bool(freq_result.get('cond_micro_spark', False))
        cond_7 = bool(freq_result.get('cond_1m_trough',   False))
        cond_8 = bool(freq_result.get('strong_alignment', False))
    else:
        cond_5 = cond_6 = cond_7 = cond_8 = False

    freq_extra = sum([cond_5, cond_6, cond_7, cond_8])
    is_strong  = cond_1 and cond_2 and cond_3 and cond_4 and freq_extra >= 2

    # LAYER 3 — Exhaustion profile
    try:
        exh = compute_exhaustion_profile(trader, symbol)
        # Boost resonance score by HT_FFT gate quality (up to +20%)
        exh["resonance_score"] = float(np.clip(
            exh.get("resonance_score", 0.0) +
            ht_fft_gate.get("resonance_extra", 0.0) * 0.20,
            0.0, 1.0
        ))
        exh["exhaustion_confirmed"] = bool(exh["resonance_score"] >= 0.68)
        combined["_exh"] = exh
        # Relax layer-2 gate if exhaustion fully confirmed
        if exh.get("exhaustion_confirmed") and freq_extra >= 1:
            is_strong = True
    except Exception:
        combined["_exh"] = {}

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

    # ── Exhaustion Profile Block (new — dipzz28) ──────────────────────
    exh = golden.get("_exh") if golden else None
    if exh:
        format_exhaustion_block(exh, W)

    # ── HT_SINE + FFT Forecast Gate Block (new — dipzz28) ────────────
    ht_fft = golden.get("_ht_fft_gate") if golden else None
    if ht_fft:
        format_ht_fft_block(ht_fft, W)

    # ── Micro-TF Dip Quality Block (v25 — kept as supplementary) ─────
    micro = golden.get("_micro") if golden else None
    if micro:
        format_micro_block(micro, W)

    all_signals = []

    # ── HT_SINE + FFT gate signals ────────────────────────────────────
    if ht_fft and ht_fft.get("gate_pass"):
        ht_d  = ht_fft.get("ht", {})
        fft_d = ht_fft.get("fft", {})
        all_signals.append("📡 HT_FFT GATE PASS — all 3 mandatory conditions confirmed")
        if ht_d.get("ht_bull_signal"):
            all_signals.append(
                f"📡 HT_SINE bull: leadsine({ht_d.get('leadsine',0):.3f}) > "
                f"sine({ht_d.get('sine',0):.3f}) — cycle turning up from trough")
        fn  = fft_d.get("forecast_next",  0.0)
        cc  = fft_d.get("current_close",  0.0)
        fd  = (fn - cc) / (cc + 1e-12) * 100.0 if cc > 0 else 0.0
        cyc = fft_d.get("dominant_cycles", [])
        all_signals.append(f"📡 FFT forecast: next={fn:.8f} ({fd:+.4f}%) cycles={cyc}")
        ldp = ht_d.get("local_dip_price", 0.0)
        llf = ht_d.get("argmin_price",    0.0)
        gap = (ldp - llf) / (llf + 1e-12) * 100.0 if llf > 0 else 0.0
        all_signals.append(
            f"📡 Local dip ({ldp:.8f}) is {gap:+.4f}% above global floor ({llf:.8f})")

    # ── EXHAUSTION signals ─────────────────────────────────────────────
    if exh:
        rs   = exh.get("resonance_score", 0.0)
        conf = exh.get("exhaustion_confirmed", False)
        if conf:
            all_signals.append(
                f"⚡ DIP EXHAUSTION CONFIRMED (resonance={rs*100:.0f}%) — "
                f"selling force proven spent across all 7 dimensions")
        if exh.get("sc_1m", {}).get("just_turned"):
            all_signals.append("⚡ Sine cycle JUST TURNED at 1m trough — mathematical reversal point hit")
        if exh.get("sc_1m", {}).get("at_trough"):
            all_signals.append("⚡ 1m sine cycle AT TROUGH — price at mathematical floor of cycle")
        if exh.get("sma_cross_count", 0) >= 2:
            sxc = exh.get("sma_cross_count", 0)
            all_signals.append(f"⚡ Frequency-locked SMA cross on {sxc}/3 TFs — cycle momentum flipping")
        if exh.get("energy_1m", {}).get("all_declining"):
            all_signals.append("⚡ Energy density COLLAPSING on all 1m wavelet bands — impulse spent")
        if exh.get("wyckoff_1m", {}).get("test"):
            all_signals.append("⚡ Wyckoff SPRING + TEST confirmed on 1m — classic capitulation pattern")
        elif exh.get("wyckoff_1m", {}).get("spring"):
            ndl = exh["wyckoff_1m"].get("new_lows", 0)
            vd  = exh["wyckoff_1m"].get("vol_decline", 0)
            all_signals.append(f"⚡ Wyckoff spring on 1m ({ndl} new lows, {vd:.0f}% vol decline)")
        if exh.get("anchor", {}).get("resonant"):
            atfs = exh["anchor"].get("anchor_tfs", [])
            all_signals.append(f"⚡ Harmonic anchor RESONANT across {atfs} — magnetic price floor")
        if exh.get("sc_5m", {}).get("at_trough"):
            all_signals.append("⚡ 5m sine cycle also AT TROUGH — multi-TF trough synchrony")

    # ── Micro-TF signals (v25 — kept as supplementary) ────────────────
    if micro:
        def _yn(v): return "YES" if v else "NO"
        ac  = micro.get("argmin_count", 0)
        hp  = micro.get("hard_pass", False)
        rsi = micro.get("rsi_1m", {})
        vol = micro.get("vol_1m", {})
        geo1 = micro.get("geo_1m", {})
        geo5 = micro.get("geo_5m", {})
        if hp:
            all_signals.append(
                f"🔬 MICRO-TF HARD PASS — argmin {ac}/3 + RSI({rsi.get('rsi_val',50):.1f}) "
                f"+ bull_vol({vol.get('bull_pct',50):.0f}%) + geometry ALL confirmed")
        else:
            if ac >= 2:
                all_signals.append(f"🔬 Argmin more recent on {ac}/3 micro TFs → partial floor")
            if rsi.get("rsi_oversold"):
                all_signals.append(f"🔬 1m RSI oversold ({rsi.get('rsi_val',50):.1f}) → deep dip")
            if vol.get("bull_majority"):
                all_signals.append(f"🔬 1m bull vol dominant ({vol.get('bull_pct',50):.1f}%) → absorption")
        if geo1.get("curvature", 0) > 0:
            all_signals.append("🔬 1m price geometry: concave-up (bowl shape forming → bounce geometry)")
        if geo5.get("curvature", 0) > 0:
            all_signals.append("🔬 5m price geometry: concave-up (deceleration confirmed)")

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
    CREDENTIALS_FILE     = "credentials.txt"
    MIN_SIGNALS_REQUIRED = 3
    MAX_SCANS_BEFORE_DAILY = 10

    print("=" * 78)
    print("  🌊  STRICT MTF CASCADE DIP DETECTOR  dipzz28")
    print("  🌊  1D→4H→2H→15M→5M(REG)→1M  +  HT_SINE/FFT HARD GATE")
    print("  🌊  (HT_SINE·EXTREMA · FFT·FORECAST · EXHAUSTION · φ · WAVELET)")
    print("=" * 78)
    print()

    try:
        trader = Trader(CREDENTIALS_FILE)
    except Exception as e:
        print(f"❌ Failed to initialize trader: {e}")
        return

    scan_count          = 0
    no_major_dip_streak = 0

    # ──────────────────────────────────────────────────────────────────
    # _deliver_cascade_best: exhaustion-sorted stall delivery
    # ──────────────────────────────────────────────────────────────────
    def _deliver_cascade_best(pool: List[str], label: str) -> bool:
        if not pool:
            return False
        W = 78
        print("\n" + "╔" + "═" * W + "╗")
        print("║" + " " * W + "║")
        print("║" + f"  🎯  CASCADE STALL — EXHAUSTION-SORTED BEST {label} CANDIDATE".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("║" + f"  {len(pool)} pair(s) at {label} level. Ranking by resonance score...".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("╚" + "═" * W + "╝")

        best = pick_best_exhausted_candidate(trader, pool, label)
        if not best:
            return False
        best_sym, best_exh = best

        _, best_score, best_combined = score_15m_candidate(trader, best_sym)
        if not best_combined:
            return False
        best_combined["_exh"]               = best_exh
        best_combined["_is_cascade_fallback"] = True
        best_combined["_cascade_label"]       = label

        cmo_val    = best_combined.get("_cmo_val",    0.0)
        vratio     = best_combined.get("_vratio",     0.0)
        bull_ratio = best_combined.get("_bull_ratio", 0.0)
        ml_prob    = best_combined.get("_ml_prob",    0.0)
        rs         = best_exh.get("resonance_score", 0.0)
        conf_tag   = "✅ EXHAUSTED" if best_exh.get("exhaustion_confirmed") else "⚠️  soft"

        print("\n" + "╔" + "═" * W + "╗")
        print("║" + " " * W + "║")
        print("║" + f"  🥇  BEST {label} — EXHAUSTION CONFIRMED".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("║" + f"  Asset          : {best_sym}".ljust(W) + "║")
        print("║" + f"  Resonance Score: {rs*100:.1f}%  {conf_tag}".ljust(W) + "║")
        print("║" + f"  RSI(14) 1m     : {best_exh.get('rsi_1m', 50):.2f}".ljust(W) + "║")
        print("║" + f"  Wyckoff spring : {'✅ YES' if best_exh.get('wyckoff_1m', {}).get('spring') else '❌ NO'}  "
              f"test={'✅' if best_exh.get('wyckoff_1m', {}).get('test') else '❌'}".ljust(W) + "║")
        print("║" + f"  Harmonic anchor: {best_exh.get('anchor', {}).get('anchor_count', 0)}/4 TFs  "
              f"resonant={'✅' if best_exh.get('anchor', {}).get('resonant') else '❌'}".ljust(W) + "║")
        print("║" + f"  SMA cross TFs  : {best_exh.get('sma_cross_count', 0)}/3".ljust(W) + "║")
        print("║" + f"  Energy collapse: 1m={'✅' if best_exh.get('energy_1m', {}).get('all_declining') else '❌'}  "
              f"5m={'✅' if best_exh.get('energy_5m', {}).get('all_declining') else '❌'}".ljust(W) + "║")
        print("║" + f"  MTF Score      : {best_score:.4f}  CMO: {cmo_val:+.2f}  ML: {ml_prob*100:.1f}%".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("╚" + "═" * W + "╝")

        klines_1m = trader.get_klines(best_sym, "1m", limit=1200, return_raw=True)
        if not klines_1m:
            print(f"❌ Could not fetch 1m klines for {best_sym}.")
            return False
        current_price = float(klines_1m[-1][4])
        sr = get_sr_targets(klines_1m, current_price)
        tf_volumes = {
            "5m":  get_volume_breakdown(trader, best_sym, "5m"),
            "15m": get_volume_breakdown(trader, best_sym, "15m"),
            "4h":  get_volume_breakdown(trader, best_sym, "4h"),
            "1d":  get_volume_breakdown(trader, best_sym, "1d"),
        }
        verdict, signal_count = format_sr_output(
            best_sym, sr, current_price, cmo_val, vratio,
            bull_ratio, ml_prob, tf_volumes, best_combined
        )
        print("\n" + "╔" + "═" * W + "╗")
        print("║" + " " * W + "║")
        print("║" + f"  🎯  {label} EXHAUSTION-SORTED DIP DELIVERED — STOPPING".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("║" + f"  Asset    : {best_sym}".ljust(W) + "║")
        print("║" + f"  Price    : {current_price:.8f} USDC".ljust(W) + "║")
        print("║" + f"  Signals  : {signal_count}".ljust(W) + "║")
        print("║" + f"  Verdict  : {verdict}".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("╚" + "═" * W + "╝")
        print(f"\n✅ Bot completed ({label} exhaustion-sorted). Exiting...")
        return True

    # ══════════════════════════════════════════════════════════════════
    # INFINITE SCAN LOOP
    # ══════════════════════════════════════════════════════════════════
    while True:
        scan_count += 1
        print_scan_header(scan_count)

        try:
            symbols = trader.get_usdc_pairs()
            if not symbols:
                print("❌ No USDC pairs found. Retrying in 10s...")
                time.sleep(10)
                continue

            # ── Major TF gates (1D → 4H → 2H) — INFINITE RETRY ────────
            daily_passed = run_tf_filter(trader, symbols, "1d", max_workers=20)
            if not daily_passed:
                no_major_dip_streak += 1
                print(f"⚠️  No 1D dips (streak={no_major_dip_streak}). Retrying immediately...")
                time.sleep(5)
                continue

            four_h_passed = run_tf_filter(trader, daily_passed, "4h", max_workers=20)
            if not four_h_passed:
                no_major_dip_streak += 1
                print(f"⚠️  4H cleared all (streak={no_major_dip_streak}). Retrying...")
                time.sleep(5)
                continue

            two_h_passed = run_tf_filter(trader, four_h_passed, "2h", max_workers=20)
            if not two_h_passed:
                no_major_dip_streak += 1
                print(f"⚠️  2H cleared all (streak={no_major_dip_streak}). Retrying...")
                time.sleep(5)
                continue

            no_major_dip_streak = 0
            print(f"\n✅ {len(two_h_passed)} pairs confirmed on ALL major TFs (1D+4H+2H).")

            # ── 15M ────────────────────────────────────────────────────
            fifteen_m_passed = run_tf_filter(trader, two_h_passed, "15m", max_workers=20)
            if not fifteen_m_passed:
                print("⚠️  15M cleared all — exhaustion-sorted 1D+4H+2H delivery...")
                if _deliver_cascade_best(two_h_passed, "1D+4H+2H"):
                    return
                time.sleep(5)
                continue

            # ── 5M regression ──────────────────────────────────────────
            five_m_passed = run_5m_regression_filter(trader, fifteen_m_passed, max_workers=20)
            if not five_m_passed:
                print("⚠️  5M regression cleared all — MTF fallback (exhaustion-sorted)...")
                fallback = run_best_mtf_fallback(trader, fifteen_m_passed, max_workers=15)
                if not fallback:
                    if _deliver_cascade_best(two_h_passed, "1D+4H+2H"):
                        return
                    time.sleep(5)
                    continue

                f_symbol   = fallback[0]
                f_cmo      = fallback[1]
                f_vratio   = fallback[2]
                f_bull     = fallback[4]
                f_ml       = fallback[5]
                f_combined = fallback[6]
                f_score    = f_combined.get("_fallback_score", 0.0)
                f_rank     = f_combined.get("_fallback_rank", "?")
                # Compute exhaustion for output
                f_exh = compute_exhaustion_profile(trader, f_symbol)
                f_combined["_exh"] = f_exh

                W = 78
                print("\n" + "╔" + "═" * W + "╗")
                print("║" + " " * W + "║")
                print("║" + "  ⚡  FALLBACK — BEST MTF DIP (NO 5M REGRESSION)".ljust(W) + "║")
                print("║" + " " * W + "║")
                print("║" + f"  Candidate   : {f_symbol}  (rank {f_rank})".ljust(W) + "║")
                print("║" + f"  MTF Score   : {f_score:.4f}".ljust(W) + "║")
                print("║" + f"  Resonance   : {f_exh.get('resonance_score',0)*100:.1f}%  "
                      f"{'✅EXHAUSTED' if f_exh.get('exhaustion_confirmed') else '·soft'}".ljust(W) + "║")
                print("║" + f"  CMO         : {f_cmo:+.2f}  ML: {f_ml*100:.1f}%  "
                      f"BullRej: {f_bull*100:.1f}%".ljust(W) + "║")
                print("║" + " " * W + "║")
                print("║" + "  ⚠️  5m regression NOT met — lower confidence".ljust(W) + "║")
                print("║" + " " * W + "║")
                print("╚" + "═" * W + "╝")

                klines_1m = trader.get_klines(f_symbol, "1m", limit=1200, return_raw=True)
                if klines_1m:
                    current_price = float(klines_1m[-1][4])
                    sr = get_sr_targets(klines_1m, current_price)
                    tf_volumes = {
                        "5m":  get_volume_breakdown(trader, f_symbol, "5m"),
                        "15m": get_volume_breakdown(trader, f_symbol, "15m"),
                        "4h":  get_volume_breakdown(trader, f_symbol, "4h"),
                        "1d":  get_volume_breakdown(trader, f_symbol, "1d"),
                    }
                    verdict, signal_count = format_sr_output(
                        f_symbol, sr, current_price, f_cmo, f_vratio,
                        f_bull, f_ml, tf_volumes, f_combined
                    )
                    print(f"\n✅ Bot completed (fallback mode). Exiting...")
                    return
                time.sleep(3)
                continue

            # ── 1M final filter ────────────────────────────────────────
            results_1m = run_1m_filter(trader, five_m_passed, max_workers=15)
            if not results_1m:
                print("⚠️  1m filter empty — exhaustion-sorted 1D+4H+2H+15M delivery...")
                if _deliver_cascade_best(fifteen_m_passed, "1D+4H+2H+15M"):
                    return
                time.sleep(5)
                continue

            # ── EXHAUSTION SORT of 1m-passed results ───────────────────
            passed_syms = [r[0] for r in results_1m]
            best_pick   = pick_best_exhausted_candidate(
                trader, passed_syms, "1D+4H+2H+15M+5M+1M"
            )
            result_lookup = {r[0]: r for r in results_1m}

            def final_score(r):
                ml  = r[5]
                gs  = r[6].get("spike_prob", 0.0)
                nb  = r[6].get("wave_near_bottom", False)
                tu  = r[6].get("turning_up", False)
                eb  = r[6].get("est_bars_to_pump", 0)
                fa  = (r[6].get("freq") or {}).get("alignment_score", 0.0)
                rv  = r[6].get("reversal_score", 0.0)
                cl  = r[6].get("cyc_at_cyclic_low", False)
                pr  = r[6].get("phi_reversal_ready", False)
                pb  = float(r[6].get("phi_fwd_bias", 0.0))
                pz  = r[6].get("phi_phase_zone", "NEUTRAL")
                # Exhaustion resonance from _exh (if already computed in check_1m_final)
                exh = r[6].get("_exh") or {}
                rs  = float(exh.get("resonance_score", 0.0))
                ec  = float(exh.get("exhaustion_confirmed", False))
                score = (ml * 0.22 + gs * 0.16 + fa * 0.10 + rv * 0.10
                         + pb * 5.0 * 0.04 + rs * 0.18)
                if nb:  score += 0.08
                if tu:  score += 0.07
                if cl:  score += 0.06
                if pr:  score += 0.05
                if pz in ("REVERSAL_LOW", "BIAS_BULL"): score += 0.04
                if eb < 15: score += 0.04
                if ec:  score += 0.10   # exhaustion confirmed bonus
                score -= r[1] * 0.0015
                return score

            results_1m.sort(key=final_score, reverse=True)

            if best_pick and best_pick[0] in result_lookup:
                best_sym, best_exh_data = best_pick
                top_candidate = result_lookup[best_sym]
                # Ensure exhaustion data attached
                if "_exh" not in top_candidate[6] or not top_candidate[6]["_exh"]:
                    top_candidate[6]["_exh"] = best_exh_data
                print(f"\n⚡ Exhaustion winner: {best_sym}  "
                      f"res={best_exh_data.get('resonance_score',0)*100:.1f}%  "
                      f"{'✅EXHAUSTED' if best_exh_data.get('exhaustion_confirmed') else '·soft'}  "
                      f"RSI={best_exh_data.get('rsi_1m',50):.1f}  "
                      f"anchor={best_exh_data.get('anchor',{}).get('anchor_count',0)}/4")
            else:
                top_candidate  = results_1m[0]
                best_exh_data  = top_candidate[6].get("_exh") or {}
                print(f"\n⚠️  Exhaustion sort had no result — using final_score top: {top_candidate[0]}")

            symbol     = top_candidate[0]
            cmo_val    = top_candidate[1]
            vratio     = top_candidate[2]
            bull_ratio = top_candidate[4]
            ml_prob    = top_candidate[5]
            golden     = top_candidate[6]

            print(f"\n🏆 TOP CANDIDATE     : {symbol}")
            print(f"   φ/wavelet score  : {final_score(top_candidate):.4f}")
            exh_now = golden.get("_exh") or {}
            rs_now  = exh_now.get("resonance_score", 0.0)
            print(f"   Resonance score  : {rs_now*100:.1f}%  "
                  f"{'✅EXHAUSTION CONFIRMED' if exh_now.get('exhaustion_confirmed') else '⚠️ soft'}")
            print(f"   RSI(14) 1m       : {exh_now.get('rsi_1m', 50):.2f}")
            print(f"   Sine trough 1m   : {exh_now.get('sc_1m', {}).get('trough_proximity',0)*100:.0f}%  "
                  f"at_trough={'✅' if exh_now.get('sc_1m',{}).get('at_trough') else '❌'}  "
                  f"just_turned={'✅' if exh_now.get('sc_1m',{}).get('just_turned') else '❌'}")
            print(f"   SMA cross TFs    : {exh_now.get('sma_cross_count',0)}/3")
            print(f"   Wyckoff spring   : {'✅' if exh_now.get('wyckoff_1m',{}).get('spring') else '❌'}  "
                  f"test={'✅' if exh_now.get('wyckoff_1m',{}).get('test') else '❌'}")
            print(f"   Harmonic anchor  : {exh_now.get('anchor',{}).get('anchor_count',0)}/4 TFs  "
                  f"resonant={'✅' if exh_now.get('anchor',{}).get('resonant') else '❌'}")
            print(f"   Energy collapse  : 1m={'✅' if exh_now.get('energy_1m',{}).get('all_declining') else '❌'}  "
                  f"5m={'✅' if exh_now.get('energy_5m',{}).get('all_declining') else '❌'}")

            klines_1m = trader.get_klines(symbol, "1m", limit=1200, return_raw=True)
            if klines_1m:
                current_price = float(klines_1m[-1][4])
                sr = get_sr_targets(klines_1m, current_price)
                tf_volumes = {
                    "5m":  get_volume_breakdown(trader, symbol, "5m"),
                    "15m": get_volume_breakdown(trader, symbol, "15m"),
                    "4h":  get_volume_breakdown(trader, symbol, "4h"),
                    "1d":  get_volume_breakdown(trader, symbol, "1d"),
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
                    print("║" + f"  Asset          : {symbol}".ljust(W) + "║")
                    print("║" + f"  Price          : {current_price:.8f} USDC".ljust(W) + "║")
                    print("║" + f"  Signals        : {signal_count}".ljust(W) + "║")
                    print("║" + f"  Resonance      : {rs_now*100:.1f}%  "
                          f"{'✅EXHAUSTION CONFIRMED' if exh_now.get('exhaustion_confirmed') else '⚠️ soft'}".ljust(W) + "║")
                    print("║" + f"  Verdict        : {verdict}".ljust(W) + "║")
                    print("║" + " " * W + "║")
                    print("║" + "  🎯  Review the analysis above and make your decision.".ljust(W) + "║")
                    print("║" + " " * W + "║")
                    print("╚" + "═" * W + "╝")
                    print("\n✅ Bot completed successfully. Exiting...")
                    return
                else:
                    print(f"\n⚠️  {signal_count} signals (need {MIN_SIGNALS_REQUIRED}). Continuing...")
                    time.sleep(3)
            else:
                print(f"\n❌ Failed to fetch 1m klines for {symbol}")
                time.sleep(3)

            # ── Daily fallback after MAX_SCANS_BEFORE_DAILY ─────────────
            if scan_count >= MAX_SCANS_BEFORE_DAILY and no_major_dip_streak == 0:
                W = 78
                print("\n" + "╔" + "═" * W + "╗")
                print("║" + " " * W + "║")
                print("║" + "  ⚠️   MAX SCANS — DAILY BEST-DIP FALLBACK".ljust(W) + "║")
                print("║" + " " * W + "║")
                print("╚" + "═" * W + "╝")
                try:
                    all_symbols = trader.get_usdc_pairs()
                except Exception as e:
                    print(f"❌ {e}")
                    scan_count = 0
                    time.sleep(5)
                    continue
                daily_fallback = run_daily_best_fallback(trader, all_symbols, max_workers=15)
                if not daily_fallback:
                    print("⚠️  Daily fallback empty. Resetting and retrying...")
                    scan_count = 0
                    time.sleep(10)
                    continue
                df_symbol   = daily_fallback[0]
                df_cmo      = daily_fallback[1]
                df_vratio   = daily_fallback[2]
                df_bull     = daily_fallback[4]
                df_ml       = daily_fallback[5]
                df_combined = daily_fallback[6]
                df_score    = df_combined.get("_daily_fallback_score", 0.0)
                df_rank     = df_combined.get("_daily_fallback_rank", "?")
                df_total    = df_combined.get("_total_daily_dips", 0)
                df_cmo_1d   = df_combined.get("_cmo_1d", 0.0)
                df_pos_1d   = df_combined.get("_pos_1d", 0.5)
                # Exhaustion profile for daily fallback
                df_exh = compute_exhaustion_profile(trader, df_symbol)
                df_combined["_exh"] = df_exh
                print(f"\n  🌅  Daily fallback: {df_symbol}  "
                      f"score={df_score:.4f}  "
                      f"resonance={df_exh.get('resonance_score',0)*100:.1f}%")
                klines_1m = trader.get_klines(df_symbol, "1m", limit=1200, return_raw=True)
                if klines_1m:
                    current_price = float(klines_1m[-1][4])
                    sr = get_sr_targets(klines_1m, current_price)
                    tf_volumes = {
                        "5m":  get_volume_breakdown(trader, df_symbol, "5m"),
                        "15m": get_volume_breakdown(trader, df_symbol, "15m"),
                        "4h":  get_volume_breakdown(trader, df_symbol, "4h"),
                        "1d":  get_volume_breakdown(trader, df_symbol, "1d"),
                    }
                    verdict, signal_count = format_sr_output(
                        df_symbol, sr, current_price, df_cmo, df_vratio,
                        df_bull, df_ml, tf_volumes, df_combined
                    )
                    print(f"\n✅ Bot completed (daily best-dip fallback). Exiting...")
                    return
                else:
                    print(f"❌ Could not fetch 1m klines for {df_symbol}. Resetting...")
                    scan_count = 0
                    time.sleep(5)
                    continue

            gc.collect()

        except KeyboardInterrupt:
            print("\n\n⚠️  Scan interrupted by user. Exiting...")
            return
        except Exception as e:
            print(f"\n❌ Error in scan: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
