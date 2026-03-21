from binance.client import Client
import numpy as np
import talib as ta
import sys, gc, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CREDENTIALS_FILE = 'credentials.txt'
MAX_WORKERS      = 12
LOOP_SLEEP       = 5
FFT_CANDLES      = 200    # more candles → more reliable FFT frequency resolution

# ─────────────────────────────────────────────
#  CLIENT
# ─────────────────────────────────────────────
class Trader:
    def __init__(self, file):
        lines = [l.rstrip('\n') for l in open(file)]
        self.client = Client(lines[0], lines[1])

    def get_usdc_pairs(self):
        """
        Returns:
          pairs      — list of raw Binance symbols e.g. ['1000BONKUSDC', 'BTCUSDC']
          label_map  — {symbol: official_base_asset_ticker}
                       Uses Binance's own baseAsset field — always the official
                       coin abbreviation regardless of numeric prefix.
                       e.g. 1000BONKUSDC → baseAsset = 'BONK'  (official ticker)
                            BTCUSDC      → baseAsset = 'BTC'
        """
        info      = self.client.get_exchange_info()
        pairs     = []
        label_map = {}
        for s in info['symbols']:
            if s['quoteAsset'] == 'USDC' and s['status'] == 'TRADING':
                sym  = s['symbol']       # e.g. 1000BONKUSDC
                base = s['baseAsset']    # e.g. BONK  ← official name from Binance
                pairs.append(sym)
                label_map[sym] = base
        return pairs, label_map

trader = Trader(CREDENTIALS_FILE)

# ─────────────────────────────────────────────
#  ORIGINAL FILTER FUNCTIONS — LOGIC UNCHANGED
# ─────────────────────────────────────────────

def _trendline_pass(klines):
    """Exact original logic. Returns (passed, close_list)."""
    close = [float(e[4]) for e in klines]
    if not close:
        return False, []
    x = close
    y = range(len(x))
    fit  = np.poly1d(np.polyfit(y, x, 1))(y)
    lo   = fit * 0.99
    return x[-1] < lo[-1], close

def filter1(pair, out, lock):
    klines = trader.client.get_klines(symbol=pair, interval='2h')
    passed, _ = _trendline_pass(klines)
    if passed:
        with lock: out.append(pair)

def filter2(pair, out, lock):
    klines = trader.client.get_klines(symbol=pair, interval='15m')
    passed, _ = _trendline_pass(klines)
    if passed:
        with lock: out.append(pair)

def filter3(pair, out, lock):
    klines = trader.client.get_klines(symbol=pair, interval='5m')
    passed, _ = _trendline_pass(klines)
    if passed:
        with lock: out.append(pair)

def compute_vwcmo(klines):
    """
    Volume-Weighted CMO — merges CMO depth with bullish volume conviction.

    raw_cmo    = CMO(14) on 1m close prices        range: -100 to +100
    bull_ratio = bull_vol / total_vol               range:  0.0 to  1.0
    multiplier = (2 * bull_ratio - 1)              range: -1.0 to +1.0
      bull_ratio=1.0  → multiplier=+1.0  → vwcmo = raw_cmo  (full depth, 100% conviction)
      bull_ratio=0.75 → multiplier=+0.5  → vwcmo = raw_cmo/2 (moderate conviction)
      bull_ratio=0.5  → multiplier= 0.0  → vwcmo = 0        (no conviction, fails)
      bull_ratio=0.25 → multiplier=-0.5  → vwcmo = positive  (bear dominant, fails hard)

    vwcmo < -50 means: deep oversold AND meaningful buying activity starting.
    Returns (vwcmo, raw_cmo, bull_pct, bear_pct) or (None, None, None, None).
    """
    close = [float(e[4]) for e in klines]
    if not close:
        return None, None, None, None

    raw_arr = ta.CMO(np.asarray(close, dtype=np.float64), timeperiod=14)
    raw_cmo = float(raw_arr[-1])
    if np.isnan(raw_cmo):
        return None, None, None, None

    bull_vol = sum(float(k[5]) for k in klines if float(k[4]) >= float(k[1]))
    total_vol = sum(float(k[5]) for k in klines)
    if total_vol == 0:
        return None, None, None, None

    bull_ratio = bull_vol / total_vol
    bear_ratio = 1.0 - bull_ratio
    multiplier = 2.0 * bull_ratio - 1.0   # +1 full bull, 0 neutral, -1 full bear
    vwcmo      = raw_cmo * multiplier

    return (round(vwcmo, 2),
            round(raw_cmo, 2),
            round(bull_ratio * 100.0, 1),
            round(bear_ratio * 100.0, 1))


VWCMO_GATE = -50   # same threshold — now volume-scaled

def momentum(pair, sel_pairs, sel_cmo, sel_detail, lock):
    klines = trader.client.get_klines(symbol=pair, interval='1m')
    if not klines:
        return
    vwcmo, raw_cmo, bull_pct, bear_pct = compute_vwcmo(klines)
    if vwcmo is None:
        return
    detail = {
        'vwcmo':    vwcmo,
        'raw_cmo':  raw_cmo,
        'bull_pct': bull_pct,
        'bear_pct': bear_pct,
    }
    if vwcmo < VWCMO_GATE:
        with lock:
            sel_pairs.append(pair)
            sel_cmo.append(vwcmo)       # rank by VWCMO (most negative = best)
            sel_detail[pair] = detail
    else:
        # store detail even for failures so diagnostic can show why
        with lock:
            sel_detail[pair] = detail

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
#  Consistent, realistic approach:
#
#  1. Detrend close prices (remove linear drift)
#  2. FFT → find dominant oscillation period
#  3. Reconstruct only the dominant + first 3
#     harmonics to get a clean cycle projection
#  4. Project one full cycle forward → FFT target
#  5. Volume-profile resistance: bin all highs
#     above current price weighted by volume →
#     nearest high-volume cluster = resistance
#  6. Final forecast = weighted blend of FFT
#     projection and nearest volume resistance
#     (volume resistance gets 60% weight as it
#     is the harder price ceiling)
# ─────────────────────────────────────────────

def _detrend(arr):
    t   = np.arange(len(arr))
    p   = np.polyfit(t, arr, 1)
    return arr - np.poly1d(p)(t), p

def fft_analysis(close_list, volume_list, high_list, current_price, tf_label,
                 sanity_cap_pct=25.0):
    """
    Each TF uses ONLY its own kline data for every calculation:
      - close/volume/high arrays are this TF's own candles
      - volume-resistance bins use this TF's own mean price as bin scalar
        (not current_price, which is always the 1m last close)
      - dominant cycle, amplitude, phase — all from this TF's own FFT

    sanity_cap_pct: max % above current_price allowed for this TF group.
    """
    n = min(FFT_CANDLES, len(close_list))
    if n < 32:
        return None

    # ── own TF data arrays ───────────────────────────────────
    close  = np.array(close_list[-n:], dtype=np.float64)
    volume = np.array(volume_list[-n:], dtype=np.float64)
    high   = np.array(high_list[-n:],  dtype=np.float64)

    # TF-own price reference for binning — median of this TF's closes
    # This ensures each TF bins resistance relative to its own price scale,
    # not the 1m spot price which may differ slightly due to timing
    tf_price_ref = float(np.median(close))

    # ── 1. detrend using this TF's own close series ──────────
    detrended, trend_coeffs = _detrend(close)

    # ── 2. FFT on this TF's detrended series ─────────────────
    spectrum = np.fft.rfft(detrended)
    freqs    = np.fft.rfftfreq(n)
    power    = np.abs(spectrum)
    power[0] = 0   # kill DC component

    # reject periods shorter than 4 bars (noise floor per TF)
    min_period   = 4
    valid_mask   = (freqs > 0) & (freqs <= 1.0 / min_period)
    if not np.any(valid_mask):
        return None
    masked_power              = power.copy()
    masked_power[~valid_mask] = 0

    dom_idx        = int(np.argmax(masked_power))
    dom_freq       = freqs[dom_idx]
    dominant_period = int(round(1.0 / dom_freq)) if dom_freq > 0 else n
    # cap period at half the data window — longer periods are unreliable
    dominant_period = min(dominant_period, n // 2)

    # ── 3. reconstruct: dominant + 3 strongest harmonics ─────
    top_indices              = np.argsort(masked_power)[-4:]
    clean_spec               = np.zeros_like(spectrum)
    clean_spec[top_indices]  = spectrum[top_indices]
    reconstructed            = np.fft.irfft(clean_spec, n=n)

    # ── 4. phase-aware projection forward ────────────────────
    trend_at_end   = float(np.poly1d(trend_coeffs)(n - 1))
    trend_slope    = float(trend_coeffs[0])
    trend_forward  = trend_at_end + trend_slope * dominant_period

    osc_amplitude  = float(np.max(reconstructed) - np.min(reconstructed)) / 2.0
    osc_now        = float(reconstructed[-1])
    osc_mean       = float(np.mean(reconstructed))

    if osc_now < osc_mean:
        # price at trough in this TF's cycle → project full amplitude up
        osc_contribution = osc_amplitude + abs(osc_now - osc_mean)
    else:
        # price mid-cycle → project remaining half
        osc_contribution = osc_amplitude * 0.5

    fft_target = trend_forward + osc_contribution
    fft_target = max(fft_target, current_price * 1.0001)  # must be above entry

    # ── 5. volume-profile resistance — using THIS TF's own data ──
    # Bin width = 0.3% of this TF's own median price (not current_price)
    # This means each TF has independently scaled bins matching its own
    # candle range, so 1m bins are narrow and 2h bins are appropriately wide
    BIN_PCT = 0.003                          # 0.3% bucket width
    bin_size = tf_price_ref * BIN_PCT        # absolute width in price units
    bins     = {}
    for h, v in zip(high, volume):
        if h > current_price * 1.001:        # only levels above entry
            # snap h to nearest bin boundary using this TF's own bin_size
            b = round(h / bin_size) * bin_size
            bins[b] = bins.get(b, 0.0) + float(v)

    res_target = None
    res_volume = 0.0
    if bins:
        # qualify: only bins in top 30% by volume weight (high-conviction walls)
        vol_threshold = float(np.percentile(list(bins.values()), 70))
        candidates    = {
            k: v for k, v in bins.items()
            if v >= vol_threshold and k > current_price
        }
        if candidates:
            # nearest qualified resistance above entry
            res_target = float(min(candidates.keys()))
            res_volume = float(candidates[res_target])

    # ── 6. blend: resistance is the harder ceiling ────────────
    if res_target and res_target > current_price:
        # if FFT projects past resistance, cap at resistance (can't ignore wall)
        if fft_target > res_target:
            # strong wall — blend pulls forecast toward resistance
            forecast = res_target * 0.65 + fft_target * 0.35
        else:
            # FFT is below resistance — blend toward FFT (still room to move)
            forecast = fft_target * 0.60 + res_target * 0.40
    else:
        forecast = fft_target

    # sanity cap and floor
    forecast = min(forecast, current_price * (1.0 + sanity_cap_pct / 100.0))
    forecast = max(forecast, current_price * 1.0001)

    upside_pct = (forecast - current_price) / current_price * 100.0

    return {
        'tf':             tf_label,
        'dominant_period': dominant_period,
        'osc_amplitude':  round(osc_amplitude, 8),
        'fft_target':     round(fft_target, 8),
        'res_target':     round(res_target, 8) if res_target else None,
        'res_volume':     round(res_volume, 2),
        'forecast':       round(forecast, 8),
        'upside_pct':     round(upside_pct, 4),
        'cascade_stop':   False,   # set by caller if needed
    }


def _run_fft_for_tfs(pair, current_price, tf_list, sanity_cap_pct=25.0):
    """
    Fetch each TF's own klines independently and run fft_analysis.
    Each TF gets its own close/volume/high arrays — no shared data.
    sanity_cap_pct is passed into fft_analysis so the cap is applied
    once inside the function using that TF's own data, not overridden here.
    """
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
        # each TF's own independent data arrays
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
        'upside_pct': round(best_upside, 4),
        'confidence': confidence,
        'spread_pct': round(spread, 4),
    }
    return tf_results, best_overall


def full_fft_report(pair, current_price):
    """
    Short-term  : 1m, 3m, 5m         — cap +25% each TF, own data each
    Higher-TF   : 15m, 30m, 1h, 2h   — cap +60% each TF, own data each
                  cascade stops only when:
                    res_target exists AND
                    res_target is meaningfully above entry (>1.5%) AND
                    fft_target reaches or exceeds res_target
                  → prevents stopping at trivial nearby resistance

    Every TF fetches its own independent klines — no shared arrays.
    Returns (stf_results, stf_best, htf_results, htf_best).
    """
    stf_tfs = [('1m', '1m'), ('3m', '3m'), ('5m', '5m')]
    htf_tfs = [('15m', '15m'), ('30m', '30m'), ('1h', '1h'), ('2h', '2h')]

    # ── short-term: each TF independent, cap +25% ────────────
    stf_results, stf_best = _run_fft_for_tfs(
        pair, current_price, stf_tfs, sanity_cap_pct=25.0
    )

    # ── HTF cascade: each TF fully independent ───────────────
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

        # this TF's own independent arrays
        close  = [float(k[4]) for k in klines]
        volume = [float(k[5]) for k in klines]
        high   = [float(k[2]) for k in klines]

        result = fft_analysis(close, volume, high, current_price,
                              label, sanity_cap_pct=60.0)
        if not result:
            continue

        result['cascade_stop'] = False
        htf_results.append(result)

        # cascade stop condition (corrected):
        # 1. resistance must exist on this TF
        # 2. resistance must be meaningfully above entry (>1.5%) — not a trivial wall
        # 3. FFT projection must reach or exceed the resistance level
        # All three must be true to stop — prevents premature stops on tiny walls
        if (result['res_target'] is not None
                and result['res_target'] > current_price * 1.015
                and result['fft_target'] >= result['res_target']):
            result['cascade_stop'] = True
            break   # no point projecting further — this wall will reject price

    # ── HTF best overall ─────────────────────────────────────
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
            'upside_pct': round(htf_best_upside, 4),
            'confidence': confidence,
            'spread_pct': round(spread, 4),
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

        bull = sum(float(k[5]) for k in k1 if float(k[4]) >= float(k[1]))
        tot  = sum(float(k[5]) for k in k1)
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

    # compute concurrently
    data    = {}
    d_lock  = threading.Lock()

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
        lbl  = label_map.get(p, p.replace('USDC', ''))
        pr_s = (f'{pr:.6f}' if pr and pr < 1 else f'{pr:.4f}') if pr else '—'
        cmo_s = f'{cmo:.2f}' if cmo is not None else '—'
        print(f'  │  {i:>3}  {lbl:<10}  {pr_s:>13}  {sc:>9.1f}  {cmo_s:>8}  │')

    print(sep + '\n')

# ─────────────────────────────────────────────
#  FFT REPORT PRINTER
# ─────────────────────────────────────────────

def _print_tf_block(r):
    """Print one timeframe block, shared by STF and HTF sections."""
    has_res  = r['res_target'] is not None
    stop_tag = '  ◄ CASCADE STOP (resistance reached)' \
               if r.get('cascade_stop') else ''
    print(f'  ┌─ [{r["tf"]}] {"─"*52}┐')
    print(f'  │  Dominant cycle  : {r["dominant_period"]} bars')
    print(f'  │  Oscillation amp : {r["osc_amplitude"]}')
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

    # ── SHORT-TERM: 1m / 3m / 5m ────────────────────────────
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

    # ── MEDIUM/LONGER-TERM: 15m / 30m / 1h / 2h ────────────
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
#  MAIN LOOP
# ─────────────────────────────────────────────

print(f'\n  MTF Dip Scanner + FFT Forecast')
print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'  All USDC pairs  |  {MAX_WORKERS} threads  |  retry {LOOP_SLEEP}s')
print(f'  Ticker names: official Binance baseAsset field (no invented labels)\n')

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

    # ── 15m ─────────────────────────────────────────────────
    fp2 = run_stage(filter2, fp1, '15m')
    print(f'  15m → {len(fp2)} passed')
    print_stage_table(fp2, label_map, '15m filter', show_cmo=True)

    if not fp2:
        del fp1, fp2; gc.collect()
        print(f'  Nothing passed 15m. Retry in {LOOP_SLEEP}s\n')
        time.sleep(LOOP_SLEEP); continue

    # ── 5m ──────────────────────────────────────────────────
    fp3 = run_stage(filter3, fp2, '5m ')
    print(f'  5m  → {len(fp3)} passed')
    print_stage_table(fp3, label_map, '5m filter', show_cmo=True)

    if not fp3:
        del fp1, fp2, fp3; gc.collect()
        print(f'  Nothing passed 5m. Retry in {LOOP_SLEEP}s\n')
        time.sleep(LOOP_SLEEP); continue

    # ── 1m VWCMO gate ────────────────────────────────────────
    sel_pairs  = []
    sel_cmo    = []
    sel_detail = {}     # {symbol: {vwcmo, raw_cmo, bull_pct, bear_pct}}
    sel_lock   = threading.Lock()
    total_1m   = len(fp3)
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
        fs = [pool.submit(_mom, s) for s in fp3]
        for f in as_completed(fs): pass
    print()

    # ── full diagnostic: show every candidate with VWCMO breakdown ──
    print(f'\n  [1m VWCMO diagnostic]  {len(fp3)} candidates'
          f'  (gate: VWCMO < {VWCMO_GATE})\n')
    diag_header = (f'  {"Ticker":<12}  {"Price":>10}  '
                   f'{"RawCMO":>8}  {"Bull%":>6}  {"Bear%":>6}  '
                   f'{"VWCMO":>8}  {"Result":<22}')
    print(diag_header)
    print('  ' + '─' * (len(diag_header) - 2))
    for p in fp3:
        d    = sel_detail.get(p, {})
        lbl  = label_map.get(p, p.replace('USDC', ''))
        _, _, pr_v = spike_score_and_cmo(p)
        pr_s = (f'{pr_v:.6f}' if pr_v and pr_v < 1 else f'{pr_v:.4f}') \
               if pr_v else '—'
        raw  = d.get('raw_cmo')
        bull = d.get('bull_pct')
        bear = d.get('bear_pct')
        vw   = d.get('vwcmo')
        if vw is None:
            result = 'no data'
        elif vw < VWCMO_GATE:
            result = f'PASS ✔  (VWCMO={vw})'
        else:
            # explain precisely why it failed
            if raw is not None and raw < VWCMO_GATE:
                reason = f'CMO deep ({raw}) but bear vol {bear}% kills it'
            elif raw is not None and raw >= VWCMO_GATE and bull is not None and bull > 50:
                reason = f'bull vol ok ({bull}%) but CMO shallow ({raw})'
            elif bull is not None and bull <= 50:
                reason = f'bear vol dominant ({bear}%)'
            else:
                reason = f'insufficient depth+conviction'
            result = f'fail  {reason}'
        raw_s  = f'{raw:.1f}'  if raw  is not None else '—'
        bull_s = f'{bull:.1f}' if bull is not None else '—'
        bear_s = f'{bear:.1f}' if bear is not None else '—'
        vw_s   = f'{vw:.2f}'   if vw   is not None else '—'
        print(f'  {lbl:<12}  {pr_s:>10}  '
              f'{raw_s:>8}  {bull_s:>6}  {bear_s:>6}  '
              f'{vw_s:>8}  {result}')
    print()

    print(f'  1m  → {len(sel_pairs)} passed VWCMO < {VWCMO_GATE}')
    print_stage_table(sel_pairs, label_map,
                      f'1m VWCMO gate (confirmed dips)', show_cmo=True)

    # ── selection: lowest VWCMO = deepest oversold + most bullish vol ──
    if len(sel_pairs) > 1:
        print(f'  more mtf dips found: '
              f'{[label_map.get(p, p) for p in sel_pairs]}')
        position    = sel_cmo.index(min(sel_cmo))
        best_symbol = sel_pairs[position]
        best_d      = sel_detail[best_symbol]
        print(f'  Best (lowest VWCMO {sel_cmo[position]:.2f}  '
              f'RawCMO={best_d["raw_cmo"]}  '
              f'Bull={best_d["bull_pct"]}%): '
              f'{label_map.get(best_symbol, best_symbol)}\n')

    elif len(sel_pairs) == 1:
        best_symbol = sel_pairs[0]
        best_d      = sel_detail[best_symbol]
        print(f'  1 mtf dip found: {label_map.get(best_symbol, best_symbol)}  '
              f'VWCMO={sel_cmo[0]}  '
              f'RawCMO={best_d["raw_cmo"]}  '
              f'Bull={best_d["bull_pct"]}%\n')

    else:
        print(f'  No MTF dips (VWCMO < {VWCMO_GATE}) found.')
        del fp1, fp2, fp3, sel_pairs, sel_cmo, sel_detail
        gc.collect()
        print(f'  GC done. Retry in {LOOP_SLEEP}s\n')
        print(f'  {"·" * 62}\n')
        time.sleep(LOOP_SLEEP); continue

    # ── FFT forecast on winner ────────────────────────────────
    _, _, current_price = spike_score_and_cmo(best_symbol)
    if current_price:
        lbl = label_map.get(best_symbol, best_symbol)
        print(f'  Running FFT forecast on {lbl}...')
        stf_results, stf_best, htf_results, htf_best = \
            full_fft_report(best_symbol, current_price)
        if stf_results or htf_results:
            print_fft_report(best_symbol, label_map,
                             stf_results, stf_best,
                             htf_results, htf_best)
        else:
            print('  FFT: insufficient data for forecast.\n')
    else:
        print('  Could not fetch current price for FFT.\n')

    del fp1, fp2, fp3, sel_pairs, sel_cmo, sel_detail
    gc.collect()
    sys.exit(0)