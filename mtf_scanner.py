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

def momentum(pair, sel_pairs, sel_cmo, lock):
    klines = trader.client.get_klines(symbol=pair, interval='1m')
    close  = [float(e[4]) for e in klines]
    if not close:
        return
    real = ta.CMO(np.asarray(close, dtype=np.float64), timeperiod=14)
    if real[-1] < -50:
        with lock:
            sel_pairs.append(pair)
            sel_cmo.append(real[-1])

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

def fft_analysis(close_list, volume_list, high_list, current_price, tf_label):
    """
    Returns dict with:
      dominant_period  — bars for one full cycle
      fft_target       — price projected one cycle forward via FFT
      res_target       — nearest volume-resistance level above price
      res_volume       — total volume at that resistance bin
      forecast         — blended target (60% res + 40% fft if res exists,
                         else pure fft)
      upside_pct       — % from current to forecast
    """
    n = min(FFT_CANDLES, len(close_list))
    if n < 32:
        return None   # not enough data for meaningful FFT

    close  = np.array(close_list[-n:], dtype=np.float64)
    volume = np.array(volume_list[-n:], dtype=np.float64)
    high   = np.array(high_list[-n:],  dtype=np.float64)

    # ── 1. detrend
    detrended, trend_coeffs = _detrend(close)

    # ── 2. FFT
    spectrum = np.fft.rfft(detrended)
    freqs    = np.fft.rfftfreq(n)
    power    = np.abs(spectrum)
    power[0] = 0   # kill DC

    # dominant frequency (skip freq bins that imply period < 4 bars — noise)
    valid_mask           = freqs > (1.0 / (n / 2))
    valid_mask[0]        = False
    if not np.any(valid_mask):
        return None
    masked_power         = power.copy()
    masked_power[~valid_mask] = 0
    dom_idx              = int(np.argmax(masked_power))
    dom_freq             = freqs[dom_idx]
    dominant_period      = int(round(1.0 / dom_freq)) if dom_freq > 0 else n

    # ── 3. reconstruct using dominant + top 3 harmonics only
    #       (removes noise, keeps the true oscillation shape)
    top_indices = np.argsort(masked_power)[-4:]   # 4 strongest components
    clean_spectrum          = np.zeros_like(spectrum)
    clean_spectrum[top_indices] = spectrum[top_indices]
    reconstructed           = np.fft.irfft(clean_spectrum, n=n)

    # ── 4. project one dominant period forward
    #       extrapolate the linear trend + the last reconstructed value
    #       shifted forward by dominant_period bars
    trend_at_end     = np.poly1d(trend_coeffs)(n - 1)
    trend_slope      = trend_coeffs[0]
    trend_forward    = trend_at_end + trend_slope * dominant_period

    # oscillation contribution: amplitude of reconstructed at end
    osc_amplitude    = float(np.max(reconstructed) - np.min(reconstructed)) / 2.0
    # current reconstructed phase: if we are at a trough (< mean), add amplitude
    osc_now          = float(reconstructed[-1])
    osc_mean         = float(np.mean(reconstructed))
    if osc_now < osc_mean:
        # at trough → project to peak
        osc_contribution = osc_amplitude - osc_now
    else:
        # mid-cycle → project half amplitude
        osc_contribution = osc_amplitude * 0.5

    fft_target = max(current_price, trend_forward + osc_contribution)

    # ── 5. volume-profile resistance
    #       bin highs into 0.3% price buckets, weight by volume
    #       find the nearest cluster above current price
    BIN_W = 0.003
    bins  = {}
    for h, v in zip(high, volume):
        if h > current_price * 1.001:  # at least 0.1% above entry
            b = round(h / (current_price * BIN_W)) * (current_price * BIN_W)
            bins[b] = bins.get(b, 0.0) + float(v)

    res_target = None
    res_volume = 0.0
    if bins:
        # nearest level with meaningful volume (top 30% by volume weight)
        vol_threshold = np.percentile(list(bins.values()), 70)
        candidates    = {k: v for k, v in bins.items()
                         if v >= vol_threshold and k > current_price}
        if candidates:
            res_target = float(min(candidates.keys()))  # nearest
            res_volume = float(candidates[res_target])

    # ── 6. blend
    if res_target and res_target > current_price:
        # volume resistance is the harder ceiling — weight it more
        forecast = res_target * 0.60 + fft_target * 0.40
    else:
        forecast = fft_target

    # sanity cap: never forecast more than +25% on short TFs
    forecast = min(forecast, current_price * 1.25)
    # never forecast below entry
    forecast = max(forecast, current_price * 1.0001)

    upside_pct = (forecast - current_price) / current_price * 100.0

    return {
        'tf':               tf_label,
        'dominant_period':  dominant_period,
        'osc_amplitude':    round(osc_amplitude, 8),
        'fft_target':       round(fft_target, 8),
        'res_target':       round(res_target, 8) if res_target else None,
        'res_volume':       round(res_volume, 2),
        'forecast':         round(forecast, 8),
        'upside_pct':       round(upside_pct, 4),
    }


def _run_fft_for_tfs(pair, current_price, tf_list, sanity_cap_pct=25.0):
    """
    Generic: run fft_analysis for each (label, interval) in tf_list.
    sanity_cap_pct controls the max % forecast above current_price.
    Returns (tf_results, best_overall) or ([], None).
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
        close  = [float(k[4]) for k in klines]
        volume = [float(k[5]) for k in klines]
        high   = [float(k[2]) for k in klines]
        result = fft_analysis(close, volume, high, current_price, label)
        if result:
            # re-apply cap with the per-group sanity limit
            result['forecast']   = min(result['forecast'],
                                       current_price * (1 + sanity_cap_pct / 100))
            result['upside_pct'] = round(
                (result['forecast'] - current_price) / current_price * 100.0, 4)
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
    Short-term  : 1m, 3m, 5m        (cap +25%)
    Medium-term : 15m, 30m, 1h, 2h  (cap +60%, cascade stops at resistance)

    For HTF cascade: we walk 15m→30m→1h→2h and stop projecting further
    once a resistance level is hit (i.e. fft_target is capped by res_target).
    Returns (stf_results, stf_best, htf_results, htf_best).
    """
    stf_tfs = [('1m', '1m'), ('3m', '3m'), ('5m', '5m')]
    htf_tfs = [('15m', '15m'), ('30m', '30m'), ('1h', '1h'), ('2h', '2h')]

    stf_results, stf_best = _run_fft_for_tfs(pair, current_price,
                                              stf_tfs, sanity_cap_pct=25.0)

    # HTF cascade: stop at first TF where resistance is hit
    # "resistance hit" = res_target exists AND fft_target >= res_target
    # meaning price would reach and test that wall within this TF's cycle
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
        result = fft_analysis(close, volume, high, current_price, label)
        if not result:
            continue

        # apply 60% cap for HTF
        result['forecast']   = min(result['forecast'],
                                   current_price * 1.60)
        result['upside_pct'] = round(
            (result['forecast'] - current_price) / current_price * 100.0, 4)

        htf_results.append(result)

        # cascade stop: resistance exists and FFT projection reaches/exceeds it
        # → this is where price will face real selling pressure; no point going higher
        if (result['res_target'] is not None
                and result['fft_target'] >= result['res_target'] * 0.98):
            result['cascade_stop'] = True
            break
        result['cascade_stop'] = False

    # compute HTF best overall
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
        htf_best = {
            'current':    current_price,
            'forecast':   round(htf_best_forecast, 8),
            'upside_pct': round(htf_best_upside, 4),
            'confidence': confidence,
            'spread_pct': round(spread, 4),
            'tfs_used':   len(htf_results),
            'stopped_at': htf_results[-1]['tf'] if htf_results else '—',
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

    # ── 1m CMO gate ──────────────────────────────────────────
    sel_pairs = []
    sel_cmo   = []
    sel_lock  = threading.Lock()
    total_1m  = len(fp3)
    done_1m   = [0]

    def _mom(sym):
        momentum(sym, sel_pairs, sel_cmo, sel_lock)
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

    # always show 5m survivors with their actual CMO — diagnostic
    print(f'\n  [1m CMO check]  {len(fp3)} candidates:')
    for p in fp3:
        _, cmo_v, pr_v = spike_score_and_cmo(p)
        status = 'PASS ✔' if p in sel_pairs else f'fail  (CMO={cmo_v}, need <-50)'
        print(f'    {label_map.get(p,p):<12}  price={pr_v}  {status}')
    print()

    print(f'  1m  → {len(sel_pairs)} passed CMO < -50')
    print_stage_table(sel_pairs, label_map, '1m CMO gate — confirmed dips', show_cmo=True)

    # ── original selection logic ──────────────────────────────
    if len(sel_pairs) > 1:
        print(f'  more mtf dips are found: '
              f'{[label_map.get(p,p) for p in sel_pairs]}')
        position    = sel_cmo.index(min(sel_cmo))
        best_symbol = sel_pairs[position]
        print(f'  Best (lowest CMO {sel_cmo[position]:.2f}): '
              f'{label_map.get(best_symbol, best_symbol)}\n')

    elif len(sel_pairs) == 1:
        best_symbol = sel_pairs[0]
        print(f'  1 mtf dip found: '
              f'{label_map.get(best_symbol, best_symbol)}\n')

    else:
        print(f'  No MTF dips (CMO < -50) found.')
        del fp1, fp2, fp3, sel_pairs, sel_cmo
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

    sys.exit(0)
