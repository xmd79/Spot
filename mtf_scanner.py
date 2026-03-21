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


def full_fft_report(pair, current_price):
    """
    Run FFT analysis on 1m, 3m, 5m.
    Returns list of per-TF result dicts + best_overall dict.
    """
    tfs = [
        ('1m',  '1m'),
        ('3m',  '3m'),
        ('5m',  '5m'),
    ]
    tf_results = []

    for label, interval in tfs:
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
            tf_results.append(result)

    if not tf_results:
        return [], None

    # ── best overall: volume-weighted average of per-TF forecasts
    #    weight each TF by its volume-resistance confidence
    #    (higher res_volume = stronger signal)
    forecasts = np.array([r['forecast'] for r in tf_results])
    weights   = np.array([
        r['res_volume'] if r['res_volume'] > 0 else 1.0
        for r in tf_results
    ], dtype=np.float64)

    best_forecast  = float(np.average(forecasts, weights=weights))
    best_upside    = (best_forecast - current_price) / current_price * 100.0

    # consensus confidence: inverse of spread between TF forecasts
    spread      = float(np.std(forecasts) / best_forecast * 100) if best_forecast > 0 else 0
    confidence  = round(max(0.0, min(100.0, 100.0 - spread * 8)), 1)

    best_overall = {
        'current':    current_price,
        'forecast':   round(best_forecast, 8),
        'upside_pct': round(best_upside, 4),
        'confidence': confidence,
        'spread_pct': round(spread, 4),
    }

    return tf_results, best_overall

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

def print_fft_report(pair, label_map, tf_results, best_overall):
    lbl = label_map.get(pair, pair.replace('USDC', ''))
    w   = 62

    print(f'\n  {"═"*w}')
    print(f'  ◈  FFT SPIKE FORECAST  ·  {lbl}  ({pair})')
    print(f'  {"═"*w}')
    print(f'  Entry price : {best_overall["current"]}')
    print()

    # per-TF breakdown
    for r in tf_results:
        has_res = r['res_target'] is not None
        print(f'  ┌─ [{r["tf"]}] ──────────────────────────────────────────────┐')
        print(f'  │  Dominant cycle  : {r["dominant_period"]} bars')
        print(f'  │  Oscillation amp : {r["osc_amplitude"]}')
        print(f'  │  FFT projection  : {r["fft_target"]}')
        if has_res:
            print(f'  │  Vol resistance  : {r["res_target"]}  '
                  f'(vol weight {r["res_volume"]:.0f})')
        else:
            print(f'  │  Vol resistance  : none found above entry')
        print(f'  │  ── [{r["tf"]}] Forecast ──────────────────────────────────')
        print(f'  │  Price target    : {r["forecast"]}')
        print(f'  │  Upside          : +{r["upside_pct"]} %')
        if has_res:
            print(f'  │  Blend           : 60% vol-res + 40% FFT')
        else:
            print(f'  │  Blend           : 100% FFT  (no resistance found)')
        print(f'  └{"─"*58}┘')
        print()

    # best overall
    print(f'  {"═"*w}')
    print(f'  ★  BEST OVERALL FORECAST')
    print(f'  {"─"*w}')
    print(f'  Consensus target : {best_overall["forecast"]}')
    print(f'  Upside           : +{best_overall["upside_pct"]} %')
    print(f'  Confidence       : {best_overall["confidence"]} %  '
          f'(TF spread {best_overall["spread_pct"]} %)')
    print(f'  Method           : volume-weighted average across 1m/3m/5m forecasts')
    print(f'  {"═"*w}\n')

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
        print(f'  Running FFT forecast on '
              f'{label_map.get(best_symbol, best_symbol)}...')
        tf_results, best_overall = full_fft_report(best_symbol, current_price)
        if tf_results:
            print_fft_report(best_symbol, label_map, tf_results, best_overall)
        else:
            print('  FFT: insufficient data for forecast.\n')
    else:
        print('  Could not fetch current price for FFT.\n')

    sys.exit(0)
