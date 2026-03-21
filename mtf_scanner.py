"""
MTF Dip Scanner — CMD output only
Usage: python mtf_scanner.py
"""

from binance.client import Client
import numpy as np
import talib as ta
import sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CREDENTIALS_FILE   = 'credentials.txt'
MAX_WORKERS        = 12
CANDLE_LIMIT       = 500
MIN_CANDLES        = 19      # CMO_PERIOD(14) + 5 safety margin
SLOPE_THRESHOLD_2H = -0.002  # 2h trendline slope must be > this
DEV_BAND_2H        = 0.010   # price must be >1.0% below 2h trendline
DEV_BAND_15M       = 0.007   # price must be >0.7% below 15m trendline
DEV_BAND_5M        = 0.005   # price must be >0.5% below 5m trendline
CMO_1M_THRESHOLD   = -50     # 1m CMO must be below this (deep oversold)
CMO_HTF_THRESHOLD  = -30     # 15m OR 5m CMO must be below this
CMO_PERIOD         = 14
VOLUME_LOOKBACK    = 20      # bars used for volume average baseline
API_SLEEP          = 0.08    # seconds between API calls per thread

# ─────────────────────────────────────────────
#  BINANCE CLIENT
# ─────────────────────────────────────────────
class Trader:
    def __init__(self, file):
        lines = [l.rstrip('\n') for l in open(file)]
        self.client = Client(lines[0], lines[1])

    def get_usdc_pairs(self):
        info = self.client.get_exchange_info()
        return [
            s['symbol'] for s in info['symbols']
            if s['quoteAsset'] == 'USDC' and s['status'] == 'TRADING'
        ]

    def get_klines(self, symbol, interval):
        try:
            k = self.client.get_klines(symbol=symbol, interval=interval,
                                       limit=CANDLE_LIMIT)
            return k or []
        except Exception:
            return []

# ─────────────────────────────────────────────
#  ANALYSIS HELPERS
# ─────────────────────────────────────────────
def extract(klines):
    close  = np.array([float(k[4]) for k in klines], dtype=np.float64)
    volume = np.array([float(k[5]) for k in klines], dtype=np.float64)
    return close, volume

def trendline(close):
    y = np.arange(len(close))
    c = np.polyfit(y, close, 1)
    return c[0], np.poly1d(c)(y[-1])   # (slope, last trendline value)

def below_band(close, dev):
    slope, trend = trendline(close)
    return close[-1] < trend * (1 - dev), slope

def vol_confirmed(volume):
    # new assets may not have enough history — pass them through
    if len(volume) < VOLUME_LOOKBACK + 1:
        return True
    return volume[-1] > np.mean(volume[-(VOLUME_LOOKBACK + 1):-1])

def cmo_last(close):
    if len(close) < CMO_PERIOD + 1:
        return None
    r = ta.CMO(close, timeperiod=CMO_PERIOD)
    v = r[-1]
    return None if np.isnan(v) else float(v)

# ─────────────────────────────────────────────
#  PER-SYMBOL SCAN  (runs inside thread pool)
# ─────────────────────────────────────────────
_lock     = threading.Lock()
_progress = [0]
_results  = []

def scan_symbol(trader, symbol, total):
    r = dict(
        symbol=symbol, price=None, slope_2h=None,
        passed_2h=False, passed_15m=False, passed_5m=False,
        vol_ok=False, htf_cmo_ok=False,
        cmo_15m=None, cmo_5m=None, cmo_1m=None,
        selected=False, reason=''
    )

    try:
        # ── 2h filter ───────────────────────────────────
        time.sleep(API_SLEEP)
        k2h = trader.get_klines(symbol, Client.KLINE_INTERVAL_2HOUR)
        if len(k2h) < MIN_CANDLES:
            r['reason'] = 'insufficient 2h candles'; return r

        c2h, v2h  = extract(k2h)
        passed, slope = below_band(c2h, DEV_BAND_2H)
        r['slope_2h'] = round(slope, 8)
        r['price']    = round(float(c2h[-1]), 8)

        if not passed:
            r['reason'] = 'above 2h band'; return r
        if slope < SLOPE_THRESHOLD_2H:
            r['reason'] = f'slope too negative ({slope:.5f})'; return r
        r['passed_2h'] = True

        # ── volume confirmation ──────────────────────────
        if not vol_confirmed(v2h):
            r['reason'] = 'low volume vs 20-bar avg'; return r
        r['vol_ok'] = True

        # ── 15m filter ──────────────────────────────────
        time.sleep(API_SLEEP)
        k15m = trader.get_klines(symbol, Client.KLINE_INTERVAL_15MINUTE)
        if len(k15m) < MIN_CANDLES:
            r['reason'] = 'insufficient 15m candles'; return r

        c15m, _ = extract(k15m)
        p15, _  = below_band(c15m, DEV_BAND_15M)
        if not p15:
            r['reason'] = 'above 15m band'; return r
        r['passed_15m'] = True

        cmo15 = cmo_last(c15m)
        r['cmo_15m'] = round(cmo15, 2) if cmo15 is not None else None

        # ── 5m filter ───────────────────────────────────
        time.sleep(API_SLEEP)
        k5m = trader.get_klines(symbol, Client.KLINE_INTERVAL_5MINUTE)
        if len(k5m) < MIN_CANDLES:
            r['reason'] = 'insufficient 5m candles'; return r

        c5m, _ = extract(k5m)
        p5, _  = below_band(c5m, DEV_BAND_5M)
        if not p5:
            r['reason'] = 'above 5m band'; return r
        r['passed_5m'] = True

        cmo5 = cmo_last(c5m)
        r['cmo_5m'] = round(cmo5, 2) if cmo5 is not None else None

        # ── HTF CMO gate (15m OR 5m must be oversold) ───
        htf_ok = (
            (cmo15 is not None and cmo15 < CMO_HTF_THRESHOLD) or
            (cmo5  is not None and cmo5  < CMO_HTF_THRESHOLD)
        )
        r['htf_cmo_ok'] = htf_ok
        if not htf_ok:
            r['reason'] = f'HTF CMO weak (15m={cmo15}, 5m={cmo5})'; return r

        # ── 1m momentum (final gate) ─────────────────────
        time.sleep(API_SLEEP)
        k1m = trader.get_klines(symbol, Client.KLINE_INTERVAL_1MINUTE)
        if len(k1m) < MIN_CANDLES:
            r['reason'] = 'insufficient 1m candles'; return r

        c1m, _ = extract(k1m)
        cmo1   = cmo_last(c1m)
        r['cmo_1m'] = round(cmo1, 2) if cmo1 is not None else None

        if cmo1 is None or cmo1 >= CMO_1M_THRESHOLD:
            r['reason'] = f'1m CMO insufficient ({cmo1})'; return r

        r['selected'] = True
        r['reason']   = 'MTF dip confirmed'

    except Exception as e:
        r['reason'] = f'error: {e}'
    finally:
        with _lock:
            _progress[0] += 1
            n   = _progress[0]
            pct = int(n / total * 100)
            bar = '█' * (pct // 4) + '░' * (25 - pct // 4)
            print(f'\r  [{bar}] {pct:3d}%  {n}/{total}  {symbol:<18}',
                  end='', flush=True)

    return r

# ─────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────
def fmt(val, decimals=2, fallback='—'):
    return f'{val:.{decimals}f}' if val is not None else fallback

def tick(b):
    return '✔' if b else '·'

def print_results(rows):
    # column widths
    CW = [16, 14, 11, 5, 5, 5, 5, 9, 9, 9, 7, 26]
    HDR = ['Symbol', 'Price', 'Slope 2h',
           '2h', '15m', '5m', 'Vol',
           'CMO 15m', 'CMO 5m', 'CMO 1m', 'HTF ok', 'Status']

    SEP = '─' * (sum(CW) + len(CW) * 2)

    def row_line(cols):
        return '  ' + '  '.join(str(cols[i]).ljust(CW[i]) for i in range(len(cols)))

    print(SEP)
    print(row_line(HDR))
    print(SEP)

    for r in rows:
        price_str = fmt(r['price'], 6) if r['price'] and r['price'] < 1 \
                    else fmt(r['price'], 4)
        status = ('★ ' if r['selected'] else '') + r['reason'][:24]
        cols = [
            r['symbol'],
            price_str,
            fmt(r['slope_2h'], 6),
            tick(r['passed_2h']),
            tick(r['passed_15m']),
            tick(r['passed_5m']),
            tick(r['vol_ok']),
            fmt(r['cmo_15m']),
            fmt(r['cmo_5m']),
            fmt(r['cmo_1m']),
            tick(r['htf_cmo_ok']),
            status,
        ]
        print(row_line(cols))

    print(SEP)

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print(f'\n  MTF Dip Scanner  ·  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Slope > {SLOPE_THRESHOLD_2H}  |  bands: 2h={DEV_BAND_2H} 15m={DEV_BAND_15M} 5m={DEV_BAND_5M}')
    print(f'  CMO: 1m<{CMO_1M_THRESHOLD}  HTF<{CMO_HTF_THRESHOLD}  |  threads={MAX_WORKERS}\n')

    trader = Trader(CREDENTIALS_FILE)
    pairs  = trader.get_usdc_pairs()
    total  = len(pairs)
    print(f'  {total} USDC pairs found — scanning...\n')

    t0 = __import__('time').time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(scan_symbol, trader, p, total): p for p in pairs}
        for f in as_completed(futures):
            res = f.result()
            with _lock:
                _results.append(res)

    elapsed = __import__('time').time() - t0
    print(f'\n\n  Done in {elapsed:.1f}s\n')

    # dips first sorted by lowest CMO 1m, then remaining by symbol
    dips = sorted([r for r in _results if r['selected']],
                  key=lambda x: x['cmo_1m'] if x['cmo_1m'] is not None else 0)
    rest = sorted([r for r in _results if not r['selected']],
                  key=lambda x: x['symbol'])

    print_results(dips + rest)

    # summary line
    p2h  = sum(1 for r in _results if r['passed_2h'])
    p15m = sum(1 for r in _results if r['passed_15m'])
    p5m  = sum(1 for r in _results if r['passed_5m'])
    print(f'\n  Scanned: {total}  |  Passed 2h: {p2h}  |  15m: {p15m}  |  5m: {p5m}  |  Dips: {len(dips)}')

    # best dip callout
    if dips:
        best = dips[0]
        w = 42
        print(f'\n  {"─"*w}')
        print(f'  ★  BEST DIP : {best["symbol"]}')
        print(f'     Price    : {best["price"]}')
        print(f'     Slope 2h : {best["slope_2h"]}')
        print(f'     CMO 1m   : {best["cmo_1m"]}')
        print(f'     CMO 15m  : {best["cmo_15m"]}')
        print(f'     CMO 5m   : {best["cmo_5m"]}')
        print(f'  {"─"*w}\n')
    else:
        print('\n  No MTF dips found at this time.\n')

    sys.exit(0)

if __name__ == '__main__':
    main()
