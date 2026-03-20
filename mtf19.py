from binance.client import Client
import numpy as np
import sys
import concurrent.futures
import scipy.signal as signal
from scipy.stats import linregress
import requests
from rich.console import Console
from rich.table import Table
from rich import box
from rich.live import Live

# ====================== CONFIG ======================
RSI_LENGTH = 14
MAX_WORKERS = 12
TF_LIST = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d']
VOLUME_LOOKBACK = 20
SCALP_LOOKBACK_CANDLES = 50
EXTREMA_LOOKBACK = 200

# ====================== TELEGRAM (optional) ======================
TELEGRAM_TOKEN = ""
TELEGRAM_CHAT_ID = ""

console = Console()

# ====================== BINANCE CLIENT ======================
class Trader:
    def __init__(self, file):
        lines = [line.rstrip('\n') for line in open(file)]
        self.client = Client(lines[0], lines[1])

    def get_usdc_pairs(self):
        info = self.client.get_exchange_info()
        pairs = []
        for s in info['symbols']:
            if s['quoteAsset'] == 'USDC' and s['status'] == 'TRADING':
                sym = s['symbol']
                if sym.isascii() and sym.isupper():
                    pairs.append(sym)
        return pairs

# ====================== INDICATORS ======================
def get_klines(client, symbol, interval):
    k = client.get_klines(symbol=symbol, interval=interval, limit=500)
    if len(k) < 50:
        return None
    o = np.array([float(x[1]) for x in k])
    h = np.array([float(x[2]) for x in k])
    l = np.array([float(x[3]) for x in k])
    c = np.array([float(x[4]) for x in k])
    v = np.array([float(x[5]) for x in k])
    return o, h, l, c, v

def rsi(c):
    if len(c) < RSI_LENGTH + 1:
        return 50.0
    deltas = np.diff(c)[-RSI_LENGTH:]
    up = np.maximum(deltas, 0)
    down = np.maximum(-deltas, 0)
    avg_up = np.mean(up)
    avg_down = np.mean(down)
    rs = avg_up / (avg_down + 1e-8)
    return 100 - 100 / (1 + rs)

def macd(c):
    ema12 = np.mean(c[-12:])
    ema26 = np.mean(c[-26:])
    return ema12 - ema26

def regression(c):
    x = np.arange(len(c))
    slope, intercept, _, _, _ = linregress(x, c)
    trend = intercept + slope * x
    std = np.std(c - trend)
    return trend, trend - std, trend + std

def fft_forecast(c):
    return np.mean(np.fft.fft(c).real)

def wick_extremes(h, l, lookback=EXTREMA_LOOKBACK):
    if len(h) < lookback:
        return -1, -1, False, 0.0, 0.0, "NEUTRAL"
    
    recent_slice_h = h[-lookback:]
    recent_slice_l = l[-lookback:]
    
    # 1. Absolute Extremes
    max_val = np.max(recent_slice_h)
    min_val = np.min(recent_slice_l)
    
    # 2. Most Recent Occurrence of these extremes
    indices_high = np.where(recent_slice_h == max_val)[0]
    indices_low  = np.where(recent_slice_l == min_val)[0]
    
    idx_high = indices_high[-1] if len(indices_high) > 0 else -1
    idx_low  = indices_low[-1] if len(indices_low) > 0 else -1
    
    price_high = max_val
    price_low = min_val
    
    # 3. Cycle Dominance
    recent_is_low = idx_low > idx_high
    last_event = "LOW_RECENT" if recent_is_low else "HIGH_RECENT"
    
    return idx_high, idx_low, recent_is_low, price_high, price_low, last_event

def volume_bull_percent(c, v, lookback=VOLUME_LOOKBACK):
    if len(c) < lookback + 1 or len(v) < lookback:
        return 50.0
    recent_c = c[-lookback:]
    recent_v = v[-lookback:]
    bull_vol = 0.0
    bear_vol = 0.0
    for i in range(1, len(recent_c)):
        if recent_c[i] > recent_c[i-1]:
            bull_vol += recent_v[i]
        elif recent_c[i] < recent_c[i-1]:
            bear_vol += recent_v[i]
    total = bull_vol + bear_vol
    if total < 1e-8:
        return 50.0
    return (bull_vol / total) * 100

def momentum_short(c, periods=5):
    if len(c) < periods + 1:
        return 0.0
    return c[-1] - c[-periods]

def orderbook(client, symbol):
    try:
        d = client.get_order_book(symbol=symbol, limit=50)
        bids = sum(float(b[1]) for b in d['bids'])
        asks = sum(float(a[1]) for a in d['asks'])
        return (bids - asks) / (bids + asks + 1e-8)
    except:
        return 0.0

# ====================== HARMONIC & ML LOGIC ======================
def harmonic_oscillator(c, v):
    c_clean = c[~np.isnan(c)]
    if len(c_clean) < 2:
        return 0.0, 0.0, 0.0, "NEUTRAL", 0.0, False, 0.0

    normalized = (c_clean - np.mean(c_clean)) / (np.std(c_clean) + 1e-8)
    analytic = signal.hilbert(normalized)
    instantaneous_phase = np.angle(analytic)
    phase_deg = np.degrees(instantaneous_phase[-1])
    if phase_deg < 0:
        phase_deg += 360
    sine_val = np.sin(instantaneous_phase[-1])
    
    current_vol = v[-1]
    avg_vol = np.mean(v[-20:]) if len(v) >= 20 else current_vol
    vol_ratio = current_vol / (avg_vol + 1e-8)
    
    price_std = np.std(c_clean)
    amplitude_price = price_std * 2.0 

    state = "NEUTRAL"
    if 315 <= phase_deg or phase_deg < 45:
        state = "RESET"
    elif 45 <= phase_deg < 135:
        state = "RISING"
    elif 135 <= phase_deg < 225:
        state = "TOP_ZONE"
    elif 225 <= phase_deg < 315:
        state = "DIP_ZONE"

    is_reversal_zone = False
    if sine_val < -0.7 and vol_ratio > 1.5:
        is_reversal_zone = True
    elif sine_val > 0.7 and vol_ratio > 1.5:
        is_reversal_zone = True

    return phase_deg, sine_val, amplitude_price, state, vol_ratio, is_reversal_zone, price_std

def ml_confidence_score(d):
    score = 0
    if d['rsi'] < 35: score += 20
    elif d['rsi'] < 45: score += 10
    if d['sine'] < -0.5: score += 25
    if d['state'] == "DIP_ZONE": score += 15
    if d['vol_energy'] > 2.0: score += 20
    elif d['vol_energy'] > 1.5: score += 10
    if d['below_reg']: score += 20
    if d['compression'] and d['state'] == "DIP_ZONE":
        score += 15
    return min(100, score)

# ====================== CORE LOGIC ======================
def compute_tf(client, symbol, tf):
    klines = get_klines(client, symbol, tf)
    if klines is None:
        return None
    o, h, l, c, v = klines

    trend, low_band, high_band = regression(c)
    
    # Extremas
    idx_hi, idx_lo, recent_is_low, price_high_200, price_low_200, last_event = wick_extremes(h, l)
    
    # Harmonic
    phase, sine_val, amplitude_price, state, vol_energy, is_reversal, price_std = harmonic_oscillator(c, v)
    
    # Forecasts
    fft_val = fft_forecast(c)
    trend_slope = (trend[-1] - trend[-5]) / 5 if len(trend) > 5 else 0
    
    # Sine Target
    target_sine_val = 1.0 
    sine_price_target = c[-1] + (amplitude_price/2.0) * (target_sine_val - sine_val)
    
    # Volatility Compression
    bandwidth = (high_band[-1] - low_band[-1]) / (c[-1] + 1e-8)
    bandwidths_hist = [(high_band[i] - low_band[i]) / (c[i] + 1e-8) for i in range(-20, 0)]
    avg_bandwidth = np.mean(bandwidths_hist)
    squeeze = bandwidth < (avg_bandwidth * 0.8)
    
    bull_vol_pct = 50.0
    mom = 0.0
    recent_max_short = np.max(h[-SCALP_LOOKBACK_CANDLES:]) if len(h) >= SCALP_LOOKBACK_CANDLES else np.max(h)

    if tf == '1m':
        bull_vol_pct = volume_bull_percent(c, v)
        mom = momentum_short(c)

    data = {
        "rsi": rsi(c),
        "macd": macd(c),
        "fft": fft_val,
        "price": c[-1],
        "low_reg": low_band[-1],
        "high_reg": high_band[-1],
        "below_reg": c[-1] < low_band[-1],
        "dist": (low_band[-1] - c[-1]) / low_band[-1] * 100 if low_band[-1] > 0 else 0,
        "recent_is_low": recent_is_low,
        "target": np.max(h[-200:]) if len(h) >= 200 else np.max(h),
        "bull_vol_pct_1m": bull_vol_pct if tf == '1m' else None,
        "momentum_1m": mom if tf == '1m' else None,
        "recent_max_short": recent_max_short,
        "phase": phase,
        "sine": sine_val,
        "amplitude": amplitude_price,
        "state": state,
        "vol_energy": vol_energy,
        "is_reversal": is_reversal,
        
        # Extremas
        "highest_high_200": price_high_200,
        "lowest_low_200": price_low_200,
        "last_event": last_event,
        
        # Forecast slopes
        "reg_slope": trend_slope,
        
        # Specific Forecasts
        "sine_price_target": sine_price_target,
        
        # Compression
        "compression": squeeze
    }
    
    data['ml_score'] = ml_confidence_score(data)
    
    return data

def all_forecasts_above_price(tf_data):
    current = tf_data['1m']['price']
    return all(d['fft'] > current for d in tf_data.values())

def all_targets_above_price(tf_data):
    current = tf_data['1m']['price']
    return all(d['target'] > current for d in tf_data.values())

def ai_score(tf_data, imbalance):
    score = 0.0
    ml_sum = sum(d['ml_score'] for d in tf_data.values())
    score += ml_sum * 2.0 

    for d in tf_data.values():
        sine_score = (1 - d["sine"]) * 50
        score += sine_score
        if d["is_reversal"] and d["sine"] < 0:
            score += 150
        if d["sine"] < -0.5 and d["vol_energy"] > 2.0:
            score += d["vol_energy"] * 20

    for d in tf_data.values():
        score += max(0, 50 - d["rsi"])
        score += max(0, d["dist"]) * 3
        score += 100 if d["recent_is_low"] else -30

    if '1m' in tf_data:
        d1m = tf_data['1m']
        score += d1m["bull_vol_pct_1m"] * 1.2
        score += d1m["momentum_1m"] * 8000

    low_count = sum(1 for d in tf_data.values() if d["recent_is_low"])
    score += low_count * 180
    score += imbalance * 300
    return score

def sniper_filter(tf_data):
    if tf_data['1m']['ml_score'] < 60: return False
    if tf_data['5m']['ml_score'] < 50: return False
    
    for tf in ['1m', '3m', '5m']:
        if tf not in tf_data: return False
        d = tf_data[tf]
        if d["state"] != "DIP_ZONE" or not d["is_reversal"]:
            return False

    higher_tfs = ['15m', '30m', '1h', '4h', '1d']
    dip_count = 0
    reversal_count = 0
    for tf in higher_tfs:
        if tf in tf_data:
            d = tf_data[tf]
            if d["state"] == "DIP_ZONE": dip_count += 1
            if d["is_reversal"]: reversal_count += 1

    if dip_count < 3 or reversal_count < 2: return False

    for tf in ['1m','3m','5m']:
        if not tf_data[tf]["recent_is_low"]: return False

    higher_count = sum(1 for tf in higher_tfs if tf in tf_data and tf_data[tf]["recent_is_low"])
    if higher_count < 2: return False

    for tf in ['1m','3m','5m']:
        if not tf_data[tf]["below_reg"]: return False

    below_count = sum(1 for tf in higher_tfs if tf in tf_data and tf_data[tf]["below_reg"])
    if below_count < 2: return False

    return True

def scan(symbol, client):
    tf_data = {}
    for tf in TF_LIST:
        d = compute_tf(client, symbol, tf)
        if d is None:
            return None
        tf_data[tf] = d

    if not all_forecasts_above_price(tf_data):
        return None
    if not all_targets_above_price(tf_data):
        return None

    imbalance = orderbook(client, symbol)
    score = ai_score(tf_data, imbalance)
    sniper = sniper_filter(tf_data)

    return {
        "symbol": symbol,
        "data": tf_data,
        "score": score,
        "sniper": sniper,
        "imbalance": imbalance
    }

# ====================== DYNAMIC TABLE ======================
def build_table(candidates):
    table = Table(
        title="🔥 HARMONIC SNIPER SCANNER (ROBUST 200 EXTREMES)",
        expand=True,
        box=box.HEAVY,
        show_lines=True,
        show_edge=True
    )
    table.add_column("Rank", justify="center", style="bold", no_wrap=True)
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right", style="green", no_wrap=True)
    table.add_column("Sniper", justify="center", no_wrap=True)
    table.add_column("ML Conf", justify="center", no_wrap=True)

    for tf in TF_LIST:
        table.add_column(f"{tf} Phase", justify="right", no_wrap=True)
        
    if not candidates:
        return table
    sorted_cands = sorted(candidates, key=lambda x: x["score"], reverse=True)

    for i, c in enumerate(sorted_cands, 1):
        row = [
            str(i),
            c["symbol"],
            f"{c['score']:.0f}",
            "[green]YES[/green]" if c["sniper"] else "[red]NO[/red]",
            f"{c['data']['1m']['ml_score']}%"
        ]

        for tf in TF_LIST:
            d = c["data"].get(tf, {})
            phase = d.get("phase", 0)
            state = d.get("state", "-")

            if state == "DIP_ZONE": state = f"[bold green]{state}[/bold green]"
            elif state == "TOP_ZONE": state = f"[bold red]{state}[/bold red]"

            row.append(f"{phase:.0f}°")
        table.add_row(*row)

    return table

# ====================== ALERT ======================
def send_alert(msg):
    if TELEGRAM_TOKEN:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})

# ====================== MAIN ======================
if __name__ == "__main__":
    trader = Trader("credentials.txt")
    symbols = trader.get_usdc_pairs()

    console.print(f"[bold cyan]Scanning {len(symbols)} English USDC Pairs for Harmonic Dips...[/bold cyan]")

    candidates = []

    with Live(build_table(candidates), refresh_per_second=3, vertical_overflow="visible") as live:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(scan, s, trader.client) for s in symbols]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    candidates.append(result)
                live.update(build_table(candidates))

    if not candidates:
        console.print("[red]No candidates passed the harmonic filters.[/red]")
        sys.exit(1)

    # ====================== BEST CANDIDATE ======================
    best = max(candidates, key=lambda x: x["score"])

    console.print("\n[bold magenta]=== BEST HARMONIC DIP CANDIDATE ===[/bold magenta]")
    console.print(f"Symbol : [bold cyan]{best['symbol']}[/bold cyan]")
    console.print(f"AI Score : [bold green]{best['score']:.1f}[/bold green]")
    console.print(f"ML Confidence (1m) : {best['data']['1m']['ml_score']}%")

    console.print("\n[bold white]--- MTF Extremas & Forecasts ---[/bold white]")
    for tf, d in best["data"].items():
        extra = ""
        if tf == '1m':
            extra = f" | VolEnergy {d['vol_energy']:.1f}x"
        
        squeeze_str = "[yellow]SQUEEZE[/yellow]" if d['compression'] else ""

        last_event_str = "[green]LOW (Dip)[/green]" if d['last_event'] == "LOW_RECENT" else "[red]HIGH (Top)[/red]"

        console.print(
            f"{tf:4} | Low200: {d['lowest_low_200']:.8f} | High200: {d['highest_high_200']:.8f} | "
            f"Last Event: {last_event_str} | FFT: {d['fft']:.8f} | SineTgt: {d['sine_price_target']:.8f} | Phase {d['phase']:.0f}° {extra} {squeeze_str}"
        )

    # ====================== TRADE SETUP ======================
    console.print("\n[bold green]=== INSTANT SPOT LONG SETUP ===[/bold green]")
    entry = best["data"]["1m"]["price"]

    # Targets Logic
    tp_scalp = best['data']['1m']['recent_max_short']
    
    # --- FIBONACCI REVERSAL CYCLE LOGIC ---
    # Using 5m Extremas as the primary cycle baseline (Intermediate horizon)
    d5m = best['data']['5m']
    cycle_low = d5m['lowest_low_200']
    cycle_high = d5m['highest_high_200']
    cycle_diff = cycle_high - cycle_low
    
    # Calculate Fibonacci Ratios (Inner Floats)
    fib_0382 = cycle_low + (cycle_diff * 0.382)
    fib_0500 = cycle_low + (cycle_diff * 0.500)
    fib_0618 = cycle_low + (cycle_diff * 0.618) # Golden Ratio
    fib_0786 = cycle_low + (cycle_diff * 0.786) # Harmonic Root
    fib_1000 = cycle_high # Structural Resistance
    
    console.print(f"Entry Price          : [bold]{entry:.8f} USDC[/bold]")
    console.print(f"Fast Scalp Target    : [yellow]{tp_scalp:.8f} USDC[/yellow] (Quick liquidity grab)")
    
    console.print(f"\n[bold white]--- Fibonacci Reversal Cycle (5m Structure) ---[/bold white]")
    console.print(f"Cycle Low (0.0)      : {cycle_low:.8f}")
    console.print(f"Cycle High (1.0)     : {cycle_high:.8f}")
    
    console.print(f"\n[bold cyan]Primary Targets (Fibo Inner Ratios):[/bold cyan]")
    console.print(f"Fibo 0.382 Target    : {fib_0382:.8f} USDC")
    console.print(f"Fibo 0.500 Target    : {fib_0500:.8f} USDC")
    console.print(f"Fibo 0.618 Target    : [bold green]{fib_0618:.8f} USDC[/bold green] (Golden Ratio - Primary)")
    console.print(f"Fibo 0.786 Target    : [bold yellow]{fib_0786:.8f} USDC[/bold yellow] (Harmonic Extension)")
    
    console.print(f"\n[bold red]Full Cycle Reversal (1.0):[/bold red]")
    console.print(f"Structural High      : {fib_1000:.8f} USDC (Max resistance)")

    send_alert(f"🔥 HARMONIC DIP: {best['symbol']} | Score {best['score']:.0f} | Entry {entry:.8f} | Fibo 0.618 {fib_0618:.8f}")

    console.print("\n[bold cyan]Scan complete.[/bold cyan]")