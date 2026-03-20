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
VOLUME_LOOKBACK = 20 # For volume bull/bear proxy
SCALP_LOOKBACK_CANDLES = 50
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
        return [s['symbol'] for s in info['symbols']
                if s['quoteAsset'] == 'USDC' and s['status'] == 'TRADING']
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
def wick_extremes(h, l, lookback=200):
    if len(h) < lookback:
        return -1, -1, False
    recent_slice_h = h[-lookback:]
    recent_slice_l = l[-lookback:]
    idx_high = np.argmax(recent_slice_h)
    idx_low = np.argmin(recent_slice_l)
    recent_is_low = idx_low > idx_high
    return idx_high, idx_low, recent_is_low
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
# ====================== NEW: HARMONIC OSCILLATOR LOGIC ======================
def harmonic_oscillator(c, v):
    """
    Core Mapping: S(t) = A * sin(omega * t + phi)
    Returns Phase (Deg), Sine Value (-1 to 1), and Volume Energy metrics.
    """
    # 1. Normalize Price to create the Oscillator S(t)
    # Detrend: Subtract mean to center around 0
    c_clean = c[~np.isnan(c)]
    if len(c_clean) < 2:
        return 0.0, 0.0, 0.0, "NEUTRAL", 0.0, False
       
    normalized = (c_clean - np.mean(c_clean)) / (np.std(c_clean) + 1e-8)
   
    # 2. Hilbert Transform to extract Phase and Sine
    analytic = signal.hilbert(normalized)
    instantaneous_phase = np.angle(analytic)
   
    # Convert to Degrees (0 to 360)
    phase_deg = np.degrees(instantaneous_phase[-1])
    if phase_deg < 0:
        phase_deg += 360
       
    # Sine Value represents Sentiment (-1 Fear, +1 Greed)
    sine_val = np.sin(instantaneous_phase[-1])
   
    # 3. Amplitude (Energy) derived from Volume
    # A(t) = alpha * V(t)
    current_vol = v[-1]
    avg_vol = np.mean(v[-20:]) if len(v) >= 20 else current_vol
    vol_ratio = current_vol / (avg_vol + 1e-8)
   
    # Amplitude scaling (Energy injection)
    amplitude = np.abs(analytic[-1]) * vol_ratio
   
    # 4. State Mapping (Phase Space)
    # 0/360: Start, 90: Rising, 180: Top, 270: Dip
    state = "NEUTRAL"
    if 315 <= phase_deg or phase_deg < 45:
        state = "RESET"
    elif 45 <= phase_deg < 135:
        state = "RISING"
    elif 135 <= phase_deg < 225:
        state = "TOP_ZONE"
    elif 225 <= phase_deg < 315:
        state = "DIP_ZONE" # Money zone
       
    # 5. Reversal Logic
    # Condition: Phase Saturation (near -1 or 1) + Volume Spike
    is_reversal_zone = False
    if sine_val < -0.7 and vol_ratio > 1.5: # Extreme Fear + High Vol
        is_reversal_zone = True # Capitulation Buy
    elif sine_val > 0.7 and vol_ratio > 1.5: # Extreme Greed + High Vol
        is_reversal_zone = True # Distribution Sell
       
    return phase_deg, sine_val, amplitude, state, vol_ratio, is_reversal_zone
# ====================== CORE LOGIC ======================
def compute_tf(client, symbol, tf):
    klines = get_klines(client, symbol, tf)
    if klines is None:
        return None
    o, h, l, c, v = klines
   
    # Standard Metrics
    trend, low_band, high_band = regression(c)
    idx_hi, idx_lo, recent_is_low = wick_extremes(h, l)
   
    # Harmonic Metrics (New)
    phase, sine_val, amplitude, state, vol_energy, is_reversal = harmonic_oscillator(c, v)
   
    bull_vol_pct = 50.0
    mom = 0.0
    recent_max_short = np.max(h[-SCALP_LOOKBACK_CANDLES:]) if len(h) >= SCALP_LOOKBACK_CANDLES else np.max(h)
   
    if tf == '1m':
        bull_vol_pct = volume_bull_percent(c, v)
        mom = momentum_short(c)
    return {
        "rsi": rsi(c),
        "macd": macd(c),
        "fft": fft_forecast(c),
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
       
        # New Harmonic Data
        "phase": phase,
        "sine": sine_val,
        "amplitude": amplitude,
        "state": state,
        "vol_energy": vol_energy,
        "is_reversal": is_reversal
    }
def all_forecasts_above_price(tf_data):
    current = tf_data['1m']['price']
    return all(d['fft'] > current for d in tf_data.values())
def all_targets_above_price(tf_data):
    current = tf_data['1m']['price']
    return all(d['target'] > current for d in tf_data.values())
def ai_score(tf_data, imbalance):
    score = 0.0
   
    # --- Harmonic Scoring (New Logic) ---
    for d in tf_data.values():
        # Reward Phase near 270 degrees (Dip Zone)
        # sine_val = -1 is ideal. Distance from -1 determines score.
        # sine_val ranges -1 to 1. We want high score when sine is -1.
        sine_score = (1 - d["sine"]) * 50 # Max 100 points if sine = -1
        score += sine_score
       
        # Reward Reversal Zones (Capitulation)
        if d["is_reversal"] and d["sine"] < 0:
            score += 150 # Strong buy signal
           
        # Volume Energy Bonus
        if d["sine"] < -0.5 and d["vol_energy"] > 2.0:
            score += d["vol_energy"] * 20 # High energy in fear zone
           
    # --- Standard Scoring ---
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
    # Harmonic Filter: Require Lower TFs to be in Dip Zone (180-360 deg, pref 270)
    for tf in ['1m','3m','5m']:
        d = tf_data.get(tf)
        if d:
            # Only buy if sine is negative (Fear zone) or just turned up (State = Reset/Rising start)
            if d["sine"] > 0.5:
                return False # Reject if in Greed zone
   
    # Standard Filter
    for tf in ['1m','3m','5m']:
        if tf in tf_data and not tf_data[tf]["recent_is_low"]:
            return False
   
    higher_count = sum(1 for tf in ['15m','30m','1h','4h','1d'] if tf in tf_data and tf_data[tf]["recent_is_low"])
    if higher_count < 2:
        return False
   
    for tf in ['1m','3m','5m']:
        if tf in tf_data and not tf_data[tf]["below_reg"]:
            return False
   
    below_count = sum(1 for tf in ['15m','30m','1h','4h','1d'] if tf in tf_data and tf_data[tf]["below_reg"])
    if below_count < 2:
        return False
   
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
        title="🔥 HARMONIC SNIPER SCANNER (Sine Wave Logic)",
        expand=True,
        box=box.HEAVY,
        show_lines=True,
        show_edge=True
    )
    table.add_column("Rank", justify="center", style="bold", no_wrap=True)
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right", style="green", no_wrap=True)
    table.add_column("Sniper", justify="center", no_wrap=True)
   
    for tf in TF_LIST:
        table.add_column(f"{tf} Phase", justify="right", no_wrap=True)
        table.add_column(f"{tf} Sine", justify="right", no_wrap=True)
        table.add_column(f"{tf} State", justify="center", no_wrap=True)
    if not candidates:
        return table
    sorted_cands = sorted(candidates, key=lambda x: x["score"], reverse=True)
   
    for i, c in enumerate(sorted_cands, 1):
        row = [
            str(i),
            c["symbol"],
            f"{c['score']:.0f}",
            "[green]YES[/green]" if c["sniper"] else "[red]NO[/red]",
        ]
       
        for tf in TF_LIST:
            d = c["data"].get(tf, {})
            phase = d.get("phase", 0)
            sine = d.get("sine", 0)
            state = d.get("state", "-")
           
            # Color code Sine
            sine_str = f"{sine:.2f}"
            if sine < -0.5: sine_str = f"[green]{sine:.2f}[/green]" # Fear
            elif sine > 0.5: sine_str = f"[red]{sine:.2f}[/red]" # Greed
           
            # Color code State
            if state == "DIP_ZONE": state = f"[bold green]{state}[/bold green]"
            elif state == "TOP_ZONE": state = f"[bold red]{state}[/bold red]"
           
            row += [
                f"{phase:.0f}°",
                sine_str,
                state
            ]
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
   
    console.print(f"[bold cyan]Scanning {len(symbols)} USDC Pairs for Harmonic Dips...[/bold cyan]")
   
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
    console.print(f"Orderbook Imb. : {best['imbalance']:.4f}")
   
    console.print("\n[bold white]--- Sine Wave Phase Analysis ---[/bold white]")
    for tf, d in best["data"].items():
        extra = ""
        if tf == '1m':
            extra = f" | VolEnergy {d['vol_energy']:.1f}x"
       
        phase_str = f"Phase {d['phase']:.0f}°"
        if d['sine'] < -0.5: phase_str += " (FEAR)"
        elif d['sine'] > 0.5: phase_str += " (GREED)"
       
        console.print(
            f"{tf:4} | {phase_str:15} | Sine {d['sine']:+.2f} | State {d['state']:10} | "
            f"ReversalZone: {d['is_reversal']} {extra}"
        )
   
    # ====================== TRADE SETUP ======================
    console.print("\n[bold green]=== INSTANT SPOT LONG SETUP ===[/bold green]")
    entry = best["data"]["1m"]["price"]
   
    # Conservative take-profit
    tp_conservative = max(d["high_reg"] for d in best["data"].values())
   
    # Fast scalp target
    d5m = best["data"].get("5m", {})
    tp_scalp = d5m.get("high_reg", entry * 1.005)
    tp_scalp_aggressive = d5m.get("recent_max_short", tp_scalp)
   
    # === NEW: INTERMEDIATE MIDDLE TARGET (between fast scalp and main TP) ===
    # This is the balanced exit level used as primary exit signal
    middle_target = (tp_scalp + tp_conservative) / 2
   
    console.print(f"Entry Price : [bold]{entry:.8f} USDC[/bold]")
    console.print(f"Fast Scalp Target : [yellow]{tp_scalp:.8f} USDC[/yellow]")
    console.print(f"Intermediate Exit Signal : [bold yellow]{middle_target:.8f} USDC[/bold yellow] (primary recommended exit)")
    console.print(f"Take Profit (main) : [green]{tp_conservative:.8f} USDC[/green]")
   
    send_alert(f"🔥 HARMONIC DIP: {best['symbol']} | Score {best['score']:.0f} | Phase {best['data']['1m']['phase']:.0f}° | Entry {entry:.8f}")
   
    console.print("\n[bold cyan]Scan complete.[/bold cyan]")