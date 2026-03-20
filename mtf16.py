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
VOLUME_LOOKBACK = 20      # for volume bull/bear proxy
SCALP_LOOKBACK_CANDLES = 50  # for fast 5m aggressive target fallback

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

def ht_sine(c):
    normalized = (c - np.mean(c)) / np.std(c)
    analytic = signal.hilbert(normalized)
    sine = np.sin(np.angle(analytic))
    lead = np.sin(np.angle(analytic) + np.pi / 4)
    if lead[-1] > sine[-1] and lead[-2] <= sine[-2]:
        return "REVERSAL_UP"
    return "HOLD"

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
    idx_low  = np.argmin(recent_slice_l)
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

# ====================== CORE LOGIC ======================
def compute_tf(client, symbol, tf):
    klines = get_klines(client, symbol, tf)
    if klines is None:
        return None
    o, h, l, c, v = klines
    trend, low_band, high_band = regression(c)
    idx_hi, idx_lo, recent_is_low = wick_extremes(h, l)
    
    bull_vol_pct = 50.0
    mom = 0.0
    recent_max_short = np.max(h[-SCALP_LOOKBACK_CANDLES:]) if len(h) >= SCALP_LOOKBACK_CANDLES else np.max(h)
    
    if tf == '1m':
        bull_vol_pct = volume_bull_percent(c, v)
        mom = momentum_short(c)

    return {
        "rsi": rsi(c),
        "macd": macd(c),
        "sine": ht_sine(c),
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
        "recent_max_short": recent_max_short,  # used for 5m fast scalp
    }

def all_forecasts_above_price(tf_data):
    current = tf_data['1m']['price']
    return all(d['fft'] > current for d in tf_data.values())

def all_targets_above_price(tf_data):
    current = tf_data['1m']['price']
    return all(d['target'] > current for d in tf_data.values())

def ai_score(tf_data, imbalance):
    score = 0.0
    
    for d in tf_data.values():
        score += max(0, 50 - d["rsi"])
        score += 120 if d["sine"] == "REVERSAL_UP" else 0
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
        title="🔥 SNIPER MTF DIP SCANNER - ALL USDC PAIRS (Ranked)",
        expand=True,
        box=box.HEAVY,
        show_lines=True,
        show_edge=True
    )
    table.add_column("Rank", justify="center", style="bold", no_wrap=True)
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right", style="green", no_wrap=True)
    table.add_column("Sniper", justify="center", no_wrap=True)
    table.add_column("1m BullVol%", justify="right", no_wrap=True)
    
    for tf in TF_LIST:
        table.add_column(f"{tf} RSI", justify="right", no_wrap=True)
        table.add_column(f"{tf} Dist%", justify="right", no_wrap=True)
        table.add_column(f"{tf} RecentLow", justify="center", no_wrap=True)

    if not candidates:
        return table

    sorted_cands = sorted(candidates, key=lambda x: x["score"], reverse=True)
    
    for i, c in enumerate(sorted_cands, 1):
        d1m = c["data"].get("1m", {})
        row = [
            str(i),
            c["symbol"],
            f"{c['score']:.0f}",
            "[green]YES[/green]" if c["sniper"] else "[red]NO[/red]",
            f"{d1m.get('bull_vol_pct_1m', 50.0):.1f}%" if d1m else "-"
        ]
        
        for tf in TF_LIST:
            d = c["data"].get(tf, {})
            row += [
                f"{d.get('rsi', 50.0):.1f}" if d else "-",
                f"{d.get('dist', 0.0):+.2f}" if d else "-",
                "✔" if d.get("recent_is_low", False) else "✖" if d else "-"
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
    
    console.print(f"[bold cyan]Scanning {len(symbols)} USDC Spot pairs...[/bold cyan]")
    
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
        console.print("[red]No candidates passed the strict forecast/target filters.[/red]")
        sys.exit(1)
    
    # ====================== BEST CANDIDATE ======================
    best = max(candidates, key=lambda x: x["score"])
    
    console.print("\n[bold magenta]=== BEST MTF DIP CANDIDATE FOUND (SPOT LONG) ===[/bold magenta]")
    console.print(f"Symbol          : [bold cyan]{best['symbol']}[/bold cyan]")
    console.print(f"AI Score        : [bold green]{best['score']:.1f}[/bold green]")
    console.print(f"Orderbook Imb.  : {best['imbalance']:.4f} (positive = buy pressure)")
    console.print(f"Sniper Filter   : {'[green]YES[/green]' if best['sniper'] else '[red]NO[/red]'}")
    
    console.print("\n[bold white]--- Multi-Timeframe Technicals & Forecasts ---[/bold white]")
    for tf, d in best["data"].items():
        extra = ""
        if tf == '1m':
            extra = f" | BullVol {d['bull_vol_pct_1m']:.1f}% | Mom {d['momentum_1m']:+.8f}"
        console.print(
            f"{tf:4} | RSI {d['rsi']:6.1f} | MACD {d['macd']:+.4f} | Sine {d['sine']:12} | "
            f"Dist {d['dist']:+.2f}% | RegLow {d['low_reg']:.8f} | RegHigh {d['high_reg']:.8f} | "
            f"FFT Forecast {d['fft']:.8f} | RecentLow {d['recent_is_low']} {extra}"
        )
    
    # ====================== TRADE SETUP (SPOT LONG ONLY) ======================
    console.print("\n[bold green]=== INSTANT SPOT LONG SETUP ===[/bold green]")
    entry = best["data"]["1m"]["price"]
    
    # Conservative take-profit (max regression high across all TFs)
    tp_conservative = max(d["high_reg"] for d in best["data"].values())
    
    # Fast scalp target — focused on 5m timeframe
    d5m = best["data"].get("5m", {})
    tp_scalp = d5m.get("high_reg", entry * 1.005)  # regression upper band on 5m
    tp_scalp_aggressive = d5m.get("recent_max_short", tp_scalp)  # recent high within ~50 candles
    
    console.print(f"Entry Price          : [bold]{entry:.8f} USDC[/bold] (Market buy now)")
    console.print(f"Take Profit (main)   : [green]{tp_conservative:.8f} USDC[/green] (conservative — max RegHigh)")
    console.print(f"Fast Scalp Target    : [yellow]{tp_scalp:.8f} USDC[/yellow] (5m RegHigh — quick exit)")
    console.print(f"Fast Scalp Aggressive: [yellow]{tp_scalp_aggressive:.8f} USDC[/yellow] (5m recent high)")
    console.print(f"Max Target (overall) : {max(d['target'] for d in best['data'].values()):.8f} USDC (most aggressive seen)")
    
    send_alert(f"🔥 BEST SPOT DIP: {best['symbol']} | Score {best['score']:.0f} | Entry {entry:.8f} | Scalp 5m {tp_scalp:.8f}")
    
    console.print("\n[bold cyan]Scan complete. Full ranked table shown above.[/bold cyan]")