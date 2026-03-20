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

def wick_extreme(h, l):
    hi = np.argmax(h[-200:])
    lo = np.argmin(l[-200:])
    return hi, lo

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
    trend, low, high = regression(c)
    hi, lo = wick_extreme(h, l)
    return {
        "rsi": rsi(c),
        "macd": macd(c),
        "sine": ht_sine(c),
        "fft": fft_forecast(c),
        "price": c[-1],
        "low_reg": low[-1],
        "high_reg": high[-1],
        "below_reg": c[-1] < low[-1],
        "dist": (low[-1] - c[-1]) / low[-1] * 100,
        "recent_low": lo > hi,
        "target": np.max(h)
    }

def ai_score(tf_data, imbalance):
    score = 0.0
    for d in tf_data.values():
        score += max(0, 50 - d["rsi"])
        score += 120 if d["sine"] == "REVERSAL_UP" else 0
        score += d["dist"] * 2
        score += 80 if d["recent_low"] else 0
    score += imbalance * 200
    return score

def sniper_filter(tf_data):
    count = 0
    for tf, d in tf_data.items():
        if tf in ['1m', '3m', '5m']:
            if not (d["below_reg"] and d["recent_low"]):
                return False
        else:
            if d["below_reg"]:
                count += 1
    return count >= 3

def scan(symbol, client):
    tf_data = {}
    for tf in TF_LIST:
        d = compute_tf(client, symbol, tf)
        if not d:
            return None
        tf_data[tf] = d
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

# ====================== DYNAMIC TABLE (ALL ASSETS) ======================
def build_table(candidates):
    table = Table(title="🔥 SNIPER MTF DIP SCANNER - ALL USDC PAIRS (Ranked by Score)", 
                  expand=True, box=box.HEAVY, show_lines=True)
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Symbol", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Sniper", justify="center")
    
    for tf in TF_LIST:
        table.add_column(f"{tf} RSI", justify="right")
        table.add_column(f"{tf} Dist%", justify="right")
        table.add_column(f"{tf} W", justify="center")

    if not candidates:
        return table

    sorted_cands = sorted(candidates, key=lambda x: x["score"], reverse=True)
    
    for i, c in enumerate(sorted_cands, 1):
        row = [
            str(i),
            c["symbol"],
            f"{c['score']:.0f}",
            "[green]YES[/green]" if c["sniper"] else "[red]NO[/red]"
        ]
        for tf in TF_LIST:
            d = c["data"][tf]
            row += [
                f"{d['rsi']:.1f}",
                f"{d['dist']:.2f}",
                "✔" if d["recent_low"] else "✖"
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
    
    console.print(f"[bold cyan]Starting scan of {len(symbols)} USDC Spot pairs...[/bold cyan]")
    
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
        console.print("[red]No valid candidates found.[/red]")
        sys.exit(1)

    # ====================== BEST MTF DIP CANDIDATE ======================
    best = max(candidates, key=lambda x: x["score"])
    
    console.print("\n[bold magenta]=== BEST MTF DIP CANDIDATE FOUND ===[/bold magenta]")
    console.print(f"Symbol          : [bold cyan]{best['symbol']}[/bold cyan]")
    console.print(f"AI Score        : [bold green]{best['score']:.1f}[/bold green]")
    console.print(f"Orderbook Imb.  : {best['imbalance']:.4f} (positive = strong buy pressure)")
    console.print(f"Sniper Filter   : {'[green]YES[/green]' if best['sniper'] else '[red]NO[/red]'}")
    
    console.print("\n[bold white]--- Multi-Timeframe Technicals ---[/bold white]")
    for tf, d in best["data"].items():
        console.print(
            f"{tf:4} | RSI {d['rsi']:6.1f} | MACD {d['macd']:+.4f} | Sine {d['sine']:12} | "
            f"Dist {d['dist']:+.2f}% | RegLow {d['low_reg']:.8f} | RegHigh {d['high_reg']:.8f} | "
            f"WickLow {d['recent_low']}"
        )
    
    # ====================== TRADE SETUP ======================
    console.print("\n[bold green]=== INSTANT TRADE SETUP ===[/bold green]")
    entry = best["data"]["1m"]["price"]
    sl = min(d["low_reg"] for d in best["data"].values())
    tp = max(d["high_reg"] for d in best["data"].values())
    
    risk_pct = (entry - sl) / entry * 100 if entry > sl else 0
    reward_pct = (tp - entry) / entry * 100
    rr = reward_pct / risk_pct if risk_pct > 0 else float('inf')
    
    console.print(f"Entry Price     : [bold]{entry:.8f} USDC[/bold] (Market buy now)")
    console.print(f"Stop Loss       : [red]{sl:.8f} USDC[/red]")
    console.print(f"Take Profit     : [green]{tp:.8f} USDC[/green]")
    console.print(f"Risk            : {risk_pct:.2f}%")
    console.print(f"Reward          : {reward_pct:.2f}%")
    console.print(f"Risk:Reward     : [bold]1:{rr:.2f}[/bold]")
    console.print(f"Max Target      : {max(d['target'] for d in best['data'].values()):.8f} USDC")
    
    send_alert(f"🔥 BEST MTF DIP: {best['symbol']} | Score {best['score']:.0f} | Entry {entry:.8f}")
    
    console.print("\n[bold cyan]Scan complete. Full ranked table shown above.[/bold cyan]")