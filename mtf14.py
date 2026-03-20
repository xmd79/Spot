from binance.client import Client
import numpy as np
import sys
import concurrent.futures
import re
import scipy.signal as signal
from datetime import datetime
import gc
from rich.console import Console
from rich.table import Table
from rich import box
from rich.live import Live
from scipy.stats import linregress

# --- Configuration ---
RSI_LENGTH = 14
LOOKBACK_LIMIT = 1200
MAX_WORKERS = 12
TF_LIST = ['1m','3m','5m','15m','30m','1h','4h','1d']

console = Console()

# ---------------- Binance Connection ----------------
class Trader:
    def __init__(self, file):
        self.connect(file)

    def connect(self, file):
        try:
            lines = [line.rstrip('\n') for line in open(file)]
            self.client = Client(lines[0], lines[1])
        except Exception as e:
            console.print(f"[red]Error connecting to Binance:[/red] {e}")
            sys.exit(1)

    def get_usdc_pairs(self):
        try:
            info = self.client.get_exchange_info()
            return [s['symbol'] for s in info['symbols']
                    if s['quoteAsset']=='USDC' and s['status']=='TRADING' 
                    and re.match(r'^[A-Z0-9]+$', s['baseAsset'])]
        except:
            return []

# ---------------- Signals & Calculations ----------------
def get_klines(client, symbol, interval, limit=500):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=min(limit,1000))
        if not klines or len(klines)<50: return None
        o = np.array([float(k[1]) for k in klines])
        h = np.array([float(k[2]) for k in klines])
        l = np.array([float(k[3]) for k in klines])
        c = np.array([float(k[4]) for k in klines])
        v = np.array([float(k[5]) for k in klines])
        return o,h,l,c,v
    except:
        return None

def rsi(closes, period=RSI_LENGTH):
    if len(closes)<period+1: return 50.0
    deltas = np.diff(closes)
    up, down = np.maximum(deltas,0), np.maximum(-deltas,0)
    avg_up, avg_down = np.mean(up[:period]), np.mean(down[:period])
    rs = avg_up/(avg_down if avg_down!=0 else 1e-8)
    rsi_val = 100-100/(1+rs)
    for i in range(period,len(deltas)):
        avg_up=(avg_up*(period-1)+up[i])/period
        avg_down=(avg_down*(period-1)+down[i])/period
        rs=avg_up/(avg_down if avg_down!=0 else 1e-8)
        rsi_val=100-100/(1+rs)
    return rsi_val

def ht_sine(closes):
    if len(closes)<50: return "HOLD",0,0
    c = closes[np.isfinite(closes)]
    if np.std(c) < 1e-8: return "HOLD",0,0
    normalized = (c - np.mean(c)) / np.std(c)
    analytic = signal.hilbert(normalized)
    sine = np.sin(np.angle(analytic))
    leadsine = np.sin(np.angle(analytic) + np.pi/4)
    if leadsine[-1]>sine[-1] and leadsine[-2]<=sine[-2]: return "REVERSAL_UP",sine[-1],leadsine[-1]
    if leadsine[-1]<sine[-1] and leadsine[-2]>=sine[-2]: return "REVERSAL_DOWN",sine[-1],leadsine[-1]
    return "HOLD", sine[-1], leadsine[-1]

def regression_channel(closes):
    x = np.arange(len(closes))
    slope, intercept, r_value, _, _ = linregress(x, closes)
    trend = intercept + slope*x
    residuals = closes - trend
    std_dev = np.std(residuals)
    upper = trend + std_dev
    lower = trend - std_dev
    middle = trend
    return trend, upper, lower, middle, slope

def fft_forecast(closes, bars=20):
    n = len(closes)
    if n<50: return np.array([closes[-1]]*bars)
    t = np.arange(n)
    coeffs = np.polyfit(t, closes, 1)
    trend = np.polyval(coeffs, t)
    detrended = closes - trend
    fft_vals = np.fft.fft(detrended)
    freqs = np.fft.fftfreq(n)
    indices = np.argsort(np.abs(fft_vals.real))[-12:]
    future_t = np.arange(n, n+bars)
    forecast = np.zeros(bars)
    for i in indices:
        amp, phase, freq = np.abs(fft_vals[i])/n, np.angle(fft_vals[i]), freqs[i]
        forecast += amp*np.cos(2*np.pi*freq*future_t + phase)
    return forecast + np.polyval(coeffs, future_t)

def compute_tf_signals(o,h,l,c,v):
    signals = {}
    signals['rsi'] = rsi(c)
    signals['sine'], signals['sine_val'], signals['lead_val'] = ht_sine(c)
    trend, upper, lower, middle, slope = regression_channel(c)
    signals['trend'] = trend
    signals['upper'] = upper
    signals['lower'] = lower
    signals['middle'] = middle
    signals['slope'] = slope
    signals['forecast'] = fft_forecast(c)
    signals['current'] = c[-1]
    signals['target'] = np.max(h)
    signals['regression_forecast'] = np.array([lower[-1], middle[-1], upper[-1]])
    signals['overall_forecast'] = np.mean(signals['forecast'])
    signals['is_dip'] = signals['rsi']<30 or signals['sine']=="REVERSAL_UP"
    return signals

# ---------------- Scan a single symbol ----------------
def scan_symbol(symbol, client):
    try:
        data_map = {}
        dip_tf_count = 0
        for tf in TF_LIST:
            data = get_klines(client, symbol, tf)
            if not data: return None
            signals = compute_tf_signals(*data)
            if signals['is_dip']:
                dip_tf_count += 1
            data_map[tf] = signals
        score = sum(max(0,50-d['rsi']) + 100*(1 if d['sine']=="REVERSAL_UP" else 0) for d in data_map.values())
        overall_forecast = np.mean([d['overall_forecast'] for d in data_map.values()])
        current_price = data_map[TF_LIST[-1]]['current']
        return {'symbol':symbol,'score':score,'data_map':data_map,'overall_forecast':overall_forecast,
                'current_price':current_price,'dip_tf_count':dip_tf_count}
    except:
        return None

# ---------------- Build Live Table ----------------
def build_table(candidates):
    table = Table(title=f"MTF Dip Leaderboard - {datetime.now().strftime('%H:%M:%S')}", box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Rank", justify="right")
    table.add_column("Symbol")
    table.add_column("Score", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Overall Forecast", justify="right")
    for tf in TF_LIST:
        table.add_column(f"{tf} RSI", justify="right")
        table.add_column(f"{tf} Sine")
        table.add_column(f"{tf} Target", justify="right")
        table.add_column(f"{tf} Reg Low", justify="right")
        table.add_column(f"{tf} Reg Mid", justify="right")
        table.add_column(f"{tf} Reg High", justify="right")
        table.add_column(f"{tf} Forecast", justify="right")
    
    for i,c in enumerate(sorted(candidates,key=lambda x:(x['dip_tf_count'],x['score']),reverse=True),1):
        row = [str(i), c['symbol'], f"{c['score']:.1f}", f"{c['current_price']:.6f}", f"{c['overall_forecast']:.6f}"]
        for tf in TF_LIST:
            d = c['data_map'][tf]
            rsi_str = f"[green]{d['rsi']:.1f}[/green]" if d['rsi']<30 else f"[red]{d['rsi']:.1f}[/red]" if d['rsi']>70 else f"{d['rsi']:.1f}"
            sine_str = f"[yellow]{d['sine']}[/yellow]" if "REVERSAL" in d['sine'] else d['sine']
            row += [
                rsi_str,
                sine_str,
                f"{d['target']:.6f}",
                f"{d['lower'][-1]:.6f}",
                f"{d['middle'][-1]:.6f}",
                f"{d['upper'][-1]:.6f}",
                f"{np.mean(d['forecast']):.6f}"
            ]
        table.add_row(*row)
    return table

# ---------------- Main Execution ----------------
if __name__=="__main__":
    trader = Trader('credentials.txt')
    symbols = trader.get_usdc_pairs()
    console.print(f"[green]Scanning {len(symbols)} USDC pairs across multiple timeframes...[/green]")

    candidates = []
    with Live(build_table(candidates), refresh_per_second=2) as live:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_symbol = {executor.submit(scan_symbol,s,trader.client):s for s in symbols}
            for future in concurrent.futures.as_completed(future_to_symbol):
                res = future.result()
                if res: candidates.append(res)
                live.update(build_table(candidates))
                gc.collect()
        
        # Stop live before printing best candidate
        live.stop()

    # Print the best MTF dip candidate after scanning
    if candidates:
        best = max(candidates, key=lambda x:(x['dip_tf_count'],x['score']))
        console.print("\n[bold magenta]=== BEST MTF DIP CANDIDATE FOUND ===[/bold magenta]")
        console.print(f"[bold]Symbol:[/bold] {best['symbol']}")
        console.print(f"[bold]Score:[/bold] {best['score']:.1f}")
        console.print(f"[bold]Current Price:[/bold] {best['current_price']:.6f}")
        console.print(f"[bold]Overall Forecast:[/bold] {best['overall_forecast']:.6f}")
        console.print(f"[bold]Timeframes indicating dip:[/bold] {best['dip_tf_count']}/{len(TF_LIST)}")
        for tf in TF_LIST:
            d = best['data_map'][tf]
            console.print(f"[bold]{tf}:[/bold] RSI {d['rsi']:.1f}, Sine {d['sine']}, Target {d['target']:.6f}, Reg [L:{d['lower'][-1]:.6f} M:{d['middle'][-1]:.6f} H:{d['upper'][-1]:.6f}], Forecast {np.mean(d['forecast']):.6f}")

    console.print("\n[bold green]Scan complete. Bot stopped after selecting best MTF dip.[/bold green]")