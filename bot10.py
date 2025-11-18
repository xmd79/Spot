#!/usr/bin/env python3
"""
1MIN SCALPER 213 — FINAL CLEAN EDITION
→ 12/12 REQUIRED
→ FULL READABLE COLUMN NAMES
→ 1m + 3m + 5m RSI REVERSAL FULLY WORKING
→ BUY_DOM = BUY % > SELL % (true dominance)
→ ENHANCED DISPLAY WITH NUMERICAL VALUES
→ SMA STACK: true/false for close < SMA12 < SMA27 < SMA56 < SMA100
→ DIP ZONE: argmin more recent than argmax + price below middle threshold
→ RATE LIMIT OPTIMIZED - AVOIDS API ERROR -1003
"""

import os
import time
import numpy as np
import pandas as pd
import threading
import signal
import talib
from binance.client import Client
from binance.exceptions import BinanceAPIException
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from colorama import init, Fore, Style
import warnings
from collections import defaultdict
import random

warnings.filterwarnings("ignore")
init(autoreset=True)

# ================= CONFIG =================
API_FILE = 'api.txt'
MAX_WORKERS = 8  # Reduced from 36 to avoid rate limits
SCAN_INTERVAL = 15  # Increased from 8 to give more time between scans
BATCH_SIZE = 10  # Process symbols in batches
BATCH_DELAY = 1.0  # Delay between batches
REQUEST_DELAY = 0.1  # Delay between individual requests

# ================= GLOBALS =================
stop_event = threading.Event()
candidates = {}
lock = threading.Lock()
current_executor = None
perfect_found = None
spike_alert = None
request_count = defaultdict(int)
last_minute = datetime.now().minute

# ================= RATE LIMIT TRACKING =================
def check_rate_limit():
    """Track and respect Binance API rate limits"""
    global last_minute, request_count
    
    current_minute = datetime.now().minute
    if current_minute != last_minute:
        request_count.clear()
        last_minute = current_minute
    
    total_requests = sum(request_count.values())
    if total_requests > 5500:  # Stay well under 6000 limit
        sleep_time = 60 - (datetime.now().second % 60)
        print(f"{Fore.YELLOW}Rate limit approaching. Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
        request_count.clear()

def make_request_with_delay(client, func, *args, **kwargs):
    """Make API request with delay and rate limit checking"""
    check_rate_limit()
    time.sleep(REQUEST_DELAY)
    
    try:
        result = func(*args, **kwargs)
        request_count[func.__name__] += 1
        return result
    except BinanceAPIException as e:
        if e.code == -1003:  # Rate limit error
            print(f"{Fore.RED}Rate limit hit. Waiting 60 seconds...")
            time.sleep(60)
            return make_request_with_delay(client, func, *args, **kwargs)
        raise e

# ================= SIGNAL HANDLING =================
def instant_stop(sig, frame):
    global current_executor
    print(f"\n{Fore.RED}{Style.BRIGHT}STOPPED BY USER")
    stop_event.set()
    if current_executor:
        current_executor.shutdown(wait=False, cancel_futures=True)
    os._exit(0)

signal.signal(signal.SIGINT, instant_stop)
signal.signal(signal.SIGTERM, instant_stop)

# ================= CLIENT =================
def get_client():
    if not os.path.exists(API_FILE):
        print(f"{Fore.RED}api.txt not found → line 1: API_KEY, line 2: SECRET")
        return None
    try:
        with open(API_FILE) as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            key, secret = lines[0], lines[1]
        client = Client(key, secret)
        client.API_URL = 'https://api.binance.com/api'
        # Set request timeout and retry settings
        client.session.timeout = (10, 30)  # (connect, read) timeout
        return client
    except Exception as e:
        print(f"{Fore.RED}API load error: {e}")
        return None

# ================= DATA FETCH — FIXED & RELIABLE =================
def fetch_ohlcv(client, symbol, interval, limit=500):
    try:
        # Use the rate-limited request function
        klines = make_request_with_delay(
            client, 
            client.get_klines,
            symbol=symbol, 
            interval=interval, 
            limit=limit
        )
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        df['taker_buy_base'] = pd.to_numeric(df['taker_buy_base'])
        return df[['close', 'volume', 'taker_buy_base']].values[-limit:]
    except Exception as e:
        # print(f"Fetch failed {symbol} {interval}: {e}")
        return None

# ================= RSI 5-LAYER REVERSAL — NOW 100% WORKING ON 1m/3m/5m =================
def rsi_reversal_confirmed(client, symbol, tf='1m'):
    try:
        data = fetch_ohlcv(client, symbol, tf, 500)
        if data is None or len(data) < 200:
            return False, 50.0
        closes = data[:, 0]
        rsi = talib.RSI(closes, timeperiod=14)
        if len(rsi) < 50 or np.isnan(rsi[-1]):
            return False, 50.0

        current = rsi[-1]
        hist = rsi[-200:]

        # 1. Extreme oversold touched
        extreme = 28 if tf == '1m' else 32
        touched = np.any(hist <= extreme)

        # 2. Last oversold more recent than overbought
        oversold_idx = np.where(hist <= 30)[0]
        overbought_idx = np.where(hist >= 70)[0]
        recent_oversold = len(oversold_idx) > 0 and (
            len(overbought_idx) == 0 or oversold_idx[-1] > overbought_idx[-1]
        )

        # 3. RSI rising
        rising = (len(rsi) > 10 and current > rsi[-5] and current > rsi[-10])

        # 4. Closer to 30 than 50
        golden_zone = abs(current - 30) < abs(current - 50)

        # 5. Still low
        cap = 42 if tf == '1m' else 45
        still_low = current < cap

        return all([touched, recent_oversold, rising, golden_zone, still_low]), current
    except:
        return False, 50.0

# ================= TRUE BUY_DOM: BUY % > SELL % =================
def buy_dominance_confirmed(client, symbol):
    try:
        data = fetch_ohlcv(client, symbol, '1m', 100)
        if data is None or len(data) < 50:
            return False, 0.0
        buy_vol = np.sum(data[-50:, 2])
        total_vol = np.sum(data[-50:, 1])
        buy_pct = (buy_vol / total_vol) * 100 if total_vol > 0 else 0.0
        return total_vol > 0 and (buy_vol / total_vol) > 0.5, buy_pct
    except:
        return False, 0.0

# ================= INDICATORS =================
def sma_stack_dip(client, symbol):
    data = fetch_ohlcv(client, symbol, '1m', 500)
    if data is None or len(data) < 100: return False
    c = data[:, 0]
    return c[-1] < np.mean(c[-12:]) < np.mean(c[-27:]) < np.mean(c[-56:]) < np.mean(c[-100:])

def golden_triangle(client, symbol):
    data = fetch_ohlcv(client, symbol, '1m', 500)
    if data is None or len(data) < 200: return False, 0.0
    p = data[:, 0][-200:]
    try:
        for i in range(10, len(p)-80, 15):
            for j in range(i+25, len(p)-50, 15):
                for k in range(j+25, len(p)-10, 15):
                    if p[k] <= p[i]: continue
                    a = abs(p[j] - p[i]); b = abs(p[k] - p[j]); c = abs(p[k] - p[i])
                    if min(a,b,c) < 1e-8: continue
                    ratios = sorted([a/b, b/c, c/a])
                    if all(1.3 < r < 2.8 for r in ratios):
                        err = abs(ratios[1]-1.618) + abs(ratios[2]/ratios[1]-1.618)
                        if err < 0.5: return True, err
        return False, 0.0
    except: return False, 0.0

def spike_incoming(client, symbol):
    data = fetch_ohlcv(client, symbol, '1m', 120)
    if data is None or len(data) < 80: return False, 0.0
    close = data[:, 0]; vol = data[:, 1]
    ratio = vol[-1] / (np.mean(vol[-60:-10]) + 1e-8)
    mom = close[-1]/close[-6] - 1 if close[-6] != 0 else 0
    t = np.arange(len(close))
    detrended = close - np.polyval(np.polyfit(t, close, 1), t)
    fft = np.abs(np.fft.fft(detrended))
    freqs = np.fft.fftfreq(len(detrended))
    peak = freqs[np.argmax(fft[1:])+1]
    phase = (len(close) * peak % 1.0) * 360
    strength = min(100, ratio * 15) if ratio > 2 else 0
    return (vol[-1] > vol[-5] and mom < 0.003 and 140 <= phase <= 240 and ratio > 2.5), strength

def dip_zone_confirmed(client, symbol):
    """Check if argmin is more recent than argmax and price is below middle threshold"""
    try:
        data = fetch_ohlcv(client, symbol, '1m', 100)
        if data is None or len(data) < 50:
            return False, 0.0, 0.0
        
        closes = data[:, 0]
        
        # Find argmin and argmax indices
        argmin_idx = np.argmin(closes)
        argmax_idx = np.argmax(closes)
        
        # Check if argmin is more recent than argmax
        min_more_recent = argmin_idx > argmax_idx
        
        # Calculate middle threshold between min and max
        min_price = closes[argmin_idx]
        max_price = closes[argmax_idx]
        middle_threshold = (min_price + max_price) / 2
        
        # Check if current close is below middle threshold
        current_close = closes[-1]
        below_middle = current_close < middle_threshold
        
        # Calculate distance from middle threshold as percentage
        distance_from_middle = (middle_threshold - current_close) / middle_threshold * 100 if middle_threshold > 0 else 0.0
        
        # Calculate how recent the min is (as percentage of total period)
        min_recency = argmin_idx / len(closes) * 100
        
        return min_more_recent and below_middle, distance_from_middle, min_recency
    except:
        return False, 0.0, 0.0

# ================= CORE ANALYSIS =================
def analyze_symbol(client, symbol, prices):
    global perfect_found, spike_alert
    if stop_event.is_set(): return
    try:
        price = float(prices.get(symbol, 0))
        if price <= 0: return

        data_1m = fetch_ohlcv(client, symbol, '1m', 500)
        if data_1m is None or len(data_1m) < 300: return
        close = data_1m[:, 0]
        volume = data_1m[:, 1]

        # === ALL 3 RSI TIMEFRAMES — NOW WORKING ===
        rsi_1m_conf, rsi_1m_val = rsi_reversal_confirmed(client, symbol, '1m')
        rsi_3m_conf, rsi_3m_val = rsi_reversal_confirmed(client, symbol, '3m')
        rsi_5m_conf, rsi_5m_val = rsi_reversal_confirmed(client, symbol, '5m')

        current_rsi = 50.0
        try:
            r = talib.RSI(close, 14)
            if len(r) > 0 and not np.isnan(r[-1]): current_rsi = r[-1]
        except: pass

        mom_val = talib.MOM(close, 14)[-1] if len(talib.MOM(close, 14)) > 0 else 0.0
        mom_ok = mom_val > 0
        is_spike, spike_val = spike_incoming(client, symbol)
        buy_dom_conf, buy_pct = buy_dominance_confirmed(client, symbol)

        # NEW DIP ZONE LOGIC
        dip_conf, dip_distance, dip_recency = dip_zone_confirmed(client, symbol)

        octant_val = int((np.arctan2(np.diff(close[-80:]), 1)[-1] * 180 / np.pi) % 360 // 45) % 8
        octant_conf = octant_val in [0,1,2,3]

        tri_conf, tri_err = golden_triangle(client, symbol)
        vol_exp_ratio = volume[-1] / np.mean(volume[-50:-10]) if np.mean(volume[-50:-10]) > 0 else 0.0
        vol_exp_conf = volume[-1] > 4.2 * np.mean(volume[-50:-10])

        poly_coeff = np.polyfit(np.arange(50), close[-50:], 2)[0]
        poly_conf = poly_coeff > 1e-8

        sma_conf = sma_stack_dip(client, symbol)

        conditions = {
            "RSI 1m Rev": rsi_1m_conf,
            "RSI 3m Rev": rsi_3m_conf,
            "RSI 5m Rev": rsi_5m_conf,
            "Dip Zone": dip_conf,
            "Octant Up": octant_conf,
            "Golden Tri": tri_conf,
            "Volume Exp": vol_exp_conf,
            "Buy Dominate": buy_dom_conf,
            "Momentum Up": mom_ok,
            "Poly Up": poly_conf,
            "Spike In": is_spike,
            "SMA Stack": sma_conf
        }

        # Store numerical values
        num_values = {
            "RSI 1m": rsi_1m_val,
            "RSI 3m": rsi_3m_val,
            "RSI 5m": rsi_5m_val,
            "Dip Dist": dip_distance,
            "Dip Rec": dip_recency,
            "Octant": octant_val,
            "Tri Err": tri_err,
            "Vol Ratio": vol_exp_ratio,
            "Buy %": buy_pct,
            "Momentum": mom_val,
            "Poly Coeff": poly_coeff
        }

        score = sum(conditions.values())
        perfect = score == 12

        with lock:
            candidates[symbol] = {
                "score": score,
                "price": price,
                "rsi": round(current_rsi, 1),
                "conds": conditions,
                "nums": num_values,
                "spike": round(spike_val, 1),
                "perfect": perfect
            }
            if perfect and perfect_found is None:
                perfect_found = {"symbol": symbol, "price": price}
                stop_event.set()
            if spike_val > 80 and spike_alert is None:
                spike_alert = symbol

    except: pass

# ================= LIVE DISPLAY — FULL READABLE LABELS =================
def live_display():
    while not stop_event.is_set():
        with lock:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"{Fore.CYAN}{Style.BRIGHT}1MIN SCALPER 213 — LIVE SCAN (RATE LIMIT OPTIMIZED)")
            print(f"{Fore.CYAN}{'═' * 280}\n")

            header = f"{'#':<3} {'SPIKE':>5} {'SCORE':>5} {'SYMBOL':<14} {'RSI':>7} {'RSI1m':>7} {'RSI3m':>7} {'RSI5m':>7} {'OCT':>5} {'TRI':>7} {'VOL%':>7} {'BUY%':>7} {'MOM':>7} {'POLY':>7} {'DIP%':>7} {'DIPR':>5} {'SPK':>5} {'SMA':>5} {'PRICE':>28}"
            print(f"{Fore.YELLOW}{Style.BRIGHT}{header}")
            print(f"{Fore.YELLOW}{'─' * 280}")

            sorted_cands = sorted(candidates.items(), key=lambda x: (x[1]["score"], x[1]["spike"]), reverse=True)[:30]
            for i, (sym, d) in enumerate(sorted_cands, 1):
                c = d["conds"]
                n = d["nums"]
                color = Fore.GREEN + Style.BRIGHT if d["score"] == 12 else \
                        Fore.RED + Style.BRIGHT if d["spike"] > 80 else \
                        Fore.YELLOW if d["score"] >= 9 else Fore.WHITE

                spike_mark = f"{Fore.RED}!!!{Style.RESET_ALL}" if d["spike"] > 80 else f"{d['spike']:>5}"

                row = f"{i:<3} {spike_mark} {Fore.CYAN}{d['score']}/12{Style.RESET_ALL} " \
                      f"{color}{sym:<14}{Style.RESET_ALL} {d['rsi']:>7.1f} " \
                      f"{n['RSI 1m']:>7.2f} {n['RSI 3m']:>7.2f} {n['RSI 5m']:>7.2f} " \
                      f"{n['Octant']:>5.0f} {n['Tri Err']:>7.4f} {n['Vol Ratio']:>7.2f} {n['Buy %']:>7.2f} " \
                      f"{n['Momentum']:>7.6f} {n['Poly Coeff']:>7.6f} {n['Dip Dist']:>7.2f} " \
                      f"{n['Dip Rec']:>5.0f} {'true' if c['Spike In'] else 'false':>5} {'true' if c['SMA Stack'] else 'false':>5} " \
                      f"${d['price']:>27.25f}"
                print(row)

            status = f"{Fore.RED}{Style.BRIGHT}SPIKE → {spike_alert}{Style.RESET_ALL}" if spike_alert else "Waiting for 12/12 perfect signal..."
            print(f"\n{Fore.WHITE}{status}")
        time.sleep(1.0)

# ================= MAIN =================
def main():
    global current_executor
    client = get_client()
    if not client: return

    print(f"{Fore.GREEN}{Style.BRIGHT}1MIN SCALPER 213 STARTED — SCANNING USDC PAIRS (RATE LIMIT OPTIMIZED)")
    threading.Thread(target=live_display, daemon=True).start()
    time.sleep(3)

    while not stop_event.is_set():
        with lock:
            candidates.clear()
            perfect_found = None
            spike_alert = None

        try:
            # Get exchange info with rate limiting
            info = make_request_with_delay(client, client.get_exchange_info)
            symbols = [s["symbol"] for s in info["symbols"] if s["quoteAsset"] == "USDC" and s["status"] == "TRADING"]
            
            # Get prices with rate limiting
            all_tickers = make_request_with_delay(client, client.get_all_tickers)
            prices = {t["symbol"]: t["price"] for t in all_tickers}
            
            # Limit to top 100 symbols by price to reduce API calls
            # You can adjust this or implement your own filtering logic
            if len(symbols) > 100:
                # Filter symbols with reasonable price ranges
                symbols = [s for s in symbols if prices.get(s, 0) and float(prices[s]) > 0][:100]
            
            print(f"{Fore.CYAN}Scanning {len(symbols)} USDC pairs...")
            
        except Exception as e:
            print(f"{Fore.RED}Error fetching data: {e}")
            time.sleep(30)
            continue

        # Process symbols in batches to avoid rate limits
        for i in range(0, len(symbols), BATCH_SIZE):
            if stop_event.is_set(): break
            
            batch = symbols[i:i + BATCH_SIZE]
            
            current_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
            futures = []
            
            for sym in batch:
                if stop_event.is_set(): break
                # Add random delay to distribute requests
                time.sleep(random.uniform(0.05, 0.15))
                futures.append(current_executor.submit(analyze_symbol, client, sym, prices))
            
            # Wait for batch to complete
            for future in as_completed(futures):
                try:
                    future.result(timeout=30)
                except Exception as e:
                    pass
            
            current_executor.shutdown(wait=True)
            
            # Delay between batches
            if i + BATCH_SIZE < len(symbols):
                time.sleep(BATCH_DELAY)

        if perfect_found:
            os.system('cls' if os.name == 'nt' else 'clear')
            p = perfect_found
            print(f"{Fore.GREEN}{Style.BRIGHT}PERFECT 12/12 ACHIEVED — ENTER LONG NOW")
            print(f"{Fore.WHITE}{Style.BRIGHT}SYMBOL → {p['symbol']}")
            print(f"PRICE  → ${float(p['price']):.25f}")
            print(f"{Fore.YELLOW}{Style.BRIGHT}ALL 12 CONDITIONS CONFIRMED — MAXIMUM PROBABILITY REVERSAL")
            break

        time.sleep(SCAN_INTERVAL)

    print(f"{Fore.MAGENTA}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Scanner stopped")

if __name__ == "__main__":
    main()