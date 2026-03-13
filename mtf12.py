from binance.client import Client
import numpy as np
import sys
import concurrent.futures
import re
import time
import scipy.signal as signal
import pandas as pd
from datetime import datetime
import gc

# --- Configuration ---
RSI_LENGTH = 14
LOOKBACK_LIMIT = 1200
GOLDEN_RATIO_DERIV = 0.61803398875

class Trader:
    def __init__(self, file):
        self.connect(file)

    def connect(self, file):
        try:
            lines = [line.rstrip('\n') for line in open(file)]
            key = lines[0]
            secret = lines[1]
            self.client = Client(key, secret)
        except Exception as e:
            print(f"Error connecting to Binance: {e}")
            sys.exit(1)

    def get_usdc_pairs(self):
        try:
            exchange_info = self.client.get_exchange_info()
            trading_pairs = []
            for symbol in exchange_info['symbols']:
                if symbol['quoteAsset'] != 'USDC': continue
                if symbol['status'] != 'TRADING': continue
                base_asset = symbol['baseAsset']
                if not re.match(r'^[A-Z0-9]+$', base_asset): continue
                trading_pairs.append(symbol['symbol'])
            return trading_pairs
        except Exception as e:
            print(f"Error fetching pairs: {e}")
            return []

filename = 'credentials.txt'
trader = Trader(filename)

def get_klines_data(client, symbol, interval, limit=1000):
    try:
        api_limit = min(limit, 1000)
        klines = client.get_klines(symbol=symbol, interval=interval, limit=api_limit)
        if not klines or len(klines) < 200:
            return None
        opens = np.array([float(entry[1]) for entry in klines], dtype=np.float64)
        highs = np.array([float(entry[2]) for entry in klines], dtype=np.float64)
        lows = np.array([float(entry[3]) for entry in klines], dtype=np.float64)
        closes = np.array([float(entry[4]) for entry in klines], dtype=np.float64)
        volumes = np.array([float(entry[5]) for entry in klines], dtype=np.float64)
        if not np.all(np.isfinite(closes)):
            return None
        return opens, highs, lows, closes, volumes
    except Exception:
        return None

def analyze_extrema(lows, highs, timeframe_name):
    window = min(LOOKBACK_LIMIT, len(lows))
    l_vals = lows[-window:]
    h_vals = highs[-window:]
    ll_val = np.min(l_vals)
    hh_val = np.max(h_vals)
    idx_ll_rel = int(np.argmin(l_vals))
    idx_hh_rel = int(np.argmax(h_vals))
    bars_ago_ll = (window - 1) - idx_ll_rel
    bars_ago_hh = (window - 1) - idx_hh_rel
    recent_extrema = "ARGMIN (Low)" if bars_ago_ll < bars_ago_hh else "ARGMAX (High)"
    return {
        'timeframe': timeframe_name, 'll': ll_val, 'hh': hh_val,
        'bars_ago_ll': bars_ago_ll, 'bars_ago_hh': bars_ago_hh,
        'recent_type': recent_extrema, 'window_used': window
    }

def analyze_volume_structure(opens, closes, volumes):
    window = min(LOOKBACK_LIMIT, len(volumes))
    o_vals = opens[-window:]
    c_vals = closes[-window:]
    v_vals = volumes[-window:]
    total_vol = np.sum(v_vals)
    if total_vol == 0: return 0, 0, 0, 0
    bull_mask = c_vals > o_vals
    bear_mask = c_vals < o_vals
    bull_vol = np.sum(v_vals[bull_mask])
    bear_vol = np.sum(v_vals[bear_mask])
    active_vol = bull_vol + bear_vol
    if active_vol == 0: return 0, 0, 0, 0
    return (bull_vol / active_vol * 100, bear_vol / active_vol * 100, bull_vol, bear_vol)

def check_volume_induction(lows, volumes):
    window = min(LOOKBACK_LIMIT, len(lows))
    l_vals = lows[-window:]
    v_vals = volumes[-window:]
    idx_ll = int(np.argmin(l_vals))
    vol_at_ll = v_vals[idx_ll]
    avg_vol = np.mean(v_vals)
    is_induction = vol_at_ll > (avg_vol * 1.5)
    return is_induction, vol_at_ll, avg_vol

def simple_rsi(closes, period=RSI_LENGTH):
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    up = np.maximum(deltas, 0)
    down = np.maximum(-deltas, 0)
    avg_up = np.mean(up[:period])
    avg_down = np.mean(down[:period])
    rs = avg_up / avg_down if avg_down != 0 else np.inf
    rsi_val = 100 - 100 / (1 + rs) if np.isfinite(rs) else 100
    for i in range(period, len(deltas)):
        curr_up = up[i]
        curr_down = down[i]
        avg_up = (avg_up * (period - 1) + curr_up) / period
        avg_down = (avg_down * (period - 1) + curr_down) / period
        rs = avg_up / avg_down if avg_down != 0 else np.inf
        rsi_val = 100 - 100 / (1 + rs) if np.isfinite(rs) else 100
    return rsi_val

def analyze_ht_sine(closes):
    if len(closes) < 50:
        return "HOLD", 0.0, 0.0
    closes = closes[np.isfinite(closes)]
    if len(closes) < 50:
        return "HOLD", 0.0, 0.0
    mean = np.mean(closes)
    std = np.std(closes)
    if std < 1e-8:
        return "HOLD", 0.0, 0.0
    closes_norm = (closes - mean) / std
    analytic = signal.hilbert(closes_norm)
    if np.any(~np.isfinite(analytic)):
        return "HOLD", 0.0, 0.0
    inst_phase = np.angle(analytic)
    sine = np.sin(inst_phase)
    lead_phase = inst_phase + np.pi / 4
    leadsine = np.sin(lead_phase)
    if len(sine) < 2:
        return "HOLD", 0.0, 0.0
    s_val = sine[-1]
    ls_val = leadsine[-1]
    s_prev = sine[-2]
    ls_prev = leadsine[-2]
    ht_signal = "HOLD"
    if ls_val > s_val and ls_prev <= s_prev:
        ht_signal = "REVERSAL_UP"
    elif ls_val < s_val and ls_prev >= s_prev:
        ht_signal = "REVERSAL_DOWN"
    return ht_signal, s_val, ls_val

def calculate_fft_forecast(closes):
    n = len(closes)
    if n < 100:
        return np.array([closes[-1]] * 20), 0.0
    t = np.arange(n)
    coeffs = np.polyfit(t, closes, 1)
    trend = np.polyval(coeffs, t)
    detrended = closes - trend
    fft_vals = np.fft.fft(detrended)
    freqs = np.fft.fftfreq(n)
    indices = np.argsort(np.abs(fft_vals.real))[::-1][:12]
    forecast_bars = 20
    future_t = np.arange(n, n + forecast_bars)
    forecast_signal = np.zeros(forecast_bars)
    for i in indices:
        if abs(fft_vals[i]) < 1e-6: continue
        amp = np.abs(fft_vals[i]) / n
        phase = np.angle(fft_vals[i])
        freq = freqs[i]
        forecast_signal += amp * np.cos(2 * np.pi * freq * future_t + phase)
    future_trend = np.polyval(coeffs, future_t)
    forecast_prices = forecast_signal + future_trend
    upward_bias = np.mean(forecast_prices > closes[-1])
    return forecast_prices, upward_bias

def compute_signal(data_map, timeframes):
    extrema_map = {}
    volume_map = {}
    fft_map = {}
    phi_map = {}
    for tf in timeframes:
        o, h, l, c, v = data_map[tf]
        extrema_map[tf] = analyze_extrema(l, h, tf)
        hh = extrema_map[tf]['hh']
        ll = extrema_map[tf]['ll']
        range_ = hh - ll
        phi_618 = ll + range_ * GOLDEN_RATIO_DERIV
        phi_100 = ll + range_ * 1.000
        phi_map[tf] = phi_618 if c[-1] < (ll + range_ * 0.3) else phi_100
        volume_map[tf] = analyze_volume_structure(o, c, v)
        fft_forecast, upward_bias = calculate_fft_forecast(c)
        fft_map[tf] = {'forecast': fft_forecast, 'up_bias': upward_bias}
    if extrema_map.get('1m', {}).get('recent_type') != "ARGMIN (Low)": return None
    if extrema_map.get('5m', {}).get('recent_type') != "ARGMIN (Low)": return None
    argmin_count = sum(1 for tf in timeframes if extrema_map.get(tf, {}).get('recent_type') == "ARGMIN (Low)")
    if argmin_count < 5: return None
    for tf in ['1m', '3m', '5m', '15m']:
        bull_p, bear_p, _, _ = volume_map.get(tf, (0,0,0,0))
        if bull_p <= bear_p: return None
    induction_1m, vol_ll, avg_vol = check_volume_induction(data_map['1m'][2], data_map['1m'][4])
    if not induction_1m: return None
    target_price = phi_map['5m']
    current_price = data_map['1m'][3][-1]
    if current_price >= target_price: return None
    if len(data_map['5m'][3]) < 50 or len(data_map['1m'][3]) < 50:
        return None
    sig_5m, _, _ = analyze_ht_sine(data_map['5m'][3])
    sig_1m, _, _ = analyze_ht_sine(data_map['1m'][3])
    if sig_1m != "REVERSAL_UP": return None
    rsi = simple_rsi(data_map['5m'][3])
    reward_dist = (target_price - current_price) / current_price if current_price > 0 else 0
    score = reward_dist * 1500
    score += max(0, (45 - rsi)) * 3
    sine_bonus = 100 if sig_1m == "REVERSAL_UP" else 0
    sine_bonus += 80 if sig_5m == "REVERSAL_UP" else 0
    score += sine_bonus
    score += fft_map['5m']['up_bias'] * 120
    return {
        'symbol': 'unknown',
        'price': current_price,
        'target': target_price,
        'rsi': rsi,
        'score': score,
        'extrema_map': extrema_map,
        'volume_map': volume_map,
        'fft_map': fft_map,
        'phi_map': phi_map,
        'induction_vol': vol_ll,
        'avg_vol': avg_vol,
        'sine_sig_5m': sig_5m,
        'sine_sig_1m': sig_1m,
        'phi_support': extrema_map['5m']['ll']
    }

def check_mtf_dip(symbols, client):
    candidates = []
    timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h']
    def process_symbol(symbol):
        time.sleep(0.25)  # Slightly increased delay to be safer with Binance rate limits
        data_map = {}
        for tf in timeframes:
            data = get_klines_data(client, symbol, tf, 1000)
            if data is None: return None
            data_map[tf] = data
        if any(len(data_map.get(tf, (0,0,0,0,0))[3]) < 50 for tf in ['1m','5m']):
            return None
        result = compute_signal(data_map, timeframes)
        if result:
            result['symbol'] = symbol
            return result
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:  # Reduced workers to lower API pressure
        results = list(executor.map(process_symbol, symbols))
    return [r for r in results if r is not None]

# --- Main Execution ---
if __name__ == "__main__":
    print("MTF Dip Scanner started. Will stop automatically when a valid signal is found.")
    print("Current date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Scanning USDC pairs repeatedly...\n")

    iteration = 0
    while True:
        iteration += 1
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Scan #{iteration} started...")
        
        try:
            trading_pairs = trader.get_usdc_pairs()
            print(f"  → Found {len(trading_pairs)} USDC trading pairs")
            
            candidates = check_mtf_dip(trading_pairs, trader.client)
            
            if candidates:
                candidates.sort(key=lambda x: x['score'], reverse=True)
                best = candidates[0]
                
                print("\n" + "="*120)
                print("!!! STRICT MTF DIP DETECTED - STOPPING SCANNER !!!")
                print("Recent ARGMIN 1m+5m | Majority TFs | Bull Vol 1-15m | Induction Spike | Sine Reversal + FFT Up Bias")
                print("="*120)
                print(f"\n[STRONGEST SIGNAL] {best['symbol']}")
                print(f"Current Price:     {best['price']:.8f} USDC")
                print(f"Target (5m Fib):   {best['target']:.8f} USDC")
                print(f"Algo Score:        {best['score']:.1f}")
                print(f"RSI (5m):          {best['rsi']:.1f}")
                print("-" * 120)
                print(f"{'TF':<5} | {'Extrema':<15} | {'LL':<12} | {'HH':<12} | {'Fib Target':<12} | {'Vol Bull %':<10} | {'Vol Bear %':<10} | {'FFT Up Bias':<12}")
                print("-" * 120)
                
                tf_order = ['1m','3m','5m','15m','30m','1h','2h','4h']
                for tf in tf_order:
                    ext = best['extrema_map'][tf]
                    vol = best['volume_map'][tf]
                    phi = best['phi_map'][tf]
                    fft_b = best['fft_map'][tf]['up_bias']
                    fft_next = best['fft_map'][tf]['forecast'][0] if len(best['fft_map'][tf]['forecast']) > 0 else best['price']
                    fft_dir = "UP" if fft_next > best['price'] else "DOWN"
                    vol_flag = "*" if vol[0] > vol[1] else " "
                    print(f"{tf:<5} | {ext['recent_type']:<15} | {ext['ll']:<12.8f} | {ext['hh']:<12.8f} | {phi:<12.8f} | {vol[0]:<10.2f} | {vol[1]:<10.2f} | {fft_b:.2f} ({fft_dir}) {vol_flag}")
                
                print("-" * 120)
                print("NOTE: '*' = Bullish volume dominance (required in 1m/3m/5m/15m)")
                print("-" * 120)
                print("VOLUME INDUCTION (1m):")
                print(f"  Vol at Low: {best['induction_vol']:.2f} vs Avg: {best['avg_vol']:.2f} (≥1.5× required)")
                print("-" * 120)
                print("SINE SIGNALS:")
                print(f"  5m: {best['sine_sig_5m']} | 1m: {best['sine_sig_1m']}")
                print("="*120)
                
                if len(candidates) > 1:
                    print("\nOther strong candidates:")
                    for i, c in enumerate(candidates[1:6], 1):
                        print(f"  {i}. {c['symbol']:12} | Score: {c['score']:>6.0f} | Price: {c['price']:.6f} | Target: {c['target']:.6f}")
                
                print("\nScanner stopping as requested when signal is found.")
                break  # <--- This line stops the loop when a candidate is found
            
            else:
                print("  → No candidates met the strict criteria this scan.")
            
        except Exception as e:
            print(f"  → Error during scan: {e}")
        
        # Clean memory
        gc.collect()
        
        # Wait before next scan
        print("Sleeping 5 seconds before next scan...\n")
        time.sleep(5)