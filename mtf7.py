from binance.client import Client
from binance.exceptions import BinanceAPIException
import numpy as np
import talib as ta
import concurrent.futures
from tabulate import tabulate
import os
import sys
import time
from datetime import datetime

# --- Global Stats Variables ---
TRADE_HISTORY = []
TOTAL_NET_PNL = 0.0
TOTAL_TRADES = 0
WINS = 0
LOSSES = 0

# Formatting constant
PRECISION = 25

class Trader:
    def __init__(self, file):
        self.connect(file)

    def connect(self, file):
        if not os.path.exists(file):
            print(f"API file {file} not found.")
            sys.exit(1)
        with open(file) as f:
            lines = [line.rstrip('\n') for line in f]
            if len(lines) < 2:
                print("API file format incorrect. Expected key and secret.")
                sys.exit(1)
            key = lines[0]
            secret = lines[1]
            self.client = Client(key, secret)

    def get_usdc_pairs(self):
        exchange_info = self.client.get_exchange_info()
        trading_pairs = [
            symbol['symbol'] 
            for symbol in exchange_info['symbols'] 
            if symbol['quoteAsset'] == 'USDC' 
            and symbol['status'] == 'TRADING'
            and symbol['symbol'].isascii() 
        ]
        return trading_pairs

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def format_float(val):
    if val is None: return "-"
    try:
        return f"{float(val):.{PRECISION}f}"
    except:
        return "-"

def print_stats_dashboard():
    win_rate = (WINS / TOTAL_TRADES * 100) if TOTAL_TRADES > 0 else 0.0
    print("\n" + "="*80)
    print(" " * 25 + "MTF SCANNER DASHBOARD")
    print("="*80)
    print("MODE: BEST UP-CYCLE CANDIDATE SEARCH")

def get_score(item):
    """
    Calculates a score based on how many criteria are met.
    Used to sort and find the 'Best' candidate.
    """
    symbol, data = item
    score = 0
    
    # MTF Trend Confirmation (1 point each)
    if data.get('2h_trend') == "PASS": score += 1
    if data.get('15m_trend') == "PASS": score += 1
    if data.get('5m_trend') == "PASS": score += 1
    if data.get('5m_mid') == "PASS": score += 1
    
    # RSI Momentum (1 point)
    if data.get('1m_rsi_valid') == "YES": score += 1
    
    # Structure & Price Action (Critical) (2 + 1 points)
    if data.get('1m_hl_struct') == "YES": score += 2
    if data.get('1m_price_up') == "YES": score += 1
    
    # Volume Confirmation (1 + 1 points)
    if data.get('1m_vol_dom') == "BULL": score += 1
    if data.get('1m_vol_inc') == "YES": score += 1
    
    # Spike Power Bonus
    try:
        score += float(data.get('spike_power', 0)) * 5
    except: pass
            
    try:
        bv = float(data.get('bull_vol', 0))
    except: bv = 0
        
    # Return tuple for sorting: (Negative Score for Descending, Negative Bull Vol for Tiebreaker, Symbol)
    return (-score, -bv, symbol) 

def print_dynamic_table(full_records, highlight_symbol=None):
    clear_screen()
    print_stats_dashboard()
    
    print(f"\nDisplaying Top Assets (Sorted by Up-Cycle Score)")

    # Sort assets
    sorted_assets = sorted(full_records.items(), key=get_score)
    
    # Limit view to top 20
    if highlight_symbol:
        # Ensure highlight is visible at top or within list
        filtered = {k:v for k,v in sorted_assets if k == highlight_symbol}
        for s, d in sorted_assets:
            if s not in filtered and len(filtered) < 20:
                filtered[s] = d
        sorted_assets = list(filtered.items())
    else:
        sorted_assets = sorted_assets[:20]

    table_data = []
    for symbol, data in sorted_assets:
        row = [
            symbol, 
            data.get('2h_trend', '-'),
            data.get('15m_trend', '-'),
            data.get('5m_trend', '-'),
            data.get('5m_mid', '-'),
            data.get('1m_rsi', '-'),
            "HL" if data.get('1m_hl_struct') == 'YES' else "LL",
            "UP" if data.get('1m_price_up') == 'YES' else "--",
            "DOM" if data.get('1m_vol_dom') == 'BULL' else "WEAK",
            "INC" if data.get('1m_vol_inc') == 'YES' else "FLAT",
            f"{data.get('spike_power', 0):.2f}",
            format_float(data.get('bull_vol')),
            format_float(data.get('bear_vol'))
        ]
        table_data.append(row)

    headers = [
        "Symbol", "2H Trnd", "15M Trnd", "5M Trnd", "5M Mid", "RSI", 
        "Struct", "Pri Act", "Vol Dom", "Vol Mom", "Power", "Bull %", "Bear %"
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))

def get_regression_breakout_status(close_array):
    try:
        lookback = 360
        if len(close_array) < lookback:
            return "FAIL"
        
        y = close_array[-lookback:]
        x = np.arange(len(y))
        
        slope, intercept = np.polyfit(x, y, 1)
        regression_line = slope * x + intercept
        residuals = y - regression_line
        std_dev = np.std(residuals)
        
        lower_channel = regression_line - (2 * std_dev)
        upper_channel = regression_line + (2 * std_dev)
        
        indices_low_breach = np.where(y < lower_channel)[0]
        indices_high_breach = np.where(y > upper_channel)[0]
        
        if len(indices_low_breach) == 0 or len(indices_high_breach) == 0:
            return "FAIL"
            
        last_low_idx = indices_low_breach[-1]
        last_high_idx = indices_high_breach[-1]
        
        if last_low_idx <= last_high_idx:
            return "FAIL"
        
        current_close = y[-1]
        price_at_low = y[last_low_idx]
        price_at_high = y[last_high_idx]
        
        dist_to_low = abs(current_close - price_at_low)
        dist_to_high = abs(current_close - price_at_high)
        
        if dist_to_low < dist_to_high:
            return "PASS"
            
    except Exception as e:
        pass
        
    return "FAIL"

def analyze_single_asset(client, symbol):
    results = {
        '2h_trend': 'FAIL', '15m_trend': 'FAIL', '5m_trend': 'FAIL', '5m_mid': 'FAIL',
        '1m_rsi': '-', '1m_rsi_valid': 'NO', 
        '1m_hl_struct': 'NO', '1m_price_up': 'NO', '1m_vol_dom': 'BEAR', '1m_vol_inc': 'NO',
        'bull_vol': 0.0, 'bear_vol': 0.0, 'vol_signal': 'N/A', 'spike_power': 0.0
    }
    
    try:
        # --- 2H Analysis ---
        try:
            k_2h = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_2HOUR, limit=1200)
            if k_2h:
                c_2h = np.array([float(k[4]) for k in k_2h])
                results['2h_trend'] = get_regression_breakout_status(c_2h)
        except: results['2h_trend'] = "ERR"

        # --- 15M Analysis ---
        try:
            k_15m = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=1200)
            if k_15m:
                c_15m = np.array([float(k[4]) for k in k_15m])
                results['15m_trend'] = get_regression_breakout_status(c_15m)
        except: results['15m_trend'] = "ERR"

        # --- 5M Analysis (Trend + Midpoint) ---
        try:
            k_5m = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_5MINUTE, limit=1200)
            if k_5m:
                c_5m = np.array([float(k[4]) for k in k_5m])
                results['5m_trend'] = get_regression_breakout_status(c_5m)
                
                try:
                    idx_min = np.argmin(c_5m)
                    idx_max = np.argmax(c_5m)
                    val_max = c_5m[idx_max]
                    val_min = c_5m[idx_min]
                    middle_threshold = (val_max + val_min) / 2
                    is_mid = c_5m[-1] < middle_threshold and idx_min > idx_max
                    results['5m_mid'] = "PASS" if is_mid else "FAIL"
                except: results['5m_mid'] = "ERR"
        except: 
            results['5m_trend'] = "ERR"
            results['5m_mid'] = "ERR"

        # --- 1M Analysis ---
        try:
            k_1m = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=1200)
            if not k_1m: raise ValueError("No 1m data")
            
            c_1m = np.array([float(k[4]) for k in k_1m])
            o_1m = np.array([float(k[1]) for k in k_1m])
            v_1m = np.array([float(k[5]) for k in k_1m])
            
            # RSI
            try:
                rsi = ta.RSI(c_1m, timeperiod=14)
                valid_rsi = rsi[np.logical_not(np.isnan(rsi))]
                if len(valid_rsi) > 0:
                    curr_rsi = valid_rsi[-1]
                    results['1m_rsi'] = f"{curr_rsi:.2f}"
                    idx_oversold, idx_overbought = -1, -1
                    for i in range(len(valid_rsi) - 1, -1, -1):
                        if valid_rsi[i] < 30 and idx_oversold == -1: idx_oversold = i
                        if valid_rsi[i] > 70 and idx_overbought == -1: idx_overbought = i
                        if idx_oversold != -1 and idx_overbought != -1: break
                    
                    if idx_oversold > idx_overbought and curr_rsi < 50.0:
                        results['1m_rsi_valid'] = "YES"
            except: pass

            # Structure (Higher Low)
            try:
                abs_low_idx = np.argmin(c_1m)
                abs_low_val = c_1m[abs_low_idx]
                
                if abs_low_idx < len(c_1m) - 10:
                    data_after_abs_low = c_1m[abs_low_idx + 1:]
                    if len(data_after_abs_low) > 0:
                        rel_low_idx_in_slice = np.argmin(data_after_abs_low)
                        last_low_idx = abs_low_idx + 1 + rel_low_idx_in_slice
                        last_low_val = c_1m[last_low_idx]
                        
                        if last_low_val > abs_low_val:
                            results['1m_hl_struct'] = "YES"
                            
                            # Momentum from HL
                            segment_closes = c_1m[last_low_idx:]
                            segment_opens = o_1m[last_low_idx:]
                            segment_vols = v_1m[last_low_idx:]
                            curr_p = c_1m[-1]
                            
                            if curr_p > last_low_val:
                                results['1m_price_up'] = "YES"
                            
                            is_bull_candle = segment_closes > segment_opens
                            is_bear_candle = segment_closes < segment_opens
                            bull_vol_segment = np.sum(np.where(is_bull_candle, segment_vols, 0))
                            bear_vol_segment = np.sum(np.where(is_bear_candle, segment_vols, 0))
                            
                            if bull_vol_segment > bear_vol_segment:
                                results['1m_vol_dom'] = "BULL"
                            
                            if len(c_1m) > last_low_idx + 3:
                                recent_bull_mask = (c_1m[-3:] > o_1m[-3:])
                                recent_bull_mean = np.mean(np.where(recent_bull_mask, v_1m[-3:], 0))
                                if last_low_idx >= 3:
                                    pre_dip_bull_mask = (c_1m[last_low_idx-3:last_low_idx] > o_1m[last_low_idx-3:last_low_idx])
                                    pre_dip_bull_mean = np.mean(np.where(pre_dip_bull_mask, v_1m[last_low_idx-3:last_low_idx], 0))
                                    if recent_bull_mean > pre_dip_bull_mean and recent_bull_mean > 0:
                                        results['1m_vol_inc'] = "YES"
            except Exception as e: pass
            
            # Spike Power & Total Vol
            try:
                is_bull_total = c_1m > o_1m
                is_bear_total = c_1m < o_1m
                bull_total = np.sum(np.where(is_bull_total, v_1m, 0))
                bear_total = np.sum(np.where(is_bear_total, v_1m, 0))
                total_vol = bull_total + bear_total
                
                if total_vol > 0:
                    results['bull_vol'] = (bull_total / total_vol) * 100
                    results['bear_vol'] = (bear_total / total_vol) * 100
                    
                    imb = (bull_total - bear_total) / total_vol
                    rsi_val = float(results['1m_rsi']) if results['1m_rsi'] != '-' else 50
                    rsi_strength = (100 - rsi_val) / 100 if rsi_val < 50 else 0
                    
                    recent_vol_avg = np.mean(v_1m[-10:])
                    avg_vol = np.mean(v_1m)
                    vol_vel = (recent_vol_avg - avg_vol) / avg_vol if avg_vol > 0 else 0
                    
                    power = abs(imb) * rsi_strength * (1 + vol_vel)
                    results['spike_power'] = power
            except: pass

        except Exception: pass

    except Exception: pass
    return results

def get_forecast_data(client, symbol):
    forecast = {
        'symbol': symbol,
        'fft_cycle': '-',
        'gann_levels': [],
        'lr_fast': 0.0, 'lr_int': 0.0, 'lr_big': 0.0,
        'trans_low': 0.0, 'trans_high': 0.0, 'trans_target': 0.0,
        'current_price': 0.0
    }
    
    try:
        k_1m = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=1200)
        closes = np.array([float(k[4]) for k in k_1m])
        forecast['current_price'] = closes[-1]
        
        # Linear Regression Forecast
        lr_lookback = 120
        if len(closes) > lr_lookback:
            y = closes[-lr_lookback:]
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            std = np.std(y - (slope * x + intercept))
            
            # Project Forward
            forecast['lr_fast'] = (slope * (len(y) + 10) + intercept) + std
            forecast['lr_int'] = (slope * (len(y) + 30) + intercept) + (2 * std)
            forecast['lr_big'] = (slope * (len(y) + 60) + intercept) + (3 * std)

        # FFT Analysis
        if len(closes) > 200:
            window = 50
            ma = ta.MA(closes, timeperiod=window)
            detrended = closes[window:] - ma[window:]
            detrended = detrended[~np.isnan(detrended)]
            
            if len(detrended) > 0:
                fft_res = np.fft.rfft(detrended)
                fft_freq = np.fft.rfftfreq(len(detrended))
                power = np.abs(fft_res)
                
                if len(power) > 2:
                    peak_idx = np.argmax(power[1:-1]) + 1
                    freq = fft_freq[peak_idx]
                    if freq > 0:
                        period = 1 / freq
                        forecast['fft_cycle'] = f"{period:.1f} mins"

        # Gann Octaves
        p_min = np.min(closes)
        p_max = np.max(closes)
        rng = p_max - p_min
        if rng > 0:
            for i in range(1, 9):
                lvl = p_min + (rng * (i/8))
                forecast['gann_levels'].append(lvl)
        else:
            forecast['gann_levels'] = [closes[-1]]

        # Argmin/Argmax Logic
        abs_min_idx = np.argmin(closes)
        abs_min_val = closes[abs_min_idx]
        
        if abs_min_idx < len(closes) - 1:
            data_after_low = closes[abs_min_idx:]
            rel_max_idx_in_slice = np.argmax(data_after_low)
            abs_max_idx = abs_min_idx + rel_max_idx_in_slice
            abs_max_val = closes[abs_max_idx]
            
            forecast['trans_low'] = abs_min_val
            forecast['trans_high'] = abs_max_val
            
            current = closes[-1]
            if current < abs_max_val:
                forecast['trans_target'] = abs_max_val
            else:
                rng = abs_max_val - abs_min_val
                forecast['trans_target'] = abs_max_val + rng
                
    except Exception as e:
        pass
        
    return forecast

# --- Main Execution Loop ---

filename = 'api.txt'
if not os.path.exists(filename):
    print("Please create 'api.txt' with API Key and Secret.")
    sys.exit(1)

trader = Trader(filename)
print("Bot started in BEST UP-CYCLE CANDIDATE SEARCH mode.")

# Define Minimum Score Threshold for "Best Candidate"
# Max score approx 10. We want "Most" criteria met, so threshold 6 (60%)
MIN_SCORE_THRESHOLD = 6

try:
    while True:
        try:
            cycle_start = time.time()
            list_all_pairs = trader.get_usdc_pairs()
            full_records = {}
            
            print(f"Scanning {len(list_all_pairs)} assets...")

            MAX_WORKERS = 5
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_sym = {executor.submit(analyze_single_asset, trader.client, s): s for s in list_all_pairs}
                for future in concurrent.futures.as_completed(future_to_sym):
                    sym = future_to_sym[future]
                    try:
                        data = future.result()
                        if not isinstance(data, dict):
                            data = {k:'ERR' for k in ['2h_trend','15m_trend','5m_trend','5m_mid','1m_rsi','1m_rsi_valid','1m_hl_struct','1m_price_up','1m_vol_dom','1m_vol_inc','bull_vol','bear_vol','spike_power']}
                        full_records[sym] = data
                    except Exception:
                        full_records[sym] = {k:'ERR' for k in ['2h_trend','15m_trend','5m_trend','5m_mid','1m_rsi','1m_rsi_valid','1m_hl_struct','1m_price_up','1m_vol_dom','1m_vol_inc','bull_vol','bear_vol','spike_power']}

            # Sort all assets by Score to find the best one
            # get_score returns tuple, sorting handles it automatically
            sorted_assets = sorted(full_records.items(), key=get_score)
            
            # --- STEP 1: CHECK FOR PERFECT SETUP ---
            # (Defined as all major criteria passing)
            winners = []
            for s in list_all_pairs:
                r = full_records[s]
                # MTF
                if r.get('2h_trend') != 'PASS': continue
                if r.get('15m_trend') != 'PASS': continue
                if r.get('5m_trend') != 'PASS': continue
                if r.get('5m_mid') != 'PASS': continue
                # 1M Logic
                if r.get('1m_hl_struct') != "YES": continue
                if r.get('1m_price_up') != "YES": continue
                if r.get('1m_vol_dom') != "BULL": continue
                if r.get('1m_vol_inc') != "YES": continue
                if r.get('1m_rsi_valid') != "YES": continue
                
                winners.append(s)
            
            if winners:
                valid_winners = []
                for s in winners:
                    try:
                        rsi_val = float(full_records[s].get('1m_rsi', '100'))
                        valid_winners.append((s, rsi_val))
                    except: pass
                valid_winners.sort(key=lambda x: x[1])
                best_symbol = valid_winners[0][0]
                
                print_dynamic_table(full_records, highlight_symbol=best_symbol)
                print("\n" + "!"*80)
                print(f" [!!!] PERFECT MTF DIP FOUND: {best_symbol} [!!!]")
                print("!"*80)
                print("Criteria Met: ALL Rules Passed")
                print("\n>>> SCRIPT EXITING. PERFECT SETUP FOUND. <<<")
                sys.exit(0)
            
            # --- STEP 2: FIND BEST UP-CYCLE CANDIDATE ---
            # If no perfect setup, pick the highest scoring asset
            best_sym, best_data = sorted_assets[0]
            
            # Calculate raw score for the best asset
            # get_score returns (-score, -bv, sym), so we take -index 0
            raw_score = -get_score((best_sym, best_data))[0]
            
            # CRITICAL CHECK: Does it fill "most" criteria?
            # We require Score >= Threshold AND Core Up-Cycle Pillars (HL Structure, Price Up, Bull Vol)
            is_structural_upcycle = (best_data.get('1m_hl_struct') == "YES" and 
                                     best_data.get('1m_price_up') == "YES" and 
                                     best_data.get('1m_vol_dom') == "BULL")
            
            if raw_score >= MIN_SCORE_THRESHOLD and is_structural_upcycle:
                # This is the best candidate filling most criteria for an up-cycle. Stop Bot.
                print_dynamic_table(full_records, highlight_symbol=best_sym)
                
                # Generate Forecast
                fc = get_forecast_data(trader.client, best_sym)
                
                print("\n" + "="*80)
                print(f" [>>>] BEST UP-CYCLE CANDIDATE FOUND: {best_sym} (Score: {raw_score})")
                print("="*80)
                
                # Score Summary
                print(f"Score Summary:")
                print(f"  - 2H Trend: {best_data.get('2h_trend')} ({'PASS' if best_data.get('2h_trend')=='PASS' else 'FAIL'})")
                print(f"  - 15M Trend: {best_data.get('15m_trend')} ({'PASS' if best_data.get('15m_trend')=='PASS' else 'FAIL'})")
                print(f"  - 5M Trend: {best_data.get('5m_trend')} ({'PASS' if best_data.get('5m_trend')=='PASS' else 'FAIL'})")
                print(f"  - 5M Mid: {best_data.get('5m_mid')} ({'PASS' if best_data.get('5m_mid')=='PASS' else 'FAIL'})")
                print(f"  - RSI Valid: {best_data.get('1m_rsi_valid')}")
                print(f"  - HL Structure: {best_data.get('1m_hl_struct')} (CRITICAL)")
                print(f"  - Price Action: {best_data.get('1m_price_up')} (CRITICAL)")
                print(f"  - Volume Dom: {best_data.get('1m_vol_dom')} (CRITICAL)")
                print(f"  - Volume Mom: {best_data.get('1m_vol_inc')}")
                print(f"  - Spike Power: {best_data.get('spike_power', 0):.4f}")
                
                # Forecast
                print(f"\n--- ADVANCED FORECAST ({best_sym}) ---")
                print(f"[1] Cycle Transformation (Last 1200 Candles):")
                print(f"    - Lowest Low (Argmin): {fc['trans_low']:.6f}")
                print(f"    - Highest High (Argmax): {fc['trans_high']:.6f}")
                print(f"    - Target in Argmax Area: {fc['trans_target']:.6f}")
                
                print(f"\n[2] Linear Regression Forecast Targets:")
                print(f"    - Current Price: {fc['current_price']:.6f}")
                pct_f = ((fc['lr_fast'] - fc['current_price'])/fc['current_price'])*100
                pct_i = ((fc['lr_int'] - fc['current_price'])/fc['current_price'])*100
                pct_b = ((fc['lr_big'] - fc['current_price'])/fc['current_price'])*100
                
                print(f"    - Fast Target (+10m): {fc['lr_fast']:.6f} ({pct_f:.2f}%)")
                print(f"    - Int. Target (+30m): {fc['lr_int']:.6f} ({pct_i:.2f}%)")
                print(f"    - Big Target (+60m): {fc['lr_big']:.6f} ({pct_b:.2f}%)")
                
                print(f"\n[3] FFT Dominant Cycle: {fc['fft_cycle']}")
                
                print(f"\n[4] Gann Octave Levels:")
                gann_str = " | ".join([f"{x:.4f}" for x in fc['gann_levels']])
                print(f"    {gann_str}")

                print("\n>>> SCRIPT EXITING. BEST UP-CYCLE CANDIDATE SELECTED. <<<")
                sys.exit(0)

            else:
                # Candidate score is too low or not structurally an up-cycle. Continue Loop.
                print_dynamic_table(full_records, highlight_symbol=best_sym)
                print(f"\n[INFO] Best Candidate ({best_sym}) Score: {raw_score}. (Need >= {MIN_SCORE_THRESHOLD} & Up-Cycle Structure)")
                print("      Continuing search...")
                
                cycle_duration = time.time() - cycle_start
                print(f"Scan Complete in {cycle_duration:.2f}s. Waiting 5s...")
                time.sleep(5)
                continue

        except KeyboardInterrupt:
            print("\nStopped by user.")
            sys.exit(0)

except KeyboardInterrupt:
    sys.exit(0)