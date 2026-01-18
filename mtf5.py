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

    def get_account_balance(self):
        account = self.client.get_account()
        for balance in account['balances']:
            if balance['asset'] == 'USDC':
                return float(balance['free'])
        return 0.0

    def get_current_price(self, symbol):
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])

    def execute_buy(self, symbol, usdc_amount):
        pass

    def execute_sell(self, symbol):
        pass

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def update_trade_stats(symbol, profit, profit_pct):
    pass

def print_stats_dashboard():
    win_rate = (WINS / TOTAL_TRADES * 100) if TOTAL_TRADES > 0 else 0.0
    print("\n" + "="*60)
    print(" " * 15 + "MTF SCANNER DASHBOARD")
    print("="*60)
    print("MODE: SIGNAL ONLY (Auto-Exit on Trigger)")
    print(f"Net PnL:      {TOTAL_NET_PNL:.2f} USDC (N/A)")
    print("="*60)

    if TRADE_HISTORY:
        history_to_show = TRADE_HISTORY[::-1][:10] 
        print("\nRECENT TRADE HISTORY:")
        print(tabulate(history_to_show, headers="keys", tablefmt="grid"))
    else:
        print("\nWaiting for signal...")

def format_float(val):
    if val is None: return "-"
    try:
        return f"{float(val):.{PRECISION}f}"
    except:
        return "-"

def print_dynamic_table(full_records):
    clear_screen()
    print_stats_dashboard()
    
    print(f"\nDisplaying All Assets (Sorted by MTF Strength)")

    # Sort Logic for the table display
    def get_score(item):
        symbol, data = item
        score = 0
        
        if data.get('2h_trend') == "PASS": score += 1
        if data.get('15m_trend') == "PASS": score += 1
        if data.get('5m_trend') == "PASS": score += 1
        if data.get('5m_mid') == "PASS": score += 1
        if data.get('1m_rsi_valid') == "YES": score += 1
        if data.get('1m_struct_valid') == "YES": score += 1
        if data.get('vol_signal') == "BULL": score += 1
            
        try:
            bv = float(data.get('bull_vol', 0))
        except: bv = 0
            
        return (-score, -bv, symbol) 

    sorted_assets = sorted(full_records.items(), key=get_score)

    table_data = []
    for symbol, data in sorted_assets:
        row = [
            symbol, 
            data.get('2h_trend', '-'),
            data.get('15m_trend', '-'),
            data.get('5m_trend', '-'),
            data.get('5m_mid', '-'),
            data.get('1m_rsi', '-'),
            "YES" if data.get('1m_rsi_valid') == 'YES' else "NO",
            "YES" if data.get('1m_struct_valid') == 'YES' else "NO",
            format_float(data.get('bull_vol')),
            format_float(data.get('bear_vol'))
        ]
        table_data.append(row)

    headers = [
        "Symbol", "2H Trend", "15M Trend", "5M Trend", "5M Mid", "1M RSI",
        "RSI OK (<50)", "Struct OK", "Bull Vol %", "Bear Vol %"
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))

# --- Analysis Logic ---

def get_regression_breakout_status(close_array):
    """
    Updated Logic:
    1. Takes input array (expected 1200 values).
    2. Uses Length (Lookback) = 360 for the Regression Channel calculation.
    3. Uses Method 1 (Standard Deviation) to define Upper/Lower channels.
    """
    try:
        lookback = 360
        
        # Ensure we have enough data (at least 360)
        if len(close_array) < lookback:
            return "FAIL"
        
        # Analyze the last 360 candles of the 1200 values
        y = close_array[-lookback:]
        x = np.arange(len(y))
        
        # --- Core Components: Linear Regression Line (Midline) ---
        # Formula: y = a + bx
        # Calculate slope (b) and intercept (a)
        slope, intercept = np.polyfit(x, y, 1)
        regression_line = slope * x + intercept
        
        # --- Core Components: Upper & Lower Channel Lines ---
        # Method 1: Standard Deviation
        # Calculate distance of price from regression line
        residuals = y - regression_line
        
        # Calculate Standard Deviation (Sigma)
        std_dev = np.std(residuals)
        
        # Define Channels: Line +/- 2 Standard Deviations
        lower_channel = regression_line - (2 * std_dev)
        upper_channel = regression_line + (2 * std_dev)
        
        # --- Breach Detection (Support/Resistance Logic) ---
        # Indices where price closed BELOW the lower channel (The Dips/Flush)
        indices_low_breach = np.where(y < lower_channel)[0]
        
        # Indices where price closed ABOVE the upper channel (The Spikes/Overextension)
        indices_high_breach = np.where(y > upper_channel)[0]
        
        # If we haven't hit both boundaries, the setup isn't valid
        if len(indices_low_breach) == 0 or len(indices_high_breach) == 0:
            return "FAIL"
            
        # --- Reversal Signal Logic ---
        # Condition 1: Recency
        # The most recent dip (last_low_idx) must be more recent than the last spike (last_high_idx)
        last_low_idx = indices_low_breach[-1]
        last_high_idx = indices_high_breach[-1]
        
        if last_low_idx <= last_high_idx:
            return "FAIL"
        
        # Condition 2: Overbought/Oversold Confirmation (Distance)
        # Current Close distance to the Low Breach < Distance to High Breach
        # This confirms we are currently sitting at the "dip" level, not recovering from it
        current_close = y[-1]
        price_at_low = y[last_low_idx]
        price_at_high = y[last_high_idx]
        
        dist_to_low = abs(current_close - price_at_low)
        dist_to_high = abs(current_close - price_at_high)
        
        if dist_to_low < dist_to_high:
            return "PASS"
            
    except Exception as e:
        # print(f"Regression Logic Error: {e}")
        pass
        
    return "FAIL"

def analyze_single_asset(client, symbol):
    # Initialize all keys with explicit defaults to ensure table is always populated
    results = {
        '2h_trend': 'FAIL', '15m_trend': 'FAIL', '5m_trend': 'FAIL', '5m_mid': 'FAIL',
        '1m_rsi': '-', '1m_rsi_valid': 'NO', 
        'argmin_price': None, 'argmax_price': None, '1m_struct_valid': 'NO',
        'bull_vol': 0.0, 'bear_vol': 0.0, 'vol_signal': 'N/A'
    }
    
    try:
        # --- 2H Analysis (Fetch 1200, Use Last 360) ---
        try:
            k_2h = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_2HOUR, limit=1200)
            if k_2h:
                c_2h = np.array([float(k[4]) for k in k_2h])
                results['2h_trend'] = get_regression_breakout_status(c_2h)
        except: 
            results['2h_trend'] = "ERR"

        # --- 15M Analysis (Fetch 1200, Use Last 360) ---
        try:
            k_15m = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=1200)
            if k_15m:
                c_15m = np.array([float(k[4]) for k in k_15m])
                results['15m_trend'] = get_regression_breakout_status(c_15m)
        except: 
            results['15m_trend'] = "ERR"

        # --- 5M Analysis (Fetch 1200, Use Last 360) ---
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
                except: 
                    results['5m_mid'] = "ERR"
        except: 
            results['5m_trend'] = "ERR"
            results['5m_mid'] = "ERR"

        # --- 1M Analysis (Structure + RSI + Volume) ---
        try:
            # 1M timeframe keeps limit 1200 as standard for high res data
            k_1m = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=1200)
            if not k_1m: raise ValueError("No 1m data")
            
            c_1m = np.array([float(k[4]) for k in k_1m])
            
            # --- 1. Structure Logic (Lowest Low vs Highest High) ---
            try:
                val_min = np.min(c_1m) 
                val_max = np.max(c_1m) 
                
                indices_min = np.where(c_1m == val_min)[0]
                indices_max = np.where(c_1m == val_max)[0]
                
                if len(indices_min) > 0 and len(indices_max) > 0:
                    idx_min = indices_min[-1] 
                    idx_max = indices_max[-1] 
                    
                    results['argmin_price'] = val_min
                    results['argmax_price'] = val_max
                    
                    if idx_min > idx_max:
                        results['1m_struct_valid'] = "YES"
                    else:
                        results['1m_struct_valid'] = "NO"
            except:
                results['1m_struct_valid'] = "ERR"
            
            # --- 2. RSI Logic ---
            try:
                rsi = ta.RSI(c_1m, timeperiod=14)
                valid_rsi = rsi[np.logical_not(np.isnan(rsi))]
                
                if len(valid_rsi) > 0:
                    curr_rsi = valid_rsi[-1]
                    results['1m_rsi'] = f"{curr_rsi:.2f}"
                    
                    idx_oversold = -1
                    idx_overbought = -1
                    
                    # Iterate backwards to find most recent occurrences
                    for i in range(len(valid_rsi) - 1, -1, -1):
                        if valid_rsi[i] < 30 and idx_oversold == -1: 
                            idx_oversold = i
                        if valid_rsi[i] > 70 and idx_overbought == -1: 
                            idx_overbought = i
                        
                        if idx_oversold != -1 and idx_overbought != -1:
                            break
                    
                    cond_recency = (idx_oversold > idx_overbought) if (idx_oversold != -1 and idx_overbought != -1) else False
                    cond_middle = curr_rsi < 50.0
                    
                    if cond_recency and cond_middle:
                        results['1m_rsi_valid'] = "YES"
                    else:
                        results['1m_rsi_valid'] = "NO"
                else:
                    results['1m_rsi'] = "-"
                    results['1m_rsi_valid'] = "NO"
            except:
                results['1m_rsi_valid'] = "ERR"
            
            # --- 3. Volume Logic ---
            try:
                bull = 0.0
                bear = 0.0
                for k in k_1m:
                    try:
                        o = float(k[1])
                        c = float(k[4])
                        v = float(k[5]) 
                        if c > o: 
                            bull += v
                        elif c < o: 
                            bear += v
                    except:
                        continue
                
                total = bull + bear
                if total > 0:
                    b_pct = (bull / total) * 100
                    be_pct = (bear / total) * 100
                    results['bull_vol'] = b_pct
                    results['bear_vol'] = be_pct
                    results['vol_signal'] = "BULL" if b_pct > be_pct else "BEAR"
                else:
                    results['vol_signal'] = "FLAT"
            except:
                results['vol_signal'] = "ERR"
                    
        except Exception:
            # If 1m fetch fails, ensure structure and RSI remain defaults (NO/-)
            pass

    except Exception:
        pass
        
    return results

# --- Main Execution Loop ---

filename = 'api.txt'
if not os.path.exists(filename):
    print("Please create 'api.txt' with API Key and Secret.")
    sys.exit(1)

trader = Trader(filename)
print("Bot started in SIGNAL ONLY mode. Press Ctrl+C to stop.")

try:
    while True:
        try:
            cycle_start = time.time()
            
            list_all_pairs = trader.get_usdc_pairs()
            full_records = {}
            
            print(f"Scanning {len(list_all_pairs)} assets for full MTF data...")

            results_list = []
            MAX_WORKERS = 5
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_sym = {executor.submit(analyze_single_asset, trader.client, s): s for s in list_all_pairs}
                for future in concurrent.futures.as_completed(future_to_sym):
                    sym = future_to_sym[future]
                    try:
                        data = future.result()
                        if not isinstance(data, dict):
                            data = {
                                '2h_trend': 'CRASH', '15m_trend': 'CRASH', '5m_trend': 'CRASH', '5m_mid': 'CRASH',
                                '1m_rsi': '-', '1m_rsi_valid': 'NO', '1m_struct_valid': 'NO',
                                'bull_vol': 0.0, 'bear_vol': 0.0, 'vol_signal': 'ERR'
                            }
                        full_records[sym] = data
                    except Exception:
                        full_records[sym] = {
                            '2h_trend': 'ERR', '15m_trend': 'ERR', '5m_trend': 'ERR', '5m_mid': 'ERR',
                            '1m_rsi': '-', '1m_rsi_valid': 'NO', '1m_struct_valid': 'NO',
                            'bull_vol': 0.0, 'bear_vol': 0.0, 'vol_signal': 'ERR'
                        }

            print_dynamic_table(full_records)
            
            cycle_duration = time.time() - cycle_start
            print(f"\nScan Complete in {cycle_duration:.2f}s")

            # --- Helper to calculate score for ranking ---
            def get_ranking_score(item):
                symbol, data = item
                score = 0
                
                if data.get('2h_trend') == 'PASS': score += 1
                if data.get('15m_trend') == 'PASS': score += 1
                if data.get('5m_trend') == 'PASS': score += 1
                if data.get('5m_mid') == 'PASS': score += 1
                if data.get('1m_struct_valid') == 'YES': score += 1
                if data.get('1m_rsi_valid') == 'YES': score += 1
                if data.get('vol_signal') == 'BULL': score += 1
                
                # Secondary metric: Bull Vol % (Higher is better)
                try:
                    bull_vol = float(data.get('bull_vol', 0))
                except:
                    bull_vol = 0
                
                # Tertiary metric: RSI (Lower is better, so we invert)
                try:
                    rsi_str = data.get('1m_rsi', '100')
                    if rsi_str == '-': rsi_val = 100
                    else: rsi_val = float(rsi_str)
                except:
                    rsi_val = 100
                
                # Return tuple for sorting: (Primary Score, Bull Vol, -RSI, Symbol)
                return (score, bull_vol, -rsi_val, symbol)

            # FIND WINNER (Strict Logic)
            winners = []
            for s in list_all_pairs:
                r = full_records[s]
                if (r.get('2h_trend') == 'PASS' and 
                    r.get('15m_trend') == 'PASS' and 
                    r.get('5m_trend') == 'PASS' and
                    r.get('5m_mid') == 'PASS' and
                    r.get('1m_struct_valid') == 'YES' and
                    r.get('1m_rsi_valid') == 'YES' and 
                    r.get('vol_signal') == 'BULL'):
                    
                    winners.append(s)
            
            if winners:
                valid_winners = []
                for s in winners:
                    try:
                        rsi_str = full_records[s].get('1m_rsi', '100')
                        rsi_val = float(rsi_str)
                        valid_winners.append((s, rsi_val))
                    except: pass
                
                valid_winners.sort(key=lambda x: x[1])
                best_symbol = valid_winners[0][0]
                
                print("\n" + "!"*60)
                print(f" [!!!] PERFECT SETUP FOUND: {best_symbol} [!!!]")
                print("!"*60)
                print("Criteria Met:")
                print(f" - 2H/15M/5M Trend: PASS (Reg Chan Logic)")
                print(f" - 5M Midpoint: PASS")
                print(f" - 1m Structure: Valid")
                print(f" - 1m RSI: {full_records[best_symbol]['1m_rsi']} (<50 & OS > OB)")
                print(f" - Volume: Bullish")
                print("\n>>> SCRIPT EXITING. PLEASE ENTER TRADE MANUALLY. <<<")
                
                sys.exit(0) 
                
            else:
                # --- NEW LOGIC: POTENTIAL CANDIDATE FILTER ---
                
                # Filter strictly for assets that have 2H, 15M, 5M Trend PASS
                potential_matches = []
                for s in list_all_pairs:
                    r = full_records[s]
                    if (r.get('2h_trend') == 'PASS' and 
                        r.get('15m_trend') == 'PASS' and 
                        r.get('5m_trend') == 'PASS'):
                        potential_matches.append(s)
                
                if potential_matches:
                    # Find best among the potential matches
                    best_sym, best_data = max(
                        [(s, full_records[s]) for s in potential_matches], 
                        key=get_ranking_score
                    )
                    
                    score_details = get_ranking_score((best_sym, best_data))
                    
                    print("\n" + "-"*60)
                    print(f" [?] BEST MTF POTENTIAL CANDIDATE: {best_sym} [?]")
                    print(f" Score: {score_details[0]}/7 criteria met")
                    print("-"*60)
                    print(f" 2H Trend: {best_data['2h_trend']} | 15M Trend: {best_data['15m_trend']} | 5M Trend: {best_data['5m_trend']}")
                    print(f" 5M Mid: {best_data['5m_mid']} | 1M Struct: {best_data['1m_struct_valid']} | RSI OK: {best_data['1m_rsi_valid']}")
                    print(f" 1M RSI: {best_data['1m_rsi']} | Vol Signal: {best_data['vol_signal']} ({best_data['bull_vol']:.2f}%)")
                    print("-"*60)
                    print(">>> SCRIPT EXITING. PLEASE REVIEW BEST CANDIDATE. <<<")
                    
                    sys.exit(0) 
                else:
                    # No assets meet the minimum Trend criteria
                    print("No assets meet minimum trend criteria (2H/15M/5M PASS). Waiting 5s...")
                    time.sleep(5)
                    continue

        except KeyboardInterrupt:
            print("\nStopped by user.")
            sys.exit(0)

except KeyboardInterrupt:
    sys.exit(0)
