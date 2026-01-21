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
    print("\n" + "="*80)
    print(" " * 25 + "MTF SCANNER DASHBOARD")
    print("="*80)
    print("MODE: SIGNAL ONLY (Auto-Exit on Trigger)")
    print(f"Net PnL:      {TOTAL_NET_PNL:.2f} USDC (N/A)")
    print("="*80)

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

    # Sort Logic
    def get_score(item):
        symbol, data = item
        score = 0
        
        if data.get('2h_trend') == "PASS": score += 1
        if data.get('15m_trend') == "PASS": score += 1
        if data.get('5m_trend') == "PASS": score += 1
        if data.get('5m_mid') == "PASS": score += 1
        if data.get('1m_rsi_valid') == "YES": score += 1
        
        # New Logic Scoring
        if data.get('1m_hl_struct') == "YES": score += 2 # Higher Low is critical
        if data.get('1m_price_up') == "YES": score += 1
        if data.get('1m_vol_dom') == "BULL": score += 1
        if data.get('1m_vol_inc') == "YES": score += 1
            
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
            "HL" if data.get('1m_hl_struct') == 'YES' else "LL",
            "UP" if data.get('1m_price_up') == 'YES' else "--",
            "DOM" if data.get('1m_vol_dom') == 'BULL' else "WEAK",
            "INC" if data.get('1m_vol_inc') == 'YES' else "FLAT",
            format_float(data.get('bull_vol')),
            format_float(data.get('bear_vol'))
        ]
        table_data.append(row)

    headers = [
        "Symbol", "2H Trnd", "15M Trnd", "5M Trnd", "5M Mid", "RSI", 
        "Struct", "Pri Act", "Vol Dom", "Vol Mom", "Bull %", "Bear %"
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))

# --- Analysis Logic ---

def get_regression_breakout_status(close_array):
    """
    New Logic: 
    Regression Channel with length 360.
    Identifies if price is near a 'Dip' zone (closer to lower channel breach) 
    and that the low breach happened after the high breach.
    """
    try:
        lookback = 360
        if len(close_array) < lookback:
            return "FAIL"
        
        y = close_array[-lookback:]
        x = np.arange(len(y))
        
        # 1. Linear Regression
        slope, intercept = np.polyfit(x, y, 1)
        regression_line = slope * x + intercept
        
        # 2. Calculate Channel (2 SD)
        residuals = y - regression_line
        std_dev = np.std(residuals)
        
        lower_channel = regression_line - (2 * std_dev)
        upper_channel = regression_line + (2 * std_dev)
        
        # 3. Identify Breaches
        indices_low_breach = np.where(y < lower_channel)[0]
        indices_high_breach = np.where(y > upper_channel)[0]
        
        if len(indices_low_breach) == 0 or len(indices_high_breach) == 0:
            return "FAIL"
            
        last_low_idx = indices_low_breach[-1]
        last_high_idx = indices_high_breach[-1]
        
        # 4. Recency: Dip must have happened AFTER the spike
        if last_low_idx <= last_high_idx:
            return "FAIL"
        
        # 5. Distance: Current Close is closer to the Dip price than the Spike price
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
    # Initialize all keys
    results = {
        '2h_trend': 'FAIL', '15m_trend': 'FAIL', '5m_trend': 'FAIL', '5m_mid': 'FAIL',
        '1m_rsi': '-', '1m_rsi_valid': 'NO', 
        '1m_hl_struct': 'NO', '1m_price_up': 'NO', '1m_vol_dom': 'BEAR', '1m_vol_inc': 'NO',
        'bull_vol': 0.0, 'bear_vol': 0.0, 'vol_signal': 'N/A'
    }
    
    try:
        # --- 2H Analysis ---
        try:
            k_2h = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_2HOUR, limit=1200)
            if k_2h:
                c_2h = np.array([float(k[4]) for k in k_2h])
                results['2h_trend'] = get_regression_breakout_status(c_2h)
        except: 
            results['2h_trend'] = "ERR"

        # --- 15M Analysis ---
        try:
            k_15m = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=1200)
            if k_15m:
                c_15m = np.array([float(k[4]) for k in k_15m])
                results['15m_trend'] = get_regression_breakout_status(c_15m)
        except: 
            results['15m_trend'] = "ERR"

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
                except: 
                    results['5m_mid'] = "ERR"
        except: 
            results['5m_trend'] = "ERR"
            results['5m_mid'] = "ERR"

        # --- 1M Analysis (Structure + RSI + Volume + NEW Logic) ---
        try:
            k_1m = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=1200)
            if not k_1m: raise ValueError("No 1m data")
            
            c_1m = np.array([float(k[4]) for k in k_1m])
            o_1m = np.array([float(k[1]) for k in k_1m])
            v_1m = np.array([float(k[5]) for k in k_1m])
            
            # --- 1. RSI Logic ---
            try:
                rsi = ta.RSI(c_1m, timeperiod=14)
                valid_rsi = rsi[np.logical_not(np.isnan(rsi))]
                
                if len(valid_rsi) > 0:
                    curr_rsi = valid_rsi[-1]
                    results['1m_rsi'] = f"{curr_rsi:.2f}"
                    
                    idx_oversold = -1
                    idx_overbought = -1
                    
                    for i in range(len(valid_rsi) - 1, -1, -1):
                        if valid_rsi[i] < 30 and idx_oversold == -1: idx_oversold = i
                        if valid_rsi[i] > 70 and idx_overbought == -1: idx_overbought = i
                        if idx_oversold != -1 and idx_overbought != -1: break
                    
                    cond_recency = (idx_oversold > idx_overbought) if (idx_oversold != -1 and idx_overbought != -1) else False
                    cond_middle = curr_rsi < 50.0
                    
                    if cond_recency and cond_middle:
                        results['1m_rsi_valid'] = "YES"
            except: pass

            # --- 2. Structure Logic (Higher Low - HL) ---
            # "Last Low is higher than the Lowest Low (absolute bottom)"
            try:
                # Absolute Lowest Point
                abs_low_idx = np.argmin(c_1m)
                abs_low_val = c_1m[abs_low_idx]
                
                # We need a "Last Low" that occurred AFTER the absolute lowest point
                # If abs_low is the very last candle, we can't form a HL yet.
                if abs_low_idx < len(c_1m) - 10:
                    # Look for the lowest point in the data AFTER the absolute low
                    data_after_abs_low = c_1m[abs_low_idx + 1:]
                    if len(data_after_abs_low) > 0:
                        rel_low_idx_in_slice = np.argmin(data_after_abs_low)
                        # This is our "Last Low" candidate
                        last_low_idx = abs_low_idx + 1 + rel_low_idx_in_slice
                        last_low_val = c_1m[last_low_idx]
                        
                        # Check Condition: Last Low > Lowest Low
                        if last_low_val > abs_low_val:
                            results['1m_hl_struct'] = "YES"
                            
                            # --- 3. Price & Volume Momentum Logic (Based on Last Low) ---
                            # Analyze data from last_low_idx to current
                            segment_closes = c_1m[last_low_idx:]
                            segment_opens = o_1m[last_low_idx:]
                            segment_vols = v_1m[last_low_idx:]
                            
                            # Current Price
                            curr_p = c_1m[-1]
                            
                            # Check 1: Price Increasing (Current Price > Last Low Price)
                            if curr_p > last_low_val:
                                results['1m_price_up'] = "YES"
                            
                            # Volume Analysis
                            # Calculate Bull vs Bear volume in the segment
                            is_bull_candle = segment_closes > segment_opens
                            is_bear_candle = segment_closes < segment_opens
                            
                            bull_vol_segment = np.sum(np.where(is_bull_candle, segment_vols, 0))
                            bear_vol_segment = np.sum(np.where(is_bear_candle, segment_vols, 0))
                            
                            # Check 2: Bullish Volume Dominance in the move
                            if bull_vol_segment > bear_vol_segment:
                                results['1m_vol_dom'] = "BULL"
                            
                            # Check 3: Bullish Volume INCREASING
                            # We compare the recent volume (last 3 candles) to the volume before the last low
                            if len(c_1m) > last_low_idx + 3:
                                # Mean Bull Vol of recent 3 candles
                                recent_bull_mask = (c_1m[-3:] > o_1m[-3:])
                                recent_bull_mean = np.mean(np.where(recent_bull_mask, v_1m[-3:], 0))
                                
                                # Mean Bull Vol of 3 candles BEFORE the last low (the dip)
                                if last_low_idx >= 3:
                                    pre_dip_bull_mask = (c_1m[last_low_idx-3:last_low_idx] > o_1m[last_low_idx-3:last_low_idx])
                                    pre_dip_bull_mean = np.mean(np.where(pre_dip_bull_mask, v_1m[last_low_idx-3:last_low_idx], 0))
                                    
                                    # Condition: Recent Bull Vol > Pre-Dip Bull Vol
                                    if recent_bull_mean > pre_dip_bull_mean and recent_bull_mean > 0:
                                        results['1m_vol_inc'] = "YES"

            except Exception as e:
                # Fallback for Structure if logic fails
                pass
            
            # --- 4. Total Volume Stats (For Table) ---
            try:
                is_bull_total = c_1m > o_1m
                is_bear_total = c_1m < o_1m
                bull_total = np.sum(np.where(is_bull_total, v_1m, 0))
                bear_total = np.sum(np.where(is_bear_total, v_1m, 0))
                total_vol = bull_total + bear_total
                
                if total_vol > 0:
                    results['bull_vol'] = (bull_total / total_vol) * 100
                    results['bear_vol'] = (bear_total / total_vol) * 100
            except: pass

        except Exception:
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
                            data = {}
                            data['2h_trend'] = 'CRASH'
                            data['15m_trend'] = 'CRASH'
                            data['5m_trend'] = 'CRASH'
                            data['5m_mid'] = 'CRASH'
                            data['1m_rsi'] = '-'
                            data['1m_rsi_valid'] = 'NO'
                            data['1m_hl_struct'] = 'NO'
                            data['1m_price_up'] = 'NO'
                            data['1m_vol_dom'] = 'BEAR'
                            data['1m_vol_inc'] = 'NO'
                            data['bull_vol'] = 0.0
                            data['bear_vol'] = 0.0
                        full_records[sym] = data
                    except Exception:
                        full_records[sym] = {
                            '2h_trend': 'ERR', '15m_trend': 'ERR', '5m_trend': 'ERR', '5m_mid': 'ERR',
                            '1m_rsi': '-', '1m_rsi_valid': 'NO', '1m_hl_struct': 'NO',
                            '1m_price_up': 'NO', '1m_vol_dom': 'BEAR', '1m_vol_inc': 'NO',
                            'bull_vol': 0.0, 'bear_vol': 0.0, 'vol_signal': 'ERR'
                        }

            print_dynamic_table(full_records)
            
            cycle_duration = time.time() - cycle_start
            print(f"\nScan Complete in {cycle_duration:.2f}s")

            # FIND WINNER (Strict Logic Updated)
            winners = []
            for s in list_all_pairs:
                r = full_records[s]
                
                # MTF Filters
                mtf_pass = (r.get('2h_trend') == 'PASS' and 
                            r.get('15m_trend') == 'PASS' and 
                            r.get('5m_trend') == 'PASS' and
                            r.get('5m_mid') == 'PASS')
                
                if not mtf_pass: continue
                
                # 1M Entry Filters (The New Logic)
                # 1. Structure: Higher Low formed (Last Low > Lowest Low)
                struct_ok = (r.get('1m_hl_struct') == "YES")
                
                # 2. Momentum: Price is increasing off that Higher Low
                price_ok = (r.get('1m_price_up') == "YES")
                
                # 3. Volume Confirmation: Bullish Dominance AND Increasing Bullish Volume
                vol_ok = (r.get('1m_vol_dom') == "BULL" and r.get('1m_vol_inc') == "YES")
                
                # 4. RSI Filter
                rsi_ok = (r.get('1m_rsi_valid') == "YES")
                
                if struct_ok and price_ok and vol_ok and rsi_ok:
                    winners.append(s)
            
            if winners:
                # Sort winners by lowest RSI to find the most oversold valid setup
                valid_winners = []
                for s in winners:
                    try:
                        rsi_str = full_records[s].get('1m_rsi', '100')
                        rsi_val = float(rsi_str)
                        valid_winners.append((s, rsi_val))
                    except: pass
                
                valid_winners.sort(key=lambda x: x[1])
                best_symbol = valid_winners[0][0]
                
                print("\n" + "!"*80)
                print(f" [!!!] BEST MTF DIP FOUND: {best_symbol} [!!!]")
                print("!"*80)
                print("Criteria Met:")
                print(f" - 2H/15M/5M Trend: PASS (Regression Dip Logic)")
                print(f" - 5M Midpoint: PASS")
                print(f" - 1m Structure: HIGHER LOW (Last Low > Lowest Low)")
                print(f" - 1m Price Action: INCREASING (off HL)")
                print(f" - 1m Volume: BULLISH DOMINANT & INCREASING")
                print(f" - 1m RSI: {full_records[best_symbol]['1m_rsi']} (<50 & OS > OB)")
                print("\n>>> SCRIPT EXITING. PLEASE ENTER TRADE MANUALLY. <<<")
                
                sys.exit(0) 
                
            else:
                print("No perfect setup found. Waiting 5s...")
                time.sleep(5)
                continue

        except KeyboardInterrupt:
            print("\nStopped by user.")
            sys.exit(0)

except KeyboardInterrupt:
    sys.exit(0)