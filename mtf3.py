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
from collections import defaultdict

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
        print(f"\n>>> EXECUTING BUY ORDER: {symbol} for {usdc_amount} USDC")
        try:
            qty_str = "{:.2f}".format(usdc_amount)
            order = self.client.order_market_buy(
                symbol=symbol,
                quoteOrderQty=qty_str
            )
            return order
        except BinanceAPIException as e:
            print(f"BUY FAILED: {e}")
            return None

    def execute_sell(self, symbol):
        print(f"\n>>> EXECUTING SELL ORDER: {symbol} (Market)")
        try:
            account = self.client.get_account()
            asset_balance = 0.0
            for bal in account['balances']:
                if bal['asset'] == symbol.replace('USDC', ''):
                    asset_balance = float(bal['free'])
            
            if asset_balance <= 0:
                print("No balance to sell.")
                return None

            info = self.client.get_symbol_info(symbol)
            step_size = 0.001
            for f in info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    step_size = float(f['step_size'])
            
            precision = int(round(-np.log10(step_size)))
            qty_str = "{0:.{1}f}".format(asset_balance, precision)
            
            order = self.client.order_market_sell(
                symbol=symbol,
                quantity=qty_str
            )
            return order
        except BinanceAPIException as e:
            print(f"SELL FAILED: {e}")
            return None

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def update_trade_stats(symbol, profit, profit_pct):
    global TOTAL_NET_PNL, TOTAL_TRADES, WINS, LOSSES, TRADE_HISTORY
    
    TOTAL_TRADES += 1
    TOTAL_NET_PNL += profit
    result = "WIN" if profit > 0 else "LOSS"
    if profit > 0: WINS += 1
    else: LOSSES += 1
    
    TRADE_HISTORY.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Symbol": symbol,
        "Result": result,
        "PnL ($)": f"{profit:.2f}",
        "PnL (%)": f"{profit_pct:.2f}%"
    })

def print_stats_dashboard():
    win_rate = (WINS / TOTAL_TRADES * 100) if TOTAL_TRADES > 0 else 0.0
    print("\n" + "="*60)
    print(" " * 15 + "TRADING BOT DASHBOARD")
    print("="*60)
    print(f"Total Trades: {TOTAL_TRADES} | Wins: {WINS} | Losses: {LOSSES}")
    print(f"Win Rate:    {win_rate:.2f}%")
    print(f"Net PnL:      {TOTAL_NET_PNL:.2f} USDC")
    print("="*60)

    if TRADE_HISTORY:
        history_to_show = TRADE_HISTORY[::-1][:10] 
        print("\nRECENT TRADE HISTORY:")
        print(tabulate(history_to_show, headers="keys", tablefmt="grid"))
    else:
        print("\nNo trades executed yet.")

def format_float(val):
    if val is None: return "-"
    try:
        return f"{float(val):.{PRECISION}f}"
    except:
        return "-"

def print_dynamic_table(full_records):
    """
    Prints the dynamic table sorted by MTF Score.
    """
    clear_screen()
    print_stats_dashboard()
    
    print(f"\nDisplaying All Assets (Sorted by MTF Strength: Most Conditions First)")

    # Sort Logic
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
            format_float(data.get('argmin_price')),
            format_float(data.get('argmax_price')),
            format_float(data.get('bull_vol')),
            format_float(data.get('bear_vol'))
        ]
        table_data.append(row)

    headers = [
        "Symbol", "2H Trend", "15M Trend", "5M Trend", "5M Mid", "1M RSI",
        "ArgMin Price", "ArgMax Price", "Bull Vol %", "Bear Vol %"
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=f".{PRECISION}f"))

# --- Analysis Logic ---

def get_regression_breakout_status(close_array):
    """
    Logic: Price < Regression Line AND 
    Most recent Lowest price below line is more recent than Most recent Highest price above line.
    Returns: 'PASS' or 'FAIL'
    """
    try:
        if len(close_array) < 50: return "FAIL"
        
        y = range(len(close_array))
        # Polyfit might fail if variance is 0, handle it
        try:
            slope, intercept = np.polyfit(y, close_array, 1)
        except:
            return "FAIL"
            
        line_values = slope * y + intercept
        
        # Indices where price is below line
        below_mask = close_array < line_values
        # Indices where price is above line
        above_mask = close_array > line_values
        
        if not np.any(below_mask): return "FAIL" # Price never below line
        if not np.any(above_mask): return "FAIL" # Price never above line
        
        # Get most recent index of Lowest Close below line
        # We want the specific point with lowest value among those below
        values_below = close_array[below_mask]
        min_val_below = np.min(values_below)
        # Find index of that min value in the original array
        idx_min_below = np.where(close_array == min_val_below)[0][-1] # Last occurrence
        
        # Get most recent index of Highest Close above line
        values_above = close_array[above_mask]
        max_val_above = np.max(values_above)
        idx_max_above = np.where(close_array == max_val_above)[0][-1]
        
        # Condition: The breakdown (min below) happened AFTER the breakout (max above)
        if idx_min_below > idx_max_above:
            return "PASS"
            
    except Exception:
        pass
    return "FAIL"

def analyze_single_asset(client, symbol):
    """
    Fetches all data for a symbol and calculates all conditions.
    Returns a dict of results.
    """
    results = {
        '2h_trend': 'FAIL', '15m_trend': 'FAIL', '5m_trend': 'FAIL', '5m_mid': 'FAIL',
        '1m_rsi': '-', '1m_rsi_valid': 'NO', 
        'argmin_price': None, 'argmax_price': None, '1m_struct_valid': 'NO',
        'bull_vol': None, 'bear_vol': None, 'vol_signal': 'BEAR'
    }
    
    try:
        # --- 2H Analysis ---
        try:
            k_2h = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_2HOUR, limit=1200)
            if k_2h:
                c_2h = np.array([float(k[4]) for k in k_2h])
                results['2h_trend'] = get_regression_breakout_status(c_2h)
        except: pass

        # --- 15M Analysis ---
        try:
            k_15m = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=1200)
            if k_15m:
                c_15m = np.array([float(k[4]) for k in k_15m])
                results['15m_trend'] = get_regression_breakout_status(c_15m)
        except: pass

        # --- 5M Analysis (Trend + Midpoint) ---
        try:
            k_5m = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_5MINUTE, limit=1200)
            if k_5m:
                c_5m = np.array([float(k[4]) for k in k_5m])
                
                # Trend
                results['5m_trend'] = get_regression_breakout_status(c_5m)
                
                # Midpoint
                y = range(len(c_5m))
                try:
                    best_fit_line = np.poly1d(np.polyfit(y, c_5m, 1))(y)
                    idx_min = np.argmin(c_5m)
                    idx_max = np.argmax(c_5m)
                    val_max = c_5m[idx_max]
                    val_min = c_5m[idx_min]
                    middle_threshold = (val_max + val_min) / 2
                    is_mid = c_5m[-1] < middle_threshold and idx_min > idx_max
                    results['5m_mid'] = "PASS" if is_mid else "FAIL"
                except: results['5m_mid'] = "FAIL"
        except: pass

        # --- 1M Analysis (Structure + RSI + Volume) ---
        try:
            k_1m = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=1200)
            if k_1m:
                c_1m = np.array([float(k[4]) for k in k_1m])
                
                # Structure (ArgMin/ArgMax)
                idx_min = np.argmin(c_1m)
                idx_max = np.argmax(c_1m)
                val_min = c_1m[idx_min]
                val_max = c_1m[idx_max]
                is_struct = idx_min > idx_max
                results['argmin_price'] = val_min
                results['argmax_price'] = val_max
                results['1m_struct_valid'] = "YES" if is_struct else "NO"
                
                # RSI (Oversold > Overbought)
                rsi = ta.RSI(c_1m, timeperiod=14)
                valid_rsi = rsi[np.logical_not(np.isnan(rsi))]
                if len(valid_rsi) > 0:
                    idx_oversold = -1
                    idx_overbought = -1
                    # Search backwards
                    for i in range(len(valid_rsi) - 1, -1, -1):
                        if valid_rsi[i] < 30 and idx_oversold == -1: idx_oversold = i
                        if valid_rsi[i] > 70 and idx_overbought == -1: idx_overbought = i
                        if idx_oversold != -1 and idx_overbought != -1: break
                    
                    results['1m_rsi'] = f"{valid_rsi[-1]:.2f}"
                    results['1m_rsi_valid'] = "YES" if idx_oversold > idx_overbought else "NO"
                
                # Volume
                bull = 0.0
                bear = 0.0
                for k in k_1m:
                    o, c, v = float(k[1]), float(k[4]), float(k[5])
                    if c > o: bull += v
                    elif c < o: bear += v
                
                total = bull + bear
                if total > 0:
                    b_pct = (bull / total) * 100
                    be_pct = (bear / total) * 100
                    results['bull_vol'] = b_pct
                    results['bear_vol'] = be_pct
                    results['vol_signal'] = "BULL" if b_pct > be_pct else "BEAR"
                    
        except: pass

    except Exception:
        pass # Symbol failed completely
        
    return results

# --- Main Execution Loop ---

filename = 'api.txt'
if not os.path.exists(filename):
    print("Please create 'api.txt' with API Key and Secret.")
    sys.exit(1)

trader = Trader(filename)
print("Bot started. Press Ctrl+C to stop.")

try:
    while True:
        try:
            cycle_start = time.time()
            
            # 1. Initialization
            list_all_pairs = trader.get_usdc_pairs()
            full_records = {}
            
            print(f"Scanning {len(list_all_pairs)} assets for full MTF data...")

            # ============================================================
            # FULL SCAN: Analyze every asset for every timeframe
            # ============================================================
            
            # Use ThreadPool to fetch data for all assets in parallel
            results_list = []
            MAX_WORKERS = 5
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_sym = {executor.submit(analyze_single_asset, trader.client, s): s for s in list_all_pairs}
                for future in concurrent.futures.as_completed(future_to_sym):
                    sym = future_to_sym[future]
                    try:
                        data = future.result()
                        full_records[sym] = data
                    except Exception:
                        full_records[sym] = {} # Handle error gracefully

            # ============================================================
            # PRINT TABLE
            # ============================================================
            print_dynamic_table(full_records)
            
            cycle_duration = time.time() - cycle_start
            print(f"\nScan Complete in {cycle_duration:.2f}s")

            # ============================================================
            # FIND WINNER (Strict Logic)
            # ============================================================
            
            # Criteria: 2H/15M/5M Trend PASS, 5M Mid PASS, 1M Struct Valid, 1M Vol Bull
            winners = []
            for s in list_all_pairs:
                r = full_records[s]
                if (r.get('2h_trend') == 'PASS' and 
                    r.get('15m_trend') == 'PASS' and 
                    r.get('5m_trend') == 'PASS' and
                    r.get('5m_mid') == 'PASS' and
                    r.get('1m_struct_valid') == 'YES' and
                    r.get('vol_signal') == 'BULL'):
                    
                    winners.append(s)
            
            if winners:
                # Pick lowest RSI
                valid_winners = []
                for s in winners:
                    try:
                        rsi = float(r['1m_rsi'])
                        valid_winners.append((s, rsi))
                    except: pass
                
                valid_winners.sort(key=lambda x: x[1])
                best_symbol = valid_winners[0][0]
                goto_trade = True
                print(f"[!] WINNER FOUND: {best_symbol}")
            else:
                print("No perfect setup found. Waiting 5s...")
                print_stats_dashboard() # Print stats at end
                time.sleep(5)
                continue

            # ============================================================
            # TRADING PHASE
            # ============================================================
            
            if goto_trade:
                print(f"\n{'='*60}")
                print(f"TRADING EXECUTION")
                print(f"SELECTED ASSET: {best_symbol}")
                print(f"{'='*60}")

                usdc_balance = trader.get_account_balance()
                if usdc_balance < 10:
                    print("Insufficient USDC balance.")
                    sys.exit(1)
                
                order = trader.execute_buy(best_symbol, usdc_balance)
                if not order:
                    time.sleep(10)
                    continue

                executed_qty = 0.0
                spent = 0.0
                for f in order['fills']:
                    executed_qty += float(f['qty'])
                    spent += float(f['quoteQty'])
                
                entry_price = spent / executed_qty
                target_price = entry_price * 1.032

                while True:
                    curr = trader.get_current_price(best_symbol)
                    pnl = ((curr - entry_price) / entry_price) * 100
                    print(f"\rPrice: {curr:.8f} | PnL: {pnl:+.2f}% | Target: {target_price:.8f}", end="")
                    
                    if curr >= target_price:
                        break
                    time.sleep(2)
                
                print("\n\nTarget Reached! Selling...")
                sell_order = trader.execute_sell(best_symbol)
                
                if sell_order:
                    got_back = 0.0
                    for f in sell_order['fills']:
                        got_back += float(f['quoteQty'])
                    
                    profit = got_back - spent
                    profit_pct = (profit / spent) * 100
                    update_trade_stats(best_symbol, profit, profit_pct)
                    
                    # Print stats at end of iteration
                    print_stats_dashboard()
                    print("Waiting 5s...")
                    time.sleep(5)

        except KeyboardInterrupt:
            print("\nStopped by user.")
            sys.exit(0)

except KeyboardInterrupt:
    sys.exit(0)
