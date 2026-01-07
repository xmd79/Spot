from binance.client import Client
from binance.exceptions import BinanceAPIException
import numpy as np
import talib as ta
import concurrent.futures
from tabulate import tabulate
import os
import sys
import time

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

def analyze_pair(symbol, interval, logic_mode='poly_only'):
    """
    Analyzes a pair based on the specific logic_mode requested.
    """
    try:
        limit = 1000 
        klines = trader.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        if not klines:
            return False, None

        close = [float(entry[4]) for entry in klines]
        x = np.array(close)
        
        # Minimum data check
        if len(x) < 50: 
            return False, None

        # --- Common Logic: Poly Fit Channel ---
        y = range(len(x))
        best_fit_line = np.poly1d(np.polyfit(y, x, 1))(y)
        lower_bound = best_fit_line * 0.99
        
        # Check Rule: Close is below lower channel
        if x[-1] >= lower_bound[-1]:
            return False, None

        # Logic Mode: Poly Only (2h, 15m)
        if logic_mode == 'poly_only':
            return True, None

        # --- Extended Logic: Argmin vs Argmax (5m, 1m) ---
        idx_min = np.argmin(x)
        idx_max = np.argmax(x)
        
        # Rule: Dip (min) is more recent than Peak (max)
        if idx_min <= idx_max:
            return False, None

        # Logic Mode: Poly Arg (5m)
        if logic_mode == 'poly_arg':
            return True, None

        # --- Final Logic: RSI (1m Final) ---
        if logic_mode == 'poly_arg_rsi':
            rsi_values = ta.RSI(x, timeperiod=14)
            current_rsi = rsi_values[-1]
            
            if current_rsi is None or np.isnan(current_rsi):
                return False, None
            
            return True, current_rsi

    except (BinanceAPIException, Exception) as e:
        return False, None

def run_scan_concurrent(pairs, interval, description, logic_mode):
    if not pairs:
        return []

    # Suppress verbose scanning logs to keep table output clean
    # print(f"Scanning {len(pairs)} pairs on {description}...")
    
    results = []
    MAX_WORKERS = 5 
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_symbol = {
                executor.submit(analyze_pair, symbol, interval, logic_mode): symbol 
                for symbol in pairs
            }
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    is_match, value = future.result()
                    if is_match:
                        if logic_mode == 'poly_arg_rsi':
                            results.append((symbol, value))
                        else:
                            results.append(symbol)
                except Exception:
                    pass
    except KeyboardInterrupt:
        # Allow interruption to propagate up
        raise
        
    return results

def print_dynamic_table(header, data):
    """Helper to print lists of data using tabulate"""
    print(f"\n{'='*20} {header} {'='*20}")
    if not data:
        print("No assets found in this stage.")
        return
    
    table_data = []
    
    # If data is a list of tuples (Symbol, Value)
    if isinstance(data[0], tuple): 
        # Sort by value (e.g., RSI) lowest first
        sorted_data = sorted(data, key=lambda x: x[1])
        for item in sorted_data:
            table_data.append([item[0], f"{item[1]:.2f}"])
        print(tabulate(table_data, headers=["Symbol", "Value"], tablefmt="grid"))
        
    # If data is a list of strings (Symbol only)
    else: 
        for item in data:
            table_data.append([item])
        # Print without headers or with simple index
        print(tabulate(table_data, headers=["Symbol"], tablefmt="grid", showindex=True))

# --- Main Execution Loop ---

filename = 'api.txt'
if not os.path.exists(filename):
    print("Please create an 'api.txt' file with your API key on the first line and Secret on the second.")
    sys.exit(1)

trader = Trader(filename)

print("Bot started. Press Ctrl+C to stop.")

try:
    while True:
        try:
            print("\n" + "#"*60)
            print(f"Starting New Scan Cycle - {time.strftime('%H:%M:%S')}")
            print("#"*60)

            # Stage 1: Get All Assets
            list_all_pairs = trader.get_usdc_pairs()
            print_dynamic_table(f"Stage 1: All USDC Assets (Total: {len(list_all_pairs)})", list_all_pairs)

            # Stage 2: 2h Filter (Polyfit Only)
            list_2h = run_scan_concurrent(list_all_pairs, Client.KLINE_INTERVAL_2HOUR, "2H", 'poly_only')
            if not list_2h:
                print("No assets passed Stage 2 (2h Filter). Retrying...")
                time.sleep(5)
                continue
            print_dynamic_table(f"Stage 2: Passed 2H Filter (Total: {len(list_2h)})", list_2h)

            # Stage 3: 15m Filter (Polyfit Only)
            list_15m = run_scan_concurrent(list_2h, Client.KLINE_INTERVAL_15MINUTE, "15M", 'poly_only')
            if not list_15m:
                print("No assets passed Stage 3 (15m Filter). Retrying...")
                time.sleep(5)
                continue
            print_dynamic_table(f"Stage 3: Passed 15M Filter (Total: {len(list_15m)})", list_15m)

            # Stage 4: 5m Filter (Polyfit + Argmin/Argmax)
            list_5m = run_scan_concurrent(list_15m, Client.KLINE_INTERVAL_5MINUTE, "5M", 'poly_arg')
            if not list_5m:
                print("No assets passed Stage 4 (5m MTF Dip). Retrying...")
                time.sleep(5)
                continue
            print_dynamic_table(f"Stage 4: Passed 5M MTF Dip (Total: {len(list_5m)})", list_5m)

            # Stage 5: 1m Final Filter (Polyfit + Argmin + RSI)
            list_final = run_scan_concurrent(list_5m, Client.KLINE_INTERVAL_1MINUTE, "1M Final", 'poly_arg_rsi')
            
            # Always print final table even if empty to show we tried, or skip to loop?
            # Requirement: "and so on til single value is found"
            if not list_final:
                print("No assets passed Stage 5 (1m Final). Retrying...")
                time.sleep(5)
                continue
            
            print_dynamic_table(f"Stage 5: Final Candidates (Sorted by RSI)", list_final)

            # Determine Single Best Value
            # list_final contains tuples (symbol, rsi). 
            # Sort by RSI again to be safe and take index 0
            list_final.sort(key=lambda x: x[1])
            best_pick = list_final[0]

            print("\n" + "#"*60)
            print(f"   BEST SINGLE VALUE FOUND")
            print(f"   Symbol: {best_pick[0]}")
            print(f"   RSI:    {best_pick[1]:.2f}")
            print("#"*60 + "\n")
            
            # Assuming "til single value is found" means stop when we identify the winner.
            # If you want it to restart looking for better ones, remove the 'break'.
            break 

        except KeyboardInterrupt:
            # This catches the Ctrl+C inside the loop
            print("\n\n[!] Scan interrupted by user (Ctrl+C). Stopping bot...")
            sys.exit(0)

except KeyboardInterrupt:
    # This catches the Ctrl+C if it happens during initialization (unlikely here but good practice)
    print("\n\n[!] Bot stopped by user.")
    sys.exit(0)