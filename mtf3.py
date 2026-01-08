

I have updated the code to include the specific logic rules you requested.

### Changes Made:
1.  **5 Minute Timeframe (New Mode: `5m_midpoint`):**
    *   Increased the candle limit to **1200** as requested.
    *   Added a check to ensure the current price is **below the middle threshold** (calculated as the average of the Highest Price and Lowest Price found in the last 1200 candles).
    *   This ensures the price is deep in the dip, not just slightly off the regression line.

2.  **1 Minute Timeframe (Enhanced `poly_arg_rsi`):**
    *   The existing code already checked that the lowest price happened *after* the highest price (`idx_min > idx_max`). I have verified this logic is strictly applied to ensure a valid "Dip" structure for the 1-minute timeframe as well.

Here is the complete, updated script:

```python
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
        """Returns free USDC balance"""
        account = self.client.get_account()
        for balance in account['balances']:
            if balance['asset'] == 'USDC':
                return float(balance['free'])
        return 0.0

    def get_current_price(self, symbol):
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])

    def execute_buy(self, symbol, usdc_amount):
        """Executes a Market Buy for specific USDC amount"""
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
        """Sells all available balance of the asset"""
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
                    step_size = float(f['stepSize'])
            
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

def print_1min_dip_details(trader, symbol):
    """Fetches 1m data for the winner and prints Argmin/Argmax details"""
    try:
        klines = trader.client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=1000)
        close = [float(entry[4]) for entry in klines]
        x = np.array(close)
        
        if len(x) < 2: return

        idx_min = np.argmin(x)
        idx_max = np.argmax(x)
        val_min = x[idx_min]
        val_max = x[idx_max]
        
        print(f"\n{'='*20} 1M MTF DIP VERIFICATION {symbol} {'='*20}")
        print(f"Total Candles Analyzed: {len(x)}")
        print(f"ARGMAX (Peak) Index: {idx_max} | Price: {val_max}")
        print(f"ARGMIN (Dip)  Index: {idx_min} | Price: {val_min}")
        
        if idx_min > idx_max:
            print("Confirmation: Dip (ARGMIN) is AFTER Peak (ARGMAX). Valid MTF Dip structure.")
        else:
            print("Warning: Dip structure invalid (Dip before Peak).")
        print("="*60)
    except Exception as e:
        print(f"Could not verify 1m dip details: {e}")

def analyze_pair(symbol, interval, logic_mode='poly_only', limit=1000):
    """
    Analyzes a pair based on the specified logic mode.
    """
    try:
        klines = trader.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines: return False, None
        close = [float(entry[4]) for entry in klines]
        x = np.array(close)
        if len(x) < 50: return False, None

        y = range(len(x))
        best_fit_line = np.poly1d(np.polyfit(y, x, 1))(y)
        lower_bound = best_fit_line * 0.99
        
        # Base Condition: Price must be below the regression trend
        if x[-1] >= lower_bound[-1]: return False, None

        if logic_mode == 'poly_only': return True, None

        # Structural Check: Peak must happen before Dip
        idx_min = np.argmin(x)
        idx_max = np.argmax(x)
        
        # If Dip is not after Peak, invalid structure
        if idx_min <= idx_max: return False, None

        if logic_mode == 'poly_arg': return True, None

        # --- New Logic for 5M ---
        if logic_mode == '5m_midpoint':
            val_max = x[idx_max]
            val_min = x[idx_min]
            
            # Calculate Middle Threshold of Argmin and Argmax
            middle_threshold = (val_max + val_min) / 2
            
            # Condition: Current price must be below this middle threshold
            if x[-1] >= middle_threshold: return False, None
            
            return True, None

        # --- Logic for 1M (RSI + Structure) ---
        if logic_mode == 'poly_arg_rsi':
            # Ensure structure is valid (already checked above idx_min > idx_max)
            
            rsi_values = ta.RSI(x, timeperiod=14)
            current_rsi = rsi_values[-1]
            if current_rsi is None or np.isnan(current_rsi): return False, None
            return True, current_rsi

    except (BinanceAPIException, Exception): return False, None

def run_scan_concurrent(pairs, interval, description, logic_mode, limit=1000):
    if not pairs: return []
    results = []
    MAX_WORKERS = 5 
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_symbol = {executor.submit(analyze_pair, s, interval, logic_mode, limit): s for s in pairs}
            for future in concurrent.futures.as_completed(future_to_symbol):
                try:
                    is_match, value = future.result()
                    if is_match:
                        if logic_mode == 'poly_arg_rsi': results.append((future_to_symbol[future], value))
                        else: results.append(future_to_symbol[future])
                except Exception: pass
    except KeyboardInterrupt: raise
    return results

def print_dynamic_table(header, data):
    print(f"\n{'='*20} {header} {'='*20}")
    if not data: print("No assets found."); return
    table_data = []
    if isinstance(data[0], tuple): 
        sorted_data = sorted(data, key=lambda x: x[1])
        for item in sorted_data: table_data.append([item[0], f"{item[1]:.2f}"])
        print(tabulate(table_data, headers=["Symbol", "Value"], tablefmt="grid"))
    else: 
        for item in data: table_data.append([item])
        print(tabulate(table_data, headers=["Symbol"], tablefmt="grid", showindex=True))

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
            print("\n" + "#"*60)
            print(f"Starting Scan Cycle - {time.strftime('%H:%M:%S')}")
            print("#"*60)

            # --- Scanning Phase ---
            list_all_pairs = trader.get_usdc_pairs()
            print_dynamic_table(f"Stage 1: All USDC Assets ({len(list_all_pairs)})", list_all_pairs)

            # Stage 2: 2H Filter (Polyfit only)
            list_2h = run_scan_concurrent(list_all_pairs, Client.KLINE_INTERVAL_2HOUR, "2H", 'poly_only')
            if not list_2h: time.sleep(5); continue
            print_dynamic_table(f"Stage 2: 2H Filter ({len(list_2h)})", list_2h)

            # Stage 3: 15M Filter (Polyfit only)
            list_15m = run_scan_concurrent(list_2h, Client.KLINE_INTERVAL_15MINUTE, "15M", 'poly_only')
            if not list_15m: time.sleep(5); continue
            print_dynamic_table(f"Stage 3: 15M Filter ({len(list_15m)})", list_15m)

            # Stage 4: 5M Filter 
            # New Logic: Polyfit + Structure + Price below middle threshold of Max/Min (1200 candles)
            list_5m = run_scan_concurrent(list_15m, Client.KLINE_INTERVAL_5MINUTE, "5M", '5m_midpoint', limit=1200)
            if not list_5m: time.sleep(5); continue
            print_dynamic_table(f"Stage 4: 5M MTF Dip ({len(list_5m)})", list_5m)

            # Stage 5: 1M Final
            # Logic: Polyfit + Structure (Argmin after Argmax) + Lowest RSI
            list_final = run_scan_concurrent(list_5m, Client.KLINE_INTERVAL_1MINUTE, "1M Final", 'poly_arg_rsi')
            
            if not list_final:
                print("No assets passed Stage 5. Retrying...")
                time.sleep(5)
                continue
            
            list_final.sort(key=lambda x: x[1]) # Sort lowest RSI first
            print_dynamic_table(f"Stage 5: Final Candidates", list_final)

            best_symbol = list_final[0][0]
            print_1min_dip_details(trader, best_symbol)

            # --- Trading Phase ---
            print("\n" + "="*60)
            print(f"SELECTED ASSET: {best_symbol}")
            print("PREPARING TO TRADE MAX USDC BALANCE")
            print("="*60)

            # 1. Get Balance
            usdc_balance = trader.get_account_balance()
            if usdc_balance < 10:
                print("Insufficient USDC balance (Minimum 10 USDC required).")
                sys.exit(1)
            
            print(f"Available USDC: {usdc_balance:.2f}")

            # 2. Execute Buy
            order = trader.execute_buy(best_symbol, usdc_balance)
            
            if not order:
                print("Trade execution failed. Restarting scan.")
                time.sleep(10)
                continue

            # Extract executed details
            executed_qty = 0.0
            cummulative_quote_qty = 0.0
            for fill in order['fills']:
                executed_qty += float(fill['qty'])
                cummulative_quote_qty += float(fill['quoteQty'])
            
            entry_price = cummulative_quote_qty / executed_qty if executed_qty > 0 else 0
            entry_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\n--- ORDER FILLED ---")
            print(f"Time:      {entry_time_str}")
            print(f"Price:     {entry_price}")
            print(f"Spent:     {cummulative_quote_qty:.2f} USDC")
            print(f"Received:  {executed_qty} {best_symbol.replace('USDC','')}")

            # 3. Calculate Target
            TARGET_MULTIPLIER = 1.032
            target_price = entry_price * TARGET_MULTIPLIER

            # --- In-Trade Monitoring Loop ---
            print("\n" + "#"*60)
            print("ENTERING TRADE MONITORING LOOP")
            print("Scanning for Exit Price (Target +3.2%)...")
            print("#"*60 + "\n")

            while True:
                try:
                    current_price = trader.get_current_price(best_symbol)
                    dist_from_entry_pct = ((current_price - entry_price) / entry_price) * 100
                    dist_to_exit_pct = ((target_price - current_price) / current_price) * 100
                    
                    data = [
                        ["Asset Traded", best_symbol],
                        ["Entry Time", entry_time_str],
                        ["Entry Price", f"{entry_price:.8f}"],
                        ["Value (USDC)", f"{cummulative_quote_qty:.2f}"],
                        ["Amount Bought", f"{executed_qty:.6f}"],
                        ["Current Price", f"{current_price:.8f}"],
                        ["Dist from Entry", f"{dist_from_entry_pct:+.2f}%"],
                        ["Dist to Exit", f"{dist_to_exit_pct:.2f}%"],
                        ["Target Price", f"{target_price:.8f}"],
                        ["Net Profit Target", "3.0% (Gross 3.2%)"]
                    ]
                    
                    print("\n" + tabulate(data, tablefmt="grid"))
                    
                    if current_price >= target_price:
                        print(f"\n!!! TARGET REACHED ({dist_from_entry_pct:.2f}) !!!")
                        break

                except KeyboardInterrupt:
                    print("\n[!] User interrupted trade. Selling manually...")
                    break
                except Exception as e:
                    print(f"\nError monitoring trade: {e}")
                
                time.sleep(5)

            # --- Exit Phase ---
            print("\nInitiating Exit (Sell)...")
            sell_order = trader.execute_sell(best_symbol)
            
            if sell_order:
                sold_quote = 0.0
                for fill in sell_order['fills']:
                    sold_quote += float(fill['quoteQty'])
                
                profit = sold_quote - cummulative_quote_qty
                profit_pct = (profit / cummulative_quote_qty) * 100
                
                print("\n" + "#"*60)
                print("TRADE COMPLETED")
                print(f"Exit Price: Avg ~ {sold_quote/executed_qty:.8f}")
                print(f"Total Return: {sold_quote:.2f} USDC")
                print(f"Net Profit: {profit:.2f} USDC ({profit_pct:.2f}%)")
                print("#"*60)
            
            # --- Cycle Restart Logic ---
            print("\n" + "#"*60)
            print("Trade Cycle Complete. Restarting Market Scan...")
            print("#"*60)
            time.sleep(10)

        except KeyboardInterrupt:
            print("\n\n[!] Scan interrupted by user. Stopping bot.")
            sys.exit(0)

except KeyboardInterrupt:
    print("\n[!] Bot stopped by user.")
    sys.exit(0)
```
