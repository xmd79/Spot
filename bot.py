import os
import time
import numpy as np
from binance.client import Client
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

# --- Configuration ---
API_FILE = 'api.txt'
TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
TOP_N_RESULTS = 20
ANALYSIS_INTERVAL_SECONDS = 120 # How often to run the full analysis

# Rate Limiting Configuration
# MAX_WORKERS controls how many API calls run at the same time.
# 10 is a safe value that respects Binance's rate limits.
MAX_WORKERS = 10

# Best Dip Criteria
BEST_DIP_RECOVERY_THRESHOLD = 2.0 # The dip must have recovered by at least this %
# --- End Configuration ---

def get_binance_client():
    """Initialize Binance client with API credentials from a file."""
    if not os.path.exists(API_FILE):
        print(f"Error: API credentials file not found at '{API_FILE}'")
        return None
    try:
        with open(API_FILE, 'r') as file:
            api_key = file.readline().strip()
            api_secret = file.readline().strip()
        return Client(api_key, api_secret)
    except Exception as e:
        print(f"Error reading API credentials: {e}")
        return None

def fetch_usdc_pairs(client):
    """Fetch all available trading pairs against USDC."""
    try:
        exchange_info = client.get_exchange_info()
        symbols = exchange_info['symbols']
        usdc_pairs = [s['symbol'] for s in symbols if s['quoteAsset'] == 'USDC' and s['status'] == 'TRADING' and s['isSpotTradingAllowed']]
        print(f"Found {len(usdc_pairs)} active USDC pairs.")
        return usdc_pairs
    except Exception as e:
        print(f"Error fetching USDC pairs: {e}")
        return []

def format_time_ago(seconds):
    """Convert seconds into a human-readable 'time ago' string."""
    dt = timedelta(seconds=int(seconds))
    days = dt.days
    hours, remainder = divmod(dt.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    
    parts = []
    if days > 0: parts.append(f"{days}d")
    if hours > 0: parts.append(f"{hours}h")
    if minutes > 0: parts.append(f"{minutes}m")
    if not parts: parts.append("just now")
        
    return " ".join(parts) + " ago"

def fetch_and_analyze_candle(client, symbol, timeframe):
    """Fetches and analyzes a single candle for a single symbol/timeframe."""
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=1000)
        if not klines or len(klines) < 10: # Ensure enough data
            return None

        close_prices = np.array([float(k[4]) for k in klines])
        times_ms = np.array([k[0] for k in klines])
        
        min_price_index = np.argmin(close_prices)
        
        # Ignore if the dip is the very last candle (not a historical event)
        if min_price_index == len(close_prices) - 1:
            return None
            
        dip_price = close_prices[min_price_index]
        dip_timestamp_ms = times_ms[min_price_index]
        current_price = close_prices[-1]
        last_timestamp_ms = times_ms[-1]
        
        time_since_dip_sec = (last_timestamp_ms - dip_timestamp_ms) / 1000
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "dip_price": dip_price,
            "current_price": current_price,
            "time_ago_seconds": time_since_dip_sec
        }
    except Exception:
        # Silently ignore errors for individual calls
        return None

def analyze_best_dip_details(client, symbol):
    """Performs a detailed min/max/avg analysis on a single symbol across all timeframes."""
    print(f"\n--- Performing Detailed Threshold Analysis for {symbol} ---")
    details = {}
    for timeframe in TIMEFRAMES:
        try:
            klines = client.get_klines(symbol=symbol, interval=timeframe, limit=1000)
            if not klines:
                continue
            
            close_prices = np.array([float(k[4]) for k in klines])
            
            min_threshold = np.min(close_prices)
            max_threshold = np.max(close_prices)
            middle_threshold = np.mean(close_prices)
            
            details[timeframe] = {
                "min": min_threshold,
                "max": max_threshold,
                "middle": middle_threshold
            }
            time.sleep(0.2) # Small delay for this sequential analysis
        except Exception as e:
            print(f"Could not get detailed data for {symbol} on {timeframe}: {e}")
            
    # Print the detailed results
    if details:
        for tf, data in sorted(details.items()):
            print(f"  | {tf:<4} | Min: {data['min']:<12.8f} | Max: {data['max']:<12.8f} | Middle (Avg): {data['middle']:.8f}")
    else:
        print("No detailed data could be retrieved.")

def main():
    """Main function to run the dip analysis loop."""
    client = get_binance_client()
    if not client:
        return

    print("--- Binance Best Dip Hunter ---")
    
    while True:
        start_time = time.time()
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting new analysis cycle...")
        
        usdc_pairs = fetch_usdc_pairs(client)
        if not usdc_pairs:
            print("No USDC pairs found. Retrying...")
            time.sleep(ANALYSIS_INTERVAL_SECONDS)
            continue

        all_dips = []
        print("Submitting tasks to thread pool...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # The ThreadPoolExecutor now handles the rate limiting by only running
            # MAX_WORKERS threads at a time. This is much faster and safer.
            futures = [
                executor.submit(fetch_and_analyze_candle, client, symbol, timeframe)
                for symbol in usdc_pairs for timeframe in TIMEFRAMES
            ]

            print(f"Processing {len(futures)} results...")
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        all_dips.append(result)
                except Exception as exc:
                    print(f"A task generated an exception: {exc}")

        # Sort all collected dips by how recent they were (ascending)
        sorted_dips = sorted(all_dips, key=lambda x: x['time_ago_seconds'])
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nAnalysis complete in {duration:.2f} seconds. Found {len(sorted_dips)} dip events.")

        # --- Print the Top Results ---
        print("\n--- Most Recent Market Dips (Ranking) ---")
        if not sorted_dips:
            print("No significant dips found in the last cycle.")
            best_dip = None
        else:
            for i, dip in enumerate(sorted_dips[:TOP_N_RESULTS]):
                time_ago_str = format_time_ago(dip['time_ago_seconds'])
                recovery_pct = ((dip['current_price'] - dip['dip_price']) / dip['dip_price']) * 100
                
                print(f"{i+1:2}. [{dip['symbol']:<10} | {dip['timeframe']:<3}] "
                      f"Dip: {dip['dip_price']:.8f} | "
                      f"Current: {dip['current_price']:.8f} ({recovery_pct:+.2f}%) | "
                      f"Occurred: {time_ago_str}")

        # --- Find and Report the Best Dip ---
        best_dip = None
        for dip in sorted_dips:
            recovery_pct = ((dip['current_price'] - dip['dip_price']) / dip['dip_price']) * 100
            if recovery_pct >= BEST_DIP_RECOVERY_THRESHOLD:
                best_dip = dip
                break # Found the best one, stop searching

        if best_dip:
            recovery_pct = ((best_dip['current_price'] - best_dip['dip_price']) / best_dip['dip_price']) * 100
            time_ago_str = format_time_ago(best_dip['time_ago_seconds'])
            print("\n" + "="*50)
            print(f"!!! BEST DIP FOUND !!!")
            print(f"  Asset:      {best_dip['symbol']}")
            print(f"  Timeframe:  {best_dip['timeframe']}")
            print(f"  Dip Price:  {best_dip['dip_price']:.8f}")
            print(f"  Current:    {best_dip['current_price']:.8f}")
            print(f"  Recovery:   {recovery_pct:+.2f}%")
            print(f"  Occurred:   {time_ago_str}")
            print("="*50)
            
            # Run detailed analysis on the best dip's symbol
            analyze_best_dip_details(client, best_dip['symbol'])
        else:
            print("\n--- No dips met the recovery threshold for 'Best Dip'. ---")
        
        print(f"\nNext run in {ANALYSIS_INTERVAL_SECONDS} seconds.")
        time.sleep(ANALYSIS_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped by user.")
