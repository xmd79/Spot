import os
import time
import numpy as np
import threading
from binance.client import Client
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

# --- Configuration ---
API_FILE = 'api.txt'
MTF_SCAN_TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h', '1d']
DETAILED_1M_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

# --- SAFETY CONFIGURATION ---
# A 5-second interval is extremely dangerous and will lead to an IP ban.
# A 60-second interval is much safer and still very effective for finding patterns.
# DO NOT set this below 30 seconds unless you want to be banned.
ANALYSIS_INTERVAL_SECONDS = 60

# Rate Limiting Configuration
# The script now intelligently manages this, but 10 is a safe value.
MAX_WORKERS = 10

# --- Spike Pattern Criteria ---
MIN_MTF_DIP_COUNT = 4
VOLUME_ANALYSIS_PERIOD = 56
PRICE_UPTREND_PERIOD = 5
# --- End Configuration ---

# Global event to signal a graceful shutdown
stop_event = threading.Event()

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
        usdc_pairs = [s['symbol'] for s in symbols if s['quoteAsset'] == 'USDC' and s['status'] == 'TRADING' and s['isSpotTradingAllowed']
                      and not any(exclude in s['symbol'] for exclude in ['UP', 'DOWN', 'BEAR', 'BULL'])]  # Filter out leverage tokens
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
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if not parts:
        parts.append("just now")
    return " ".join(parts) + " ago"

def check_mtf_dip(client, symbol, timeframe):
    """Checks if an asset is near a recent low for a given timeframe."""
    if stop_event.is_set():
        return None  # Exit early if shutdown is signaled
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=100)
        if not klines or len(klines) < 20:
            return False

        close_prices = np.array([float(k[4]) for k in klines])
        current_price = close_prices[-1]

        min_price = np.min(close_prices)
        max_price = np.max(close_prices)

        price_range = max_price - min_price
        if price_range == 0:
            return False

        is_near_low = (current_price - min_price) / price_range < 0.10
        return is_near_low
    except Exception:
        return False

def analyze_spike_pattern(client, symbol):
    """Analyzes the 1m chart for signs of an imminent price spike."""
    if stop_event.is_set():
        return None  # Exit early if shutdown is signaled
    try:
        klines = client.get_klines(symbol=symbol, interval='1m', limit=max(VOLUME_ANALYSIS_PERIOD + 10, PRICE_UPTREND_PERIOD + 10))
        if not klines or len(klines) < VOLUME_ANALYSIS_PERIOD:
            return None

        close_prices = np.array([float(k[4]) for k in klines])
        volumes = np.array([float(k[5]) for k in klines])

        recent_closes = close_prices[-PRICE_UPTREND_PERIOD:]
        is_price_increasing = all(recent_closes[i] > recent_closes[i-1] for i in range(1, len(recent_closes)))

        recent_volumes = volumes[-VOLUME_ANALYSIS_PERIOD:]
        x_axis = np.arange(len(recent_volumes))
        volume_slope, _ = np.polyfit(x_axis, recent_volumes, 1)
        is_volume_rising = volume_slope > 0

        if is_price_increasing and is_volume_rising:
            score = volume_slope * 1000
            return {
                "symbol": symbol,
                "score": score,
                "price_slope": "Increasing",
                "volume_slope": f"{volume_slope:.2f}",
                "current_price": close_prices[-1]
            }

        return None
    except Exception:
        return None

def analyze_best_asset_details(client, symbol):
    """Performs a detailed min/max/avg analysis on a single symbol across all timeframes."""
    print(f"\n--- Performing Detailed Threshold Analysis for {symbol} ---")
    details = {}
    for timeframe in DETAILED_1M_TIMEFRAMES:
        if stop_event.is_set():
            break  # Exit early if shutdown is signaled
        try:
            klines = client.get_klines(symbol=symbol, interval=timeframe, limit=1000)
            if not klines:
                continue

            close_prices = np.array([float(k[4]) for k in klines])

            details[timeframe] = {
                "min": np.min(close_prices),
                "max": np.max(close_prices),
                "middle": np.mean(close_prices)
            }
            time.sleep(0.1)  # Small delay to be respectful
        except Exception as e:
            print(f"Could not get detailed data for {symbol} on {timeframe}: {e}")

    if details:
        for tf, data in sorted(details.items()):
            print(f"  | {tf:<4} | Min: {data['min']:<12.8f} | Max: {data['max']:<12.8f} | Middle (Avg): {data['middle']:.8f}")
    else:
        print("No detailed data could be retrieved.")


def main():
    """Main function to run the MTF dip and spike pattern analysis loop."""
    client = get_binance_client()
    if not client:
        return

    print("--- Binance MTF Dip & Spike Pattern Hunter (Safe Mode) ---")
    print("Press Ctrl+C to stop the bot at any time.")

    usdc_pairs = fetch_usdc_pairs(client)
    if not usdc_pairs:
        print("No USDC pairs found. Exiting.")
        return

    last_best_asset_key = None

    try:
        while not stop_event.is_set():
            start_time = time.time()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning for MTF dips & spike patterns...")

            # --- Phase 1: MTF Dip Scanning ---
            mtf_dip_counts = {}
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_symbol = {}
                for symbol in usdc_pairs:
                    for timeframe in MTF_SCAN_TIMEFRAMES:
                        if stop_event.is_set():
                            break
                        future = executor.submit(check_mtf_dip, client, symbol, timeframe)
                        future_to_symbol[future] = symbol

                for future in as_completed(future_to_symbol):
                    if stop_event.is_set():
                        break
                    try:
                        is_dip = future.result()
                        symbol = future_to_symbol[future]
                        if is_dip:
                            mtf_dip_counts[symbol] = mtf_dip_counts.get(symbol, 0) + 1
                    except Exception:
                        pass

            # --- Phase 2: Spike Pattern Analysis on Top Candidates ---
            top_candidates = [s for s, count in sorted(mtf_dip_counts.items(), key=lambda item: item[1], reverse=True) if count >= MIN_MTF_DIP_COUNT]

            final_spike_candidates = []
            if top_candidates:
                print(f"Top {len(top_candidates)} MTF dip candidates found. Analyzing for spike patterns...")
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_symbol = {executor.submit(analyze_spike_pattern, client, symbol): symbol for symbol in top_candidates[:5]}

                    for future in as_completed(future_to_symbol):
                        if stop_event.is_set():
                            break
                        try:
                            result = future.result()
                            if result:
                                final_spike_candidates.append(result)
                        except Exception:
                            pass

            # --- Final Ranking and Reporting ---
            final_spike_candidates.sort(key=lambda x: x['score'], reverse=True)

            # FIX: Initialize best_asset to None to prevent UnboundLocalError
            best_asset = None
            current_best_asset_key = None
            if final_spike_candidates:
                best_asset = final_spike_candidates[0]
                current_best_asset_key = best_asset['symbol']

            if best_asset and current_best_asset_key != last_best_asset_key:
                print("\n" + "="*60)
                print(f"!!! NEW TOP MTF DIP & SPIKE PATTERN FOUND !!!")
                print(f"  Asset:      {best_asset['symbol']}")
                print(f"  Score:      {best_asset['score']:.2f}")
                print(f"  Price Trend: {best_asset['price_slope']}")
                print(f"  Volume Trend: {best_asset['volume_slope']}")
                print(f"  Current Price: {best_asset['current_price']:.8f}")
                print("="*60)

                analyze_best_asset_details(client, best_asset['symbol'])
                last_best_asset_key = current_best_asset_key
            else:
                print("No new top patterns found. Continuing to monitor.")

            # Calculate sleep time, accounting for the duration of the scan
            scan_duration = time.time() - start_time
            sleep_time = max(0, ANALYSIS_INTERVAL_SECONDS - scan_duration)
            print(f"Scan complete. Next run in {sleep_time:.0f} seconds.")

            # Use stop_event.wait() instead of time.sleep() to allow for instant interruption
            stop_event.wait(sleep_time)

    except KeyboardInterrupt:
        # This block is now a fallback, the primary shutdown is via the stop_event
        print("\nShutdown signal received.")
    finally:
        stop_event.set()
        print("Bot stopped gracefully.")

if __name__ == "__main__":
    main()