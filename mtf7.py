from binance.client import Client
import numpy as np
import talib as ta
import sys
import concurrent.futures

# --- Configuration ---
# RSI Length for the final 3m decision. 
# Standard is 14, but you can lower this (e.g., 7) for more sensitivity if desired.
RSI_LENGTH = 14 
# ---------------------

class Trader:
    def __init__(self, file):
        self.connect(file)

    def connect(self, file):
        lines = [line.rstrip('\n') for line in open(file)]
        key = lines[0]
        secret = lines[1]
        self.client = Client(key, secret)

    def get_usdc_pairs(self):
        exchange_info = self.client.get_exchange_info()
        trading_pairs = [symbol['symbol'] for symbol in exchange_info['symbols'] 
                         if symbol['quoteAsset'] == 'USDC' and symbol['status'] == 'TRADING']
        return trading_pairs

filename = 'credentials.txt'
trader = Trader(filename)

def get_klines_data(client, symbol, interval, limit=200):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines or len(klines) < 50: 
            return None
        
        closes = np.array([float(entry[4]) for entry in klines], dtype=np.float64)
        highs = np.array([float(entry[2]) for entry in klines], dtype=np.float64)
        lows = np.array([float(entry[3]) for entry in klines], dtype=np.float64)
        raw_closes = [float(entry[4]) for entry in klines]
        
        return closes, highs, lows, raw_closes
    except Exception:
        return None

def check_regression_dip(symbol, interval, client):
    """ Checks if price is below the lower regression band (0.99 factor) """
    data = get_klines_data(client, symbol, interval)
    if data is None:
        return False
    
    close = data[0]
    
    # Skip if price is flat
    if np.std(close) == 0:
        return False

    # Regression Logic
    x = close
    y = range(len(x))
    
    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line3 = best_fit_line1 * 0.99
    
    if x[-1] < best_fit_line3[-1]:
        return True
    return False

def analyze_3m_rsi(symbol, client):
    """
    Final analysis: Calculates RSI on the 3-minute timeframe.
    Selects the pair with the lowest RSI value.
    """
    # Using '3m' interval as requested
    data = get_klines_data(client, symbol, '3m', limit=100)
    if data is None:
        return None

    close, _, _, raw_closes = data

    # Volatility Check
    if np.std(close) == 0:
        return None

    # Calculate RSI
    try:
        rsi_array = ta.RSI(close, timeperiod=RSI_LENGTH)
        if np.isnan(rsi_array[-1]):
            return None
        current_rsi = rsi_array[-1]
    except Exception:
        return None

    return {
        'symbol': symbol,
        'rsi': current_rsi,
        'last_close': close[-1],
        'last_10_closes': raw_closes[-10:]
    }

def run_stage_filter(symbols, interval, client):
    found_pairs = []
    print(f"Scanning {len(symbols)} pairs on {interval} timeframe...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_symbol = {executor.submit(check_regression_dip, symbol, interval, client): symbol for symbol in symbols}
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                if future.result():
                    found_pairs.append(symbol)
            except Exception:
                pass
    
    return found_pairs

def run_final_analysis(symbols, client):
    candidates = []
    print(f"Analyzing {len(symbols)} candidates on 3m timeframe (RSI)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_symbol = {executor.submit(analyze_3m_rsi, symbol, client): symbol for symbol in symbols}
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result:
                    candidates.append(result)
            except Exception:
                pass
    
    # Sort by Lowest RSI value
    candidates.sort(key=lambda x: x['rsi'])
    return candidates

# --- Main Execution ---

print("Fetching trading pairs...")
trading_pairs = trader.get_usdc_pairs()

# Stage 1: 2h Filter
filtered_pairs_2h = run_stage_filter(trading_pairs, '2h', trader.client)
print(f"Found {len(filtered_pairs_2h)} dips on 2h.")

if not filtered_pairs_2h:
    sys.exit(0)

# Stage 2: 15m Filter
filtered_pairs_15m = run_stage_filter(filtered_pairs_2h, '15m', trader.client)
print(f"Found {len(filtered_pairs_15m)} dips on 15m.")

if not filtered_pairs_15m:
    sys.exit(0)

# Stage 3: 5m Filter
filtered_pairs_5m = run_stage_filter(filtered_pairs_15m, '5m', trader.client)
print(f"Found {len(filtered_pairs_5m)} dips on 5m.")

if not filtered_pairs_5m:
    sys.exit(0)

# Stage 4: 3m RSI Analysis
final_candidates = run_final_analysis(filtered_pairs_5m, trader.client)

# Results
if final_candidates:
    print("\n" + "="*60)
    print(f"=== BEST MTF DIP CANDIDATE (Lowest 3m RSI) ===")
    print("="*60)
    
    best = final_candidates[0]
    
    print(f"\n[WINNER] {best['symbol']}")
    print(f"Current Price: {best['last_close']}")
    print(f"RSI (3m, Period {RSI_LENGTH}): {best['rsi']:.2f}")
    
    print("-" * 60)
    print("DATA VERIFICATION (Last 10 3-minute Closing Prices):")
    print("-" * 60)
    last_prices = best['last_10_closes']
    for i, price in enumerate(last_prices):
        direction = ""
        if i > 0:
            if price < last_prices[i-1]: direction = "  (Drop)"
            elif price > last_prices[i-1]: direction = "  (Rise)"
        print(f"3m Candle -{10-i}: {price:.5f}{direction}")
    
    print("="*60)

    print("\nTop 5 List (Sorted by Lowest RSI):")
    for i, c in enumerate(final_candidates[:5]):
        print(f"{i+1}. {c['symbol']} | Price: {c['last_close']:.5f} | RSI(3m): {c['rsi']:.2f}")
else:
    print("No candidates found.")

sys.exit(0)
