from binance.client import Client
import matplotlib.pyplot as plt
import numpy as np
import talib as ta
import os
import sys
import time
import pandas as pd
from datetime import datetime
import concurrent.futures
from functools import partial

class Trader:
    def __init__(self, file):
        self.connect(file)

    """ Creates Binance client """
    def connect(self, file):
        lines = [line.rstrip('\n') for line in open(file)]
        key = lines[0]
        secret = lines[1]
        self.client = Client(key, secret)

    """ Gets all account balances """
    def getBalances(self):
        account = self.client.get_account()
        balances = account['balances']
        return {balance['asset']: float(balance['free']) for balance in balances if float(balance['free']) > 0}

    """ Get all pairs traded against USDC """
    def get_usdc_pairs(self):
        exchange_info = self.client.get_exchange_info()
        trading_pairs = [symbol['symbol'] for symbol in exchange_info['symbols'] 
                        if symbol['quoteAsset'] == 'USDC' and symbol['status'] == 'TRADING']
        return trading_pairs

    """ Get historical klines data with error handling """
    def get_klines_safe(self, symbol, interval, limit=500):
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            return klines
        except Exception as e:
            print(f"Error fetching data for {symbol} on {interval}: {e}")
            return None

filename = 'api.txt'
trader = Trader(filename)

# Global lists to store filtered pairs at each timeframe
filtered_pairs_daily = []
filtered_pairs_2h = []
filtered_pairs_15m = []
filtered_pairs_5m = []
filtered_pairs_3m = []
selected_pairs = []
selected_pairs_data = []

def check_poly_dip(symbol, interval, trader, threshold=0.01):
    """
    Generic function to check if price is below polynomial trendline
    Returns: (symbol, is_dip, current_price, trendline_value, deviation)
    """
    klines = trader.get_klines_safe(symbol, interval)
    
    if klines is None:
        return (symbol, False, None, None, None)
        
    close = [float(entry[4]) for entry in klines]
    
    if len(close) < 10:  # Need minimum data points
        return (symbol, False, None, None, None)
    
    try:
        x = close
        y = np.arange(len(x))
        
        # Calculate polynomial trendline (degree 1 for linear)
        poly_coeffs = np.polyfit(y, x, 1)
        trendline = np.poly1d(poly_coeffs)(y)
        
        current_price = x[-1]
        current_trendline = trendline[-1]
        
        # Calculate deviation percentage
        deviation = (current_price - current_trendline) / current_trendline
        
        # Check if price is below threshold
        is_dip = deviation < -threshold
        
        return (symbol, is_dip, current_price, current_trendline, deviation)
        
    except Exception as e:
        print(f"Error processing {symbol} on {interval}: {e}")
        return (symbol, False, None, None, None)

def check_momentum(symbol, trader):
    """
    Check CMO momentum on 1-minute timeframe
    Returns: (symbol, cmo_value, is_oversold)
    """
    klines = trader.get_klines_safe(symbol, '1m', limit=100)
    
    if klines is None:
        return (symbol, None, False)
        
    close = [float(entry[4]) for entry in klines]
    
    if len(close) < 14:
        return (symbol, None, False)
    
    try:
        close_array = np.asarray(close)
        cmo = ta.CMO(close_array, timeperiod=14)
        current_cmo = cmo[-1]
        is_oversold = current_cmo < -50
        
        return (symbol, current_cmo, is_oversold)
        
    except Exception as e:
        print(f"Error calculating momentum for {symbol}: {e}")
        return (symbol, None, False)

def process_pairs_batch(pairs, interval, trader, max_workers=10):
    """
    Process a batch of pairs concurrently for a specific timeframe
    """
    filtered_pairs = []
    
    # Create partial function with fixed parameters
    check_func = partial(check_poly_dip, interval=interval, trader=trader)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {executor.submit(check_func, pair): pair for pair in pairs}
        
        for future in concurrent.futures.as_completed(future_to_pair):
            symbol, is_dip, price, trendline, deviation = future.result()
            
            if is_dip:
                filtered_pairs.append({
                    'symbol': symbol,
                    'price': price,
                    'trendline': trendline,
                    'deviation': deviation,
                    'interval': interval
                })
                print(f"✓ Dip found on {interval}: {symbol} (deviation: {deviation:.2%})")
    
    return filtered_pairs

def process_momentum_batch(pairs_data, trader, max_workers=10):
    """
    Process momentum check for final filtered pairs
    """
    momentum_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {executor.submit(check_momentum, pair['symbol'], trader): pair for pair in pairs_data}
        
        for future in concurrent.futures.as_completed(future_to_pair):
            pair_data = future_to_pair[future]
            symbol, cmo, is_oversold = future.result()
            
            if is_oversold:
                momentum_results.append({
                    'symbol': symbol,
                    'cmo': cmo,
                    'price': pair_data['price'],
                    'deviation_3m': pair_data['deviation'],
                    'all_data': pair_data
                })
                print(f"✓ Momentum oversold: {symbol} (CMO: {cmo:.2f})")
    
    return momentum_results

def plot_mtf_chart(symbol, trader):
    """
    Plot multi-timeframe chart for the selected symbol
    """
    intervals = ['1d', '2h', '15m', '5m', '3m', '1m']
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, interval in enumerate(intervals):
        klines = trader.get_klines_safe(symbol, interval)
        if klines is None:
            continue
            
        close = [float(entry[4]) for entry in klines]
        timestamps = [datetime.fromtimestamp(int(entry[0])/1000) for entry in klines]
        
        axes[idx].plot(timestamps, close, label='Price', linewidth=1)
        
        # Add trendline
        x = close
        y = np.arange(len(x))
        poly_coeffs = np.polyfit(y, x, 1)
        trendline = np.poly1d(poly_coeffs)(y)
        axes[idx].plot(timestamps, trendline, 'r--', label='Trendline', linewidth=1)
        
        # Add bands
        axes[idx].plot(timestamps, trendline * 1.01, 'g--', alpha=0.5, label='+1%')
        axes[idx].plot(timestamps, trendline * 0.99, 'b--', alpha=0.5, label='-1%')
        
        axes[idx].set_title(f'{symbol} - {interval}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'MTF_Dip_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

def main():
    print("=== Multi-Timeframe Dip Finder ===")
    print(f"Starting scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get all USDC pairs
    trading_pairs = trader.get_usdc_pairs()
    print(f"Found {len(trading_pairs)} USDC trading pairs")
    
    # Filter 1: Daily timeframe
    print("\n=== Filter 1: Daily Timeframe ===")
    daily_filtered = process_pairs_batch(trading_pairs, '1d', trader, max_workers=15)
    daily_symbols = [p['symbol'] for p in daily_filtered]
    print(f"Daily filter passed: {len(daily_symbols)} pairs")
    
    if not daily_symbols:
        print("No pairs passed daily filter. Exiting.")
        return
    
    # Filter 2: 2h timeframe
    print("\n=== Filter 2: 2-Hour Timeframe ===")
    h2_filtered = process_pairs_batch(daily_symbols, '2h', trader, max_workers=10)
    h2_symbols = [p['symbol'] for p in h2_filtered]
    print(f"2h filter passed: {len(h2_symbols)} pairs")
    
    if not h2_symbols:
        print("No pairs passed 2h filter. Exiting.")
        return
    
    # Filter 3: 15m timeframe
    print("\n=== Filter 3: 15-Minute Timeframe ===")
    m15_filtered = process_pairs_batch(h2_symbols, '15m', trader, max_workers=10)
    m15_symbols = [p['symbol'] for p in m15_filtered]
    print(f"15m filter passed: {len(m15_symbols)} pairs")
    
    if not m15_symbols:
        print("No pairs passed 15m filter. Exiting.")
        return
    
    # Filter 4: 5m timeframe
    print("\n=== Filter 4: 5-Minute Timeframe ===")
    m5_filtered = process_pairs_batch(m15_symbols, '5m', trader, max_workers=10)
    m5_symbols = [p['symbol'] for p in m5_filtered]
    print(f"5m filter passed: {len(m5_symbols)} pairs")
    
    if not m5_symbols:
        print("No pairs passed 5m filter. Exiting.")
        return
    
    # Filter 5: 3m timeframe
    print("\n=== Filter 5: 3-Minute Timeframe ===")
    m3_filtered = process_pairs_batch(m5_symbols, '3m', trader, max_workers=10)
    print(f"3m filter passed: {len(m3_filtered)} pairs")
    
    if not m3_filtered:
        print("No pairs passed 3m filter. Exiting.")
        return
    
    # Final Filter: Momentum check on 1m
    print("\n=== Final Filter: 1-Minute Momentum ===")
    momentum_results = process_momentum_batch(m3_filtered, trader, max_workers=10)
    print(f"Momentum filter passed: {len(momentum_results)} pairs")
    
    if not momentum_results:
        print("No pairs passed momentum filter. No MTF dips found.")
        return
    
    # Sort by lowest CMO (most oversold)
    momentum_results.sort(key=lambda x: x['cmo'])
    
    print("\n=== MTF DIPS FOUND ===")
    print(f"Total MTF dips: {len(momentum_results)}")
    
    for i, result in enumerate(momentum_results, 1):
        print(f"\n{i}. {result['symbol']}")
        print(f"   CMO: {result['cmo']:.2f}")
        print(f"   Current Price: {result['price']:.6f}")
        print(f"   3m Deviation: {result['deviation_3m']:.2%}")
    
    # Select the most oversold (lowest CMO)
    best_dip = momentum_results[0]
    print(f"\n=== BEST MTF DIP ===")
    print(f"Symbol: {best_dip['symbol']}")
    print(f"CMO: {best_dip['cmo']:.2f}")
    print(f"Current Price: {best_dip['price']:.6f}")
    print(f"3m Deviation: {best_dip['deviation_3m']:.2%}")
    
    # Generate MTF chart for the best dip
    print(f"\nGenerating MTF chart for {best_dip['symbol']}...")
    plot_mtf_chart(best_dip['symbol'], trader)
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(momentum_results)
    df.to_csv(f'MTF_Dips_{timestamp}.csv', index=False)
    print(f"Results saved to MTF_Dips_{timestamp}.csv")
    
    print(f"\nScan completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
    sys.exit(0)