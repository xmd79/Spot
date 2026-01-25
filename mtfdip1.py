from binance.client import Client
import matplotlib.pyplot as plt
import numpy as np
import talib as ta
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re
from collections import defaultdict
import unicodedata

class Trader:
    def __init__(self, file):
        self.connect(file)

    """ Creates Binance client """
    def connect(self, file):
        lines = [line.rstrip('\n') for line in open(file)]
        key = lines[0]
        secret = lines[1]
        self.client = Client(key, secret)

    """ Get all USDC trading pairs with proper mapping """
    def get_all_usdc_pairs(self):
        exchange_info = self.client.get_exchange_info()
        
        # Dictionary to store: {standardized_symbol: original_symbol}
        symbol_map = {}
        
        for symbol_info in exchange_info['symbols']:
            if symbol_info['quoteAsset'] == 'USDC' and symbol_info['status'] == 'TRADING':
                original_symbol = symbol_info['symbol']
                base_asset = symbol_info['baseAsset']  # Get the actual base asset
                
                # Create meaningful standardized name
                standardized = self.create_meaningful_symbol(base_asset, original_symbol)
                
                if standardized:
                    symbol_map[standardized] = original_symbol
                    print(f"📋 {original_symbol} → {standardized}")
        
        print(f"\n✅ Found {len(symbol_map)} USDC trading pairs")
        return symbol_map
    
    def create_meaningful_symbol(self, base_asset, original_symbol):
        """Create meaningful standardized symbol from base asset"""
        # First, try to use the actual base asset name
        if base_asset and isinstance(base_asset, str):
            # Convert to English/ASCII representation
            base_english = self.convert_to_english(base_asset)
            if base_english:
                return f"{base_english}USDC"
        
        # Fallback: extract from original symbol
        # Remove USDC and any separators, keep everything before USDC
        pattern = r'^(.*?)[\.\-_]?USDC$'
        match = re.match(pattern, original_symbol, re.IGNORECASE)
        if match:
            base_part = match.group(1)
            base_english = self.convert_to_english(base_part)
            if base_english:
                return f"{base_english}USDC"
        
        return "UNKNOWNUSDC"
    
    def convert_to_english(self, text):
        """Convert any text to English ASCII representation"""
        if not text:
            return ""
        
        # Normalize unicode (convert accented characters to their base form)
        normalized = unicodedata.normalize('NFKD', text)
        
        # Remove non-ASCII characters but keep the text
        ascii_text = ''
        for char in normalized:
            # Check if character is ASCII
            if ord(char) < 128:
                ascii_text += char
            else:
                # For non-ASCII, try to get a meaningful representation
                # Keep the character but we'll transliterate it
                ascii_text += char
        
        # Now transliterate any remaining non-ASCII to closest ASCII
        # Simple manual mapping for common cases
        translit_map = {
            '币': 'BI',
            '安': 'AN',
            '人': 'REN',
            '生': 'SHENG',
            '生': 'SHENG',
            '幣': 'BI',
            '安': 'AN',
            'の': 'NO',
            'コ': 'KO',
            'イ': 'I',
            'ン': 'N',
            'ト': 'TO',
        }
        
        result = ''
        for char in ascii_text:
            if char in translit_map:
                result += translit_map[char]
            elif ord(char) < 128:
                result += char
            else:
                # For other non-ASCII, use Unicode name or skip
                try:
                    name = unicodedata.name(char).split()[-1][:3]
                    result += name
                except:
                    result += 'X'
        
        # Remove any remaining non-alphanumeric except uppercase letters
        cleaned = re.sub(r'[^A-Z0-9]', '', result.upper())
        
        # Ensure it's not empty
        if not cleaned:
            cleaned = "UNKNOWN"
        
        return cleaned
    
    def safe_get_klines(self, symbol, interval, retries=3):
        """Safely get klines with error handling and retries"""
        for attempt in range(retries):
            try:
                klines = self.client.get_klines(symbol=symbol, interval=interval)
                return klines
            except Exception as e:
                if "Invalid symbol" in str(e):
                    print(f"❌ Invalid symbol for API: {symbol}")
                    return None
                elif attempt < retries - 1:
                    print(f"⚠️  Retry {attempt + 1}/{retries} for {symbol} on {interval}")
                    import time
                    time.sleep(1)
                else:
                    print(f"❌ Failed to get klines for {symbol} on {interval}: {e}")
                    return None

filename = 'credentials.txt'
trader = Trader(filename)

# Get all symbols with mapping
symbol_map = trader.get_all_usdc_pairs()

# Standardized symbols for processing
standardized_symbols = list(symbol_map.keys())

# Thread-safe storage
filtered_pairs1 = []
filtered_pairs2 = []
filtered_pairs3 = []
selected_pair = []
selected_pairCMO = []

# Store additional info
results_map = defaultdict(dict)

# Thread locks
locks = [threading.Lock() for _ in range(4)]

def check_trend(prices, threshold_multiplier=0.99):
    """Helper function to check if price is below regression line"""
    if not prices or len(prices) < 10:
        return False
    
    try:
        x = prices
        y = range(len(x))
        
        best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
        best_fit_line3 = best_fit_line1 * threshold_multiplier
        
        return x[-1] < best_fit_line3[-1]
    except Exception as e:
        print(f"Error in trend check: {e}")
        return False

def filter1(standardized_symbol):
    """Check 2h timeframe"""
    interval = '2h'
    original_symbol = symbol_map[standardized_symbol]
    
    try:
        klines = trader.safe_get_klines(symbol=original_symbol, interval=interval)
        if not klines:
            return
        
        close = [float(entry[4]) for entry in klines]
        
        if not close or len(close) < 20:
            return
        
        print(f"🔍 2h: {standardized_symbol} ← {original_symbol}")
        
        if check_trend(close, 0.99):
            with locks[0]:
                filtered_pairs1.append(standardized_symbol)
                results_map[standardized_symbol]['original'] = original_symbol
                results_map[standardized_symbol]['2h'] = True
                print(f'✅ 2h PASS: {standardized_symbol}')
    except Exception as e:
        print(f"Error processing {standardized_symbol} on 2h: {e}")

def filter2(standardized_symbol):
    """Check 15m timeframe"""
    interval = '15m'
    original_symbol = symbol_map[standardized_symbol]
    
    try:
        klines = trader.safe_get_klines(symbol=original_symbol, interval=interval)
        if not klines:
            return
        
        close = [float(entry[4]) for entry in klines]
        
        if not close or len(close) < 20:
            return
        
        print(f"🔍 15m: {standardized_symbol} ← {original_symbol}")
        
        if check_trend(close, 0.99):
            with locks[1]:
                filtered_pairs2.append(standardized_symbol)
                results_map[standardized_symbol]['15m'] = True
                print(f'✅ 15m PASS: {standardized_symbol}')
    except Exception as e:
        print(f"Error processing {standardized_symbol} on 15m: {e}")

def filter3(standardized_symbol):
    """Check 5m timeframe"""
    interval = '5m'
    original_symbol = symbol_map[standardized_symbol]
    
    try:
        klines = trader.safe_get_klines(symbol=original_symbol, interval=interval)
        if not klines:
            return
        
        close = [float(entry[4]) for entry in klines]
        
        if not close or len(close) < 20:
            return
        
        print(f"🔍 5m: {standardized_symbol} ← {original_symbol}")
        
        if check_trend(close, 0.99):
            with locks[2]:
                filtered_pairs3.append(standardized_symbol)
                results_map[standardized_symbol]['5m'] = True
                print(f'✅ 5m PASS: {standardized_symbol}')
    except Exception as e:
        print(f"Error processing {standardized_symbol} on 5m: {e}")

def momentum_filter(standardized_symbol):
    """Check 1m momentum"""
    interval = '1m'
    original_symbol = symbol_map[standardized_symbol]
    
    try:
        klines = trader.safe_get_klines(symbol=original_symbol, interval=interval)
        if not klines:
            return
        
        close = [float(entry[4]) for entry in klines]
        
        if not close or len(close) < 20:
            return
        
        print(f"🔍 1m: {standardized_symbol} ← {original_symbol}")
        
        close_array = np.asarray(close)
        real = ta.CMO(close_array, timeperiod=14)
        
        if len(real) > 0 and real[-1] < -50:
            with locks[3]:
                selected_pair.append(standardized_symbol)
                selected_pairCMO.append(real[-1])
                results_map[standardized_symbol]['cmo'] = real[-1]
                results_map[standardized_symbol]['final'] = True
                print(f'🎯 MTF DIP: {standardized_symbol} (CMO: {real[-1]:.2f})')
    except Exception as e:
        print(f"Error processing {standardized_symbol} on 1m: {e}")

def concurrent_filter_stage(pairs_list, filter_func, stage_name, max_workers=10):
    """Run filter function concurrently"""
    print(f"\n{'='*70}")
    print(f"🚀 {stage_name.upper()}")
    print(f"📊 Scanning {len(pairs_list)} pairs...")
    print(f"{'='*70}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(filter_func, pair): pair for pair in pairs_list}
        
        completed = 0
        for future in as_completed(futures):
            pair = futures[future]
            completed += 1
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error for {pair}: {e}")
            
            if completed % 10 == 0:
                print(f"📈 Progress: {completed}/{len(pairs_list)}")

def main():
    if not symbol_map:
        print("❌ No trading pairs found. Exiting.")
        sys.exit(1)
    
    # Show sample with meaningful names
    print("\n📋 SAMPLE OF TRADING PAIRS:")
    print("Standardized Name ← Original Name")
    print("-" * 50)
    samples = list(symbol_map.items())[:15]
    for i, (std, orig) in enumerate(samples):
        print(f"{i+1:2d}. {std:20} ← {orig}")
    if len(symbol_map) > 15:
        print(f"... and {len(symbol_map) - 15} more")
    
    # Stage 1: 2h scan
    concurrent_filter_stage(standardized_symbols, filter1, "2-hour filter")
    
    print(f"\n✅ 2H FILTER: {len(filtered_pairs1)}/{len(standardized_symbols)} passed")
    
    # Stage 2: 15m scan
    if filtered_pairs1:
        concurrent_filter_stage(filtered_pairs1, filter2, "15-minute filter")
        print(f"\n✅ 15M FILTER: {len(filtered_pairs2)}/{len(filtered_pairs1)} passed")
    else:
        print("\n❌ No pairs passed 2h filter")
        sys.exit(0)
    
    # Stage 3: 5m scan
    if filtered_pairs2:
        concurrent_filter_stage(filtered_pairs2, filter3, "5-minute filter")
        print(f"\n✅ 5M FILTER: {len(filtered_pairs3)}/{len(filtered_pairs2)} passed")
    else:
        print("\n❌ No pairs passed 15m filter")
        sys.exit(0)
    
    # Stage 4: 1m momentum
    if filtered_pairs3:
        concurrent_filter_stage(filtered_pairs3, momentum_filter, "1-minute momentum")
        
        print(f"\n{'='*70}")
        print("🏆 FINAL QUALIFIED PAIRS")
        print(f"{'='*70}")
        
        if selected_pair:
            # Sort by CMO (most oversold first)
            sorted_pairs = sorted(zip(selected_pair, selected_pairCMO), 
                                key=lambda x: x[1])
            
            print(f"🎯 Found {len(sorted_pairs)} MTF dips (ranked by oversold condition):\n")
            
            for rank, (std_symbol, cmo) in enumerate(sorted_pairs, 1):
                orig_symbol = symbol_map[std_symbol]
                print(f"{rank:2d}. {std_symbol:20} ← {orig_symbol}")
                print(f"    📉 CMO: {cmo:7.2f} | Oversold level: {'🔴' if cmo < -60 else '🟠' if cmo < -55 else '🟡' if cmo < -50 else '⚪'}")
                print()
            
            # Select the best
            best_std, best_cmo = sorted_pairs[0]
            best_orig = symbol_map[best_std]
            
            print(f"{'='*70}")
            print(f"🏆 TOP SELECTION:")
            print(f"{'='*70}")
            print(f"Symbol:     {best_std}")
            print(f"Original:   {best_orig}")
            print(f"CMO Score:  {best_cmo:.2f}")
            print(f"Status:     {'EXTREMELY OVERSOLD 🔴' if best_cmo < -60 else 'STRONGLY OVERSOLD 🟠' if best_cmo < -55 else 'OVERSOLD 🟡'}")
            
            # Show alternatives
            if len(sorted_pairs) > 1:
                print(f"\n🎖️  ALTERNATIVES:")
                for i, (std_symbol, cmo) in enumerate(sorted_pairs[1:4], 2):
                    orig_symbol = symbol_map[std_symbol]
                    print(f"  {i}. {std_symbol:15} (CMO: {cmo:6.2f}) ← {orig_symbol}")
        else:
            print("❌ No MTF dips found")
    else:
        print("\n❌ No pairs passed 5m filter")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 SCAN SUMMARY")
    print(f"{'='*70}")
    print(f"Total pairs scanned:     {len(symbol_map):4d}")
    print(f"Passed 2h filter:        {len(filtered_pairs1):4d}  ({len(filtered_pairs1)/len(symbol_map)*100:.1f}%)")
    print(f"Passed 15m filter:       {len(filtered_pairs2):4d}  ({len(filtered_pairs2)/len(filtered_pairs1)*100:.1f}% of previous)")
    print(f"Passed 5m filter:        {len(filtered_pairs3):4d}  ({len(filtered_pairs3)/len(filtered_pairs2)*100:.1f}% of previous)")
    print(f"Final MTF dip candidates: {len(selected_pair):4d}  ({len(selected_pair)/len(filtered_pairs3)*100:.1f}% of previous)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
    sys.exit(0)