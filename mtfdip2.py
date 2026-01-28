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
import time

# --- CONFIGURATION ---
LOOKBACK_LENGTH = 360
MIN_SAMPLES = LOOKBACK_LENGTH 
# ---------------------

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
        
        symbol_map = {}
        
        for symbol_info in exchange_info['symbols']:
            if symbol_info['quoteAsset'] == 'USDC' and symbol_info['status'] == 'TRADING':
                original_symbol = symbol_info['symbol']
                base_asset = symbol_info['baseAsset']
                
                standardized = self.create_meaningful_symbol(base_asset, original_symbol)
                
                if standardized:
                    symbol_map[standardized] = original_symbol
                    print(f"📋 {original_symbol} -> {standardized}")
        
        print(f"\n✅ Found {len(symbol_map)} USDC trading pairs")
        return symbol_map
    
    def create_meaningful_symbol(self, base_asset, original_symbol):
        """Create meaningful standardized symbol from base asset"""
        if base_asset and isinstance(base_asset, str):
            base_english = self.convert_to_english(base_asset)
            if base_english:
                return f"{base_english}USDC"
        
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
        
        normalized = unicodedata.normalize('NFKD', text)
        ascii_text = ''
        for char in normalized:
            if ord(char) < 128:
                ascii_text += char
            else:
                ascii_text += char
        
        translit_map = {
            '币': 'BI', '安': 'AN', '人': 'REN', '生': 'SHENG', '幣': 'BI', 'の': 'NO', 'コ': 'KO', 'イ': 'I', 'ン': 'N', 'ト': 'TO',
        }
        
        result = ''
        for char in ascii_text:
            if char in translit_map:
                result += translit_map[char]
            elif ord(char) < 128:
                result += char
            else:
                try:
                    name = unicodedata.name(char).split()[-1][:3]
                    result += name
                except:
                    result += 'X'
        
        cleaned = re.sub(r'[^A-Z0-9]', '', result.upper())
        
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
                    return None
                elif attempt < retries - 1:
                    time.sleep(1)
                else:
                    return None

# --- ACCUMULATION LOGIC ---

def check_trend_structure(prices):
    """
    LOGIC:
    1. Current Close < Middle Line.
    2. Lowest value below Lower Line is MORE RECENT than Highest value above Upper Line.
    """
    
    if not prices or len(prices) < MIN_SAMPLES:
        return False
    
    try:
        prices_recent = np.asarray(prices[-LOOKBACK_LENGTH:])
        x = np.arange(len(prices_recent))
        y = prices_recent
        
        # 1. Calculate Linear Regression
        m, b = np.polyfit(x, y, 1)
        middle_line = m * x + b
        
        # 2. Calculate Standard Deviation
        residuals = y - middle_line
        std_dev = np.std(residuals, ddof=1) 
        
        lower_band = middle_line - (1.0 * std_dev)
        upper_band = middle_line + (1.0 * std_dev)
        
        # 3. CHECK CONDITION A: Current Close < Middle Line
        current_price = y[-1]
        if current_price >= middle_line[-1]:
            return False 
        
        # 4. CHECK CONDITION B: Recent Dip vs Recent Peak
        dip_indices = np.where(y < lower_band)[0]
        peak_indices = np.where(y > upper_band)[0]
        
        if len(dip_indices) == 0:
            return False
            
        last_dip_index = dip_indices[-1]
        
        if len(peak_indices) == 0:
            return True
            
        last_peak_index = peak_indices[-1]
        
        # 5. COMPARE RECENCY
        if last_dip_index > last_peak_index:
            return True
        
        return False
        
    except Exception as e:
        print(f"Error in trend check: {e}")
        return False

# --- FILTER FUNCTIONS ---

def filter1(standardized_symbol):
    """Check 2h timeframe for Accumulation Structure"""
    interval = '2h'
    original_symbol = symbol_map[standardized_symbol]
    
    try:
        klines = trader.safe_get_klines(symbol=original_symbol, interval=interval)
        if not klines: return
        
        close = [float(entry[4]) for entry in klines]
        if not close or len(close) < MIN_SAMPLES: return
        
        if check_trend_structure(close): 
            with locks[0]:
                filtered_pairs1.append(standardized_symbol)
                results_map[standardized_symbol]['original'] = original_symbol
                results_map[standardized_symbol]['2h'] = True
                print(f'✅ 2h ACCUMULATION: {standardized_symbol}')
    except Exception as e:
        pass

def filter2(standardized_symbol):
    """Check 15m timeframe for Accumulation Structure"""
    interval = '15m'
    original_symbol = symbol_map[standardized_symbol]
    
    try:
        klines = trader.safe_get_klines(symbol=original_symbol, interval=interval)
        if not klines: return
        
        close = [float(entry[4]) for entry in klines]
        if not close or len(close) < MIN_SAMPLES: return
        
        if check_trend_structure(close):
            with locks[1]:
                filtered_pairs2.append(standardized_symbol)
                results_map[standardized_symbol]['15m'] = True
                print(f'✅ 15m ACCUMULATION: {standardized_symbol}')
    except Exception as e:
        pass

def filter3(standardized_symbol):
    """
    Check 5m timeframe for Accumulation Structure AND Calculate HT_SINE
    """
    interval = '5m'
    original_symbol = symbol_map[standardized_symbol]
    
    try:
        klines = trader.safe_get_klines(symbol=original_symbol, interval=interval)
        if not klines: return
        
        close = [float(entry[4]) for entry in klines]
        if not close or len(close) < MIN_SAMPLES: return
        
        if check_trend_structure(close):
            
            # --- NEW LOGIC: HT_SINE CALCULATION ---
            close_array = np.asarray(close)
            
            # Calculate Hilbert Transform - Sine Wave
            # Returns (sine, leadsine)
            sine, _ = ta.HT_SINE(close_array)
            
            # Filter NaN and 0 values as requested
            # We create a mask of valid values
            valid_mask = (~np.isnan(sine)) & (sine != 0)
            clean_sine = sine[valid_mask]
            
            current_sine = None
            if len(clean_sine) > 0:
                current_sine = clean_sine[-1]
            else:
                # Fallback if no clean data (shouldn't happen often with 360 candles)
                current_sine = 0 
            
            with locks[2]:
                filtered_pairs3.append(standardized_symbol)
                results_map[standardized_symbol]['5m'] = True
                results_map[standardized_symbol]['sine'] = float(current_sine)
                print(f'✅ 5m ACCUMULATION: {standardized_symbol} | Sine: {current_sine:.4f}')
    except Exception as e:
        pass

# --- MAIN EXECUTION BLOCK ---

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
# Removed selected_pair/selected_pairCMO as we use sine now

# Store additional info
results_map = defaultdict(dict)

# Thread locks
locks = [threading.Lock() for _ in range(3)] # Reduced to 3 locks

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
                pass
            
            if completed % 50 == 0 or completed == len(pairs_list):
                print(f"📈 Progress: {completed}/{len(pairs_list)}")

def main():
    if not symbol_map:
        print("❌ No trading pairs found. Exiting.")
        sys.exit(1)
    
    # Show sample with meaningful names
    print("\n📋 SAMPLE OF TRADING PAIRS:")
    print("Standardized Name <- Original Name")
    print("-" * 50)
    samples = list(symbol_map.items())[:15]
    for i, (std, orig) in enumerate(samples):
        print(f"{i+1:2d}. {std:20} <- {orig}")
    if len(symbol_map) > 15:
        print(f"... and {len(symbol_map) - 15} more")
    
    # Stage 1: 2h scan
    concurrent_filter_stage(standardized_symbols, filter1, "2-HOUR FILTER (Accumulation Logic)")
    print(f"\n✅ 2H FILTER: {len(filtered_pairs1)}/{len(standardized_symbols)} passed")
    
    # Stage 2: 15m scan
    if filtered_pairs1:
        concurrent_filter_stage(filtered_pairs1, filter2, "15-MINUTE FILTER (Accumulation Logic)")
        print(f"\n✅ 15M FILTER: {len(filtered_pairs2)}/{len(filtered_pairs1)} passed")
    else:
        print("\n❌ No pairs passed 2h filter")
        sys.exit(0)
    
    # Stage 3: 5m scan (Includes Sine Calculation)
    if filtered_pairs2:
        concurrent_filter_stage(filtered_pairs2, filter3, "5-MINUTE FILTER (Accumulation + Sine Wave)")
        print(f"\n✅ 5M FILTER: {len(filtered_pairs3)}/{len(filtered_pairs2)} passed")
    else:
        print("\n❌ No pairs passed 15m filter")
        sys.exit(0)
    
    # --- FINAL SORTING BASED ON SINE WAVE ---
    if filtered_pairs3:
        print(f"\n{'='*70}")
        print("🏆 FINAL QUALIFIED PAIRS (HT_SINE Cycle Bottom)")
        print(f"{'='*70}")
        
        # Prepare list for sorting: [(symbol, sine_value), ...]
        sine_candidates = []
        for symbol in filtered_pairs3:
            sine_val = results_map[symbol].get('sine', 0)
            if sine_val is not None:
                sine_candidates.append((symbol, sine_val))
        
        if sine_candidates:
            # Sort by Sine value (Ascending). 
            # The most negative value (closest to -1.0) is the cycle bottom.
            sorted_candidates = sorted(sine_candidates, key=lambda x: x[1])
            
            print(f"🎯 Found {len(sorted_candidates)} dip candidates ranked by Cycle Bottom (Lowest Sine):\n")
            
            for rank, (std_symbol, sine_val) in enumerate(sorted_candidates, 1):
                orig_symbol = symbol_map[std_symbol]
                
                # Visualization: -0.9 is bottom, +0.9 is top
                if sine_val < -0.8: level = '🔴 DEEP CYCLE BOTTOM'
                elif sine_val < -0.5: level = '🟠 NEAR BOTTOM'
                elif sine_val < 0: level = '🟡 LOWER HALF'
                else: level = '⚪ NEUTRAL/RISING'
                
                print(f"{rank:2d}. {std_symbol:20} <- {orig_symbol}")
                print(f"    🌊 HT_Sine: {sine_val:.4f} | {level}")
                print()
            
            # Select the best (Most negative Sine)
            best_std, best_sine = sorted_candidates[0]
            best_orig = symbol_map[best_std]
            
            print(f"{'='*70}")
            print(f"🏆 TOP SELECTION (Lowest Cycle Point):")
            print(f"{'='*70}")
            print(f"Symbol:     {best_std}")
            print(f"Original:   {best_orig}")
            print(f"Sine Value: {best_sine:.4f}")
            
            # Show alternatives
            if len(sorted_candidates) > 1:
                print(f"\n🎖️  ALTERNATIVES:")
                for i, (std_symbol, sine_val) in enumerate(sorted_candidates[1:4], 2):
                    orig_symbol = symbol_map[std_symbol]
                    print(f"  {i}. {std_symbol:15} (Sine: {sine_val:.4f}) <- {orig_symbol}")
        else:
            print("❌ No valid Sine wave data found.")
            
        # --- Report on ALL 5m passers ---
        print(f"\n{'='*70}")
        print(f"🔍 5m Accumulation Candidates (Total: {len(filtered_pairs3)})")
        print(f"{'='*70}")
        
        for symbol in filtered_pairs3:
            sine_val = results_map[symbol].get('sine', 'N/A')
            print(f"{symbol:20} | Accumulation Structure | Sine: {sine_val}")
            
    else:
        print("\n❌ No pairs passed 5m filter")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 SCAN SUMMARY")
    print(f"{'='*70}")
    print(f"Total pairs scanned:     {len(symbol_map):4d}")
    print(f"Passed 2h filter:        {len(filtered_pairs1):4d}  ({len(filtered_pairs1)/len(symbol_map)*100:.1f}%)")
    
    def safe_pct(numerator, denominator):
        if denominator == 0: return 0.0
        return numerator/denominator*100

    print(f"Passed 15m filter:       {len(filtered_pairs2):4d}  ({safe_pct(len(filtered_pairs2), len(filtered_pairs1)):.1f}% of previous)")
    print(f"Passed 5m filter:        {len(filtered_pairs3):4d}  ({safe_pct(len(filtered_pairs3), len(filtered_pairs2)):.1f}% of previous)")
    # Removed final spike candidates count
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
    sys.exit(0)