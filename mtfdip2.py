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
LOOKBACK_LENGTH = 1000 
MIN_SAMPLES = LOOKBACK_LENGTH 
STD_DEV_MULTIPLIER = 2.0 
SLEEP_TIME = 5 
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

# --- CORE STRUCTURE LOGIC (UPDATED) ---

def check_trend_structure(prices):
    """
    LOGIC:
    1. Current Close < Middle Line.
    2. ArgMin (Most Extreme Dip) vs ArgMax (Most Extreme Peak).
    3. Most Extreme Dip must be MORE RECENT.
    4. Returns detailed data dictionary.
    """
    
    if not prices or len(prices) < MIN_SAMPLES:
        return {'pass': False}
    
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
        
        # 3. Define Bands
        lower_band = middle_line - (STD_DEV_MULTIPLIER * std_dev)
        upper_band = middle_line + (STD_DEV_MULTIPLIER * std_dev)
        
        # Current Values
        current_price = y[-1]
        current_middle = middle_line[-1]
        current_lower = lower_band[-1]
        current_upper = upper_band[-1]
        
        # 4. CHECK POSITION
        position_status = "ABOVE" if current_price >= current_middle else "BELOW"
        
        if current_price >= current_middle:
            return {'pass': False, 'reason': 'Price Above Middle'}
        
        # 5. FIND INDICES
        dip_indices = np.where(y < lower_band)[0]
        peak_indices = np.where(y > upper_band)[0]
        
        # Extremas
        if len(dip_indices) == 0:
            return {'pass': False, 'reason': 'No Dip Found'}
            
        # Find ArgMin (Most Extreme Dip)
        dip_values = y[dip_indices]
        index_of_min_in_subset = np.argmin(dip_values)
        most_extreme_dip_index = dip_indices[index_of_min_in_subset]
        argmin_val = dip_values[index_of_min_in_subset]
        
        # Find ArgMax (Most Extreme Peak)
        argmax_val = 0
        most_extreme_peak_index = -1
        
        if len(peak_indices) > 0:
            peak_values = y[peak_indices]
            index_of_max_in_subset = np.argmax(peak_values)
            most_extreme_peak_index = peak_indices[index_of_max_in_subset]
            argmax_val = peak_values[index_of_max_in_subset]
            
            # Compare Recency
            if most_extreme_dip_index > most_extreme_peak_index:
                last_event = "ARGMIN (Dip)"
            else:
                return {'pass': False, 'reason': 'Peak More Recent'}
        else:
            last_event = "ARGMIN (Dip)" # No peak, dip is definitely the main event
            
        # Return Success Data
        return {
            'pass': True,
            'current': current_price,
            'middle': current_middle,
            'lower': current_lower,
            'upper': current_upper,
            'position': position_status,
            'argmin': argmin_val,
            'argmax': argmax_val,
            'last_event': last_event
        }
        
    except Exception as e:
        return {'pass': False, 'reason': str(e)}

# --- FILTER FUNCTIONS ---

def filter_15m(standardized_symbol):
    """Step 1: Check 15m timeframe"""
    interval = '15m'
    original_symbol = symbol_map[standardized_symbol]
    
    try:
        klines = trader.safe_get_klines(symbol=original_symbol, interval=interval)
        if not klines: return
        
        close = [float(entry[4]) for entry in klines]
        if not close or len(close) < MIN_SAMPLES: return
        
        data = check_trend_structure(close)
        
        if data['pass']: 
            with locks[0]:
                filtered_pairs15.append(standardized_symbol)
                results_map[standardized_symbol]['original'] = original_symbol
                results_map[standardized_symbol]['15m_data'] = data
                print(f'✅ 15m PASS: {standardized_symbol}')
    except Exception as e:
        pass

def filter_5m(standardized_symbol):
    """Step 2: Check 5m timeframe"""
    interval = '5m'
    original_symbol = symbol_map[standardized_symbol]
    
    try:
        klines = trader.safe_get_klines(symbol=original_symbol, interval=interval)
        if not klines: return
        
        close = [float(entry[4]) for entry in klines]
        if not close or len(close) < MIN_SAMPLES: return
        
        data = check_trend_structure(close)
        
        if data['pass']:
            with locks[1]:
                filtered_pairs5.append(standardized_symbol)
                results_map[standardized_symbol]['5m_data'] = data
                print(f'✅ 5m PASS: {standardized_symbol}')
    except Exception as e:
        pass

def filter_3m(standardized_symbol):
    """Step 3: Check 3m timeframe"""
    interval = '3m'
    original_symbol = symbol_map[standardized_symbol]
    
    try:
        klines = trader.safe_get_klines(symbol=original_symbol, interval=interval)
        if not klines: return
        
        close = [float(entry[4]) for entry in klines]
        if not close or len(close) < MIN_SAMPLES: return
        
        data = check_trend_structure(close)
        
        if data['pass']:
            with locks[2]:
                filtered_pairs3.append(standardized_symbol)
                results_map[standardized_symbol]['3m_data'] = data
                print(f'✅ 3m PASS: {standardized_symbol}')
    except Exception as e:
        pass

def sine_sorter_1m(standardized_symbol):
    """Step 4: Calculate HT_SINE"""
    interval = '1m'
    original_symbol = symbol_map[standardized_symbol]
    
    try:
        klines = trader.safe_get_klines(symbol=original_symbol, interval=interval)
        if not klines: return
        
        close = [float(entry[4]) for entry in klines]
        if not close or len(close) < 200: return
        
        close_array = np.asarray(close)
        sine, _ = ta.HT_SINE(close_array)
        
        valid_mask = (~np.isnan(sine)) & (sine != 0)
        clean_sine = sine[valid_mask]
        
        current_sine = clean_sine[-1] if len(clean_sine) > 0 else 0 
        
        with locks[3]:
            filtered_pairs_final.append(standardized_symbol)
            results_map[standardized_symbol]['sine'] = float(current_sine)
            
    except Exception as e:
        pass

def print_detailed_report(tf_name, pairs_list):
    """Prints the detailed regression data for passed assets"""
    print(f"\n{'='*100}")
    print(f"📊 {tf_name} DETAILED ANALYSIS (PASSED ASSETS)")
    print(f"{'='*100}")
    # Header
    print(f"{'Asset Name':<20} | {'Current Price':<12} | {'Lower Band':<12} | {'Middle Line':<12} | {'Upper Band':<12} | {'Pos':<5}")
    print("-" * 100)
    
    for symbol in pairs_list:
        # Symbol is the standardized English name
        data = results_map[symbol][f'{tf_name.lower()}_data']
        
        print(f"{symbol:<20} | "
              f"{data['current']:<12.4f} | "
              f"{data['lower']:<12.4f} | "
              f"{data['middle']:<12.4f} | "
              f"{data['upper']:<12.4f} | "
              f"{data['position']:<5}")

    print(f"\n{'Asset Name':<20} | {'ArgMin (Dip)':<12} | {'ArgMax (Peak)':<12} | {'Last Occurrence':<20}")
    print("-" * 70)
    
    for symbol in pairs_list:
        data = results_map[symbol][f'{tf_name.lower()}_data']
        
        print(f"{symbol:<20} | "
              f"{data['argmin']:<12.4f} | "
              f"{data['argmax']:<12.4f} | "
              f"{data['last_event']:<20}")
    
    print(f"{'='*100}")

# --- MAIN EXECUTION BLOCK ---

filename = 'credentials.txt'
trader = Trader(filename)

symbol_map = trader.get_all_usdc_pairs()
standardized_symbols = list(symbol_map.keys())

# Thread-safe storage
filtered_pairs15 = []
filtered_pairs5 = []
filtered_pairs3 = []
filtered_pairs_final = []

results_map = defaultdict(dict)

locks = [threading.Lock() for _ in range(4)]

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

    scan_attempt = 0

    while True:
        scan_attempt += 1
        print(f"\n\n{'#'*70}")
        print(f"# 🔄 STARTING MARKET SCAN #{scan_attempt}")
        print(f"{'#'*70}")

        # Clean Data
        filtered_pairs15.clear()
        filtered_pairs5.clear()
        filtered_pairs3.clear()
        filtered_pairs_final.clear()
        results_map.clear()

        # --- STAGE 1: 15m ---
        concurrent_filter_stage(standardized_symbols, filter_15m, "15-MINUTE FILTER (Extrema Logic)")
        
        if filtered_pairs15:
            print(f"\n✅ {len(filtered_pairs15)} pairs passed 15m structure.")
            print_detailed_report('15m', filtered_pairs15)
        else:
            print(f"⏳ No 15m dips found. Sleeping {SLEEP_TIME}s...")
            time.sleep(SLEEP_TIME)
            continue

        # --- STAGE 2: 5m ---
        concurrent_filter_stage(filtered_pairs15, filter_5m, "5-MINUTE FILTER (Extrema Logic)")
        
        if filtered_pairs5:
            print(f"\n✅ {len(filtered_pairs5)} pairs passed 5m structure.")
            print_detailed_report('5m', filtered_pairs5)
        else:
            print(f"⏳ No 5m dips found. Sleeping {SLEEP_TIME}s...")
            time.sleep(SLEEP_TIME)
            continue
            
        # --- STAGE 3: 3m ---
        concurrent_filter_stage(filtered_pairs5, filter_3m, "3-MINUTE FILTER (Extrema Logic)")
        
        if filtered_pairs3:
            print(f"\n✅ {len(filtered_pairs3)} pairs passed 3m structure.")
            print_detailed_report('3m', filtered_pairs3)
        else:
            print(f"⏳ No 3m dips found. Sleeping {SLEEP_TIME}s...")
            time.sleep(SLEEP_TIME)
            continue
        
        # --- STAGE 4: 1m Sine Sorter ---
        concurrent_filter_stage(filtered_pairs3, sine_sorter_1m, "1-MINUTE SORTER (HT_SINE Cycle)")
        
        if filtered_pairs_final:
            print(f"\n✅ MTF DIP FOUND! ANALYSIS COMPLETE.")
            
            print(f"\n{'='*70}")
            print("🏆 FINAL QUALIFIED PAIRS (1m HT_SINE Cycle Bottom)")
            print(f"{'='*70}")
            
            sine_candidates = []
            for symbol in filtered_pairs_final:
                sine_val = results_map[symbol].get('sine', 0)
                if sine_val is not None:
                    sine_candidates.append((symbol, sine_val))
            
            if sine_candidates:
                sorted_candidates = sorted(sine_candidates, key=lambda x: x[1])
                
                print(f"🎯 Found {len(sorted_candidates)} dip candidates ranked by Cycle Bottom (Lowest Sine):\n")
                
                for rank, (std_symbol, sine_val) in enumerate(sorted_candidates, 1):
                    if sine_val < -0.8: level = '🔴 DEEP CYCLE BOTTOM'
                    elif sine_val < -0.5: level = '🟠 NEAR BOTTOM'
                    elif sine_val < 0: level = '🟡 LOWER HALF'
                    else: level = '⚪ NEUTRAL/RISING'
                    
                    print(f"{rank:2d}. {std_symbol:20} | 🌊 1m HT_Sine: {sine_val:.4f} | {level}")
                
                # Best Selection
                best_std, best_sine = sorted_candidates[0]
                print(f"\n🏆 TOP SELECTION: {best_std} (Sine: {best_sine:.4f})")
            
            # Summary
            print(f"\n{'='*70}")
            print("📊 SCAN SUMMARY")
            print(f"{'='*70}")
            print(f"Total pairs scanned:     {len(symbol_map)}")
            print(f"Passed 15m filter:       {len(filtered_pairs15)}")
            print(f"Passed 5m filter:        {len(filtered_pairs5)}")
            print(f"Passed 3m filter:        {len(filtered_pairs3)}")
            print(f"Final Sine Candidates:   {len(filtered_pairs_final)}")
            print(f"{'='*70}")

            break # Loop ends on success
            
        else:
            print(f"⏳ No candidates met final Sine criteria. Sleeping {SLEEP_TIME}s...")
            time.sleep(SLEEP_TIME)

if __name__ == "__main__":
    main()
    sys.exit(0)
