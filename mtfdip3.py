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
LOOKBACK_LENGTH = 100  # Last 100 values to check
MIN_REQUIRED_SAMPLES = 20 # Minimum to attempt calculation
STD_DEV_MULTIPLIER = 2.0 # Multiplier 2.0 as requested
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
    
    def get_ticker_fallback(self, symbol):
        """Fallback to get current price if klines fail"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except:
            return 0.0

# --- CORE STRUCTURE LOGIC (STRICT BREACHES) ---

def check_trend_structure(prices):
    """
    Calculates Global ArgMin and ArgMax based on last 100 candles (or max available).
    STRICT LOGIC:
    1. ArgMin MUST be below Lower Band.
    2. ArgMax MUST be above Upper Band.
    3. ArgMin (Dip) must be more recent than ArgMax (Peak).
    4. Current Close MUST be below Middle Line.
    """
    default_resp = {
        'pass': False, 
        'reason': 'NO_DATA', 
        'current': 0, 'middle': 0, 'lower': 0, 'upper': 0, 
        'position': 'N/A', 'argmin': 0, 'argmax': 0, 'last_event': 'N/A'
    }

    if not prices or len(prices) < MIN_REQUIRED_SAMPLES:
        return {**default_resp, 'reason': 'INSUFFICIENT_DATA'}
    
    try:
        # Dynamic Lookback: Use LOOKBACK_LENGTH or available data (up to LOOKBACK_LENGTH)
        # If we have less than LOOKBACK_LENGTH (e.g. new token), we use all available.
        prices_recent = np.asarray(prices[-LOOKBACK_LENGTH:])
        x = np.arange(len(prices_recent))
        y = prices_recent
        
        # 1. Calculate Linear Regression
        m, b = np.polyfit(x, y, 1)
        middle_line = m * x + b
        
        # 2. Calculate Standard Deviation
        residuals = y - middle_line
        std_dev = np.std(residuals, ddof=1) 
        
        if std_dev == 0:
            return {**default_resp, 'reason': 'FLAT_LINE'}

        # 3. Define Bands (Multiplier 2.0)
        lower_band = middle_line - (STD_DEV_MULTIPLIER * std_dev)
        upper_band = middle_line + (STD_DEV_MULTIPLIER * std_dev)
        
        # 4. Current Values
        current_price = y[-1]
        current_middle = middle_line[-1]
        current_lower = lower_band[-1]
        current_upper = upper_band[-1]
        
        # 5. Position Status
        position_status = "ABOVE" if current_price >= current_middle else "BELOW"
        
        # 6. FIND GLOBAL EXTREMAS (ARGMIN / ARGMAX)
        global_min_idx = np.argmin(y)
        global_max_idx = np.argmax(y)
        
        argmin_val = y[global_min_idx]
        argmax_val = y[global_max_idx]
        
        # 7. STRICT BREACH CHECKS
        # Check A: Is ArgMin STRICTLY below the Lower Band at that time?
        is_argmin_dip = argmin_val < lower_band[global_min_idx]
        # Check B: Is ArgMax STRICTLY above the Upper Band at that time?
        is_argmax_peak = argmax_val > upper_band[global_max_idx]
        
        if not is_argmin_dip:
            return {
                'pass': False, 'reason': 'MIN_NOT_BELOW',
                'current': current_price, 'middle': current_middle, 'lower': current_lower, 'upper': current_upper,
                'position': position_status, 'argmin': argmin_val, 'argmax': argmax_val, 'last_event': 'N/A'
            }
            
        if not is_argmax_peak:
            return {
                'pass': False, 'reason': 'MAX_NOT_ABOVE',
                'current': current_price, 'middle': current_middle, 'lower': current_lower, 'upper': current_upper,
                'position': position_status, 'argmin': argmin_val, 'argmax': argmax_val, 'last_event': 'N/A'
            }

        # 8. RECENCY CHECK (CIRCUIT)
        # We want ArgMin (Dip) to be more recent than ArgMax (Peak)
        if global_min_idx <= global_max_idx:
            # Peak is recent or equal -> Fail
            return {
                'pass': False, 'reason': 'RECENT_PEAK',
                'current': current_price, 'middle': current_middle, 'lower': current_lower, 'upper': current_upper,
                'position': position_status, 'argmin': argmin_val, 'argmax': argmax_val, 'last_event': 'ARGMAX'
            }
        
        # 9. CURRENT POSITION CHECK
        # Current Price must be below Middle Line
        if position_status == "ABOVE":
            return {
                'pass': False, 'reason': 'CURRENT_ABOVE_MIDDLE',
                'current': current_price, 'middle': current_middle, 'lower': current_lower, 'upper': current_upper,
                'position': position_status, 'argmin': argmin_val, 'argmax': argmax_val, 'last_event': 'ARGMIN'
            }
        
        # ALL CHECKS PASSED
        return {
            'pass': True, 
            'reason': 'PASS',
            'current': current_price, 'middle': current_middle, 'lower': current_lower, 'upper': current_upper,
            'position': position_status, 'argmin': argmin_val, 'argmax': argmax_val, 'last_event': 'ARGMIN'
        }
        
    except Exception as e:
        return {**default_resp, 'reason': f"ERROR: {str(e)[:20]}"}

# --- FFT FORECAST LOGIC ---

def calculate_fft_forecast(prices):
    """
    Uses FFT to smooth noise and project the next immediate trend.
    Returns (forecasted_price, spike_confirmed_bool).
    """
    if not prices or len(prices) < 32:
        return 0.0, False
    
    try:
        # Take last 32 points for FFT analysis
        recent_prices = np.array(prices[-32:])
        
        # Apply FFT
        fft_vals = np.fft.fft(recent_prices)
        
        # Keep only the lowest 4 frequencies (trend) to filter out noise
        # Zero out high frequencies
        fft_vals[4:] = 0
        
        # Inverse FFT to get the smoothed signal
        smoothed_signal = np.fft.ifft(fft_vals)
        
        # The last real value of the smoothed signal is our "current trend" value
        current_trend_val = smoothed_signal[-1].real
        
        # Simple projection: use the momentum of the smoothed signal
        # If the last value of smoothed signal is higher than the previous, it's bullish
        momentum = current_trend_val - smoothed_signal[-2].real
        
        # Forecasted next price (Linear projection based on smoothed momentum)
        forecast = current_trend_val + momentum
        
        # Check for "Imminent Spike"
        # If the forecast is significantly higher than current actual price
        current_actual = recent_prices[-1]
        spike_confirmed = (forecast > current_actual * 1.005) # 0.5% projected rise threshold
        
        return forecast, spike_confirmed
        
    except:
        return 0.0, False

# --- FILTER FUNCTIONS ---

def filter_15m(standardized_symbol):
    interval = '15m'
    original_symbol = symbol_map[standardized_symbol]
    
    try:
        klines = trader.safe_get_klines(symbol=original_symbol, interval=interval)
        close = []
        if klines:
            close = [float(entry[4]) for entry in klines]
        
        data = check_trend_structure(close)
        
        if not data['pass'] and (data['reason'] == 'INSUFFICIENT_DATA' or data['current'] == 0):
             fallback_price = trader.get_ticker_fallback(original_symbol)
             data['current'] = fallback_price

        with locks[0]:
            results_map[standardized_symbol]['15m_data'] = data
            results_map[standardized_symbol]['original'] = original_symbol 
        
        if data['pass']: 
            filtered_pairs15.append(standardized_symbol)
            
    except Exception as e:
        with locks[0]:
            fallback_price = trader.get_ticker_fallback(symbol_map[standardized_symbol])
            results_map[standardized_symbol]['15m_data'] = {
                'pass': False, 'reason': 'API_CRASH', 'current': fallback_price, 'middle': 0, 
                'lower': 0, 'upper': 0, 'position': 'ERR', 'argmin': 0, 'argmax': 0, 'last_event': 'CRASH'
            }

def filter_5m(standardized_symbol):
    interval = '5m'
    original_symbol = symbol_map[standardized_symbol]
    
    try:
        klines = trader.safe_get_klines(symbol=original_symbol, interval=interval)
        close = []
        if klines:
            close = [float(entry[4]) for entry in klines]
            
        data = check_trend_structure(close)
        
        if not data['pass'] and (data['reason'] == 'INSUFFICIENT_DATA' or data['current'] == 0):
             fallback_price = trader.get_ticker_fallback(original_symbol)
             data['current'] = fallback_price

        with locks[1]:
            results_map[standardized_symbol]['5m_data'] = data
        
        if data['pass']:
            filtered_pairs5.append(standardized_symbol)
            
    except Exception as e:
        fallback_price = trader.get_ticker_fallback(symbol_map[standardized_symbol])
        with locks[1]:
            results_map[standardized_symbol]['5m_data'] = {
                'pass': False, 'reason': 'API_CRASH', 'current': fallback_price, 'middle': 0, 
                'lower': 0, 'upper': 0, 'position': 'ERR', 'argmin': 0, 'argmax': 0, 'last_event': 'CRASH'
            }

def filter_3m(standardized_symbol):
    interval = '3m'
    original_symbol = symbol_map[standardized_symbol]
    
    try:
        klines = trader.safe_get_klines(symbol=original_symbol, interval=interval)
        close = []
        if klines:
            close = [float(entry[4]) for entry in klines]
            
        data = check_trend_structure(close)

        if not data['pass'] and (data['reason'] == 'INSUFFICIENT_DATA' or data['current'] == 0):
             fallback_price = trader.get_ticker_fallback(original_symbol)
             data['current'] = fallback_price

        with locks[2]:
            results_map[standardized_symbol]['3m_data'] = data
        
        if data['pass']:
            filtered_pairs3.append(standardized_symbol)
            
    except Exception as e:
        fallback_price = trader.get_ticker_fallback(symbol_map[standardized_symbol])
        with locks[2]:
            results_map[standardized_symbol]['3m_data'] = {
                'pass': False, 'reason': 'API_CRASH', 'current': fallback_price, 'middle': 0, 
                'lower': 0, 'upper': 0, 'position': 'ERR', 'argmin': 0, 'argmax': 0, 'last_event': 'CRASH'
            }

def sine_sorter_1m(standardized_symbol):
    interval = '1m'
    original_symbol = symbol_map[standardized_symbol]
    
    try:
        klines = trader.safe_get_klines(symbol=original_symbol, interval=interval)
        if klines:
            close = [float(entry[4]) for entry in klines]
            
            # HT_SINE Calculation
            if len(close) >= 200:
                close_array = np.asarray(close)
                sine, _ = ta.HT_SINE(close_array)
                valid_mask = (~np.isnan(sine)) & (sine != 0)
                clean_sine = sine[valid_mask]
                current_sine = clean_sine[-1] if len(clean_sine) > 0 else 0
                
                with locks[3]:
                    filtered_pairs_final.append(standardized_symbol)
                    results_map[standardized_symbol]['sine'] = float(current_sine)
                    results_map[standardized_symbol]['1m_close_list'] = close # Store for FFT later
            else:
                with locks[3]:
                    results_map[standardized_symbol]['sine'] = None
    except:
        pass

def initialize_results_map(symbol_list):
    """Pre-populate results map"""
    for symbol in symbol_list:
        if symbol not in results_map:
            results_map[symbol] = {
                'original': symbol_map.get(symbol, symbol),
                '15m_data': None,
                '5m_data': None,
                '3m_data': None,
                'sine': None,
                '1m_close_list': None
            }

def print_consolidated_report(tf_name, scanned_list):
    """Prints a single comprehensive table with 25 decimal floats"""
    print(f"\n{'='*160}")
    print(f"📊 {tf_name} CONSOLIDATED ANALYSIS (SCANNED: {len(scanned_list)})")
    print(f"{'='*160}")
    
    # Using .25f format for 25 decimal places as requested
    print(f"{'Asset':<18} | {'Current':<32} | {'Lower':<32} | {'Middle':<32} | {'Upper':<32} | {'Pos':<6} | {'ArgMin':<32} | {'ArgMax':<32} | {'STATUS'}")
    print("-" * 160)
    
    count_pass = 0
    
    for symbol in scanned_list:
        if symbol in results_map and f'{tf_name.lower()}_data' in results_map[symbol]:
            d = results_map[symbol][f'{tf_name.lower()}_data']
            
            if d is None:
                print(f"{symbol:<18} | {'N/A':<32} | ...")
                continue

            # YES/NO based on logic pass
            status_str = "YES" if d['pass'] else "NO"
            
            print(f"{symbol:<18} | "
                  f"{d['current']:<32.25f} | "
                  f"{d['lower']:<32.25f} | "
                  f"{d['middle']:<32.25f} | "
                  f"{d['upper']:<32.25f} | "
                  f"{d['position']:<6} | "
                  f"{d['argmin']:<32.25f} | "
                  f"{d['argmax']:<32.25f} | "
                  f"{status_str}")
            
            if d['pass']:
                count_pass += 1
        else:
            print(f"{symbol:<18} | MISSING DATA")

    print(f"{'='*160}")

# --- MAIN EXECUTION BLOCK ---

filename = 'credentials.txt'
trader = Trader(filename)

symbol_map = trader.get_all_usdc_pairs()
standardized_symbols = list(symbol_map.keys())

filtered_pairs15 = []
filtered_pairs5 = []
filtered_pairs3 = []
filtered_pairs_final = []

results_map = defaultdict(dict)
locks = [threading.Lock() for _ in range(4)]

def concurrent_filter_stage(pairs_list, filter_func, stage_name, max_workers=10):
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

        initialize_results_map(standardized_symbols)

        filtered_pairs15.clear()
        filtered_pairs5.clear()
        filtered_pairs3.clear()
        filtered_pairs_final.clear()

        # --- STAGE 1: 15m ---
        concurrent_filter_stage(standardized_symbols, filter_15m, "15-MINUTE FILTER")
        print_consolidated_report('15m', standardized_symbols)
        
        if not filtered_pairs15:
            print(f"⏳ No 15m dips found. Sleeping {SLEEP_TIME}s...")
            time.sleep(SLEEP_TIME)
            continue

        # --- STAGE 2: 5m ---
        concurrent_filter_stage(filtered_pairs15, filter_5m, "5-MINUTE FILTER")
        print_consolidated_report('5m', filtered_pairs15)
        
        if not filtered_pairs5:
            print(f"⏳ No 5m dips found. Sleeping {SLEEP_TIME}s...")
            time.sleep(SLEEP_TIME)
            continue
            
        # --- STAGE 3: 3m ---
        concurrent_filter_stage(filtered_pairs5, filter_3m, "3-MINUTE FILTER")
        print_consolidated_report('3m', filtered_pairs5)
        
        if not filtered_pairs3:
            print(f"⏳ No 3m dips found. Sleeping {SLEEP_TIME}s...")
            time.sleep(SLEEP_TIME)
            continue
        
        # --- STAGE 4: 1m Sine + FFT ---
        concurrent_filter_stage(filtered_pairs3, sine_sorter_1m, "1-MINUTE SORTER (HT_SINE)")
        
        if filtered_pairs_final:
            print(f"\n✅ MTF DIP FOUND! ANALYSIS COMPLETE.")
            
            print(f"\n{'='*70}")
            print("🏆 FINAL QUALIFIED PAIRS (HT_SINE Cycle Bottom)")
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
                    
                    print(f"{rank:2d}. {std_symbol:20} | 🌊 1m HT_Sine: {sine_val:.25f} | {level}")
                
                # --- FFT FORECAST FOR THE WINNER ---
                best_std, best_sine = sorted_candidates[0]
                best_orig = symbol_map[best_std]
                close_list_1m = results_map[best_std].get('1m_close_list', [])
                
                print(f"\n🏆 TOP SELECTION: {best_std} (Original: {best_orig})")
                print(f"{'='*70}")
                print("📡 FFT FORECAST ANALYSIS (Imminent Spike Detection)")
                print(f"{'='*70}")
                
                forecast_price, spike_detected = calculate_fft_forecast(close_list_1m)
                
                print(f"Current Price:    {close_list_1m[-1]:.25f}")
                print(f"FFT Projection:   {forecast_price:.25f}")
                
                if spike_detected:
                    print(f"🚨 STATUS: 🚨 IMMINENT SPIKE DETECTED! 🚨")
                else:
                    print(f"STATUS: 📊 Stabilization / No immediate spike projected.")
                
                print(f"{'='*70}")
            
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
