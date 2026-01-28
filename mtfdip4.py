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
MIN_REQUIRED_SAMPLES = 20 # Minimum candles to perform regression calculation
STD_DEV_MULTIPLIER = 1.0 # Multiplier 1.0 (1 Deviation) as requested
SLEEP_TIME = 5 

# --- NEW FFT MTF CONFIGURATION ---
FFT_TFS = {
    '1m': 32,
    '3m': 64,
    '5m': 96,
    '15m': 128,
    '30m': 160,
    '1h': 192
}

FFT_LOW_FREQS = 4
VOLUME_LOOKBACK = 20
MIN_VOLUME_INCREASE = 1.25  # 25% accumulation threshold
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

# --- ORIGINAL FFT FORECAST LOGIC ---

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
        
        # Simple projection: use momentum of the smoothed signal
        # If the last value of smoothed signal is higher than previous, it's bullish
        momentum = current_trend_val - smoothed_signal[-2].real
        
        # Forecast next price (Linear projection based on smoothed momentum)
        forecast = current_trend_val + momentum
        
        # Check for "Imminent Spike"
        # If the forecast is significantly higher than the current actual price
        current_actual = recent_prices[-1]
        spike_confirmed = (forecast > current_actual * 1.005) # 0.5% projected rise threshold
        
        return forecast, spike_confirmed
        
    except:
        return 0.0, False

# --- NEW MTF FFT ENGINE FUNCTIONS ---

def fft_mtf_forecast(prices, lookback):
    """
    Flexible FFT forecast for variable lookbacks.
    Returns dictionary with forecast, momentum, current price and strength.
    """
    if not prices or len(prices) < lookback:
        return None

    try:
        data = np.array(prices[-lookback:])
        fft_vals = np.fft.fft(data)
        
        # Filter noise using FFT_LOW_FREQS
        # Ensure we don't exceed array bounds
        cutoff = min(FFT_LOW_FREQS, len(fft_vals))
        fft_vals[cutoff:] = 0
        
        smooth = np.fft.ifft(fft_vals).real

        momentum = smooth[-1] - smooth[-2]
        forecast = smooth[-1] + momentum

        return {
            'forecast': float(forecast),
            'momentum': float(momentum),
            'current': float(data[-1]),
            'strength': abs(momentum) / data[-1] if data[-1] != 0 else 0
        }
    except Exception as e:
        return None

def volume_accumulation(volumes):
    """
    Checks if recent volume is significantly higher than previous volume.
    Returns (bool_passed, ratio).
    """
    if not volumes or len(volumes) < VOLUME_LOOKBACK * 2:
        return False, 0.0

    try:
        recent_vol = np.mean(volumes[-VOLUME_LOOKBACK:])
        previous_vol = np.mean(volumes[-2*VOLUME_LOOKBACK:-VOLUME_LOOKBACK])

        if previous_vol == 0:
            return False, 0.0

        ratio = recent_vol / previous_vol
        return ratio >= MIN_VOLUME_INCREASE, ratio
    except:
        return False, 0.0

def argmax_target(prices):
    """
    Finds the highest price (structural resistance) in the dataset.
    """
    if not prices or len(prices) == 0:
        return None
    try:
        return float(np.max(prices))
    except:
        return None

# --- CORE STRUCTURE LOGIC (STRICT BREACHES) ---

def check_trend_structure(prices):
    """
    Calculates Global ArgMin and ArgMax based on last 100 candles (or max available).
    STRICT LOGIC with STD_DEV_MULTIPLIER = 1.0:
    1. ArgMin MUST be below Lower Band (1.0 * StdDev).
    2. ArgMax MUST be above Upper Band (1.0 * StdDev).
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
        # Dynamic Lookback: Use LOOKBACK_LENGTH or available data
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

        # 3. Define Bands (Using 1.0 Multiplier)
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
        # Check A: Is ArgMin STRICTLY below Lower Band at that time?
        is_argmin_dip = argmin_val < lower_band[global_min_idx]
        # Check B: Is ArgMax STRICTLY above Upper Band at that time?
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
        # We want ArgMin (Dip) to be most recent event.
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
            'current': current_price,
            'middle': current_middle,
            'lower': current_lower,
            'upper': current_upper,
            'position': position_status,
            'argmin': argmin_val,
            'argmax': argmax_val,
            'last_event': 'ARGMIN'
        }
        
    except Exception as e:
        return {**default_resp, 'reason': f"ERROR: {str(e)[:20]}"}

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
            close_array = np.asarray(close)
            
            if len(close_array) >= 200:
                sine, _ = ta.HT_SINE(close_array)
                valid_mask = (~np.isnan(sine)) & (sine != 0)
                clean_sine = sine[valid_mask]
                current_sine = clean_sine[-1] if len(clean_sine) > 0 else 0
                
                # --- CALL ORIGINAL FFT FORECAST HERE ---
                fft_price, fft_spike = calculate_fft_forecast(close)
                
                with locks[3]:
                    filtered_pairs_final.append(standardized_symbol)
                    results_map[standardized_symbol]['sine'] = float(current_sine)
                    results_map[standardized_symbol]['1m_close_list'] = close # Store for FFT later
                    results_map[standardized_symbol]['fft_forecast'] = fft_price
                    results_map[standardized_symbol]['fft_spike'] = fft_spike
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
                '1m_close_list': None,
                'fft_forecast': None,
                'fft_spike': False
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
                
                # --- ORIGINAL 1m FFT FORECAST FOR THE WINNER ---
                best_std, best_sine = sorted_candidates[0]
                best_orig = symbol_map[best_std]
                
                print(f"\n🏆 TOP SELECTION: {best_std} (Original: {best_orig})")
                print(f"{'='*70}")
                print("📡 1m FFT FORECAST ANALYSIS (Initial Spike Detection)")
                print(f"{'='*70}")
                
                fft_price = results_map[best_std].get('fft_forecast', 0.0)
                fft_spike = results_map[best_std].get('fft_spike', False)
                current_best = results_map[best_std].get('1m_close_list', [0])[-1]
                
                print(f"Current Price:    {current_best:.25f}")
                print(f"1m FFT Projection: {fft_price:.25f}")
                
                if fft_spike:
                    print(f"🚨 STATUS: 🚨 1m IMMINENT SPIKE DETECTED! 🚨")
                else:
                    print(f"STATUS: 📊 Stabilization / No immediate 1m spike projected.")
                
                print(f"{'='*70}")

                # --- NEW MTF FFT REVERSAL TARGET ENGINE (FIXED) ---
                print(f"\n{'='*160}")
                print("📡 MULTI-TIMEFRAME FFT REVERSAL TARGET ENGINE (INSTITUTIONAL GRADE)")
                print(f"{'='*160}")
                
                mtf_targets = []
                best_score = -1
                best_tf = 'None'
                
                # Headers for MTF Table
                print(f"{'TF':<6} | {'FFT Forecast':<32} | {'ArgMax Target':<32} | {'Final Target':<32} | {'Status':<10} | {'Confidence'}")
                print("-" * 160)

                for tf, lb in FFT_TFS.items():
                    try:
                        kl = trader.safe_get_klines(best_orig, tf)
                        if not kl:
                            continue

                        close = [float(x[4]) for x in kl]
                        vol = [float(x[5]) for x in kl]

                        fft = fft_mtf_forecast(close, lb)
                        if not fft:
                            continue

                        argmax_price = argmax_target(close)
                        vol_ok, vol_ratio = volume_accumulation(vol)

                        # --- FIX START ---
                        # Determine if FFT is Bullish or Bearish relative to Current Price
                        current_tf_price = fft['current']
                        fft_proj = fft['forecast']
                        
                        fft_bullish = fft_proj >= current_tf_price
                        status_str = "BULLISH" if fft_bullish else "BEARISH"

                        if fft_bullish:
                            # FFT agrees with direction. Use the higher of FFT or ArgMax.
                            projected_target = max(fft_proj, argmax_price) if argmax_price else fft_proj
                            confidence_mult = 1.0
                        else:
                            # FFT points down. This is bad for a LONG trade.
                            # We ignore FFT projection and rely solely on Structural ArgMax.
                            projected_target = argmax_price
                            # HEAVILY PENALIZE confidence so bearish TFs don't win
                            confidence_mult = 0.1 
                        # --- FIX END ---

                        # Confidence calculation
                        confidence = (
                            fft['strength'] *
                            confidence_mult * # Apply multiplier based on FFT direction
                            (1.5 if vol_ok else 1.0) *
                            (1 + vol_ratio)
                        )

                        mtf_targets.append({
                            'tf': tf,
                            'forecast': fft_proj,
                            'argmax': argmax_price,
                            'final_target': projected_target,
                            'volume_ratio': vol_ratio,
                            'confidence': confidence
                        })

                        # Format string for table row
                        print(f"{tf:<6} | "
                              f"{fft_proj:<32.25f} | "
                              f"{argmax_price if argmax_price else 0.0:<32.25f} | "
                              f"{projected_target:<32.25f} | "
                              f"{status_str:<10} | "
                              f"{confidence:.6f}")

                        if confidence > best_score:
                            best_score = confidence
                            best_tf = tf

                    except Exception as e:
                        # Silently skip timeframe errors to keep engine running
                        pass

                # --- FINAL BEST MTF TARGET OUTPUT ---
                if mtf_targets:
                    # Find best target based on calculated confidence score
                    best = max(mtf_targets, key=lambda x: x['confidence'])
                    
                    print(f"\n{'='*160}")
                    print("🏆 FINAL HIGH-PROBABILITY MTF TARGET")
                    print(f"{'='*160}")
                    print(f"Asset:          {best_std}")
                    print(f"Best Timeframe: {best['tf']}")
                    print(f"FFT Target:     {best['forecast']:.25f}")
                    print(f"ArgMax Target:  {best['argmax']:.25f}")
                    print(f"FINAL TARGET:   {best['final_target']:.25f}")
                    print(f"Volume Ratio:   x{best['volume_ratio']:.2f}")
                    print(f"Confidence:     {best['confidence']:.6f}")
                    print("🚀 EXPECTED: STRUCTURAL REVERSAL → VOLUME EXPANSION → SPIKE")
                    print(f"{'='*160}")
                else:
                    print("\n⚠️ Could not generate MTF targets due to insufficient data for all timeframes.")

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