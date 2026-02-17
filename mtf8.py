from binance.client import Client
import numpy as np
import talib as ta
import sys
import concurrent.futures
import re

# --- Configuration ---
RSI_LENGTH = 14
VOLUME_SMA_LENGTH = 20
LOOKBACK_LIMIT = 500  # Window for Extrema calculation

# Physics/Thermo Constants
PHI = 1.61803398875
GOLDEN_RATIO_DERIV = 0.61803398875

class Trader:
    def __init__(self, file):
        self.connect(file)

    def connect(self, file):
        try:
            lines = [line.rstrip('\n') for line in open(file)]
            key = lines[0]
            secret = lines[1]
            self.client = Client(key, secret)
        except Exception as e:
            print(f"Error connecting to Binance: {e}")
            sys.exit(1)

    def get_usdc_pairs(self):
        try:
            exchange_info = self.client.get_exchange_info()
            trading_pairs = []
            
            for symbol in exchange_info['symbols']:
                # 1. Must be USDC pair
                if symbol['quoteAsset'] != 'USDC':
                    continue
                    
                # 2. Must be Trading
                if symbol['status'] != 'TRADING':
                    continue
                
                # 3. Strict English Abbreviation Check
                base_asset = symbol['baseAsset']
                if not re.match(r'^[A-Z0-9]+$', base_asset):
                    continue
                
                trading_pairs.append(symbol['symbol'])
                
            return trading_pairs
        except Exception as e:
            print(f"Error fetching pairs: {e}")
            return []

filename = 'credentials.txt'
trader = Trader(filename)

def get_klines_data(client, symbol, interval, limit=500):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines or len(klines) < 50: 
            return None
        
        closes = np.array([float(entry[4]) for entry in klines], dtype=np.float64)
        highs = np.array([float(entry[2]) for entry in klines], dtype=np.float64)
        lows = np.array([float(entry[3]) for entry in klines], dtype=np.float64)
        volumes = np.array([float(entry[5]) for entry in klines], dtype=np.float64)
        
        return closes, highs, lows, volumes
    except Exception:
        return None

def analyze_extrema(lows, highs, timeframe_name):
    """
    Analyzes Lowest Low and Highest High over the LOOKBACK_LIMIT.
    Determines which occurred more recently (ArgMin vs ArgMax).
    """
    l_vals = lows[-LOOKBACK_LIMIT:]
    h_vals = highs[-LOOKBACK_LIMIT:]
    
    ll_val = np.min(l_vals)
    hh_val = np.max(h_vals)
    
    idx_ll = int(np.argmin(l_vals))
    idx_hh = int(np.argmax(h_vals))
    
    # Convert to "bars ago" (0 = most recent)
    bars_ago_ll = (LOOKBACK_LIMIT - 1) - idx_ll
    bars_ago_hh = (LOOKBACK_LIMIT - 1) - idx_hh
    
    recent_extrema = "NONE"
    if bars_ago_ll < bars_ago_hh:
        recent_extrema = "ARGMIN (Low)"
    else:
        recent_extrema = "ARGMAX (High)"
        
    return {
        'timeframe': timeframe_name,
        'll': ll_val,
        'hh': hh_val,
        'bars_ago_ll': bars_ago_ll,
        'bars_ago_hh': bars_ago_hh,
        'recent_type': recent_extrema
    }

def check_volume_induction(lows, volumes):
    """
    Checks if the lowest low was accompanied by a volume spike.
    """
    l_vals = lows[-LOOKBACK_LIMIT:]
    v_vals = volumes[-LOOKBACK_LIMIT:]
    
    idx_ll = int(np.argmin(l_vals))
    vol_at_ll = v_vals[idx_ll]
    
    avg_vol = np.mean(v_vals)
    
    is_induction = vol_at_ll > (avg_vol * 1.2)
    return is_induction, vol_at_ll, avg_vol

def analyze_ht_sine(closes):
    """
    Uses Hilbert Transform Sine Wave to detect cycle turns.
    """
    try:
        sine, leadsine = ta.HT_SINE(closes)
        
        s_val = sine[-1]
        ls_val = leadsine[-1]
        s_prev = sine[-2]
        ls_prev = leadsine[-2]
        
        signal = "HOLD"
        
        if ls_val > s_val and ls_prev <= s_prev:
            signal = "REVERSAL_UP" 
        elif ls_val < s_val and ls_prev >= s_prev:
            signal = "REVERSAL_DOWN" 
            
        return signal, s_val, ls_val
    except:
        return "ERROR", 0, 0

def calculate_fft_forecast(closes):
    """
    FFT Forecast - Used for direction validation only.
    """
    n = len(closes)
    t = np.arange(n)
    coeffs = np.polyfit(t, closes, 1)
    trend = np.polyval(coeffs, t)
    detrended = closes - trend
    
    fft_vals = np.fft.fft(detrended)
    freqs = np.fft.fftfreq(n)
    
    indices = np.argsort(np.abs(fft_vals))[::-1]
    
    forecast_bars = 10
    future_t = np.arange(n, n + forecast_bars)
    forecast_signal = np.zeros(forecast_bars)
    
    keep_n = 3
    for i in indices[:keep_n]:
        amp = np.abs(fft_vals[i]) / n
        phase = np.angle(fft_vals[i])
        freq = freqs[i]
        forecast_signal += amp * np.cos(2 * np.pi * freq * future_t + phase)
    
    future_trend = np.polyval(coeffs, future_t)
    forecast_prices = forecast_signal + future_trend
    
    return forecast_prices

def check_mtf_dip(symbols, client):
    candidates = []
    
    # Expanded Timeframes
    timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h']
    
    def process_symbol(symbol):
        data_map = {}
        
        # Fetch all TFs
        for tf in timeframes:
            data = get_klines_data(client, symbol, tf, 500)
            if data:
                data_map[tf] = data
            else:
                return None # Skip if any TF data missing
        
        # Analyze Extrema for all TFs
        extrema_map = {}
        for tf, data in data_map.items():
            c, h, l, v = data
            extrema_map[tf] = analyze_extrema(l, h, tf)
        
        # --- CRITERIA CHECK ---
        
        # 1. Strict 1m and 5m Criteria: MUST be ArgMin Recent
        if extrema_map['1m']['recent_type'] != "ARGMIN (Low)":
            return None
        if extrema_map['5m']['recent_type'] != "ARGMIN (Low)":
            return None
            
        # 2. Majority TF Criteria: Most TFs must have ArgMin Recent
        argmin_count = 0
        for tf in timeframes:
            if extrema_map[tf]['recent_type'] == "ARGMIN (Low)":
                argmin_count += 1
        
        # Require strict majority (e.g., > 4 out of 8)
        if argmin_count < 5:
            return None
            
        # 3. Volume Induction (1m)
        c_1m, h_1m, l_1m, v_1m = data_map['1m']
        induction_1m, vol_ll, avg_vol = check_volume_induction(l_1m, v_1m)
        if not induction_1m:
            return None

        # 4. Target Calculation (Phi Resistance 0.618 from 5m)
        c_5m, h_5m, l_5m, v_5m = data_map['5m']
        hh_5m = extrema_map['5m']['hh']
        ll_5m = extrema_map['5m']['ll']
        price_range = hh_5m - ll_5m
        
        target_price = ll_5m + (price_range * GOLDEN_RATIO_DERIV)
        current_price = c_5m[-1]
        
        if current_price >= target_price:
            return None

        # 5. Sine Signal Confirmation
        sig_5m, _, _ = analyze_ht_sine(c_5m)
        sig_1m, _, _ = analyze_ht_sine(c_1m)
        
        if sig_5m != "REVERSAL_UP" and sig_1m != "REVERSAL_UP":
            return None

        # FFT Forecast
        forecast = calculate_fft_forecast(c_5m)
        
        # Scoring
        rsi = ta.RSI(c_5m, 14)[-1]
        score = (target_price - current_price) * 1000
        score += (100 - rsi)
        if sig_5m == "REVERSAL_UP": score += 50
        
        return {
            'symbol': symbol,
            'price': current_price,
            'target': target_price,
            'rsi': rsi,
            'score': score,
            'extrema_map': extrema_map, # Pass all extrema data
            'induction_vol': vol_ll,
            'avg_vol': avg_vol,
            'sine_sig_5m': sig_5m,
            'sine_sig_1m': sig_1m,
            'forecast': forecast,
            'phi_support': ll_5m
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(process_symbol, symbols))
        
    return [r for r in results if r is not None]

# --- Main Execution ---

print("Fetching trading pairs...")
trading_pairs = trader.get_usdc_pairs()
print(f"Scanning {len(trading_pairs)} USDC pairs ...")

candidates = check_mtf_dip(trading_pairs, trader.client)

if candidates:
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    best = candidates[0]
    
    print("\n" + "="*100)
    print(f"=== CRITERIA MET: Strict ArgMin Recent (1m/5m) & Majority TFs ArgMin ===")
    print("="*100)
    
    print(f"\n[WINNER] {best['symbol']}")
    print(f"Current Price: {best['price']:.25f}")
    print(f"Algo Score:    {best['score']:.25f}")
    print(f"RSI (5m):      {best['rsi']:.25f}")
    
    print("-" * 100)
    print("EXTREMA ANALYSIS (Per Timeframe - 25 Decimal Precision):")
    print("-" * 100)
    
    # Print logic for all TFs
    tf_order = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h']
    for tf in tf_order:
        ext = best['extrema_map'][tf]
        print(f"{tf:<5} | Recent: {ext['recent_type']:<15} | LL: {ext['ll']:.25f} | HH: {ext['hh']:.25f}")
    
    print("-" * 100)
    print("VOLUME INDUCTION (1m Support Confirmation):")
    print("-" * 100)
    print(f"Vol at Low: {best['induction_vol']:.25f} vs Avg Vol: {best['avg_vol']:.25f}")
    
    print("-" * 100)
    print("TARGET LOGIC:")
    print("-" * 100)
    print(f"Target (Phi Resistance 0.618): {best['target']:.25f}")
    print(f"Stop Loss (5m LL):             {best['phi_support']:.25f}")
    
    print("-" * 100)
    print("SINE SIGNALS:")
    print("-" * 100)
    print(f"5m Sine: {best['sine_sig_5m']} | 1m Sine: {best['sine_sig_1m']}")
    
    print("="*100)
    
    print("\nTop 5 Candidates:")
    for i, c in enumerate(candidates[:5]):
        print(f"{i+1}. {c['symbol']} | Score: {c['score']:.0f} | Tgt: {c['target']:.25f}")

else:
    print("No candidates found matching strict ArgMin recency and Volume Induction criteria.")

sys.exit(0)
