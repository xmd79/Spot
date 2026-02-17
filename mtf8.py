from binance.client import Client
import numpy as np
import talib as ta
import sys
import concurrent.futures

# --- Configuration ---
RSI_LENGTH = 14
VOLUME_SMA_LENGTH = 20  # For volume average calculation
VOLUME_MULTIPLIER = 1.5 # Threshold for "Spike" (e.g. 1.5x average volume)
FFT_FORECAST_BARS = 10  # Project next 30 mins (10 * 3m)

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
            trading_pairs = [symbol['symbol'] for symbol in exchange_info['symbols'] 
                             if symbol['quoteAsset'] == 'USDC' and symbol['status'] == 'TRADING']
            return trading_pairs
        except Exception as e:
            print(f"Error fetching pairs: {e}")
            return []

filename = 'credentials.txt'
trader = Trader(filename)

def get_klines_data(client, symbol, interval, limit=100):
    try:
        # Increased limit slightly for better FFT resolution
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines or len(klines) < 50: 
            return None
        
        closes = np.array([float(entry[4]) for entry in klines], dtype=np.float64)
        highs = np.array([float(entry[2]) for entry in klines], dtype=np.float64)
        lows = np.array([float(entry[3]) for entry in klines], dtype=np.float64)
        volumes = np.array([float(entry[5]) for entry in klines], dtype=np.float64)
        raw_closes = [float(entry[4]) for entry in klines]
        
        return closes, highs, lows, volumes, raw_closes
    except Exception:
        return None

def check_regression_dip(symbol, interval, client):
    """ Checks if price is below the lower regression band (0.99 factor) """
    data = get_klines_data(client, symbol, interval, limit=100)
    if data is None:
        return False
    
    close = data[0]
    
    if np.std(close) == 0:
        return False

    # Regression Logic
    x = close
    y = range(len(x))
    
    try:
        coeffs = np.polyfit(y, x, 1)
        best_fit_line1 = np.polyval(coeffs, y)
        best_fit_line3 = best_fit_line1 * 0.99  # Lower band
        
        if x[-1] < best_fit_line3[-1]:
            return True
    except:
        pass
        
    return False

def calculate_fft_forecast(closes):
    """
    Fast FFT Forecast Algo:
    1. Detrend data (remove linear trend).
    2. FFT to find frequencies.
    3. Filter noise (keep top 5 dominant cycles).
    4. Reconstruct and project forward.
    """
    n = len(closes)
    
    # 1. Detrend
    t = np.arange(n)
    coeffs = np.polyfit(t, closes, 1)
    trend = np.polyval(coeffs, t)
    detrended = closes - trend
    
    # 2. FFT
    fft_vals = np.fft.fft(detrended)
    freqs = np.fft.fftfreq(n)
    
    # 3. Filter Noise (Keep top dominant frequencies)
    # Sort by amplitude (absolute value)
    indices = np.argsort(np.abs(fft_vals))[::-1]
    
    # Zero out small frequencies (noise cancellation)
    filtered_fft = np.zeros_like(fft_vals)
    keep_n = 5 # Keep top 5 cycles for 'fast' signal
    for i in indices[:keep_n]:
        filtered_fft[i] = fft_vals[i]
        
    # 4. Inverse FFT to get cleaned cycle
    cleaned_cycle = np.fft.ifft(filtered_fft).real
    
    # Forecast next steps
    forecast = []
    last_t = n - 1
    
    # We manually project the dominant frequencies forward
    # Recalculate the future values using the filtered spectrum
    future_t = np.arange(n, n + FFT_FORECAST_BARS)
    forecast_signal = np.zeros(FFT_FORECAST_BARS)
    
    for i in indices[:keep_n]:
        amp = np.abs(fft_vals[i]) / n
        phase = np.angle(fft_vals[i])
        freq = freqs[i]
        # Forecast component: amp * cos(2*pi*freq*t + phase)
        forecast_signal += amp * np.cos(2 * np.pi * freq * future_t + phase)
    
    # Re-add the trend
    future_trend = np.polyval(coeffs, future_t)
    forecast_prices = forecast_signal + future_trend
    
    return forecast_prices

def analyze_final_candidate(symbol, client):
    """
    Final analysis: RSI, Volume Spike, and FFT Target Calculation.
    """
    data = get_klines_data(client, symbol, '3m', limit=100)
    if data is None:
        return None

    closes, highs, lows, volumes, raw_closes = data

    if np.std(closes) == 0:
        return None

    # 1. RSI Calculation
    try:
        rsi_array = ta.RSI(closes, timeperiod=RSI_LENGTH)
        if np.isnan(rsi_array[-1]):
            return None
        current_rsi = rsi_array[-1]
    except:
        return None

    # 2. Volume Spike Check
    vol_sma = ta.SMA(volumes, timeperiod=VOLUME_SMA_LENGTH)
    current_vol = volumes[-1]
    avg_vol = vol_sma[-1]
    
    # If volume is 0 or NaN, skip
    if np.isnan(avg_vol) or avg_vol == 0:
        return None
        
    vol_ratio = current_vol / avg_vol
    
    # 3. FFT Forecast
    try:
        forecast_prices = calculate_fft_forecast(closes)
        # Target is the highest price in the forecast window
        target_price = np.max(forecast_prices)
        # Confidence: Check if forecast is generally upward
        trend_direction = "UP" if forecast_prices[-1] > closes[-1] else "DOWN/NEUTRAL"
    except:
        return None

    # Algo Scoring: 
    # We want Low RSI (oversold) and High Volume (reversal pressure)
    # Score = (100 - RSI) * VolumeRatio. Higher is better.
    score = (100 - current_rsi) * vol_ratio

    return {
        'symbol': symbol,
        'rsi': current_rsi,
        'score': score,
        'last_close': closes[-1],
        'volume_ratio': vol_ratio,
        'target_price': target_price,
        'forecast_trend': trend_direction,
        'forecast_data': forecast_prices,
        'last_10_closes': raw_closes[-10:]
    }

def run_stage_filter(symbols, interval, client):
    found_pairs = []
    # print(f"Scanning {len(symbols)} pairs on {interval}...") # Reduced verbosity for speed
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
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
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_symbol = {executor.submit(analyze_final_candidate, symbol, client): symbol for symbol in symbols}
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result:
                    # Filter: Only keep if Volume Spike exists (Ratio > 1.0 minimum) and RSI < 40
                    if result['volume_ratio'] > 1.0 and result['rsi'] < 40:
                        candidates.append(result)
            except Exception:
                pass
    
    # Sort by Highest Score (Best Risk/Reward)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates

# --- Main Execution ---

print("Fetching trading pairs...")
trading_pairs = trader.get_usdc_pairs()

# Stage 1: 2h Filter
filtered_pairs_2h = run_stage_filter(trading_pairs, '2h', trader.client)
if not filtered_pairs_2h:
    print("No dips on 2h.")
    sys.exit(0)

# Stage 2: 15m Filter
filtered_pairs_15m = run_stage_filter(filtered_pairs_2h, '15m', trader.client)
if not filtered_pairs_15m:
    print("No dips on 15m.")
    sys.exit(0)

# Stage 3: 5m Filter
filtered_pairs_5m = run_stage_filter(filtered_pairs_15m, '5m', trader.client)
if not filtered_pairs_5m:
    print("No dips on 5m.")
    sys.exit(0)

# Stage 4: 3m Analysis (RSI + FFT + Volume)
final_candidates = run_final_analysis(filtered_pairs_5m, trader.client)

# Results
if final_candidates:
    print("\n" + "="*70)
    print(f"=== BEST MTF DIP CANDIDATE (FFT Forecast & Volume Confirmed) ===")
    print("="*70)
    
    best = final_candidates[0]
    
    print(f"\n[WINNER] {best['symbol']}")
    print(f"Current Price: {best['last_close']:.5f}")
    print(f"RSI (3m): {best['rsi']:.2f}")
    print(f"Vol Spike: {best['volume_ratio']:.2f}x Avg")
    print(f"Algo Score: {best['score']:.2f}")
    
    print("-" * 70)
    print("TRADE SETUP (Forecast FFT):")
    print("-" * 70)
    
    # Entry Logic: Current price or slight dip
    entry_price = best['last_close']
    target_price = best['target_price']
    
    # Stop Loss: Below recent low (safety)
    # We can approximate a stop for display purposes based on last 10 candles low
    stop_price = min(best['last_10_closes']) * 0.999 
    
    print(f"Best Entry Price: {entry_price:.5f} (Market or Limit)")
    print(f"Target Exit Price (FFT): {target_price:.5f}")
    print(f"Forecast Direction: {best['forecast_trend']}")
    print(f"Suggested Stop Loss: {stop_price:.5f}")
    
    # Verify FFT Forecast Prints
    print("-" * 70)
    print("FFT PRICE PROJECTION (Next 30 mins):")
    print("-" * 70)
    for i, f_price in enumerate(best['forecast_data']):
        # Assuming 3m interval, steps are 3 mins
        minutes = (i+1) * 3
        print(f" +{minutes}m: {f_price:.5f}")
        
    print("="*70)
    
    print("\nTop 5 Candidates (Score Ranked):")
    for i, c in enumerate(final_candidates[:5]):
        print(f"{i+1}. {c['symbol']} | RSI: {c['rsi']:.2f} | Vol: {c['volume_ratio']:.2f}x | Target: {c['target_price']:.5f}")
else:
    print("No candidates found matching MTF dip + Volume Spike criteria.")

sys.exit(0)