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
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

class Trader:
    def __init__(self, file):
        self.connect(file)

    """ Creates Binance client """
    def connect(self, file):
        lines = [line.rstrip('\n') for line in open(file)]
        key = lines[0]
        secret = lines[1]
        self.client = Client(key, secret)

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

def analyze_fft(price_data, sampling_rate=1.0):
    """
    Perform FFT analysis on price data to identify dominant cycles
    Returns: dominant frequencies and their power
    """
    try:
        # Normalize price data
        normalized = (price_data - np.mean(price_data)) / np.std(price_data)
        
        # Apply window function to reduce spectral leakage
        windowed = normalized * np.hanning(len(normalized))
        
        # Compute FFT
        fft_values = fft(windowed)
        frequencies = fftfreq(len(windowed), d=1.0/sampling_rate)
        
        # Get power spectrum (magnitude squared)
        power = np.abs(fft_values) ** 2
        
        # Only keep positive frequencies
        pos_freq_idx = frequencies > 0
        frequencies = frequencies[pos_freq_idx]
        power = power[pos_freq_idx]
        
        # Find dominant frequencies (top 5)
        dominant_idx = np.argsort(power)[-5:]
        dominant_freqs = frequencies[dominant_idx]
        dominant_power = power[dominant_idx]
        
        return dominant_freqs, dominant_power
    except Exception as e:
        print(f"Error in FFT analysis: {e}")
        return np.array([]), np.array([])

def forecast_price(price_data, dominant_freqs, forecast_periods=10):
    """
    Forecast future prices using dominant frequencies from FFT
    Returns: forecasted prices
    """
    try:
        if len(dominant_freqs) == 0:
            return np.array([])
            
        # Create a model based on dominant frequencies
        t = np.arange(len(price_data))
        forecast_t = np.arange(len(price_data), len(price_data) + forecast_periods)
        
        # Initialize forecast
        forecast = np.zeros(forecast_periods)
        
        # Fit and predict using each dominant frequency
        for freq in dominant_freqs:
            if freq > 0:  # Skip zero frequency
                # Create sine and cosine components
                sin_component = np.sin(2 * np.pi * freq * t)
                cos_component = np.cos(2 * np.pi * freq * t)
                
                # Fit coefficients
                A = np.column_stack([sin_component, cos_component, np.ones_like(t)])
                coeffs, _, _, _ = np.linalg.lstsq(A, price_data, rcond=None)
                
                # Forecast
                sin_forecast = np.sin(2 * np.pi * freq * forecast_t)
                cos_forecast = np.cos(2 * np.pi * freq * forecast_t)
                forecast += coeffs[0] * sin_forecast + coeffs[1] * cos_forecast + coeffs[2]
        
        # Normalize forecast to match the last price
        if len(price_data) > 0 and len(forecast) > 0:
            last_price = price_data[-1]
            first_forecast = forecast[0]
            if first_forecast != 0:
                forecast = forecast * (last_price / first_forecast)
        
        return forecast
    except Exception as e:
        print(f"Error in price forecasting: {e}")
        return np.array([])

def calculate_ht_sine_momentum(close_prices, volume):
    """
    Calculate HT_SINE indicators with volume confirmation and impulse momentum
    Returns: sine, cosine, impulse, volume_confirmed
    """
    try:
        close_array = np.asarray(close_prices, dtype=np.float64)
        volume_array = np.asarray(volume, dtype=np.float64)
        
        # Calculate HT_SINE
        sine, cosine = ta.HT_SINE(close_array)
        
        # Calculate impulse momentum (rate of change of sine)
        impulse = np.zeros_like(sine)
        if len(sine) > 1:
            impulse[1:] = np.diff(sine)
        
        # Calculate volume confirmation
        # 1. Volume SMA (20 period)
        volume_sma = ta.SMA(volume_array, timeperiod=20)
        
        # 2. Current volume relative to average
        volume_ratio = np.zeros_like(volume_array)
        valid_idx = volume_sma > 0
        volume_ratio[valid_idx] = volume_array[valid_idx] / volume_sma[valid_idx]
        
        # 3. Volume confirmation (current volume > 1.2x average)
        volume_confirmed = volume_ratio > 1.2
        
        return sine, cosine, impulse, volume_confirmed
    except Exception as e:
        print(f"Error in HT_SINE calculation: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([])

def check_advanced_dip(symbol, interval, trader, threshold=0.01):
    """
    Advanced dip detection using polynomial trendline, HT_SINE, volume, and FFT
    Returns: detailed analysis results
    """
    klines = trader.get_klines_safe(symbol, interval)
    
    if klines is None:
        return {
            'symbol': symbol,
            'is_dip': False,
            'score': 0,
            'price': None,
            'trendline': None,
            'deviation': None,
            'sine': None,
            'cosine': None,
            'impulse': None,
            'volume_confirmed': False,
            'fft_signal': False,
            'forecast_change': None
        }
        
    close = [float(entry[4]) for entry in klines]
    volume = [float(entry[5]) for entry in klines]
    
    if len(close) < 50:  # Need minimum data points for all indicators
        return {
            'symbol': symbol,
            'is_dip': False,
            'score': 0,
            'price': None,
            'trendline': None,
            'deviation': None,
            'sine': None,
            'cosine': None,
            'impulse': None,
            'volume_confirmed': False,
            'fft_signal': False,
            'forecast_change': None
        }
    
    try:
        # 1. Polynomial trendline analysis
        x = close
        y = np.arange(len(x))
        
        poly_coeffs = np.polyfit(y, x, 1)
        trendline = np.poly1d(poly_coeffs)(y)
        
        current_price = x[-1]
        current_trendline = trendline[-1]
        deviation = (current_price - current_trendline) / current_trendline
        is_below_trendline = deviation < -threshold
        
        # 2. HT_SINE analysis
        sine, cosine, impulse, volume_confirmed = calculate_ht_sine_momentum(close, volume)
        
        # Check for sine crossing below -0.5 (indicative of dip)
        sine_dip = len(sine) > 0 and sine[-1] < -0.5
        
        # Check for negative impulse (downward momentum)
        impulse_negative = len(impulse) > 0 and impulse[-1] < 0
        
        # 3. Volume confirmation
        volume_signal = len(volume_confirmed) > 0 and volume_confirmed[-1]
        
        # 4. FFT analysis
        dominant_freqs, dominant_power = analyze_fft(close)
        forecast = forecast_price(close, dominant_freqs, forecast_periods=5)
        
        # Check if forecast shows upward reversal
        fft_signal = False
        forecast_change = None
        if len(forecast) > 0:
            forecast_change = (forecast[-1] - current_price) / current_price
            fft_signal = forecast_change > 0.01  # At least 1% expected increase
        
        # 5. Calculate overall score
        score = 0
        if is_below_trendline:
            score += 2
        if sine_dip:
            score += 2
        if impulse_negative:
            score += 1
        if volume_signal:
            score += 1
        if fft_signal:
            score += 2
        
        # Determine if it's a dip based on combined signals
        is_dip = (is_below_trendline and sine_dip and 
                  (volume_signal or fft_signal) and score >= 4)
        
        return {
            'symbol': symbol,
            'is_dip': is_dip,
            'score': score,
            'price': current_price,
            'trendline': current_trendline,
            'deviation': deviation,
            'sine': sine[-1] if len(sine) > 0 else None,
            'cosine': cosine[-1] if len(cosine) > 0 else None,
            'impulse': impulse[-1] if len(impulse) > 0 else None,
            'volume_confirmed': volume_signal,
            'fft_signal': fft_signal,
            'forecast_change': forecast_change
        }
        
    except Exception as e:
        print(f"Error processing {symbol} on {interval}: {e}")
        return {
            'symbol': symbol,
            'is_dip': False,
            'score': 0,
            'price': None,
            'trendline': None,
            'deviation': None,
            'sine': None,
            'cosine': None,
            'impulse': None,
            'volume_confirmed': False,
            'fft_signal': False,
            'forecast_change': None
        }

def process_pairs_batch(pairs, interval, trader, max_workers=10):
    """
    Process a batch of pairs concurrently for a specific timeframe
    """
    filtered_pairs = []
    
    # Create partial function with fixed parameters
    check_func = partial(check_advanced_dip, interval=interval, trader=trader)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {executor.submit(check_func, pair): pair for pair in pairs}
        
        for future in concurrent.futures.as_completed(future_to_pair):
            result = future.result()
            
            if result['is_dip']:
                filtered_pairs.append(result)
                print(f"✓ Dip found on {interval}: {result['symbol']} (score: {result['score']}, "
                      f"sine: {result['sine']:.3f}, vol_conf: {result['volume_confirmed']}, "
                      f"fft: {result['fft_signal']}, forecast: {result['forecast_change']:.2%})")
    
    return filtered_pairs

def plot_advanced_mtf_chart(symbol, trader):
    """
    Plot advanced multi-timeframe chart with all indicators
    """
    intervals = ['1d', '2h', '15m', '5m', '3m', '1m']
    fig, axes = plt.subplots(6, 2, figsize=(15, 18))
    
    for idx, interval in enumerate(intervals):
        klines = trader.get_klines_safe(symbol, interval)
        if klines is None:
            continue
            
        close = [float(entry[4]) for entry in klines]
        volume = [float(entry[5]) for entry in klines]
        timestamps = [datetime.fromtimestamp(int(entry[0])/1000) for entry in klines]
        
        # Price chart with trendline
        axes[idx, 0].plot(timestamps, close, label='Price', linewidth=1)
        
        # Add trendline
        x = close
        y = np.arange(len(x))
        poly_coeffs = np.polyfit(y, x, 1)
        trendline = np.poly1d(poly_coeffs)(y)
        axes[idx, 0].plot(timestamps, trendline, 'r--', label='Trendline', linewidth=1)
        
        # Add bands
        axes[idx, 0].plot(timestamps, trendline * 1.01, 'g--', alpha=0.5, label='+1%')
        axes[idx, 0].plot(timestamps, trendline * 0.99, 'b--', alpha=0.5, label='-1%')
        
        axes[idx, 0].set_title(f'{symbol} - {interval} Price')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)
        
        # HT_SINE and volume
        sine, cosine, impulse, volume_confirmed = calculate_ht_sine_momentum(close, volume)
        
        # Create second y-axis for volume
        ax2 = axes[idx, 1].twinx()
        ax2.bar(timestamps, volume, alpha=0.3, color='gray', label='Volume')
        
        # Plot HT_SINE
        if len(sine) > 0:
            axes[idx, 1].plot(timestamps, sine, 'b-', label='Sine', linewidth=1)
            axes[idx, 1].plot(timestamps, cosine, 'r-', label='Cosine', linewidth=1)
            axes[idx, 1].axhline(y=-0.5, color='b', linestyle='--', alpha=0.3)
            axes[idx, 1].axhline(y=0.5, color='b', linestyle='--', alpha=0.3)
            
            # Highlight volume confirmed periods
            if len(volume_confirmed) > 0:
                for i, confirmed in enumerate(volume_confirmed):
                    if confirmed and i < len(timestamps):
                        axes[idx, 1].axvspan(timestamps[i], timestamps[min(i+1, len(timestamps)-1)], 
                                           alpha=0.1, color='green')
        
        axes[idx, 1].set_title(f'{symbol} - {interval} HT_SINE & Volume')
        axes[idx, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        axes[idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Advanced_MTF_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

def detect_imminent_spikes(pairs_data, trader):
    """
    Advanced detection of imminent price spikes using multiple indicators
    """
    spike_candidates = []
    
    for pair_data in pairs_data:
        symbol = pair_data['symbol']
        
        # Get 1-minute data for detailed analysis
        klines = trader.get_klines_safe(symbol, '1m', limit=100)
        if klines is None:
            continue
            
        close = [float(entry[4]) for entry in klines]
        volume = [float(entry[5]) for entry in klines]
        
        if len(close) < 50:
            continue
        
        try:
            # 1. Price compression detection (Bollinger Band Width)
            bb_upper, bb_middle, bb_lower = ta.BBANDS(np.array(close), timeperiod=20, nbdevup=2, nbdevdn=2)
            bb_width = (bb_upper - bb_lower) / bb_middle
            bb_squeeze = bb_width[-1] < np.percentile(bb_width, 10)  # In lowest 10%
            
            # 2. Volume spike detection
            volume_sma = ta.SMA(np.array(volume), timeperiod=20)
            volume_spike = volume[-1] > volume_sma[-1] * 2.0  # Current volume > 2x average
            
            # 3. RSI divergence
            rsi = ta.RSI(np.array(close), timeperiod=14)
            rsi_oversold = rsi[-1] < 30
            
            # 4. Stochastic oscillator
            slowk, slowd = ta.STOCH(np.array(close), np.array([max(entry[2], entry[3]) for entry in klines]), 
                                   np.array([min(entry[2], entry[3]) for entry in klines]), 
                                   fastk_period=14, slowk_period=3, slowd_period=3)
            stoch_oversold = slowk[-1] < 20 and slowd[-1] < 20
            
            # 5. MACD histogram turning positive
            macd, macdsignal, macdhist = ta.MACD(np.array(close), fastperiod=12, slowperiod=26, signalperiod=9)
            macd_turning = len(macdhist) > 1 and macdhist[-2] < 0 and macdhist[-1] > 0
            
            # 6. Calculate spike probability
            spike_score = 0
            if bb_squeeze:
                spike_score += 2
            if volume_spike:
                spike_score += 2
            if rsi_oversold:
                spike_score += 1
            if stoch_oversold:
                spike_score += 1
            if macd_turning:
                spike_score += 2
            
            # Consider it a spike candidate if score is high enough
            if spike_score >= 4:
                spike_candidates.append({
                    'symbol': symbol,
                    'score': spike_score,
                    'bb_squeeze': bb_squeeze,
                    'volume_spike': volume_spike,
                    'rsi_oversold': rsi_oversold,
                    'stoch_oversold': stoch_oversold,
                    'macd_turning': macd_turning,
                    'price': close[-1]
                })
                
        except Exception as e:
            print(f"Error in spike detection for {symbol}: {e}")
            continue
    
    # Sort by spike score (highest first)
    spike_candidates.sort(key=lambda x: x['score'], reverse=True)
    return spike_candidates

def main():
    print("=== Advanced Multi-Timeframe Dip & Spike Finder ===")
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
    
    # Spike Detection
    print("\n=== Spike Detection ===")
    spike_candidates = detect_imminent_spikes(m3_filtered, trader)
    print(f"Found {len(spike_candidates)} spike candidates")
    
    # Combine MTF dips with spike candidates
    final_candidates = []
    
    # Add all spike candidates
    for candidate in spike_candidates:
        # Find corresponding MTF data
        mtf_data = next((p for p in m3_filtered if p['symbol'] == candidate['symbol']), None)
        if mtf_data:
            combined_data = {**mtf_data, **candidate}
            final_candidates.append(combined_data)
    
    # Sort by combined score (MTF score + spike score)
    final_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n=== FINAL CANDIDATES ===")
    print(f"Total candidates: {len(final_candidates)}")
    
    for i, candidate in enumerate(final_candidates[:5], 1):  # Show top 5
        print(f"\n{i}. {candidate['symbol']}")
        print(f"   MTF Score: {candidate['score']}")
        print(f"   Current Price: {candidate['price']:.6f}")
        print(f"   Deviation: {candidate['deviation']:.2%}")
        print(f"   Sine: {candidate['sine']:.3f}")
        print(f"   Volume Confirmed: {candidate['volume_confirmed']}")
        print(f"   FFT Signal: {candidate['fft_signal']}")
        print(f"   Forecast Change: {candidate['forecast_change']:.2%}")
        print(f"   BB Squeeze: {candidate.get('bb_squeeze', False)}")
        print(f"   Volume Spike: {candidate.get('volume_spike', False)}")
        print(f"   RSI Oversold: {candidate.get('rsi_oversold', False)}")
        print(f"   MACD Turning: {candidate.get('macd_turning', False)}")
    
    if final_candidates:
        # Select the top candidate
        best_candidate = final_candidates[0]
        print(f"\n=== BEST CANDIDATE ===")
        print(f"Symbol: {best_candidate['symbol']}")
        print(f"Combined Score: {best_candidate['score']}")
        print(f"Current Price: {best_candidate['price']:.6f}")
        
        # Generate advanced MTF chart for the best candidate
        print(f"\nGenerating advanced MTF chart for {best_candidate['symbol']}...")
        plot_advanced_mtf_chart(best_candidate['symbol'], trader)
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(final_candidates)
        df.to_csv(f'Advanced_Candidates_{timestamp}.csv', index=False)
        print(f"Results saved to Advanced_Candidates_{timestamp}.csv")
    else:
        print("No candidates found with sufficient signals.")
    
    print(f"\nScan completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
    sys.exit(0)