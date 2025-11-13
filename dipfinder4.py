from binance.client import Client
import matplotlib.pyplot as plt
import numpy as np
import talib as ta
import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures
from functools import partial
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
import gc
from collections import defaultdict
warnings.filterwarnings('ignore')

class ContinuousMTFScanner:
    def __init__(self, api_file):
        self.connect(api_file)
        self.data_cache = {}  # Cache for klines data
        self.last_scan_time = {}
        self.scan_interval = 5  # seconds
        self.max_cache_age = 300  # 5 minutes
        self.results_history = []
        self.best_candidate = None
        self.running = True
        
    def connect(self, file):
        lines = [line.rstrip('\n') for line in open(file)]
        key = lines[0]
        secret = lines[1]
        self.client = Client(key, secret)
        
    def get_usdc_pairs(self):
        try:
            exchange_info = self.client.get_exchange_info()
            trading_pairs = [symbol['symbol'] for symbol in exchange_info['symbols'] 
                           if symbol['quoteAsset'] == 'USDC' and symbol['status'] == 'TRADING']
            return trading_pairs
        except Exception as e:
            print(f"Error fetching pairs: {e}")
            return []
    
    def get_klines_cached(self, symbol, interval, limit=500):
        """
        Get klines data with caching to reduce API calls
        """
        current_time = time.time()
        cache_key = f"{symbol}_{interval}"
        
        # Check if we have fresh cached data
        if cache_key in self.data_cache:
            cached_data, cache_time = self.data_cache[cache_key]
            age = current_time - cache_time
            
            # For 1m interval, refresh every 30 seconds
            # For other intervals, refresh every 2 minutes
            refresh_interval = 30 if interval == '1m' else 120
            
            if age < refresh_interval:
                return cached_data
        
        # Fetch fresh data
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            self.data_cache[cache_key] = (klines, current_time)
            return klines
        except Exception as e:
            return None
    
    def cleanup_cache(self):
        """
        Remove old cached data to prevent memory leaks
        """
        current_time = time.time()
        keys_to_remove = []
        
        for key, (_, cache_time) in self.data_cache.items():
            if current_time - cache_time > self.max_cache_age:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.data_cache[key]
        
        # Force garbage collection
        gc.collect()
    
    def analyze_fft(self, price_data, sampling_rate=1.0):
        """FFT analysis with error handling"""
        try:
            if len(price_data) < 20:
                return np.array([]), np.array([])
            
            normalized = (price_data - np.mean(price_data)) / (np.std(price_data) + 1e-8)
            windowed = normalized * np.hanning(len(normalized))
            
            fft_values = fft(windowed)
            frequencies = fftfreq(len(windowed), d=1.0/sampling_rate)
            
            pos_freq_idx = frequencies > 0
            frequencies = frequencies[pos_freq_idx]
            power = np.abs(fft_values[pos_freq_idx]) ** 2
            
            if len(power) > 5:
                dominant_idx = np.argsort(power)[-5:]
                return frequencies[dominant_idx], power[dominant_idx]
            
            return np.array([]), np.array([])
        except:
            return np.array([]), np.array([])
    
    def calculate_ht_sine_momentum(self, close_prices, volume):
        """HT_SINE with volume confirmation"""
        try:
            close_array = np.asarray(close_prices, dtype=np.float64)
            volume_array = np.asarray(volume, dtype=np.float64)
            
            sine, cosine = ta.HT_SINE(close_array)
            
            # Calculate impulse momentum
            impulse = np.zeros_like(sine)
            if len(sine) > 1:
                impulse[1:] = np.diff(sine)
            
            # Volume confirmation
            volume_sma = ta.SMA(volume_array, timeperiod=20)
            volume_ratio = np.zeros_like(volume_array)
            valid_idx = volume_sma > 0
            volume_ratio[valid_idx] = volume_array[valid_idx] / (volume_sma[valid_idx] + 1e-8)
            volume_confirmed = volume_ratio > 1.5
            
            return sine, cosine, impulse, volume_confirmed
        except:
            return np.array([]), np.array([]), np.array([]), np.array([])
    
    def detect_price_patterns(self, klines):
        """
        Detect candlestick patterns that indicate reversal
        """
        try:
            close = np.array([float(entry[4]) for entry in klines])
            high = np.array([float(entry[2]) for entry in klines])
            low = np.array([float(entry[3]) for entry in klines])
            open_price = np.array([float(entry[1]) for entry in klines])
            
            # Hammer pattern
            hammer = ta.CDLHAMMER(open_price, high, low, close)
            
            # Engulfing patterns
            engulfing = ta.CDLENGULFING(open_price, high, low, close)
            
            # Doji patterns
            doji = ta.CDLDOJI(open_price, high, low, close)
            
            # Check for recent bullish patterns (last 3 candles)
            recent_hammer = np.any(hammer[-3:] > 0)
            recent_bullish_engulfing = np.any(engulfing[-3:] > 0)
            recent_doji = np.any(doji[-3:] > 0)
            
            return {
                'hammer': bool(recent_hammer),
                'bullish_engulfing': bool(recent_bullish_engulfing),
                'doji': bool(recent_doji),
                'has_bullish_pattern': bool(recent_hammer or recent_bullish_engulfing or recent_doji)
            }
        except:
            return {
                'hammer': False,
                'bullish_engulfing': False,
                'doji': False,
                'has_bullish_pattern': False
            }
    
    def calculate_volume_acceleration(self, volume_data):
        """
        Calculate the acceleration of volume (rate of change of volume increase)
        """
        try:
            volume_array = np.array(volume_data)
            
            # Calculate volume SMA
            volume_sma = ta.SMA(volume_array, timeperiod=10)
            
            # Calculate volume rate of change
            volume_roc = ta.ROC(volume_array, timeperiod=3)
            
            # Calculate volume acceleration (second derivative)
            if len(volume_roc) > 1:
                volume_acceleration = np.diff(volume_roc)
                recent_acceleration = volume_acceleration[-3:]
                avg_acceleration = np.mean(recent_acceleration)
                return bool(avg_acceleration > 0)  # Positive acceleration
            return False
        except:
            return False
    
    def calculate_volatility_contraction(self, close_prices):
        """
        Detect volatility contraction patterns that often precede breakouts
        """
        try:
            close_array = np.array(close_prices)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = ta.BBANDS(close_array, timeperiod=20, nbdevup=2, nbdevdn=2)
            
            # Calculate BB Width
            bb_width = (bb_upper - bb_lower) / bb_middle
            
            # Check if BB Width is in the lowest 10% of recent values
            bb_width_percentile = np.percentile(bb_width, 10)
            is_bb_squeeze = bb_width[-1] < bb_width_percentile
            
            # ATR (Average True Range) to measure volatility
            atr = ta.ATR(close_array, close_array, close_array, timeperiod=14)
            
            # Check if ATR is decreasing (volatility contracting)
            atr_decreasing = atr[-1] < atr[-5] if len(atr) > 5 else False
            
            return {
                'bb_squeeze': bool(is_bb_squeeze),
                'atr_decreasing': bool(atr_decreasing),
                'volatility_contraction': bool(is_bb_squeeze and atr_decreasing)
            }
        except:
            return {
                'bb_squeeze': False,
                'atr_decreasing': False,
                'volatility_contraction': False
            }
    
    def calculate_spike_probability(self, symbol):
        """
        Calculate a comprehensive spike probability score based on multiple factors
        """
        try:
            # Get 1-minute data for detailed analysis
            klines_1m = self.get_klines_cached(symbol, '1m', limit=100)
            if klines_1m is None:
                return 0
            
            close = [float(entry[4]) for entry in klines_1m]
            volume = [float(entry[5]) for entry in klines_1m]
            
            if len(close) < 50:
                return 0
            
            close_array = np.array(close)
            volume_array = np.array(volume)
            
            # 1. Volume acceleration (weight: 20%)
            volume_accelerating = self.calculate_volume_acceleration(volume)
            volume_score = 20 if volume_accelerating else 0
            
            # 2. Volatility contraction (weight: 20%)
            volatility_data = self.calculate_volatility_contraction(close)
            volatility_score = 20 if volatility_data['volatility_contraction'] else 0
            
            # 3. Price patterns (weight: 15%)
            pattern_data = self.detect_price_patterns(klines_1m)
            pattern_score = 15 if pattern_data['has_bullish_pattern'] else 0
            
            # 4. RSI oversold (weight: 15%)
            rsi = ta.RSI(close_array, timeperiod=14)
            rsi_oversold = rsi[-1] < 30 if len(rsi) > 0 else False
            rsi_score = 15 if rsi_oversold else 0
            
            # 5. MACD turning positive (weight: 15%)
            macd, macdsignal, macdhist = ta.MACD(close_array, fastperiod=12, slowperiod=26, signalperiod=9)
            macd_turning = len(macdhist) > 1 and macdhist[-2] < 0 and macdhist[-1] > 0
            macd_score = 15 if macd_turning else 0
            
            # 6. Price position relative to recent range (weight: 15%)
            recent_high = np.max(close_array[-20:])
            recent_low = np.min(close_array[-20:])
            price_position = (close_array[-1] - recent_low) / (recent_high - recent_low)
            near_bottom = price_position < 0.2  # In the bottom 20% of recent range
            position_score = 15 if near_bottom else 0
            
            # Calculate total spike probability score
            total_score = volume_score + volatility_score + pattern_score + rsi_score + macd_score + position_score
            
            return total_score
            
        except Exception as e:
            return 0
    
    def check_advanced_dip(self, symbol, intervals=['1d', '2h', '15m', '5m', '3m']):
        """
        Check dip across multiple timeframes
        """
        results = {}
        total_score = 0
        valid_timeframes = 0
        
        for interval in intervals:
            klines = self.get_klines_cached(symbol, interval)
            if klines is None:
                continue
                
            close = [float(entry[4]) for entry in klines]
            volume = [float(entry[5]) for entry in klines]
            
            if len(close) < 30:
                continue
            
            try:
                # Polynomial trendline
                x = close
                y = np.arange(len(x))
                poly_coeffs = np.polyfit(y, x, 1)
                trendline = np.poly1d(poly_coeffs)(y)
                
                current_price = x[-1]
                current_trendline = trendline[-1]
                deviation = (current_price - current_trendline) / (current_trendline + 1e-8)
                is_below_trendline = deviation < -0.01
                
                # HT_SINE analysis
                sine, cosine, impulse, volume_confirmed = self.calculate_ht_sine_momentum(close, volume)
                sine_dip = len(sine) > 0 and sine[-1] < -0.5
                impulse_negative = len(impulse) > 0 and impulse[-1] < -0.01
                volume_signal = len(volume_confirmed) > 0 and volume_confirmed[-1]
                
                # FFT analysis
                dominant_freqs, _ = self.analyze_fft(close)
                fft_signal = len(dominant_freqs) > 0
                
                # Scoring
                timeframe_score = 0
                if is_below_trendline:
                    timeframe_score += 2
                if sine_dip:
                    timeframe_score += 2
                if impulse_negative:
                    timeframe_score += 1
                if volume_signal:
                    timeframe_score += 1
                if fft_signal:
                    timeframe_score += 1
                
                results[interval] = {
                    'price': current_price,
                    'deviation': deviation,
                    'sine': sine[-1] if len(sine) > 0 else None,
                    'impulse': impulse[-1] if len(impulse) > 0 else None,
                    'volume_confirmed': volume_signal,
                    'score': timeframe_score
                }
                
                total_score += timeframe_score
                valid_timeframes += 1
                
            except Exception:
                continue
        
        # Calculate overall score
        avg_score = total_score / valid_timeframes if valid_timeframes > 0 else 0
        
        # Determine if it's a valid MTF dip
        is_mtf_dip = (avg_score >= 4 and valid_timeframes >= 3 and 
                      all(results.get(tf, {}).get('score', 0) >= 2 for tf in ['1d', '2h', '15m']))
        
        return {
            'symbol': symbol,
            'is_mtf_dip': is_mtf_dip,
            'total_score': total_score,
            'avg_score': avg_score,
            'valid_timeframes': valid_timeframes,
            'timeframes': results,
            'timestamp': datetime.now()
        }
    
    def scan_pairs_batch(self, pairs, max_workers=10):
        """
        Scan a batch of pairs concurrently
        """
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {executor.submit(self.check_advanced_dip, pair): pair for pair in pairs}
            
            for future in concurrent.futures.as_completed(future_to_pair):
                try:
                    result = future.result(timeout=10)  # 10 second timeout per pair
                    if result['is_mtf_dip']:
                        # Calculate spike probability
                        spike_probability = self.calculate_spike_probability(result['symbol'])
                        result['spike_probability'] = spike_probability
                        result['combined_score'] = result['total_score'] + spike_probability
                        
                        results.append(result)
                        
                except Exception:
                    continue
        
        return results
    
    def update_display(self, results):
        """
        Update the display with latest results
        """
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print(f"CONTINUOUS MTF DIP SCANNER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        if not results:
            print("No MTF dips found in current scan")
            print("\nMonitoring... (Press Ctrl+C to stop)")
            return
        
        # Sort by spike probability (highest first)
        results.sort(key=lambda x: x['spike_probability'], reverse=True)
        
        # Get the best candidate
        best_candidate = results[0]
        
        print(f"\n🚀 BEST CANDIDATE FOR IMMEDIATE SPIKE: {best_candidate['symbol']} 🚀")
        print("-" * 80)
        print(f"Spike Probability Score: {best_candidate['spike_probability']}/100")
        print(f"MTF Dip Score: {best_candidate['total_score']}")
        print(f"Combined Score: {best_candidate['combined_score']}")
        
        # Show detailed analysis for best candidate
        print("\nDetailed Analysis:")
        
        # Get 1-minute data for detailed analysis
        klines_1m = self.get_klines_cached(best_candidate['symbol'], '1m', limit=100)
        if klines_1m:
            close = [float(entry[4]) for entry in klines_1m]
            volume = [float(entry[5]) for entry in klines_1m]
            
            # Volume acceleration
            volume_accelerating = self.calculate_volume_acceleration(volume)
            print(f"Volume Acceleration: {'✓' if volume_accelerating else '✗'}")
            
            # Volatility contraction
            volatility_data = self.calculate_volatility_contraction(close)
            print(f"Volatility Contraction: {'✓' if volatility_data['volatility_contraction'] else '✗'}")
            print(f"  - BB Squeeze: {'✓' if volatility_data['bb_squeeze'] else '✗'}")
            print(f"  - ATR Decreasing: {'✓' if volatility_data['atr_decreasing'] else '✗'}")
            
            # Price patterns
            pattern_data = self.detect_price_patterns(klines_1m)
            print(f"Bullish Candlestick Pattern: {'✓' if pattern_data['has_bullish_pattern'] else '✗'}")
            if pattern_data['has_bullish_pattern']:
                print(f"  - Hammer: {'✓' if pattern_data['hammer'] else '✗'}")
                print(f"  - Bullish Engulfing: {'✓' if pattern_data['bullish_engulfing'] else '✗'}")
                print(f"  - Doji: {'✓' if pattern_data['doji'] else '✗'}")
            
            # RSI
            close_array = np.array(close)
            rsi = ta.RSI(close_array, timeperiod=14)
            rsi_oversold = rsi[-1] < 30 if len(rsi) > 0 else False
            print(f"RSI Oversold (<30): {'✓' if rsi_oversold else '✗'}")
            if len(rsi) > 0:
                print(f"  - Current RSI: {rsi[-1]:.2f}")
            
            # MACD
            macd, macdsignal, macdhist = ta.MACD(close_array, fastperiod=12, slowperiod=26, signalperiod=9)
            macd_turning = len(macdhist) > 1 and macdhist[-2] < 0 and macdhist[-1] > 0
            print(f"MACD Turning Positive: {'✓' if macd_turning else '✗'}")
            
            # Price position
            recent_high = np.max(close_array[-20:])
            recent_low = np.min(close_array[-20:])
            price_position = (close_array[-1] - recent_low) / (recent_high - recent_low)
            near_bottom = price_position < 0.2
            print(f"Price Near Bottom of Range: {'✓' if near_bottom else '✗'}")
            print(f"  - Position in Range: {price_position:.1%}")
        
        # Show MTF analysis
        print("\nMTF Analysis:")
        for tf, data in best_candidate['timeframes'].items():
            if data['score'] > 0:
                print(f"  {tf}: score={data['score']}, "
                      f"dev={data['deviation']:.2%}, "
                      f"sine={data['sine']:.2f}" if data['sine'] else "")
        
        # Show other candidates if they exist
        if len(results) > 1:
            print(f"\nOther MTF Dips Found ({len(results)-1} more):")
            for i, result in enumerate(results[1:6], 2):  # Show top 5 more
                print(f"{i}. {result['symbol']} - Spike Probability: {result['spike_probability']}/100")
        
        # Update best candidate
        self.best_candidate = best_candidate
        
        print("\n" + "=" * 80)
        print("Monitoring... (Press Ctrl+C to stop)")
    
    def run_continuous_scan(self):
        """
        Main continuous scanning loop
        """
        print("Starting continuous MTF dip scanner...")
        print("Initializing...")
        
        # Get trading pairs
        trading_pairs = self.get_usdc_pairs()
        if not trading_pairs:
            print("No trading pairs found. Exiting.")
            return
        
        print(f"Monitoring {len(trading_pairs)} USDC pairs")
        time.sleep(2)  # Brief pause before starting
        
        try:
            while self.running:
                start_time = time.time()
                
                # Scan pairs
                results = self.scan_pairs_batch(trading_pairs, max_workers=15)
                
                # Update display
                self.update_display(results)
                
                # Cleanup cache
                self.cleanup_cache()
                
                # Calculate sleep time to maintain 5-second intervals
                elapsed = time.time() - start_time
                sleep_time = max(0, self.scan_interval - elapsed)
                
                # Sleep until next scan
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n\nStopping scanner...")
            self.running = False
            print("Scanner stopped.")
        except Exception as e:
            print(f"\nError in scanning loop: {e}")
            self.running = False

def main():
    # Initialize scanner
    api_file = 'api.txt'
    scanner = ContinuousMTFScanner(api_file)
    
    # Start continuous scanning
    scanner.run_continuous_scan()
    
    sys.exit(0)

if __name__ == "__main__":
    main()