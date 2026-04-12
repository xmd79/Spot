from binance.client import Client
import numpy as np
import talib as ta
import time
import sys
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from threading import Lock
from collections import deque


class RateLimiter:
    """Token bucket rate limiter to avoid API bans"""
    def __init__(self, requests_per_second: float = 15, burst: int = 25):
        self.rate = requests_per_second
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self.lock = Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_update = now
                
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            time.sleep(0.05)


class Trader:
    def __init__(self, credentials_file: str):
        self.connect(credentials_file)
        self.rate_limiter = RateLimiter(requests_per_second=15, burst=30)

    def connect(self, file: str):
        with open(file) as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) < 2:
            raise ValueError("credentials.txt must contain API key on line 1 and secret on line 2")
        self.client = Client(lines[0], lines[1])

    def get_usdc_pairs(self) -> List[str]:
        exchange_info = self.client.get_exchange_info()
        pairs = [
            symbol['symbol']
            for symbol in exchange_info['symbols']
            if symbol['quoteAsset'] == 'USDC' and symbol['status'] == 'TRADING'
        ]
        print(f"Found {len(pairs)} USDC trading pairs")
        return pairs

    def get_klines(self, symbol: str, interval: str, limit: int = 500, return_volume: bool = False):
        self.rate_limiter.acquire()
        for attempt in range(3):
            try:
                klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
                close = [float(k[4]) for k in klines]
                if return_volume:
                    volume = [float(k[5]) for k in klines]
                    return close, volume
                return close
            except Exception as e:
                if 'rate limit' in str(e).lower():
                    time.sleep(2 ** attempt * 2)
                else:
                    time.sleep(0.5)
        return [] if not return_volume else ([], [])


# ==========================================
# INDICATORS & FORECASTING ENGINE
# ==========================================

def linear_regression_dip(close: List[float], deviation: float = 0.01) -> bool:
    if len(close) < 20:
        return False
    x = np.arange(len(close))
    slope, intercept = np.polyfit(x, close, 1)
    trend = slope * x + intercept
    lower_band = trend * (1 - deviation)
    return close[-1] < lower_band[-1]


def has_pre_spike_volume_infusion(volumes: List[float], window: int = 50, threshold: float = 2.0) -> bool:
    if len(volumes) < window + 10:
        return False
    avg_volume = np.mean(volumes[-window-1:-1])
    last_volume = volumes[-1]
    recent_5_avg = np.mean(volumes[-6:-1]) if len(volumes) > 6 else last_volume
    build_up = recent_5_avg > avg_volume * 1.3
    spike = last_volume > avg_volume * threshold
    return spike and build_up


def fft_ht_sine_forecast(close: List[float]) -> Optional[Dict]:
    """
    FFT Forecast using HT_SINE between most negative and most positive frequencies.
    Maps argmin/argmax of the sinusoidal wave to actual price Hi-Los to find cycle target.
    """
    if len(close) < 500:
        return None
    
    data = np.array(close[-500:], dtype='float64')
    
    try:
        # Get Hilbert Transform Sine Wave
        sine, leadsine = ta.HT_SINE(data)
    except Exception:
        return None
        
    # Remove NaNs from TA-Lib output
    valid_idx = ~np.isnan(sine)
    if np.sum(valid_idx) < 100:
        return None
        
    sine_clean = sine[valid_idx]
    leadsine_clean = leadsine[valid_idx]
    close_clean = data[valid_idx]
    
    # 1. Find Argmin and Argmax of the Sine Wave (Rejection/Reversal points)
    idx_min = np.argmin(sine_clean)
    idx_max = np.argmax(sine_clean)
    
    cycle_low = close_clean[idx_min]
    cycle_high = close_clean[idx_max]
    amplitude = abs(cycle_high - cycle_low)
    
    # 2. FFT to find dominant frequency (most significant cycle)
    fft_vals = np.fft.rfft(sine_clean)
    if len(fft_vals) > 1:
        # Ignore DC component (index 0)
        fft_magnitudes = np.abs(fft_vals[1:])
        dominant_idx = np.argmax(fft_magnitudes) + 1 
        freqs = np.fft.rfftfreq(len(sine_clean))
        dominant_freq = freqs[dominant_idx]
        dominant_period = int(1.0 / dominant_freq) if dominant_freq > 0 else len(sine_clean)
    else:
        dominant_period = len(sine_clean)
        
    current_price = close_clean[-1]
    
    # Phase check: Standard HT_Sine leading crossover
    is_rising = sine_clean[-1] > leadsine_clean[-1]
    
    # 3. Forecast Target based on Hi-Lo range
    if is_rising:
        # Cycle turning up: Target is current price + full amplitude of the dominant cycle
        forecast_target = current_price + amplitude
    else:
        # Still dipping: Project half amplitude recovery as initial target
        forecast_target = current_price + (amplitude * 0.5)
        
    return {
        'cycle_low': cycle_low,
        'cycle_high': cycle_high,
        'amplitude': amplitude,
        'dominant_period': dominant_period,
        'is_rising': is_rising,
        'forecast_target': forecast_target,
        'current_price': current_price
    }


# ==========================================
# CONCURRENT FILTER FUNCTIONS
# ==========================================

def check_tf_dip(trader: Trader, symbol: str, interval: str) -> Tuple[str, bool]:
    close = trader.get_klines(symbol, interval, limit=300)
    is_dip = linear_regression_dip(close, 0.01)
    return (symbol, is_dip)


def check_1m_momentum_volume(trader: Trader, symbol: str) -> Tuple[str, float, float, bool]:
    """Returns (symbol, cmo, volume_ratio, is_strong_pass)"""
    close, volumes = trader.get_klines(symbol, '1m', limit=200, return_volume=True)
    if len(close) < 50 or len(volumes) < 50:
        return (symbol, 0.0, 0.0, False)
        
    cmo = ta.CMO(np.asarray(close), timeperiod=14)
    cmo_val = cmo[-1] if not np.isnan(cmo[-1]) else 0.0
    
    avg_vol = np.mean(volumes[-51:-1]) if len(volumes) > 51 else np.mean(volumes[:-1])
    vratio = volumes[-1] / avg_vol if avg_vol > 0 else 0.0
    
    is_strong = (cmo_val < -50) and has_pre_spike_volume_infusion(volumes)
    return (symbol, cmo_val, vratio, is_strong)


class ProgressTracker:
    def __init__(self, total: int, label: str):
        self.total = total
        self.label = label
        self.completed = 0
        self.passed = 0
        self.lock = Lock()
        self.start_time = time.time()

    def update(self, passed: bool = False):
        with self.lock:
            self.completed += 1
            if passed: self.passed += 1

    def get_stats(self) -> str:
        with self.lock:
            elapsed = time.time() - self.start_time
            rate = self.completed / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.completed) / rate if rate > 0 else 0
            return (f"\r{self.label}: {self.completed}/{self.total} | "
                   f"✓{self.passed} | {rate:.1f}/s | ETA: {remaining:.0f}s")


def run_tf_filter_concurrent(trader: Trader, symbols: List[str], interval: str, max_workers: int = 20) -> List[str]:
    passed_symbols = []
    tracker = ProgressTracker(len(symbols), f"{interval} filter")
    print(f"Running {interval} filter on {len(symbols)} pairs...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_tf_dip, trader, sym, interval): sym for sym in symbols}
        for future in as_completed(futures):
            try:
                sym, is_dip = future.result()
                if is_dip: passed_symbols.append(sym)
                tracker.update(passed=is_dip)
                print(tracker.get_stats(), end="", flush=True)
            except: tracker.update()
            
    print(f"\r{tracker.get_stats()}" + " " * 20)
    return passed_symbols


def run_1m_filter_concurrent(trader: Trader, symbols: List[str], max_workers: int = 15) -> List[Tuple[str, float, float, bool]]:
    results = []
    tracker = ProgressTracker(len(symbols), "1m filter")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_1m_momentum_volume, trader, sym): sym for sym in symbols}
        for future in as_completed(futures):
            try:
                res = future.result()
                results.append(res)
                tracker.update(passed=res[3])
                print(tracker.get_stats(), end="", flush=True)
            except: tracker.update()
            
    print(f"\r{tracker.get_stats()}" + " " * 20)
    return results


def get_multi_tf_forecast(trader: Trader, symbol: str) -> Dict[str, Optional[Dict]]:
    """Fetches 500 candles for 4 timeframes and runs FFT/HT_SINE forecast on each."""
    timeframes = ['2h', '15m', '5m', '1m']
    forecasts = {}
    
    # Sequential here is fine, it's only 4 API calls for ONE final symbol
    for tf in timeframes:
        close = trader.get_klines(symbol, tf, limit=500)
        forecasts[tf] = fft_ht_sine_forecast(close)
        
    return forecasts


def format_forecast_output(symbol: str, forecasts: Dict[str, Optional[Dict]], current_price_1m: float):
    print("\n" + "="*60)
    print(f"  ★ CYCLE FORECAST: {symbol} ★")
    print("="*60)
    
    weights = {'2h': 0.10, '15m': 0.20, '5m': 0.30, '1m': 0.40}
    weighted_target = 0.0
    total_weight = 0.0
    
    for tf in ['2h', '15m', '5m', '1m']:
        f = forecasts.get(tf)
        if f:
            w = weights[tf]
            weighted_target += f['forecast_target'] * w
            total_weight += w
            
            phase = "🟢 RISING" if f['is_rising'] else "🔴 FALLING"
            change = ((f['forecast_target'] - f['current_price']) / f['current_price']) * 100
            
            print(f"  [{tf.upper():4}] Cycle Range: ${f['cycle_low']:.4f} -> ${f['cycle_high']:.4f}")
            print(f"         Amplitude: ${f['amplitude']:.4f} | Dominant Period: {f['dominant_period']} candles | Phase: {phase}")
            print(f"         Local Target: ${f['forecast_target']:.4f} ({change:+.2f}%)\n")
        else:
            print(f"  [{tf.upper():4}] Insufficient data for FFT/HT_SINE\n")
            
    if total_weight > 0:
        final_target = weighted_target / total_weight
        total_change = ((final_target - current_price_1m) / current_price_1m) * 100
        print("  " + "-"*56)
        print(f"  🎯 CONSENSUS WEIGHTED TARGET: ${final_target:.4f} ({total_change:+.2f}%)")
        print(f"  📍 Current 1m Price:         ${current_price_1m:.4f}")
    print("="*60 + "\n")


# ==========================================
# MAIN LOGIC
# ==========================================

def main():
    start_time = time.time()
    trader = Trader('credentials.txt')
    trading_pairs = trader.get_usdc_pairs()

    print("=" * 60)
    print("  MTF DIP SCANNER + FFT/HT_SINE CYCLE FORECASTER")
    print("=" * 60 + "\n")

    # MTF Filters
    filtered1 = run_tf_filter_concurrent(trader, trading_pairs, '2h', 20)
    if not filtered1: print("No 2h dips. Exiting."), sys.exit(0)

    filtered2 = run_tf_filter_concurrent(trader, filtered1, '15m', 15)
    if not filtered2: print("No 15m dips. Exiting."), sys.exit(0)

    filtered3 = run_tf_filter_concurrent(trader, filtered2, '5m', 15)
    if not filtered3: print("No 5m dips. Exiting."), sys.exit(0)

    # 1m Momentum/Volume Filter
    results_1m = run_1m_filter_concurrent(trader, filtered3, 15)
    
    strong_candidates = [r for r in results_1m if r[3] is True]
    
    final_choice = None
    mode = "NONE"

    if strong_candidates:
        # Strict mode passed
        final_choice = max(strong_candidates, key=lambda x: (-x[1], x[2]))
        mode = "STRONG (CMO < -50 + Volume Infusion)"
    else:
        # FALLBACK: Choose best candidate from the 5m dip list even if 1m conditions aren't perfect
        if results_1m:
            final_choice = min(results_1m, key=lambda x: x[1]) # Lowest CMO available
            mode = "FALLBACK (Best available 1m CMO from 5m dips)"
        else:
            print("\nFailed to fetch 1m data for fallback. Exiting.")
            sys.exit(0)

    sym, cmo_val, vratio, _ = final_choice
    
    print("\n" + "-"*60)
    print(f"  SELECTED SYMBOL: {sym}")
    print(f"  SELECTION MODE:  {mode}")
    print(f"  1m CMO:          {cmo_val:.2f}")
    print(f"  1m Volume Ratio: x{vratio:.1f}")
    print("-"*60)
    
    # Run Multi-Timeframe FFT Forecasting on the chosen asset
    print("\nCalculating FFT Dominant Cycles & HT_SINE Reversals...")
    forecasts = get_multi_tf_forecast(trader, sym)
    
    # Get current accurate 1m price for final calculation
    current_1m_close = trader.get_klines(sym, '1m', limit=1)
    current_price = current_1m_close[-1] if current_1m_close else 0.0
    
    format_forecast_output(sym, forecasts, current_price)
    
    print(f"Total Execution Time: {time.time()-start_time:.1f}s")


if __name__ == "__main__":
    main()