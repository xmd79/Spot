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

    def get_klines(self, symbol: str, interval: str, limit: int = 500, return_raw: bool = False):
        self.rate_limiter.acquire()
        for attempt in range(3):
            try:
                klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
                if return_raw:
                    return klines
                close = [float(k[4]) for k in klines]
                return close
            except Exception as e:
                if 'rate limit' in str(e).lower():
                    time.sleep(2 ** attempt * 2)
                else:
                    time.sleep(0.5)
        return [] if not return_raw else []


# ==========================================
# INDICATORS & PURE DATA FORECASTING ENGINE
# ==========================================

def linear_regression_dip(close: List[float], deviation: float = 0.01) -> bool:
    if len(close) < 20: return False
    x = np.arange(len(close))
    slope, intercept = np.polyfit(x, close, 1)
    trend = slope * x + intercept
    lower_band = trend * (1 - deviation)
    return close[-1] < lower_band[-1]


def has_pre_spike_volume_infusion(volumes: List[float], window: int = 50, threshold: float = 2.0) -> bool:
    if len(volumes) < window + 10: return False
    avg_volume = np.mean(volumes[-window-1:-1])
    last_volume = volumes[-1]
    recent_5_avg = np.mean(volumes[-6:-1]) if len(volumes) > 6 else last_volume
    build_up = recent_5_avg > avg_volume * 1.3
    spike = last_volume > avg_volume * threshold
    return spike and build_up


def has_bullish_rejection_volume(raw_klines: list, window: int = 10) -> Tuple[bool, float]:
    if not raw_klines or len(raw_klines) < window:
        return False, 0.0
    
    recent_klines = raw_klines[-window:]
    bull_vol, bear_vol = 0.0, 0.0
    
    for k in recent_klines:
        o, c, v = float(k[1]), float(k[4]), float(k[5])
        if c > o:      bull_vol += v
        elif c < o:    bear_vol += v
            
    total_dir_vol = bull_vol + bear_vol
    if total_dir_vol == 0: return False, 0.0
        
    bull_ratio = bull_vol / total_dir_vol
    is_confirmed = bull_ratio > 0.65
    return is_confirmed, bull_ratio


def fft_ht_sine_forecast(close: List[float], volumes: List[float], bull_rejection_ratio: float) -> Optional[Dict]:
    if len(close) < 500 or len(volumes) < 500:
        return None
    
    data_close = np.array(close[-500:], dtype='float64')
    data_vol = np.array(volumes[-500:], dtype='float64')
    
    try:
        sine, leadsine = ta.HT_SINE(data_close)
    except Exception:
        return None
        
    valid_idx = ~np.isnan(sine)
    if np.sum(valid_idx) < 100:
        return None
        
    sine_clean = sine[valid_idx]
    leadsine_clean = leadsine[valid_idx]
    close_clean = data_close[valid_idx]
    vol_clean = data_vol[valid_idx]
    
    # 1. FFT to find dominant cycle period
    fft_vals = np.fft.rfft(sine_clean)
    dominant_period = len(sine_clean)
    if len(fft_vals) > 1:
        fft_magnitudes = np.abs(fft_vals[1:])
        dominant_idx = np.argmax(fft_magnitudes) + 1 
        freqs = np.fft.rfftfreq(len(sine_clean))
        dominant_freq = freqs[dominant_idx]
        if dominant_freq > 0:
            dominant_period = int(1.0 / dominant_freq)
            
    # 2. FIX: Find TRUE Structural Highs/Lows using LeadSine/Sine Crossovers
    # This prevents the inversion bug where argmin happens at a higher price than argmax
    bottoms, tops = [], []
    for i in range(1, len(sine_clean)):
        if leadsine_clean[i-1] <= sine_clean[i-1] and leadsine_clean[i] > sine_clean[i]:
            bottoms.append(i)
        if leadsine_clean[i-1] >= sine_clean[i-1] and leadsine_clean[i] < sine_clean[i]:
            tops.append(i)

    cycle_low, cycle_high = 0.0, 0.0
    if bottoms and tops:
        last_bottom_idx = bottoms[-1]
        valid_tops = [t for t in tops if t > last_bottom_idx]
        
        if valid_tops:
            last_top_idx = valid_tops[-1]
            cycle_low = np.min(close_clean[last_bottom_idx:last_top_idx+1])
            cycle_high = np.max(close_clean[last_bottom_idx:last_top_idx+1])
        else:
            if len(bottoms) >= 2 and len(tops) >= 1:
                prev_bottom_idx = bottoms[-2]
                prev_top_idx = tops[-1]
                cycle_low = np.min(close_clean[prev_bottom_idx:prev_top_idx+1])
                cycle_high = np.max(close_clean[prev_bottom_idx:prev_top_idx+1])

    # Absolute Fallback if no crossovers found
    if cycle_high <= cycle_low:
        lookback = min(dominant_period * 2, len(close_clean))
        cycle_low = np.min(close_clean[-lookback:])
        cycle_high = np.max(close_clean[-lookback:])

    # 3. Determine Phase & Distance
    current_price = close_clean[-1]
    distance_from_bottom = len(sine_clean) - 1 - bottoms[-1] if bottoms else 999
    leadsine_crossed_up = leadsine_clean[-1] > sine_clean[-1]
    at_recent_bottom = distance_from_bottom <= (dominant_period * 1.2)

    base_target = cycle_high
    
    # 4. FIX: Dynamic Scaling that NEVER zeroes out
    vol_at_top_idx = tops[-1] if tops else -1
    vol_at_cycle_top = np.mean(vol_clean[max(0, vol_at_top_idx-2):vol_at_top_idx+3]) if vol_at_top_idx > 0 else np.mean(vol_clean)
    current_vol = vol_clean[-1]
    
    vol_top_ratio = (current_vol / vol_at_cycle_top) if vol_at_cycle_top > 0 else 1.0
    
    # If bull_ratio is 0 (fallback mode), treat it as 1.0 (neutral). If > 0, scale it.
    rejection_momentum = max(bull_rejection_ratio * 2.0, 1.0) if bull_rejection_ratio > 0 else 1.0
    dynamic_scale = vol_top_ratio * rejection_momentum

    if leadsine_crossed_up:
        phase_status = "🚀 REVERSAL UP"
        forecast_target = base_target * dynamic_scale
    elif at_recent_bottom:
        phase_status = "⚡ EXHAUSTION DIP"
        forecast_target = base_target * dynamic_scale
    else:
        phase_status = "🔄 ACCUMULATING"
        mid_cycle_point = cycle_low + ((cycle_high - cycle_low) / 2)
        forecast_target = mid_cycle_point
        
    # Hard Floor: Never project below current price when scanning for long dips
    forecast_target = max(forecast_target, current_price)
        
    return {
        'cycle_low': cycle_low,
        'cycle_high': cycle_high,
        'dynamic_scale': dynamic_scale,
        'dominant_period': dominant_period,
        'phase_status': phase_status,
        'forecast_target': forecast_target,
        'current_price': current_price,
        'distance_from_bottom': distance_from_bottom
    }


# ==========================================
# CONCURRENT FILTER FUNCTIONS
# ==========================================

def check_tf_dip(trader: Trader, symbol: str, interval: str) -> Tuple[str, bool]:
    close = trader.get_klines(symbol, interval, limit=300)
    return (symbol, linear_regression_dip(close, 0.01))

def check_5m_rejection(trader: Trader, symbol: str) -> Tuple[str, bool, float]:
    klines = trader.get_klines(symbol, '5m', limit=300, return_raw=True)
    if not klines: return (symbol, False, 0.0)
    close = [float(k[4]) for k in klines]
    if linear_regression_dip(close, 0.01):
        is_rejection, ratio = has_bullish_rejection_volume(klines, window=10)
        return (symbol, is_rejection, ratio)
    return (symbol, False, 0.0)

def check_1m_final(trader: Trader, symbol: str) -> Tuple[str, float, float, bool, float]:
    klines = trader.get_klines(symbol, '1m', limit=200, return_raw=True)
    if not klines or len(klines) < 50: return (symbol, 0.0, 0.0, False, 0.0)
    close = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]
    cmo = ta.CMO(np.asarray(close), timeperiod=14)
    cmo_val = cmo[-1] if not np.isnan(cmo[-1]) else 0.0
    avg_vol = np.mean(volumes[-51:-1]) if len(volumes) > 51 else np.mean(volumes[:-1])
    vratio = volumes[-1] / avg_vol if avg_vol > 0 else 0.0
    is_rejection, bull_ratio = has_bullish_rejection_volume(klines, window=10)
    is_strong = (cmo_val < -50) and has_pre_spike_volume_infusion(volumes) and is_rejection
    return (symbol, cmo_val, vratio, is_strong, bull_ratio)

class ProgressTracker:
    def __init__(self, total: int, label: str):
        self.total, self.label, self.completed, self.passed = total, label, 0, 0
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
            return (f"\r{self.label}: {self.completed}/{self.total} | ✓{self.passed} | {rate:.1f}/s | ETA: {remaining:.0f}s")

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

def run_5m_filter_concurrent(trader: Trader, symbols: List[str], max_workers: int = 15) -> Tuple[List[str], float]:
    passed_symbols, best_ratio = [], 0.0
    tracker = ProgressTracker(len(symbols), "5m+Rej filter")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_5m_rejection, trader, sym): sym for sym in symbols}
        for future in as_completed(futures):
            try:
                sym, passed, ratio = future.result()
                if passed: 
                    passed_symbols.append(sym)
                    if ratio > best_ratio: best_ratio = ratio
                tracker.update(passed=passed)
                print(tracker.get_stats(), end="", flush=True)
            except: tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20)
    return passed_symbols, best_ratio

def run_1m_filter_concurrent(trader: Trader, symbols: List[str], max_workers: int = 15) -> List[Tuple[str, float, float, bool, float]]:
    results = []
    tracker = ProgressTracker(len(symbols), "1m filter")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_1m_final, trader, sym): sym for sym in symbols}
        for future in as_completed(futures):
            try:
                res = future.result()
                results.append(res)
                tracker.update(passed=res[3])
                print(tracker.get_stats(), end="", flush=True)
            except: tracker.update()
    print(f"\r{tracker.get_stats()}" + " " * 20)
    return results

def get_multi_tf_forecast(trader: Trader, symbol: str, best_bull_ratio: float) -> Dict[str, Optional[Dict]]:
    # All requested timeframes
    timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h']
    forecasts = {}
    print("Fetching 7 Timeframes for Cycle Mapping...", end=" ", flush=True)
    
    for tf in timeframes:
        klines = trader.get_klines(symbol, tf, limit=500, return_raw=True)
        if klines:
            close = [float(k[4]) for k in klines]
            volumes = [float(k[5]) for k in klines]
            
            # Calculate live rejection ratio for active intra-day timeframes
            if tf in ['1m', '3m', '5m']:
                _, live_ratio = has_bullish_rejection_volume(klines, window=10)
                forecasts[tf] = fft_ht_sine_forecast(close, volumes, live_ratio)
            else:
                # Higher TFs use the best established ratio to avoid noise
                forecasts[tf] = fft_ht_sine_forecast(close, volumes, best_bull_ratio)
        else:
            forecasts[tf] = None
            
    print("Done.")
    return forecasts

def format_forecast_output(symbol: str, forecasts: Dict[str, Optional[Dict]], current_price_1m: float):
    print("\n" + "="*60)
    print(f"  ★ STRUCTURAL REJECTION CYCLE FORECAST: {symbol} ★")
    print("="*60)
    
    # Re-weight to account for 7 timeframes
    weights = {'1m': 0.25, '3m': 0.20, '5m': 0.15, '15m': 0.15, '30m': 0.10, '1h': 0.10, '2h': 0.05}
    weighted_target = 0.0
    total_weight = 0.0
    
    for tf in ['1m', '3m', '5m', '15m', '30m', '1h', '2h']:
        f = forecasts.get(tf)
        if f:
            w = weights[tf]
            weighted_target += f['forecast_target'] * w
            total_weight += w
            
            change = ((f['forecast_target'] - f['current_price']) / f['current_price']) * 100
            
            print(f"  [{tf.upper():4}] True Cycle Low:  ${f['cycle_low']:.4f}")
            print(f"         True Cycle High: ${f['cycle_high']:.4f}")
            print(f"         Dominant Period: {f['dominant_period']} candles | Since Bottom: {f['distance_from_bottom']}")
            print(f"         Dynamic Scale:   x{f['dynamic_scale']:.2f} | Phase: {f['phase_status']}")
            print(f"         Projected Target: ${f['forecast_target']:.4f} ({change:+.2f}%)\n")
        else:
            print(f"  [{tf.upper():4}] Insufficient data\n")
            
    if total_weight > 0:
        final_target = weighted_target / total_weight
        total_change = ((final_target - current_price_1m) / current_price_1m) * 100
        print("  " + "-"*56)
        print(f"  🎯 7-TF CONSENSUS TARGET: ${final_target:.4f} ({total_change:+.2f}%)")
        print(f"  📍 Current Entry Price:   ${current_price_1m:.4f}")
    print("="*60 + "\n")


# ==========================================
# MAIN LOGIC
# ==========================================

def main():
    start_time = time.time()
    trader = Trader('credentials.txt')
    trading_pairs = trader.get_usdc_pairs()

    print("=" * 60)
    print("  7-TF MTF SCANNER + OHLCV REJECTION FORECASTER")
    print("=" * 60 + "\n")

    filtered1 = run_tf_filter_concurrent(trader, trading_pairs, '2h', 20)
    if not filtered1: print("No 2h dips. Exiting."), sys.exit(0)

    filtered2 = run_tf_filter_concurrent(trader, filtered1, '15m', 15)
    if not filtered2: print("No 15m dips. Exiting."), sys.exit(0)

    filtered3, best_5m_ratio = run_5m_filter_concurrent(trader, filtered2, 15)
    if not filtered3: 
        print("\nNo 5m dips with Bullish Rejection Volume. Exiting.")
        sys.exit(0)

    results_1m = run_1m_filter_concurrent(trader, filtered3, 15)
    
    strong_candidates = [r for r in results_1m if r[3] is True]
    final_choice = None
    mode = "NONE"

    if strong_candidates:
        final_choice = max(strong_candidates, key=lambda x: (-x[1], x[4]))
        mode = "STRONG (CMO + Spike + OHLCV Bull Rejection)"
        live_bull_ratio = final_choice[4]
    else:
        if results_1m:
            final_choice = min(results_1m, key=lambda x: x[1]) 
            mode = "FALLBACK (Best CMO, 1m Rejection weak)"
            live_bull_ratio = final_choice[4] if final_choice[4] > 0 else best_5m_ratio # Cascade ratio
        else:
            print("\nFailed to fetch 1m data. Exiting.")
            sys.exit(0)

    sym, cmo_val, vratio, _, live_bull_ratio = final_choice
    
    print("\n" + "-"*60)
    print(f"  SELECTED SYMBOL: {sym}")
    print(f"  SELECTION MODE:  {mode}")
    print(f"  1m CMO:          {cmo_val:.2f}")
    print(f"  1m Volume Ratio: x{vratio:.1f}")
    print(f"  Bull Rej Vol:    {live_bull_ratio*100:.1f}% (Green Candles)")
    print("-"*60)
    
    forecasts = get_multi_tf_forecast(trader, sym, live_bull_ratio)
    
    current_1m_close = trader.get_klines(sym, '1m', limit=1)
    current_price = current_1m_close[-1] if current_1m_close else 0.0
    
    format_forecast_output(sym, forecasts, current_price)
    
    print(f"Total Execution Time: {time.time()-start_time:.1f}s")


if __name__ == "__main__":
    main()
