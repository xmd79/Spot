import os
import time
import numpy as np
import pandas as pd
import threading
import signal
import sys
import warnings
from binance.client import Client
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# --- Suppress all warnings for a clean output ---
warnings.filterwarnings("ignore")

# --- ML & Signal Processing Libraries ---
try:
    import pandas_ta as ta
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import pmdarima as pm
    from scipy.fft import fft, fftfreq
except ImportError as e:
    print("="*80)
    print("!!! IMPORT ERROR !!!")
    print(f"A required library is missing: {e}")
    print("Please install all required libraries by running the following command in your terminal:")
    print("pip install pandas pandas-ta scikit-learn pmdarima scipy")
    print("="*80)
    sys.exit()

# --- Configuration ---
API_FILE = 'api.txt'
MTF_SCAN_TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h', '1d']
DETAILED_1M_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
# NOTE: This version is API-intensive. If you get rate-limited, increase this interval.
ANALYSIS_INTERVAL_SECONDS = 30
MAX_WORKERS = 15

# --- Scoring & Criteria Configuration ---
MIN_WEIGHTED_DIP_SCORE = 4.0  # New threshold based on weighted score
VOLUME_ANALYSIS_PERIOD = 56
PRICE_UPTREND_PERIOD = 5
# A candidate must achieve this score to trigger final analysis
WINNING_SCORE_THRESHOLD = 500

# --- NEW: Optimization and Weighting Configuration ---
TIMEFRAME_WEIGHTS = {
    '5m': 1.0, '15m': 1.2, '30m': 1.4, '1h': 1.6, '4h': 1.8, '1d': 2.0
}
ASSET_SCAN_LIMIT = 100  # Limit number of assets to scan for speed
MIN_24H_VOLUME_USD = 500000  # Minimum 24h volume in USDC to consider an asset
MIN_24H_PRICE_CHANGE_PCT = 0.5 # Minimum 24h price change to consider an asset
ANALYSIS_COOLDOWN_SECONDS = 300 # 5 minutes cooldown for recently analyzed assets
BATCH_SIZE = 20 # Number of assets to analyze in a single batch
# --- End Configuration ---

# Global event to signal a graceful shutdown
stop_event = threading.Event()

def signal_handler(sig, frame):
    """Handles Ctrl+C signal gracefully."""
    print('\nCtrl+C pressed! Bot is shutting down...')
    stop_event.set()

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def get_binance_client():
    """Initialize Binance client."""
    if not os.path.exists(API_FILE):
        print(f"Error: API credentials file not found at '{API_FILE}'")
        return None
    try:
        with open(API_FILE, 'r') as file:
            api_key = file.readline().strip()
            api_secret = file.readline().strip()
        return Client(api_key, api_secret)
    except Exception as e:
        print(f"Error reading API credentials: {e}")
        return None

def fetch_usdc_pairs(client):
    """Fetch and pre-filter USDC pairs for higher potential."""
    try:
        print("Fetching and filtering USDC pairs...")
        exchange_info = client.get_exchange_info()
        symbols = exchange_info['symbols']
        
        # Basic filter as before
        usdc_pairs = [s['symbol'] for s in symbols if s['quoteAsset'] == 'USDC' and s['status'] == 'TRADING' and s['isSpotTradingAllowed']
                      and not any(exclude in s['symbol'] for exclude in ['UP', 'DOWN', 'BEAR', 'BULL'])]
        
        # Additional pre-filtering for higher potential assets
        print(f"Found {len(usdc_pairs)} total USDC pairs. Fetching 24h ticker data for filtering...")
        tickers = client.get_ticker()
        
        # Create a quick lookup for ticker data
        ticker_data = {t['symbol']: t for t in tickers}
        
        filtered_pairs = []
        for symbol in usdc_pairs:
            if symbol in ticker_data:
                ticker = ticker_data[symbol]
                # Check for minimum volume and price movement
                volume = float(ticker['quoteVolume'])
                price_change = abs(float(ticker['priceChangePercent']))
                
                if volume > MIN_24H_VOLUME_USD and price_change > MIN_24H_PRICE_CHANGE_PCT:
                    filtered_pairs.append(symbol)
        
        # Sort by volume (descending) to prioritize more liquid assets
        filtered_pairs.sort(key=lambda x: ticker_data[x]['quoteVolume'], reverse=True)
        
        # Limit to top N pairs to speed up analysis
        final_pairs = filtered_pairs[:ASSET_SCAN_LIMIT]
        print(f"Filtered down to {len(final_pairs)} high-potential assets.")
        return final_pairs
        
    except Exception as e:
        print(f"Error fetching USDC pairs: {e}")
        return []

def analyze_asset_for_table(client, symbol):
    """Gathers all required data for an asset with enhanced dip scoring."""
    if stop_event.is_set(): return None

    result = {"symbol": symbol}
    weighted_dip_score = 0
    spike_score = 0

    try:
        # --- Concurrently fetch all MTF data ---
        with ThreadPoolExecutor(max_workers=6) as mtf_executor:
            mtf_futures = [mtf_executor.submit(get_mtf_data, client, symbol, tf) for tf in MTF_SCAN_TIMEFRAMES]

            for future in as_completed(mtf_futures):
                if stop_event.is_set(): return None
                data = future.result()
                if not data: continue

                tf = data['timeframe']
                result[f'{tf}_price_change_pct'] = data['price_change_pct']
                result[f'{tf}_volume_change_pct'] = data['volume_change_pct']
                result['current_price'] = data['current_price']

                if data['is_dip']:
                    # Add weighted score instead of just counting
                    weight = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                    dip_strength = data.get('dip_strength', 50) / 100.0 # Normalize to 0-1
                    weighted_dip_score += weight * dip_strength

        # --- Spike Pattern Analysis (1m) ---
        spike_result = analyze_spike_pattern(client, symbol)
        if spike_result:
            result["spike_score"] = spike_result['score']
            result["current_price"] = spike_result['current_price']

        # --- Price & Volume Change (1h) ---
        klines_1h = client.get_klines(symbol=symbol, interval='1h', limit=2)
        if klines_1h:
            current_c, past_c = float(klines_1h[-1][4]), float(klines_1h[-2][4])
            current_v, past_v = float(klines_1h[-1][5]), float(klines_1h[-2][5])
            result["current_price"] = current_c
            result["price_change_1h_pct"] = ((current_c - past_c) / past_c) * 100 if past_c > 0 else 0
            result["volume_change_1h_pct"] = ((current_v - past_v) / past_v) * 100 if past_v > 0 else 0

    except Exception as e:
        # print(f"Error analyzing {symbol} for table: {e}")
        return None

    # Finalize result dictionary for the table
    result['weighted_dip_score'] = weighted_dip_score
    result['spike_score'] = spike_score
    result['power_score'] = (weighted_dip_score * 100) + spike_score
    return result

def get_mtf_data(client, symbol, timeframe):
    """Enhanced MTF data analysis with improved dip detection."""
    if stop_event.is_set(): return None
    try:
        # Fetch more candles for better context
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=200)
        if not klines or len(klines) < 20: return None

        closes = np.array([float(k[4]) for k in klines])
        volumes = np.array([float(k[5]) for k in klines])
        times_ms = np.array([k[0] for k in klines])
        
        # Enhanced dip detection with multiple criteria
        current_price = closes[-1]
        
        # Calculate percentiles instead of just min/max
        p10 = np.percentile(closes, 10)
        median = np.median(closes)
        
        # Criteria 1: Price in bottom 15% of recent range (less strict than 10%)
        is_dip_price = current_price <= p10
        
        # Criteria 2: Price below recent moving average (trend confirmation)
        ma20 = np.mean(closes[-20:])
        is_dip_trend = current_price < ma20
        
        # Criteria 3: Recent price decline (momentum confirmation)
        recent_change = (closes[-1] - closes[-10]) / closes[-10]
        is_dip_momentum = recent_change < -0.02  # 2% decline in last 10 periods
        
        # Combined dip condition (must meet at least 2 of 3 criteria)
        dip_criteria_met = sum([is_dip_price, is_dip_trend, is_dip_momentum])
        is_dip = dip_criteria_met >= 2
        
        # Calculate dip strength (0-100)
        dip_strength = 0
        if is_dip:
            # How far below the 10th percentile
            price_factor = max(0, (p10 - current_price) / p10 * 100) if p10 > 0 else 0
            
            # How far below the moving average
            ma_factor = max(0, (ma20 - current_price) / ma20 * 100) if ma20 > 0 else 0
            
            # Recent decline factor
            momentum_factor = abs(recent_change) * 100
            
            dip_strength = min(100, (price_factor + ma_factor + momentum_factor) / 3)

        # --- Change Calculation ---
        price_change_pct = 0
        volume_change_pct = 0
        if len(klines) >= 2:
            past_price, past_volume = closes[-2], volumes[-2]
            if past_price > 0: price_change_pct = ((current_price - past_price) / past_price) * 100
            if past_volume > 0: volume_change_pct = ((volumes[-1] - past_volume) / past_volume) * 100

        # --- Time Since Last Dip ---
        time_ago_sec = None
        if is_dip:
            min_price_index = np.argmin(closes)
            dip_timestamp_ms = times_ms[min_price_index]
            last_timestamp_ms = times_ms[-1]
            time_ago_sec = (last_timestamp_ms - dip_timestamp_ms) / 1000

        return {
            "timeframe": timeframe,
            "is_dip": is_dip,
            "dip_strength": dip_strength,
            "current_price": current_price,
            "price_change_pct": price_change_pct,
            "volume_change_pct": volume_change_pct,
            "time_ago_seconds": time_ago_sec
        }
    except Exception: return None

def analyze_spike_pattern(client, symbol):
    """Analyzes the 1m chart for signs of an imminent price spike."""
    if stop_event.is_set(): return None
    try:
        klines = client.get_klines(symbol=symbol, interval='1m', limit=max(VOLUME_ANALYSIS_PERIOD + 10, PRICE_UPTREND_PERIOD + 10))
        if not klines or len(klines) < VOLUME_ANALYSIS_PERIOD: return None
        close_prices = np.array([float(k[4]) for k in klines])
        volumes = np.array([float(k[5]) for k in klines])
        recent_closes = close_prices[-PRICE_UPTREND_PERIOD:]
        is_price_increasing = all(recent_closes[i] > recent_closes[i-1] for i in range(1, len(recent_closes)))
        recent_volumes = volumes[-VOLUME_ANALYSIS_PERIOD:]
        x_axis = np.arange(len(recent_volumes))
        volume_slope, _ = np.polyfit(x_axis, recent_volumes, 1)
        is_volume_rising = volume_slope > 0
        # --- Slightly Relaxed Spike Criteria: Only require 3 out of 5 recent candles to be increasing ---
        increasing_candles = sum(recent_closes[i] > recent_closes[i-1] for i in range(1, len(recent_closes)))
        if is_price_increasing and is_volume_rising and increasing_candles >= 3:
            score = volume_slope * 1000
            return {"score": score, "current_price": close_prices[-1]}
        return None
    except Exception: return None

def print_dynamic_table(all_results, near_misses, scan_stats):
    """Clears the screen and prints a sorted table of top candidates."""
    os.system('cls' if os.name == 'nt' else 'clear')

    print("--- Live Market Scan - Top Candidates ---")
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.sort_values(by='power_score', ascending=False).reset_index(drop=True)
        print(df.head(15).to_string(index=False, float_format="%.2f"))

    print("\n--- Near Misses (Almost Candidates) ---")
    if near_misses:
        df_near = pd.DataFrame(near_misses)
        df_near = df_near.sort_values(by='weighted_dip_score', ascending=False).reset_index(drop=True)
        print(df_near.to_string(index=False, float_format="%.2f"))

    print("\n--- Scan Statistics ---")
    for reason, count in scan_stats.items():
        print(f"  - {reason:<25}: {count}")

def format_time_ago(seconds):
    """Convert seconds into a human-readable 'time ago' string."""
    if seconds is None: return "N/A"
    dt = timedelta(seconds=int(seconds))
    days = dt.days
    hours, remainder = divmod(dt.seconds, 3600)
    minutes, _ = divmod(remainder, 60)

    parts = []
    if days > 0: parts.append(f"{days}d")
    if hours > 0: parts.append(f"{hours}h")
    if minutes > 0: parts.append(f"{minutes}m")
    if not parts: parts.append("just now")
    return " ".join(parts) + " ago"

# --- ML and Final Analysis Functions ---
def perform_final_analysis(client, symbol):
    """Performs full ML and threshold analysis on the best MTF dip."""
    print(f"\n{'='*80}")
    print(f"!!! BEST MTF DIP FOUND: {symbol} - STARTING FINAL ANALYSIS !!!")
    print(f"{'='*80}")

    print("\nCalculating MTF Min/Middle/Max Thresholds...")
    mtf_thresholds = get_mtf_thresholds(client, symbol)

    print("\nFetching data for ML models...")
    try:
        klines = client.get_klines(symbol=symbol, interval='1h', limit=2000)
        if not klines:
            print("Failed to fetch historical data. Aborting ML analysis.")
            return
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df.set_index('timestamp', inplace=True)
        current_price = df['close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return

    print("Calculating indicators and running models...")
    # Use individual, stable pandas-ta calls
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.sma(length=50, append=True)
    # Use modern pandas fillna syntax
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    # Run all models
    rf_target = run_random_forest(df)
    arima_target = run_arima(df['close'])
    svm_target = run_svm(df)
    lr_target = run_linear_regression(df)
    fft_target = run_fft_analysis(df['close'])

    # Collect all model targets
    model_targets = {
        "Random Forest": rf_target,
        "ARIMA": arima_target,
        "SVM": svm_target,
        "Linear Regression": lr_target,
        "FFT Cycle": fft_target
    }

    valid_targets = [t for t in model_targets.values() if t is not None and t > 0]
    consensus_target = np.mean(valid_targets) if valid_targets else current_price
    potential_change = ((consensus_target - current_price) / current_price) * 100

    # --- Final Report ---
    print("\n" + "="*80)
    print("!!! FINAL ANALYSIS REPORT !!!")
    print(f"{'='*80}")
    print(f"Asset: {symbol}")
    print(f"Current Price: {current_price:.8f}")
    print("-" * 40)
    print("Model Predictions:")
    for name, target in model_targets.items():
        if target: print(f"  - {name:<18}: {target:.8f}")
    print("-" * 40)
    print(f"!!! CONSENSUS INCOMING TARGET: {consensus_target:.8f} ({potential_change:+.2f}%) !!!")
    print("-" * 40)
    print("MTF Thresholds & Predictive Zones:")
    if mtf_thresholds:
        # Define the correct order of timeframes from shortest to longest
        timeframe_order = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
        
        # Iterate through timeframes in the correct order
        for tf in timeframe_order:
            if tf in mtf_thresholds:
                data = mtf_thresholds[tf]
                current_price_for_tf = data.get('current_price', 0.0)
                min_p, max_p, middle_p, std_dev = data['min'], data['max'], data['middle'], data['std_dev']

                dist_to_min_pct = ((current_price_for_tf - min_p) / min_p) * 100 if min_p > 0 else 0
                dist_to_max_pct = ((max_p - current_price_for_tf) / max_p) * 100 if max_p > 0 else 0
                target_zone_low = middle_p - (0.5 * std_dev)
                target_zone_high = middle_p + (0.5 * std_dev)

                print(f"  | {tf:<4} | Min: {min_p:<12.8f} | Max: {max_p:<12.8f} | Middle: {middle_p:<12.8f} | StdDev: {std_dev:<12.8f}")
                print(f"  |     | Current: {current_price_for_tf:<12.8f} ({dist_to_min_pct:+.2f}% from Min, {dist_to_max_pct:+.2f}% from Max)")
                print(f"  |     | Target Zone: {target_zone_low:<12.8f} - {target_zone_high:<12.8f}")
    print(f"{'='*80}")
    print("Analysis complete. Bot will now stop.")
    stop_event.set()

def run_random_forest(df):
    """Trains a Random Forest model and predicts the next price."""
    try:
        df['future_close'] = df['close'].shift(-12)
        df_ml = df.dropna()
        features = df_ml.drop(['future_close'], axis=1)
        target = df_ml['future_close']
        X_train, _, y_train, _ = train_test_split(features, target, test_size=0.2, shuffle=False)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        last_features_df = pd.DataFrame([features.iloc[-1].values], columns=features.columns)
        prediction = model.predict(last_features_df)[0]
        return prediction
    except Exception: return None

def run_arima(series):
    """Trains an ARIMA model and predicts the next price."""
    try:
        model = pm.auto_arima(series, start_p=1, start_q=1, test='adf', max_p=3, max_q=3, m=1, d=None, seasonal=False, start_P=0, D=0, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
        prediction = model.predict(n_periods=12).iloc[-1]
        return prediction
    except Exception: return None

def run_svm(df):
    """Trains a Support Vector Machine (SVR) model and predicts the next price."""
    try:
        df['future_close'] = df['close'].shift(-12)
        df_ml = df.dropna()
        features = df_ml.drop(['future_close'], axis=1)
        target = df_ml['future_close']
        X_train, _, y_train, _ = train_test_split(features, target, test_size=0.2, shuffle=False)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = SVR(kernel='rbf', C=1.0, gamma='auto')
        model.fit(X_train_scaled, y_train)
        last_features_scaled = scaler.transform(features.iloc[-1:].values)
        prediction = model.predict(last_features_scaled)[0]
        return prediction
    except Exception: return None

def run_linear_regression(df):
    """Trains a Linear Regression model and predicts the next price."""
    try:
        df['future_close'] = df['close'].shift(-12)
        df_ml = df.dropna()
        features = df_ml.drop(['future_close'], axis=1)
        target = df_ml['future_close']
        X_train, _, y_train, _ = train_test_split(features, target, test_size=0.2, shuffle=False)
        model = LinearRegression()
        model.fit(X_train, y_train)
        prediction = model.predict(features.iloc[-1:].values)[0]
        return prediction
    except Exception: return None

def run_fft_analysis(series):
    """Performs FFT to find dominant cycles and project a target."""
    try:
        detrended = series - series.rolling(window=20).mean()
        detrended = detrended.dropna()
        if len(detrended) < 100: return None
        fft_values = fft(detrended.values)
        frequencies = fftfreq(len(detrended))
        magnitude = np.abs(fft_values)
        dominant_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
        amplitude = magnitude[dominant_freq_idx] / len(detrended)
        target = series.iloc[-1] + amplitude
        return target
    except Exception: return None

def get_mtf_thresholds(client, symbol):
    """Calculates min, middle, max, and std dev for each timeframe."""
    thresholds = {}
    for timeframe in DETAILED_1M_TIMEFRAMES:
        if stop_event.is_set(): break
        try:
            klines = client.get_klines(symbol=symbol, interval=timeframe, limit=1000)
            if not klines: continue
            close_prices = np.array([float(k[4]) for k in klines])
            current_price = close_prices[-1]
            thresholds[timeframe] = {
                "min": np.min(close_prices),
                "max": np.max(close_prices),
                "middle": np.mean(close_prices),
                "std_dev": np.std(close_prices),
                "current_price": current_price
            }
            time.sleep(0.1)
        except Exception as e:
            print(f"Could not get thresholds for {symbol} on {timeframe}: {e}")
    return thresholds
# --- End ML Functions ---


def main():
    """Optimized main function for faster dip detection."""
    client = get_binance_client()
    if not client: return

    print("--- Binance Instant Dynamic Table & Analysis Bot (Optimized) ---")
    print("Press Ctrl+C to stop the bot at any time.")

    usdc_pairs = fetch_usdc_pairs(client)
    if not usdc_pairs:
        print("No USDC pairs found. Exiting.")
        return

    last_analysis_winner_key = None
    recently_analyzed = {} # Track recently analyzed assets

    while not stop_event.is_set():
        start_time = time.time()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning {len(usdc_pairs)} assets for MTF dips...")

        all_results = []
        near_misses = []
        scan_stats = {
            "Total Assets Scanned": 0,
            "Not Enough MTF Dips": 0,
            "No Spike Pattern": 0,
            "Other Errors": 0,
            "Skipped (Cooldown)": 0
        }

        # Create a list of assets to scan (excluding those on cooldown)
        assets_to_scan = []
        current_time = time.time()
        for symbol in usdc_pairs:
            if symbol not in recently_analyzed or (current_time - recently_analyzed[symbol]) > ANALYSIS_COOLDOWN_SECONDS:
                assets_to_scan.append(symbol)
            else:
                scan_stats["Skipped (Cooldown)"] += 1
        
        # Process in batches to manage API limits
        for i in range(0, len(assets_to_scan), BATCH_SIZE):
            if stop_event.is_set(): break
            
            batch = assets_to_scan[i:i+BATCH_SIZE]
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_symbol = {executor.submit(analyze_asset_for_table, client, symbol): symbol for symbol in batch}

                for future in as_completed(future_to_symbol):
                    if stop_event.is_set(): break
                    try:
                        result = future.result()
                        if result:
                            all_results.append(result)
                        else:
                            scan_stats["Other Errors"] += 1
                    except Exception:
                        scan_stats["Other Errors"] += 1

                    scan_stats["Total Assets Scanned"] += 1
            
            # Small delay between batches to respect API limits
            time.sleep(0.5)

        # Post-scan processing to find candidates and near misses
        for res in all_results:
            if res.get('weighted_dip_score', 0) < MIN_WEIGHTED_DIP_SCORE:
                near_misses.append(res)
            elif not res.get('spike_score', 0) > 0:
                scan_stats["No Spike Pattern"] += 1

        # Find the winner for the FINAL ANALYSIS (highest weighted dip score)
        analysis_winner = None
        if all_results:
            analysis_winner = max(all_results, key=lambda x: x.get('weighted_dip_score', 0))

        # Print the dynamic table and statistics
        print_dynamic_table(all_results, near_misses, scan_stats)

        if analysis_winner and analysis_winner['symbol'] != last_analysis_winner_key:
            print(f"\n!!! WINNER FOR ANALYSIS: {analysis_winner['symbol']} with weighted score {analysis_winner.get('weighted_dip_score', 0):.2f} !!!")
            
            # Mark this asset as recently analyzed
            recently_analyzed[analysis_winner['symbol']] = time.time()
            
            perform_final_analysis(client, analysis_winner['symbol'])
            last_analysis_winner_key = analysis_winner['symbol']
        else:
            # No new winner found, so we continue the loop
            pass

        if not stop_event.is_set():
            scan_duration = time.time() - start_time
            sleep_time = max(0, ANALYSIS_INTERVAL_SECONDS - scan_duration)
            print(f"\nScan complete. Next run in {sleep_time:.0f} seconds...")
            stop_event.wait(sleep_time)

    print("Bot stopped gracefully.")

if __name__ == "__main__":
    main()