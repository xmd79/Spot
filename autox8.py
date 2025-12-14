import numpy as np
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import datetime
import time
import concurrent.futures
import talib
import gc
import math
from decimal import Decimal, getcontext
import requests
import os
import pandas as pd
import warnings
from scipy.fft import fft, fftfreq, ifft

# Set Decimal precision to 25
getcontext().prec = 25

# Exchange constants
TRADE_SYMBOL = "BTCUSDC"

# Timezone Configuration
LOCAL_TIMEZONE = datetime.timezone(datetime.timedelta(hours=2))  # GMT+2

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Load credentials from file
with open("api.txt", "r") as f:
    lines = f.readlines()
    api_key = lines[0].strip()
    api_secret = lines[1].strip()

# Initialize Binance client with increased timeout
client = BinanceClient(api_key, api_secret, requests_params={"timeout": 30})  # Timeout set to 30 seconds

# Trading Configuration
PROFIT_TARGET_PERCENT = 0.35  # 0.35% profit target (changed from 0.65%)
TOTAL_FEE_PERCENT = 0.22  # Total fee percentage (0.1% for buy + 0.1% for sell + 0.02% buffer)
MIN_TRADE_AMOUNT = 10

# Technical Indicators Configuration
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Updated configurable conditions - Replaced conditions as requested
CONFIG = {
    "conditions": {
        # FFT forecast next targets condition
        "fft_forecast_next_targets": True,  # FFT Forecast Next Targets
        
        # Conditions from second code (with some removed)
        "fft_forecast_price_15s": True,  # FFT Forecast Price (15s)
        "time_distance_condition": True,  # Time Distance Condition (replaced close_below_sma200_15s)
        "momentum_positive_1m": True,  # Momentum Positive (1m)
        "fft_forecast_up_1m": True,  # FFT Forecast Up (1m)
        "fft_forecast_up_3m": True,  # FFT Forecast Up (3m)
        "fft_forecast_up_5m": True,  # FFT Forecast Up (5m)
        
        # Momentum condition for 15s
        "momentum_positive_15sec": True, # Momentum Positive (15s)
    },
    "min_conditions_met": 8  # ALL 8 conditions must be met to trigger a trade
}

# Utility Functions
def fetch_candles_in_parallel(timeframes, symbol=TRADE_SYMBOL, limit=1200):
    def fetch_candles(timeframe):
        return get_candles(symbol, timeframe, limit)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_candles, timeframes))
    return dict(zip(timeframes, results))

def get_candles(symbol, timeframe, limit=1200, retries=5, delay=5):
    for attempt in range(retries):
        try:
            klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
            candles = []
            for k in klines:
                candle = {
                    "time": k[0] / 1000,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "timeframe": timeframe
                }
                candles.append(candle)
            return candles
        except BinanceAPIException as e:
            print(f"Binance API Error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
        except requests.exceptions.ReadTimeout as e:
            print(f"Read Timeout fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except Exception as e:
            print(f"Unexpected error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    print(f"Failed to fetch candles for {timeframe} after {retries} attempts. Skipping timeframe.")
    return []

def get_current_price(retries=5, delay=5):
    for attempt in range(retries):
        try:
            ticker = client.get_symbol_ticker(symbol=TRADE_SYMBOL)
            price = Decimal(str(ticker['price']))
            if price > Decimal('0'):
                return price
            print(f"Invalid price {price:.25f} on attempt {attempt + 1}/{retries}")
        except BinanceAPIException as e:
            print(f"Error fetching {TRADE_SYMBOL} price (attempt {attempt + 1}/{retries}): {e.message}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except requests.exceptions.ReadTimeout as e:
            print(f"Read Timeout fetching price (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    print(f"Failed to fetch valid {TRADE_SYMBOL} price after {retries} attempts.")
    return Decimal('0.0')

def get_balance(asset='USDC'):
    try:
        balance_info = client.get_asset_balance(asset)
        return Decimal(str(balance_info['free'])) if balance_info else Decimal('0.0')
    except BinanceAPIException as e:
        print(f"Error fetching balance for {asset}: {e.message}")
        return Decimal('0.0')

def get_last_buy_trade():
    try:
        trades = client.get_my_trades(symbol=TRADE_SYMBOL)
        if not trades:
            print("No trades found.")
            return None
        for trade in reversed(trades):
            if trade['isBuyer']:
                return {
                    "price": Decimal(str(trade['price'])),
                    "qty": Decimal(str(trade['qty'])),
                    "time": trade['time']
                }
    except BinanceAPIException as e:
        print(f"Error fetching trade history: {e.message}")
    return None

def get_average_entry_price():
    last_trade = get_last_buy_trade()
    if last_trade:
        entry_price = last_trade['price']
        print(f"Using last buy trade price as entry price: {entry_price:.25f}")
        return entry_price
    print(f"No valid last buy trade found for {TRADE_SYMBOL}; cannot calculate entry price.")
    return Decimal('0.0')

def get_symbol_lot_size_info(symbol):
    try:
        exchange_info = client.get_symbol_info(symbol)
        for filter in exchange_info['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                return {
                    'minQty': Decimal(str(filter['minQty'])),
                    'stepSize': Decimal(str(filter['stepSize']))
                }
        print(f"Could not find LOT_SIZE filter for {symbol}. Using defaults.")
        return {'minQty': Decimal('0.00001'), 'stepSize': Decimal('0.00001')}
    except BinanceAPIException as e:
        print(f"Error fetching symbol info for {symbol}: {e.message}")
        return {'minQty': Decimal('0.00001'), 'stepSize': Decimal('0.00001')}

def buy_asset():
    try:
        current_price = get_current_price()
        if current_price <= Decimal('0'):
            print(f"Invalid current price {current_price:.25f} for buy order.")
            return None, None, None, None
        usdc_balance = get_balance('USDC')
        if usdc_balance <= Decimal('0'):
            print("No USDC balance available to place buy order.")
            return None, None, None, None
        raw_quantity = usdc_balance / current_price
        print(f"Raw quantity calculated: {raw_quantity:.25f} (USDC: {usdc_balance:.25f}, Price: {current_price:.25f})")
        step_precision = int(-math.log10(float(step_size))) if step_size > Decimal('0') else 8
        adjusted_quantity = (raw_quantity // step_size) * step_size
        adjusted_quantity = adjusted_quantity.quantize(Decimal('0.' + '0' * step_precision))
        cost = adjusted_quantity * current_price
        print(f"Adjusted quantity (max balance): {adjusted_quantity:.25f}, Cost: {cost:.25f} (Step size: {step_size:.25f}, Precision: {step_precision})")
        min_notional = Decimal('10.0')
        if cost < min_notional:
            print(f"Cost {cost:.25f} USDC is below minimum notional value {min_notional:.25f}. Adjusting quantity.")
            min_quantity_for_notional = min_notional / current_price
            adjusted_quantity = ((min_quantity_for_notional + step_size - Decimal('1E-25')) // step_size) * step_size
            adjusted_quantity = adjusted_quantity.quantize(Decimal('0.' + '0' * step_precision))
            cost = adjusted_quantity * current_price
            print(f"Re-adjusted quantity for notional: {adjusted_quantity:.25f}, New Cost: {cost:.25f}")
        if adjusted_quantity < min_trade_size:
            print(f"Adjusted quantity {adjusted_quantity:.25f} is below minimum trade size {min_trade_size:.25f}. Cannot execute trade.")
            return None, None, None, None
        if cost > usdc_balance:
            print(f"Cost {cost:.25f} exceeds available balance {usdc_balance:.25f}. Re-adjusting.")
            adjusted_quantity = ((usdc_balance / current_price) // step_size) * step_size
            adjusted_quantity = adjusted_quantity.quantize(Decimal('0.' + '0' * step_precision))
            cost = adjusted_quantity * current_price
            print(f"Re-adjusted quantity to fit balance: {adjusted_quantity:.25f}, Final Cost: {cost:.25f}")
        if adjusted_quantity < min_trade_size:
            print(f"Final adjusted quantity {adjusted_quantity:.25f} still below minimum trade size {min_trade_size:.25f}. Cannot execute trade.")
            return None, None, None, None
        remaining_balance = usdc_balance - cost
        print(f"Using {cost:.25f} of {usdc_balance:.25f} USDC, Remaining Balance: {remaining_balance:.25f} USDC")
        order = client.order_market_buy(symbol=TRADE_SYMBOL, quantity=float(adjusted_quantity))
        print(f"Market buy order executed: {order}")
        entry_price = Decimal(str(order['fills'][0]['price']))
        entry_datetime = datetime.datetime.now()
        return entry_price, adjusted_quantity, entry_datetime, cost
    except BinanceAPIException as e:
        print(f"Error executing buy order: {e.message}")
        return None, None, None, None

def sell_asset(asset_balance):
    try:
        current_price = get_current_price()
        if current_price <= Decimal('0'):
            print(f"Invalid current price {current_price:.25f} for sell order.")
            return False
        asset_balance_dec = Decimal(str(asset_balance))
        step_precision = int(-math.log10(float(step_size))) if step_size > Decimal('0') else 8
        sell_quantity = (asset_balance_dec // step_size) * step_size
        sell_quantity = sell_quantity.quantize(Decimal('0.' + '0' * step_precision))
        if sell_quantity < min_trade_size:
            print(f"Cannot sell: Adjusted quantity {sell_quantity:.25f} is below minimum trade size {min_trade_size:.25f}.")
            return False
        sell_order = client.order_market_sell(symbol=TRADE_SYMBOL, quantity=float(sell_quantity))
        print(f"Market sell order executed: {sell_order}")
        return True
    except BinanceAPIException as e:
        print(f"Error executing sell order: {e.message}")
        return False

def check_exit_condition(initial_investment, asset_balance, entry_price):
    if initial_investment <= Decimal('0.0') or asset_balance <= Decimal('0.0') or entry_price <= Decimal('0.0'):
        print("Invalid initial investment, asset balance, or entry price for exit condition check.")
        return False
    current_price = get_current_price()
    if current_price <= Decimal('0.0'):
        print("Invalid current price for exit condition check.")
        return False
    current_value = asset_balance * current_price
    target_value = initial_investment * Decimal('1.0035')  # 0.35% profit target (changed from 0.65%)
    target_price = target_value / asset_balance
    print(f"Exit Check: Current Price: {current_price:.25f}, Target Price: {target_price:.25f}, Current Value: {current_value:.25f}, Target Value: {target_value:.25f}")
    return current_price >= target_price

# Analysis Functions
def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3, range_distance=Decimal('0.05')):
    close_prices = np.array([float(x) for x in close_prices if not np.isnan(x) and x > 0], dtype=np.float64)
    if len(close_prices) == 0:
        return None, None, None, None, None, None, None
    min_close = Decimal(str(np.nanmin(close_prices)))
    max_close = Decimal(str(np.nanmax(close_prices)))
    momentum = talib.MOM(close_prices, timeperiod=period)
    min_momentum = Decimal(str(np.nanmin(momentum)))
    max_momentum = Decimal(str(np.nanmax(momentum)))
    min_percentage_custom = Decimal(str(minimum_percentage)) / Decimal('100')
    max_percentage_custom = Decimal(str(maximum_percentage)) / Decimal('100')
    min_threshold = min(min_close - (max_close - min_close) * min_percentage_custom, Decimal(str(close_prices[-1])))
    max_threshold = max(max_close + (max_close - min_close) * max_percentage_custom, Decimal(str(close_prices[-1])))
    range_price = [Decimal(str(x)) for x in np.linspace(float(close_prices[-1]) * (1 - float(range_distance)), float(close_prices[-1]) * (1 + float(range_distance)), num=50)]
    with np.errstate(invalid='ignore'):
        filtered_close = np.where(close_prices < float(min_threshold), float(min_threshold), close_prices)
        filtered_close = np.where(filtered_close > float(max_threshold), float(max_threshold), filtered_close)
    avg_mtf = Decimal(str(np.nanmean(filtered_close)))
    current_momentum = Decimal(str(momentum[-1]))
    with np.errstate(invalid='ignore', divide='ignore'):
        percent_to_min_momentum = (max_momentum - current_momentum) / (max_momentum - min_momentum) * Decimal('100') if max_momentum != min_momentum else Decimal('NaN')
        percent_to_max_momentum = (current_momentum - min_momentum) / (max_momentum - min_momentum) * Decimal('100') if max_momentum != min_momentum else Decimal('NaN')
    percent_to_min_combined = (Decimal(str(minimum_percentage)) + percent_to_min_momentum) / Decimal('2')
    percent_to_max_combined = (Decimal(str(maximum_percentage)) + percent_to_max_momentum) / Decimal('2')
    momentum_signal = percent_to_max_combined - percent_to_min_combined
    return min_threshold, max_threshold, avg_mtf, momentum_signal, range_price, percent_to_min_momentum, percent_to_max_momentum

def calculate_buy_sell_volume(candle_map):
    buy_volume, sell_volume = {}, {}
    for timeframe in candle_map:
        buy_volume[timeframe] = []
        sell_volume[timeframe] = []
        total_buy = Decimal('0.0')
        total_sell = Decimal('0.0')
        for candle in candle_map[timeframe]:
            if Decimal(str(candle["close"])) > Decimal(str(candle["open"])):
                total_buy += Decimal(str(candle["volume"]))
            elif Decimal(str(candle["close"])) < Decimal(str(candle["open"])):
                total_sell += Decimal(str(candle["volume"]))
            buy_volume[timeframe].append(total_buy)
            sell_volume[timeframe].append(total_sell)
    return buy_volume, sell_volume

def calculate_volume_ratio(buy_volume, sell_volume):
    volume_ratio = {}
    for timeframe in buy_volume.keys():
        total_volume = buy_volume[timeframe][-1] + sell_volume[timeframe][-1]
        if total_volume > Decimal('0'):
            ratio = (buy_volume[timeframe][-1] / total_volume) * Decimal('100')
            volume_ratio[timeframe] = {"buy_ratio": ratio, "sell_ratio": Decimal('100') - ratio, "status": "Bullish" if ratio > Decimal('50') else "Bearish" if ratio < Decimal('50') else "Neutral"}
        else:
            volume_ratio[timeframe] = {"buy_ratio": Decimal('0'), "sell_ratio": Decimal('0'), "status": "No Activity"}
    return volume_ratio

def find_major_reversals(candles, current_close, min_threshold, max_threshold):
    lows = [Decimal(str(candle['low'])) for candle in candles if Decimal(str(candle['low'])) >= min_threshold]
    highs = [Decimal(str(candle['high'])) for candle in candles if Decimal(str(candle['high'])) <= max_threshold]
    last_bottom = min(lows) if lows else None
    last_top = max(highs) if highs else None
    closest_reversal = None
    closest_type = None
    current_close_dec = Decimal(str(current_close))
    if last_bottom is not None:
        if closest_reversal is None or abs(last_bottom - current_close_dec) < abs(closest_reversal - current_close_dec):
            closest_reversal = last_bottom
            closest_type = 'DIP'
    if last_top is not None:
        if closest_reversal is None or abs(last_top - current_close_dec) < abs(closest_reversal - current_close_dec):
            closest_reversal = last_top
            closest_type = 'TOP'
    if closest_type == 'TOP' and closest_reversal <= current_close_dec:
        closest_type = None
        closest_reversal = None
    elif closest_type == 'DIP' and closest_reversal >= current_close_dec:
        closest_type = None
        closest_reversal = None
    return last_bottom, last_top, closest_reversal, closest_type

def calculate_momentum(candles, period=10):
    """Calculate momentum indicator."""
    try:
        if not candles or len(candles) < period + 1:
            return None, 0.0, {"error": "Insufficient data for momentum analysis"}
        
        close_prices = np.array([candle["close"] for candle in candles], dtype=np.float64)
        
        # Calculate momentum (current price minus price N periods ago)
        momentum = np.zeros(len(close_prices))
        for i in range(period, len(close_prices)):
            momentum[i] = close_prices[i] - close_prices[i - period]
        
        current_momentum = float(momentum[-1])
        
        # Check if momentum is positive
        momentum_positive = current_momentum > 0
        
        details = {
            "timeframe": candles[0]["timeframe"] if candles else "unknown",
            "current_momentum": current_momentum,
            "momentum_positive": momentum_positive,
            "period": period
        }
        
        return momentum_positive, current_momentum, details
        
    except Exception as e:
        print(f"calculate_momentum error: {e}")
        return False, 0.0, {"error": str(e)}

def improved_fft_forecast(candles, forecast_periods=4):
    """Improved FFT analysis with robust frequency filtering and proper data cleaning."""
    try:
        if not candles or len(candles) < 10:
            return np.array([candles[-1]["close"]] * forecast_periods) if candles else np.array([1.0] * forecast_periods)
        
        # Extract close prices
        close_prices = np.array([candle["close"] for candle in candles], dtype=np.float64)
        
        # Ensure we have enough data
        if len(close_prices) < 20:
            # Pad with last value if needed
            padding = np.full(20 - len(close_prices), close_prices[-1])
            close_prices = np.concatenate([padding, close_prices])
        
        # Detrend the data
        mean_val = np.mean(close_prices)
        detrended = close_prices - mean_val
        
        # Apply FFT
        fft_values = fft(detrended)
        
        # Calculate power spectrum
        power = np.abs(fft_values) ** 2
        
        # Filter frequencies (keep only significant ones)
        threshold = np.max(power) * 0.1  # Keep frequencies with 10% of max power
        significant_freqs = np.where(power > threshold)[0]
        
        # Create filtered FFT
        filtered_fft = np.zeros_like(fft_values, dtype=complex)
        filtered_fft[significant_freqs] = fft_values[significant_freqs]
        
        # Inverse FFT to get forecast
        forecast = ifft(filtered_fft).real
        
        # Add trend back
        forecast = forecast + mean_val
        
        # Return only the forecast periods
        forecast_result = forecast[-forecast_periods:]
        
        # Ensure forecast is valid
        forecast_result = np.where(np.isfinite(forecast_result), forecast_result, mean_val)
        
        return forecast_result
    except Exception as e:
        print(f"Error in improved FFT forecast: {e}")
        return np.array([candles[-1]["close"]] * forecast_periods) if candles else np.array([1.0] * forecast_periods)

def analyze_fft_cycle_with_natural_argmin_argmax(candles, timeframe):
    """
    Analyze FFT cycle using natural argmin and argmax from the last 1200 candles.
    No pattern enforcement - uses the actual values as they naturally occur.
    """
    try:
        if not candles:
            return {"error": "No data provided"}
        
        # Extract price data
        close_prices = np.array([candle["close"] for candle in candles], dtype=np.float64)
        low_prices = np.array([candle["low"] for candle in candles], dtype=np.float64)
        high_prices = np.array([candle["high"] for candle in candles], dtype=np.float64)
        
        # Find natural argmin and argmax from the entire dataset
        # For argmin: find absolute lowest low
        # For argmax: find absolute highest high
        
        # Find absolute minimum (lowest low)
        min_idx = np.argmin(low_prices)
        lowest_low_price = low_prices[min_idx]
        lowest_low_time = datetime.datetime.fromtimestamp(candles[min_idx]["time"], tz=LOCAL_TIMEZONE)
        
        # Find absolute maximum (highest high)
        max_idx = np.argmax(high_prices)
        highest_high_price = high_prices[max_idx]
        highest_high_time = datetime.datetime.fromtimestamp(candles[max_idx]["time"], tz=LOCAL_TIMEZONE)
        
        # Determine which occurred more recently
        dip_more_recent = min_idx > max_idx
        cycle_direction = "up" if dip_more_recent else "down"
        
        current_price = close_prices[-1]
        
        # Use improved FFT for forecasting
        forecast_prices = improved_fft_forecast(candles, forecast_periods=4)
        forecast_target = forecast_prices[-1]
        
        # Calculate percentage difference to forecast target
        forecast_diff_pct = ((forecast_target - current_price) / current_price) * 100
        
        # Prepare results
        results = {
            "timeframe": timeframe,
            "cycle_direction": cycle_direction,
            "current_price": current_price,
            "forecast_target": forecast_target,
            "forecast_diff_pct": forecast_diff_pct,
            "lowest_low_price": lowest_low_price,
            "lowest_low_time": lowest_low_time,
            "lowest_low_idx": int(min_idx),
            "highest_high_price": highest_high_price,
            "highest_high_time": highest_high_time,
            "highest_high_idx": int(max_idx),
            "dip_more_recent": dip_more_recent,
            "data_points": len(candles)
        }
        
        return results
        
    except Exception as e:
        print(f"Error analyzing FFT cycle with natural argmin/argmax: {e}")
        return {"error": str(e)}

def analyze_fft_cycle_with_natural_values(candles_1m, candles_3m, candles_5m):
    """
    Analyze FFT cycles for all timeframes using natural argmin and argmax values.
    No pattern enforcement - uses the actual values as they naturally occur.
    """
    try:
        # Analyze each timeframe separately with natural argmin/argmax
        fft_1m = analyze_fft_cycle_with_natural_argmin_argmax(candles_1m, '1m')
        fft_3m = analyze_fft_cycle_with_natural_argmin_argmax(candles_3m, '3m')
        fft_5m = analyze_fft_cycle_with_natural_argmin_argmax(candles_5m, '5m')
        
        # Check for errors
        if 'error' in fft_1m or 'error' in fft_3m or 'error' in fft_5m:
            return {
                "1m": fft_1m,
                "3m": fft_3m,
                "5m": fft_5m,
                "natural_analysis": False
            }
        
        # Print detailed argmin and argmax information for each timeframe
        print(f"\n--- Natural Argmin/Argmax Analysis ---")
        print(f"1m - Argmin (Lowest Low): {fft_1m['lowest_low_price']:.2f} at index {fft_1m['lowest_low_idx']} ({fft_1m['lowest_low_time'].strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"1m - Argmax (Highest High): {fft_1m['highest_high_price']:.2f} at index {fft_1m['highest_high_idx']} ({fft_1m['highest_high_time'].strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"3m - Argmin (Lowest Low): {fft_3m['lowest_low_price']:.2f} at index {fft_3m['lowest_low_idx']} ({fft_3m['lowest_low_time'].strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"3m - Argmax (Highest High): {fft_3m['highest_high_price']:.2f} at index {fft_3m['highest_high_idx']} ({fft_3m['highest_high_time'].strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"5m - Argmin (Lowest Low): {fft_5m['lowest_low_price']:.2f} at index {fft_5m['lowest_low_idx']} ({fft_5m['lowest_low_time'].strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"5m - Argmax (Highest High): {fft_5m['highest_high_price']:.2f} at index {fft_5m['highest_high_idx']} ({fft_5m['highest_high_time'].strftime('%Y-%m-%d %H:%M:%S')})")
        
        # Note: We are NOT checking or enforcing any patterns
        # Using the natural values as they occur in market
        
        print(f"\n--- Extrema Values Summary ---")
        print(f"Using natural argmin/argmax values without pattern enforcement")
        print(f"1m: Low={fft_1m['lowest_low_price']:.2f}, High={fft_1m['highest_high_price']:.2f}")
        print(f"3m: Low={fft_3m['lowest_low_price']:.2f}, High={fft_3m['highest_high_price']:.2f}")
        print(f"5m: Low={fft_5m['lowest_low_price']:.2f}, High={fft_5m['highest_high_price']:.2f}")
        
        return {
            "1m": fft_1m,
            "3m": fft_3m,
            "5m": fft_5m,
            "natural_analysis": True
        }
        
    except Exception as e:
        print(f"Error analyzing FFT cycles with natural values: {e}")
        return {
            "1m": {"error": str(e)},
            "3m": {"error": str(e)},
            "5m": {"error": str(e)},
            "natural_analysis": False
        }

def generate_15s_data_from_1m(df_1m):
    """
    Generate 15-second OHLCV data from 1-minute data.
    This function creates 4 15-second candles from each 1-minute candle.
    """
    if df_1m is None or df_1m.empty:
        return None
    
    # Create a new DataFrame for 15s data
    df_15s = pd.DataFrame()
    
    # For each 1-minute candle, create 4 15-second candles
    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    for idx, row in df_1m.iterrows():
        # Get the 1-minute candle data
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']
        volume = row['volume']
        
        # Create 4 15-second candles
        for i in range(4):
            # Calculate timestamp for each 15-second candle
            timestamp = row['timestamp'] + datetime.timedelta(seconds=15 * i)
            timestamps.append(timestamp)
            
            # For simplicity, distribute the OHLC and volume evenly
            # In a real scenario, you might want to use a more sophisticated method
            if i == 0:
                opens.append(open_price)
                closes.append(open_price + (close_price - open_price) * 0.25)
            elif i == 1:
                opens.append(open_price + (close_price - open_price) * 0.25)
                closes.append(open_price + (close_price - open_price) * 0.5)
            elif i == 2:
                opens.append(open_price + (close_price - open_price) * 0.5)
                closes.append(open_price + (close_price - open_price) * 0.75)
            else:  # i == 3
                opens.append(open_price + (close_price - open_price) * 0.75)
                closes.append(close_price)
            
            # For high and low, we'll use the high and low of the 1-minute candle
            # In a real scenario, you might want to distribute these more realistically
            highs.append(high_price)
            lows.append(low_price)
            
            # Distribute volume evenly
            volumes.append(volume / 4)
    
    # Create the 15s DataFrame
    df_15s = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    return df_15s

def analyze_fft_forecast_price_15s(candles_1m):
    """
    Analyze FFT forecast price for 15-second timeframe derived from 1-minute data.
    """
    try:
        if not candles_1m:
            return {"error": "No 1m data provided"}
        
        # Convert 1m candles to DataFrame
        df_1m = pd.DataFrame(candles_1m)
        
        # Convert timestamp to datetime in GMT+2 timezone
        df_1m['timestamp'] = pd.to_datetime(df_1m['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
        
        # Generate 15s data from 1m data
        df_15s = generate_15s_data_from_1m(df_1m)
        
        if df_15s is None or df_15s.empty:
            return {"error": "Failed to generate 15s data"}
        
        # Convert DataFrame back to candles format
        candles_15s = []
        for idx, row in df_15s.iterrows():
            candles_15s.append({
                "time": int(row['timestamp'].timestamp()),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume']),
                "timeframe": "15s"
            })
        
        # Extract price data for FFT analysis
        close_prices = np.array([candle["close"] for candle in candles_15s], dtype=np.float64)
        current_price = close_prices[-1]
        
        # Use improved FFT for forecasting
        forecast_prices = improved_fft_forecast(candles_15s, forecast_periods=16)  # 4 minutes ahead (16 * 15s)
        forecast_target = forecast_prices[-1]
        
        # Calculate percentage difference to forecast target
        forecast_diff_pct = ((forecast_target - current_price) / current_price) * 100
        
        # Determine if forecast is up (positive)
        forecast_up = forecast_target > current_price
        
        return {
            "timeframe": "15s",
            "current_price": current_price,
            "forecast_target": forecast_target,
            "forecast_diff_pct": forecast_diff_pct,
            "forecast_up": forecast_up,
            "forecast_prices": forecast_prices.tolist()
        }
        
    except Exception as e:
        print(f"Error analyzing FFT forecast price (15s): {e}")
        return {"error": str(e)}

def analyze_momentum_15sec(candles_1m):
    """
    Analyze if momentum is positive for 15-second timeframe derived from 1-minute data.
    """
    try:
        if not candles_1m:
            return {"error": "No 1m data provided"}
        
        # Convert 1m candles to DataFrame
        df_1m = pd.DataFrame(candles_1m)
        
        # Convert timestamp to datetime in GMT+2 timezone
        df_1m['timestamp'] = pd.to_datetime(df_1m['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
        
        # Generate 15s data from 1m data
        df_15s = generate_15s_data_from_1m(df_1m)
        
        if df_15s is None or df_15s.empty:
            return {"error": "Failed to generate 15s data"}
        
        # Convert DataFrame back to candles format
        candles_15s = []
        for idx, row in df_15s.iterrows():
            candles_15s.append({
                "time": int(row['timestamp'].timestamp()),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume']),
                "timeframe": "15s"
            })
        
        # Calculate momentum for 15s
        momentum_positive, momentum_value, momentum_details = calculate_momentum(candles_15s, period=10)
        
        return {
            "timeframe": "15s",
            "momentum_positive": momentum_positive,
            "current_momentum": momentum_value,
            "period": momentum_details.get('period', 10),
            "momentum_strength": 'Strong' if abs(momentum_value) > 100 else 'Moderate' if abs(momentum_value) > 50 else 'Weak'
        }
        
    except Exception as e:
        print(f"Error analyzing momentum (15s): {e}")
        return {"error": str(e)}

def get_time_to_next_targets(current_time):
    """
    Calculates the duration in minutes/seconds to the exact next 1, 3, 5, 10, and 15-minute marks.
    """
    targets = [1, 3, 5, 10, 15]
    results = {}

    for interval in targets:
        # Calculate the next time mark by adding enough minutes to surpass the current minute,
        # then rounding down to the nearest interval
        total_minutes = current_time.hour * 60 + current_time.minute
        minutes_past_interval = total_minutes % interval
        minutes_to_add = interval - minutes_past_interval
        
        # If we are exactly at the interval mark, add a full interval
        if current_time.second == 0 and current_time.microsecond == 0 and minutes_past_interval == 0:
             minutes_to_add += interval

        next_target_time = current_time + datetime.timedelta(minutes=minutes_to_add)
        # Round the target time to the nearest minute/second mark for accuracy
        next_target_time = next_target_time.replace(second=0, microsecond=0)

        # Calculate the duration until the target
        duration = next_target_time - current_time
        total_seconds = duration.total_seconds()
        
        results[f"Next {interval}-min mark"] = {
            "Target Time": next_target_time.strftime("%H:%M:%S"),
            "Duration (seconds)": total_seconds,
            "Duration (minutes)": round(total_seconds / 60, 2)
        }
    return results

def analyze_fft_forecast_next_targets(candles_1m, candles_3m, candles_5m):
    """
    Analyze FFT forecast for next target timeframes (1, 3, 5, 10, 15 minutes).
    """
    try:
        # Get current local time
        current_time = datetime.datetime.now(LOCAL_TIMEZONE)
        
        # Calculate time to next targets
        time_targets = get_time_to_next_targets(current_time)
        
        # Analyze FFT for each timeframe
        fft_1m = analyze_fft_cycle_with_natural_argmin_argmax(candles_1m, '1m')
        fft_3m = analyze_fft_cycle_with_natural_argmin_argmax(candles_3m, '3m')
        fft_5m = analyze_fft_cycle_with_natural_argmin_argmax(candles_5m, '5m')
        
        # Check for errors
        if 'error' in fft_1m or 'error' in fft_3m or 'error' in fft_5m:
            return {
                "error": "Error in FFT analysis",
                "time_targets": time_targets,
                "fft_1m": fft_1m,
                "fft_3m": fft_3m,
                "fft_5m": fft_5m
            }
        
        # Determine if forecast is up for each timeframe
        forecast_up_1m = fft_1m['forecast_target'] > fft_1m['current_price']
        forecast_up_3m = fft_3m['forecast_target'] > fft_3m['current_price']
        forecast_up_5m = fft_5m['forecast_target'] > fft_5m['current_price']
        
        # Calculate forecast differences
        forecast_diff_1m = ((fft_1m['forecast_target'] - fft_1m['current_price']) / fft_1m['current_price']) * 100
        forecast_diff_3m = ((fft_3m['forecast_target'] - fft_3m['current_price']) / fft_3m['current_price']) * 100
        forecast_diff_5m = ((fft_5m['forecast_target'] - fft_5m['current_price']) / fft_5m['current_price']) * 100
        
        # Combine results
        results = {
            "current_time": current_time.strftime("%H:%M:%S"),
            "time_targets": time_targets,
            "fft_analysis": {
                "1m": {
                    "current_price": fft_1m['current_price'],
                    "forecast_target": fft_1m['forecast_target'],
                    "forecast_diff_pct": forecast_diff_1m,
                    "forecast_up": forecast_up_1m,
                    "time_to_target": time_targets["Next 1-min mark"]["Duration (seconds)"]
                },
                "3m": {
                    "current_price": fft_3m['current_price'],
                    "forecast_target": fft_3m['forecast_target'],
                    "forecast_diff_pct": forecast_diff_3m,
                    "forecast_up": forecast_up_3m,
                    "time_to_target": time_targets["Next 3-min mark"]["Duration (seconds)"]
                },
                "5m": {
                    "current_price": fft_5m['current_price'],
                    "forecast_target": fft_5m['forecast_target'],
                    "forecast_diff_pct": forecast_diff_5m,
                    "forecast_up": forecast_up_5m,
                    "time_to_target": time_targets["Next 5-min mark"]["Duration (seconds)"]
                }
            },
            "overall_forecast_up": forecast_up_1m and forecast_up_3m and forecast_up_5m
        }
        
        return results
        
    except Exception as e:
        print(f"Error analyzing FFT forecast next targets: {e}")
        return {"error": str(e)}

def analyze_time_distance_condition():
    """
    Analyze time distance condition based on next target marks.
    This replaces the close_below_sma200_15s condition.
    """
    try:
        # Get current local time
        current_time = datetime.datetime.now(LOCAL_TIMEZONE)
        
        # Calculate time to next targets
        time_targets = get_time_to_next_targets(current_time)
        
        # Define condition: Check if we are within 30 seconds of any target mark
        # This creates a timing-based entry condition
        condition_met = False
        
        for target_name, target_data in time_targets.items():
            # Check if we're within 30 seconds of the target
            if target_data["Duration (seconds)"] <= 30:
                condition_met = True
                break
        
        # Additional logic: Check if we're in the first 10 seconds after a target mark
        # This provides another timing window for entries
        if not condition_met:
            for target_name, target_data in time_targets.items():
                # Check if we're in the first 10 seconds after the target mark
                if target_data["Duration (seconds)"] >= (target_data["Duration (minutes)"] * 60 - 10):
                    condition_met = True
                    break
        
        return {
            "current_time": current_time.strftime("%H:%M:%S"),
            "time_targets": time_targets,
            "condition_met": condition_met,
            "condition_description": "Within 30 seconds before or 10 seconds after any target mark"
        }
        
    except Exception as e:
        print(f"Error analyzing time distance condition: {e}")
        return {"error": str(e)}

# Initialize lot size info
lot_size_info = get_symbol_lot_size_info(TRADE_SYMBOL)
min_trade_size = lot_size_info['minQty']
step_size = lot_size_info['stepSize']
print(f"Initialized {TRADE_SYMBOL} - Min Trade Size: {min_trade_size:.25f}, Step Size: {step_size:.25f}")

# Initialize trade state
position_open = False
initial_investment = Decimal('0.0')
asset_balance = Decimal('0.0')
entry_price = Decimal('0.0')
entry_datetime = None  # Added to track entry time

# Initial balance check
usdc_balance = get_balance('USDC')
asset_balance = get_balance(TRADE_SYMBOL.split('USDC')[0])
print("Trading Bot Initialized!")

# Main trading loop
try:
    while True:
        current_local_time = datetime.datetime.now()
        current_local_time_str = current_local_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\nCurrent Local Time: {current_local_time_str}")

        usdc_balance = get_balance('USDC')
        asset_balance = get_balance(TRADE_SYMBOL.split('USDC')[0])
        current_price = get_current_price()
        btc_value_in_usdc = asset_balance * current_price

        if btc_value_in_usdc > usdc_balance and not position_open:
            print(f"BTC Value in USDC ({btc_value_in_usdc:.25f}) > USDC Balance ({usdc_balance:.25f}). Entering in-trade mode.")
            position_open = True
            entry_price = get_average_entry_price()
            if entry_price > Decimal('0'):
                initial_investment = asset_balance * entry_price
                print(f"Estimated Initial Investment: {initial_investment:.25f} USDC based on last buy trade entry price {entry_price:.25f}")
                last_trade = get_last_buy_trade()
                if last_trade:
                    entry_datetime = datetime.datetime.fromtimestamp(last_trade['time'] / 1000)
                    print(f"Entry Datetime set from last trade: {entry_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    entry_datetime = current_local_time
                    print(f"No trade history found. Using current time as Entry Datetime: {entry_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                initial_investment = asset_balance * current_price
                print(f"No valid entry price found. Using current price {current_price:.25f} to estimate Initial Investment: {initial_investment:.25f} USDC")
                entry_price = current_price
                entry_datetime = current_local_time
                print(f"Entry Datetime set to current time: {entry_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

        # Fetch candles for all timeframes
        candle_map = fetch_candles_in_parallel(['1m', '3m', '5m'])
        
        # Also fetch 1m candles for 15s analysis
        candles_1m = get_candles(TRADE_SYMBOL, '1m', limit=1200)
        
        if not candle_map.get('1m'):
            print("Error: '1m' candles not fetched. Check API connectivity or symbol.")
        if current_price == Decimal('0.0'):
            print(f"Warning: Current {TRADE_SYMBOL} price is {current_price:.25f}. API may be failing.")

        # Initialize all condition results - Updated with new conditions
        conditions_status = {
            # FFT forecast next targets condition
            "fft_forecast_next_targets": False,
            
            # Conditions from second code (with some removed)
            "fft_forecast_price_15s": False,
            "time_distance_condition": False,  # New time distance condition (replaced close_below_sma200_15s)
            "momentum_positive_1m": False,
            "fft_forecast_up_1m": False,
            "fft_forecast_up_3m": False,
            "fft_forecast_up_5m": False,
            
            # Momentum condition for 15s
            "momentum_positive_15sec": False,
        }

        buy_volume, sell_volume = calculate_buy_sell_volume(candle_map)
        volume_ratios = calculate_volume_ratio(buy_volume, sell_volume)

        # Print timeframe details first
        print("\n" + "="*80)
        print("TIMEFRAME ANALYSIS")
        print("="*80)
        
        # Analyze FFT cycles with natural argmin/argmax values
        print("\n--- FFT Cycle Analysis with Natural Values ---")
        fft_cycles = analyze_fft_cycle_with_natural_values(
            candle_map.get('1m', []), 
            candle_map.get('3m', []), 
            candle_map.get('5m', [])
        )
        
        # Print timeframe details
        for timeframe in ['1m', '3m', '5m']:
            if timeframe in candle_map and candle_map[timeframe]:
                print(f"\n--- {timeframe} Timeframe Details ---")
                closes = [candle['close'] for candle in candle_map[timeframe]]
                current_close = Decimal(str(closes[-1]))
                high_tf = Decimal(str(np.nanmax([float(x) for x in closes])))
                low_tf = Decimal(str(np.nanmin([float(x) for x in closes])))
                min_threshold_tf, max_threshold_tf, avg_mtf, momentum_signal, _, _, _ = calculate_thresholds(closes, period=14, minimum_percentage=2, maximum_percentage=2)
                last_bottom, last_top, closest_reversal, closest_type = find_major_reversals(candle_map[timeframe], current_price, min_threshold_tf, max_threshold_tf)
                if closest_reversal is not None:
                    print(f"Most Recent Major Reversal Type: {closest_type}")
                    print(f"Last Major Reversal Found at Price: {closest_reversal:.25f}")
                else:
                    print("No Major Reversal Found")
                
                print(f"Current Close: {current_close:.25f}")
                print(f"Minimum Threshold: {min_threshold_tf:.25f}" if min_threshold_tf is not None else "Minimum Threshold: Not available")
                print(f"Maximum Threshold: {max_threshold_tf:.25f}" if max_threshold_tf is not None else "Maximum Threshold: Not available")
                print(f"Average MTF: {avg_mtf:.25f}" if avg_mtf is not None else "Average MTF: Not available")
                print(f"Momentum Signal: {momentum_signal:.25f}" if momentum_signal is not None else "Momentum Signal: Not available")
                print(f"Volume Bullish Ratio: {volume_ratios[timeframe]['buy_ratio']:.25f}%" if timeframe in volume_ratios else "Volume Bullish Ratio: Not available")
                print(f"Volume Bearish Ratio: {volume_ratios[timeframe]['sell_ratio']:.25f}%" if timeframe in volume_ratios else "Volume Bearish Ratio: Not available")
                print(f"Status: {volume_ratios[timeframe]['status']}" if timeframe in volume_ratios else "Status: Not available")
            else:
                print(f"\n--- {timeframe} --- No data available.")

        # Print conditions in order
        print("\n" + "="*80)
        print("TRADING CONDITIONS")
        print("="*80)
        
        # Condition 1: FFT Forecast Next Targets
        print("\n--- Condition 1: FFT Forecast Next Targets ---")
        fft_next_targets_result = analyze_fft_forecast_next_targets(
            candle_map.get('1m', []), 
            candle_map.get('3m', []), 
            candle_map.get('5m', [])
        )
        if 'error' not in fft_next_targets_result:
            conditions_status["fft_forecast_next_targets"] = fft_next_targets_result['overall_forecast_up']
            print(f"Current Time: {fft_next_targets_result['current_time']}")
            
            for target_name, target_data in fft_next_targets_result['time_targets'].items():
                print(f"{target_name}: {target_data['Target Time']} ({target_data['Duration (minutes)']} min)")
            
            print("\nFFT Analysis:")
            for tf, data in fft_next_targets_result['fft_analysis'].items():
                print(f"{tf} - Current: {data['current_price']:.2f}, Target: {data['forecast_target']:.2f}, Diff: {data['forecast_diff_pct']:.4f}%, Up: {data['forecast_up']}")
            
            print(f"\nOverall Forecast Up: {fft_next_targets_result['overall_forecast_up']}")
            print(f"Condition Met: {conditions_status['fft_forecast_next_targets']}")
        else:
            print(f"Error analyzing FFT forecast next targets: {fft_next_targets_result['error']}")
            print(f"Condition Met: {conditions_status['fft_forecast_next_targets']}")
        
        # Condition 2: FFT Forecast Price (15s)
        print("\n--- Condition 2: FFT Forecast Price (15s) ---")
        fft_15s_result = analyze_fft_forecast_price_15s(candles_1m)
        if 'error' not in fft_15s_result:
            conditions_status["fft_forecast_price_15s"] = fft_15s_result['forecast_up']
            print(f"Current Price: {fft_15s_result['current_price']:.2f}")
            print(f"Forecast Target: {fft_15s_result['forecast_target']:.2f}")
            print(f"Forecast Difference: {fft_15s_result['forecast_diff_pct']:.4f}%")
            print(f"Forecast Direction: {'Upward' if fft_15s_result['forecast_up'] else 'Downward'}")
            print(f"Forecast Up: {fft_15s_result['forecast_up']}")
            print(f"Condition Met: {conditions_status['fft_forecast_price_15s']}")
        else:
            print(f"Error analyzing FFT forecast price (15s): {fft_15s_result['error']}")
            print(f"Condition Met: {conditions_status['fft_forecast_price_15s']}")
        
        # Condition 3: Time Distance Condition (replaced close_below_sma200_15s)
        print("\n--- Condition 3: Time Distance Condition ---")
        time_distance_result = analyze_time_distance_condition()
        if 'error' not in time_distance_result:
            conditions_status["time_distance_condition"] = time_distance_result['condition_met']
            print(f"Current Time: {time_distance_result['current_time']}")
            
            for target_name, target_data in time_distance_result['time_targets'].items():
                print(f"{target_name}: {target_data['Target Time']} ({target_data['Duration (minutes)']} min)")
            
            print(f"\nCondition Description: {time_distance_result['condition_description']}")
            print(f"Condition Met: {time_distance_result['condition_met']}")
            print(f"Condition Met: {conditions_status['time_distance_condition']}")
        else:
            print(f"Error analyzing time distance condition: {time_distance_result['error']}")
            print(f"Condition Met: {conditions_status['time_distance_condition']}")
        
        # Condition 4: Momentum Positive (1m)
        print("\n--- Condition 4: Momentum Positive (1m) ---")
        momentum_1m_positive, momentum_1m_value, momentum_1m_details = calculate_momentum(candle_map['1m'])
        conditions_status["momentum_positive_1m"] = momentum_1m_positive
        print(f"Current Momentum: {momentum_1m_value:.4f}")
        print(f"Momentum Period: {momentum_1m_details.get('period', 10)}")
        print(f"Momentum Direction: {'Positive' if momentum_1m_positive else 'Negative'}")
        print(f"Momentum Strength: {'Strong' if abs(momentum_1m_value) > 100 else 'Moderate' if abs(momentum_1m_value) > 50 else 'Weak'}")
        print(f"Momentum Positive: {momentum_1m_positive}")
        print(f"Condition Met: {conditions_status['momentum_positive_1m']}")
        
        # Condition 5: FFT Forecast Up (1m)
        print("\n--- Condition 5: FFT Forecast Up (1m) ---")
        if '1m' in fft_cycles and 'error' not in fft_cycles['1m']:
            fft_1m_result = fft_cycles['1m']
            conditions_status["fft_forecast_up_1m"] = fft_1m_result['forecast_target'] > fft_1m_result['current_price']
            print(f"Current Price: {fft_1m_result['current_price']:.2f}")
            print(f"Forecast Target: {fft_1m_result['forecast_target']:.2f}")
            print(f"Forecast Difference: {fft_1m_result['forecast_diff_pct']:.4f}%")
            print(f"Lowest Low: {fft_1m_result['lowest_low_price']:.2f}")
            print(f"Highest High: {fft_1m_result['highest_high_price']:.2f}")
            print(f"Cycle Direction: {fft_1m_result['cycle_direction']}")
            print(f"Dip More Recent: {fft_1m_result['dip_more_recent']}")
            print(f"Forecast Up: {conditions_status['fft_forecast_up_1m']}")
            print(f"Condition Met: {conditions_status['fft_forecast_up_1m']}")
        else:
            print(f"Error analyzing 1m FFT cycle: {fft_cycles.get('1m', {}).get('error', 'Unknown error')}")
            print(f"Condition Met: {conditions_status['fft_forecast_up_1m']}")
        
        # Condition 6: FFT Forecast Up (3m)
        print("\n--- Condition 6: FFT Forecast Up (3m) ---")
        if '3m' in fft_cycles and 'error' not in fft_cycles['3m']:
            fft_3m_result = fft_cycles['3m']
            conditions_status["fft_forecast_up_3m"] = fft_3m_result['forecast_target'] > fft_3m_result['current_price']
            print(f"Current Price: {fft_3m_result['current_price']:.2f}")
            print(f"Forecast Target: {fft_3m_result['forecast_target']:.2f}")
            print(f"Forecast Difference: {fft_3m_result['forecast_diff_pct']:.4f}%")
            print(f"Lowest Low: {fft_3m_result['lowest_low_price']:.2f}")
            print(f"Highest High: {fft_3m_result['highest_high_price']:.2f}")
            print(f"Cycle Direction: {fft_3m_result['cycle_direction']}")
            print(f"Dip More Recent: {fft_3m_result['dip_more_recent']}")
            print(f"Forecast Up: {conditions_status['fft_forecast_up_3m']}")
            print(f"Condition Met: {conditions_status['fft_forecast_up_3m']}")
        else:
            print(f"Error analyzing 3m FFT cycle: {fft_cycles.get('3m', {}).get('error', 'Unknown error')}")
            print(f"Condition Met: {conditions_status['fft_forecast_up_3m']}")
        
        # Condition 7: FFT Forecast Up (5m)
        print("\n--- Condition 7: FFT Forecast Up (5m) ---")
        if '5m' in fft_cycles and 'error' not in fft_cycles['5m']:
            fft_5m_result = fft_cycles['5m']
            conditions_status["fft_forecast_up_5m"] = fft_5m_result['forecast_target'] > fft_5m_result['current_price']
            print(f"Current Price: {fft_5m_result['current_price']:.2f}")
            print(f"Forecast Target: {fft_5m_result['forecast_target']:.2f}")
            print(f"Forecast Difference: {fft_5m_result['forecast_diff_pct']:.4f}%")
            print(f"Lowest Low: {fft_5m_result['lowest_low_price']:.2f}")
            print(f"Highest High: {fft_5m_result['highest_high_price']:.2f}")
            print(f"Cycle Direction: {fft_5m_result['cycle_direction']}")
            print(f"Dip More Recent: {fft_5m_result['dip_more_recent']}")
            print(f"Forecast Up: {conditions_status['fft_forecast_up_5m']}")
            print(f"Condition Met: {conditions_status['fft_forecast_up_5m']}")
        else:
            print(f"Error analyzing 5m FFT cycle: {fft_cycles.get('5m', {}).get('error', 'Unknown error')}")
            print(f"Condition Met: {conditions_status['fft_forecast_up_5m']}")
            
        # Condition 8: Momentum Positive (15s)
        print("\n--- Condition 8: Momentum Positive (15s) ---")
        momentum_15s_result = analyze_momentum_15sec(candles_1m)
        if 'error' not in momentum_15s_result:
            conditions_status["momentum_positive_15sec"] = momentum_15s_result['momentum_positive']
            print(f"Current Momentum: {momentum_15s_result['current_momentum']:.4f}")
            print(f"Momentum Period: {momentum_15s_result['period']}")
            print(f"Momentum Direction: {'Positive' if momentum_15s_result['momentum_positive'] else 'Negative'}")
            print(f"Momentum Strength: {momentum_15s_result['momentum_strength']}")
            print(f"Momentum Positive: {momentum_15s_result['momentum_positive']}")
            print(f"Condition Met: {conditions_status['momentum_positive_15sec']}")
        else:
            print(f"Error analyzing momentum (15s): {momentum_15s_result['error']}")
            print(f"Condition Met: {conditions_status['momentum_positive_15sec']}")

        # Print all conditions with true/false values
        print("\n" + "="*80)
        print("TRADING CONDITIONS STATUS")
        print("="*80)
        
        true_conditions_count = sum(int(status) for status in conditions_status.values())
        false_conditions_count = len(conditions_status) - true_conditions_count
        print(f"Overall Conditions Status: {true_conditions_count} True, {false_conditions_count} False")
        print(f"Minimum Required: {CONFIG['min_conditions_met']}")
        
        print("\nCondition Summary:")
        print("-" * 65)
        for condition_name, result in conditions_status.items():
            status = "TRUE" if result else "FALSE"
            print(f"{condition_name:<50}{status}")
        print("-" * 65)
        
        all_conditions_met = all(conditions_status.values())
        print(f"\nAll Conditions Met for Entry: {'Yes' if all_conditions_met else 'No'}")

        if position_open:
            print()
            print("Current In-Trade Status:")
            current_value_in_usdc = asset_balance * current_price
            if current_value_in_usdc < Decimal('0'):
                print("Error: Current BTC Balance Value in USDC is negative. Check balance or price.")
                current_value_in_usdc = Decimal('0.0')
            print(f"Current BTC Balance Value in USDC: {current_value_in_usdc:.25f}")

            target_value = initial_investment * Decimal('1.0035')  # 0.35% profit target (changed from 0.65%)
            entry_time_str = entry_datetime.strftime("%H:%M") if entry_datetime else "Unknown"
            time_span = (current_local_time - entry_datetime) if entry_datetime else None
            if time_span:
                total_seconds = int(time_span.total_seconds())
                days = total_seconds // (24 * 3600)
                hours = (total_seconds % (24 * 3600)) // 3600
                minutes = (total_seconds % 3600) // 60
                time_span_str = f"{days} days, {hours} hours, {minutes} minutes"
            else:
                time_span_str = "Unknown"
            
            if initial_investment <= Decimal('0'):
                print("Error: Initial investment is zero or negative. Using default value for display.")
                initial_investment_display = Decimal('1.0')
            else:
                initial_investment_display = initial_investment
            print(f"Initial USDC amount: {initial_investment_display:.25f}, Expected USDC amount after exit: {target_value:.25f}, Entry Price for last BTC purchased: {entry_price:.25f}")
            print(f"Entry Time (HH:MM): {entry_time_str}, Time Span from Entry: {time_span_str}")

            # Value Change Percentage
            if initial_investment_display > Decimal('0'):
                value_change_percentage = ((current_value_in_usdc - initial_investment) / initial_investment) * Decimal('100')
            else:
                value_change_percentage = Decimal('0.0')
            print(f"Value Change Percentage from Initial Investment: {value_change_percentage:.25f}%")

            # Price for 0.35% Profit Target
            if asset_balance > Decimal('0'):
                target_price = target_value / asset_balance
            else:
                target_price = Decimal('0.0')
                print("Error: BTC balance is zero or negative. Target price set to 0.")
            print(f"Price for 0.35% Profit Target: {target_price:.25f}")

            # Percentage Price Differences
            if entry_price > Decimal('0') and target_price > entry_price:
                # Entry to Current percentage change
                entry_to_current_pct = ((current_price - entry_price) / entry_price) * Decimal('100')
                
                # Current to Target percentage change (what we need to gain)
                current_to_target_pct = ((target_price - current_price) / current_price) * Decimal('100')
                
                # Entry to Target percentage change (should be 0.35%)
                entry_to_target_pct = ((target_price - entry_price) / entry_price) * Decimal('100')
                
                print(f"Entry Price: {entry_price:.2f}")
                print(f"Current Price: {current_price:.2f}")
                print(f"Target Price: {target_price:.2f}")
                print(f"Entry to Current: {entry_to_current_pct:.2f}%")
                print(f"Current to Target: {current_to_target_pct:.2f}%")
                print(f"Entry to Target: {entry_to_target_pct:.2f}%")
                
            else:
                entry_to_current_pct = Decimal('0.0')
                current_to_target_pct = Decimal('0.0')
                entry_to_target_pct = Decimal('0.0')
                print("Error: Invalid entry or target price for percentage calculations.")

            print(f"Price Change from Entry: {entry_to_current_pct:.2f}%")
            print(f"Needed Gain to Target: {current_to_target_pct:.2f}%")

            print()

            if check_exit_condition(initial_investment, asset_balance, entry_price):
                print("Target profit of 0.35% reached or exceeded. Initiating exit...")
                if sell_asset(float(asset_balance)):
                    exit_usdc_balance = get_balance('USDC')
                    profit = exit_usdc_balance - initial_investment
                    profit_percentage = (profit / initial_investment) * Decimal('100') if initial_investment > Decimal('0') else Decimal('0.0')
                    print(f"Position closed. Sold BTC for USDC: {exit_usdc_balance:.25f}")
                    print(f"Trade log: Time: {current_local_time_str}, Entry Price: {entry_price:.25f}, Exit Balance: {exit_usdc_balance:.25f}, Profit: {profit:.25f} USDC, Profit Percentage: {profit_percentage:.25f}%")
                    position_open = False
                    initial_investment = Decimal('0.0')
                    asset_balance = Decimal('0.0')
                    entry_price = Decimal('0.0')
                    entry_datetime = None  # Reset entry datetime
        else:
            if usdc_balance > Decimal('0'):
                print(f"Current USDC balance found: {usdc_balance:.25f}")
            else:
                print("No USDC balance available.")
            print(f"Current BTC balance: {asset_balance:.25f} BTC")

            # Check if all conditions are met for entry
            if all_conditions_met:
                usdc_balance = get_balance('USDC')
                if usdc_balance > Decimal('0'):
                    print(f"\n!!! ALL {CONFIG['min_conditions_met']} CONDITIONS MET - EXECUTING TRADE !!!")
                    print(f"Trigger signal detected! Attempting to buy {TRADE_SYMBOL} with entire USDC balance: {usdc_balance:.25f} at price {current_price:.25f}")
                    entry_price, quantity_bought, entry_datetime, cost = buy_asset()
                    if entry_price is not None and quantity_bought is not None and cost is not None:
                        initial_investment = cost
                        print(f"BTC was bought at entry price of {entry_price:.25f} USDC for quantity: {quantity_bought:.25f} BTC, Cost: {cost:.25f} USDC")
                        print(f"Entry Datetime: {entry_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                        position_open = True
                        print(f"New position opened with {cost:.25f} USDC at price {entry_price:.25f}.")
                        usdc_balance = get_balance('USDC')
                        asset_balance = get_balance(TRADE_SYMBOL.split('USDC')[0])
                    else:
                        print("Error placing buy order.")
                else:
                    print("No USDC balance to invest in BTC.")
            else:
                print(f"\n!!! INSUFFICIENT CONDITIONS MET - NO TRADE EXECUTED !!!")
                print(f"Only {true_conditions_count}/{CONFIG['min_conditions_met']} conditions met.")
                print("Waiting for next iteration...")

        print(f"\nCurrent USDC balance: {usdc_balance:.25f}")
        print(f"Current BTC balance: {asset_balance:.25f} BTC")
        print(f"Current {TRADE_SYMBOL} price: {current_price:.25f}\n")

        del candle_map
        gc.collect()
        time.sleep(5)
except KeyboardInterrupt:
    print("\nBot stopped by user. Exiting gracefully...")
    # Save any important state before exiting
    if position_open:
        print(f"Position is currently open with entry price: {entry_price:.25f}")
        print(f"Current profit/loss: {((asset_balance * current_price) - initial_investment):.25f} USDC")
    print("Cleaning up resources...")
    gc.collect()
    print("Bot shutdown complete.")
except Exception as e:
    print(f"Unexpected error in main loop: {e}")
    print("Attempting to save state before exit...")
    gc.collect()
    print("Bot shutdown due to error.")