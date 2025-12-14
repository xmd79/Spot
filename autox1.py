import numpy as np
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import datetime
import time
import concurrent.futures
import talib
import gc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import scipy.fftpack as fftpack
import math
from decimal import Decimal, getcontext
import requests
import os
import pandas as pd
import warnings
from scipy.signal import hilbert, argrelextrema, find_peaks
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
PROFIT_TARGET_PERCENT = 1.25  # 1.25% profit target
TOTAL_FEE_PERCENT = 0.22  # Total fee percentage (0.1% for buy + 0.1% for sell + 0.02% buffer)
MIN_TRADE_AMOUNT = 10

# Technical Indicators Configuration
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Updated configurable conditions - Now 12 conditions
CONFIG = {
    "conditions": {
        # Original conditions from first code (with some removed)
        "ML_Forecasted_Price_over_Current_Close": True,  # ML Forecast Price over Current Close
        "fibo_forecast_target_5m": True,  # NEW: Fibonacci forecast target (5m) above current close
        "dip_confirmed_1m": True,  # Dip Confirmed (1m)
        "dip_confirmed_3m": True,  # Dip Confirmed (3m)
        "dip_confirmed_5m": True,  # Dip Confirmed (5m)
        "volume_bullish_1m": True,  # Volume Bullish (1m)
        
        # Conditions from second code (with some removed)
        "fft_forecast_price_15s": True,  # FFT Forecast Price (15s)
        "close_below_sma200_15s": True,  # Close Below SMA200 (15s)
        "momentum_positive_1m": True,  # Momentum Positive (1m)
        "fft_forecast_up_1m": True,  # FFT Forecast Up (1m)
        "fft_forecast_up_3m": True,  # FFT Forecast Up (3m)
        "fft_forecast_up_5m": True,  # FFT Forecast Up (5m)
    },
    "min_conditions_met": 12  # ALL 12 conditions must be met to trigger a trade
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
    target_value = initial_investment * Decimal('1.0125')  # 1.25% profit target
    target_price = target_value / asset_balance
    print(f"Exit Check: Current Price: {current_price:.25f}, Target Price: {target_price:.25f}, Current Value: {current_value:.25f}, Target Value: {target_value:.25f}")
    return current_price >= target_price

# Analysis Functions
def backtest_model(candles):
    closes = np.array([float(candle["close"]) for candle in candles], dtype=np.float64)
    X = np.arange(len(closes)).reshape(-1, 1)
    y = closes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = Decimal(str(mean_absolute_error(y_test, predictions)))
    return model, mae, predictions, y_test

def forecast_next_price(model, num_steps=1):
    last_index = model.n_features_in_
    future_steps = np.arange(last_index, last_index + num_steps).reshape(-1, 1)
    return [Decimal(str(price)) for price in model.predict(future_steps)]

def forecast_price_range(model, num_steps=10):
    """
    Forecast a range of prices using the ML model.
    Returns min and max forecasted prices.
    """
    last_index = model.n_features_in_
    future_steps = np.arange(last_index, last_index + num_steps).reshape(-1, 1)
    forecasted_prices = model.predict(future_steps)
    min_forecast = Decimal(str(np.min(forecasted_prices)))
    max_forecast = Decimal(str(np.max(forecasted_prices)))
    return min_forecast, max_forecast, [Decimal(str(price)) for price in forecasted_prices]

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

def scale_to_sine(close_prices):
    close_prices_np = np.array([float(x) for x in close_prices], dtype=np.float64)
    sine_wave, _ = talib.HT_SINE(close_prices_np)
    current_sine = Decimal(str(np.nan_to_num(sine_wave)[-1]))
    sine_wave_min = Decimal(str(np.nanmin(sine_wave)))
    sine_wave_max = Decimal(str(np.nanmax(sine_wave)))
    dist_from_close_to_min = ((current_sine - sine_wave_min) / (sine_wave_max - sine_wave_min)) * Decimal('100') if sine_wave_max != sine_wave_min else Decimal('0')
    dist_from_close_to_max = ((sine_wave_max - current_sine) / (sine_wave_max - sine_wave_min)) * Decimal('100') if sine_wave_max != sine_wave_min else Decimal('0')
    return dist_from_close_to_min, dist_from_close_to_max, current_sine

def calculate_spectral_analysis(prices):
    prices_np = np.array([float(x) for x in prices], dtype=np.float64)
    fft_result = np.fft.fft(prices_np)
    power_spectrum = np.abs(fft_result) ** 2
    frequencies = np.fft.fftfreq(len(prices_np))
    neg_freqs = [Decimal(str(f)) for f in frequencies[frequencies < 0]]
    neg_powers = [Decimal(str(p)) for p in power_spectrum[frequencies < 0]]
    pos_freqs = [Decimal(str(f)) for f in frequencies[frequencies >= 0]]
    pos_powers = [Decimal(str(p)) for p in power_spectrum[frequencies >= 0]]
    return neg_freqs, neg_powers, pos_freqs, pos_powers

def determine_market_sentiment(negative_freqs, negative_powers, positive_freqs, positive_powers, last_major_reversal_type, buy_volume, sell_volume):
    total_negative_power = sum(negative_powers)
    total_positive_power = sum(positive_powers)
    if last_major_reversal_type == 'DIP':
        return "Bullish" if buy_volume > sell_volume else "Accumulation"
    elif last_major_reversal_type == 'TOP':
        if sell_volume > buy_volume:
            return "Bearish" if total_positive_power > total_negative_power else "Distribution"
        return "Bullish"
    return "Neutral"

def calculate_45_degree_projection(last_bottom, last_top):
    if last_bottom is not None and last_top is not None:
        distance = last_top - last_bottom
        return last_top + distance
    return None

def calculate_wave_price(t, avg, min_price, max_price, omega, phi):
    t_dec = Decimal(str(t))
    avg_dec = Decimal(str(avg))
    min_price_dec = Decimal(str(min_price))
    max_price_dec = Decimal(str(max_price))
    omega_dec = Decimal(str(omega))
    phi_dec = Decimal(str(phi))
    amplitude = (max_price_dec - min_price_dec) / Decimal('2')
    return avg_dec + amplitude * Decimal(str(math.sin(float(omega_dec * t_dec + phi_dec))))

def calculate_independent_wave_price(current_price, avg, min_price, max_price, range_distance):
    current_price_dec = Decimal(str(current_price))
    avg_dec = Decimal(str(avg))
    min_price_dec = Decimal(str(min_price))
    max_price_dec = Decimal(str(max_price))
    range_distance_dec = Decimal(str(range_distance))
    noise = Decimal(str(np.random.uniform(-1, 1))) * range_distance_dec
    return avg_dec + (noise * (max_price_dec - min_price_dec) / Decimal('2'))

def find_specific_support_resistance(candle_map, min_threshold, max_threshold, current_close):
    support_levels = []
    resistance_levels = []
    current_close_dec = Decimal(str(current_close))
    for timeframe in candle_map:
        for candle in candle_map[timeframe]:
            close_dec = Decimal(str(candle["close"]))
            if close_dec < current_close_dec and close_dec >= min_threshold:
                support_levels.append((close_dec, Decimal(str(candle["volume"])), candle["time"]))
            if close_dec > current_close_dec and close_dec <= max_threshold:
                resistance_levels.append((close_dec, Decimal(str(candle["volume"])), candle["time"]))
    support_levels.sort(key=lambda x: x[1], reverse=True)
    resistance_levels.sort(key=lambda x: x[1], reverse=True)
    significant_support = support_levels[:3]
    significant_resistance = resistance_levels[:3]
    print("Most Significant Support Levels (Price, Volume):")
    for price, volume, _ in significant_support:
        print(f"Support Price: {price:.25f}, Volume: {volume:.25f}")
    print("Most Significant Resistance Levels (Price, Volume):")
    for price, volume, _ in significant_resistance:
        print(f"Resistance Price: {price:.25f}, Volume: {volume:.25f}")
    return [level[0] for level in significant_support], [level[0] for level in significant_resistance]

def calculate_stochastic_rsi(close_prices, length_rsi=14, length_stoch=14, smooth_k=3, smooth_d=3):
    close_prices_np = np.array([float(x) for x in close_prices], dtype=np.float64)
    rsi = talib.RSI(close_prices_np, timeperiod=length_rsi)
    min_rsi = talib.MIN(rsi, timeperiod=length_stoch)
    max_rsi = talib.MAX(rsi, timeperiod=length_stoch)
    stoch_k = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
    stoch_k_smooth = talib.EMA(stoch_k, timeperiod=smooth_k)
    stoch_d = talib.EMA(stoch_k_smooth, timeperiod=smooth_d)
    return [Decimal(str(k)) for k in stoch_k_smooth], [Decimal(str(d)) for d in stoch_d]

def calculate_bullish_bearish_volume_ratios(candle_map):
    volume_ratios = {}
    for timeframe, candles in candle_map.items():
        bullish_volume = sum(Decimal(str(candle["volume"])) for candle in candles if Decimal(str(candle["close"])) > Decimal(str(candle["open"])))
        bearish_volume = sum(Decimal(str(candle["volume"])) for candle in candles if Decimal(str(candle["close"])) < Decimal(str(candle["open"])))
        total_volume = bullish_volume + bearish_volume
        ratio = bullish_volume / bearish_volume if bearish_volume > Decimal('0') else Decimal('Infinity')
        volume_ratios[timeframe] = {
            "bullish_volume": bullish_volume,
            "bearish_volume": bearish_volume,
            "ratio": ratio,
            "status": "Bullish" if ratio > Decimal('1') else "Bearish" if ratio < Decimal('1') else "Neutral"
        }
        print(f"{timeframe} - Bullish Volume: {bullish_volume:.25f}, Bearish Volume: {bearish_volume:.25f}, Ratio: {ratio:.25f} ({volume_ratios[timeframe]['status']})")
    return volume_ratios

def analyze_volume_changes_over_time(candle_map):
    volume_trends = {}
    for timeframe, candles in candle_map.items():
        if len(candles) < 2:
            print(f"{timeframe} - Not enough data to analyze volume changes.")
            continue
        last_volume = Decimal(str(candles[-1]["volume"]))
        previous_volume = Decimal(str(candles[-2]["volume"]))
        volume_change = last_volume - previous_volume
        change_status = "Increasing" if volume_change > Decimal('0') else "Decreasing" if volume_change < Decimal('0') else "Stable"
        volume_trends[timeframe] = {
            "last_volume": last_volume,
            "previous_volume": previous_volume,
            "change": volume_change,
            "status": change_status
        }
        print(f"{timeframe} - Last Volume: {last_volume:.25f}, Previous Volume: {previous_volume:.25f}, Change: {volume_change:.25f} ({change_status})")
    return volume_trends

def calculate_fibonacci_levels_from_reversal(last_reversal, max_threshold, min_threshold, last_major_reversal_type, current_price):
    levels = {}
    current_price_dec = Decimal(str(current_price))
    if last_reversal is None:
        last_reversal = current_price_dec
        last_major_reversal_type = 'DIP'
    else:
        last_reversal = Decimal(str(last_reversal))
    if last_major_reversal_type == 'DIP':
        levels['0.0'] = last_reversal
        levels['1.0'] = Decimal(str(max_threshold)) if max_threshold is not None else current_price_dec * Decimal('1.05')
        levels['0.5'] = last_reversal + (levels['1.0'] - last_reversal) / Decimal('2')
    elif last_major_reversal_type == 'TOP':
        levels['0.0'] = last_reversal
        levels['1.0'] = Decimal(str(min_threshold)) if min_threshold is not None else current_price_dec * Decimal('0.95')
        levels['0.5'] = last_reversal - (last_reversal - levels['1.0']) / Decimal('2')
    return levels

def forecast_fibo_target_price(fib_levels):
    return fib_levels.get('1.0', None)

def forecast_volume_based_on_conditions(volume_ratios, min_threshold, current_price):
    current_price_dec = Decimal(str(current_price))
    if '1m' in volume_ratios and volume_ratios['1m']['buy_ratio'] > Decimal('50'):
        forecasted_price = current_price_dec + (current_price_dec * Decimal('0.0267'))
        print(f"Forecasting a bullish price increase to {forecasted_price:.25f}.")
        return forecasted_price
    elif '1m' in volume_ratios and volume_ratios['1m']['sell_ratio'] > Decimal('50'):
        forecasted_price = current_price_dec - (current_price_dec * Decimal('0.0267'))
        print(f"Forecasting a bearish price decrease to {forecasted_price:.25f}.")
        return forecasted_price
    print("No clear forecast direction based on volume ratios.")
    return None

def check_market_conditions_and_forecast(support_levels, resistance_levels, current_price):
    current_price_dec = Decimal(str(current_price))
    if not support_levels and not resistance_levels:
        print("No support or resistance levels found; trade cautiously.")
        return "No trading signals available."
    first_support = support_levels[0] if support_levels else None
    first_resistance = resistance_levels[0] if resistance_levels else None
    if first_support and current_price_dec < first_support:
        print(f"Current price {current_price_dec:.25f} is below support {first_support:.25f}.")
        return "Current price below key support level; consider selling."
    elif first_resistance and current_price_dec > first_resistance:
        print(f"Current price {current_price_dec:.25f} is above resistance {first_resistance:.25f}.")
        return "Current price above key resistance level; consider buying."
    print(f"Current price {current_price_dec:.25f} is within support {first_support} and resistance {first_resistance}.")
    return None

def get_target(closes, n_components, last_major_reversal_type, buy_volume, sell_volume):
    closes_np = np.array([float(x) for x in closes], dtype=np.float64)
    fft = fftpack.rfft(closes_np)
    frequencies = fftpack.rfftfreq(len(closes_np))
    idx = np.argsort(np.abs(fft))[::-1][:n_components]
    filtered_fft = np.zeros_like(fft)
    filtered_fft[idx] = fft[idx]
    filtered_signal = fftpack.irfft(filtered_fft)
    current_close = Decimal(str(closes_np[-1]))
    target_price = Decimal(str(np.nanmax(filtered_signal)))
    current_time = datetime.datetime.now()
    stop_loss = current_close - Decimal(str(np.std(closes_np)))
    market_mood = "Neutral"
    if last_major_reversal_type == 'DIP' and buy_volume > sell_volume:
        market_mood = "Bullish"
    elif last_major_reversal_type == 'TOP' and sell_volume > buy_volume:
        market_mood = "Bearish" if float(sell_volume) >= float(buy_volume) else "Choppy"
    return current_time, current_close, stop_loss, target_price, market_mood

def forecast_price_per_time_pythagorean(timeframe, candles, min_threshold, max_threshold, current_price, time_window_minutes, last_reversal, last_reversal_type):
    current_price_dec = Decimal(str(current_price))
    if min_threshold is None or max_threshold is None:
        min_threshold = current_price_dec * Decimal('0.95')
        max_threshold = current_price_dec * Decimal('1.05')
    else:
        min_threshold = Decimal(str(min_threshold))
        max_threshold = Decimal(str(max_threshold))
    threshold_range = max_threshold - min_threshold
    time_leg = Decimal(str(time_window_minutes))
    hypotenuse = Decimal(str(math.sqrt(float(time_leg**2 + threshold_range**2))))
    price_per_minute = threshold_range / time_leg if time_leg > Decimal('0') else Decimal('0.0')
    forecast_price = max_threshold
    if last_reversal_type == 'DIP' and last_reversal is not None:
        last_reversal = Decimal(str(last_reversal))
        dist_from_dip = current_price_dec - last_reversal
        if dist_from_dip >= threshold_range:
            print(f"Pump incoming detected for {timeframe}: Distance from DIP ({dist_from_dip:.25f}) >= Threshold Range ({threshold_range:.25f})")
            forecast_price = last_reversal + (threshold_range * Decimal('1.5'))
        else:
            forecast_price = max_threshold
    elif last_reversal is None:
        forecast_price = current_price_dec * Decimal('1.05')
    print(f"\n--- Fixed Pythagorean Forecast for {timeframe} ---")
    print(f"Min Threshold: {min_threshold:.25f}")
    print(f"Max Threshold: {max_threshold:.25f}")
    print(f"Threshold Range (Leg B): {threshold_range:.25f}")
    print(f"Time Window (Leg A): {time_leg:.25f} minutes")
    print(f"Hypotenuse (Price-Time Distance): {hypotenuse:.25f}")
    print(f"Base Price Change Rate: {price_per_minute:.25f} USDC per minute")
    if last_reversal_type == 'DIP' and last_reversal is not None:
        print(f"Last DIP Reversal: {last_reversal:.25f}")
        print(f"Distance from DIP to Current Price: {dist_from_dip:.25f}")
    print(f"Fixed Forecast Price: {forecast_price:.25f}")
    return forecast_price, price_per_minute

# NEW FUNCTIONS FROM SECOND CODE

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

def calculate_volume_dominance(candles):
    """
    Analyze volume dominance for specified timeframe.
    Returns whether bullish volume is greater than bearish volume.
    """
    try:
        if not candles:
            return {"error": "No data provided"}
        
        # Determine bullish vs bearish volume
        bullish_volume = 0.0
        bearish_volume = 0.0
        
        for i in range(1, len(candles)):
            if candles[i]["close"] > candles[i-1]["close"]:  # Bullish candle
                bullish_volume += candles[i]["volume"]
            elif candles[i]["close"] < candles[i-1]["close"]:  # Bearish candle
                bearish_volume += candles[i]["volume"]
            else:  # Doji - split volume
                bullish_volume += candles[i]["volume"] / 2
                bearish_volume += candles[i]["volume"] / 2
        
        # Calculate percentages
        total_volume = bullish_volume + bearish_volume
        if total_volume > 0:
            bullish_pct = (bullish_volume / total_volume) * 100
            bearish_pct = (bearish_volume / total_volume) * 100
        else:
            bullish_pct = 50.0
            bearish_pct = 50.0
        
        # Determine dominance
        if bullish_pct > bearish_pct:
            dominance = "Bullish"
            dominance_pct = bullish_pct
        elif bearish_pct > bullish_pct:
            dominance = "Bearish"
            dominance_pct = bearish_pct
        else:
            dominance = "Neutral"
            dominance_pct = 50.0
        
        # Determine if bullish volume is greater than bearish volume
        bullish_dominance = bullish_pct > bearish_pct
        
        return {
            "timeframe": candles[0]["timeframe"] if candles else "unknown",
            "bullish_pct": bullish_pct,
            "bearish_pct": bearish_pct,
            "dominance": dominance,
            "dominance_pct": dominance_pct,
            "bullish_dominance": bullish_dominance
        }
        
    except Exception as e:
        print(f"Error analyzing volume dominance: {e}")
        # Return default values when error occurs
        return {
            "timeframe": "unknown",
            "bullish_pct": 50.0,
            "bearish_pct": 50.0,
            "dominance": "Neutral",
            "dominance_pct": 50.0,
            "bullish_dominance": False
        }

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
        fft_freq = fftfreq(len(detrended))
        
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
        # Using the natural values as they occur in the market
        
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

def analyze_sma200_condition_15s(candles_1m):
    """
    Analyze if current close price is below SMA200 for 15-second timeframe.
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
        
        # Extract close prices
        close_prices = df_15s['close'].values.astype(float)
        
        # Calculate SMA200
        if len(close_prices) < 200:
            return {"error": "Not enough data for SMA200 calculation"}
        
        sma200 = talib.SMA(close_prices, timeperiod=200)
        current_sma200 = sma200[-1]
        current_close = close_prices[-1]
        
        # Check if current close is below SMA200
        close_below_sma200 = current_close < current_sma200
        
        return {
            "timeframe": "15s",
            "current_close": current_close,
            "current_sma200": current_sma200,
            "close_below_sma200": close_below_sma200,
            "difference": current_close - current_sma200,
            "difference_pct": ((current_close - current_sma200) / current_sma200) * 100
        }
        
    except Exception as e:
        print(f"Error analyzing SMA200 condition (15s): {e}")
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

        # Initialize all condition results
        conditions_status = {
            # Original conditions from first code (with some removed)
            "ML_Forecasted_Price_over_Current_Close": False,
            "fibo_forecast_target_5m": False,  # NEW: Fibonacci forecast target (5m) above current close
            "dip_confirmed_1m": False,
            "dip_confirmed_3m": False,
            "dip_confirmed_5m": False,
            "volume_bullish_1m": False,
            
            # Conditions from second code (with some removed)
            "fft_forecast_price_15s": False,
            "close_below_sma200_15s": False,
            "momentum_positive_1m": False,
            "fft_forecast_up_1m": False,
            "fft_forecast_up_3m": False,
            "fft_forecast_up_5m": False,
        }

        if "1m" in candle_map and candle_map['1m']:
            model, backtest_mae, last_predictions, actuals = backtest_model(candle_map["1m"])
            print(f"Backtest MAE: {backtest_mae:.25f}")
            forecasted_prices = forecast_next_price(model, num_steps=1)
            closes = [candle['close'] for candle in candle_map["1m"]]
            min_threshold, max_threshold, _, _, _, _, _ = calculate_thresholds(closes)
            adjusted_forecasted_price = min(max(forecasted_prices[-1], min_threshold), max_threshold) if min_threshold is not None and max_threshold is not None else forecasted_prices[-1]
            print(f"Forecasted Price: {adjusted_forecasted_price:.25f}")
            
            # Check ML_Forecasted_Price_over_Current_Close condition
            conditions_status["ML_Forecasted_Price_over_Current_Close"] = adjusted_forecasted_price > current_price
        else:
            min_threshold, max_threshold, adjusted_forecasted_price = None, None, None
            print("No 1m data available for forecasting.")

        buy_volume, sell_volume = calculate_buy_sell_volume(candle_map)
        volume_ratios = calculate_volume_ratio(buy_volume, sell_volume)
        support_levels, resistance_levels = find_specific_support_resistance(candle_map, min_threshold or current_price * Decimal('0.95'), max_threshold or current_price * Decimal('1.05'), current_price)
        volume_ratios_details = calculate_bullish_bearish_volume_ratios(candle_map)
        volume_trends_details = analyze_volume_changes_over_time(candle_map)

        fib_info = {}
        pythagorean_forecasts = {'1m': {'price': None, 'rate': None}, '3m': {'price': None, 'rate': None}, '5m': {'price': None, 'rate': None}}
        last_major_reversal_type = None

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
                min_threshold_tf, max_threshold_tf, avg_mtf, momentum_signal, _, percent_to_min_momentum, percent_to_max_momentum = calculate_thresholds(closes, period=14, minimum_percentage=2, maximum_percentage=2)
                last_bottom, last_top, closest_reversal, closest_type = find_major_reversals(candle_map[timeframe], current_price, min_threshold_tf, max_threshold_tf)
                if closest_reversal is not None:
                    print(f"Most Recent Major Reversal Type: {closest_type}")
                    print(f"Last Major Reversal Found at Price: {closest_reversal:.25f}")
                    last_major_reversal_type = closest_type
                else:
                    print("No Major Reversal Found")
                
                # Check dip conditions
                dip_confirmed = closest_type == 'DIP'
                conditions_status[f'dip_confirmed_{timeframe}'] = dip_confirmed
                
                # Check volume_bullish_1m condition
                if timeframe == '1m':
                    conditions_status["volume_bullish_1m"] = buy_volume.get('1m', [Decimal('0')])[-1] > sell_volume.get('1m', [Decimal('0')])[-1]
                
                valid_closes = np.array([float(c) for c in closes if not np.isnan(c) and c > 0], dtype=np.float64)
                sma_lengths = [5, 7, 9, 12]
                smas = {length: Decimal(str(talib.SMA(valid_closes, timeperiod=length)[-1])) for length in sma_lengths if not np.isnan(talib.SMA(valid_closes, timeperiod=length)[-1])}
                print("Simple Moving Averages:")
                for length, sma in smas.items():
                    print(f"SMA-{length}: {sma:.25f}, Current Close: {current_close:.25f} - {'Above' if current_close > sma else 'Below'}")
                if all(current_close > sma for sma in smas.values()):
                    print("SELL Signal: Current close is above all SMAs.")
                elif all(current_close < sma for sma in smas.values()):
                    print("BUY Signal: Current close is below all SMAs.")
                stoch_k, stoch_d = calculate_stochastic_rsi(closes)
                print(f"Current Stochastic K: {stoch_k[-1]:.25f}")
                print(f"Current Stochastic D: {stoch_d[-1]:.25f}")
                negative_freqs, negative_powers, positive_freqs, positive_powers = calculate_spectral_analysis(closes)
                market_sentiment = determine_market_sentiment(negative_freqs, negative_powers, positive_freqs, positive_powers, closest_type, buy_volume[timeframe][-1], sell_volume[timeframe][-1])
                print(f"Market Sentiment: {market_sentiment}")
                projected_price = calculate_45_degree_projection(last_bottom, last_top)
                print(f"Projected Price Using 45-Degree Angle: {projected_price:.25f}" if projected_price is not None else "No projection available")
                print(f"Current Close: {current_close:.25f}")
                print(f"Minimum Threshold: {min_threshold_tf:.25f}" if min_threshold_tf is not None else "Minimum Threshold: Not available")
                print(f"Maximum Threshold: {max_threshold_tf:.25f}" if max_threshold_tf is not None else "Maximum Threshold: Not available")
                print(f"Average MTF: {avg_mtf:.25f}" if avg_mtf is not None else "Average MTF: Not available")
                print(f"Momentum Signal: {momentum_signal:.25f}" if momentum_signal is not None else "Momentum Signal: Not available")
                print(f"Volume Bullish Ratio: {volume_ratios[timeframe]['buy_ratio']:.25f}%" if timeframe in volume_ratios else "Volume Bullish Ratio: Not available")
                print(f"Volume Bearish Ratio: {volume_ratios[timeframe]['sell_ratio']:.25f}%" if timeframe in volume_ratios else "Volume Bearish Ratio: Not available")
                print(f"Status: {volume_ratios[timeframe]['status']}" if timeframe in volume_ratios else "Status: Not available")
                avg = (min_threshold_tf + max_threshold_tf) / Decimal('2') if min_threshold_tf is not None and max_threshold_tf is not None else current_close
                wave_price = calculate_wave_price(len(closes), avg, min_threshold_tf or Decimal('0'), max_threshold_tf or Decimal('Infinity'), omega=Decimal('0.1'), phi=Decimal('0'))
                print(f"Calculated Wave Price: {wave_price:.25f}")
                independent_wave_price = calculate_independent_wave_price(current_price, avg, min_threshold_tf or Decimal('0'), max_threshold_tf or Decimal('Infinity'), range_distance=Decimal('0.1'))
                print(f"Calculated Independent Wave Price: {independent_wave_price:.25f}")
                current_time, entry_price_usdc, stop_loss, reversal_target, market_mood = get_target(closes, n_components=5, last_major_reversal_type=last_major_reversal_type, buy_volume=buy_volume[timeframe][-1], sell_volume=sell_volume[timeframe][-1])
                print(f"Current Time: {current_time}")
                print(f"Entry Price: {entry_price_usdc:.25f}")
                print(f"Stop Loss: {stop_loss:.25f}")
                print(f"Reversal Target: {reversal_target:.25f}")
                print(f"Market Mood (from FFT): {market_mood}")
                fib_levels = calculate_fibonacci_levels_from_reversal(closest_reversal, max_threshold_tf, min_threshold_tf, closest_type, current_price)
                fib_info[timeframe] = fib_levels
                print(f"Fibonacci Levels for {timeframe}:")
                for level, price in fib_levels.items():
                    print(f"Level {level}: {price:.25f}")
                fib_reversal_price = forecast_fibo_target_price(fib_info[timeframe])
                print(f"{timeframe} Incoming Fibonacci Reversal Target (Forecast): {fib_reversal_price:.25f}" if fib_reversal_price is not None else f"{timeframe} Incoming Fibonacci Reversal Target: price not available.")
                
                # Check fibo_forecast_target_5m condition for 5m timeframe
                if timeframe == '5m':
                    if fib_reversal_price is not None:
                        conditions_status["fibo_forecast_target_5m"] = fib_reversal_price > current_price
                        print(f"Fibonacci Forecast Target > Current Price: {conditions_status['fibo_forecast_target_5m']}")
                    else:
                        conditions_status["fibo_forecast_target_5m"] = False
                        print("Fibonacci Forecast Target not available")
                
                dist_to_min = ((current_price - low_tf) / (high_tf - low_tf)) * Decimal('100') if (high_tf - low_tf) != Decimal('0') else Decimal('0')
                dist_to_max = ((high_tf - current_price) / (high_tf - low_tf)) * Decimal('100') if (high_tf - low_tf) != Decimal('0') else Decimal('0')
                print(f"Distance from Current Close to Min Threshold ({low_tf:.25f}): {dist_to_min:.25f}%")
                print(f"Distance from Current Close to Max Threshold ({high_tf:.25f}): {dist_to_max:.25f}%")
                symmetrical_min_distance = (high_tf - current_price) / (high_tf - low_tf) * Decimal('100') if (high_tf - low_tf) != Decimal('0') else Decimal('0')
                symmetrical_max_distance = (current_price - low_tf) / (high_tf - low_tf) * Decimal('100') if (high_tf - low_tf) != Decimal('0') else Decimal('0')
                print(f"Normalized Distance to Min Threshold (Symmetrical): {symmetrical_max_distance:.25f}%")
                print(f"Normalized Distance to Max Threshold (Symmetrical): {symmetrical_min_distance:.25f}%")
                time_window = {'1m': 1, '3m': 3, '5m': 5}[timeframe]
                forecast_price, price_rate = forecast_price_per_time_pythagorean(timeframe, candle_map[timeframe], min_threshold_tf, max_threshold_tf, current_price, time_window, closest_reversal, closest_type)
                pythagorean_forecasts[timeframe] = {'price': forecast_price, 'rate': price_rate}
                print(f"Pythagorean Forecast Price for {timeframe}: {pythagorean_forecasts[timeframe]['price']:.25f}")
                print(f"Pythagorean Price Rate for {timeframe}: {pythagorean_forecasts[timeframe]['rate']:.25f} USDC/min")
            else:
                print(f"\n--- {timeframe} --- No data available.")
                fib_levels = calculate_fibonacci_levels_from_reversal(None, None, None, 'DIP', current_price)
                fib_info[timeframe] = fib_levels
                print(f"Fibonacci Levels for {timeframe} (Fallback):")
                for level, price in fib_levels.items():
                    print(f"Level {level}: {price:.25f}")
                time_window = {'1m': 1, '3m': 3, '5m': 5}[timeframe]
                forecast_price, price_rate = forecast_price_per_time_pythagorean(timeframe, [], None, None, current_price, time_window, None, 'DIP')
                pythagorean_forecasts[timeframe] = {'price': forecast_price, 'rate': price_rate}
                print(f"Pythagorean Forecast Price for {timeframe}: {pythagorean_forecasts[timeframe]['price']:.25f}")
                print(f"Pythagorean Price Rate for {timeframe}: {pythagorean_forecasts[timeframe]['rate']:.25f} USDC/min")

        # Print conditions in order
        print("\n" + "="*80)
        print("TRADING CONDITIONS")
        print("="*80)
        
        # Condition 1: ML Forecasted Price over Current Close
        print("\n--- Condition 1: ML Forecasted Price over Current Close ---")
        if "1m" in candle_map and candle_map['1m']:
            print(f"Current Price: {current_price:.25f}")
            print(f"Forecasted Price: {adjusted_forecasted_price:.25f}")
            price_diff = adjusted_forecasted_price - current_price
            price_diff_pct = (price_diff / current_price) * 100 if current_price > 0 else 0
            print(f"Price Difference: {price_diff:.25f} ({price_diff_pct:.4f}%)")
            print(f"Forecasted > Current: {conditions_status['ML_Forecasted_Price_over_Current_Close']}")
            print(f"Condition Met: {conditions_status['ML_Forecasted_Price_over_Current_Close']}")
        else:
            print("No 1m data available for ML forecast.")
            print(f"Condition Met: {conditions_status['ML_Forecasted_Price_over_Current_Close']}")
        
        # Condition 2: Fibonacci Forecast Target (5m) above Current Close
        print("\n--- Condition 2: Fibonacci Forecast Target (5m) above Current Close ---")
        if '5m' in candle_map and candle_map['5m']:
            fib_5m_levels = fib_info.get('5m', {})
            fib_5m_target = forecast_fibo_target_price(fib_5m_levels)
            if fib_5m_target is not None:
                print(f"Current Price: {current_price:.25f}")
                print(f"Fibonacci Forecast Target (5m): {fib_5m_target:.25f}")
                price_diff = fib_5m_target - current_price
                price_diff_pct = (price_diff / current_price) * 100 if current_price > 0 else 0
                print(f"Price Difference: {price_diff:.25f} ({price_diff_pct:.4f}%)")
                print(f"Fibonacci Target > Current: {conditions_status['fibo_forecast_target_5m']}")
                print(f"Condition Met: {conditions_status['fibo_forecast_target_5m']}")
            else:
                print("Fibonacci forecast target not available for 5m timeframe.")
                print(f"Condition Met: {conditions_status['fibo_forecast_target_5m']}")
        else:
            print("No 5m data available for Fibonacci forecast.")
            print(f"Condition Met: {conditions_status['fibo_forecast_target_5m']}")
        
        # Condition 3: Dip Confirmed (1m)
        print("\n--- Condition 3: Dip Confirmed (1m) ---")
        if '1m' in candle_map and candle_map['1m']:
            closes = [candle['close'] for candle in candle_map['1m']]
            min_threshold_tf, max_threshold_tf, _, _, _, _, _ = calculate_thresholds(closes, period=14, minimum_percentage=2, maximum_percentage=2)
            last_bottom, last_top, closest_reversal, closest_type = find_major_reversals(candle_map['1m'], current_price, min_threshold_tf, max_threshold_tf)
            print(f"Most Recent Reversal Type: {closest_type if closest_type else 'None'}")
            print(f"Reversal Price: {closest_reversal:.25f}" if closest_reversal else "No reversal found")
            print(f"Current Price vs Reversal: {'Above dip' if closest_type == 'DIP' and current_price > closest_reversal else 'Below top' if closest_type == 'TOP' and current_price < closest_reversal else 'Neutral'}")
            print(f"Dip Confirmed: {conditions_status['dip_confirmed_1m']}")
            print(f"Condition Met: {conditions_status['dip_confirmed_1m']}")
        else:
            print("No 1m data available for dip confirmation.")
            print(f"Condition Met: {conditions_status['dip_confirmed_1m']}")
        
        # Condition 4: Dip Confirmed (3m)
        print("\n--- Condition 4: Dip Confirmed (3m) ---")
        if '3m' in candle_map and candle_map['3m']:
            closes = [candle['close'] for candle in candle_map['3m']]
            min_threshold_tf, max_threshold_tf, _, _, _, _, _ = calculate_thresholds(closes, period=14, minimum_percentage=2, maximum_percentage=2)
            last_bottom, last_top, closest_reversal, closest_type = find_major_reversals(candle_map['3m'], current_price, min_threshold_tf, max_threshold_tf)
            print(f"Most Recent Reversal Type: {closest_type if closest_type else 'None'}")
            print(f"Reversal Price: {closest_reversal:.25f}" if closest_reversal else "No reversal found")
            print(f"Current Price vs Reversal: {'Above dip' if closest_type == 'DIP' and current_price > closest_reversal else 'Below top' if closest_type == 'TOP' and current_price < closest_reversal else 'Neutral'}")
            print(f"Dip Confirmed: {conditions_status['dip_confirmed_3m']}")
            print(f"Condition Met: {conditions_status['dip_confirmed_3m']}")
        else:
            print("No 3m data available for dip confirmation.")
            print(f"Condition Met: {conditions_status['dip_confirmed_3m']}")
        
        # Condition 5: Dip Confirmed (5m)
        print("\n--- Condition 5: Dip Confirmed (5m) ---")
        if '5m' in candle_map and candle_map['5m']:
            closes = [candle['close'] for candle in candle_map['5m']]
            min_threshold_tf, max_threshold_tf, _, _, _, _, _ = calculate_thresholds(closes, period=14, minimum_percentage=2, maximum_percentage=2)
            last_bottom, last_top, closest_reversal, closest_type = find_major_reversals(candle_map['5m'], current_price, min_threshold_tf, max_threshold_tf)
            print(f"Most Recent Reversal Type: {closest_type if closest_type else 'None'}")
            print(f"Reversal Price: {closest_reversal:.25f}" if closest_reversal else "No reversal found")
            print(f"Current Price vs Reversal: {'Above dip' if closest_type == 'DIP' and current_price > closest_reversal else 'Below top' if closest_type == 'TOP' and current_price < closest_reversal else 'Neutral'}")
            print(f"Dip Confirmed: {conditions_status['dip_confirmed_5m']}")
            print(f"Condition Met: {conditions_status['dip_confirmed_5m']}")
        else:
            print("No 5m data available for dip confirmation.")
            print(f"Condition Met: {conditions_status['dip_confirmed_5m']}")
        
        # Condition 6: Volume Bullish (1m)
        print("\n--- Condition 6: Volume Bullish (1m) ---")
        if '1m' in volume_ratios:
            bullish_vol = buy_volume.get('1m', [Decimal('0')])[-1]
            bearish_vol = sell_volume.get('1m', [Decimal('0')])[-1]
            total_vol = bullish_vol + bearish_vol
            bullish_pct = (bullish_vol / total_vol * 100) if total_vol > 0 else 0
            bearish_pct = (bearish_vol / total_vol * 100) if total_vol > 0 else 0
            print(f"Bullish Volume: {bullish_vol:.25f} ({bullish_pct:.2f}%)")
            print(f"Bearish Volume: {bearish_vol:.25f} ({bearish_pct:.2f}%)")
            print(f"Total Volume: {total_vol:.25f}")
            print(f"Volume Ratio (Bullish/Bearish): {bullish_vol/bearish_vol:.4f}" if bearish_vol > 0 else "Infinity")
            print(f"Bullish > Bearish: {conditions_status['volume_bullish_1m']}")
            print(f"Condition Met: {conditions_status['volume_bullish_1m']}")
        else:
            print("No volume data available for 1m timeframe.")
            print(f"Condition Met: {conditions_status['volume_bullish_1m']}")
        
        # Condition 7: FFT Forecast Price (15s)
        print("\n--- Condition 7: FFT Forecast Price (15s) ---")
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
        
        # Condition 8: Close Below SMA200 (15s)
        print("\n--- Condition 8: Close Below SMA200 (15s) ---")
        sma200_15s_result = analyze_sma200_condition_15s(candles_1m)
        if 'error' not in sma200_15s_result:
            conditions_status["close_below_sma200_15s"] = sma200_15s_result['close_below_sma200']
            print(f"Current Close: {sma200_15s_result['current_close']:.2f}")
            print(f"Current SMA200: {sma200_15s_result['current_sma200']:.2f}")
            print(f"Difference: {sma200_15s_result['difference']:.4f} ({sma200_15s_result['difference_pct']:.4f}%)")
            print(f"Position Relative to SMA200: {'Below' if sma200_15s_result['close_below_sma200'] else 'Above'}")
            print(f"Close Below SMA200: {sma200_15s_result['close_below_sma200']}")
            print(f"Condition Met: {conditions_status['close_below_sma200_15s']}")
        else:
            print(f"Error analyzing SMA200 condition (15s): {sma200_15s_result['error']}")
            print(f"Condition Met: {conditions_status['close_below_sma200_15s']}")
        
        # Condition 9: Momentum Positive (1m)
        print("\n--- Condition 9: Momentum Positive (1m) ---")
        momentum_1m_positive, momentum_1m_value, momentum_1m_details = calculate_momentum(candle_map['1m'])
        conditions_status["momentum_positive_1m"] = momentum_1m_positive
        print(f"Current Momentum: {momentum_1m_value:.4f}")
        print(f"Momentum Period: {momentum_1m_details.get('period', 10)}")
        print(f"Momentum Direction: {'Positive' if momentum_1m_positive else 'Negative'}")
        print(f"Momentum Strength: {'Strong' if abs(momentum_1m_value) > 100 else 'Moderate' if abs(momentum_1m_value) > 50 else 'Weak'}")
        print(f"Momentum Positive: {momentum_1m_positive}")
        print(f"Condition Met: {conditions_status['momentum_positive_1m']}")
        
        # Condition 10: FFT Forecast Up (1m)
        print("\n--- Condition 10: FFT Forecast Up (1m) ---")
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
        
        # Condition 11: FFT Forecast Up (3m)
        print("\n--- Condition 11: FFT Forecast Up (3m) ---")
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
        
        # Condition 12: FFT Forecast Up (5m)
        print("\n--- Condition 12: FFT Forecast Up (5m) ---")
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

            target_value = initial_investment * Decimal('1.0125')  # 1.25% profit target
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

            # Price for 1.25% Profit Target
            if asset_balance > Decimal('0'):
                target_price = target_value / asset_balance
            else:
                target_price = Decimal('0.0')
                print("Error: BTC balance is zero or negative. Target price set to 0.")
            print(f"Price for 1.25% Profit Target: {target_price:.25f}")

            # Percentage Price Differences
            if entry_price > Decimal('0') and target_price > entry_price:
                # Entry to Current percentage change
                entry_to_current_pct = ((current_price - entry_price) / entry_price) * Decimal('100')
                
                # Current to Target percentage change (what we need to gain)
                current_to_target_pct = ((target_price - current_price) / current_price) * Decimal('100')
                
                # Entry to Target percentage change (should be 1.25%)
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
                print("Target profit of 1.25% reached or exceeded. Initiating exit...")
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
