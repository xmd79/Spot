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
import pandas as pd
import warnings
import os

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
client = BinanceClient(api_key, api_secret, requests_params={"timeout": 30})

# Trading Configuration
PROFIT_TARGET_PERCENT = 1.00  # 1.00% net profit target
TOTAL_FEE_PERCENT = 0.22  # Total fee percentage
MIN_TRADE_AMOUNT = 10

# CHANGE 1: Completely removed two conditions from CONFIG and updated min_conditions_met.
CONFIG = {
    "conditions": {
        "momentum_positive_1m": True,
        "momentum_positive_15sec": True,
        "linear_regression_channel_break": True,
        "aroon_only_signal": True,
        "volume_bias_condition": True,
        "thresholds_1m": True,
        "thresholds_3m": True,
        "thresholds_5m": True,
    },
    "min_conditions_met": 8  # Corrected to 8 active conditions
}

# Global variables for market state tracking
last_reversal_type = None
last_reversal_time = None
current_major_trend = "UNKNOWN"

# Variables for extrema tracking across timeframes
timeframe_extrema = {
    '1m': {'recent_high': None, 'recent_low': None, 'recent_high_idx': None, 'recent_low_idx': None},
    '3m': {'recent_high': None, 'recent_low': None, 'recent_high_idx': None, 'recent_low_idx': None},
    '5m': {'recent_high': None, 'recent_low': None, 'recent_high_idx': None, 'recent_low_idx': None}
}

# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def get_symbol_lot_size_info(symbol):
    try:
        exchange_info = client.get_symbol_info(symbol)
        for filter_info in exchange_info['filters']:
            if filter_info['filterType'] == 'LOT_SIZE':
                return {
                    'minQty': Decimal(str(filter_info['minQty'])),
                    'stepSize': Decimal(str(filter_info['stepSize']))
                }
        print(f"Could not find LOT_SIZE filter for {symbol}. Using defaults.")
        return {'minQty': Decimal('0.00001'), 'stepSize': Decimal('0.00001')}
    except BinanceAPIException as e:
        print(f"Error fetching symbol info for {symbol}: {e.message}")
        return {'minQty': Decimal('0.00001'), 'stepSize': Decimal('0.00001')}

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
                time.sleep(delay * (attempt + 1))
        except Exception as e:
            print(f"Unexpected error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    print(f"Failed to fetch candles for {timeframe} after {retries} attempts.")
    return []

def fetch_candles_in_parallel(timeframes, symbol=TRADE_SYMBOL, limit=1200):
    def fetch_candles(timeframe):
        return get_candles(symbol, timeframe, limit)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_candles, timeframes))
    return dict(zip(timeframes, results))

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
        except Exception as e:
            print(f"Unexpected error fetching price (attempt {attempt + 1}/{retries}): {e}")
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
        
        lot_size_info = get_symbol_lot_size_info(TRADE_SYMBOL)
        step_size = lot_size_info['stepSize']
        min_trade_size = lot_size_info['minQty']
        
        step_precision = int(-math.log10(float(step_size))) if step_size > Decimal('0') else 8
        adjusted_quantity = (raw_quantity // step_size) * step_size
        adjusted_quantity = adjusted_quantity.quantize(Decimal('0.' + '0' * step_precision))
        cost = adjusted_quantity * current_price
        print(f"Adjusted quantity (max balance): {adjusted_quantity:.25f}, Cost: {cost:.25f}")
        
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
        
        lot_size_info = get_symbol_lot_size_info(TRADE_SYMBOL)
        step_size = lot_size_info['stepSize']
        min_trade_size = lot_size_info['minQty']
        
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
    
    # To achieve 1% net profit after 0.22% fees, we need 1% + 0.22% = 1.22% gross profit
    target_value = initial_investment * Decimal('1.0122')  # 1.22% gross profit for 1% net profit
    target_price = target_value / asset_balance
    
    print(f"Exit Check: Current Price: {current_price:.25f}, Target Price: {target_price:.25f}, Current Value: {current_value:.25f}, Target Value: {target_value:.25f}")
    return current_price >= target_price

def save_signal_to_file(signal_data):
    try:
        if not os.path.exists("signals.txt"):
            with open("signals.txt", "w") as f:
                f.write("TIMESTAMP,SYMBOL,PRICE,SIGNAL,CONDITIONS_MET\n")
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        symbol = TRADE_SYMBOL
        price = signal_data.get("current_price", "N/A")
        signal = signal_data.get("signal", "N/A")
        conditions_met = signal_data.get("conditions_met", "N/A")
        
        with open("signals.txt", "a") as f:
            f.write(f"{timestamp},{symbol},{price},{signal},{conditions_met}\n")
        
        print(f"Signal saved to signals.txt: {timestamp} - {signal}")
    except Exception as e:
        print(f"Error saving signal to file: {e}")

# =========================================================
# VOLUME BIAS ANALYSIS FUNCTION
# =========================================================

def analyze_volume_bias(candles):
    try:
        if not candles or len(candles) < 20:
            return {"error": "Insufficient data for volume analysis", "condition_met": False}
        
        bullish_volume = 0
        bearish_volume = 0
        
        for candle in candles:
            if candle["close"] > candle["open"]:
                bullish_volume += candle["volume"]
            else:
                bearish_volume += candle["volume"]
        
        total_volume = bullish_volume + bearish_volume
        
        if total_volume == 0:
            return {"error": "No volume data", "condition_met": False}
        
        bullish_percentage = (bullish_volume / total_volume) * 100
        bearish_percentage = (bearish_volume / total_volume) * 100
        
        condition_met = bullish_percentage > bearish_percentage
        
        return {
            "condition_met": condition_met,
            "bullish_volume": bullish_volume,
            "bearish_volume": bearish_volume,
            "total_volume": total_volume,
            "bullish_percentage": bullish_percentage,
            "bearish_percentage": bearish_percentage,
            "volume_bias": "Bullish" if bullish_percentage > bearish_percentage else "Bearish"
        }
        
    except Exception as e:
        print(f"Error in analyze_volume_bias: {e}")
        return {"error": str(e), "condition_met": False}

# =========================================================
# NEW THRESHOLDS ANALYSIS FUNCTION
# =========================================================

def analyze_thresholds_by_timeframe(candles, timeframe):
    try:
        if not candles:
            return {"error": f"No {timeframe} data provided", "condition_met": False}
        
        close_prices = np.array([candle["close"] for candle in candles], dtype=np.float64)
        
        if len(close_prices) < 14:
            return {"error": f"Insufficient data for {timeframe} analysis", "condition_met": False}
        
        lookback = min(1200, len(close_prices))
        recent_closes = close_prices[-lookback:]
        
        min_idx = np.argmin(recent_closes)
        max_idx = np.argmax(recent_closes)
        
        actual_min_idx = len(close_prices) - lookback + min_idx
        actual_max_idx = len(close_prices) - lookback + max_idx
        
        min_value = recent_closes[min_idx]
        max_value = recent_closes[max_idx]
        
        current_price = close_prices[-1]
        
        hilo_range = max_value - min_value
        
        dist_to_min = current_price - min_value
        dist_to_max = max_value - current_price
        
        percent_to_min = (dist_to_min / hilo_range) * 100 if hilo_range > 0 else 0
        percent_to_max = (dist_to_max / hilo_range) * 100 if hilo_range > 0 else 0
        
        condition_met = dist_to_min < dist_to_max
        
        return {
            "timeframe": timeframe,
            "condition_met": condition_met,
            "current_price": current_price,
            "min_value": min_value,
            "max_value": max_value,
            "min_idx": actual_min_idx,
            "max_idx": actual_max_idx,
            "hilo_range": hilo_range,
            "dist_to_min": dist_to_min,
            "dist_to_max": dist_to_max,
            "percent_to_min": percent_to_min,
            "percent_to_max": percent_to_max,
            "description": f"Distance to min ({dist_to_min:.2f}) is {'less than' if condition_met else 'greater than'} distance to max ({dist_to_max:.2f})"
        }
        
    except Exception as e:
        print(f"Error analyzing thresholds ({timeframe}): {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "condition_met": False}

def analyze_thresholds_1m(candles_1m):
    return analyze_thresholds_by_timeframe(candles_1m, "1m")

def analyze_thresholds_3m(candles_3m):
    return analyze_thresholds_by_timeframe(candles_3m, "3m")

def analyze_thresholds_5m(candles_5m):
    return analyze_thresholds_by_timeframe(candles_5m, "5m")

# =========================================================
# ENHANCED TECHNICAL ANALYSIS FUNCTIONS
# =========================================================

def calculate_extrema_with_indices(prices, period=14):
    if len(prices) < period:
        return None, None, None, None, None, None
    
    recent_prices = prices[-period:]
    
    recent_high_idx = np.argmax(recent_prices)
    recent_low_idx = np.argmin(recent_prices)
    
    actual_high_idx = len(prices) - period + recent_high_idx
    actual_low_idx = len(prices) - period + recent_low_idx
    
    recent_high = prices[actual_high_idx]
    recent_low = prices[actual_low_idx]
    
    current_idx = len(prices) - 1
    periods_since_high = current_idx - actual_high_idx
    periods_since_low = current_idx - actual_low_idx
    
    return recent_high, recent_low, periods_since_high, periods_since_low, actual_high_idx, actual_low_idx

def enhanced_aroon(high, low, close, period=14):
    try:
        if len(high) < period or len(low) < period:
            return None, None, None, None, None, None
        
        recent_high, recent_low, periods_since_high, periods_since_low, recent_high_idx, recent_low_idx = calculate_extrema_with_indices(close, period)
        
        if recent_high is None or recent_low is None:
            return None, None, None, None, None, None
        
        aroon_up = ((period - periods_since_high) / period) * 100
        aroon_down = ((period - periods_since_low) / period) * 100
        
        return aroon_up, aroon_down, periods_since_high, periods_since_low, recent_high_idx, recent_low_idx
        
    except Exception as e:
        print(f"Error in enhanced_aroon: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None

def analyze_multiple_timeframe_extrema(candles_1m, candles_3m=None, candles_5m=None):
    global timeframe_extrema
    
    try:
        close_1m = [c["close"] for c in candles_1m]
        high_1m = [c["high"] for c in candles_1m]
        low_1m = [c["low"] for c in candles_1m]
        
        aroon_1m = enhanced_aroon(high_1m, low_1m, close_1m, period=14)
        
        if aroon_1m[0] is not None:
            recent_high = high_1m[aroon_1m[4]] if aroon_1m[4] is not None and 0 <= aroon_1m[4] < len(high_1m) else None
            recent_low = low_1m[aroon_1m[5]] if aroon_1m[5] is not None and 0 <= aroon_1m[5] < len(low_1m) else None
            
            timeframe_extrema['1m'] = {
                'recent_high': recent_high,
                'recent_low': recent_low,
                'recent_high_idx': aroon_1m[4],
                'recent_low_idx': aroon_1m[5],
                'aroon_up': aroon_1m[0],
                'aroon_down': aroon_1m[1]
            }
        
        if candles_3m and len(candles_3m) >= 14:
            close_3m = [c["close"] for c in candles_3m]
            high_3m = [c["high"] for c in candles_3m]
            low_3m = [c["low"] for c in candles_3m]
            
            aroon_3m = enhanced_aroon(high_3m, low_3m, close_3m, period=14)
            
            if aroon_3m[0] is not None:
                recent_high = high_3m[aroon_3m[4]] if aroon_3m[4] is not None and 0 <= aroon_3m[4] < len(high_3m) else None
                recent_low = low_3m[aroon_3m[5]] if aroon_3m[5] is not None and 0 <= aroon_3m[5] < len(low_3m) else None
                
                timeframe_extrema['3m'] = {
                    'recent_high': recent_high,
                    'recent_low': recent_low,
                    'recent_high_idx': aroon_3m[4],
                    'recent_low_idx': aroon_3m[5],
                    'aroon_up': aroon_3m[0],
                    'aroon_down': aroon_3m[1]
                }
        
        if candles_5m and len(candles_5m) >= 14:
            close_5m = [c["close"] for c in candles_5m]
            high_5m = [c["high"] for c in candles_5m]
            low_5m = [c["low"] for c in candles_5m]
            
            aroon_5m = enhanced_aroon(high_5m, low_5m, close_5m, period=14)
            
            if aroon_5m[0] is not None:
                recent_high = high_5m[aroon_5m[4]] if aroon_5m[4] is not None and 0 <= aroon_5m[4] < len(high_5m) else None
                recent_low = low_5m[aroon_5m[5]] if aroon_5m[5] is not None and 0 <= aroon_5m[5] < len(low_5m) else None
                
                timeframe_extrema['5m'] = {
                    'recent_high': recent_high,
                    'recent_low': recent_low,
                    'recent_high_idx': aroon_5m[4],
                    'recent_low_idx': aroon_5m[5],
                    'aroon_up': aroon_5m[0],
                    'aroon_down': aroon_5m[1]
                }
        
        return timeframe_extrema
        
    except Exception as e:
        print(f"Error in analyze_multiple_timeframe_extrema: {e}")
        import traceback
        traceback.print_exc()
        return timeframe_extrema

def calculate_momentum(candles, period=10):
    try:
        if not candles or len(candles) < period + 1:
            return None, 0.0, {"error": "Insufficient data for momentum analysis"}
        
        close_prices = np.array([candle["close"] for candle in candles], dtype=np.float64)
        
        momentum = np.zeros(len(close_prices))
        for i in range(period, len(close_prices)):
            momentum[i] = close_prices[i] - close_prices[i - period]
        
        current_momentum = float(momentum[-1])
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

def generate_15s_data_from_1m(df_1m):
    if df_1m is None or df_1m.empty:
        return None
    
    df_15s = pd.DataFrame()
    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    for idx, row in df_1m.iterrows():
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']
        volume = row['volume']
        
        for i in range(4):
            timestamp = row['timestamp'] + datetime.timedelta(seconds=15 * i)
            timestamps.append(timestamp)
            
            if i == 0:
                opens.append(open_price)
                closes.append(open_price + (close_price - open_price) * 0.25)
            elif i == 1:
                opens.append(open_price + (close_price - open_price) * 0.25)
                closes.append(open_price + (close_price - open_price) * 0.5)
            elif i == 2:
                opens.append(open_price + (close_price - open_price) * 0.5)
                closes.append(open_price + (close_price - open_price) * 0.75)
            else:
                opens.append(open_price + (close_price - open_price) * 0.75)
                closes.append(close_price)
            
            highs.append(high_price)
            lows.append(low_price)
            volumes.append(volume / 4)
    
    df_15s = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    return df_15s

def analyze_momentum_15sec(candles_1m):
    try:
        if not candles_1m:
            return {"error": "No 1m data provided"}
        
        df_1m = pd.DataFrame(candles_1m)
        df_1m['timestamp'] = pd.to_datetime(df_1m['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
        
        df_15s = generate_15s_data_from_1m(df_1m)
        if df_15s is None or df_15s.empty:
            return {"error": "Failed to generate 15s data"}
        
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

def calculate_linear_regression_channel(candles, period=500, dev_multiplier=2.0):
    try:
        if not candles or len(candles) < period:
            return {"error": "Insufficient data for linear regression channel"}
        
        close_prices = np.array([candle["close"] for candle in candles[-period:]], dtype=np.float64)
        n = len(close_prices)
        
        x = np.arange(n)
        coeffs = np.polyfit(x, close_prices, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        regression_line = slope * x + intercept
        residuals = close_prices - regression_line
        std_dev = np.std(residuals)
        
        upper_channel = regression_line + dev_multiplier * std_dev
        lower_channel = regression_line - dev_multiplier * std_dev
        
        current_price = close_prices[-1]
        current_upper = upper_channel[-1]
        current_lower = lower_channel[-1]
        
        above_upper = current_price > current_upper
        below_lower = current_price < current_lower
        
        below_lower_indices = []
        above_upper_indices = []
        
        for i in range(n):
            if close_prices[i] < lower_channel[i]:
                below_lower_indices.append(i)
            if close_prices[i] > upper_channel[i]:
                above_upper_indices.append(i)
        
        most_recent_below_lower = len(below_lower_indices) > 0
        most_recent_above_upper = len(above_upper_indices) > 0
        most_recent_below_lower_index = max(below_lower_indices) if below_lower_indices else None
        most_recent_above_upper_index = max(above_upper_indices) if above_upper_indices else None
        
        up_cycle = False
        if most_recent_below_lower and most_recent_above_upper:
            up_cycle = most_recent_below_lower_index > most_recent_above_upper_index
        elif most_recent_below_lower and not most_recent_above_upper:
            up_cycle = True
        elif not most_recent_below_lower and most_recent_above_upper:
            up_cycle = False
        
        return {
            "current_price": current_price,
            "current_upper": current_upper,
            "current_lower": current_lower,
            "above_upper": above_upper,
            "below_lower": below_lower,
            "most_recent_below_lower": most_recent_below_lower,
            "most_recent_above_upper": most_recent_above_upper,
            "most_recent_below_lower_index": most_recent_below_lower_index,
            "most_recent_above_upper_index": most_recent_above_upper_index,
            "up_cycle": up_cycle,
            "slope": slope,
            "intercept": intercept,
            "std_dev": std_dev,
            "period": period,
            "dev_multiplier": dev_multiplier,
            "below_lower_indices": below_lower_indices[-10:] if len(below_lower_indices) > 0 else [],
            "above_upper_indices": above_upper_indices[-10:] if len(above_upper_indices) > 0 else []
        }
        
    except Exception as e:
        print(f"Error calculating linear regression channel: {e}")
        return {"error": str(e)}

def analyze_linear_regression_channel_break(candles_1m):
    try:
        if not candles_1m:
            return {"error": "No 1m data provided"}
        
        lrc_result = calculate_linear_regression_channel(candles_1m, period=500, dev_multiplier=2.0)
        
        if "error" in lrc_result:
            return lrc_result
        
        below_lower_indices = lrc_result["below_lower_indices"]
        above_upper_indices = lrc_result["above_upper_indices"]
        
        most_recent_below_lower_index = max(below_lower_indices) if below_lower_indices else None
        most_recent_above_upper_index = max(above_upper_indices) if above_upper_indices else None
        
        condition_met = False
        description = "No channel breaks detected in the analysis window."
        
        is_most_recent_below_lower = False
        is_most_recent_above_upper = False
        
        if most_recent_below_lower_index is not None and most_recent_above_upper_index is not None:
            if most_recent_below_lower_index > most_recent_above_upper_index:
                condition_met = True
                description = "Most recent occurrence was price below the lower channel (indicating upward cycle)"
                is_most_recent_below_lower = True
            else:
                condition_met = False
                description = "Most recent occurrence was price above the upper channel (indicating downward cycle)"
                is_most_recent_above_upper = True
        elif most_recent_below_lower_index is not None:
            condition_met = True
            description = "Most recent occurrence was price below the lower channel (indicating upward cycle)"
            is_most_recent_below_lower = True
        elif most_recent_above_upper_index is not None:
            condition_met = False
            description = "Most recent occurrence was price above the upper channel (indicating downward cycle)"
            is_most_recent_above_upper = True
        
        return {
            "condition_met": condition_met,
            "current_price": lrc_result["current_price"],
            "upper_channel": lrc_result["current_upper"],
            "lower_channel": lrc_result["current_lower"],
            "above_upper": lrc_result["above_upper"],
            "below_lower": lrc_result["below_lower"],
            "most_recent_below_lower": is_most_recent_below_lower,
            "most_recent_above_upper": is_most_recent_above_upper,
            "most_recent_below_lower_index": most_recent_below_lower_index,
            "most_recent_above_upper_index": most_recent_above_upper_index,
            "up_cycle": condition_met,
            "description": description,
            "below_lower_indices": below_lower_indices[-10:] if len(below_lower_indices) > 0 else [],
            "above_upper_indices": above_upper_indices[-10:] if len(above_upper_indices) > 0 else []
        }
        
    except Exception as e:
        print(f"Error analyzing linear regression channel break: {e}")
        return {"error": str(e)}

# =========================================================
# AROON-ONLY SIGNAL ANALYSIS FUNCTION
# =========================================================
def analyze_aroon_only_signal(candles, aroon_period=14):
    """
    Generates a definitive "UP" or "DOWN" signal based solely on the Aroon indicator.
    - If Aroon Up > Aroon Down, signal is "UP" and condition_met is True.
    - If Aroon Down > Aroon Up, signal is "DOWN" and condition_met is False.
    """
    try:
        if not candles or len(candles) < aroon_period + 1:
            return {"error": "Insufficient data for Aroon analysis", "condition_met": False, "signal": "DOWN"}

        close = [c["close"] for c in candles]
        high = [c["high"] for c in candles]
        low = [c["low"] for c in candles]
        current_price = close[-1]
        
        aroon_result = enhanced_aroon(high, low, close, aroon_period)
        
        if aroon_result[0] is None:
            return {"error": "Failed to calculate Aroon values", "condition_met": False, "signal": "DOWN"}
        
        aroon_up, aroon_down, _, _, recent_high_idx, recent_low_idx = aroon_result
        
        # The core logic: determine signal based on Aroon comparison
        if aroon_up > aroon_down:
            signal = "UP"
            condition_met = True
        else:
            signal = "DOWN"
            condition_met = False
        
        recent_high = high[recent_high_idx] if recent_high_idx is not None and 0 <= recent_high_idx < len(high) else None
        recent_low = low[recent_low_idx] if recent_low_idx is not None and 0 <= recent_low_idx < len(low) else None
        
        return {
            "signal": signal,
            "condition_met": condition_met,
            "signal_reason": f"Aroon Up ({aroon_up:.2f}) is {'greater than' if condition_met else 'less than'} Aroon Down ({aroon_down:.2f})",
            "aroon_up": round(aroon_up, 2),
            "aroon_down": round(aroon_down, 2),
            "current_price": round(current_price, 2),
            "recent_high": round(recent_high, 2) if recent_high is not None else None,
            "recent_low": round(recent_low, 2) if recent_low is not None else None,
        }
        
    except Exception as e:
        print(f"Error in analyze_aroon_only_signal: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "condition_met": False, "signal": "DOWN"}

# =========================================================
# MAIN TRADING LOOP
# =========================================================

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
entry_datetime = None

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

        # Fetch candles for multiple timeframes
        candle_map = fetch_candles_in_parallel(['1m', '3m', '5m'])
        candles_1m = candle_map.get('1m', [])
        
        if not candles_1m:
            print("Error: '1m' candles not fetched. Check API connectivity or symbol.")
        if current_price == Decimal('0.0'):
            print(f"Warning: Current {TRADE_SYMBOL} price is {current_price:.25f}. API may be failing.")

        # Initialize all condition results
        conditions_status = {
            "momentum_positive_1m": False,
            "momentum_positive_15sec": False,
            "linear_regression_channel_break": False,
            "aroon_only_signal": False,
            "volume_bias_condition": False,
            "thresholds_1m": False,
            "thresholds_3m": False,
            "thresholds_5m": False,
        }

        # Print conditions in order
        print("\n" + "="*80)
        print("TRADING CONDITIONS")
        print("="*80)
        
        # Condition 1: Momentum Positive (1m)
        print("\n--- Condition 1: Momentum Positive (1m) ---")
        momentum_1m_positive, momentum_1m_value, momentum_1m_details = calculate_momentum(candles_1m)
        conditions_status["momentum_positive_1m"] = momentum_1m_positive
        print(f"Current Momentum: {momentum_1m_value:.4f}")
        print(f"Momentum Period: {momentum_1m_details.get('period', 10)}")
        print(f"Momentum Direction: {'Positive' if momentum_1m_positive else 'Negative'}")
        print(f"Momentum Strength: {'Strong' if abs(momentum_1m_value) > 100 else 'Moderate' if abs(momentum_1m_value) > 50 else 'Weak'}")
        print(f"Condition Met: {conditions_status['momentum_positive_1m']}")
        
        # Condition 2: Momentum Positive (15s)
        print("\n--- Condition 2: Momentum Positive (15s) ---")
        momentum_15s_result = analyze_momentum_15sec(candles_1m)
        if 'error' not in momentum_15s_result:
            conditions_status["momentum_positive_15sec"] = momentum_15s_result['momentum_positive']
            print(f"Current Momentum: {momentum_15s_result['current_momentum']:.4f}")
            print(f"Momentum Period: {momentum_15s_result['period']}")
            print(f"Momentum Direction: {'Positive' if momentum_15s_result['momentum_positive'] else 'Negative'}")
            print(f"Momentum Strength: {momentum_15s_result['momentum_strength']}")
            print(f"Condition Met: {conditions_status['momentum_positive_15sec']}")
        else:
            print(f"Error analyzing momentum (15s): {momentum_15s_result['error']}")
            print(f"Condition Met: {conditions_status['momentum_positive_15sec']}")
        
        # Condition 3: Linear Regression Channel Break
        print("\n--- Condition 3: Linear Regression Channel Break ---")
        lrc_break_result = analyze_linear_regression_channel_break(candles_1m)
        if 'error' not in lrc_break_result:
            conditions_status["linear_regression_channel_break"] = lrc_break_result['condition_met']
            print(f"Current Price: {lrc_break_result['current_price']:.2f}")
            print(f"Upper Channel: {lrc_break_result['upper_channel']:.2f}")
            print(f"Lower Channel: {lrc_break_result['lower_channel']:.2f}")
            print(f"Most Recent Below Lower: {lrc_break_result['most_recent_below_lower']}")
            print(f"Most Recent Above Upper: {lrc_break_result['most_recent_above_upper']}")
            print(f"Up Cycle: {lrc_break_result['up_cycle']}")
            print(f"Description: {lrc_break_result['description']}")
            print(f"Condition Met: {conditions_status['linear_regression_channel_break']}")
        else:
            print(f"Error analyzing linear regression channel break: {lrc_break_result['error']}")
            print(f"Condition Met: {conditions_status['linear_regression_channel_break']}")
            
        # Condition 4: Aroon-Only Signal
        print("\n--- Condition 4: Aroon-Only Signal ---")
        aroon_signal_result = analyze_aroon_only_signal(candles_1m, aroon_period=14)
        if 'error' not in aroon_signal_result:
            conditions_status["aroon_only_signal"] = aroon_signal_result['condition_met']
            print(f"Signal: {aroon_signal_result['signal']}")
            print(f"Aroon Up: {aroon_signal_result['aroon_up']}")
            print(f"Aroon Down: {aroon_signal_result['aroon_down']}")
            print(f"Current Price: {aroon_signal_result['current_price']}")
            print(f"Recent High: {aroon_signal_result['recent_high']}")
            print(f"Recent Low: {aroon_signal_result['recent_low']}")
            print(f"Reason: {aroon_signal_result['signal_reason']}")
            print(f"Condition Met: {conditions_status['aroon_only_signal']}")
        else:
            print(f"Error analyzing Aroon signal: {aroon_signal_result['error']}")
            print(f"Condition Met: {conditions_status['aroon_only_signal']}")

        # Condition 5: Volume Bias Condition
        print("\n--- Condition 5: Volume Bias Condition ---")
        volume_bias_result = analyze_volume_bias(candles_1m)
        if 'error' not in volume_bias_result:
            conditions_status["volume_bias_condition"] = volume_bias_result['condition_met']
            print(f"Total Volume: {volume_bias_result['total_volume']:.2f}")
            print(f"Bullish Volume: {volume_bias_result['bullish_volume']:.2f} ({volume_bias_result['bullish_percentage']:.2f}%)")
            print(f"Bearish Volume: {volume_bias_result['bearish_volume']:.2f} ({volume_bias_result['bearish_percentage']:.2f}%)")
            print(f"Volume Bias: {volume_bias_result['volume_bias']}")
            print(f"Condition Met: {conditions_status['volume_bias_condition']}")
        else:
            print(f"Error analyzing volume bias: {volume_bias_result['error']}")
            print(f"Condition Met: {conditions_status['volume_bias_condition']}")

        # Condition 6: Thresholds Analysis (1m)
        print("\n--- Condition 6: Thresholds Analysis (1m) ---")
        thresholds_1m_result = analyze_thresholds_1m(candles_1m)
        if 'error' not in thresholds_1m_result:
            conditions_status["thresholds_1m"] = thresholds_1m_result['condition_met']
            print(f"Timeframe: {thresholds_1m_result['timeframe']}")
            print(f"Current Price: {thresholds_1m_result['current_price']:.2f}")
            print(f"Min Value: {thresholds_1m_result['min_value']:.2f} at index {thresholds_1m_result['min_idx']}")
            print(f"Max Value: {thresholds_1m_result['max_value']:.2f} at index {thresholds_1m_result['max_idx']}")
            print(f"HiLo Range: {thresholds_1m_result['hilo_range']:.2f}")
            print(f"Distance to Min: {thresholds_1m_result['dist_to_min']:.2f}")
            print(f"Distance to Max: {thresholds_1m_result['dist_to_max']:.2f}")
            print(f"Percent to Min: {thresholds_1m_result['percent_to_min']:.2f}%")
            print(f"Percent to Max: {thresholds_1m_result['percent_to_max']:.2f}%")
            print(f"Description: {thresholds_1m_result['description']}")
            print(f"Condition Met: {conditions_status['thresholds_1m']}")
        else:
            print(f"Error analyzing thresholds (1m): {thresholds_1m_result['error']}")
            print(f"Condition Met: {conditions_status['thresholds_1m']}")
            
        # Condition 7: Thresholds Analysis (3m)
        print("\n--- Condition 7: Thresholds Analysis (3m) ---")
        thresholds_3m_result = analyze_thresholds_3m(candle_map.get('3m', []))
        if 'error' not in thresholds_3m_result:
            conditions_status["thresholds_3m"] = thresholds_3m_result['condition_met']
            print(f"Timeframe: {thresholds_3m_result['timeframe']}")
            print(f"Current Price: {thresholds_3m_result['current_price']:.2f}")
            print(f"Min Value: {thresholds_3m_result['min_value']:.2f} at index {thresholds_3m_result['min_idx']}")
            print(f"Max Value: {thresholds_3m_result['max_value']:.2f} at index {thresholds_3m_result['max_idx']}")
            print(f"HiLo Range: {thresholds_3m_result['hilo_range']:.2f}")
            print(f"Distance to Min: {thresholds_3m_result['dist_to_min']:.2f}")
            print(f"Distance to Max: {thresholds_3m_result['dist_to_max']:.2f}")
            print(f"Percent to Min: {thresholds_3m_result['percent_to_min']:.2f}%")
            print(f"Percent to Max: {thresholds_3m_result['percent_to_max']:.2f}%")
            print(f"Description: {thresholds_3m_result['description']}")
            print(f"Condition Met: {conditions_status['thresholds_3m']}")
        else:
            print(f"Error analyzing thresholds (3m): {thresholds_3m_result['error']}")
            print(f"Condition Met: {conditions_status['thresholds_3m']}")
            
        # Condition 8: Thresholds Analysis (5m)
        print("\n--- Condition 8: Thresholds Analysis (5m) ---")
        thresholds_5m_result = analyze_thresholds_5m(candle_map.get('5m', []))
        if 'error' not in thresholds_5m_result:
            conditions_status["thresholds_5m"] = thresholds_5m_result['condition_met']
            print(f"Timeframe: {thresholds_5m_result['timeframe']}")
            print(f"Current Price: {thresholds_5m_result['current_price']:.2f}")
            print(f"Min Value: {thresholds_5m_result['min_value']:.2f} at index {thresholds_5m_result['min_idx']}")
            print(f"Max Value: {thresholds_5m_result['max_value']:.2f} at index {thresholds_5m_result['max_idx']}")
            print(f"HiLo Range: {thresholds_5m_result['hilo_range']:.2f}")
            print(f"Distance to Min: {thresholds_5m_result['dist_to_min']:.2f}")
            print(f"Distance to Max: {thresholds_5m_result['dist_to_max']:.2f}")
            print(f"Percent to Min: {thresholds_5m_result['percent_to_min']:.2f}%")
            print(f"Percent to Max: {thresholds_5m_result['percent_to_max']:.2f}%")
            print(f"Description: {thresholds_5m_result['description']}")
            print(f"Condition Met: {conditions_status['thresholds_5m']}")
        else:
            print(f"Error analyzing thresholds (5m): {thresholds_5m_result['error']}")
            print(f"Condition Met: {conditions_status['thresholds_5m']}")

        # Print all conditions with true/false values
        print("\n" + "="*80)
        print("TRADING CONDITIONS STATUS")
        print("="*80)
        
        # CHANGE 2 & 3: Get results only for active conditions and require ALL to be true.
        active_conditions_results = [
            conditions_status[condition_name]
            for condition_name in CONFIG['conditions']
        ]
        true_conditions_count = sum(int(status) for status in active_conditions_results)
        false_conditions_count = len(active_conditions_results) - true_conditions_count
        
        print(f"Overall Conditions Status: {true_conditions_count} True, {false_conditions_count} False")
        print(f"Total Active Conditions: {len(active_conditions_results)}")
        
        print("\nCondition Summary (Active):")
        print("-" * 65)
        for condition_name in CONFIG['conditions']:
            status = "TRUE" if conditions_status[condition_name] else "FALSE"
            print(f"{condition_name:<50}{status}")
        print("-" * 65)
        
        all_conditions_met = all(active_conditions_results)
        print(f"\nAll Active Conditions Met for Entry: {'Yes' if all_conditions_met else 'No'}")

        # Save signal to file if all conditions are met
        if all_conditions_met:
            signal_data = {
                "current_price": float(current_price),
                "signal": "BUY",
                "conditions_met": f"{true_conditions_count}/{len(active_conditions_results)}"
            }
            save_signal_to_file(signal_data)

        if position_open:
            print()
            print("Current In-Trade Status:")
            current_value_in_usdc = asset_balance * current_price
            if current_value_in_usdc < Decimal('0'):
                print("Error: Current BTC Balance Value in USDC is negative. Check balance or price.")
                current_value_in_usdc = Decimal('0.0')
            print(f"Current BTC Balance Value in USDC: {current_value_in_usdc:.25f}")

            # Updated to use 1.0122 for 1% net profit after 0.22% fees
            target_value = initial_investment * Decimal('1.0122')
            entry_time_str = entry_datetime.strftime("%H:%M") if entry_datetime else "Unknown"
            time_span = (current_local_time - entry_datetime) if entry_datetime else None
            time_span_str = "Unknown"
            if time_span:
                total_seconds = int(time_span.total_seconds())
                days = total_seconds // (24 * 3600)
                hours = (total_seconds % (24 * 3600)) // 3600
                minutes = (total_seconds % 3600) // 60
                time_span_str = f"{days} days, {hours} hours, {minutes} minutes"
            
            if initial_investment <= Decimal('0'):
                print("Error: Initial investment is zero or negative. Using default value for display.")
                initial_investment_display = Decimal('1.0')
            else:
                initial_investment_display = initial_investment
            print(f"Initial USDC amount: {initial_investment_display:.25f}, Expected USDC amount after exit: {target_value:.25f}, Entry Price for last BTC purchased: {entry_price:.25f}")
            print(f"Entry Time (HH:MM): {entry_time_str}, Time Span from Entry: {time_span_str}")

            if initial_investment_display > Decimal('0'):
                value_change_percentage = ((current_value_in_usdc - initial_investment) / initial_investment) * Decimal('100')
            else:
                value_change_percentage = Decimal('0.0')
            print(f"Value Change Percentage from Initial Investment: {value_change_percentage:.25f}%")

            if asset_balance > Decimal('0'):
                target_price = target_value / asset_balance
            else:
                target_price = Decimal('0.0')
                print("Error: BTC balance is zero or negative. Target price set to 0.")
            # Updated print statement to reflect 1% net profit target
            print(f"Price for 1.00% Net Profit Target (after fees): {target_price:.25f}")

            if entry_price > Decimal('0') and target_price > entry_price:
                entry_to_current_pct = ((current_price - entry_price) / entry_price) * Decimal('100')
                current_to_target_pct = ((target_price - current_price) / current_price) * Decimal('100')
                entry_to_target_pct = ((target_price - entry_price) / entry_price) * Decimal('100')
                
                print(f"Entry Price: {entry_price:.2f}")
                print(f"Current Price: {current_price:.2f}")
                print(f"Target Price: {target_price:.2f}")
                print(f"Entry to Current: {entry_to_current_pct:.2f}%")
                print(f"Entry to Target: {entry_to_target_pct:.2f}%")
            else:
                print("Error: Invalid entry or target price for percentage calculations.")

            print(f"Price Change from Entry: {entry_to_current_pct:.2f}%")
            print(f"Needed Gain to Target: {current_to_target_pct:.2f}%")

            print()

            if check_exit_condition(initial_investment, asset_balance, entry_price):
                # Updated print statement to reflect 1% net profit target
                print("Target net profit of 1.00% (after fees) reached or exceeded. Initiating exit...")
                if sell_asset(float(asset_balance)):
                    exit_usdc_balance = get_balance('USDC')
                    profit = exit_usdc_balance - initial_investment
                    profit_percentage = (profit / initial_investment) * Decimal('100') if initial_investment > Decimal('0.0') else Decimal('0.0')
                    print(f"Position closed. Sold BTC for USDC: {exit_usdc_balance:.25f}")
                    print(f"Trade log: Time: {current_local_time_str}, Entry Price: {entry_price:.25f}, Exit Balance: {exit_usdc_balance:.25f}")
                    print(f"Trade log: Time: {current_local_time_str}, Entry Price: {entry_price:.25f}, Exit Balance: {exit_usdc_balance:.25f}, Profit: {profit:.25f} Net Profit Percentage: {profit_percentage:.25f}%")
                    position_open = False
                    initial_investment = Decimal('0.0')
                    asset_balance = Decimal('0.0')
                    entry_price = Decimal('0.0')
                    entry_datetime = None
        else:
            if usdc_balance > Decimal('0'):
                print(f"Current USDC balance found: {usdc_balance:.25f}")
            else:
                print("No USDC balance available.")
            print(f"Current BTC balance: {asset_balance:.25f}")

            if all_conditions_met:
                usdc_balance = get_balance('USDC')
                if usdc_balance > Decimal('0'):
                    print(f"\n!!! ALL {len(active_conditions_results)} ACTIVE CONDITIONS MET - EXECUTING TRADE !!!")
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
                print(f"Only {true_conditions_count}/{len(active_conditions_results)} conditions met.")
                print("Waiting for next iteration...")

        print(f"\nCurrent USDC balance: {usdc_balance:.25f}")
        print(f"Current BTC balance: {asset_balance:.25f} BTC")
        print(f"Current {TRADE_SYMBOL} price: {current_price:.25f}\n")

        del candle_map
        gc.collect()
        # CHANGE 4: Strict sleep time of 5 seconds.
        time.sleep(5)

except KeyboardInterrupt:
    print("\nBot stopped by user. Exiting gracefully...")
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