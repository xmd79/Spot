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
from scipy import signal
from scipy.fft import fft, ifft, fftfreq

# Set Decimal precision to 25
getcontext().prec = 25

# Exchange constants
TRADE_SYMBOL = "BTCUSDC"

# Timezone Configuration
LOCAL_TIMEZONE = datetime.timezone(datetime.timedelta(hours=2))  # GMT+2

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Load credentials from file
try:
    with open("api.txt", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        api_secret = lines[1].strip()
except FileNotFoundError:
    print("Error: api.txt file not found.")
    exit(1)

# Initialize Binance client with increased timeout
client = BinanceClient(api_key, api_secret, requests_params={"timeout": 30})

# Trading Configuration
PROFIT_TARGET_PERCENT = 0.35   # CHANGED: 0.35% net profit goal (was 1.5%)
TOTAL_FEE_PERCENT = 0.35      # 0.35% total fee percentage
MIN_TRADE_AMOUNT = 10

# CONFIGURATION UPDATED
CONFIG = {
    "conditions": {
        "momentum_positive_1m": True,
        "momentum_positive_15sec": True,
        "linear_regression_channel_break": True,
        "aroon_only_signal": True,
        "thresholds_15sec": True,
        "volume_bullish_1min": True,
        "stoch_rsi_precise_15sec": True,
        "stoch_rsi_precise_1min": True,
        "dip_confirmation_15sec": True,
        "dip_confirmation_1min": True,
        "stoch_oversold_vs_overbought_1min": True,
        "rsi_oversold_vs_overbought_15sec": True,
    },
    "min_conditions_met": 7  # Trigger trade if 7 out of 12 conditions are True
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
# UPDATED UTILITY: DIP CONFIRMATION (Argmin vs Argmax - MOST RECENT)
# =========================================================

def get_dip_confirmation(closes, lookback=1200):
    """
    Analyzes price structure using GLOBAL Argmin vs Argmax for last 1200 values.
    Returns: (is_dip_confirmed, abs_min_idx, abs_max_idx, val_at_min, val_at_max)
    """
    try:
        np_closes = np.array(closes, dtype=np.float64)
        recent_data = np_closes[-lookback:]
        n = len(recent_data)
        
        if n < 5:
            return False, -1, -1, 0.0, 0.0

        clean_data = recent_data[~np.isnan(recent_data)]
        if len(clean_data) == 0:
            return False, -1, -1, 0.0, 0.0

        min_val = np.nanmin(clean_data)
        max_val = np.nanmax(clean_data)

        min_indices_relative = np.where(clean_data == min_val)[0]
        max_indices_relative = np.where(clean_data == max_val)[0]

        last_min_idx_relative = min_indices_relative[-1] if len(min_indices_relative) > 0 else 0
        last_max_idx_relative = max_indices_relative[-1] if len(max_indices_relative) > 0 else 0

        abs_min_idx = -1
        abs_max_idx = -1
        
        found_min = False
        found_max = False
        
        for i in range(len(np_closes) - 1, -1, -1):
            val = np_closes[i]
            if np.isnan(val):
                continue
            
            if not found_min and abs(val - min_val) < 1e-8:
                abs_min_idx = i
                found_min = True
            
            if not found_max and abs(val - max_val) < 1e-8:
                abs_max_idx = i
                found_max = True
                
            if found_min and found_max:
                break

        if abs_min_idx == -1 or abs_max_idx == -1:
            return False, -1, -1, 0.0, 0.0

        # Get price values at the found indices
        val_at_min = np_closes[abs_min_idx]
        val_at_max = np_closes[abs_max_idx]

        is_dip_confirmed = abs_min_idx > abs_max_idx
        return is_dip_confirmed, abs_min_idx, abs_max_idx, val_at_min, val_at_max

    except Exception as e:
        print(f"Error in get_dip_confirmation: {e}")
        import traceback
        traceback.print_exc()
        return False, -1, -1, 0.0, 0.0

# =========================================================
# UPDATED FUNCTION: STOCH OVERSOLD VS OVERBOUGHT (1MIN)
# =========================================================

def check_stoch_oversold_vs_overbought_1min(candles_1m):
    """
    Checks if Oversold (<20) happened more recently than Overbought (>80)
    on the Stochastic %K line for the 1m timeframe.
    """
    try:
        if not candles_1m or len(candles_1m) < 15:
            return False, "Insufficient data", -1, -1, 0.0

        closes = np.array([c['close'] for c in candles_1m], dtype=np.float64)
        highs = np.array([c['high'] for c in candles_1m], dtype=np.float64)
        lows = np.array([c['low'] for c in candles_1m], dtype=np.float64)

        # Stoch: %K 14, Smooth 1, %D 4
        slowk, slowd = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=1, slowd_period=4)
        
        oversold_limit = 20.0
        overbought_limit = 80.0

        last_oversold_idx = -1
        last_overbought_idx = -1
        
        # Iterate backwards to find most recent occurrences
        for i in range(len(slowk) - 1, -1, -1):
            val = slowk[i]
            if np.isnan(val):
                continue
            
            if last_oversold_idx == -1 and val <= oversold_limit:
                last_oversold_idx = i
            
            if last_overbought_idx == -1 and val >= overbought_limit:
                last_overbought_idx = i
                
            if last_oversold_idx != -1 and last_overbought_idx != -1:
                break

        condition_met = last_oversold_idx > last_overbought_idx
        
        description = ""
        if condition_met:
            description = "Most recent extreme was OVERSOLD (Bullish Potential)"
        else:
            description = "Most recent extreme was OVERBOUGHT (Bearish Potential) or None"

        # Added current_val to return tuple for detailed printing
        current_val = slowk[-1]
        return condition_met, description, last_oversold_idx, last_overbought_idx, current_val

    except Exception as e:
        print(f"Error checking Stoch Oversold vs Overbought (1m): {e}")
        import traceback
        traceback.print_exc()
        return False, str(e), -1, -1, 0.0


# =========================================================
# UPDATED FUNCTION: RSI OVERSOLD VS OVERBOUGHT (15SEC)
# =========================================================

def check_rsi_oversold_vs_overbought_15sec(candles_15s):
    """
    Checks if Oversold (<30) happened more recently than Overbought (>70)
    on the RSI line for the 15s timeframe.
    Uses RSI Length 14.
    """
    try:
        if not candles_15s or len(candles_15s) < 20:
            return False, "Insufficient data", -1, -1, 0.0

        closes = np.array([c['close'] for c in candles_15s], dtype=np.float64)

        # RSI Length 14
        rsi_values = talib.RSI(closes, timeperiod=14)
        
        # Standard RSI Oversold/Overbought limits
        oversold_limit = 30.0
        overbought_limit = 70.0

        last_oversold_idx = -1
        last_overbought_idx = -1
        
        # Iterate backwards to find most recent occurrences
        for i in range(len(rsi_values) - 1, -1, -1):
            val = rsi_values[i]
            if np.isnan(val):
                continue
            
            if last_oversold_idx == -1 and val <= oversold_limit:
                last_oversold_idx = i
            
            if last_overbought_idx == -1 and val >= overbought_limit:
                last_overbought_idx = i
                
            if last_oversold_idx != -1 and last_overbought_idx != -1:
                break

        condition_met = last_oversold_idx > last_overbought_idx
        
        description = ""
        if condition_met:
            description = "Most recent extreme was OVERSOLD (Bullish Potential)"
        else:
            description = "Most recent extreme was OVERBOUGHT (Bearish Potential) or None"

        # Added current_val to return tuple for detailed printing
        current_val = rsi_values[-1]
        return condition_met, description, last_oversold_idx, last_overbought_idx, current_val

    except Exception as e:
        print(f"Error checking RSI Oversold vs Overbought (15s): {e}")
        import traceback
        traceback.print_exc()
        return False, str(e), -1, -1, 0.0


# =========================================================
# NEW FUNCTIONS: STOCH + RSI PRECISE STRATEGIES (UPDATED WITH RSI %D)
# =========================================================

def analyze_stoch_rsi_precise_strategy_15sec(candles_15s):
    """
    Analyzes 15s timeframe.
    Trigger: RSI < 61.8 AND Stoch %K < 61.8 AND Dip Confirmation.
    Added: RSI %D Calculation and Print.
    """
    try:
        if not candles_15s or len(candles_15s) < 20:
            return {"error": "Insufficient 15s data", "condition_met": False}

        closes = np.array([c['close'] for c in candles_15s], dtype=np.float64)
        highs = np.array([c['high'] for c in candles_15s], dtype=np.float64)
        lows = np.array([c['low'] for c in candles_15s], dtype=np.float64)

        # --- 1. CALCULATE INDICATORS ---
        # RSI Length: 3
        rsi_values = talib.RSI(closes, timeperiod=3)
        
        # RSI %D: Smoothed RSI (Simple Moving Average of RSI, period 3)
        rsi_d_values = talib.MA(rsi_values, timeperiod=3, matype=0) # 0=SMA
        current_rsi = rsi_values[-1]
        current_rsi_d = rsi_d_values[-1]
        
        # Stoch: %K 14, Smooth 1, %D 4
        slowk, slowd = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=1, slowd_period=4)
        current_stoch_k = slowk[-1]
        current_stoch_d = slowd[-1]

        # --- 2. DIP CONFIRMATION (Argmin vs Argmax) ---
        # Updated to unpack 5 values (added price values at end)
        is_dip_15s, idx_min, idx_max, val_min, val_max = get_dip_confirmation(closes, lookback=1200)

        # --- 3. TRIGGER CONDITIONS ---
        threshold = 61.8
        
        # Condition: RSI < 61.8
        rsi_condition_met = current_rsi < threshold
        
        # Condition: Stoch %K < 61.8
        stoch_k_condition_met = current_stoch_k < threshold
        
        # Combined Logic: RSI, Stoch K, AND Dip Confirmation must all be True
        condition_met = rsi_condition_met and stoch_k_condition_met and is_dip_15s

        # --- PRINT SPECS ---
        print(f"====== 15s PRECISE STRATEGY ======")
        print(f"RSI (Len 3):           {current_rsi:.2f} < {threshold} : {rsi_condition_met}")
        print(f"RSI %D (MA 3):         {current_rsi_d:.2f}") # NEW PRINT
        print(f"Stoch %K (14,1,4):     {current_stoch_k:.2f} < {threshold} : {stoch_k_condition_met}")
        print(f"Stoch %D:              {current_stoch_d:.2f}")
        print(f"Dip Conf (MinIdx {idx_min} > MaxIdx {idx_max}): {is_dip_15s}")
        print(f"-----------------------------------")
        print(f"TOTAL CONDITION MET:   {condition_met}")
        print(f"===================================")

        return {
            "timeframe": "15s",
            "condition_met": condition_met,
            "current_rsi": current_rsi,
            "current_rsi_d": current_rsi_d,
            "current_stoch_k": current_stoch_k,
            "current_stoch_d": current_stoch_d,
            "dip_confirmed": is_dip_15s,
            "rsi_trigger": rsi_condition_met,
            "stoch_k_trigger": stoch_k_condition_met
        }

    except Exception as e:
        print(f"Error in analyze_stoch_rsi_precise_strategy_15sec: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "condition_met": False}


def analyze_stoch_rsi_precise_strategy_1min(candles_1m):
    """
    Analyzes 1min timeframe.
    Trigger: RSI < 61.8 AND Stoch %K < 61.8 AND Dip Confirmation.
    Added: RSI %D Calculation and Print.
    """
    try:
        if not candles_1m or len(candles_1m) < 20:
            return {"error": "Insufficient 1m data", "condition_met": False}

        closes = np.array([c['close'] for c in candles_1m], dtype=np.float64)
        highs = np.array([c['high'] for c in candles_1m], dtype=np.float64)
        lows = np.array([c['low'] for c in candles_1m], dtype=np.float64)

        # --- 1. CALCULATE INDICATORS ---
        # RSI Length: 3
        rsi_values = talib.RSI(closes, timeperiod=3)
        
        # RSI %D: Smoothed RSI (Simple Moving Average of RSI, period 3)
        rsi_d_values = talib.MA(rsi_values, timeperiod=3, matype=0) # 0=SMA
        current_rsi = rsi_values[-1]
        current_rsi_d = rsi_d_values[-1]

        # Stoch: %K 14, Smooth 1, %D 4
        slowk, slowd = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=1, slowd_period=4)
        current_stoch_k = slowk[-1]
        current_stoch_d = slowd[-1]

        # --- 2. DIP CONFIRMATION (Argmin vs Argmax) ---
        # Updated to unpack 5 values (added price values at end)
        is_dip_1m, idx_min, idx_max, val_min, val_max = get_dip_confirmation(closes, lookback=1200)

        # --- 3. TRIGGER CONDITIONS ---
        threshold = 61.8
        
        # Condition: RSI < 61.8
        rsi_condition_met = current_rsi < threshold
        
        # Condition: Stoch %K < 61.8
        stoch_k_condition_met = current_stoch_k < threshold
        
        # Combined Logic: RSI, Stoch K, AND Dip Confirmation must all be True
        condition_met = rsi_condition_met and stoch_k_condition_met and is_dip_1m

        # --- PRINT SPECS ---
        print(f"====== 1m PRECISE STRATEGY ======")
        print(f"RSI (Len 3):           {current_rsi:.2f} < {threshold} : {rsi_condition_met}")
        print(f"RSI %D (MA 3):         {current_rsi_d:.2f}") # NEW PRINT
        print(f"Stoch %K (14,1,4):     {current_stoch_k:.2f} < {threshold} : {stoch_k_condition_met}")
        print(f"Stoch %D:              {current_stoch_d:.2f}")
        print(f"Dip Conf (MinIdx {idx_min} > MaxIdx {idx_max}): {is_dip_1m}")
        print(f"-----------------------------------")
        print(f"TOTAL CONDITION MET:   {condition_met}")
        print(f"===================================")

        return {
            "timeframe": "1m",
            "condition_met": condition_met,
            "current_rsi": current_rsi,
            "current_rsi_d": current_rsi_d,
            "current_stoch_k": current_stoch_k,
            "current_stoch_d": current_stoch_d,
            "dip_confirmed": is_dip_1m,
            "rsi_trigger": rsi_condition_met,
            "stoch_k_trigger": stoch_k_condition_met
        }

    except Exception as e:
        print(f"Error in analyze_stoch_rsi_precise_strategy_1min: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "condition_met": False}


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

def get_candles(symbol, timeframe, limit=1200, retries=5, delay=5, endTime=None):
    for attempt in range(retries):
        try:
            actual_limit = min(limit, 1000)
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': actual_limit
            }
            if endTime is not None:
                params['endTime'] = endTime
            klines = client.get_klines(**params)
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
            return None, None, None, None
        usdc_balance = get_balance('USDC')
        if usdc_balance <= Decimal('0'):
            return None, None, None, None
        raw_quantity = usdc_balance / current_price
        
        lot_size_info = get_symbol_lot_size_info(TRADE_SYMBOL)
        step_size = lot_size_info['stepSize']
        min_trade_size = lot_size_info['minQty']
        
        step_precision = int(-math.log10(float(step_size))) if step_size > Decimal('0') else 8
        adjusted_quantity = (raw_quantity // step_size) * step_size
        adjusted_quantity = adjusted_quantity.quantize(Decimal('0.' + '0' * step_precision))
        cost = adjusted_quantity * current_price
        
        min_notional = Decimal('10.0')
        if cost < min_notional:
            min_quantity_for_notional = min_notional / current_price
            adjusted_quantity = ((min_quantity_for_notional + step_size - Decimal('1E-25')) // step_size) * step_size
            adjusted_quantity = adjusted_quantity.quantize(Decimal('0.' + '0' * step_precision))
            cost = adjusted_quantity * current_price
        
        if adjusted_quantity < min_trade_size:
            return None, None, None, None
        
        if cost > usdc_balance:
            adjusted_quantity = ((usdc_balance / current_price) // step_size) * step_size
            adjusted_quantity = adjusted_quantity.quantize(Decimal('0.' + '0' * step_precision))
            cost = adjusted_quantity * current_price
        
        if adjusted_quantity < min_trade_size:
            return None, None, None, None
        
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
            return False
        asset_balance_dec = Decimal(str(asset_balance))
        
        lot_size_info = get_symbol_lot_size_info(TRADE_SYMBOL)
        step_size = lot_size_info['stepSize']
        min_trade_size = lot_size_info['minQty']
        
        step_precision = int(-math.log10(float(step_size))) if step_size > Decimal('0') else 8
        sell_quantity = (asset_balance_dec // step_size) * step_size
        sell_quantity = sell_quantity.quantize(Decimal('0.' + '0' * step_precision))
        
        if sell_quantity < min_trade_size:
            return False
        
        sell_order = client.order_market_sell(symbol=TRADE_SYMBOL, quantity=float(sell_quantity))
        print(f"Market sell order executed: {sell_order}")
        return True
    except BinanceAPIException as e:
        print(f"Error executing sell order: {e.message}")
        return False

def check_exit_condition(initial_investment, asset_balance, entry_price):
    """
    Calculates exit condition based on Clean Net Profit Target + Fees.
    """
    if initial_investment <= Decimal('0.0') or asset_balance <= Decimal('0.0') or entry_price <= Decimal('0.0'):
        return False
    current_price = get_current_price()
    if current_price <= Decimal('0.0'):
        return False
    
    total_required_gross_percent = PROFIT_TARGET_PERCENT + TOTAL_FEE_PERCENT
    multiplier = Decimal(str(1.0 + total_required_gross_percent / 100))
    
    target_value = initial_investment * multiplier
    target_price = target_value / asset_balance
    
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
# THRESHOLDS CALCULATION FUNCTION
# =========================================================

def calculate_thresholds(close_prices, period=14, minimum_percentage=3, maximum_percentage=3, range_distance=0.05):
    min_close = np.nanmin(close_prices)
    max_close = np.nanmax(close_prices)
    close_prices = np.array(close_prices)
    momentum = talib.MOM(close_prices, timeperiod=period)
    min_momentum = np.nanmin(momentum)   
    max_momentum = np.nanmax(momentum)
    min_percentage_custom = minimum_percentage / 100  
    max_percentage_custom = maximum_percentage / 100
    min_threshold = np.minimum(min_close - (max_close - min_close) * min_percentage_custom, close_prices[-1])
    max_threshold = np.maximum(max_close + (max_close - min_close) * max_percentage_custom, close_prices[-1])
    range_price = np.linspace(close_prices[-1] * (1 - range_distance), close_prices[-1] * (1 + range_distance), num=50)
    with np.errstate(invalid='ignore'):
        filtered_close = np.where(close_prices < min_threshold, min_threshold, close_prices)      
        filtered_close = np.where(filtered_close > max_threshold, max_threshold, filtered_close)
    avg_mtf = np.nanmean(filtered_close)
    current_momentum = momentum[-1]
    with np.errstate(invalid='ignore', divide='ignore'):
        percent_to_min_momentum = ((max_momentum - current_momentum) /   
                                   (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan               
        percent_to_max_momentum = ((current_momentum - min_momentum) / 
                                   (max_momentum - min_momentum)) * 100 if max_momentum - min_momentum != 0 else np.nan
    percent_to_min_combined = (minimum_percentage + percent_to_min_momentum) / 2         
    percent_to_max_combined = (maximum_percentage + percent_to_max_momentum) / 2      
    momentum_signal = percent_to_max_combined - percent_to_min_combined
    return min_threshold, max_threshold, avg_mtf, momentum_signal, range_price

# =========================================================
# NEW FUNCTION: VOLUME SENTIMENT ANALYSIS
# =========================================================

def analyze_volume_sentiment(candles):
    """
    Calculates bullish vs bearish volume ratio.
    Returns: condition_met (bullish > bearish), bull_pct, bear_pct
    """
    try:
        if not candles:
            return False, 0.0, 0.0, {"error": "No candles provided"}

        total_bullish_vol = 0.0
        total_bearish_vol = 0.0

        for candle in candles:
            vol = candle.get('volume', 0)
            close = candle.get('close', 0)
            open_p = candle.get('open', 0)

            if close > open_p:
                total_bullish_vol += vol
            elif close < open_p:
                total_bearish_vol += vol

        total_vol = total_bullish_vol + total_bearish_vol
        
        if total_vol == 0:
            return False, 0.0, 0.0, {"error": "Total volume is zero"}

        bull_pct = (total_bullish_vol / total_vol) * 100
        bear_pct = (total_bearish_vol / total_vol) * 100
        
        condition_met = total_bullish_vol > total_bearish_vol
        
        return condition_met, bull_pct, bear_pct, {}
        
    except Exception as e:
        print(f"Error analyzing volume sentiment: {e}")
        return False, 0.0, 0.0, {"error": str(e)}

# =========================================================
# NEW FUNCTION: THRESHOLDS ANALYSIS (15SEC)
# =========================================================

def analyze_thresholds_15sec(candles_15s):
    """
    Calculates thresholds for 15s timeframe.
    Logic: Condition is True if current close is below the middle point
    between Min and Max of the last 1200 values.
    """
    try:
        if not candles_15s:
            return {"error": "No 15s data provided", "condition_met": False}
        
        recent_candles = candles_15s[-1200:] if len(candles_15s) >= 1200 else candles_15s
        
        close_prices = np.array([candle["close"] for candle in recent_candles], dtype=np.float64)
        
        if len(close_prices) < 2:
            return {"error": "Insufficient data for 15s analysis", "condition_met": False}
        
        min_value = np.min(close_prices)
        max_value = np.max(close_prices)
        current_price = close_prices[-1]
        
        midpoint = (min_value + max_value) / 2
        
        condition_met = current_price < midpoint
        
        return {
            "timeframe": "15s",
            "condition_met": condition_met,
            "current_price": current_price,
            "min_value": min_value,
            "max_value": max_value,
            "midpoint": midpoint,
            "description": f"Current Price ({current_price:.2f}) is {'below' if condition_met else 'above'} Midpoint ({midpoint:.2f})"
        }
        
    except Exception as e:
        print(f"Error analyzing thresholds (15s): {e}")
        return {"error": str(e), "condition_met": False}

# =========================================================
# NEW FUNCTION: SINE WAVE OSCILLATOR WITH FFT (15SEC) - REMOVED FROM ACTIVE LOGIC
# =========================================================

def analyze_sine_wave_oscillator_15sec(candles_15s):
    """
    Analyzes sine wave cycles on the 15s timeframe.
    Uses simulated 15s candles.
    """
    try:
        if not candles_15s:
            return {"error": "No 15s data provided", "condition_met": False}
        
        recent_candles = candles_15s[-1000:] if len(candles_15s) > 1000 else candles_15s
        
        if len(recent_candles) < 300:
            return {"error": f"Insufficient 15s data: only {len(recent_candles)} candles", "condition_met": False}
        
        close_prices = np.array([candle["close"] for candle in recent_candles], dtype=np.float64)
        
        minima_indices = signal.argrelextrema(close_prices, np.less, order=30)[0]
        maxima_indices = signal.argrelextrema(close_prices, np.greater, order=30)[0]
        
        if len(minima_indices) < 2 or len(maxima_indices) < 2:
            return {"error": "Not enough extrema for cycle analysis", "condition_met": False}
        
        last_min_idx = minima_indices[-1]
        last_max_idx = maxima_indices[-1]
        current_idx = len(close_prices) - 1
        current_price = close_prices[-1]
        
        x = np.arange(len(close_prices))
        coeffs = np.polyfit(x, close_prices, 1)
        detrended = close_prices - (coeffs[0] * x + coeffs[1])
        fft_values = fft(detrended)
        fft_freq = fftfreq(len(detrended))
        
        valid_indices = np.where((np.abs(fft_freq) > 1/3000) & (np.abs(fft_freq) < 1/20))
        valid_freq = fft_freq[valid_indices]
        valid_fft = np.abs(fft_values[valid_indices])
        
        if len(valid_fft) == 0:
            return {"error": "No valid frequency components found", "condition_met": False}
        
        dominant_freq_idx = np.argmax(valid_fft)
        dominant_freq = valid_freq[dominant_freq_idx]
        dominant_period = 1 / np.abs(dominant_freq)
        
        min_threshold, max_threshold, avg_mtf, momentum_signal, range_price = calculate_thresholds(
            close_prices, period=14, minimum_percentage=2, maximum_percentage=2, range_distance=0.05
        )
        
        midpoint_threshold = (min_threshold + max_threshold) / 2
        
        if last_min_idx > last_max_idx:
            cycle_type = "UP"
            next_extremum_type = "MAXIMUM"
            if current_price < midpoint_threshold:
                threshold_signal = "UP_STAGE_1"
                stage_description = "In dip area, moving toward middle area"
            else:
                threshold_signal = "UP_STAGE_2"
                stage_description = "In middle area, moving toward top area"
            
            prev_max_idx = maxima_indices[-2]
            wave_period = last_min_idx - prev_max_idx
            last_min_value = close_prices[last_min_idx]
            prev_max_value = close_prices[prev_max_idx]
            amplitude = prev_max_value - last_min_value
            fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
            fib_amplitude = amplitude * fib_ratios[3]
            forecast_price = last_min_value + fib_amplitude
            
            if forecast_price <= current_price:
                forecast_price = current_price + (current_price * 0.001)
            
            time_to_next_max = wave_period - (current_idx - last_min_idx)
            primary_validation = forecast_price > current_price
            
            periods_between_extrema = []
            all_extrema = sorted(np.concatenate([minima_indices, maxima_indices]))
            for i in range(1, len(all_extrema)):
                if (all_extrema[i] in minima_indices and all_extrema[i-1] in minima_indices) or \
                   (all_extrema[i] in maxima_indices and all_extrema[i-1] in maxima_indices):
                    continue
                periods_between_extrema.append(abs(all_extrema[i] - all_extrema[i-1]))
            
            if periods_between_extrema:
                avg_period_between_extrema = np.mean(periods_between_extrema)
                fft_validation = (avg_period_between_extrema * 0.75 <= wave_period <= avg_period_between_extrema * 1.25)
            else:
                avg_period_between_extrema = 0
                fft_validation = False
            
            condition_met = True
        else:
            cycle_type = "DOWN"
            next_extremum_type = "MINIMUM"
            if current_price > midpoint_threshold:
                threshold_signal = "DOWN_STAGE_1"
                stage_description = "In top area, moving toward middle area"
            else:
                threshold_signal = "DOWN_STAGE_2"
                stage_description = "In middle area, moving toward dip area"
            
            prev_min_idx = minima_indices[-2]
            wave_period = last_max_idx - prev_min_idx
            last_max_value = close_prices[last_max_idx]
            prev_min_value = close_prices[prev_min_idx]
            amplitude = last_max_value - prev_min_value
            fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
            fib_amplitude = amplitude * fib_ratios[3]
            forecast_price = last_max_value - fib_amplitude
            
            if forecast_price >= current_price:
                forecast_price = current_price - (current_price * 0.001)
            
            time_to_next_min = wave_period - (current_idx - last_max_idx)
            primary_validation = forecast_price < current_price
            
            periods_between_extrema = []
            all_extrema = sorted(np.concatenate([minima_indices, maxima_indices]))
            for i in range(1, len(all_extrema)):
                if (all_extrema[i] in minima_indices and all_extrema[i-1] in minima_indices) or \
                   (all_extrema[i] in maxima_indices and all_extrema[i-1] in maxima_indices):
                    continue
                periods_between_extrema.append(abs(all_extrema[i] - all_extrema[i-1]))
            
            if periods_between_extrema:
                avg_period_between_extrema = np.mean(periods_between_extrema)
                fft_validation = (avg_period_between_extrema * 0.75 <= wave_period <= avg_period_between_extrema * 1.25)
            else:
                avg_period_between_extrema = 0
                fft_validation = False
            
            condition_met = False
        
        if cycle_type == "UP":
            cycle_position = (current_idx - last_min_idx) / wave_period if wave_period > 0 else 0
        else:
            cycle_position = (current_idx - last_max_idx) / wave_period if wave_period > 0 else 0
        
        return {
            "condition_met": condition_met,
            "cycle_type": cycle_type,
            "current_price": current_price,
            "forecast_price": forecast_price,
            "next_extremum_type": next_extremum_type,
            "last_min_idx": last_min_idx,
            "last_max_idx": last_max_idx,
            "last_min_value": close_prices[last_min_idx],
            "last_max_value": close_prices[last_max_idx],
            "cycle_position": cycle_position,
            "wave_period": wave_period,
            "fib_amplitude": fib_amplitude,
            "dominant_period": dominant_period,
            "min_threshold": min_threshold,
            "max_threshold": max_threshold,
            "midpoint_threshold": midpoint_threshold,
            "momentum_signal": momentum_signal,
            "threshold_signal": threshold_signal,
            "stage_description": stage_description,
            "avg_period_between_extrema": avg_period_between_extrema,
            "description": f"Current cycle is {cycle_type}, expecting a new {next_extremum_type} at price {forecast_price:.2f}. {stage_description}"
        }
        
    except Exception as e:
        print(f"Error in analyze_sine_wave_oscillator_15sec: {e}")
        return {"error": str(e), "condition_met": False}

# =========================================================
# UPDATED FUNCTION: SEPARATED INDICATOR CONDITIONS (STOCH ONLY) - REMOVED FROM ACTIVE LOGIC
# =========================================================

def analyze_stoch_conditions(candles_1m, candles_15s, lookback=200):
    """
    Analyzes Stoch on BOTH 1m and 15s timeframes independently.
    RSI has been removed as per request.
    - Checks Most Recent State (Oversold vs Overbought).
    - Returns 2 separate boolean conditions.
    """
    
    def get_most_recent_state(values, oversold_limit, overbought_limit):
        # Iterate backwards to find most recent extreme
        for val in reversed(values):
            if val <= oversold_limit:
                return "OVERSOLD", True  # Strict: Found Oversold
            if val >= overbought_limit:
                return "OVERBOUGHT", False # Strict: Found Overbought
        return "NEUTRAL", False 

    results = {}

    # --- 1m Analysis ---
    try:
        df_1m = pd.DataFrame(candles_1m)
        high_1m = df_1m["high"]
        low_1m = df_1m["low"]
        close_1m = df_1m["close"]

        # 1m Stoch
        lowest_low = low_1m.rolling(14).min()
        highest_high = high_1m.rolling(14).max()
        stoch_k_1m = 100 * (close_1m - lowest_low) / (highest_high - lowest_low)
        
        state, is_oversold = get_most_recent_state(stoch_k_1m.tail(lookback).values, oversold_limit=20, overbought_limit=80)
        results["stoch_1m"] = {"state": state, "condition_met": is_oversold, "current_val": stoch_k_1m.iloc[-1] if not stoch_k_1m.empty else 0}

    except Exception as e:
        print(f"Error analyzing 1m indicators: {e}")
        results["stoch_1m"] = {"state": "ERROR", "condition_met": False, "current_val": 0}

    # --- 15s Analysis ---
    try:
        df_15s = pd.DataFrame(candles_15s)
        high_15s = df_15s["high"]
        low_15s = df_15s["low"]
        close_15s = df_15s["close"]

        # 15s Stoch
        lowest_low = low_15s.rolling(14).min()
        highest_high = high_15s.rolling(14).max()
        stoch_k_15s = 100 * (close_15s - lowest_low) / (highest_high - lowest_low)
        
        state, is_oversold = get_most_recent_state(stoch_k_15s.tail(lookback).values, oversold_limit=20, overbought_limit=80)
        results["stoch_15s"] = {"state": state, "condition_met": is_oversold, "current_val": stoch_k_15s.iloc[-1] if not stoch_k_15s.empty else 0}

    except Exception as e:
        print(f"Error analyzing 15s indicators: {e}")
        results["stoch_15s"] = {"state": "ERROR", "condition_met": False, "current_val": 0}

    return results

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
        
        # 5m Extrema calculation kept but logic removed from active conditions
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
    """
    UPDATED: Simulates realistic price movement inside a 1m candle.
    Instead of linear interpolation (which causes 100% identical volume sentiment),
    this uses the Highs and Lows to create micro-trends.
    """
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
        
        if high_price == low_price:
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
        else:
            o1 = open_price
            c1 = high_price
            timestamps.append(row['timestamp'])
            opens.append(o1); closes.append(c1); highs.append(high_price); lows.append(low_price); volumes.append(volume / 4)

            o2 = c1
            c2 = low_price
            timestamps.append(row['timestamp'] + datetime.timedelta(seconds=15))
            opens.append(o2); closes.append(c2); highs.append(high_price); lows.append(low_price); volumes.append(volume / 4)

            o3 = c2
            c3 = (open_price + close_price) / 2
            timestamps.append(row['timestamp'] + datetime.timedelta(seconds=30))
            opens.append(o3); closes.append(c3); highs.append(high_price); lows.append(low_price); volumes.append(volume / 4)

            o4 = c3
            c4 = close_price
            timestamps.append(row['timestamp'] + datetime.timedelta(seconds=45))
            opens.append(o4); closes.append(c4); highs.append(high_price); lows.append(low_price); volumes.append(volume / 4)

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
            "momentum_strength": 'Strong' if abs(momentum_value) > 100 else 'Moderate' if abs(momentum_value) > 50 else 'Weak',
            "candles_15s": candles_15s 
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

def analyze_linear_regression_channel_break(candles):
    """
    Note: Parameter 'candles' is now expected to be 15s timeframe data.
    """
    try:
        if not candles:
            return {"error": "No data provided"}
        
        lrc_result = calculate_linear_regression_channel(candles, period=500, dev_multiplier=2.0)
        
        if "error" in lrc_result:
            return lrc_result
        
        below_lower_indices = lrc_result["below_lower_indices"]
        above_upper_indices = lrc_result["above_upper_indices"]
        
        most_recent_below_lower_index = max(below_lower_indices) if below_lower_indices else None
        most_recent_above_upper_index = max(above_upper_indices) if above_upper_indices else None
        
        condition_met = False
        description = "No channel breaks detected in analysis window."
        
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

        # Fetch candles for multiple timeframes (Removed 5m from fetch list)
        print("Fetching candle data...")
        candle_map = fetch_candles_in_parallel(['1m', '3m'])
        candles_1m = candle_map.get('1m', [])
        candles_3m = candle_map.get('3m', [])
        
        print(f"Candle counts - 1m: {len(candles_1m)}, 3m: {len(candles_3m)}")
        
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
            "thresholds_15sec": False,
            "volume_bullish_1min": False,
            # REMOVED "volume_bullish_15sec"
            "stoch_rsi_precise_15sec": False,
            "stoch_rsi_precise_1min": False,
            "dip_confirmation_15sec": False,
            "dip_confirmation_1min": False,
            "stoch_oversold_vs_overbought_1min": False, # REPLACED 15s WITH 1m
            "rsi_oversold_vs_overbought_15sec": False, # NEW
        }

        # Generate 15s candles early for other conditions
        momentum_15s_result = analyze_momentum_15sec(candles_1m)
        candles_15s = []
        if 'error' not in momentum_15s_result:
            candles_15s = momentum_15s_result.get('candles_15s', [])

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
        
        # Condition 3: Linear Regression Channel Break (UPDATED TO 15s)
        print("\n--- Condition 3: Linear Regression Channel Break (15s) ---")
        lrc_break_result = analyze_linear_regression_channel_break(candles_15s) 
        if 'error' not in lrc_break_result:
            conditions_status["linear_regression_channel_break"] = lrc_break_result['condition_met']
            print(f"Timeframe: 15s")
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

        # Condition 5: Thresholds 15sec
        print("\n--- Condition 5: Thresholds Analysis (15s) ---")
        thresholds_15s_result = analyze_thresholds_15sec(candles_15s)
        if 'error' not in thresholds_15s_result:
            conditions_status["thresholds_15sec"] = thresholds_15s_result['condition_met']
            print(f"Timeframe: {thresholds_15s_result['timeframe']}")
            print(f"Current Price: {thresholds_15s_result['current_price']:.2f}")
            print(f"Min Value: {thresholds_15s_result['min_value']:.2f}")
            print(f"Max Value: {thresholds_15s_result['max_value']:.2f}")
            print(f"Midpoint: {thresholds_15s_result['midpoint']:.2f}")
            print(f"Description: {thresholds_15s_result['description']}")
            print(f"Condition Met: {conditions_status['thresholds_15sec']}")
        else:
            print(f"Error analyzing thresholds (15s): {thresholds_15s_result['error']}")
            print(f"Condition Met: {conditions_status['thresholds_15sec']}")

        # Condition 6: Volume 1m Sentiment
        print("\n--- Condition 6: Volume Sentiment (1m) ---")
        vol_1m_cond, bull_pct_1m, bear_pct_1m, err_1m = analyze_volume_sentiment(candles_1m)
        if not err_1m:
            conditions_status["volume_bullish_1min"] = vol_1m_cond
            print(f"Bullish Vol %: {bull_pct_1m:.2f}%")
            print(f"Bearish Vol %: {bear_pct_1m:.2f}%")
            print(f"Sentence: {'Bullish' if vol_1m_cond else 'Bearish'} Volume Dominates")
            print(f"Condition Met: {vol_1m_cond}")
        else:
            print(f"Error analyzing 1m volume: {err_1m.get('error', 'Unknown')}")
            print(f"Condition Met: {conditions_status['volume_bullish_1min']}")

        # REMOVED Condition 7: Volume 15s Sentiment
            
        # Condition 7: Stoch RSI Precise (15sec) - UPDATED
        print("\n--- Condition 7: Stoch + RSI Precise (15s) ---")
        precise_15s_result = analyze_stoch_rsi_precise_strategy_15sec(candles_15s)
        if 'error' not in precise_15s_result:
            conditions_status["stoch_rsi_precise_15sec"] = precise_15s_result['condition_met']
            conditions_status["dip_confirmation_15sec"] = precise_15s_result['dip_confirmed'] 
        else:
            conditions_status["stoch_rsi_precise_15sec"] = False
            conditions_status["dip_confirmation_15sec"] = False

        # Condition 8: Stoch RSI Precise (1min) - UPDATED
        print("\n--- Condition 8: Stoch + RSI Precise (1m) ---")
        precise_1m_result = analyze_stoch_rsi_precise_strategy_1min(candles_1m)
        if 'error' not in precise_1m_result:
            conditions_status["stoch_rsi_precise_1min"] = precise_1m_result['condition_met']
            conditions_status["dip_confirmation_1min"] = precise_1m_result['dip_confirmed'] 
        else:
            conditions_status["stoch_rsi_precise_1min"] = False
            conditions_status["dip_confirmation_1min"] = False
            
        # Condition 9: Dip Confirmation (15s) - UPDATED PRINTS
        print("\n--- Condition 9: Dip Confirmation (15s) ---")
        # Fetch indices and prices specifically for printing
        is_dip_15s_print, idx_min_15s_print, idx_max_15s_print, price_min_15s, price_max_15s = get_dip_confirmation([c['close'] for c in candles_15s], lookback=1200)
        print(f"Last Argmin Index (Global): {idx_min_15s_print}")
        print(f"Last Argmin Price: {price_min_15s:.2f}")
        print(f"Last Argmax Index (Global): {idx_max_15s_print}")
        print(f"Last Argmax Price: {price_max_15s:.2f}")
        print(f"Argmin > Argmax: {'YES' if is_dip_15s_print else 'NO'}")
        print(f"Condition Met: {conditions_status['dip_confirmation_15sec']}")

        # Condition 10: Dip Confirmation (1m) - UPDATED PRINTS
        print("\n--- Condition 10: Dip Confirmation (1m) ---")
        # Fetch indices and prices specifically for printing
        is_dip_1m_print, idx_min_1m_print, idx_max_1m_print, price_min_1m, price_max_1m = get_dip_confirmation([c['close'] for c in candles_1m], lookback=1200)
        print(f"Last Argmin Index (Global): {idx_min_1m_print}")
        print(f"Last Argmin Price: {price_min_1m:.2f}")
        print(f"Last Argmax Index (Global): {idx_max_1m_print}")
        print(f"Last Argmax Price: {price_max_1m:.2f}")
        print(f"Argmin > Argmax: {'YES' if is_dip_1m_print else 'NO'}")
        print(f"Condition Met: {conditions_status['dip_confirmation_1min']}")

        # Condition 11: Stoch Oversold vs Overbought (1m) - REPLACED 15s
        print("\n--- Condition 11: Stoch Oversold vs Overbought (1m) ---")
        stoch_1m_os_result = check_stoch_oversold_vs_overbought_1min(candles_1m)
        # Unpack: condition_met, description, last_oversold_idx, last_overbought_idx, current_val
        conditions_status["stoch_oversold_vs_overbought_1min"] = stoch_1m_os_result[0]
        
        print(f"Current Stoch %K: {stoch_1m_os_result[4]:.2f}")
        print(f"Last Oversold Index (<20): {stoch_1m_os_result[2]}")
        print(f"Last Overbought Index (>80): {stoch_1m_os_result[3]}")
        print(f"Description: {stoch_1m_os_result[1]}")
        print(f"Condition Met: {stoch_1m_os_result[0]}")

        # Condition 12: RSI Oversold vs Overbought (15s) - NEW
        print("\n--- Condition 12: RSI Oversold vs Overbought (15s) ---")
        rsi_15s_os_result = check_rsi_oversold_vs_overbought_15sec(candles_15s)
        # Unpack: condition_met, description, last_oversold_idx, last_overbought_idx, current_val
        conditions_status["rsi_oversold_vs_overbought_15sec"] = rsi_15s_os_result[0]
        
        print(f"Current RSI: {rsi_15s_os_result[4]:.2f}")
        print(f"Last Oversold Index (<30): {rsi_15s_os_result[2]}")
        print(f"Last Overbought Index (>70): {rsi_15s_os_result[3]}")
        print(f"Description: {rsi_15s_os_result[1]}")
        print(f"Condition Met: {rsi_15s_os_result[0]}")

        # Print all conditions with true/false values
        print("\n" + "="*80)
        print("TRADING CONDITIONS STATUS")
        print("="*80)
        
        # Get results only for active conditions and require COUNT to be >= min_conditions_met
        active_conditions_results = [
            conditions_status[condition_name]
            for condition_name in CONFIG['conditions']
        ]
        true_conditions_count = sum(int(status) for status in active_conditions_results)
        false_conditions_count = len(active_conditions_results) - true_conditions_count
        
        min_required = CONFIG['min_conditions_met']
        
        print(f"Overall Conditions Status: {true_conditions_count} True, {false_conditions_count} False")
        print(f"Total Active Conditions: {len(active_conditions_results)}")
        print(f"Minimum Required to Trigger: {min_required}")
        
        print("\nCondition Summary (Active):")
        print("-" * 65)
        for condition_name in CONFIG['conditions']:
            status = "TRUE" if conditions_status[condition_name] else "FALSE"
            print(f"{condition_name:<50}{status}")
        print("-" * 65)
        
        # =========================================================
        # UPDATED LOGIC: STRICT CONDITIONS + COUNT THRESHOLD
        # =========================================================
        
        # 1. Check Count Condition (7 from 12)
        count_condition_met = true_conditions_count >= min_required
        
        # 2. Define Strict Conditions List (Stoch OS/OB 1m REMOVED)
        strict_checks_list = [
            # REMOVED ("Stoch OS/OB (1m)", conditions_status["stoch_oversold_vs_overbought_1min"]),
            ("RSI OS/OB (15s)", conditions_status["rsi_oversold_vs_overbought_15sec"]),
            ("Momentum Positive (1m)", conditions_status["momentum_positive_1m"]),
            ("Momentum Positive (15s)", conditions_status["momentum_positive_15sec"]),
            ("Linear Reg Channel Break", conditions_status["linear_regression_channel_break"]),
            ("Aroon Only Signal", conditions_status["aroon_only_signal"]),
            ("Thresholds (15s)", conditions_status["thresholds_15sec"]),
        ]

        # Calculate Total Checks (1 Count + 6 Strict)
        total_checks = 1 + len(strict_checks_list)
        
        # 3. Print Logic Checks Sequentially
        print("\n" + "="*80)
        print("FINAL TRIGGER LOGIC CHECK")
        print("="*80)
        
        # Check 1: Count Requirement
        print(f"[1/{total_checks}] Count Requirement ({true_conditions_count} >= {min_required}): {'PASS' if count_condition_met else 'FAIL'}")
        
        # Checks 2 through 7: Strict Conditions
        check_num = 2
        all_strict_conditions_met = True
        
        for name, passed in strict_checks_list:
            status_str = "PASS" if passed else "FAIL"
            print(f"[{check_num}/{total_checks}] Strict: {name:<30} {status_str}")
            if not passed:
                all_strict_conditions_met = False
            check_num += 1

        # 4. Final Signal Trigger (Must be True AND True AND True)
        signal_triggered = count_condition_met and all_strict_conditions_met
        
        print("-" * 80)
        print(f"FINAL SIGNAL TRIGGERED: {'YES' if signal_triggered else 'NO'}")
        print("="*80)

        # Save signal to file if signal is triggered
        if signal_triggered:
            signal_data = {
                "current_price": float(current_price),
                "signal": "BUY",
                "conditions_met": f"{true_conditions_count}/{len(active_conditions_results)} (Strict: RSI OS/OB 15s, Mom 1m, Mom 15s, LRC, Aroon, Thresh 15s)"
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

            # Calculate target based on clean net profit + fees
            total_required_gross_percent = PROFIT_TARGET_PERCENT + TOTAL_FEE_PERCENT
            target_multiplier = Decimal(str(1.0 + total_required_gross_percent / 100))
            target_value = initial_investment * target_multiplier
            
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
            
            print(f"Net Profit Goal: {PROFIT_TARGET_PERCENT}%")
            print(f"Total Fees: {TOTAL_FEE_PERCENT}%")
            print(f"Price for {PROFIT_TARGET_PERCENT}% Clean Net Profit (+ {TOTAL_FEE_PERCENT}% Fees): {target_price:.25f}")

            if entry_price > Decimal('0') and target_price > entry_price:
                entry_to_current_pct = ((current_price - entry_price) / entry_price) * Decimal('100')
                current_to_target_pct = ((target_price - current_price) / current_price) * Decimal('100')
                entry_to_target_pct = ((target_price - entry_price) / entry_price) * Decimal('100')
                
                print(f"Entry Price: {entry_price:.2f}")
                print(f"Current Price: {current_price:.2f}")
                print(f"Target Price: {target_price:.2f}")
                print(f"Entry to Current (Gross): {entry_to_current_pct:.2f}%")
                print(f"Total Required Gross Move (Net {PROFIT_TARGET_PERCENT}% + Fees {TOTAL_FEE_PERCENT}%): {entry_to_target_pct:.2f}%")
                print(f"Net Profit at Target (after fees): {PROFIT_TARGET_PERCENT:.2f}%")
            else:
                print("Error: Invalid entry or target price for percentage calculations.")

            print(f"Price Change from Entry: {entry_to_current_pct:.2f}%")
            print(f"Needed Gain to Target (Gross): {current_to_target_pct:.2f}%")

            print()

            # Exit logic (profit target)
            is_profit_exit = check_exit_condition(initial_investment, asset_balance, entry_price)
            
            if is_profit_exit:
                print(f"Profit Target Reached. Initiating exit...")
                if sell_asset(float(asset_balance)):
                    exit_usdc_balance = get_balance('USDC')
                    profit = exit_usdc_balance - initial_investment
                    profit_percentage = (profit / initial_investment) * Decimal('100') if initial_investment > Decimal('0.0') else Decimal('0.0')
                    print(f"Position closed. Sold BTC for USDC: {exit_usdc_balance:.25f}")
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

            if signal_triggered:
                usdc_balance = get_balance('USDC')
                if usdc_balance > Decimal('0'):
                    print(f"\n!!! SIGNAL TRIGGERED: {true_conditions_count} CONDITIONS MET (>= {min_required}) + STRICT CONDITIONS MET - EXECUTING TRADE !!!")
                    print(f"Strict Conditions Check: RSI OS/OB (15s), Momentum (1m/15s), LRC, Aroon, Thresh (15s)")
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
                print(f"Only {true_conditions_count}/{min_required} conditions met (Required: {min_required}).")
                
                # List failed strict conditions
                failed_strict = [name for name, passed in strict_checks_list if not passed]

                if not all_strict_conditions_met:
                    print(f"STRICT CONDITIONS FAILED: {', '.join(failed_strict)}")

                print("Waiting for next iteration...")

        print(f"\nCurrent USDC balance: {usdc_balance:.25f}")
        print(f"Current BTC balance: {asset_balance:.25f} BTC")
        print(f"Current {TRADE_SYMBOL} price: {current_price:.25f}\n")

        del candle_map
        gc.collect()
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
    import traceback
    traceback.print_exc()
    print("Attempting to save state before exit...")
    gc.collect()
    print("Bot shutdown due to error.")