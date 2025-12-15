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
PROFIT_TARGET_PERCENT = 0.35  # 0.35% profit target
TOTAL_FEE_PERCENT = 0.22  # Total fee percentage
MIN_TRADE_AMOUNT = 10

# Updated configurable conditions - Now includes FFT + Aroon + ML
CONFIG = {
    "conditions": {
        "momentum_positive_1m": True,  # Momentum Positive (1m)
        "momentum_positive_15sec": True, # Momentum Positive (15s)
        "linear_regression_forecast_15s": True,  # Linear Regression Forecast (15s)
        "linear_regression_channel_break": True,  # Linear Regression Channel Break condition
        "fft_prediction_1m": True,  # FFT Prediction (1m)
        "fft_aroon_ml_reversal": True,  # FFT + Aroon + ML Reversal
    },
    "min_conditions_met": 6  # ALL 6 conditions must be met to trigger a trade
}

# Global variables to track reversal history
last_reversal_type = None
last_reversal_time = None

# =========================================================
# 1. FFT CYCLE + PHASE EXTRACTION
# =========================================================

def fft_cycle_phase(close, keep_ratio=0.10):
    """
    Extract dominant cycle and its slope using FFT.
    Optimized for fast scalping (noise suppression).
    """
    close = np.asarray(close, dtype=np.float64)
    n = len(close)

    if n < 16:
        return 0.0, 0.0

    detrended = close - np.mean(close)
    fft_vals = np.fft.fft(detrended)
    magnitudes = np.abs(fft_vals)

    threshold = np.quantile(magnitudes, 1 - keep_ratio)
    fft_vals[magnitudes < threshold] = 0

    cycle = np.real(np.fft.ifft(fft_vals))

    cycle_value = cycle[-1]
    cycle_slope = cycle[-1] - cycle[-2]

    return cycle_value, cycle_slope


# =========================================================
# 2. AROON (TIME-BASED EXHAUSTION)
# =========================================================

def aroon(high, low, period=14):
    """
    Ultra-fast Aroon calculation.
    """
    if len(high) < period:
        return None, None

    window_high = high[-period:]
    window_low = low[-period:]

    hh_index = np.argmax(window_high)
    ll_index = np.argmin(window_low)

    aroon_up = (period - (period - hh_index)) / period * 100.0
    aroon_down = (period - (period - ll_index)) / period * 100.0

    return aroon_up, aroon_down


# =========================================================
# 3. LIGHTWEIGHT ML (LOGISTIC PROBABILITY MODEL)
# =========================================================

def ml_reversal_probability(features):
    """
    Fixed logistic model (hand-tuned for scalping).
    No sklearn, no retraining lag.
    """
    weights = np.array([1.5, 1.1, 0.9, 0.6])
    bias = -0.20

    z = np.dot(weights, features) + bias
    probability = 1.0 / (1.0 + math.exp(-z))

    return probability


# =========================================================
# 4. FFT + AROON + ML REVERSAL ANALYSIS (ENHANCED)
# =========================================================

def analyze_fft_aroon_ml_reversal(candles, aroon_period=14):
    """
    Enhanced FFT + Aroon + ML Reversal analysis with:
    - Current trend detection (UP or DOWN cycle)
    - Forecast price for reversal
    - Confirmation at exact reversal point (DIP or TOP)
    - Continuation detection for sideways markets
    
    candles = list of dicts:
    {
        "open": float,
        "high": float,
        "low": float,
        "close": float
    }
    """
    global last_reversal_type, last_reversal_time

    if len(candles) < 20:
        return {"error": "Not enough candles", "condition_met": False}

    close = [c["close"] for c in candles]
    high = [c["high"] for c in candles]
    low = [c["low"] for c in candles]

    # FFT
    cycle_value, cycle_slope = fft_cycle_phase(close)

    # Aroon
    aroon_up, aroon_down = aroon(high, low, aroon_period)
    if aroon_up is None:
        return {"error": "Aroon unavailable", "condition_met": False}

    # Feature engineering
    price_vs_cycle = close[-1] - cycle_value
    aroon_diff = (aroon_up - aroon_down) / 100.0
    aroon_strength = abs(aroon_diff)

    features = np.array([
        cycle_slope,
        price_vs_cycle,
        aroon_diff,
        aroon_strength
    ])

    probability = ml_reversal_probability(features)
    
    # Determine current trend
    current_trend = "DOWN" if cycle_slope < 0 and aroon_down > aroon_up else "UP" if cycle_slope > 0 and aroon_up > aroon_down else "SIDEWAYS"
    
    # Calculate reversal forecast price
    # For a DOWN to UP reversal, forecast a higher price
    # For an UP to DOWN reversal, forecast a lower price
    current_price = close[-1]
    reversal_strength = abs(cycle_slope) * 10  # Scale the slope to estimate reversal magnitude
    
    if current_trend == "DOWN":
        # DOWN to UP reversal - forecast higher price
        reversal_forecast = current_price + (reversal_strength * 0.5)  # Conservative estimate
        reversal_type = "DIP REVERSAL (DOWN to UP)"
        incoming_reversal_type = "UP CYCLE"
    elif current_trend == "UP":
        # UP to DOWN reversal - forecast lower price
        reversal_forecast = current_price - (reversal_strength * 0.5)
        reversal_type = "TOP REVERSAL (UP to DOWN)"
        incoming_reversal_type = "DOWN CYCLE"
    else:
        # SIDEWAYS market - analyze recent price action and cycle slope
        # Check if recent price action suggests upward movement
        price_change_5 = close[-1] - close[-6] if len(close) >= 6 else close[-1] - close[0]
        
        # Check if cycle slope is slightly positive or negative
        slope_direction = "UPWARD" if cycle_slope > 0.01 else "DOWNWARD" if cycle_slope < -0.01 else "FLAT"
        
        # Determine likely continuation direction
        if price_change_5 > 0 and slope_direction == "UPWARD":
            # Price is rising and slope is upward - likely to continue UP
            reversal_forecast = current_price + (reversal_strength * 0.2)  # Smaller forecast for continuation
            reversal_type = "CONTINUATION UP"
            incoming_reversal_type = "UP CYCLE"
        elif price_change_5 < 0 and slope_direction == "DOWNWARD":
            # Price is falling and slope is downward - likely to continue DOWN
            reversal_forecast = current_price - (reversal_strength * 0.2)
            reversal_type = "CONTINUATION DOWN"
            incoming_reversal_type = "DOWN CYCLE"
        else:
            # Mixed signals - use dominant recent direction
            recent_prices = close[-10:] if len(close) >= 10 else close
            if sum(1 for p in recent_prices if p > recent_prices[0]) > sum(1 for p in recent_prices if p < recent_prices[0]):
                # More upward movement than downward
                reversal_forecast = current_price + (reversal_strength * 0.2)
                reversal_type = "CONTINUATION UP"
                incoming_reversal_type = "UP CYCLE"
            else:
                # More downward movement than upward
                reversal_forecast = current_price - (reversal_strength * 0.2)
                reversal_type = "CONTINUATION DOWN"
                incoming_reversal_type = "DOWN CYCLE"
    
    # =====================================================
    # REVERSAL CONFIRMATION LOGIC (DIP OR TOP)
    # =====================================================
    
    # Check for trend exhaustion (the current trend is ending)
    trend_exhaustion = False
    if current_trend == "DOWN":
        # DOWN trend exhaustion: high Aroon Down, low Aroon Up, but cycle slope starting to turn positive
        trend_exhaustion = (aroon_down > 70 and aroon_up < 30 and cycle_slope > 0)
    elif current_trend == "UP":
        # UP trend exhaustion: high Aroon Up, low Aroon Down, but cycle slope starting to turn negative
        trend_exhaustion = (aroon_up > 70 and aroon_down < 30 and cycle_slope < 0)
    
    # Check for early signs of new trend (the reversal is starting)
    new_trend_emerging = False
    if current_trend == "DOWN":
        # Early signs of UP cycle: cycle slope turning positive, price starting to rise above cycle
        new_trend_emerging = (cycle_slope > 0 and price_vs_cycle > 0)
    elif current_trend == "UP":
        # Early signs of DOWN cycle: cycle slope turning negative, price starting to fall below cycle
        new_trend_emerging = (cycle_slope < 0 and price_vs_cycle < 0)
    
    # Reversal confirmation - requires both trend exhaustion and new trend emerging
    reversal_confirmed = (
        probability > 0.7 and 
        trend_exhaustion and 
        new_trend_emerging
    )
    
    # =====================================================
    # DECISION LOGIC (DIP, TOP, OR CONTINUATION)
    # =====================================================

    # Generate a LONG signal for dip reversal confirmation
    long_signal = (
        reversal_confirmed and
        current_trend == "DOWN" and  # Must be in a downtrend (dip)
        reversal_type == "DIP REVERSAL (DOWN to UP)" # Must be a reversal to an uptrend
    )

    # Generate a SHORT signal for top reversal confirmation
    short_signal = (
        reversal_confirmed and
        current_trend == "UP" and  # Must be in an uptrend (top)
        reversal_type == "TOP REVERSAL (UP to DOWN)" # Must be a reversal to a downtrend
    )

    # Generate a CONTINUATION signal for sideways markets
    continuation_signal = (
        current_trend == "SIDEWAYS" and
        (reversal_type == "CONTINUATION UP" or reversal_type == "CONTINUATION DOWN")
    )

    if long_signal:
        signal = "LONG"
        condition_met = True
        # Update the last reversal tracking variables
        last_reversal_type = "DIP"
        last_reversal_time = datetime.datetime.now()
    elif short_signal:
        signal = "SHORT"
        condition_met = True
        # Update the last reversal tracking variables
        last_reversal_type = "TOP"
        last_reversal_time = datetime.datetime.now()
    elif continuation_signal:
        signal = "CONTINUATION"
        condition_met = True
        # Update the last reversal tracking variables
        last_reversal_type = "CONTINUATION"
        last_reversal_time = datetime.datetime.now()
    else:
        signal = "NO TRADE"
        condition_met = False

    return {
        "signal": signal,
        "condition_met": condition_met,
        "probability": round(probability, 4),
        "cycle_slope": round(cycle_slope, 6),
        "price_vs_cycle": round(price_vs_cycle, 6),
        "aroon_up": round(aroon_up, 2),
        "aroon_down": round(aroon_down, 2),
        "current_trend": current_trend,
        "reversal_type": reversal_type,
        "reversal_forecast": round(reversal_forecast, 2),
        "trend_exhaustion": trend_exhaustion,
        "new_trend_emerging": new_trend_emerging,
        "reversal_confirmed": reversal_confirmed,
        "last_reversal_type": last_reversal_type,
        "last_reversal_time": last_reversal_time.strftime("%Y-%m-%d %H:%M:%S") if last_reversal_time else "None",
        "incoming_reversal_type": incoming_reversal_type,
        "logic": "Reversal confirmation: Detects the end of a trend and the start of a new trend"
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
                time.sleep(delay * (attempt + 1))
        except Exception as e:
            print(f"Unexpected error fetching candles for {timeframe} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    print(f"Failed to fetch candles for {timeframe} after {retries} attempts.")
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
    target_value = initial_investment * Decimal('1.0035')  # 0.35% profit target
    target_price = target_value / asset_balance
    print(f"Exit Check: Current Price: {current_price:.25f}, Target Price: {target_price:.25f}, Current Value: {current_value:.25f}, Target Value: {target_value:.25f}")
    return current_price >= target_price

# Analysis Functions
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

def analyze_linear_regression_forecast_15s(candles_1m):
    """
    Analyze linear regression forecast for 15-second timeframe derived from 1-minute data.
    Forecast direction is enough (no threshold check).
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
        
        # Extract close prices for linear regression
        close_prices = df_15s['close'].values
        n = len(close_prices)
        
        # Need at least 2 points for linear regression
        if n < 2:
            return {"error": "Insufficient data for linear regression"}
        
        # Create x values (time indices)
        x = np.arange(n)
        
        # Calculate linear regression coefficients
        # y = mx + b where m is slope, b is intercept
        coeffs = np.polyfit(x, close_prices, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # Forecast next value (at index n)
        forecast_target = slope * n + intercept
        current_price = close_prices[-1]
        
        # Calculate percentage difference
        forecast_diff_pct = ((forecast_target - current_price) / current_price) * 100
        
        # Determine if forecast is up (positive)
        forecast_up = forecast_target > current_price
        
        # Calculate R-squared to measure fit quality
        y_pred = slope * x + intercept
        ss_res = np.sum((close_prices - y_pred) ** 2)
        ss_tot = np.sum((close_prices - np.mean(close_prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "timeframe": "15s",
            "current_price": current_price,
            "forecast_target": forecast_target,
            "forecast_diff_pct": forecast_diff_pct,
            "forecast_up": forecast_up,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "data_points": n,
            "fit_quality": "Excellent" if r_squared > 0.9 else "Good" if r_squared > 0.7 else "Fair" if r_squared > 0.5 else "Poor",
            "forecast_direction": "Upward" if forecast_up else "Downward"
        }
        
    except Exception as e:
        print(f"Error analyzing linear regression forecast (15s): {e}")
        return {"error": str(e)}

def calculate_linear_regression_channel(candles, period=500, dev_multiplier=2.0):
    """
    Calculate Linear Regression Channel similar to Pine Script implementation.
    Returns upper and lower channel lines and channel break information.
    """
    try:
        if not candles or len(candles) < period:
            return {"error": "Insufficient data for linear regression channel"}
        
        # Extract close prices
        close_prices = np.array([candle["close"] for candle in candles[-period:]], dtype=np.float64)
        n = len(close_prices)
        
        # Create x values (time indices)
        x = np.arange(n)
        
        # Calculate linear regression coefficients
        coeffs = np.polyfit(x, close_prices, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # Calculate regression line values
        regression_line = slope * x + intercept
        
        # Calculate standard deviation of residuals
        residuals = close_prices - regression_line
        std_dev = np.std(residuals)
        
        # Calculate upper and lower channel lines
        upper_channel = regression_line + dev_multiplier * std_dev
        lower_channel = regression_line - dev_multiplier * std_dev
        
        # Get current price and channel values at the most recent point
        current_price = close_prices[-1]
        current_upper = upper_channel[-1]
        current_lower = lower_channel[-1]
        
        # Check for channel breaks
        above_upper = current_price > current_upper
        below_lower = current_price < current_lower
        
        # Check for recent channel crossings and track their indices
        # Expand the detection window to look at more history
        detection_window = min(100, n-1)  # Look at up to 100 periods
        
        below_lower_indices = []
        above_upper_indices = []
        
        # Check if we crossed below the lower channel
        for i in range(detection_window):
            if close_prices[n-2-i] >= lower_channel[n-2-i] and close_prices[n-1-i] < lower_channel[n-1-i]:
                below_lower_indices.append(n-1-i)
        
        # Check if we crossed above the upper channel
        for i in range(detection_window):
            if close_prices[n-2-i] <= upper_channel[n-2-i] and close_prices[n-1-i] > upper_channel[n-1-i]:
                above_upper_indices.append(n-1-i)
        
        # Determine which crossing was most recent
        most_recent_below_lower = len(below_lower_indices) > 0
        most_recent_above_upper = len(above_upper_indices) > 0
        
        # If both types of crossings occurred, find which was most recent
        if most_recent_below_lower and most_recent_above_upper:
            most_recent_below_lower_index = max(below_lower_indices)
            most_recent_above_upper_index = max(above_upper_indices)
            
            # The most recent crossing is the one with the higher index
            most_recent_below_lower = most_recent_below_lower_index > most_recent_above_upper_index
            most_recent_above_upper = most_recent_above_upper_index > most_recent_below_lower_index
        
        # Determine the trading signal based on the most recent crossing
        # According to the user's clarification:
        # - When most recent was close below the lowest line (compared to close above the highest line), it's an up cycle
        # - When most recent was close above the upper line, it's a down cycle
        up_cycle = most_recent_below_lower and not most_recent_above_upper
        
        return {
            "current_price": current_price,
            "current_upper": current_upper,
            "current_lower": current_lower,
            "above_upper": above_upper,
            "below_lower": below_lower,
            "most_recent_below_lower": most_recent_below_lower,
            "most_recent_above_upper": most_recent_above_upper,
            "up_cycle": up_cycle,
            "slope": slope,
            "intercept": intercept,
            "std_dev": std_dev,
            "period": period,
            "dev_multiplier": dev_multiplier,
            "detection_window": detection_window
        }
        
    except Exception as e:
        print(f"Error calculating linear regression channel: {e}")
        return {"error": str(e)}

def analyze_linear_regression_channel_break(candles_1m):
    """
    Analyze linear regression channel break condition.
    Returns True if the most recent occurrence between close below the lowest line and 
    close above the highest line is close below the lowest line of channel (indicating up cycle).
    """
    try:
        if not candles_1m:
            return {"error": "No 1m data provided"}
        
        # Calculate linear regression channel
        lrc_result = calculate_linear_regression_channel(candles_1m, period=500, dev_multiplier=2.0)
        
        if "error" in lrc_result:
            return lrc_result
        
        # The condition is met if up_cycle is True
        condition_met = lrc_result["up_cycle"]
        
        # Create a dynamic description based on the actual analysis
        if lrc_result["most_recent_below_lower"] and not lrc_result["most_recent_above_upper"]:
            description = "Most recent occurrence was a close below the lowest line (indicating upward cycle)"
        elif lrc_result["most_recent_above_upper"] and not lrc_result["most_recent_below_lower"]:
            description = "Most recent occurrence was a close above the highest line (indicating downward cycle)"
        elif lrc_result["most_recent_below_lower"] and lrc_result["most_recent_above_upper"]:
            # If both are true, check which happened more recently
            if lrc_result["up_cycle"]:
                description = "Most recent occurrence was a close below the lowest line (indicating upward cycle)"
            else:
                description = "Most recent occurrence was a close above the highest line (indicating downward cycle)"
        else:
            # If no recent breaks detected, look for any breaks in the entire period
            # This is a fallback to ensure we always find some breaks
            close_prices = np.array([candle["close"] for candle in candles_1m[-500:]], dtype=np.float64)
            n = len(close_prices)
            
            # Recalculate the channel lines
            x = np.arange(n)
            coeffs = np.polyfit(x, close_prices, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            regression_line = slope * x + intercept
            residuals = close_prices - regression_line
            std_dev = np.std(residuals)
            upper_channel = regression_line + 2.0 * std_dev
            lower_channel = regression_line - 2.0 * std_dev
            
            # Look for any breaks in the entire period
            below_lower_indices = []
            above_upper_indices = []
            
            for i in range(1, n):
                if close_prices[i-1] >= lower_channel[i-1] and close_prices[i] < lower_channel[i]:
                    below_lower_indices.append(i)
                if close_prices[i-1] <= upper_channel[i-1] and close_prices[i] > upper_channel[i]:
                    above_upper_indices.append(i)
            
            if below_lower_indices and above_upper_indices:
                # Find the most recent break
                most_recent_below_lower_index = max(below_lower_indices)
                most_recent_above_upper_index = max(above_upper_indices)
                
                if most_recent_below_lower_index > most_recent_above_upper_index:
                    condition_met = True
                    description = "Most recent occurrence in the entire period was a close below the lowest line (indicating upward cycle)"
                else:
                    condition_met = False
                    description = "Most recent occurrence in the entire period was a close above the highest line (indicating downward cycle)"
            elif below_lower_indices:
                condition_met = True
                description = "Most recent occurrence in the entire period was a close below the lowest line (indicating upward cycle)"
            elif above_upper_indices:
                condition_met = False
                description = "Most recent occurrence in the entire period was a close above the highest line (indicating downward cycle)"
            else:
                # If still no breaks found, use the current position relative to the channel
                if lrc_result["below_lower"]:
                    condition_met = True
                    description = "Currently below the lower channel (indicating upward cycle)"
                elif lrc_result["above_upper"]:
                    condition_met = False
                    description = "Currently above the upper channel (indicating downward cycle)"
                else:
                    # As a last resort, use the slope of the regression line
                    if lrc_result["slope"] > 0:
                        condition_met = True
                        description = "No channel breaks detected, but regression slope is positive (indicating upward trend)"
                    else:
                        condition_met = False
                        description = "No channel breaks detected, and regression slope is negative or flat (indicating downward trend)"
        
        return {
            "condition_met": condition_met,
            "current_price": lrc_result["current_price"],
            "upper_channel": lrc_result["current_upper"],
            "lower_channel": lrc_result["current_lower"],
            "above_upper": lrc_result["above_upper"],
            "below_lower": lrc_result["below_lower"],
            "most_recent_below_lower": lrc_result["most_recent_below_lower"],
            "most_recent_above_upper": lrc_result["most_recent_above_upper"],
            "up_cycle": lrc_result["up_cycle"],
            "description": description
        }
        
    except Exception as e:
        print(f"Error analyzing linear regression channel break: {e}")
        return {"error": str(e)}

def predict_prices_with_fft(candles, num_predictions=5, filter_ratio=0.8):
    """
    Predict future prices using Fast Fourier Transform.
    Heavily modified for ultra short-term prediction (next minute only).
    
    Args:
        candles: List of candle data with 'close' prices
        num_predictions: Number of future values to predict (default: 5)
        filter_ratio: Ratio of frequency components to keep (0.8 = keep top 80%)
    
    Returns:
        Dictionary with prediction results
    """
    try:
        # Use only the most recent 15 candles for ultra short-term prediction
        if not candles or len(candles) < 10:
            return {"error": "Insufficient data for FFT prediction"}
        
        # Extract only the most recent candles
        recent_candles = candles[-15:] if len(candles) > 15 else candles
        close_prices = np.array([candle["close"] for candle in recent_candles], dtype=np.float64)
        
        # Use a very small window for detrending (3-5 candles)
        window_size = min(3, len(close_prices) // 3)
        if window_size < 3:
            window_size = 3
        
        # Calculate moving average
        weights = np.ones(window_size) / window_size
        moving_avg = np.convolve(close_prices, weights, mode='valid')
        
        # Pad the moving average to match the original length
        padding = np.zeros(len(close_prices) - len(moving_avg))
        moving_avg = np.concatenate([padding, moving_avg])
        
        # Detrend the data
        detrended = close_prices - moving_avg
        
        # Apply FFT
        fft_values = np.fft.fft(detrended)
        
        # For ultra short-term prediction, keep most of the signal
        fft_abs = np.abs(fft_values)
        threshold = np.max(fft_abs) * (1 - filter_ratio)  # Keep 80% of frequencies
        
        # Create a mask for frequencies above threshold
        mask = fft_abs > threshold
        
        # Apply mask to filter out noise
        filtered_fft = fft_values * mask
        
        # Predict just ONE step ahead
        extended_fft = np.zeros(len(filtered_fft) + 1, dtype=complex)
        extended_fft[:len(filtered_fft)] = filtered_fft
        
        # Apply Inverse FFT
        extended_detrended = np.fft.ifft(extended_fft).real
        
        # Get the last moving average value
        last_ma_value = moving_avg[-1]
        
        # Calculate the next predicted value (just one step)
        next_pred_detrended = extended_detrended[-1]
        next_prediction = next_pred_detrended + last_ma_value
        
        # Apply very strict constraints to prevent unrealistic predictions
        current_price = close_prices[-1]
        
        # Limit the maximum change to 0.1% for the next minute
        max_change_pct = 0.001  # 0.1% maximum change
        
        # Calculate the percentage change
        change_pct = (next_prediction - current_price) / current_price
        
        # Constrain the prediction
        if abs(change_pct) > max_change_pct:
            # Keep the direction but limit the magnitude
            constrained_prediction = current_price * (1 + max_change_pct if change_pct > 0 else 1 - max_change_pct)
        else:
            constrained_prediction = next_prediction
        
        # Create multiple predictions by slightly varying the constrained prediction
        # This gives us the 5 values requested but keeps them very close together
        predictions = []
        for i in range(num_predictions):
            # Small variations that decrease with each step
            variation = (1 - (i * 0.02)) if change_pct > 0 else (1 + (i * 0.02))
            predictions.append(constrained_prediction * variation)
        
        # Calculate confidence based on the proportion of retained frequency components
        confidence = np.sum(mask) / len(mask)
        
        # Calculate prediction direction (up/down)
        prediction_direction = "Up" if predictions[0] > current_price else "Down"
        
        # Calculate percentage change for the first prediction
        pct_change = ((predictions[0] - current_price) / current_price) * 100
        
        return {
            "current_price": current_price,
            "predictions": predictions,
            "num_predictions": num_predictions,
            "confidence": confidence,
            "prediction_direction": prediction_direction,
            "pct_change": pct_change,
            "filter_ratio": filter_ratio,
            "window_size": window_size,
            "data_points": len(close_prices),
            "max_change_pct": max_change_pct * 100  # Convert to percentage
        }
        
    except Exception as e:
        print(f"Error in FFT prediction: {e}")
        return {"error": str(e)}

def analyze_fft_prediction_1m(candles_1m):
    """
    Analyze FFT prediction for 1-minute timeframe.
    Modified for ultra short-term prediction (next minute).
    Returns True if prediction is upward.
    """
    try:
        if not candles_1m:
            return {"error": "No 1m data provided"}
        
        # Use only the most recent 15 candles for ultra short-term prediction
        recent_candles = candles_1m[-15:] if len(candles_1m) > 15 else candles_1m
        
        # Predict next values using FFT with modified parameters
        fft_result = predict_prices_with_fft(recent_candles, num_predictions=5, filter_ratio=0.8)
        
        if "error" in fft_result:
            return fft_result
        
        # The condition is met if prediction is upward
        condition_met = fft_result["prediction_direction"] == "Up"
        
        return {
            "condition_met": condition_met,
            "current_price": fft_result["current_price"],
            "predictions": fft_result["predictions"],
            "prediction_direction": fft_result["prediction_direction"],
            "pct_change": fft_result["pct_change"],
            "confidence": fft_result["confidence"],
            "max_change_pct": fft_result["max_change_pct"],
            "description": "FFT predicts ultra short-term upward price movement (next minute)"
        }
        
    except Exception as e:
        print(f"Error analyzing FFT prediction: {e}")
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

        # Fetch candles for all timeframes
        candle_map = fetch_candles_in_parallel(['1m'])
        
        # Also fetch 1m candles for 15s analysis
        candles_1m = get_candles(TRADE_SYMBOL, '1m', limit=1200)
        
        if not candle_map.get('1m'):
            print("Error: '1m' candles not fetched. Check API connectivity or symbol.")
        if current_price == Decimal('0.0'):
            print(f"Warning: Current {TRADE_SYMBOL} price is {current_price:.25f}. API may be failing.")

        # Initialize all condition results - Updated with remaining conditions
        conditions_status = {
            "momentum_positive_1m": False,
            "momentum_positive_15sec": False,
            "linear_regression_forecast_15s": False,
            "linear_regression_channel_break": False,
            "fft_prediction_1m": False,
            "fft_aroon_ml_reversal": False,
        }

        # Print conditions in order
        print("\n" + "="*80)
        print("TRADING CONDITIONS")
        print("="*80)
        
        # Condition 1: Momentum Positive (1m)
        print("\n--- Condition 1: Momentum Positive (1m) ---")
        momentum_1m_positive, momentum_1m_value, momentum_1m_details = calculate_momentum(candle_map['1m'])
        conditions_status["momentum_positive_1m"] = momentum_1m_positive
        print(f"Current Momentum: {momentum_1m_value:.4f}")
        print(f"Momentum Period: {momentum_1m_details.get('period', 10)}")
        print(f"Momentum Direction: {'Positive' if momentum_1m_positive else 'Negative'}")
        print(f"Momentum Strength: {'Strong' if abs(momentum_1m_value) > 100 else 'Moderate' if abs(momentum_1m_value) > 50 else 'Weak'}")
        print(f"Momentum Positive: {momentum_1m_positive}")
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
            print(f"Momentum Positive: {momentum_15s_result['momentum_positive']}")
            print(f"Condition Met: {conditions_status['momentum_positive_15sec']}")
        else:
            print(f"Error analyzing momentum (15s): {momentum_15s_result['error']}")
            print(f"Condition Met: {conditions_status['momentum_positive_15sec']}")
        
        # Condition 3: Linear Regression Forecast (15s)
        print("\n--- Condition 3: Linear Regression Forecast (15s) ---")
        linear_regression_result = analyze_linear_regression_forecast_15s(candles_1m)
        if 'error' not in linear_regression_result:
            conditions_status["linear_regression_forecast_15s"] = linear_regression_result['forecast_up']
            print(f"Current Price: {linear_regression_result['current_price']:.2f}")
            print(f"Forecast Target: {linear_regression_result['forecast_target']:.2f}")
            print(f"Forecast Difference: {linear_regression_result['forecast_diff_pct']:.4f}%")
            print(f"Slope: {linear_regression_result['slope']:.6f}")
            print(f"Intercept: {linear_regression_result['intercept']:.2f}")
            print(f"R-squared: {linear_regression_result['r_squared']:.4f}")
            print(f"Fit Quality: {linear_regression_result['fit_quality']}")
            print(f"Data Points: {linear_regression_result['data_points']}")
            print(f"Forecast Direction: {linear_regression_result['forecast_direction']}")
            print(f"Forecast Up: {linear_regression_result['forecast_up']}")
            print(f"Condition Met: {conditions_status['linear_regression_forecast_15s']}")
        else:
            print(f"Error analyzing linear regression forecast (15s): {linear_regression_result['error']}")
            print(f"Condition Met: {conditions_status['linear_regression_forecast_15s']}")
        
        # Condition 4: Linear Regression Channel Break
        print("\n--- Condition 4: Linear Regression Channel Break ---")
        lrc_break_result = analyze_linear_regression_channel_break(candles_1m)
        if 'error' not in lrc_break_result:
            conditions_status["linear_regression_channel_break"] = lrc_break_result['condition_met']
            print(f"Current Price: {lrc_break_result['current_price']:.2f}")
            print(f"Upper Channel: {lrc_break_result['upper_channel']:.2f}")
            print(f"Lower Channel: {lrc_break_result['lower_channel']:.2f}")
            print(f"Above Upper Channel: {lrc_break_result['above_upper']}")
            print(f"Below Lower Channel: {lrc_break_result['below_lower']}")
            print(f"Most Recent Below Lower: {lrc_break_result['most_recent_below_lower']}")
            print(f"Most Recent Above Upper: {lrc_break_result['most_recent_above_upper']}")
            print(f"Up Cycle: {lrc_break_result['up_cycle']}")
            print(f"Description: {lrc_break_result['description']}")
            print(f"Condition Met: {conditions_status['linear_regression_channel_break']}")
        else:
            print(f"Error analyzing linear regression channel break: {lrc_break_result['error']}")
            print(f"Condition Met: {conditions_status['linear_regression_channel_break']}")

        # Condition 5: FFT Prediction (1m)
        print("\n--- Condition 5: FFT Prediction (1m) ---")
        fft_prediction_result = analyze_fft_prediction_1m(candles_1m)
        if 'error' not in fft_prediction_result:
            conditions_status["fft_prediction_1m"] = fft_prediction_result['condition_met']
            print(f"Current Price: {fft_prediction_result['current_price']:.2f}")
            print(f"Next Prediction: {fft_prediction_result['predictions'][0]:.2f}")
            print(f"Prediction Direction: {fft_prediction_result['prediction_direction']}")
            print(f"Percentage Change: {fft_prediction_result['pct_change']:.4f}%")
            print(f"Confidence: {fft_prediction_result['confidence']:.4f}")
            print(f"Condition Met: {conditions_status['fft_prediction_1m']}")
        else:
            print(f"Error analyzing FFT prediction: {fft_prediction_result['error']}")
            print(f"Condition Met: {conditions_status['fft_prediction_1m']}")
            
        # Condition 6: FFT + Aroon + ML Reversal (ENHANCED)
        print("\n--- Condition 6: FFT + Aroon + ML Reversal (ENHANCED) ---")
        fft_aroon_ml_result = analyze_fft_aroon_ml_reversal(candles_1m)
        if 'error' not in fft_aroon_ml_result:
            conditions_status["fft_aroon_ml_reversal"] = fft_aroon_ml_result['condition_met']
            print(f"Signal: {fft_aroon_ml_result['signal']}")
            print(f"Probability: {fft_aroon_ml_result['probability']}")
            print(f"Cycle Slope: {fft_aroon_ml_result['cycle_slope']}")
            print(f"Price vs Cycle: {fft_aroon_ml_result['price_vs_cycle']}")
            print(f"Aroon Up: {fft_aroon_ml_result['aroon_up']}")
            print(f"Aroon Down: {fft_aroon_ml_result['aroon_down']}")
            print(f"Current Trend: {fft_aroon_ml_result['current_trend']}")
            print(f"Reversal Type: {fft_aroon_ml_result['reversal_type']}")
            print(f"Reversal Forecast: {fft_aroon_ml_result['reversal_forecast']}")
            print(f"Trend Exhaustion: {fft_aroon_ml_result['trend_exhaustion']}")
            print(f"New Trend Emerging: {fft_aroon_ml_result['new_trend_emerging']}")
            print(f"Reversal Confirmed: {fft_aroon_ml_result['reversal_confirmed']}")
            print(f"Last Reversal Type: {fft_aroon_ml_result['last_reversal_type']}")
            print(f"Last Reversal Time: {fft_aroon_ml_result['last_reversal_time']}")
            print(f"Incoming Reversal Type: {fft_aroon_ml_result['incoming_reversal_type']}")
            print(f"Condition Met: {conditions_status['fft_aroon_ml_reversal']}")
        else:
            print(f"Error analyzing FFT + Aroon + ML Reversal: {fft_aroon_ml_result['error']}")
            print(f"Condition Met: {conditions_status['fft_aroon_ml_reversal']}")

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

            target_value = initial_investment * Decimal('1.0035')  # 0.35% profit target
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
                    entry_datetime = None
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
