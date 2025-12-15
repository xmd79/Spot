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

# Updated configurable conditions
CONFIG = {
    "conditions": {
        "momentum_positive_1m": True,
        "momentum_positive_15sec": True,
        "linear_regression_forecast_15s": True,
        "linear_regression_channel_break": True,
        "fft_prediction_1m": True,
        "fft_aroon_ml_reversal": True,
        "ema_crossover_orderflow": True,  # New condition
    },
    "min_conditions_met": 7
}

# Global variables for market state tracking
last_reversal_type = None
last_reversal_time = None
current_major_trend = "UNKNOWN"
last_major_high = None
last_major_low = None
resistance_level = None
support_level = None
market_cycle_position = 0

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
    target_value = initial_investment * Decimal('1.0035')
    target_price = target_value / asset_balance
    print(f"Exit Check: Current Price: {current_price:.25f}, Target Price: {target_price:.25f}, Current Value: {current_value:.25f}, Target Value: {target_value:.25f}")
    return current_price >= target_price

# =========================================================
# EMA CROSSOVER WITH ORDER FLOW ANALYSIS FUNCTIONS
# =========================================================

def get_order_book(symbol=TRADE_SYMBOL, limit=100):
    """
    Fetch order book for a symbol.
    """
    try:
        depth = client.get_order_book(symbol=symbol, limit=limit)
        return depth
    except BinanceAPIException as e:
        print(f"Error fetching order book: {e.message}")
        return None
    except Exception as e:
        print(f"Unexpected error fetching order book: {e}")
        return None

def analyze_order_book(order_book, threshold_ratio=3.0):
    """
    Analyze order book for large orders and potential price movements.
    
    Args:
        order_book: Order book data from Binance
        threshold_ratio: Minimum ratio of a large order to average nearby orders
        
    Returns:
        Dictionary with order book analysis results
    """
    if not order_book or 'bids' not in order_book or 'asks' not in order_book:
        return {"error": "Invalid order book data"}
    
    try:
        bids = order_book['bids']
        asks = order_book['asks']
        
        # Calculate total volume at bid and ask sides
        total_bid_volume = sum(float(bid[1]) for bid in bids[:10])  # Top 10 bids
        total_ask_volume = sum(float(ask[1]) for ask in asks[:10])  # Top 10 asks
        
        # Find iceberg orders (unusually large orders at a single price)
        large_bids = []
        large_asks = []
        
        # Check for large bid orders
        for i, bid in enumerate(bids[:10]):
            bid_price = float(bid[0])
            bid_volume = float(bid[1])
            
            # Calculate average volume of nearby bids
            nearby_volumes = []
            for j in range(max(0, i-2), min(len(bids), i+3)):
                if j != i:
                    nearby_volumes.append(float(bids[j][1]))
            
            if nearby_volumes:
                avg_nearby_volume = sum(nearby_volumes) / len(nearby_volumes)
                if avg_nearby_volume > 0 and bid_volume / avg_nearby_volume > threshold_ratio:
                    large_bids.append({
                        "price": bid_price,
                        "volume": bid_volume,
                        "ratio": bid_volume / avg_nearby_volume
                    })
        
        # Check for large ask orders
        for i, ask in enumerate(asks[:10]):
            ask_price = float(ask[0])
            ask_volume = float(ask[1])
            
            # Calculate average volume of nearby asks
            nearby_volumes = []
            for j in range(max(0, i-2), min(len(asks), i+3)):
                if j != i:
                    nearby_volumes.append(float(asks[j][1]))
            
            if nearby_volumes:
                avg_nearby_volume = sum(nearby_volumes) / len(nearby_volumes)
                if avg_nearby_volume > 0 and ask_volume / avg_nearby_volume > threshold_ratio:
                    large_asks.append({
                        "price": ask_price,
                        "volume": ask_volume,
                        "ratio": ask_volume / avg_nearby_volume
                    })
        
        # Calculate bid-ask spread
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid) * 100
        
        # Determine order book imbalance
        if total_bid_volume + total_ask_volume > 0:
            bid_ratio = total_bid_volume / (total_bid_volume + total_ask_volume)
        else:
            bid_ratio = 0.5
        
        # Calculate volume-weighted average price (VWAP) from order book
        total_volume = total_bid_volume + total_ask_volume
        if total_volume > 0:
            vwap_bid = sum(float(bid[0]) * float(bid[1]) for bid in bids[:10]) / total_bid_volume if total_bid_volume > 0 else best_bid
            vwap_ask = sum(float(ask[0]) * float(ask[1]) for ask in asks[:10]) / total_ask_volume if total_ask_volume > 0 else best_ask
            vwap_mid = (vwap_bid + vwap_ask) / 2
        else:
            vwap_bid = best_bid
            vwap_ask = best_ask
            vwap_mid = (best_bid + best_ask) / 2
        
        # Check for buying pressure (more large bids than large asks)
        buying_pressure = len(large_bids) > len(large_asks)
        
        # Calculate order flow imbalance
        order_flow_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if total_bid_volume + total_ask_volume > 0 else 0
        
        return {
            "total_bid_volume": total_bid_volume,
            "total_ask_volume": total_ask_volume,
            "bid_ratio": bid_ratio,
            "spread": spread,
            "spread_pct": spread_pct,
            "large_bids": large_bids,
            "large_asks": large_asks,
            "buying_pressure": buying_pressure,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "vwap_mid": vwap_mid,
            "order_flow_imbalance": order_flow_imbalance,
            "strong_buy_signal": buying_pressure and order_flow_imbalance > 0.2,
            "strong_sell_signal": not buying_pressure and order_flow_imbalance < -0.2
        }
        
    except Exception as e:
        print(f"Error analyzing order book: {e}")
        return {"error": str(e)}

def calculate_vwap(candles):
    """
    Calculate Volume Weighted Average Price (VWAP) from candle data.
    
    Args:
        candles: List of candle data
        
    Returns:
        VWAP value
    """
    if not candles:
        return None
    
    try:
        total_volume = sum(candle["volume"] for candle in candles)
        if total_volume == 0:
            return None
        
        # Use typical price (HLC/3) for VWAP calculation
        vwap = sum(((candle["high"] + candle["low"] + candle["close"]) / 3) * candle["volume"] for candle in candles) / total_volume
        return vwap
    except Exception as e:
        print(f"Error calculating VWAP: {e}")
        return None

def analyze_ema_crossover_with_orderflow(candles, order_book=None):
    """
    Analyze EMA crossover signals with order book confirmation.
    
    Args:
        candles: List of 1-minute candle data
        order_book: Optional order book data
        
    Returns:
        Dictionary with analysis results and trading signals
    """
    try:
        if len(candles) < 30:  # Need enough data for EMAs
            return {"error": "Not enough candle data for EMA analysis"}
        
        # Extract close prices and volumes
        close_prices = np.array([candle["close"] for candle in candles], dtype=np.float64)
        volumes = np.array([candle["volume"] for candle in candles], dtype=np.float64)
        
        # Calculate EMAs using TALIB
        ema9 = talib.EMA(close_prices, timeperiod=9)
        ema21 = talib.EMA(close_prices, timeperiod=21)
        
        # Calculate SMAs for comparison
        sma9 = talib.SMA(close_prices, timeperiod=9)
        sma21 = talib.SMA(close_prices, timeperiod=21)
        
        # Calculate VWAP
        vwap = calculate_vwap(candles)
        
        # Current values
        current_price = close_prices[-1]
        current_ema9 = ema9[-1] if not np.isnan(ema9[-1]) else None
        current_ema21 = ema21[-1] if not np.isnan(ema21[-1]) else None
        current_sma9 = sma9[-1] if not np.isnan(sma9[-1]) else None
        current_sma21 = sma21[-1] if not np.isnan(sma21[-1]) else None
        
        # Previous values (to detect crossovers)
        prev_ema9 = ema9[-2] if len(ema9) > 1 and not np.isnan(ema9[-2]) else None
        prev_ema21 = ema21[-2] if len(ema21) > 1 and not np.isnan(ema21[-2]) else None
        
        # Calculate EMA distances
        ema_distance = current_ema9 - current_ema21 if current_ema9 and current_ema21 else None
        ema_distance_pct = (ema_distance / current_ema21) * 100 if current_ema9 and current_ema21 and current_ema21 != 0 else None
        
        # Detect potential crossover (close < EMA9 < EMA21)
        potential_crossover = (
            current_price < current_ema9 and 
            current_ema9 < current_ema21 and
            ema_distance_pct and abs(ema_distance_pct) < 0.5  # EMAs are close
        )
        
        # Detect actual crossover (EMA9 crossed above EMA21)
        actual_crossover = (
            prev_ema9 and prev_ema21 and current_ema9 and current_ema21 and
            prev_ema9 <= prev_ema21 and current_ema9 > current_ema21
        )
        
        # Detect actual crossunder (EMA9 crossed below EMA21)
        actual_crossunder = (
            prev_ema9 and prev_ema21 and current_ema9 and current_ema21 and
            prev_ema9 >= prev_ema21 and current_ema9 < current_ema21
        )
        
        # Volume confirmation (increasing volume)
        recent_volume_avg = np.mean(volumes[-10:]) if len(volumes) >= 10 else np.mean(volumes)
        current_volume = volumes[-1] if len(volumes) > 0 else 0
        volume_increasing = current_volume > recent_volume_avg * 1.2  # 20% above average
        
        # Calculate volume trend
        volume_trend = 0
        if len(volumes) >= 5:
            volume_trend = (volumes[-1] - volumes[-5]) / volumes[-5] if volumes[-5] > 0 else 0
        
        # Analyze order book if provided
        order_book_analysis = None
        buying_pressure = False
        strong_buy_signal = False
        strong_sell_signal = False
        
        if order_book:
            order_book_analysis = analyze_order_book(order_book)
            buying_pressure = order_book_analysis.get("buying_pressure", False)
            strong_buy_signal = order_book_analysis.get("strong_buy_signal", False)
            strong_sell_signal = order_book_analysis.get("strong_sell_signal", False)
        
        # Calculate price position relative to EMAs
        price_above_ema9 = current_price > current_ema9 if current_ema9 else False
        price_above_ema21 = current_price > current_ema21 if current_ema21 else False
        
        # Calculate EMA convergence rate
        ema_convergence_rate = 0
        if len(ema9) >= 5 and len(ema21) >= 5:
            current_diff = ema9[-1] - ema21[-1]
            prev_diff = ema9[-5] - ema21[-5]
            ema_convergence_rate = (current_diff - prev_diff) / abs(prev_diff) if prev_diff != 0 else 0
        
        # Generate signals
        signal = "NO TRADE"
        signal_strength = 0
        signal_reason = ""
        condition_met = False
        
        # Priority 1: Actual crossover with volume and order book confirmation
        if actual_crossover and volume_increasing and strong_buy_signal:
            signal = "STRONG BUY"
            signal_strength = 5
            signal_reason = "EMA9 crossed above EMA21 with volume and order book confirmation"
            condition_met = True
        
        # Priority 2: Actual crossover with volume confirmation
        elif actual_crossover and volume_increasing:
            signal = "BUY"
            signal_strength = 4
            signal_reason = "EMA9 crossed above EMA21 with volume confirmation"
            condition_met = True
        
        # Priority 3: Potential crossover with volume and order book confirmation
        elif potential_crossover and volume_increasing and strong_buy_signal:
            signal = "POTENTIAL BUY"
            signal_strength = 3
            signal_reason = "Potential EMA9 crossover above EMA21 with volume and order book confirmation"
            condition_met = True
        
        # Priority 4: Potential crossover with volume confirmation
        elif potential_crossover and volume_increasing:
            signal = "POTENTIAL BUY"
            signal_strength = 2
            signal_reason = "Potential EMA9 crossover above EMA21 with volume confirmation"
            condition_met = True
        
        # Priority 5: Strong buying pressure with price above EMAs
        elif strong_buy_signal and price_above_ema9 and price_above_ema21:
            signal = "BUY"
            signal_strength = 3
            signal_reason = "Strong buying pressure with price above EMAs"
            condition_met = True
        
        # Priority 6: Price below EMAs but EMAs converging upward
        elif not price_above_ema9 and ema_convergence_rate > 0.1 and volume_increasing:
            signal = "POTENTIAL BUY"
            signal_strength = 2
            signal_reason = "Price below EMAs but EMAs converging upward with volume"
            condition_met = True
        
        # Priority 7: Price above both EMAs with increasing volume
        elif price_above_ema9 and price_above_ema21 and volume_increasing:
            signal = "BUY"
            signal_strength = 2
            signal_reason = "Price above both EMAs with increasing volume"
            condition_met = True
        
        return {
            "signal": signal,
            "signal_strength": signal_strength,
            "signal_reason": signal_reason,
            "condition_met": condition_met,
            "current_price": current_price,
            "ema9": current_ema9,
            "ema21": current_ema21,
            "sma9": current_sma9,
            "sma21": current_sma21,
            "ema_distance": ema_distance,
            "ema_distance_pct": ema_distance_pct,
            "vwap": vwap,
            "potential_crossover": potential_crossover,
            "actual_crossover": actual_crossover,
            "actual_crossunder": actual_crossunder,
            "volume_increasing": volume_increasing,
            "current_volume": current_volume,
            "avg_volume": recent_volume_avg,
            "volume_trend": volume_trend,
            "order_book_analysis": order_book_analysis,
            "buying_pressure": buying_pressure,
            "strong_buy_signal": strong_buy_signal,
            "strong_sell_signal": strong_sell_signal,
            "price_above_ema9": price_above_ema9,
            "price_above_ema21": price_above_ema21,
            "ema_convergence_rate": ema_convergence_rate
        }
        
    except Exception as e:
        print(f"Error in EMA crossover analysis: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "condition_met": False}

# =========================================================
# ENHANCED TECHNICAL ANALYSIS FUNCTIONS
# =========================================================

def calculate_extrema_with_indices(prices, period=14):
    """
    Calculate recent highs and lows with their indices using argmax/argmin.
    """
    if len(prices) < period:
        return None, None, None, None
    
    # Get recent window
    recent_prices = prices[-period:]
    
    # Find indices within the recent window
    recent_high_idx = np.argmax(recent_prices)
    recent_low_idx = np.argmin(recent_prices)
    
    # Calculate actual indices in the full array
    actual_high_idx = len(prices) - period + recent_high_idx
    actual_low_idx = len(prices) - period + recent_low_idx
    
    # Get values
    recent_high = prices[actual_high_idx]
    recent_low = prices[actual_low_idx]
    
    # Calculate periods since high/low
    current_idx = len(prices) - 1
    periods_since_high = current_idx - actual_high_idx
    periods_since_low = current_idx - actual_low_idx
    
    return recent_high, recent_low, periods_since_high, periods_since_low, actual_high_idx, actual_low_idx

def calculate_thresholds_with_extrema(close_prices, period=14, minimum_percentage=3, maximum_percentage=3):
    """
    Calculate thresholds using recent extrema with indices.
    """
    try:
        if len(close_prices) < period:
            return None, None, None, None, None, None, None, None, None, None
        
        # Calculate recent extrema with indices
        recent_high, recent_low, periods_since_high, periods_since_low, high_idx, low_idx = calculate_extrema_with_indices(close_prices, period)
        
        if recent_high is None or recent_low is None:
            return None, None, None, None, None, None, None, None, None, None
        
        # Calculate thresholds based on recent extrema
        min_percentage = Decimal(str(minimum_percentage)) / Decimal('100')
        max_percentage = Decimal(str(maximum_percentage)) / Decimal('100')
        
        # Middle threshold (average of recent high and low)
        middle_threshold = Decimal(str((recent_high + recent_low) / 2))
        
        # Dynamic thresholds based on volatility
        recent_range = recent_high - recent_low
        min_threshold = Decimal(str(recent_low - (recent_range * float(min_percentage))))
        max_threshold = Decimal(str(recent_high + (recent_range * float(max_percentage))))
        
        # Calculate momentum
        close_array = np.array(close_prices, dtype=np.float64)
        momentum = talib.MOM(close_array, timeperiod=period)
        
        # Find momentum extremes
        if len(momentum) >= period:
            recent_momentum = momentum[-period:]
            recent_max_momentum_idx = np.argmax(recent_momentum)
            recent_min_momentum_idx = np.argmin(recent_momentum)
            recent_max_momentum = Decimal(str(momentum[len(momentum) - period + recent_max_momentum_idx]))
            recent_min_momentum = Decimal(str(momentum[len(momentum) - period + recent_min_momentum_idx]))
        else:
            recent_max_momentum = Decimal(str(np.nanmax(momentum)))
            recent_min_momentum = Decimal(str(np.nanmin(momentum)))
        
        current_momentum = Decimal(str(momentum[-1])) if len(momentum) > 0 else Decimal('0')
        
        # Calculate momentum percentages
        if recent_max_momentum != recent_min_momentum:
            percent_to_min_momentum = (recent_max_momentum - current_momentum) / (recent_max_momentum - recent_min_momentum) * Decimal('100')
            percent_to_max_momentum = (current_momentum - recent_min_momentum) / (recent_max_momentum - recent_min_momentum) * Decimal('100')
        else:
            percent_to_min_momentum = Decimal('50')
            percent_to_max_momentum = Decimal('50')
        
        return (
            min_threshold, 
            max_threshold, 
            middle_threshold,
            periods_since_high,
            periods_since_low,
            percent_to_min_momentum,
            percent_to_max_momentum,
            high_idx,
            low_idx
        )
        
    except Exception as e:
        print(f"Error in calculate_thresholds_with_extrema: {e}")
        return None, None, None, None, None, None, None, None, None, None

def enhanced_aroon(high, low, close, period=14):
    """
    Enhanced Aroon calculation using recent extrema indices.
    """
    try:
        if len(high) < period or len(low) < period:
            return None, None, None, None, None, None
        
        # Calculate thresholds to get recent indices
        thresholds = calculate_thresholds_with_extrema(close, period)
        
        if thresholds[7] is not None and thresholds[8] is not None:
            # Use indices from threshold calculation
            periods_since_high = thresholds[3]
            periods_since_low = thresholds[4]
            recent_high_idx = thresholds[7]
            recent_low_idx = thresholds[8]
        else:
            # Fallback to standard calculation
            recent_high, recent_low, periods_since_high, periods_since_low, recent_high_idx, recent_low_idx = calculate_extrema_with_indices(close, period)
        
        # Calculate Aroon values
        aroon_up = ((period - periods_since_high) / period) * 100.0
        aroon_down = ((period - periods_since_low) / period) * 100.0
        
        return aroon_up, aroon_down, periods_since_high, periods_since_low, recent_high_idx, recent_low_idx
        
    except Exception as e:
        print(f"Error in enhanced_aroon: {e}")
        return None, None, None, None, None, None

def analyze_multiple_timeframe_extrema(candles_1m, candles_3m=None, candles_5m=None):
    """
    Analyze extrema across multiple timeframes.
    """
    global timeframe_extrema
    
    try:
        # Analyze 1m timeframe
        close_1m = [c["close"] for c in candles_1m]
        high_1m = [c["high"] for c in candles_1m]
        low_1m = [c["low"] for c in candles_1m]
        
        aroon_1m = enhanced_aroon(high_1m, low_1m, close_1m, period=14)
        thresholds_1m = calculate_thresholds_with_extrema(close_1m, period=14)
        
        if aroon_1m[0] is not None and thresholds_1m[0] is not None:
            timeframe_extrema['1m'] = {
                'recent_high': high_1m[aroon_1m[4]] if aroon_1m[4] is not None else None,
                'recent_low': low_1m[aroon_1m[5]] if aroon_1m[5] is not None else None,
                'recent_high_idx': aroon_1m[4],
                'recent_low_idx': aroon_1m[5],
                'min_threshold': thresholds_1m[0],
                'max_threshold': thresholds_1m[1],
                'middle_threshold': thresholds_1m[2],
                'aroon_up': aroon_1m[0],
                'aroon_down': aroon_1m[1]
            }
        
        # Analyze 3m timeframe if available
        if candles_3m and len(candles_3m) >= 14:
            close_3m = [c["close"] for c in candles_3m]
            high_3m = [c["high"] for c in candles_3m]
            low_3m = [c["low"] for c in candles_3m]
            
            aroon_3m = enhanced_aroon(high_3m, low_3m, close_3m, period=14)
            thresholds_3m = calculate_thresholds_with_extrema(close_3m, period=14)
            
            if aroon_3m[0] is not None and thresholds_3m[0] is not None:
                timeframe_extrema['3m'] = {
                    'recent_high': high_3m[aroon_3m[4]] if aroon_3m[4] is not None else None,
                    'recent_low': low_3m[aroon_3m[5]] if aroon_3m[5] is not None else None,
                    'recent_high_idx': aroon_3m[4],
                    'recent_low_idx': aroon_3m[5],
                    'min_threshold': thresholds_3m[0],
                    'max_threshold': thresholds_3m[1],
                    'middle_threshold': thresholds_3m[2],
                    'aroon_up': aroon_3m[0],
                    'aroon_down': aroon_3m[1]
                }
        
        # Analyze 5m timeframe if available
        if candles_5m and len(candles_5m) >= 14:
            close_5m = [c["close"] for c in candles_5m]
            high_5m = [c["high"] for c in candles_5m]
            low_5m = [c["low"] for c in candles_5m]
            
            aroon_5m = enhanced_aroon(high_5m, low_5m, close_5m, period=14)
            thresholds_5m = calculate_thresholds_with_extrema(close_5m, period=14)
            
            if aroon_5m[0] is not None and thresholds_5m[0] is not None:
                timeframe_extrema['5m'] = {
                    'recent_high': high_5m[aroon_5m[4]] if aroon_5m[4] is not None else None,
                    'recent_low': low_5m[aroon_5m[5]] if aroon_5m[5] is not None else None,
                    'recent_high_idx': aroon_5m[4],
                    'recent_low_idx': aroon_5m[5],
                    'min_threshold': thresholds_5m[0],
                    'max_threshold': thresholds_5m[1],
                    'middle_threshold': thresholds_5m[2],
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

def analyze_linear_regression_forecast_15s(candles_1m):
    try:
        if not candles_1m:
            return {"error": "No 1m data provided"}
        
        df_1m = pd.DataFrame(candles_1m)
        df_1m['timestamp'] = pd.to_datetime(df_1m['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
        
        df_15s = generate_15s_data_from_1m(df_1m)
        if df_15s is None or df_15s.empty:
            return {"error": "Failed to generate 15s data"}
        
        close_prices = df_15s['close'].values
        n = len(close_prices)
        
        if n < 2:
            return {"error": "Insufficient data for linear regression"}
        
        x = np.arange(n)
        coeffs = np.polyfit(x, close_prices, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        forecast_target = slope * n + intercept
        current_price = close_prices[-1]
        forecast_diff_pct = ((forecast_target - current_price) / current_price) * 100
        forecast_up = forecast_target > current_price
        
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
        
        # Track all instances where price is below lower channel or above upper channel
        below_lower_indices = []
        above_upper_indices = []
        
        for i in range(n):
            if close_prices[i] < lower_channel[i]:
                below_lower_indices.append(i)
            if close_prices[i] > upper_channel[i]:
                above_upper_indices.append(i)
        
        # Find the most recent instances
        most_recent_below_lower = len(below_lower_indices) > 0
        most_recent_above_upper = len(above_upper_indices) > 0
        most_recent_below_lower_index = max(below_lower_indices) if below_lower_indices else None
        most_recent_above_upper_index = max(above_upper_indices) if above_upper_indices else None
        
        # Determine if we're in an up cycle (most recent was below lower channel)
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
            "below_lower_indices": below_lower_indices[-10:] if len(below_lower_indices) > 0 else [],  # Last 10 instances
            "above_upper_indices": above_upper_indices[-10:] if len(above_upper_indices) > 0 else []   # Last 10 instances
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
        
        # Get all instances of channel breaks
        below_lower_indices = lrc_result["below_lower_indices"]
        above_upper_indices = lrc_result["above_upper_indices"]
        
        # Find the most recent occurrences (the one with the highest index)
        most_recent_below_lower_index = max(below_lower_indices) if below_lower_indices else None
        most_recent_above_upper_index = max(above_upper_indices) if above_upper_indices else None
        
        # Determine which event was more recent to decide the cycle direction
        condition_met = False
        description = "No channel breaks detected in the analysis window."
        
        # Determine which event was most recent
        is_most_recent_below_lower = False
        is_most_recent_above_upper = False
        
        if most_recent_below_lower_index is not None and most_recent_above_upper_index is not None:
            # Both types of breaks occurred, compare which was more recent
            if most_recent_below_lower_index > most_recent_above_upper_index:
                condition_met = True  # Most recent was below lower channel
                description = "Most recent occurrence was price below the lower channel (indicating upward cycle)"
                is_most_recent_below_lower = True
            else:
                condition_met = False  # Most recent was above upper channel
                description = "Most recent occurrence was price above the upper channel (indicating downward cycle)"
                is_most_recent_above_upper = True
        elif most_recent_below_lower_index is not None:
            # Only below lower channel breaks occurred
            condition_met = True
            description = "Most recent occurrence was price below the lower channel (indicating upward cycle)"
            is_most_recent_below_lower = True
        elif most_recent_above_upper_index is not None:
            # Only above upper channel breaks occurred
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
            "most_recent_below_lower": is_most_recent_below_lower,  # Fixed: Only true if this was the most recent event
            "most_recent_above_upper": is_most_recent_above_upper,  # Fixed: Only true if this was the most recent event
            "most_recent_below_lower_index": most_recent_below_lower_index,
            "most_recent_above_upper_index": most_recent_above_upper_index,
            "up_cycle": condition_met,  # True if condition is met
            "description": description,
            "below_lower_indices": below_lower_indices[-10:] if len(below_lower_indices) > 0 else [],
            "above_upper_indices": above_upper_indices[-10:] if len(above_upper_indices) > 0 else []
        }
        
    except Exception as e:
        print(f"Error analyzing linear regression channel break: {e}")
        return {"error": str(e)}

def predict_prices_with_fft(candles, num_predictions=5, filter_ratio=0.8):
    try:
        if not candles or len(candles) < 10:
            return {"error": "Insufficient data for FFT prediction"}
        
        recent_candles = candles[-15:] if len(candles) > 15 else candles
        close_prices = np.array([candle["close"] for candle in recent_candles], dtype=np.float64)
        
        window_size = min(3, len(close_prices) // 3)
        if window_size < 3:
            window_size = 3
        
        weights = np.ones(window_size) / window_size
        moving_avg = np.convolve(close_prices, weights, mode='valid')
        padding = np.zeros(len(close_prices) - len(moving_avg))
        moving_avg = np.concatenate([padding, moving_avg])
        
        detrended = close_prices - moving_avg
        fft_values = np.fft.fft(detrended)
        
        fft_abs = np.abs(fft_values)
        threshold = np.max(fft_abs) * (1 - filter_ratio)
        mask = fft_abs > threshold
        filtered_fft = fft_values * mask
        
        extended_fft = np.zeros(len(filtered_fft) + 1, dtype=complex)
        extended_fft[:len(filtered_fft)] = filtered_fft
        extended_detrended = np.fft.ifft(extended_fft).real
        
        last_ma_value = moving_avg[-1]
        next_pred_detrended = extended_detrended[-1]
        next_prediction = next_pred_detrended + last_ma_value
        
        current_price = close_prices[-1]
        max_change_pct = 0.001
        change_pct = (next_prediction - current_price) / current_price
        
        if abs(change_pct) > max_change_pct:
            constrained_prediction = current_price * (1 + max_change_pct if change_pct > 0 else 1 - max_change_pct)
        else:
            constrained_prediction = next_prediction
        
        predictions = []
        for i in range(num_predictions):
            variation = (1 - (i * 0.02)) if change_pct > 0 else (1 + (i * 0.02))
            predictions.append(constrained_prediction * variation)
        
        confidence = np.sum(mask) / len(mask)
        prediction_direction = "Up" if predictions[0] > current_price else "Down"
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
            "max_change_pct": max_change_pct * 100
        }
        
    except Exception as e:
        print(f"Error in FFT prediction: {e}")
        return {"error": str(e)}

def analyze_fft_prediction_1m(candles_1m):
    try:
        if not candles_1m:
            return {"error": "No 1m data provided"}
        
        recent_candles = candles_1m[-15:] if len(candles_1m) > 15 else candles_1m
        fft_result = predict_prices_with_fft(recent_candles, num_predictions=5, filter_ratio=0.8)
        
        if "error" in fft_result:
            return fft_result
        
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

def fft_cycle_phase(close, keep_ratio=0.10):
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

def ml_reversal_probability(features):
    # Adjusted weights to prevent extreme probabilities
    weights = np.array([0.8, 0.6, 0.5, 0.4])  # Reduced weights
    bias = 0.1  # Adjusted bias to center the probability

    z = np.dot(weights, features) + bias
    
    # Apply a scaling factor to prevent extreme values
    z = np.clip(z, -3, 3)  # Clip the value to prevent saturation
    
    probability = 1.0 / (1.0 + math.exp(-z))
    
    # Ensure probability is in a reasonable range
    probability = np.clip(probability, 0.05, 0.95)  # Prevent extreme probabilities
    
    return probability

# =========================================================
# FIXED FFT + AROON + ML REVERSAL ANALYSIS
# =========================================================

def analyze_fft_aroon_ml_reversal(candles, aroon_period=14):
    global timeframe_extrema, current_major_trend, last_major_high, last_major_low
    
    try:
        if len(candles) < 50:
            return {"error": "Not enough candles", "condition_met": False}

        close = [c["close"] for c in candles]
        high = [c["high"] for c in candles]
        low = [c["low"] for c in candles]
        
        current_time = datetime.datetime.now()
        current_price = close[-1] if len(close) > 0 else 0
        
        # =====================================================
        # CALCULATE REAL MIN/MAX THRESHOLDS FROM 1200 CANDLES
        # =====================================================
        # Use the last 1200 candles (or all available if less)
        lookback_candles = min(1200, len(candles))
        recent_candles = candles[-lookback_candles:]
        
        # Initialize thresholds with safe defaults
        min_threshold = current_price * 0.99
        max_threshold = current_price * 1.01
        middle_threshold = current_price
        
        if len(recent_candles) > 0:
            try:
                # Extract close prices for threshold calculation
                recent_closes = [c["close"] for c in recent_candles]
                
                if len(recent_closes) > 0:
                    # Find argmin and argmax for min/max thresholds
                    min_idx = np.argmin(recent_closes)
                    max_idx = np.argmax(recent_closes)
                    
                    # Get actual min and max values
                    min_threshold = recent_closes[min_idx]
                    max_threshold = recent_closes[max_idx]
                    
                    # Calculate middle threshold (average of min and max)
                    middle_threshold = (min_threshold + max_threshold) / 2
            except Exception as e:
                print(f"Error calculating thresholds: {e}")
                # Keep default values
        
        # =====================================================
        # MULTI-TIMEFRAME EXTREMA ANALYSIS
        # =====================================================
        extrema_analysis = {}
        try:
            extrema_analysis = analyze_multiple_timeframe_extrema(candles)
        except Exception as e:
            print(f"Error in multi-timeframe analysis: {e}")
        
        # Use 1m timeframe for Aroon and recent highs/lows
        aroon_up = 50.0  # Default values
        aroon_down = 50.0
        recent_high = None
        recent_low = None
        
        if '1m' in extrema_analysis and extrema_analysis['1m'].get('recent_high') is not None:
            tf_data = extrema_analysis['1m']
            aroon_up = tf_data.get('aroon_up', 50.0)
            aroon_down = tf_data.get('aroon_down', 50.0)
            recent_high = tf_data.get('recent_high')
            recent_low = tf_data.get('recent_low')
        else:
            # Fallback to enhanced_aroon if extrema analysis failed
            try:
                aroon_result = enhanced_aroon(high, low, close, aroon_period)
                if aroon_result and aroon_result[0] is not None:
                    aroon_up, aroon_down, periods_since_high, periods_since_low, recent_high_idx, recent_low_idx = aroon_result
                    if recent_high_idx is not None and recent_high_idx < len(high):
                        recent_high = high[recent_high_idx]
                    if recent_low_idx is not None and recent_low_idx < len(low):
                        recent_low = low[recent_low_idx]
            except Exception as e:
                print(f"Error in Aroon calculation: {e}")
        
        # =====================================================
        # FFT ANALYSIS
        # =====================================================
        cycle_value = 0.0
        cycle_slope = 0.0
        try:
            cycle_value, cycle_slope = fft_cycle_phase(close)
        except Exception as e:
            print(f"Error in FFT analysis: {e}")
        
        # =====================================================
        # ML PROBABILITY
        # =====================================================
        probability = 0.5  # Default neutral probability
        try:
            price_vs_cycle = current_price - cycle_value if cycle_value is not None else 0.0
            aroon_diff = aroon_up - aroon_down
            aroon_diff_normalized = aroon_diff / 100.0 if aroon_diff != 0 else 0.0
            aroon_strength = max(aroon_up, aroon_down)
            
            features = np.array([
                float(cycle_slope),
                float(price_vs_cycle),
                float(aroon_diff_normalized),
                float(aroon_strength / 100.0)
            ], dtype=np.float64)

            probability = ml_reversal_probability(features)
        except Exception as e:
            print(f"Error in ML probability calculation: {e}")
        
        # =====================================================
        # TREND DETECTION - CORRECTED
        # =====================================================
        # Determine Aroon trend - only UP or DOWN
        aroon_trend = "UP" if aroon_up > aroon_down else "DOWN"
        
        # Check for strong trends
        if aroon_up > 70 and aroon_down < 30:
            aroon_trend_strength = "STRONG_UP"
        elif aroon_down > 70 and aroon_up < 30:
            aroon_trend_strength = "STRONG_DOWN"
        else:
            aroon_trend_strength = "WEAK"
        
        # Current trend based on multiple indicators
        current_trend = "UP"
        try:
            trend_score = 0
            if aroon_trend == "UP":
                trend_score += 1
            if float(cycle_slope) > 0:
                trend_score += 1
            if current_price > middle_threshold:
                trend_score += 1
            
            current_trend = "UP" if trend_score >= 2 else "DOWN"
        except Exception as e:
            print(f"Error determining current trend: {e}")
        
        # Update major trend
        try:
            if current_major_trend == "UNKNOWN":
                current_major_trend = current_trend
            elif aroon_trend_strength == "STRONG_UP" and float(cycle_slope) > 0.05:
                current_major_trend = "UP"
            elif aroon_trend_strength == "STRONG_DOWN" and float(cycle_slope) < -0.05:
                current_major_trend = "DOWN"
        except Exception as e:
            print(f"Error updating major trend: {e}")
        
        # =====================================================
        # MARKET CYCLE POSITION
        # =====================================================
        market_cycle_position = 50
        market_phase = "MIDDLE ZONE"
        
        try:
            threshold_range = max_threshold - min_threshold
            if threshold_range > 0:
                market_cycle_position = ((current_price - min_threshold) / threshold_range) * 100
            else:
                market_cycle_position = 50
            
            market_cycle_position = max(0, min(100, market_cycle_position))
            
            # Market phase
            if market_cycle_position > 70:
                market_phase = "TOP ZONE"
            elif market_cycle_position < 30:
                market_phase = "BOTTOM ZONE"
            else:
                market_phase = "MIDDLE ZONE"
        except Exception as e:
            print(f"Error calculating market cycle: {e}")
        
        # =====================================================
        # SUPPORT AND RESISTANCE
        # =====================================================
        resistance_level = max_threshold
        support_level = min_threshold
        
        # Update major highs/lows
        try:
            if recent_high is not None and (last_major_high is None or recent_high > last_major_high):
                last_major_high = recent_high
            
            if recent_low is not None and (last_major_low is None or recent_low < last_major_low):
                last_major_low = recent_low
        except Exception as e:
            print(f"Error updating major highs/lows: {e}")
        
        # =====================================================
        # POSITION DETECTION
        # =====================================================
        # Initialize position variables
        support_distance = 0.0
        resistance_distance = 0.0
        support_distance_pct = 0.0
        resistance_distance_pct = 0.0
        near_support = False
        near_resistance = False
        in_middle_range = False
        
        try:
            # Calculate percentage distance to support and resistance
            if support_level is not None and support_level > 0:
                support_distance = current_price - support_level
                support_distance_pct = (support_distance / support_level) * 100
                near_support = support_distance_pct >= 0 and support_distance_pct <= 0.5  # Increased range
            
            if resistance_level is not None and resistance_level > 0:
                resistance_distance = resistance_level - current_price
                resistance_distance_pct = (resistance_distance / resistance_level) * 100
                near_resistance = resistance_distance_pct >= 0 and resistance_distance_pct <= 0.5  # Increased range
            
            # Check if in middle range
            in_middle_range = not near_support and not near_resistance
        except Exception as e:
            print(f"Error in position detection: {e}")
        
        # =====================================================
        # REVERSAL SIGNALS - CORRECTED LOGIC
        # =====================================================
        reversal_dip = False
        reversal_top = False
        
        try:
            # Reversal dip: Price near support, major trend was DOWN, now showing UP signals
            reversal_dip = (
                market_phase == "BOTTOM ZONE" and
                aroon_up > 70 and  # Strong bullish Aroon
                aroon_down < 30 and  # Weak bearish Aroon
                cycle_slope > 0.01 and  # Positive cycle (lowered threshold)
                probability > 0.6  # High reversal probability (lowered threshold)
            )
            
            # Reversal top: Price near resistance, major trend was UP, now showing DOWN signals
            reversal_top = (
                market_phase == "TOP ZONE" and
                aroon_down > 70 and  # Strong bearish Aroon
                aroon_up < 30 and  # Weak bullish Aroon
                cycle_slope < -0.01 and  # Negative cycle (lowered threshold)
                probability > 0.6  # High reversal probability (lowered threshold)
            )
        except Exception as e:
            print(f"Error in reversal detection: {e}")
        
        # =====================================================
        # CONTINUATION SIGNALS
        # =====================================================
        up_continuation = False
        down_continuation = False
        
        try:
            # Up continuation: Price has broken resistance, strong UP signals
            up_continuation = (
                current_trend == "UP" and
                aroon_up > 70 and  # Strong bullish Aroon
                current_price > resistance_level * 0.98 and  # Near or above resistance
                cycle_slope > 0  # Positive cycle
            )
            
            # Down continuation: Price has broken support, strong DOWN signals
            down_continuation = (
                current_trend == "DOWN" and
                aroon_down > 70 and  # Strong bearish Aroon
                current_price < support_level * 1.02 and  # Near or below support
                cycle_slope < 0  # Negative cycle
            )
        except Exception as e:
            print(f"Error in continuation detection: {e}")
        
        # =====================================================
        # SIGNAL GENERATION - REVISED LOGIC
        # =====================================================
        condition_met = False
        signal = "NO TRADE"
        signal_reason = ""
        
        try:
            # Priority 1: Major reversals (strongest signals)
            if reversal_dip:
                signal = "LONG"
                signal_reason = "REVERSAL DIP (BOTTOM ZONE WITH STRONG UP SIGNALS)"
                condition_met = True
            elif reversal_top:
                signal = "SHORT"
                signal_reason = "REVERSAL TOP (TOP ZONE WITH STRONG DOWN SIGNALS)"
                condition_met = False  # Always false for top reversal
            
            # Priority 2: Strong Aroon signals regardless of zone
            elif aroon_up > 70 and aroon_down < 30:
                signal = "LONG"
                signal_reason = "STRONG AROON UP SIGNAL"
                condition_met = True
            elif aroon_down > 70 and aroon_up < 30:
                signal = "SHORT"
                signal_reason = "STRONG AROON DOWN SIGNAL"
                condition_met = False  # Always false for down signals
            
            # Priority 3: Continuations (momentum signals)
            elif up_continuation:
                signal = "LONG"
                signal_reason = "UP CONTINUATION (BREAKING RESISTANCE)"
                condition_met = True
            elif down_continuation:
                signal = "SHORT"
                signal_reason = "DOWN CONTINUATION (BREAKING SUPPORT)"
                condition_met = False  # Always false for down signals
            
            # Priority 4: Bottom zone with positive indicators
            elif market_phase == "BOTTOM ZONE" and (
                (aroon_up > aroon_down and cycle_slope > 0) or
                (current_trend == "UP" and probability > 0.6)
            ):
                signal = "LONG"
                signal_reason = "BOTTOM ZONE WITH POSITIVE INDICATORS"
                condition_met = True
            
            # Priority 5: General upward trend
            elif current_trend == "UP" and aroon_up > 60:
                signal = "LONG"
                signal_reason = "GENERAL UPWARD TREND"
                condition_met = True
                
        except Exception as e:
            print(f"Error in signal generation: {e}")
        
        # =====================================================
        # FORECAST CALCULATION
        # =====================================================
        forecast_pct = 0.0
        forecast_move = 0.0
        reversal_forecast = current_price
        
        try:
            if signal == "LONG":
                # Calculate forecast for LONG
                recent_price_data = close[-10:] if len(close) >= 10 else close
                if len(recent_price_data) > 1:
                    recent_volatility = np.std(recent_price_data)
                else:
                    recent_volatility = current_price * 0.001
                
                base_move = min(recent_volatility, current_price * 0.005)
                aroon_factor = aroon_up / 100.0 if aroon_up > 0 else 0.5
                confidence = probability if probability > 0 else 0.5
                cycle_factor = 1 + (abs(cycle_slope) * 0.001)
                position_factor = 1.5 if near_support else 1.0
                
                forecast_move = base_move * aroon_factor * confidence * cycle_factor * position_factor
                
                max_move = current_price * 0.01
                min_move = current_price * 0.0005
                forecast_move = max(min(forecast_move, max_move), min_move)
                
                reversal_forecast = current_price + forecast_move
                forecast_pct = (forecast_move / current_price) * 100 if current_price > 0 else 0
                
            elif signal == "SHORT":
                # Similar calculation for SHORT
                recent_price_data = close[-10:] if len(close) >= 10 else close
                if len(recent_price_data) > 1:
                    recent_volatility = np.std(recent_price_data)
                else:
                    recent_volatility = current_price * 0.001
                
                base_move = min(recent_volatility, current_price * 0.005)
                aroon_factor = aroon_down / 100.0 if aroon_down > 0 else 0.5
                confidence = probability if probability > 0 else 0.5
                cycle_factor = 1 + (abs(cycle_slope) * 0.001)
                position_factor = 1.5 if near_resistance else 1.0
                
                forecast_move = base_move * aroon_factor * confidence * cycle_factor * position_factor
                
                max_move = current_price * 0.01
                min_move = current_price * 0.0005
                forecast_move = max(min(forecast_move, max_move), min_move)
                
                reversal_forecast = current_price - forecast_move
                forecast_pct = (forecast_move / current_price) * 100 if current_price > 0 else 0
                
        except Exception as e:
            print(f"Error in forecast calculation: {e}")
        
        # =====================================================
        # RETURN RESULTS
        # =====================================================
        return {
            "signal": f"{signal} ({signal_reason})" if signal != "NO TRADE" else "NO TRADE",
            "condition_met": condition_met,
            "probability": round(probability, 4),
            "cycle_slope": round(cycle_slope, 6),
            "price_vs_cycle": round(price_vs_cycle, 6) if 'price_vs_cycle' in locals() else 0.0,
            "aroon_up": round(aroon_up, 2),
            "aroon_down": round(aroon_down, 2),
            "aroon_trend": aroon_trend,
            "aroon_trend_strength": aroon_trend_strength,
            "current_trend": current_trend,
            "current_major_trend": current_major_trend,
            "market_phase": market_phase,
            "market_cycle_position": round(market_cycle_position, 2),
            "current_price": round(current_price, 2),
            "support_level": round(support_level, 2) if support_level is not None else None,
            "resistance_level": round(resistance_level, 2) if resistance_level is not None else None,
            "middle_threshold": round(middle_threshold, 2) if middle_threshold is not None else None,
            "min_threshold": round(min_threshold, 2) if min_threshold is not None else None,
            "max_threshold": round(max_threshold, 2) if max_threshold is not None else None,
            "support_distance": round(support_distance, 2),
            "resistance_distance": round(resistance_distance, 2),
            "support_distance_pct": round(support_distance_pct, 3),
            "resistance_distance_pct": round(resistance_distance_pct, 3),
            "near_support": near_support,
            "near_resistance": near_resistance,
            "in_middle_range": in_middle_range,
            "reversal_forecast": round(reversal_forecast, 2),
            "forecast_percentage": round(forecast_pct, 4),
            "recent_high": round(recent_high, 2) if recent_high is not None else None,
            "recent_low": round(recent_low, 2) if recent_low is not None else None,
            "last_major_high": round(last_major_high, 2) if last_major_high is not None else None,
            "last_major_low": round(last_major_low, 2) if last_major_low is not None else None,
            "reversal_dip": reversal_dip,
            "reversal_top": reversal_top,
            "up_continuation": up_continuation,
            "down_continuation": down_continuation,
            "multi_timeframe_analysis": extrema_analysis
        }
        
    except Exception as e:
        print(f"Error in analyze_fft_aroon_ml_reversal: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "condition_met": False}

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
        
        candles_1m = get_candles(TRADE_SYMBOL, '1m', limit=1200)
        
        if not candle_map.get('1m'):
            print("Error: '1m' candles not fetched. Check API connectivity or symbol.")
        if current_price == Decimal('0.0'):
            print(f"Warning: Current {TRADE_SYMBOL} price is {current_price:.25f}. API may be failing.")

        # Initialize all condition results
        conditions_status = {
            "momentum_positive_1m": False,
            "momentum_positive_15sec": False,
            "linear_regression_forecast_15s": False,
            "linear_regression_channel_break": False,
            "fft_prediction_1m": False,
            "fft_aroon_ml_reversal": False,
            "ema_crossover_orderflow": False,  # New condition
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
            print(f"Most Recent Below Lower: {lrc_break_result['most_recent_below_lower']}")
            print(f"Most Recent Above Upper: {lrc_break_result['most_recent_above_upper']}")
            print(f"Most Recent Below Lower Index: {lrc_break_result['most_recent_below_lower_index']}")
            print(f"Most Recent Above Upper Index: {lrc_break_result['most_recent_above_upper_index']}")
            print(f"Up Cycle: {lrc_break_result['up_cycle']}")
            print(f"Description: {lrc_break_result['description']}")
    
            # Print the last few instances of channel breaks for debugging
            if lrc_break_result['below_lower_indices']:
                print(f"Recent Below Lower Indices: {lrc_break_result['below_lower_indices']}")
            if lrc_break_result['above_upper_indices']:
                print(f"Recent Above Upper Indices: {lrc_break_result['above_upper_indices']}")
        
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
            
        # Condition 6: FFT + Aroon + ML Reversal (FIXED)
        print("\n--- Condition 6: FFT + Aroon + ML Reversal (FIXED) ---")
        fft_aroon_ml_result = analyze_fft_aroon_ml_reversal(candles_1m)
        if 'error' not in fft_aroon_ml_result:
            conditions_status["fft_aroon_ml_reversal"] = fft_aroon_ml_result['condition_met']
            print(f"Signal: {fft_aroon_ml_result['signal']}")
            print(f"Probability: {fft_aroon_ml_result['probability']}")
            print(f"Cycle Slope: {fft_aroon_ml_result['cycle_slope']}")
            print(f"Price vs Cycle: {fft_aroon_ml_result['price_vs_cycle']}")
            print(f"Aroon Up: {fft_aroon_ml_result['aroon_up']}")
            print(f"Aroon Down: {fft_aroon_ml_result['aroon_down']}")
            print(f"Aroon Trend: {fft_aroon_ml_result['aroon_trend']}")
            print(f"Current Trend: {fft_aroon_ml_result['current_trend']}")
            print(f"Current Major Trend: {fft_aroon_ml_result['current_major_trend']}")
            print(f"Market Phase: {fft_aroon_ml_result['market_phase']}")
            print(f"Market Cycle Position: {fft_aroon_ml_result['market_cycle_position']}%")
            print(f"Reversal Forecast: {fft_aroon_ml_result['reversal_forecast']}")
            print(f"Recent High: {fft_aroon_ml_result['recent_high']}")
            print(f"Recent Low: {fft_aroon_ml_result['recent_low']}")
            print(f"Min Threshold: {fft_aroon_ml_result['min_threshold']}")
            print(f"Max Threshold: {fft_aroon_ml_result['max_threshold']}")
            print(f"Middle Threshold: {fft_aroon_ml_result['middle_threshold']}")
            print(f"Resistance Level: {fft_aroon_ml_result['resistance_level']}")
            print(f"Support Level: {fft_aroon_ml_result['support_level']}")
            print(f"Last Major High: {fft_aroon_ml_result['last_major_high']}")
            print(f"Last Major Low: {fft_aroon_ml_result['last_major_low']}")
            print(f"Condition Met: {conditions_status['fft_aroon_ml_reversal']}")

            # Show multi-timeframe analysis
            if 'multi_timeframe_analysis' in fft_aroon_ml_result:
                print("\nMulti-Timeframe Analysis:")
                for tf, data in fft_aroon_ml_result['multi_timeframe_analysis'].items():
                    if data and data['recent_high'] is not None:
                        print(f"  {tf}: High={data['recent_high']:.2f}, Low={data['recent_low']:.2f}, "
                              f"Aroon Up={data['aroon_up']:.2f}, Aroon Down={data['aroon_down']:.2f}")
        else:
            print(f"Error analyzing FFT + Aroon + ML Reversal: {fft_aroon_ml_result['error']}")
            print(f"Condition Met: {conditions_status['fft_aroon_ml_reversal']}")

        # Condition 7: EMA Crossover with Order Flow Analysis
        print("\n--- Condition 7: EMA Crossover with Order Flow Analysis ---")
        order_book = get_order_book(TRADE_SYMBOL)
        ema_crossover_result = analyze_ema_crossover_with_orderflow(candles_1m, order_book)
        
        if 'error' not in ema_crossover_result:
            conditions_status["ema_crossover_orderflow"] = ema_crossover_result['condition_met']
            print(f"Signal: {ema_crossover_result['signal']}")
            print(f"Signal Strength: {ema_crossover_result['signal_strength']}")
            print(f"Signal Reason: {ema_crossover_result['signal_reason']}")
            print(f"Current Price: {ema_crossover_result['current_price']:.2f}")
            print(f"EMA9: {ema_crossover_result['ema9']:.2f}")
            print(f"EMA21: {ema_crossover_result['ema21']:.2f}")
            print(f"SMA9: {ema_crossover_result['sma9']:.2f}")
            print(f"SMA21: {ema_crossover_result['sma21']:.2f}")
            print(f"EMA Distance: {ema_crossover_result['ema_distance']:.4f}")
            print(f"EMA Distance %: {ema_crossover_result['ema_distance_pct']:.4f}%")
            print(f"VWAP: {ema_crossover_result['vwap']:.2f}")
            print(f"Potential Crossover: {ema_crossover_result['potential_crossover']}")
            print(f"Actual Crossover: {ema_crossover_result['actual_crossover']}")
            print(f"Actual Crossunder: {ema_crossover_result['actual_crossunder']}")
            print(f"Volume Increasing: {ema_crossover_result['volume_increasing']}")
            print(f"Current Volume: {ema_crossover_result['current_volume']:.2f}")
            print(f"Average Volume: {ema_crossover_result['avg_volume']:.2f}")
            print(f"Volume Trend: {ema_crossover_result['volume_trend']:.2f}")
            print(f"Buying Pressure: {ema_crossover_result['buying_pressure']}")
            print(f"Strong Buy Signal: {ema_crossover_result['strong_buy_signal']}")
            print(f"Strong Sell Signal: {ema_crossover_result['strong_sell_signal']}")
            print(f"Price Above EMA9: {ema_crossover_result['price_above_ema9']}")
            print(f"Price Above EMA21: {ema_crossover_result['price_above_ema21']}")
            print(f"EMA Convergence Rate: {ema_crossover_result['ema_convergence_rate']:.4f}")
            print(f"Condition Met: {conditions_status['ema_crossover_orderflow']}")
            
            # Print order book details if available
            if ema_crossover_result['order_book_analysis']:
                ob = ema_crossover_result['order_book_analysis']
                print(f"Order Book Analysis:")
                print(f"  Total Bid Volume: {ob['total_bid_volume']:.2f}")
                print(f"  Total Ask Volume: {ob['total_ask_volume']:.2f}")
                print(f"  Bid Ratio: {ob['bid_ratio']:.2f}")
                print(f"  Spread: {ob['spread']:.2f}")
                print(f"  Spread %: {ob['spread_pct']:.4f}%")
                print(f"  Large Bids: {len(ob['large_bids'])}")
                print(f"  Large Asks: {len(ob['large_asks'])}")
                print(f"  VWAP Mid: {ob['vwap_mid']:.2f}")
                print(f"  Order Flow Imbalance: {ob['order_flow_imbalance']:.4f}")
        else:
            print(f"Error analyzing EMA crossover: {ema_crossover_result['error']}")
            print(f"Condition Met: {conditions_status['ema_crossover_orderflow']}")

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

            target_value = initial_investment * Decimal('1.0035')
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
                print("Target profit of 0.35% reached or exceeded. Initiating exit...")
                # Fixed syntax error: added missing closing parenthesis
                if sell_asset(float(asset_balance)):
                    exit_usdc_balance = get_balance('USDC')
                    profit = exit_usdc_balance - initial_investment
                    profit_percentage = (profit / initial_investment) * Decimal('100') if initial_investment > Decimal('0.0') else Decimal('0.0')
                    print(f"Position closed. Sold BTC for USDC: {exit_usdc_balance:.25f}")
                    print(f"Trade log: Time: {current_local_time_str}, Entry Price: {entry_price:.25f}, Exit Balance: {exit_usdc_balance:.25f}")
                    print(f"Trade log: Time: {current_local_time_str}, Entry Price: {entry_price:.25f}, Exit Balance: {exit_usdc_balance:.25f}, Profit: {profit:.25f} Profit Percentage: {profit_percentage:.25f}%")
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
                        # Fixed syntax error: added missing closing parenthesis
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
