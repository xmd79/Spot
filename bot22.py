#!/usr/bin/env python3
"""
ENHANCED BTCUSDC TRADING BOT - SINGLE ITERATION FRAME
Specialized for single asset trading with enhanced dip detection
"""

import os
import time
import numpy as np
import pandas as pd
import threading
import signal
import sys
import warnings
import math
import gc
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# --- Optional libraries with graceful fallbacks ---
try:
    import pandas_ta as ta
    import talib
except Exception as e:
    print(f"Optional libs warning: {e}")

# ------------------ Configuration ------------------
API_FILE = 'api.txt'
SYMBOL = 'BTCUSDC'

# Timeframes for analysis
TIMEFRAMES = ['1m', '3m', '5m']

# Trading Configuration
PROFIT_TARGET_PERCENT = 1.45
TOTAL_FEE_PERCENT = 0.22
MIN_TRADE_AMOUNT = 10

# Dust Conversion Configuration
MIN_DUST_CONVERSION_AMOUNT = 0.0001
MAX_DUST_CONVERSION_AMOUNT = 0.001

# Technical Indicators Configuration
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
STOCH_OVERSOLD = 20
STOCH_OVERBOUGHT = 80

BB_PERIOD = 360
BB_STD = 2

# Global stop event
stop_event = threading.Event()

# Trade state variables
trade_active = False
trade_info = {}

def signal_handler(sig, frame):
    print('\nCtrl+C pressed! Shutting down gracefully...')
    stop_event.set()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ------------------ Enhanced BTC Dust Conversion ------------------

def convert_btc_dust_to_usdc(client):
    """Convert only BTC dust balances to USDC before trading with proper error handling."""
    try:
        print("Checking for BTC dust to convert...")
        account_info = client.get_account()
        
        dust_assets = []
        
        for balance in account_info['balances']:
            asset = balance['asset']
            free_balance = float(balance['free'])
            locked_balance = float(balance['locked'])
            total_balance = free_balance + locked_balance
            
            # Check for BTC dust specifically
            if asset == 'BTC' and MIN_DUST_CONVERSION_AMOUNT <= total_balance <= MAX_DUST_CONVERSION_AMOUNT:
                try:
                    # Get current price to calculate USD value
                    symbol = f"{asset}USDC"
                    ticker = client.get_symbol_ticker(symbol=symbol)
                    price = float(ticker['price'])
                    usd_value = total_balance * price
                    
                    if usd_value < MIN_TRADE_AMOUNT:
                        print(f"Found BTC dust: {asset} - {total_balance:.8f} (≈${usd_value:.2f})")
                        dust_assets.append(asset)
                        
                except Exception as e:
                    print(f"Error checking {asset} value: {e}")
                    continue
        
        # Convert dust assets if any found
        if dust_assets:
            try:
                print(f"Attempting to convert dust assets: {dust_assets}")
                
                # Use the correct API endpoint for dust conversion
                result = client.transfer_dust(asset=dust_assets)
                
                if result.get('success', False):
                    print("BTC dust conversion successful!")
                    for item in result.get('transferResult', []):
                        print(f"  {item['fromAsset']}: {item['amount']} → USDC")
                    return True
                else:
                    print("Dust conversion failed or not supported for these assets")
                    return False
                    
            except BinanceAPIException as e:
                if 'illegal parameter' in str(e).lower():
                    print("Dust conversion not supported for these assets or invalid parameters")
                elif 'insufficient balance' in str(e).lower():
                    print("Insufficient balance for dust conversion")
                else:
                    print(f"Binance API error during dust conversion: {e}")
                return False
                
            except Exception as e:
                print(f"Unexpected error during dust conversion: {e}")
                return False
        else:
            print("No BTC dust found for conversion")
            return True
            
    except Exception as e:
        print(f"Error in BTC dust conversion process: {e}")
        return False

# ------------------ Enhanced Technical Analysis Functions ------------------

def calculate_rsi(df, period=RSI_PERIOD):
    """Calculate RSI indicator."""
    try:
        if df is None or len(df) < period + 1:
            return None
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        df = df.copy()
        df[f'RSI_{period}'] = rsi
        return df
    except Exception as e:
        print(f"calculate_rsi error: {e}")
        return None

def calculate_stochastic(df, k_period=STOCH_K_PERIOD, d_period=STOCH_D_PERIOD):
    """Calculate Stochastic Oscillator."""
    try:
        if df is None or len(df) < k_period:
            return None
            
        df = df.copy()
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        
        # Calculate %K
        lowest_low = pd.Series(low_prices).rolling(window=k_period, min_periods=1).min()
        highest_high = pd.Series(high_prices).rolling(window=k_period, min_periods=1).max()
        
        k_percent = 100 * ((close_prices - lowest_low) / (highest_high - lowest_low))
        k_percent = k_percent.fillna(50)
        
        # Calculate %D as the moving average of %K
        d_percent = pd.Series(k_percent).rolling(window=d_period, min_periods=1).mean()
        
        # Add to dataframe
        df['STOCH_K'] = k_percent
        df['STOCH_D'] = d_percent
        
        return df
    except Exception as e:
        print(f"calculate_stochastic error: {e}")
        return None

def calculate_bollinger_bands(df, period=BB_PERIOD, std_dev=BB_STD):
    """Calculate Bollinger Bands."""
    try:
        if df is None or len(df) < period:
            return None
            
        df = df.copy()
        close_prices = df['close'].values
        
        # Calculate middle band (SMA)
        middle_band = pd.Series(close_prices).rolling(window=period, min_periods=1).mean()
        
        # Calculate standard deviation
        std = pd.Series(close_prices).rolling(window=period, min_periods=1).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        # Add to dataframe
        df[f'BB_MIDDLE_{period}'] = middle_band
        df[f'BB_UPPER_{period}'] = upper_band
        df[f'BB_LOWER_{period}'] = lower_band
        
        return df
    except Exception as e:
        print(f"calculate_bollinger_bands error: {e}")
        return None

def check_polynomial_fit(prices, timestamps):
    """Check if current price is below polynomial fit line."""
    try:
        if len(prices) < 10:
            return False
            
        x = timestamps
        y = prices
        
        best_fit_line1 = np.poly1d(np.polyfit(x, y, 1))(x)
        best_fit_line3 = best_fit_line1 * 0.99
        
        if y[-1] < best_fit_line3[-1]:
            return True
        
        return False
    except Exception as e:
        print(f"check_polynomial_fit error: {e}")
        return False

# ------------------ Enhanced Analysis Functions ------------------

def analyze_argmin_argmax_condition(client, symbol, timeframe='1m', lookback=1200):
    """Analyze argmin vs argmax condition for dip detection."""
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=lookback)
        if not klines or len(klines) < 100:
            return False, 0.0, 0.0, {"error": "Insufficient data"}
            
        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
        
        for c in ['open','high','low','close','volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df.fillna(0.0, inplace=True)
        
        close_prices = df['close'].values
        
        # Find argmin and argmax
        argmin_idx = np.argmin(close_prices)
        argmax_idx = np.argmax(close_prices)
        
        current_price = close_prices[-1]
        min_price = close_prices[argmin_idx]
        max_price = close_prices[argmax_idx]
        
        # Determine which is more recent
        min_more_recent = argmin_idx > argmax_idx
        
        # Calculate distance percentages
        dist_to_min = abs((current_price - min_price) / min_price * 100) if min_price > 0 else 0
        dist_to_max = abs((current_price - max_price) / max_price * 100) if max_price > 0 else 0
        
        # Condition: min more recent AND distance to min < distance to max
        condition_met = min_more_recent and (dist_to_min < dist_to_max)
        
        details = {
            "min_more_recent": min_more_recent,
            "dist_to_min": dist_to_min,
            "dist_to_max": dist_to_max,
            "argmin_idx": argmin_idx,
            "argmax_idx": argmax_idx,
            "min_price": min_price,
            "max_price": max_price,
            "current_price": current_price
        }
        
        return condition_met, dist_to_min, dist_to_max, details
        
    except Exception as e:
        print(f"analyze_argmin_argmax_condition error: {e}")
        return False, 0.0, 0.0, {"error": str(e)}

def analyze_volume_condition(client, symbol, timeframe='1m', lookback=50):
    """Analyze volume bullish vs bearish condition."""
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=lookback)
        if not klines or len(klines) < 20:
            return False, 0.0, 0.0, {"error": "Insufficient data"}
            
        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
        
        for c in ['open','high','low','close','volume','taker_buy_base_asset_volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df.fillna(0.0, inplace=True)
        
        # Calculate bullish and bearish volume
        bullish_volume = df['taker_buy_base_asset_volume'].sum()
        total_volume = df['volume'].sum()
        bearish_volume = total_volume - bullish_volume
        
        # Calculate percentages
        bullish_pct = (bullish_volume / total_volume * 100) if total_volume > 0 else 0
        bearish_pct = (bearish_volume / total_volume * 100) if total_volume > 0 else 0
        
        # Condition: bullish volume percentage > bearish volume percentage
        condition_met = bullish_pct > bearish_pct
        
        details = {
            "bullish_volume": bullish_volume,
            "bearish_volume": bearish_volume,
            "total_volume": total_volume,
            "bullish_pct": bullish_pct,
            "bearish_pct": bearish_pct
        }
        
        return condition_met, bullish_pct, bearish_pct, details
        
    except Exception as e:
        print(f"analyze_volume_condition error: {e}")
        return False, 0.0, 0.0, {"error": str(e)}

def analyze_rsi_condition(client, symbol, timeframe='1m', lookback=1200):
    """Analyze RSI oversold/overbought most recent condition."""
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=lookback)
        if not klines or len(klines) < 100:
            return False, False, 0.0, {"error": "Insufficient data"}
            
        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
        
        for c in ['open','high','low','close']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df.fillna(0.0, inplace=True)
        
        # Calculate RSI
        df_rsi = calculate_rsi(df, RSI_PERIOD)
        if df_rsi is None or f'RSI_{RSI_PERIOD}' not in df_rsi.columns:
            return False, False, 0.0, {"error": "RSI calculation failed"}
        
        rsi_values = df_rsi[f'RSI_{RSI_PERIOD}'].values
        current_rsi = rsi_values[-1]
        
        # Find last oversold and overbought occurrences
        last_oversold_idx = None
        last_overbought_idx = None
        
        for i in range(len(rsi_values) - 1, -1, -1):
            if last_oversold_idx is None and rsi_values[i] <= RSI_OVERSOLD:
                last_oversold_idx = i
            if last_overbought_idx is None and rsi_values[i] >= RSI_OVERBOUGHT:
                last_overbought_idx = i
            if last_oversold_idx is not None and last_overbought_idx is not None:
                break
        
        # Determine which is most recent
        oversold_most_recent = False
        overbought_most_recent = False
        
        if last_oversold_idx is not None and last_overbought_idx is not None:
            oversold_most_recent = last_oversold_idx > last_overbought_idx
            overbought_most_recent = last_overbought_idx > last_oversold_idx
        elif last_oversold_idx is not None:
            oversold_most_recent = True
        elif last_overbought_idx is not None:
            overbought_most_recent = True
        
        details = {
            "current_rsi": current_rsi,
            "last_oversold_idx": last_oversold_idx,
            "last_overbought_idx": last_overbought_idx,
            "oversold_most_recent": oversold_most_recent,
            "overbought_most_recent": overbought_most_recent
        }
        
        return oversold_most_recent, overbought_most_recent, current_rsi, details
        
    except Exception as e:
        print(f"analyze_rsi_condition error: {e}")
        return False, False, 0.0, {"error": str(e)}

def analyze_merged_stochastic_condition(client, symbol, timeframes=['1m', '3m', '5m'], lookback=500):
    """Analyze merged stochastic condition across multiple timeframes."""
    try:
        all_k_values = []
        all_d_values = []
        
        for timeframe in timeframes:
            klines = client.get_klines(symbol=symbol, interval=timeframe, limit=lookback)
            if not klines or len(klines) < 50:
                continue
                
            df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
            
            for c in ['open','high','low','close']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df.fillna(0.0, inplace=True)
            
            # Calculate Stochastic
            df_stoch = calculate_stochastic(df)
            if df_stoch is None or 'STOCH_K' not in df_stoch.columns:
                continue
            
            # Add to merged arrays
            all_k_values.extend(df_stoch['STOCH_K'].values)
            all_d_values.extend(df_stoch['STOCH_D'].values)
        
        if len(all_k_values) == 0:
            return False, False, 0.0, 0.0, {"error": "No stochastic data"}
        
        # Use the last values from the merged arrays
        current_k = all_k_values[-1] if all_k_values else 50
        current_d = all_d_values[-1] if all_d_values else 50
        
        # Find last oversold and overbought in merged data
        last_oversold_idx = None
        last_overbought_idx = None
        
        for i in range(len(all_k_values) - 1, -1, -1):
            k_val = all_k_values[i]
            d_val = all_d_values[i] if i < len(all_d_values) else k_val
            
            if last_oversold_idx is None and k_val <= STOCH_OVERSOLD and d_val <= STOCH_OVERSOLD:
                last_oversold_idx = i
            if last_overbought_idx is None and k_val >= STOCH_OVERBOUGHT and d_val >= STOCH_OVERBOUGHT:
                last_overbought_idx = i
            if last_oversold_idx is not None and last_overbought_idx is not None:
                break
        
        # Determine which is most recent
        oversold_most_recent = False
        overbought_most_recent = False
        
        if last_oversold_idx is not None and last_overbought_idx is not None:
            oversold_most_recent = last_oversold_idx > last_overbought_idx
            overbought_most_recent = last_overbought_idx > last_oversold_idx
        elif last_oversold_idx is not None:
            oversold_most_recent = True
        elif last_overbought_idx is not None:
            overbought_most_recent = True
        
        details = {
            "current_k": current_k,
            "current_d": current_d,
            "last_oversold_idx": last_oversold_idx,
            "last_overbought_idx": last_overbought_idx,
            "oversold_most_recent": oversold_most_recent,
            "overbought_most_recent": overbought_most_recent,
            "total_periods": len(all_k_values)
        }
        
        return oversold_most_recent, overbought_most_recent, current_k, current_d, details
        
    except Exception as e:
        print(f"analyze_merged_stochastic_condition error: {e}")
        return False, False, 0.0, 0.0, {"error": str(e)}

def analyze_bollinger_bands_condition(client, symbol, timeframe='1m', period=BB_PERIOD, lookback=500):
    """Analyze Bollinger Bands lowest/highest extremes condition."""
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=lookback)
        if not klines or len(klines) < period:
            return False, False, {"error": "Insufficient data"}
            
        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
        
        for c in ['open','high','low','close']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df.fillna(0.0, inplace=True)
        
        # Calculate Bollinger Bands
        df_bb = calculate_bollinger_bands(df, period)
        if df_bb is None:
            return False, False, {"error": "Bollinger Bands calculation failed"}
        
        close_prices = df_bb['close'].values
        upper_band = df_bb[f'BB_UPPER_{period}'].values
        lower_band = df_bb[f'BB_LOWER_{period}'].values
        
        # Find prices below lower band and above upper band
        below_lower_mask = close_prices < lower_band
        above_upper_mask = close_prices > upper_band
        
        below_lower_indices = np.where(below_lower_mask)[0]
        above_upper_indices = np.where(above_upper_mask)[0]
        
        # Find most recent occurrences
        last_below_lower_idx = below_lower_indices[-1] if len(below_lower_indices) > 0 else None
        last_above_upper_idx = above_upper_indices[-1] if len(above_upper_indices) > 0 else None
        
        # Determine which is more recent
        lowest_below_more_recent = False
        highest_above_more_recent = False
        
        if last_below_lower_idx is not None and last_above_upper_idx is not None:
            lowest_below_more_recent = last_below_lower_idx > last_above_upper_idx
            highest_above_more_recent = last_above_upper_idx > last_below_lower_idx
        elif last_below_lower_idx is not None:
            lowest_below_more_recent = True
        elif last_above_upper_idx is not None:
            highest_above_more_recent = True
        
        details = {
            "last_below_lower_idx": last_below_lower_idx,
            "last_above_upper_idx": last_above_upper_idx,
            "lowest_below_more_recent": lowest_below_more_recent,
            "highest_above_more_recent": highest_above_more_recent,
            "current_price": close_prices[-1],
            "bb_upper": upper_band[-1],
            "bb_lower": lower_band[-1]
        }
        
        return lowest_below_more_recent, highest_above_more_recent, details
        
    except Exception as e:
        print(f"analyze_bollinger_bands_condition error: {e}")
        return False, False, {"error": str(e)}

def analyze_momentum_condition(client, symbol, timeframe='1m', lookback=1200):
    """Analyze momentum condition with most recent negative/positive detection."""
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=lookback)
        if not klines or len(klines) < 100:
            return False, 0.0, {"error": "Insufficient data"}
            
        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
        
        for c in ['open','high','low','close']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df.fillna(0.0, inplace=True)
        
        close_prices = df['close'].values
        
        # Calculate momentum (10-period)
        momentum_values = []
        for i in range(10, len(close_prices)):
            momentum = ((close_prices[i] - close_prices[i-10]) / close_prices[i-10]) * 100
            momentum_values.append(momentum)
        
        # Pad beginning with zeros
        momentum_values = [0] * 10 + momentum_values
        
        current_momentum = momentum_values[-1] if momentum_values else 0
        momentum_positive = current_momentum > 0
        
        # Find most negative and most positive momentum in last 1200 values
        recent_momentum = momentum_values[-1200:] if len(momentum_values) >= 1200 else momentum_values
        
        if len(recent_momentum) > 0:
            min_momentum_idx = np.argmin(recent_momentum)
            max_momentum_idx = np.argmax(recent_momentum)
            
            # Convert to global indices
            global_min_idx = min_momentum_idx + (len(momentum_values) - len(recent_momentum))
            global_max_idx = max_momentum_idx + (len(momentum_values) - len(recent_momentum))
            
            # Determine which is more recent
            negative_more_recent = global_min_idx > global_max_idx
            
            details = {
                "current_momentum": current_momentum,
                "momentum_positive": momentum_positive,
                "min_momentum_idx": global_min_idx,
                "max_momentum_idx": global_max_idx,
                "negative_more_recent": negative_more_recent,
                "most_negative_momentum": recent_momentum[min_momentum_idx],
                "most_positive_momentum": recent_momentum[max_momentum_idx]
            }
            
            return momentum_positive and negative_more_recent, current_momentum, details
        else:
            return False, current_momentum, {"error": "No momentum data"}
            
    except Exception as e:
        print(f"analyze_momentum_condition error: {e}")
        return False, 0.0, {"error": str(e)}

def analyze_poly_fit_condition(client, symbol, timeframe='1m', lookback=200):
    """Analyze polynomial fit condition."""
    try:
        klines = client.get_klines(symbol=symbol, interval=timeframe, limit=lookback)
        if not klines or len(klines) < 50:
            return False, {"error": "Insufficient data"}
            
        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
        
        for c in ['open','high','low','close']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df.fillna(0.0, inplace=True)
        
        close_prices = df['close'].values
        timestamps = np.arange(len(close_prices))
        
        # Check polynomial fit condition
        below_poly_fit = check_polynomial_fit(close_prices, timestamps)
        
        details = {
            "below_poly_fit": below_poly_fit,
            "current_price": close_prices[-1]
        }
        
        return below_poly_fit, details
        
    except Exception as e:
        print(f"analyze_poly_fit_condition error: {e}")
        return False, {"error": str(e)}

# ------------------ Trade Execution Functions ------------------

def execute_buy_order(client, symbol, usdc_amount):
    """Execute a market buy order using 100% of available USDC."""
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        
        quantity = usdc_amount / current_price * 0.99  # 1% buffer for fees
        
        symbol_info = client.get_symbol_info(symbol)
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = float(lot_size_filter['stepSize'])
            quantity = round(quantity - (quantity % step_size), 8)
        
        order = client.order_market_buy(
            symbol=symbol,
            quantity=quantity
        )
        
        return {
            'success': True,
            'order_id': order['orderId'],
            'symbol': symbol,
            'quantity': quantity,
            'price': current_price,
            'cost': quantity * current_price,
            'timestamp': datetime.now(),
            'order': order
        }
    except BinanceAPIException as e:
        return {
            'success': False,
            'error': str(e)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def execute_sell_order(client, symbol, quantity):
    """Execute a market sell order."""
    try:
        symbol_info = client.get_symbol_info(symbol)
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = float(lot_size_filter['stepSize'])
            quantity = round(quantity - (quantity % step_size), 8)
        
        order = client.order_market_sell(
            symbol=symbol,
            quantity=quantity
        )
        
        return {
            'success': True,
            'order_id': order['orderId'],
            'symbol': symbol,
            'quantity': quantity,
            'timestamp': datetime.now(),
            'order': order
        }
    except BinanceAPIException as e:
        return {
            'success': False,
            'error': str(e)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_current_price(client, symbol):
    """Get the current price for a symbol."""
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        print(f"Error getting current price: {e}")
        return None

def get_account_balance(client, asset):
    """Get the balance of a specific asset in the account."""
    try:
        account_info = client.get_account()
        for balance in account_info['balances']:
            if balance['asset'] == asset:
                return float(balance['free'])
        return 0.0
    except Exception as e:
        print(f"Error getting account balance: {e}")
        return 0.0

def monitor_trade(client, symbol, entry_price, entry_time, quantity):
    """Monitor the trade and sell when profit target is reached."""
    global trade_active, trade_info
    
    target_price = entry_price * (1 + (PROFIT_TARGET_PERCENT + TOTAL_FEE_PERCENT) / 100)
    
    while trade_active and not stop_event.is_set():
        try:
            current_price = get_current_price(client, symbol)
            if current_price is None:
                time.sleep(5)
                continue
            
            price_diff = current_price - entry_price
            price_diff_pct = (price_diff / entry_price) * 100
            time_elapsed = datetime.now() - entry_time
            
            target_diff = target_price - current_price
            target_diff_pct = (target_diff / current_price) * 100
            
            trade_info = {
                'symbol': symbol,
                'entry_price': entry_price,
                'current_price': current_price,
                'price_diff': price_diff,
                'price_diff_pct': price_diff_pct,
                'time_elapsed': time_elapsed,
                'target_price': target_price,
                'target_diff': target_diff,
                'target_diff_pct': target_diff_pct,
                'quantity': quantity
            }
            
            os.system('cls' if os.name == 'nt' else 'clear')
            print("="*80)
            print("TRADE MONITOR - ACTIVE POSITION")
            print("="*80)
            print(f"Symbol: {symbol}")
            print(f"Entry Price: {entry_price:.6f}")
            print(f"Entry Time: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Current Price: {current_price:.6f}")
            print(f"Price Difference: {price_diff:.6f} ({price_diff_pct:+.2f}%)")
            print(f"Time Elapsed: {time_elapsed}")
            print(f"Target Price: {target_price:.6f}")
            print(f"Distance to Target: {target_diff:.6f} ({target_diff_pct:.2f}%)")
            print(f"Quantity: {quantity}")
            print("="*80)
            
            if current_price >= target_price:
                print(f"PROFIT TARGET REACHED! Selling at {current_price:.6f}")
                sell_result = execute_sell_order(client, symbol, quantity)
                
                if sell_result['success']:
                    print(f"SELL ORDER EXECUTED SUCCESSFULLY!")
                    print(f"Order ID: {sell_result['order_id']}")
                    print(f"Quantity Sold: {sell_result['quantity']}")
                    print(f"Estimated Profit: {(current_price - entry_price) * quantity:.6f} USDC")
                    trade_active = False
                    return True
                else:
                    print(f"ERROR EXECUTING SELL ORDER: {sell_result['error']}")
            
            time.sleep(5)
        except Exception as e:
            print(f"Error in trade monitoring: {e}")
            time.sleep(5)
    
    return False

# ------------------ Main Analysis Function ------------------

def perform_single_iteration_analysis(client):
    """Perform single iteration analysis with all conditions."""
    global trade_active
    
    if trade_active:
        print("Trade already active, skipping analysis...")
        return
    
    # Clear screen for fresh iteration
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("="*80)
    print(f"BTCUSDC TRADING BOT - SINGLE ITERATION ANALYSIS")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Step 1: Check and convert BTC dust
    print("\n1. Checking for BTC dust...")
    dust_success = convert_btc_dust_to_usdc(client)
    if not dust_success:
        print("Dust conversion failed or not needed")
    
    # Step 2: Get current balance
    usdc_balance = get_account_balance(client, 'USDC')
    print(f"USDC Balance: {usdc_balance:.2f}")
    
    if usdc_balance < MIN_TRADE_AMOUNT:
        print(f"!!! INSUFFICIENT USDC BALANCE - MINIMUM REQUIRED: {MIN_TRADE_AMOUNT} !!!")
        return
    
    # Step 3: Perform all condition analyses
    print("\n2. Performing technical analysis...")
    
    conditions_met = 0
    total_conditions = 7
    condition_details = {}
    
    # Condition 1: ArgMin vs ArgMax for all timeframes
    print("\n--- Condition 1: ArgMin vs ArgMax Analysis ---")
    all_tf_argmin_met = True
    for timeframe in TIMEFRAMES:
        argmin_met, dist_min, dist_max, details = analyze_argmin_argmax_condition(client, SYMBOL, timeframe, 1200)
        condition_details[f'argmin_{timeframe}'] = {
            'met': argmin_met,
            'dist_min': dist_min,
            'dist_max': dist_max,
            'details': details
        }
        print(f"{timeframe}: Min More Recent: {details.get('min_more_recent', False)}, "
              f"Dist to Min: {dist_min:.2f}%, Dist to Max: {dist_max:.2f}%, "
              f"Condition: {argmin_met}")
        
        if not argmin_met:
            all_tf_argmin_met = False
    
    if all_tf_argmin_met:
        conditions_met += 1
        print("✓ ALL TIMEFRAMES: ArgMin condition MET")
    else:
        print("✗ ArgMin condition NOT met for all timeframes")
    
    # Condition 2: Volume Bullish vs Bearish
    print("\n--- Condition 2: Volume Analysis ---")
    volume_met, bull_pct, bear_pct, vol_details = analyze_volume_condition(client, SYMBOL, '1m', 50)
    condition_details['volume'] = {
        'met': volume_met,
        'bull_pct': bull_pct,
        'bear_pct': bear_pct,
        'details': vol_details
    }
    print(f"Bullish Volume: {bull_pct:.2f}%, Bearish Volume: {bear_pct:.2f}%")
    print(f"Condition: {volume_met}")
    
    if volume_met:
        conditions_met += 1
        print("✓ Volume condition MET")
    else:
        print("✗ Volume condition NOT met")
    
    # Condition 3: RSI Condition
    print("\n--- Condition 3: RSI Analysis ---")
    rsi_oversold_recent, rsi_overbought_recent, current_rsi, rsi_details = analyze_rsi_condition(client, SYMBOL, '1m', 1200)
    condition_details['rsi'] = {
        'oversold_most_recent': rsi_oversold_recent,
        'overbought_most_recent': rsi_overbought_recent,
        'current_rsi': current_rsi,
        'details': rsi_details
    }
    print(f"Current RSI: {current_rsi:.2f}")
    print(f"Oversold Most Recent: {rsi_oversold_recent}")
    print(f"Overbought Most Recent: {rsi_overbought_recent}")
    print(f"Condition: {rsi_oversold_recent and not rsi_overbought_recent}")
    
    if rsi_oversold_recent and not rsi_overbought_recent:
        conditions_met += 1
        print("✓ RSI condition MET")
    else:
        print("✗ RSI condition NOT met")
    
    # Condition 4: Merged Stochastic Condition
    print("\n--- Condition 4: Merged Stochastic Analysis ---")
    stoch_oversold_recent, stoch_overbought_recent, stoch_k, stoch_d, stoch_details = analyze_merged_stochastic_condition(client, SYMBOL, TIMEFRAMES, 500)
    condition_details['stochastic'] = {
        'oversold_most_recent': stoch_oversold_recent,
        'overbought_most_recent': stoch_overbought_recent,
        'stoch_k': stoch_k,
        'stoch_d': stoch_d,
        'details': stoch_details
    }
    print(f"Merged Stochastic K: {stoch_k:.2f}, D: {stoch_d:.2f}")
    print(f"Oversold Most Recent: {stoch_oversold_recent}")
    print(f"Overbought Most Recent: {stoch_overbought_recent}")
    print(f"Condition: {stoch_oversold_recent and not stoch_overbought_recent}")
    
    if stoch_oversold_recent and not stoch_overbought_recent:
        conditions_met += 1
        print("✓ Stochastic condition MET")
    else:
        print("✗ Stochastic condition NOT met")
    
    # Condition 5: Bollinger Bands Condition
    print("\n--- Condition 5: Bollinger Bands Analysis ---")
    bb_lowest_recent, bb_highest_recent, bb_details = analyze_bollinger_bands_condition(client, SYMBOL, '1m', BB_PERIOD, 500)
    condition_details['bollinger_bands'] = {
        'lowest_below_more_recent': bb_lowest_recent,
        'highest_above_more_recent': bb_highest_recent,
        'details': bb_details
    }
    print(f"Lowest Below More Recent: {bb_lowest_recent}")
    print(f"Highest Above More Recent: {bb_highest_recent}")
    print(f"Condition: {bb_lowest_recent and not bb_highest_recent}")
    
    if bb_lowest_recent and not bb_highest_recent:
        conditions_met += 1
        print("✓ Bollinger Bands condition MET")
    else:
        print("✗ Bollinger Bands condition NOT met")
    
    # Condition 6: Momentum Condition
    print("\n--- Condition 6: Momentum Analysis ---")
    momentum_met, current_momentum, mom_details = analyze_momentum_condition(client, SYMBOL, '1m', 1200)
    condition_details['momentum'] = {
        'met': momentum_met,
        'current_momentum': current_momentum,
        'details': mom_details
    }
    print(f"Current Momentum: {current_momentum:.4f}")
    print(f"Momentum > 0: {mom_details.get('momentum_positive', False)}")
    print(f"Negative More Recent: {mom_details.get('negative_more_recent', False)}")
    print(f"Condition: {momentum_met}")
    
    if momentum_met:
        conditions_met += 1
        print("✓ Momentum condition MET")
    else:
        print("✗ Momentum condition NOT met")
    
    # Condition 7: Polynomial Fit Condition
    print("\n--- Condition 7: Polynomial Fit Analysis ---")
    poly_met, poly_details = analyze_poly_fit_condition(client, SYMBOL, '1m', 200)
    condition_details['poly_fit'] = {
        'met': poly_met,
        'details': poly_details
    }
    print(f"Below Poly Fit: {poly_met}")
    
    if poly_met:
        conditions_met += 1
        print("✓ Polynomial Fit condition MET")
    else:
        print("✗ Polynomial Fit condition NOT met")
    
    # Step 4: Trading Decision
    print("\n" + "="*80)
    print("TRADING DECISION")
    print("="*80)
    print(f"Conditions Met: {conditions_met}/{total_conditions}")
    
    # All conditions must be met for trade entry
    if conditions_met == total_conditions:
        print("!!! ALL CONDITIONS MET - EXECUTING TRADE !!!")
        
        # Execute buy order with 100% USDC balance
        buy_result = execute_buy_order(client, SYMBOL, usdc_balance)
        
        if buy_result['success']:
            print(f"BUY ORDER EXECUTED SUCCESSFULLY!")
            print(f"Order ID: {buy_result['order_id']}")
            print(f"Quantity: {buy_result['quantity']:.6f}")
            print(f"Price: {buy_result['price']:.6f}")
            print(f"Cost: {buy_result['cost']:.2f} USDC")
            
            # Start trade monitoring
            trade_active = True
            monitor_trade(client, SYMBOL, buy_result['price'], buy_result['timestamp'], buy_result['quantity'])
        else:
            print(f"ERROR EXECUTING BUY ORDER: {buy_result['error']}")
    else:
        print("!!! CONDITIONS NOT MET - NO TRADE EXECUTED !!!")
        print("Waiting for next iteration...")
    
    # Step 5: Cleanup for next iteration
    print("\nCleaning up for next iteration...")
    gc.collect()

# ------------------ Main Loop ------------------

def get_binance_client():
    if not os.path.exists(API_FILE):
        print(f"API file '{API_FILE}' not found. Create with key then secret on two lines.")
        return None
    try:
        with open(API_FILE, 'r') as f:
            api_key = f.readline().strip()
            api_secret = f.readline().strip()
        return Client(api_key, api_secret)
    except Exception as e:
        print(f"Error reading API file: {e}")
        return None

# ------------------ Main Loop (UPDATED PART) ------------------
def main():
    client = get_binance_client()
    if not client:
        print("No client available. Exiting.")
        return
   
    print("=== BTCUSDC TRADING BOT - SINGLE ITERATION MODE ===")
    print("Press Ctrl+C to stop monitoring.")
    print("Each iteration will:")
    print("1. Check and convert BTC dust")
    print("2. Analyze all 7 trading conditions")
    print("3. Execute trade if ALL conditions met")
    print("4. Use 100% USDC balance for entry")
    print("5. Clean up for next iteration")
    print("="*60)
   
    iteration_count = 0
   
    while not stop_event.is_set():
        iteration_count += 1
        print(f"\n>>> Starting Iteration #{iteration_count} <<<")
       
        try:
            perform_single_iteration_analysis(client)
        except Exception as e:
            print(f"Error in iteration #{iteration_count}: {e}")
       
        # ----- NEW: Always wait exactly 5 seconds between iterations -----
        if stop_event.is_set():
            break
            
        print(f"\nWaiting 5 seconds before next iteration...")
        for i in range(5, 0, -1):
            if stop_event.is_set():
                break
            print(f"\rNext iteration in: {i} seconds", end="", flush=True)
            time.sleep(1)
        print("\r" + " " * 40 + "\r", end="")  # clear the countdown line

        # Even when a trade is active we still run a new analysis every 5 seconds
        # (the monitor thread already runs independently)
   
    print("\nTrading bot stopped.")

if __name__ == "__main__":
    main()