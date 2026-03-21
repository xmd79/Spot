from binance.client import Client
import matplotlib.pyplot as plt
import numpy as np
import talib as ta
import os
import sys

class Trader:
    def __init__(self, file):
        self.connect(file)

    """ Creates Binance client """
    def connect(self, file):
        lines = [line.rstrip('\n') for line in open(file)]
        key = lines[0]
        secret = lines[1]
        self.client = Client(key, secret)

    """ Gets all account balances """
    def getBalances(self):
        prices = self.client.get_withdraw_history()
        return prices

    """ Get all pairs traded against USDC """
    def get_usdc_pairs(self):
        exchange_info = self.client.get_exchange_info()
        trading_pairs = [symbol['symbol'] for symbol in exchange_info['symbols'] if symbol['quoteAsset'] == 'USDC' and symbol['status'] == 'TRADING']
        return trading_pairs

filename = 'credentials.txt'
trader = Trader(filename)

filtered_pairs1 = []
filtered_pairs2 = []
filtered_pairs3 = []
selected_pair = []
selected_pairCMO = []

# Dynamically fetch trading pairs against USDC
trading_pairs = trader.get_usdc_pairs()

def filter1(pair):
    interval = '2h'
    symbol = pair
    klines = trader.client.get_klines(symbol=symbol, interval=interval)
    close = [float(entry[4]) for entry in klines]

    if not close:
        return  # Skip if no close price available

    print("on 2h timeframe " + symbol)

    x = close
    y = range(len(x))

    best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
    best_fit_line2 = best_fit_line1 * 1.01
    best_fit_line3 = best_fit_line1 * 0.99
 
    if x[-1] < best_fit_line3[-1]:
        filtered_pairs1.append(symbol)
        print('found')

def filter2(filtered_pairs1):
    interval = '15m'
    for symbol in filtered_pairs1:  # Loop through each symbol in the filtered pair list
        klines = trader.client.get_klines(symbol=symbol, interval=interval)
        close = [float(entry[4]) for entry in klines]

        if not close:
            continue  # Skip if no close price available

        print("on 15min timeframe " + symbol)

        x = close
        y = range(len(x))

        best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
        best_fit_line2 = best_fit_line1 * 1.01
        best_fit_line3 = best_fit_line1 * 0.99

        if x[-1] < best_fit_line3[-1]:
            filtered_pairs2.append(symbol)
            print('found')

def filter3(filtered_pairs2):
    interval = '5m'
    for symbol in filtered_pairs2:  # Loop through each symbol in the filtered pair list
        klines = trader.client.get_klines(symbol=symbol, interval=interval)
        close = [float(entry[4]) for entry in klines]

        if not close:
            continue  # Skip if no close price available

        print("on 5m timeframe " + symbol)
        
        x = close
        y = range(len(x))

        best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
        best_fit_line2 = best_fit_line1 * 1.01
        best_fit_line3 = best_fit_line1 * 0.99

        if x[-1] < best_fit_line3[-1]:
            filtered_pairs3.append(symbol)
            print('found')

def momentum(filtered_pairs3):
    interval = '1m'
    for symbol in filtered_pairs3:  # Loop through each symbol in the filtered pair list
        klines = trader.client.get_klines(symbol=symbol, interval=interval)
        close = [float(entry[4]) for entry in klines]

        if not close:
            continue  # Skip if no close price available

        print("on 1m timeframe " + symbol)

        close_array = np.asarray(close)

        real = ta.CMO(close_array, timeperiod=14)

        if real[-1] < -50:
            print('mtf dip found')
            selected_pair.append(symbol)
            selected_pairCMO.append(real[-1])

# Run filters on trading pairs
for i in trading_pairs:
    filter1(i)

filter2(filtered_pairs1)
filter3(filtered_pairs2)
momentum(filtered_pairs3)

if len(selected_pair) > 1:
    print('more mtf dips are found') 
    print(selected_pair) 

    if min(selected_pairCMO) in selected_pairCMO:
        position = selected_pairCMO.index(min(selected_pairCMO))

        for id, value in enumerate(selected_pair):
            if id == position:
                print(selected_pair[id])

elif len(selected_pair) == 1:
    print('1 mtf dip found')   
    print(selected_pair) 

sys.exit(0)
exit()