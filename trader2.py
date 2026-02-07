import numpy as np
import pandas as pd
import signal
import threading
from datetime import datetime
from scipy.signal import argrelextrema
from scipy.fft import fft
from decimal import Decimal, getcontext
from binance.client import Client

# ==========================
# GLOBAL DECIMAL PRECISION
# ==========================

getcontext().prec = 50

def D(x):
    return Decimal(str(x))

def f25(x):
    return f"{D(x):.25f}"

# ==========================
# CONFIG
# ==========================

SYMBOL = "BTCUSDC"

FAST_TF = ["1m","3m","5m"]
INTER_TF = ["15m","30m","1h","2h"]
MAJOR_TF = ["4h","8h","12h","1d"]

ALL_TF = FAST_TF + INTER_TF + MAJOR_TF

TF_WEIGHTS = {
    "1m":1,"3m":1,"5m":1,
    "15m":2,"30m":2,"1h":3,"2h":3,
    "4h":5,"8h":6,"12h":7,"1d":10
}

LOOKBACK = 1000
SCAN_INTERVAL = 5

client = Client()

# ==========================
# STOP EVENT
# ==========================

stop_event = threading.Event()

def signal_handler(sig, frame):
    print("\nStopping Predictive Cycle Engine...")
    stop_event.set()

signal.signal(signal.SIGINT, signal_handler)

# ==========================
# STATE
# ==========================

class CycleEngineState:
    def __init__(self):
        self.targets = {}
        self.cycles = {}
        self.phase_score = {}
        self.vol_comp = {}
        self.bull_bear_vol = {}
        self.resonance_score = {}
        self.phase_ladder = {}
        self.flow_index = 0

state = CycleEngineState()

# ==========================
# DATA
# ==========================

def get_data(symbol, tf):
    klines = client.get_klines(symbol=symbol, interval=tf, limit=LOOKBACK)
    df = pd.DataFrame(klines, columns=[
        "time","open","high","low","close","vol",
        "ct","qv","nt","tb","tq","ig"
    ])
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(inplace=True)
    return df

# ==========================
# CORE ANALYSIS
# ==========================

def spectral_phase(close):
    f = fft(close)
    return float(np.angle(f[1]))

def detect_cycle(phase):
    return "UP" if phase > 0 else "DOWN"

def vol_compression(close):
    return float(np.std(close[-50:]) / np.std(close))

def bullish_bearish_vol(df):
    bullish = df[df.close >= df.open]["vol"].sum()
    bearish = df[df.close < df.open]["vol"].sum()
    total = bullish + bearish
    if total == 0:
        return 0.5,0.5
    return float(bullish/total), float(bearish/total)

# ==========================
# STRUCTURED SR
# ==========================

def structured_support_resistance(df):

    close = df["close"].values
    vol = df["vol"].values

    mins = argrelextrema(close, np.less_equal, order=5)[0]
    maxs = argrelextrema(close, np.greater_equal, order=5)[0]

    support = np.average(close[mins],weights=vol[mins]) if len(mins)>0 else np.min(close)
    resistance = np.average(close[maxs],weights=vol[maxs]) if len(maxs)>0 else np.max(close)

    last = close[-1]

    if support >= last:
        support = last*0.995

    if resistance <= last:
        resistance = last*1.005

    if last < support:
        support = last*0.999

    if last > resistance:
        resistance = last*1.001

    return float(support), float(resistance)

# ==========================
# INSTITUTIONAL PHASE LADDER
# ==========================

def institutional_phase_ladder(phase, vol_comp, bull, bear):

    imbalance = bull-bear

    if vol_comp < 0.5 and abs(imbalance) < 0.1:
        return "Accumulation"

    if phase > 0 and imbalance > 0.1:
        return "Expansion"

    if vol_comp > 0.8 and abs(imbalance)<0.1:
        return "Distribution"

    if phase < 0 and imbalance < -0.1:
        return "Capitulation"

    return "Expansion"

# ==========================
# LIQUIDITY MAGNET TARGET
# ==========================

def liquidity_magnet_target(df, cycle):

    close = df["close"].values
    vol = df["vol"].values
    last = close[-1]

    mins = argrelextrema(close,np.less_equal,order=5)[0]
    maxs = argrelextrema(close,np.greater_equal,order=5)[0]

    liquidity_lows = close[mins] if len(mins)>0 else [np.min(close)]
    liquidity_highs = close[maxs] if len(maxs)>0 else [np.max(close)]

    low_cluster = np.average(liquidity_lows)
    high_cluster = np.average(liquidity_highs)

    if cycle=="UP":
        target = max(high_cluster,last*1.005)
    else:
        target = min(low_cluster,last*0.995)

    return float(target)

# ==========================
# TARGET STATE ENGINE
# ==========================

def update_stable_target(tf, cycle, new_target, price):

    old_cycle = state.cycles.get(tf)
    old_target = state.targets.get(tf)

    if old_cycle != cycle:
        state.cycles[tf]=cycle
        state.targets[tf]=new_target
        return

    if old_target is None:
        state.targets[tf]=new_target
        return

    if cycle=="UP" and price>=old_target:
        state.targets[tf]=new_target

    if cycle=="DOWN" and price<=old_target:
        state.targets[tf]=new_target

# ==========================
# FLOW INDEX
# ==========================

def compute_flow_index():

    total_weight=0
    flow=0

    for tf,(bull,bear) in state.bull_bear_vol.items():

        weight=TF_WEIGHTS.get(tf,1)
        imbalance=bull-bear

        flow+=imbalance*weight
        total_weight+=weight

    if total_weight==0:
        return 0

    return flow/total_weight

# ==========================
# RESONANCE
# ==========================

def resonance_alignment(cycles):
    up=cycles.count("UP")
    down=cycles.count("DOWN")
    return (up-down)/len(cycles)

# ==========================
# MAIN LOOP
# ==========================

print("=== INSTITUTIONAL PREDICTIVE CYCLE ENGINE STARTED ===")

while not stop_event.is_set():

    try:

        table=[]
        cycles_all=[]

        for tf in ALL_TF:

            df=get_data(SYMBOL,tf)
            close=df["close"].values
            price=float(close[-1])

            phase=spectral_phase(close)
            cycle=detect_cycle(phase)

            bull,bear=bullish_bearish_vol(df)
            vcomp=vol_compression(close)

            support,resistance=structured_support_resistance(df)

            target=liquidity_magnet_target(df,cycle)
            update_stable_target(tf,cycle,target,price)

            ladder= institutional_phase_ladder(phase,vcomp,bull,bear)

            state.phase_score[tf]=phase
            state.vol_comp[tf]=vcomp
            state.bull_bear_vol[tf]=(bull,bear)
            state.phase_ladder[tf]=ladder

            cycles_all.append(cycle)

            table.append({
                "tf":tf,
                "cycle":cycle,
                "price":price,
                "target":state.targets[tf],
                "bull":bull,
                "bear":bear,
                "support":support,
                "resistance":resistance,
                "ladder":ladder
            })

        state.flow_index=compute_flow_index()

        resonance=resonance_alignment(cycles_all)

        # ======================
        # PRINT TABLE
        # ======================

        print("\n",datetime.now())
        print("\n=== FULL MTF INSTITUTIONAL TABLE ===")

        for row in table:

            print("\nTF:",row["tf"])
            print("Cycle:",row["cycle"])
            print("PhaseLadder:",row["ladder"])
            print("Price:",f25(row["price"]))
            print("Target:",f25(row["target"]))
            print("BullVol%:",f25(row["bull"]))
            print("BearVol%:",f25(row["bear"]))
            print("Support:",f25(row["support"]))
            print("Resistance:",f25(row["resistance"]))
            print("PhaseScore:",f25(state.phase_score[row["tf"]]))

        # ======================
        # OVERALL SUMMARY
        # ======================

        up=cycles_all.count("UP")
        down=cycles_all.count("DOWN")

        print("\n=== OVERALL INSTITUTIONAL SUMMARY ===")
        print("Total TF UP:",up)
        print("Total TF DOWN:",down)
        print("Overall Cycle Bias:","UP" if up>=down else "DOWN")
        print("MTF Resonance Score:",f25(resonance))
        print("Institutional Flow Index:",f25(state.flow_index))

        price_now=table[0]["price"]

        priority=sorted(
            [(r["tf"],r["target"]) for r in table],
            key=lambda x:abs(x[1]-price_now)
        )

        print("\nTarget Priority (closest first):")

        for i,(tf,t) in enumerate(priority,1):
            print(f"{i}. {tf} Target {f25(t)} (distance {f25(abs(t-price_now))})")

        stop_event.wait(SCAN_INTERVAL)

    except Exception as e:
        print("Engine Error:",e)
        stop_event.wait(2)

print("Engine stopped cleanly.")
