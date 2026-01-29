import ephem
import requests
import numpy as np
import pandas as pd
import math
import pytz
from datetime import datetime, timezone, timedelta
from binance.client import Client
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION & ASTRO-PHYSICS
# ==========================================

PLATONIC_SOLIDS = {
    "Sun": "Sphere (Monad)", "Moon": "Icosahedron (Water)",
    "Mercury": "Octahedron (Air)", "Venus": "Dodecahedron (Aether)",
    "Mars": "Tetrahedron (Fire)", "Jupiter": "Cube (Earth Structure)",
    "Saturn": "Cube (Limitation)", "Uranus": "Dodecahedron (Higher Ether)",
    "Neptune": "Icosahedron (Oceanic)", "Pluto": "Tetrahedron (Underworld Fire)"
}

PLANETARY_FREQUENCIES = {
    "Sun": 126.22, "Moon": 210.42, "Mercury": 141.27, "Venus": 221.23,
    "Mars": 144.72, "Jupiter": 183.58, "Saturn": 147.85,
    "Uranus": 207.36, "Neptune": 211.44, "Pluto": 140.64
}

# Planetary Hours Sequence
PLANETARY_HOUR_ORDER = ["Saturn", "Jupiter", "Mars", "Sun", "Venus", "Mercury", "Moon"]
DAY_RULERS = {
    "Monday": "Moon", "Tuesday": "Mars", "Wednesday": "Mercury",
    "Thursday": "Jupiter", "Friday": "Venus", "Saturday": "Saturn", "Sunday": "Sun"
}

# Obliquity of Ecliptic
EPSILON = math.radians(23.44)

# ==========================================
# 2. HELPER FUNCTIONS (ASTRO MATH)
# ==========================================

def to_geocentric(body):
    """Manually converts Equatorial RA/Dec to Ecliptic Longitude."""
    try:
        ra = body.ra 
        dec = body.dec
    except AttributeError:
        ra = body.ra
        dec = body.dec
        
    ra_deg = math.degrees(ra)
    dec_deg = math.degrees(dec)
    
    sin_ra = math.sin(math.radians(ra_deg))
    cos_ra = math.cos(math.radians(ra_deg))
    sin_dec = math.sin(math.radians(dec_deg))
    cos_dec = math.cos(math.radians(dec_deg))
    sin_eps = math.sin(EPSILON)
    cos_eps = math.cos(EPSILON)
    
    y = sin_dec * cos_eps - cos_dec * sin_eps * cos_ra
    x = cos_dec * cos_ra
    
    ecl_lon = math.degrees(math.atan2(y, x))
    if ecl_lon < 0: ecl_lon += 360
    return ecl_lon

def get_planetary_hour_ruler(dt):
    """Calculates planetary hour ruler for a specific datetime."""
    day_name = dt.strftime("%A")
    start_index = PLANETARY_HOUR_ORDER.index(DAY_RULERS.get(day_name, "Sun"))
    hour_num = int(dt.hour) 
    ruler_index = (start_index + hour_num) % 7
    return PLANETARY_HOUR_ORDER[ruler_index]

# ==========================================
# 3. CLASS DEFINITIONS
# ==========================================

class CelestialBody:
    def __init__(self, name, key):
        self.name = name
        self.key = key
        self.body = self._get_body(key)
        self.frequency = PLANETARY_FREQUENCIES.get(key, 0.0)
        self.current_aspects = []
        self.lon = 0.0
        self.lat = 0.0
        self.alt = 0.0
        self.speed = 0.0 
        self.is_retrograde = False
        self.zodiac_sign = ""
        self.deg_in_sign = 0.0
        self.platonic = PLATONIC_SOLIDS.get(name, "Unknown")
        
    def _get_body(self, key):
        map_ = {
            "Sun": ephem.Sun(), "Moon": ephem.Moon(), "Mercury": ephem.Mercury(),
            "Venus": ephem.Venus(), "Mars": ephem.Mars(), "Jupiter": ephem.Jupiter(),
            "Saturn": ephem.Saturn(), "Uranus": ephem.Uranus(),
            "Neptune": ephem.Neptune(), "Pluto": ephem.Pluto()
        }
        return map_.get(key)

    def compute(self, observer):
        if not self.body: return
        self.body.compute(observer)
        self.lon = to_geocentric(self.body)
        self.lat = math.degrees(self.body.dec) 
        self.alt = math.degrees(self.body.alt)
        
        signs = ["Ari", "Tau", "Gem", "Can", "Leo", "Vir", "Lib", "Sco", "Sag", "Cap", "Aqu", "Pis"]
        sign_idx = int(self.lon / 30) % 12
        self.zodiac_sign = signs[sign_idx]
        self.deg_in_sign = self.lon % 30
        
        # Retrograde calc
        obs_f = observer.copy()
        obs_f.date = observer.date + 1
        body_f = self._get_body(self.key)
        body_f.compute(obs_f)
        
        lon_f = to_geocentric(body_f)
        diff = lon_f - self.lon
        if diff < -180: diff += 360
        if diff > 180: diff -= 360
        
        self.is_retrograde = diff < 0
        self.speed = abs(diff)

class MarketEngine:
    def __init__(self):
        self.client = Client("", "")
        # Storage structure: { '1h': { 'Fast': model, 'Medium': model ... }, '4h': ... }
        self.models = {}
        self.scalers = {}

    def fetch_data(self, symbol='BTCUSDC', interval='1h', limit=1000):
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qv', 'trades', 'tbv', 'tv', 'ignore'])
            df['close'] = pd.to_numeric(df['close'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['volume'] = pd.to_numeric(df['volume'])
            return df
        except Exception as e:
            print(f"[ERROR] Binance API Error for {interval}: {e}")
            return pd.DataFrame()

    def calculate_technicals(self, df):
        """Calculates technicals on provided dataframe."""
        if df.empty: return df
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        try:
            import talib
            df['sine'], _ = talib.HT_SINE(df['close'])
            df['sine'] = df['sine'].fillna(0)
        except:
            std_dev = df['close'].std()
            t = np.arange(len(df))
            df['sine'] = np.sin(2 * np.pi * 0.1 * t) * std_dev

        df['Returns'] = df['close'].pct_change()
        df['Lagged_Close'] = df['close'].shift(1)
        return df.dropna().reset_index(drop=True)

    def train_and_predict_tf(self, df, tf_name):
        """
        Trains models specifically for the passed Timeframe (df).
        Returns a dictionary of predictions for Fast, Med, Large.
        """
        if df.empty or 'close' not in df.columns:
            return None

        print(f"    [ML] Training model for {tf_name}...", end="\r")
        
        df_train = df.copy()
        # Define Targets (Fixed candle shifts regardless of TF)
        # Fast = 4 candles, Med = 24 candles, Large = 168 candles
        df_train['Target_Fast'] = df_train['close'].shift(-4)
        df_train['Target_Med'] = df_train['close'].shift(-24)
        df_train['Target_Large'] = df_train['close'].shift(-168)
        
        features = ['RSI', 'sine', 'Returns', 'Lagged_Close']
        
        tf_models = {}
        tf_scalers = {}
        predictions = {}

        targets_map = {'Fast': df_train['Target_Fast'], 'Medium': df_train['Target_Med'], 'Large': df_train['Target_Large']}
        
        # Train separate models for each horizon
        for name, y in targets_map.items():
            temp_df = pd.concat([df_train[features], y], axis=1).dropna()
            
            # Ensure we have enough data
            if len(temp_df) < 50: 
                continue
            
            X_train = temp_df[features].values
            y_train = temp_df[y.name].values
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_train)
            
            # Reduced max_iter to 500 for speed across multiple TFs
            model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
            model.fit(X_scaled, y_train)
            
            tf_models[name] = model
            tf_scalers[name] = scaler

        # Save for potential later use
        self.models[tf_name] = tf_models
        self.scalers[tf_name] = tf_scalers

        # Generate Predictions using the most recent data point
        # We use the 'Fast' scaler (shortest term) to normalize input features, as it's most sensitive to immediate changes
        if 'Fast' in tf_scalers:
            last_row = df.iloc[-1:][features].values
            X_input = tf_scalers['Fast'].transform(last_row)
            
            for name in ['Fast', 'Medium', 'Large']:
                if name in tf_models:
                    pred = tf_models[name].predict(X_input)[0]
                    predictions[name] = pred
        
        print(f"    [ML] Training model for {tf_name}... DONE")
        return predictions

    def detect_gann_cycle(self, df):
        """
        Refines Gann Logic:
        1. Distinguishes Accumulation (Buying at Dip) vs Distribution (Selling at Peak).
        2. Returns Gann Level, Structural Status, and Argument.
        """
        if len(df) < 50: return "NEUTRAL", 0, 0, "No Data", "NEUTRAL"
        
        recent = df.tail(1000)
        max_idx = recent['high'].idxmax()
        min_idx = recent['low'].idxmin()
        max_price = recent['high'].max()
        min_price = recent['low'].min()
        current_price = df['close'].iloc[-1]
        
        is_low_recent = min_idx > max_idx
        is_high_recent = max_idx > min_idx
        
        status = "NEUTRAL"
        gann_level = 0
        arg_desc = ""
        structural_status = "NEUTRAL" 
        
        swing_size = max_price - min_price
        
        if is_low_recent:
            gann_level_up = min_price + swing_size
            if current_price > gann_level_up:
                status = "UP CYCLE"
                arg_desc = f"Breaking {gann_level_up:.0f}"
                structural_status = "ACCUMULATION (Dip Zone)"
            elif current_price < min_price:
                status = "EXTENSION"
                arg_desc = f"Below {min_price:.0f}"
                structural_status = "EXTENSION (Bear Trap)"
            else:
                status = "ACCUMULATION (Choppy)"
                arg_desc = "Ranging at Bottom"
                structural_status = "ACCUMULATION (Waiting)"
        else:
            gann_level_down = max_price - swing_size
            if current_price < gann_level_down:
                status = "DOWN CYCLE"
                arg_desc = f"Breaking {gann_level_down:.0f}"
                structural_status = "DISTRIBUTION (Peak Zone)"
            elif current_price > max_price:
                status = "EXTENSION"
                arg_desc = f"Above {max_price:.0f}"
                structural_status = "EXTENSION (Bull Trap)"
            else:
                status = "DISTRIBUTION (Choppy)"
                arg_desc = "Ranging at Top"
                structural_status = "DISTRIBUTION (Waiting)"
        
        return status, gann_level, arg_desc, structural_status

    def get_fear_greed(self, df):
        if df.empty: return 50, "NEUTRAL"
        rsi = df['RSI'].iloc[-1]
        volatility = df['Returns'].std()
        score = 50
        if rsi > 70: score += 30 
        if rsi < 30: score -= 30 
        if volatility > 0.015: score -= 10 
        score = max(0, min(100, score)) 
        sentiment = "EXTREME GREED" if score > 75 else "GREED" if score > 55 else "NEUTRAL" if score > 45 else "FEAR" if score > 25 else "EXTREME FEAR"
        return score, sentiment

# ==========================================
# 5. GEOLOCATION & TIME SYNC
# ==========================================

def get_utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)

def get_observer_data():
    sources = ["https://ipapi.co/json/", "http://ip-api.com/json/", "https://ipwho.is/"]
    lat, lon, city, tz_str = None, None, None, None
    
    print("[GEO] Searching for location...")
    for url in sources:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                lat_raw = data.get('lat') or data.get('latitude')
                lon_raw = data.get('lon') or data.get('longitude')
                city_raw = data.get('city') or data.get('region')
                tz_raw = data.get('timezone') or data.get('time_zone')
                if lat_raw and lon_raw:
                    lat, lon = float(lat_raw), float(lon_raw)
                    city, tz_str = str(city_raw), str(tz_raw)
                    print(f"[GEO] SUCCESS: {city} (Lat: {lat}, Lon: {lon}, TZ: {tz_str})")
                    break 
        except:
            continue

    if not lat:
        print("[WARN] All Geo APIs failed. Defaulting to Greenwich.")
        lat, lon, city, tz_str = 51.4778, -0.0014, "Greenwich", "Europe/London"

    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.elevation = 0
    live_time_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    observer.date = live_time_utc
    
    try:
        local_tz = pytz.timezone(tz_str)
        if live_time_utc.tzinfo is None:
            live_time_utc = pytz.UTC.localize(live_time_utc)
        local_time = live_time_utc.astimezone(local_tz)
    except:
        local_time = live_time_utc
        
    return observer, local_time, city, live_time_utc

# ==========================================
# 6. ASTRO & GANN LOGIC
# ==========================================

def calculate_aspects(bodies):
    major_aspects = {0: "Conj", 60: "Sextile", 90: "Square", 120: "Trine", 180: "Opp"}
    orb_limit = 6 
    for i, b1 in enumerate(bodies):
        for j, b2 in enumerate(bodies):
            if i >= j: continue
            diff = abs(b1.lon - b2.lon)
            if diff > 180: diff = 360 - diff
            found_aspect = None
            for angle, name in major_aspects.items():
                if abs(diff - angle) <= orb_limit:
                    found_aspect = name
                    orb_used = abs(diff - angle)
                    break
            if found_aspect:
                asp_str = f"{found_aspect} {b2.name} ({orb_used:.1f}°)"
                b1.current_aspects.append(asp_str)
                b2.current_aspects.append(asp_str)

# ==========================================
# 7. MAIN EXECUTION
# ==========================================

def main():
    print("\n" + "="*80)
    print("  KABBALISTIC-QUANTUM ASTRO-MARKET ENGINE (BTC/USDC) - MULTI-TF")
    print("="*80)

    mkt = MarketEngine()
    
    # 1. Time & Observer
    obs, local_dt, city, live_time_utc = get_observer_data()
    print(f"Observer: {city} | Local Time: {local_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 2. Astro Calculations
    solar_system = [
        CelestialBody("Sun", "Sun"), CelestialBody("Moon", "Moon"),
        CelestialBody("Mercury", "Mercury"), CelestialBody("Venus", "Venus"),
        CelestialBody("Mars", "Mars"), CelestialBody("Jupiter", "Jupiter"),
        CelestialBody("Saturn", "Saturn"), CelestialBody("Uranus", "Uranus"),
        CelestialBody("Neptune", "Neptune"), CelestialBody("Pluto", "Pluto")
    ]
    for b in solar_system: b.compute(obs)
    calculate_aspects(solar_system)
    
    # 3. Print Astro Report
    print("\n" + "="*80)
    print("  GEOCENTRIC ASTRO-PHYSICS REPORT (REAL TIME)")
    print("="*80)
    print(f"{'BODY':<10} | {'SIGN':<12} | {'DEG':<6} | {'PLATONIC':<20} | {'FREQ(Hz)':<10} | {'MOMENTUM':<12} | {'ASPECTS'}")
    print("-" * 100)
    for b in solar_system:
        aspect_str = ", ".join(b.current_aspects[:2])
        if not aspect_str: aspect_str = "-"
        print(f"{b.name:<10} | {b.zodiac_sign:<12} | {b.deg_in_sign:<5.1f}° | {b.platonic:<20} | {b.frequency:<10} | {b.speed:.2f}°/d {'(Rx)' if b.is_retrograde else ''} | {aspect_str}")

    # 4. Define Timeframes & Process All
    timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '12h', '1d', '1w']
    tf_limits = {k: 1000 for k in timeframes}
    
    # Storage for results
    analysis_results = {}

    print("\n" + "="*80)
    print("  PROCESSING ALL TIMEFRAMES (Fetching, Training, Forecasting)...")
    print("="*80)
    
    for tf in timeframes:
        # Fetch
        tf_df_raw = mkt.fetch_data(symbol='BTCUSDC', interval=tf, limit=tf_limits[tf])
        if tf_df_raw.empty:
            continue
        
        # Techs
        tf_df = mkt.calculate_technicals(tf_df_raw)
        
        # Train & Predict AI
        preds = mkt.train_and_predict_tf(tf_df, tf)
        
        # Gann
        status, lvl, args, struct_status = mkt.detect_gann_cycle(tf_df)
        
        # Store
        analysis_results[tf] = {
            'df': tf_df,
            'raw': tf_df_raw,
            'preds': preds,
            'gann': {'status': status, 'level': lvl, 'args': args, 'struct': struct_status},
            'price': tf_df['close'].iloc[-1]
        }

    # 5. Print Gann Table (Existing Logic)
    print("\n" + "="*80)
    print("  MULTI-TIMEFRAME GANN CYCLE ANALYSIS")
    print("="*80)
    print(f"{'TF':<8} | {'STATUS':<25} | {'STRUCT STATUS':<20} | {'GANN LVL':<15} | {'ARGUMENT':<25} | {'PLANETARY RULER'}")
    print("-" * 120)
    
    for tf in timeframes:
        if tf not in analysis_results: continue
        
        res = analysis_results[tf]
        gann = res['gann']
        raw_df = res['raw']
        
        # Calc Gann Target logic for display
        last_high = res['df']['high'].max()
        last_low = res['df']['low'].min()
        swing_size = last_high - last_low
        
        if "ACCUMULATION" in gann['struct']:
            gann_target = last_low + swing_size
        elif "DISTRIBUTION" in gann['struct']:
            gann_target = last_high - swing_size
        else:
            gann_target = 0
            
        last_time = pd.to_datetime(raw_df['time'].iloc[-1], unit='ms').to_pydatetime()
        ruler = get_planetary_hour_ruler(last_time)
        
        print(f"{tf:<8} | {gann['status']:<25} | {gann['struct']:<20} | {gann_target:.0f}   | {gann['args']:<25} | {ruler}")

    # 6. NEW: Multi-TF AI Forecast Table
    print("\n" + "="*80)
    print("  MULTI-TIMEFRAME AI FORECAST (NEURAL NETWORK PREDICTIONS)")
    print("="*80)
    print(f"{'TF':<8} | {'PRICE':<10} | {'FAST (4c)':<15} | {'MED (24c)':<15} | {'LARGE (168c)':<15} | {'CONSENSUS'}")
    print("-" * 100)
    
    for tf in timeframes:
        if tf not in analysis_results: continue
        
        res = analysis_results[tf]
        price = res['price']
        preds = res['preds']
        
        if not preds:
            print(f"{tf:<8} | ${price:.2f} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15} | NO DATA")
            continue
            
        p_fast = preds['Fast']
        p_med = preds['Medium']
        p_large = preds['Large']
        
        def get_dir(curr, pred):
            return "UP" if pred > curr else "DOWN"
            
        d_fast = get_dir(price, p_fast)
        d_med = get_dir(price, p_med)
        d_large = get_dir(price, p_large)
        
        # Simple consensus
        votes = [d_fast, d_med, d_large]
        if votes.count("UP") >= 2: consensus = "BULLISH"
        elif votes.count("DOWN") >= 2: consensus = "BEARISH"
        else: consensus = "MIXED"
        
        print(f"{tf:<8} | ${price:<9.2f} | ${p_fast:<10.2f} {d_fast:<4} | ${p_med:<10.2f} {d_med:<4} | ${p_large:<10.2f} {d_large:<4} | {consensus}")

    # 7. Detailed 1H Analysis (Primary Reference)
    if '1h' in analysis_results:
        print("\n" + "="*80)
        print("  DETAILED 1H FORECAST & SYNTHESIS")
        print("="*80)
        
        df_1h = analysis_results['1h']['df']
        preds_1h = analysis_results['1h']['preds']
        curr_price = analysis_results['1h']['price']
        
        last_high = df_1h['high'].iloc[-1]
        last_low = df_1h['low'].iloc[-1]
        
        # Gann Calcs
        gann_high_45 = last_high + (last_high * 0.125) 
        gann_low_45 = last_low - (last_low * 0.125)
        gann_sr = math.sqrt(curr_price)
        
        print(f"Current Price:    ${curr_price:.2f}")
        print(f"Resistance 45 deg: ${gann_high_45:.2f}")
        print(f"Support 45 deg:    ${gann_low_45:.2f}")
        print(f"Gann SR:          {gann_sr:.2f}")
        
        # Fear Greed
        fg_score, fg_sentiment = mkt.get_fear_greed(df_1h)
        print(f"Fear & Greed:    {fg_score}/100 ({fg_sentiment})")
        
        # Dates
        if preds_1h:
            t_fast = local_dt + timedelta(hours=4)
            t_med = local_dt + timedelta(hours=24)
            t_large = local_dt + timedelta(days=7)
            
            print(f"\n1H AI TARGETS (LOCAL TIME):")
            print(f"Fast (4h):    ${preds_1h['Fast']:.2f} ({t_fast.strftime('%m-%d %H:%M')}) ({'UP' if preds_1h['Fast'] > curr_price else 'DOWN'})")
            print(f"Med (24h):    ${preds_1h['Medium']:.2f} ({t_med.strftime('%m-%d %H:%M')}) ({'UP' if preds_1h['Medium'] > curr_price else 'DOWN'})")
            print(f"Large (7d):   ${preds_1h['Large']:.2f} ({t_large.strftime('%m-%d %H:%M')}) ({'UP' if preds_1h['Large'] > curr_price else 'DOWN'})")

    # 8. Final Aggregated Signal
    print("\n" + "="*80)
    print("  FINAL AGGREGATED SIGNAL")
    print("="*80)
    
    bull_votes = 0
    bear_votes = 0
    total_votes = 0
    
    for tf, res in analysis_results.items():
        if not res['preds']: continue
        # Weight higher TFs more
        weight = 1
        if tf in ['4h', '12h', '1d', '1w']: weight = 2
        elif tf == '1h': weight = 1.5
        
        if res['preds']['Large'] > res['price']: 
            bull_votes += weight
        else: 
            bear_votes += weight
        total_votes += weight
        
    print(f"Weighted Bullish Strength: {bull_votes}")
    print(f"Weighted Bearish Strength: {bear_votes}")
    
    if bull_votes > bear_votes * 1.2:
        print(">>> GLOBAL SIGNAL: STRONG BUY (Majority of Timeframes Bullish)")
    elif bear_votes > bull_votes * 1.2:
        print(">>> GLOBAL SIGNAL: STRONG SELL (Majority of Timeframes Bearish)")
    elif bull_votes > bear_votes:
        print(">>> GLOBAL SIGNAL: BUY (Leaning Bullish)")
    elif bear_votes > bull_votes:
        print(">>> GLOBAL SIGNAL: SELL (Leaning Bearish)")
    else:
        print(">>> GLOBAL SIGNAL: NEUTRAL / CHOPPY (Indecision across Timeframes)")

if __name__ == "__main__":
    main()
