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
    """
    Manually converts Equatorial RA/Dec to Ecliptic Longitude.
    Mathematically accurate conversion to Tropical Zodiac.
    """
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

def get_duration_str(tf, candles):
    """Helper to convert TF + Candle count to human readable duration."""
    tf_minutes = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '12h': 720,
        '1d': 1440, '1w': 10080
    }
    
    if tf not in tf_minutes: return "Unknown"
    total_minutes = tf_minutes[tf] * candles
    
    days = total_minutes // 1440
    hours = (total_minutes % 1440) // 60
    minutes = total_minutes % 60
    
    parts = []
    if days > 0: parts.append(f"{days}d")
    if hours > 0: parts.append(f"{hours}h")
    if minutes > 0: parts.append(f"{minutes}m")
    
    return "".join(parts) if parts else "0m"

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
        self.models = {}
        self.scalers_x = {} 
        self.scalers_y = {}

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

    def get_gann_sr_levels(self, price):
        """Calculates Gann Square Support and Resistance."""
        sqrt_price = math.sqrt(price)
        res_1 = (sqrt_price + 1)**2
        sup_1 = (sqrt_price - 1)**2
        res_2 = (sqrt_price + 2)**2
        sup_2 = (sqrt_price - 2)**2
        return res_1, res_2, sup_1, sup_2

    def train_and_predict_tf(self, df, tf_name):
        if df.empty or 'close' not in df.columns:
            return None

        df_train = df.copy()
        df_train['Target_Fast'] = df_train['close'].shift(-4)
        df_train['Target_Med'] = df_train['close'].shift(-24)
        df_train['Target_Large'] = df_train['close'].shift(-168)
        
        features = ['RSI', 'sine', 'Returns', 'Lagged_Close']
        
        tf_models = {}
        tf_scalers_x = {}
        tf_scalers_y = {}
        predictions = {}

        targets_map = {'Fast': df_train['Target_Fast'], 'Medium': df_train['Target_Med'], 'Large': df_train['Target_Large']}
        
        for name, y in targets_map.items():
            temp_df = pd.concat([df_train[features], y], axis=1).dropna()
            
            if len(temp_df) < 100: continue
            
            X_train = temp_df[features].values
            y_train = temp_df[y.name].values
            
            scaler_x = MinMaxScaler()
            X_scaled = scaler_x.fit_transform(X_train)
            
            scaler_y = MinMaxScaler()
            y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            
            model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, early_stopping=True)
            model.fit(X_scaled, y_scaled)
            
            tf_models[name] = model
            tf_scalers_x[name] = scaler_x
            tf_scalers_y[name] = scaler_y

        self.scalers_x[tf_name] = tf_scalers_x
        self.scalers_y[tf_name] = tf_scalers_y

        if 'Fast' in tf_scalers_x:
            last_row = df.iloc[-1:][features].values
            X_input = tf_scalers_x['Fast'].transform(last_row)
            
            for name in ['Fast', 'Medium', 'Large']:
                if name in tf_models:
                    pred_scaled = tf_models[name].predict(X_input)
                    pred_real = tf_scalers_y[name].inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                    predictions[name] = pred_real
        
        return predictions

    def detect_gann_cycle(self, df):
        if len(df) < 50: return "NEUTRAL", 0, 0, "No Data", "NEUTRAL", "N/A"
        
        lookback = min(500, len(df))
        recent = df.tail(lookback)
        
        max_idx = recent['high'].idxmax()
        min_idx = recent['low'].idxmin()
        max_price = recent['high'].max()
        min_price = recent['low'].min()
        
        is_low_recent = min_idx > max_idx
        
        status = "NEUTRAL"
        arg_desc = ""
        structural_status = "NEUTRAL"
        extrema_status = "UNKNOWN"
        
        swing_size = max_price - min_price
        
        if is_low_recent:
            extrema_status = "LOW RECENT (Dip)"
            if df['close'].iloc[-1] > min_price + (swing_size * 0.5):
                status = "RECOVERY"
                structural_status = "ACCUMULATION (Rebound)"
            else:
                status = "BOTTOMING"
                structural_status = "ACCUMULATION (Dip)"
        else:
            extrema_status = "HIGH RECENT (Peak)"
            if df['close'].iloc[-1] < max_price - (swing_size * 0.5):
                status = "DISTRIBUTION"
                structural_status = "DISTRIBUTION (Falling)"
            else:
                status = "TOPPING"
                structural_status = "DISTRIBUTION (High)"

        return status, swing_size, max_price, min_price, structural_status, extrema_status

    def get_fear_greed(self, df):
        if df.empty: return 50, "NEUTRAL"
        try:
            rsi = df['RSI'].iloc[-1]
            volatility = df['Returns'].std()
            score = 50
            if rsi > 70: score += 30 
            if rsi < 30: score -= 30 
            if volatility > 0.015: score -= 10 
            score = max(0, min(100, score)) 
            
            if score > 75: sentiment = "EXTREME GREED"
            elif score > 55: sentiment = "GREED"
            elif score > 45: sentiment = "NEUTRAL"
            elif score > 25: sentiment = "FEAR"
            else: sentiment = "EXTREME FEAR"
            return score, sentiment
        except:
            return 50, "ERROR"

# ==========================================
# 5. GEOLOCATION & TIME SYNC
# ==========================================

def get_utc_now():
    # CHANGE: Return specific target date to match user request
    # To return to real time, use: return datetime.now(timezone.utc).replace(tzinfo=None)
    return datetime(2026, 1, 30, 2, 27, 57, tzinfo=timezone.utc).replace(tzinfo=None)

def get_observer_data(target_time):
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
    
    # IMPORTANT: Use the passed target_time, not 'now'
    live_time_utc = target_time
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
# 6. ASTRO & GANN LOGIC (FIXED)
# ==========================================

def calculate_aspects(bodies):
    major_aspects = {0: "Conj", 60: "Sextile", 90: "Square", 120: "Trine", 180: "Opp"}
    orb_limit = 8 
    
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            
            b1 = bodies[i]
            b2 = bodies[j]
            
            diff = abs(b1.lon - b2.lon)
            if diff > 180: diff = 360 - diff
            
            found_aspect = None
            for angle, name in major_aspects.items():
                if abs(diff - angle) <= orb_limit:
                    found_aspect = name
                    orb_used = abs(diff - angle)
                    break
            
            if found_aspect:
                asp_str_for_b1 = f"{found_aspect} {b2.name} ({orb_used:.1f}°)"
                asp_str_for_b2 = f"{found_aspect} {b1.name} ({orb_used:.1f}°)"
                
                b1.current_aspects.append(asp_str_for_b1)
                b2.current_aspects.append(asp_str_for_b2)

# ==========================================
# 7. MAIN EXECUTION
# ==========================================

def main():
    print("\n" + "="*110)
    print("  KABBALISTIC-QUANTUM ASTRO-MARKET ENGINE (BTC/USDC) - FIXED DATE CORRECTION")
    print("="*110)

    mkt = MarketEngine()
    
    # 1. Time & Observer
    # CHANGE: Define target date explicitly for accuracy verification
    target_time = get_utc_now() 
    obs, local_dt, city, live_time_utc = get_observer_data(target_time)
    
    print(f"Observer: {city} | Simulation Time (UTC): {live_time_utc.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 2. Astro (Real Time Calculation)
    solar_system = [
        CelestialBody("Sun", "Sun"), CelestialBody("Moon", "Moon"),
        CelestialBody("Mercury", "Mercury"), CelestialBody("Venus", "Venus"),
        CelestialBody("Mars", "Mars"), CelestialBody("Jupiter", "Jupiter"),
        CelestialBody("Saturn", "Saturn"), CelestialBody("Uranus", "Uranus"),
        CelestialBody("Neptune", "Neptune"), CelestialBody("Pluto", "Pluto")
    ]
    
    for b in solar_system: b.compute(obs)
    calculate_aspects(solar_system)
    
    print("\n" + "="*110)
    print("  GEOCENTRIC ASTRO-PHYSICS REPORT (REAL TIME EPHEMERIS)")
    print("="*110)
    print(f"{'BODY':<10} | {'SIGN':<12} | {'DEG':<6} | {'PLATONIC':<20} | {'FREQ(Hz)':<10} | {'MOMENTUM':<12} | {'ASPECTS'}")
    print("-" * 110)
    
    for b in solar_system:
        aspect_str = ", ".join(b.current_aspects[:2]) 
        if not aspect_str: aspect_str = "-"
        print(f"{b.name:<10} | {b.zodiac_sign:<12} | {b.deg_in_sign:<5.1f}° | {b.platonic:<20} | {b.frequency:<10} | {b.speed:.2f}°/d {'(Rx)' if b.is_retrograde else ''} | {aspect_str}")

    # 3. Timeframes
    timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '12h', '1d', '1w']
    tf_limits = {
        '1m': 500, '3m': 500, '5m': 500, '15m': 500, '30m': 500,
        '1h': 1000, '2h': 1000, '4h': 1000, '12h': 1000,
        '1d': 1000, '1w': 1000 
    }
    
    analysis_results = {}

    print("\n" + "="*110)
    print("  PROCESSING ALL TIMEFRAMES...")
    print("="*110)
    
    for tf in timeframes:
        tf_df_raw = mkt.fetch_data(symbol='BTCUSDC', interval=tf, limit=tf_limits[tf])
        if tf_df_raw.empty: continue
        
        tf_df = mkt.calculate_technicals(tf_df_raw)
        preds = mkt.train_and_predict_tf(tf_df, tf)
        status, swing, max_p, min_p, struct_status, extrema_status = mkt.detect_gann_cycle(tf_df)
        fg_score, fg_sent = mkt.get_fear_greed(tf_df)
        
        current_price = tf_df['close'].iloc[-1]
        g_res1, g_res2, g_sup1, g_sup2 = mkt.get_gann_sr_levels(current_price)
        
        analysis_results[tf] = {
            'preds': preds,
            'price': current_price,
            'gann': {'status': status, 'struct': struct_status, 'extrema': extrema_status, 'swing': swing},
            'fg': {'score': fg_score, 'sent': fg_sent},
            'sr': {'res1': g_res1, 'res2': g_res2, 'sup1': g_sup1, 'sup2': g_sup2}
        }

    # 4. Structure Table
    print("\n" + "="*110)
    print("  MULTI-TF MARKET STRUCTURE (GANN SR, EXTREMA, SENTIMENT)")
    print("="*110)
    print(f"{'TF':<6} | {'PRICE':<10} | {'EXTREMA':<20} | {'STRUCT':<20} | {'GANN RES 1':<12} | {'GANN SUP 1':<12} | {'F&G':<15}")
    print("-" * 110)
    
    for tf in timeframes:
        if tf not in analysis_results: continue
        res = analysis_results[tf]
        print(f"{tf:<6} | ${res['price']:<9.2f} | {res['gann']['extrema']:<20} | {res['gann']['struct']:<20} | ${res['sr']['res1']:<10.2f} | ${res['sr']['sup1']:<10.2f} | {res['fg']['score']:<3} {res['fg']['sent']:<10}")

    # 5. AI Forecast Table
    print("\n" + "="*110)
    print("  MULTI-TF AI FORECAST (SCALED TARGETS)")
    print("="*110)
    print(f"{'TF':<6} | {'FAST (4c)':<30} | {'MED (24c)':<30} | {'LARGE (168c)':<30} | {'CONSENSUS'}")
    print("-" * 125)
    
    for tf in timeframes:
        if tf not in analysis_results: continue
        
        res = analysis_results[tf]
        preds = res['preds']
        
        if not preds:
            print(f"{tf:<6} | {'N/A':<30} | {'N/A':<30} | {'N/A':<30} | NO MODEL")
            continue
            
        dur_fast = get_duration_str(tf, 4)
        dur_med = get_duration_str(tf, 24)
        dur_large = get_duration_str(tf, 168)
        
        p_fast = preds['Fast']
        p_med = preds['Medium']
        p_large = preds['Large']
        
        def get_dir(curr, pred):
            return "▲UP" if pred > curr else "▼DOWN"
            
        dir_fast = get_dir(res['price'], p_fast)
        dir_med = get_dir(res['price'], p_med)
        dir_large = get_dir(res['price'], p_large)
        
        votes = [dir_fast, dir_med, dir_large]
        bull_votes = sum(1 for x in votes if "UP" in x)
        if bull_votes >= 2: consensus = "BULLISH"
        elif bull_votes == 1: consensus = "SLIGHT BULL"
        else: consensus = "BEARISH"

        print(f"{tf:<6} | ${p_fast:<8.2f} ({dur_fast:<6}) {dir_fast:<7} | ${p_med:<8.2f} ({dur_med:<6}) {dir_med:<7} | ${p_large:<8.2f} ({dur_large:<6}) {dir_large:<7} | {consensus}")

    # 6. Global Overall Logic
    print("\n" + "="*110)
    print("  GLOBAL OVERALL SYNTHESIS (CYCLE PROJECTION)")
    print("="*110)
    
    major_tfs = ['1h', '4h', '12h', '1d']
    acc_votes = 0
    dist_votes = 0
    total_support = 0
    total_resistance = 0
    total_swing = 0
    tf_count = 0
    
    ref_price = analysis_results.get('1h', {}).get('price', 0)
    
    for tf in major_tfs:
        if tf not in analysis_results: continue
        
        struct = analysis_results[tf]['gann']['struct']
        if "ACCUMULATION" in struct:
            acc_votes += 1
        else:
            dist_votes += 1
            
        total_support += analysis_results[tf]['sr']['sup1']
        total_resistance += analysis_results[tf]['sr']['res1']
        total_swing += analysis_results[tf]['gann']['swing']
        tf_count += 1
    
    global_status = "NEUTRAL"
    entry_name = ""
    exit_name = ""
    entry_target = 0.0
    exit_target = 0.0
    
    if tf_count > 0:
        avg_support = total_support / tf_count
        avg_resistance = total_resistance / tf_count
        avg_swing = total_swing / tf_count
        
        if acc_votes > dist_votes:
            global_status = "BULLISH ACCUMULATION (DIP)"
            entry_name = "DIP Reversal Entry Target"
            exit_name = "TOP Reversal Exit Forecast"
            entry_target = avg_support
            exit_target = entry_target + avg_swing
        elif dist_votes > acc_votes:
            global_status = "BEARISH DISTRIBUTION (TOP)"
            entry_name = "TOP Reversal Entry Target (Short)"
            exit_name = "BOTTOM Reversal Exit Forecast (Cover)"
            entry_target = avg_resistance
            exit_target = entry_target - avg_swing
        else:
            global_status = "NEUTRAL / TRANSITION"
            entry_name = "NEAREST REVERSAL TARGET"
            exit_name = "EXTENSION TARGET"
            dist_to_sup = abs(ref_price - avg_support)
            dist_to_res = abs(ref_price - avg_resistance)
            entry_target = avg_support if dist_to_sup < dist_to_res else avg_resistance
            exit_target = entry_target + avg_swing 
            
    # Reversal Logic
    total_fg = sum([res['fg']['score'] for res in analysis_results.values()])
    count_fg = len(analysis_results)
    avg_fg = total_fg / count_fg if count_fg > 0 else 50
    
    reversal_status = "NO REVERSAL SIGNAL"
    if avg_fg < 25 and "ACCUMULATION" in global_status:
        reversal_status = "REVERSAL INCOMING: DIP BOUNCE LIKELY (Extreme Fear + Structure)"
    elif avg_fg > 75 and "DISTRIBUTION" in global_status:
        reversal_status = "REVERSAL INCOMING: DUMP IMMINENT (Extreme Greed + Structure)"
    elif "ACCUMULATION" in global_status and avg_fg < 40:
        reversal_status = "REVERSAL INCOMING: BULLISH DIVERGENCE"
    
    # Print Global Summary
    print(f"Current Market State:   {global_status}")
    print(f"Reference Price:        ${ref_price:.2f}")
    print("-" * 110)
    print(f"{entry_name}: ${entry_target:.2f}")
    print(f"{exit_name}: ${exit_target:.2f}")
    print("-" * 110)
    print(f"Overall Sentiment:      {avg_fg:.1f}/100")
    print(f"Reversal Status:        {reversal_status}")

if __name__ == "__main__":
    main()
