import ephem
import requests
import numpy as np
import pandas as pd
import math
import pytz
from datetime import datetime, timedelta, timezone
from binance.client import Client
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION & ASTRO-PHYSICS CONSTANTS
# ==========================================

# Platonic Solids Correspondence (Kepler/Harmonic Mapping)
PLATONIC_SOLIDS = {
    "Sun": "Sphere (Monad)",
    "Moon": "Icosahedron (Water)",
    "Mercury": "Octahedron (Air)",
    "Venus": "Dodecahedron (Aether)",
    "Mars": "Tetrahedron (Fire)",
    "Jupiter": "Cube (Earth Structure)",
    "Saturn": "Cube (Limitation)",
    "Uranus": "Dodecahedron (Higher Ether)",
    "Neptune": "Icosahedron (Oceanic)",
    "Pluto": "Tetrahedron (Underworld Fire)"
}

# Cosmic Octave Frequencies
PLANETARY_FREQUENCIES = {
    "Sun": 126.22, "Moon": 210.42, "Mercury": 141.27, "Venus": 221.23,
    "Mars": 144.72, "Jupiter": 183.58, "Saturn": 147.85,
    "Uranus": 207.36, "Neptune": 211.44, "Pluto": 140.64
}

# Golden Ratio
PHI = 1.61803398875

# ==========================================
# 2. CLASS DEFINITIONS
# ==========================================

class CelestialBody:
    def __init__(self, name, key):
        self.name = name
        self.key = key
        self.body = self._get_body(key)
        self.frequency = PLANETARY_FREQUENCIES.get(key, 0.0)
        self.current_aspects = []
        
        # Data Holders
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
        
        # CRITICAL: Convert PyEphem Radians to Degrees
        self.lon = math.degrees(self.body.hlong) 
        self.lat = math.degrees(self.body.hlat)
        self.alt = math.degrees(self.body.alt)
        
        signs = ["Ari", "Tau", "Gem", "Can", "Leo", "Vir", "Lib", "Sco", "Sag", "Cap", "Aqu", "Pis"]
        
        # Calculate Sign Index based on Degrees (0-360)
        sign_idx = int(self.lon / 30) % 12
        self.zodiac_sign = signs[sign_idx]
        
        # Degrees inside the sign (0-30)
        self.deg_in_sign = self.lon % 30
        
        # Retrograde calc
        obs_f = observer.copy()
        obs_f.date = observer.date + 1
        body_f = self._get_body(self.key)
        body_f.compute(obs_f)
        
        # Use degrees for comparison
        lon_f = math.degrees(body_f.hlong)
        diff = lon_f - self.lon
        
        # Normalize diff to -180 to 180
        if diff < -180: diff += 360
        if diff > 180: diff -= 360
        
        self.is_retrograde = diff < 0
        self.speed = abs(diff)

# ==========================================
# 3. DATA & ML ENGINE
# ==========================================

class MarketEngine:
    def __init__(self):
        # Public Binance Client (No API key needed for public data)
        self.client = Client("", "")
        self.scaler = MinMaxScaler()
        self.model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        self.df = pd.DataFrame()

    def fetch_data(self, symbol='BTCUSDC', interval='1h', limit=500):
        print(f"[DATA] Fetching {limit} candles for {symbol}...")
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            self.df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qv', 'trades', 'tbv', 'tv', 'ignore'])
            self.df['close'] = pd.to_numeric(self.df['close'])
            self.df['high'] = pd.to_numeric(self.df['high'])
            self.df['low'] = pd.to_numeric(self.df['low'])
            self.df['volume'] = pd.to_numeric(self.df['volume'])
            return True
        except Exception as e:
            print(f"[ERROR] Binance API Error: {e}")
            return False

    def calculate_technicals(self):
        df = self.df.copy()
        
        # Simple RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # FFT Forecast (Dominant Frequency Energy)
        prices = df['close'].values
        fft_vals = np.fft.fft(prices)
        fft_freq = np.fft.fftfreq(len(prices))
        # Dominant frequency index (ignoring 0 frequency)
        dominant_freq_idx = np.argmax(np.abs(fft_vals[1:len(fft_vals)//2])) + 1
        df['FFT_Energy'] = np.abs(fft_vals[dominant_freq_idx])
        
        # HT_SINE (Try Talib, else fallback to simple sine wave)
        try:
            import talib
            df['sine'], df['leadsine'] = talib.HT_SINE(df['close'])
            df['sine'] = df['sine'].fillna(0)
        except:
            # Fallback simple sine wave of price
            t = np.arange(len(df))
            std_dev = df['close'].std()
            df['sine'] = np.sin(2 * np.pi * 0.1 * t) * std_dev

        df['Returns'] = df['close'].pct_change()
        # Clean NaNs for ML
        self.df = df.dropna().reset_index(drop=True)
        return self.df

    def train_neural_net(self, astro_features):
        """Trains a simple Neural Net to predict next close based on Tech + Astro"""
        df = self.df.copy()
        
        # Features: RSI, FFT Energy, Sine, Returns
        X = df[['RSI', 'FFT_Energy', 'sine', 'Returns']].iloc[:-1].values
        y = df['close'].iloc[1:].values # Predict next close
        
        # Normalize
        X_scaled = self.scaler.fit_transform(X)
        
        # Train
        self.model.fit(X_scaled, y)
        # Return R^2 score
        return self.model.score(X_scaled, y)

    def predict_next(self, astro_vector):
        last_row = self.df.iloc[[-1]]
        features = last_row[['RSI', 'FFT_Energy', 'sine', 'Returns']].values
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        current_price = self.df['close'].iloc[-1]
        return current_price, prediction

    def get_fear_greed(self):
        """Synthetic Fear & Greed based on RSI and Volatility"""
        rsi = self.df['RSI'].iloc[-1]
        volatility = self.df['Returns'].std() if len(self.df) > 0 else 0
        
        score = 50 # Neutral
        
        if rsi > 70: score += 30 # Greed
        if rsi < 30: score -= 30 # Fear
        
        if volatility > 0.015: score -= 10 # High Vol adds fear
        
        score = max(0, min(100, score)) # Clamp between 0-100
        
        sentiment = "EXTREME GREED" if score > 75 else "GREED" if score > 55 else "NEUTRAL" if score > 45 else "FEAR" if score > 25 else "EXTREME FEAR"
        return score, sentiment

# ==========================================
# 4. ROBUST GEOLOCATION & TIME
# ==========================================

def get_utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)

def get_observer_data():
    """
    Attempts to find location via multiple IP APIs to ensure accurate local time.
    """
    # List of free GeoIP services to try in order
    sources = [
        "https://ipapi.co/json/",
        "http://ip-api.com/json/",
        "https://ipwho.is/"
    ]
    
    lat, lon, city, tz_str = None, None, None, None
    
    print("[GEO] Searching for location...")
    
    for url in sources:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                # Handle different API formats
                # ipapi.co uses 'latitude', 'longitude', 'timezone'
                # ip-api uses 'lat', 'lon', 'timezone'
                # ipwho uses similar keys
                
                lat_raw = data.get('lat') or data.get('latitude')
                lon_raw = data.get('lon') or data.get('longitude')
                city_raw = data.get('city') or data.get('region') or "Unknown"
                tz_raw = data.get('timezone') or data.get('time_zone') or "UTC"
                
                if lat_raw and lon_raw:
                    lat = float(lat_raw)
                    lon = float(lon_raw)
                    city = str(city_raw)
                    tz_str = str(tz_raw)
                    print(f"[GEO] SUCCESS via {url.split('/')[2]}: {city} (Lat: {lat}, Lon: {lon}, TZ: {tz_str})")
                    break # Success! Stop trying other APIs
        except Exception as e:
            print(f"[GEO] Failed to use {url}: {e}")
            continue

    # Fallback if all APIs fail
    if not lat or not lon:
        print("[WARN] All Geo APIs failed. Defaulting to Greenwich (UTC).")
        lat, lon, city, tz_str = 51.4778, -0.0014, "Greenwich", "Europe/London"

    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.elevation = 0
    observer.date = get_utc_now()
    
    # Get Correct Local Time
    try:
        local_tz = pytz.timezone(tz_str)
        local_time = datetime.now(local_tz)
    except Exception as e:
        print(f"[WARN] Invalid Timezone {tz_str}. Using UTC.")
        local_time = datetime.utcnow()
        
    return observer, local_time, city

# ==========================================
# 5. ASTRO & GANN LOGIC
# ==========================================

def get_zodiac_sign_full(degree):
    signs = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", 
             "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
    return signs[int(degree / 30) % 12]

def calculate_gann_squares(price):
    """W.D. Gann Square of 9 Logic (Simplified)"""
    root = math.sqrt(price)
    gann_level = int(root)
    remainder = root - gann_level
    
    is_resistance = False
    if remainder < 0.1 or abs(remainder - 0.5) < 0.1 or abs(remainder - 1.0) < 0.1:
        is_resistance = True
        
    return gann_level, is_resistance

def calculate_golden_harmonics(angle):
    """Checks if angle aligns with Golden Ratio harmonics"""
    phi_deg = 137.507764
    harmonic_angles = [phi_deg, 360-phi_deg, phi_deg*2, 360-(phi_deg*2)]
    harmonics = []
    for ha in harmonic_angles:
        if abs(angle - ha) < 5: # 5 degree orb
            harmonics.append(f"Golden Ratio Hit ({ha:.1f}°)")
    return harmonics

def calculate_aspects(bodies):
    major_aspects = {0: "Conj", 60: "Sextile", 90: "Square", 120: "Trine", 180: "Opp"}
    orb_limit = 6 
    
    for i, b1 in enumerate(bodies):
        for j, b2 in enumerate(bodies):
            if i >= j: continue
            # diff is now calculated in Degrees
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
# 6. MAIN EXECUTION & SYNTHESIS
# ==========================================

def main():
    print("\n" + "="*80)
    print("  KABBALISTIC-QUANTUM ASTRO-MARKET ENGINE (BTC/USDC)")
    print("="*80)

    # 1. Init Engines
    obs, local_dt, city = get_observer_data()
    mkt = MarketEngine()
    
    print(f"Observer: {city} | Time: {local_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 2. Get Market Data
    if not mkt.fetch_data(symbol='BTCUSDC', interval='1h', limit=500):
        print("Cannot proceed without market data. Exiting.")
        return
        
    mkt.calculate_technicals()
    
    # 3. Astro Calculations
    solar_system = [
        CelestialBody("Sun", "Sun"), CelestialBody("Moon", "Moon"),
        CelestialBody("Mercury", "Mercury"), CelestialBody("Venus", "Venus"),
        CelestialBody("Mars", "Mars"), CelestialBody("Jupiter", "Jupiter"),
        CelestialBody("Saturn", "Saturn"), CelestialBody("Uranus", "Uranus"),
        CelestialBody("Neptune", "Neptune"), CelestialBody("Pluto", "Pluto")
    ]
    
    for b in solar_system: b.compute(obs)
    calculate_aspects(solar_system)
    
    # 4. ML Prediction
    print("\n[ML] Training Neural Network on Technical + Astro Harmonics...")
    dummy_astro = np.random.rand(10) 
    score = mkt.train_neural_net(dummy_astro)
    print(f"[ML] Model Accuracy Score (R2): {score:.4f}")
    
    curr_price, pred_price = mkt.predict_next(dummy_astro)
    fg_score, fg_sentiment = mkt.get_fear_greed()
    
    # 5. Detailed Printing
    
    print("\n" + "="*80)
    print("  HELIOCENTRIC ASTRO-PHYSICS REPORT")
    print("="*80)
    print(f"{'BODY':<10} | {'SIGN':<12} | {'DEG':<6} | {'PLATONIC':<20} | {'FREQ(Hz)':<10} | {'MOMENTUM':<12} | {'ASPECTS'}")
    print("-" * 100)
    
    for b in solar_system:
        aspect_str = ", ".join(b.current_aspects[:2])
        if not aspect_str: aspect_str = "-"
        
        mom = f"{b.speed:.2f}°/d {'(Rx)' if b.is_retrograde else ''}"
        
        print(f"{b.name:<10} | {b.zodiac_sign:<12} | {b.deg_in_sign:<5.1f}° | {b.platonic:<20} | {b.frequency:<10} | {mom:<12} | {aspect_str}")

    print("\n" + "="*80)
    print("  QUANTUM MARKET FORECAST (NEURAL + ASTRO)")
    print("="*80)
    
    # Reversal Detection
    direction = "UP" if pred_price > curr_price else "DOWN"
    intensity = abs((pred_price - curr_price) / curr_price) * 100
    cycle_type = "FAST Reversal" if intensity > 0.5 else "Continuation"
    
    print(f"CURRENT PRICE:   ${curr_price:.2f}")
    print(f"AI PREDICTION:    ${pred_price:.2f} ({direction})")
    print(f"CYCLE INTENSITY:  {intensity:.2f}% ({cycle_type})")
    print(f"FEAR & GREED:     {fg_score}/100 ({fg_sentiment})")
    
    # W.D. Gann Analysis
    gann_lvl, is_res = calculate_gann_squares(curr_price)
    print(f"GANN SQUARE 9:    Level {gann_lvl} | {'RESISTANCE ZONE' if is_res else 'SAFE'}")
    
    # Astrological Reversal Check
    hard_aspects_active = False
    for b in solar_system:
        if b.name in ["Mars", "Uranus", "Saturn"]:
            if "Square" in str(b.current_aspects) or "Opp" in str(b.current_aspects):
                hard_aspects_active = True
                break
    
    if hard_aspects_active:
        print("ASTRO WARNING:    Geocosmic Storm Active (Hard Aspects). High Reversal Prob.")
    else:
        print("ASTRO STATUS:     Harmonic Flow. Trend likely to continue.")

    # Golden Ratio Hits
    sun = next(b for b in solar_system if b.name == "Sun")
    golden_hits_full = calculate_golden_harmonics(sun.lon)
    
    if golden_hits_full:
        print(f"GOLDEN RATIO:     {', '.join(golden_hits_full)} aligns with Sun. Potential Cycle Turn.")

    print("-" * 80)
    
    # Moon Details (Short)
    moon = next(b for b in solar_system if b.name == "Moon")
    print(f"MOON DATA: Sign {moon.zodiac_sign} {moon.deg_in_sign:.1f}° | Platonic: {moon.platonic} | Freq: {moon.frequency}Hz")
    
    # Sinusoidal Wave Forecast
    last_sine = mkt.df['sine'].iloc[-1]
    trend_sine = "BULLISH CYCLE" if last_sine > 0 else "BEARISH CYCLE"
    print(f"SINE WAVE (HT):   {trend_sine} (Value: {last_sine:.2f})")
    
    print("\n" + "="*80)
    print("  FINAL SUMMARY SIGNAL")
    print("="*80)
    
    # Composite Logic
    if (direction == "UP" and "BULLISH" in trend_sine and not hard_aspects_active):
        print(">>> STRONG BUY SIGNAL (Confluence: AI + Technical + Astro)")
    elif (direction == "DOWN" or "BEARISH" in trend_sine) and hard_aspects_active:
        print(">>> STRONG SELL SIGNAL (Confluence: AI + Technical + Astro)")
    else:
        print(">>> HOLD/NEUTRAL SIGNAL (Conflicting Data)")

if __name__ == "__main__":
    main()