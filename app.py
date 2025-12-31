import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import sqlalchemy
from sqlalchemy import create_engine, text
import bcrypt
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="FS Pro Analytics", layout="wide", page_icon="⛳")

# --- DATABASE SETUP ---
try:
    if "DATABASE_URL" in st.secrets:
        DB_URL = st.secrets["DATABASE_URL"]
    else:
        DB_URL = os.getenv("DATABASE_URL")
except FileNotFoundError:
    DB_URL = os.getenv("DATABASE_URL")

if not DB_URL:
    st.error("⚠️ Database URL not found. Please set DATABASE_URL in Railway Variables.")
    st.stop()

try:
    engine = create_engine(DB_URL)
except Exception as e:
    st.error(f"Failed to connect to Database: {e}")
    st.stop()

# --- CSS STYLING (FIXED) ---
st.markdown("""
<style>
    /* 1. FORCE DARK MODE TEXT */
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown { color: #FAFAFA !important; }
    
    /* 2. FIX BUTTONS (Invisible Text Fix) */
    .stButton > button {
        background-color: #262730;
        color: #FAFAFA !important;
        border: 1px solid #4DD0E1;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #4DD0E1;
        color: #000 !important;
        border-color: #4DD0E1;
    }
    
    /* 3. FIX INPUT FIELDS */
    [data-testid="stTextInput"] input {
        color: #FAFAFA !important;
        background-color: #262730 !important; 
    }
    [data-testid="stNumberInput"] input {
        color: #FAFAFA !important;
        background-color: #262730 !important;
    }
    
    /* 4. SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #12151d;
        border-right: 1px solid #333;
    }
    
    /* 5. METRIC CARDS */
    .metric-card {
        background: linear-gradient(145deg, #1E222B, #262730);
        border: 1px solid #444;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    
    /* 6. HERO CARDS */
    .hero-card {
        background: linear-gradient(145deg, #1E222B, #262730);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid #444;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    
    /* 7. PROGRESS BARS */
    .eff-container { background-color: #111; border-radius: 10px; padding: 4px; margin-top: 10px; border: 1px solid #333; }
    .eff-bar-fill { height: 12px; background: linear-gradient(90deg, #FF4081, #00E5FF); border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# --- AUTH FUNCTIONS ---

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def login_user(username, password):
    try:
        with engine.connect() as conn:
            query = text("SELECT id, password_hash FROM users WHERE username = :u")
            result = conn.execute(query, {"u": username}).fetchone()
            if result and check_password(password, result[1]):
                return result[0] 
    except Exception as e:
        st.error(f"Login Error: {e}")
    return None

def register_user(username, password):
    try:
        hashed = hash_password(password)
        with engine.connect() as conn:
            query = text("INSERT INTO users (username, password_hash) VALUES (:u, :p)")
            conn.execute(query, {"u": username, "p": hashed})
            conn.commit()
            
            # Initialize default prefs for new user
            uid_query = text("SELECT id FROM users WHERE username = :u")
            uid = conn.execute(uid_query, {"u": username}).fetchone()[0]
            
            pref_query = text("INSERT INTO user_prefs (user_id) VALUES (:uid)")
            conn.execute(pref_query, {"uid": uid})
            conn.commit()
            
        return True
    except sqlalchemy.exc.IntegrityError:
        st.warning("Username already taken.")
        return False
    except Exception as e:
        st.error(f"Registration Error: {e}")
        return False

# --- DATA FUNCTIONS ---

def load_user_data(user_id):
    # 1. Load Shots
    try:
        query = text(f"SELECT * FROM shots WHERE user_id = :uid")
        df = pd.read_sql(query, engine, params={"uid": user_id})
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
    except:
        df = pd.DataFrame()

    # 2. Load Bag Lofts
    try:
        query = text("SELECT club_name, loft FROM bag_settings WHERE user_id = :uid")
        bag_df = pd.read_sql(query, engine, params={"uid": user_id})
        if not bag_df.empty:
            bag = dict(zip(bag_df.club_name, bag_df.loft))
        else:
            bag = DEFAULT_LOFTS.copy()
    except:
        bag = DEFAULT_LOFTS.copy()
    
    # 3. Load User Prefs (Handicap, etc)
    prefs = {'handicap': 15, 'temp': 75, 'altitude': 500, 'ball': 'Premium (100%)'}
    try:
        query = text("SELECT handicap, temp, altitude, ball FROM user_prefs WHERE user_id = :uid")
        with engine.connect() as conn:
            result = conn.execute(query, {"uid": user_id}).fetchone()
            if result:
                prefs = {
                    'handicap': result[0], 
                    'temp': result[1], 
                    'altitude': result[2], 
                    'ball': result[3]
                }
    except Exception as e:
        pass # Keep defaults if fails
        
    return df, bag, prefs

def save_user_pref(user_id, col, val):
    try:
        query = text(f"""
            INSERT INTO user_prefs (user_id, {col}) VALUES (:uid, :val)
            ON CONFLICT (user_id) DO UPDATE SET {col} = :val
        """)
        with engine.connect() as conn:
            conn.execute(query, {"uid": user_id, "val": val})
            conn.commit()
    except Exception as e:
        st.error(f"Could not save setting: {e}")

def save_bag_loft(user_id, club, loft):
    with engine.connect() as conn:
        sql = text("""
            INSERT INTO bag_settings (user_id, club_name, loft) 
            VALUES (:uid, :c, :l)
            ON CONFLICT (user_id, club_name) 
            DO UPDATE SET loft = EXCLUDED.loft;
        """)
        conn.execute(sql, {"uid": user_id, "c": club, "l": loft})
        conn.commit()

# --- CONSTANTS ---
DEFAULT_LOFTS = {
    'Driver': 10.5, '3 Wood': 15.0, '5 Wood': 18.0, 'Hybrid': 21.0, 
    '3 Iron': 21.0, '4 Iron': 24.0, '5 Iron': 27.0, '6 Iron': 30.0, 
    '7 Iron': 34.0, '8 Iron': 38.0, '9 Iron': 42.0, 'PW': 46.0, 
    'GW': 50.0, 'SW': 54.0, 'LW': 58.0
}
CLUB_SORT_ORDER = ['Driver', '3 Wood', '5 Wood', '7 Wood', 'Hybrid', '2 Iron', '3 Iron', '4 Iron', '5 Iron', '6 Iron', '7 Iron', '8 Iron', '9 Iron', 'PW', 'GW', 'SW', 'LW']

# --- SESSION STATE INITIALIZATION ---
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'master_df' not in st.session_state:
    st.session_state['master_df'] = pd.DataFrame()
if 'my_bag' not in st.session_state:
    st.session_state['my_bag'] = DEFAULT_LOFTS.copy()
if 'my_prefs' not in st.session_state:
    st.session_state['my_prefs'] = {'handicap': 15, 'temp': 75, 'altitude': 500, 'ball': 'Premium (100%)'}

# --- CLEANING HELPER ---
@st.cache_data
def clean_mevo_data(df, filename, selected_date):
    df_clean = df[df['Shot'].astype(str).str.isdigit()].copy()
    df_clean['Session'] = filename.replace('.csv', '')
    if 'Date' in df_clean.columns:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce').fillna(pd.to_datetime(selected_date))
    else:
        df_clean['Date'] = pd.to_datetime(selected_date)
    
    if 'Lateral (yds)' in df_clean.columns:
        lat_str = df_clean['Lateral (yds)'].astype(str).str.upper()
        is_left = lat_str.str.contains('L')
        nums = lat_str.str.extract(r'(\d+\.?\d*)')[0].astype(float).fillna(0.0)
        df_clean['Lateral_Clean'] = np.where(is_left, -nums, nums)
    else:
        df_clean['Lateral_Clean'] = 0.0

    cols = ['Carry (yds)', 'Total (yds)', 'Ball (mph)', 'Club (mph)', 'Smash', 'Spin (rpm)', 'Height (ft)', 'AOA (°)', 'Launch V (°)']
    for c in cols:
        if c in df_clean.columns: df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')
    
    if 'Altitude (ft)' in df_clean.columns:
        df_clean['Altitude (ft)'] = pd.to_numeric(df_clean['Altitude (ft)'].astype(str).str.replace(',','').str.replace(' ft',''), errors='coerce').fillna(0.0)
    else:
        df_clean['Altitude (ft)'] = 0.0

    df_clean['SL_Carry'] = df_clean['Carry (yds)'] / (1 + (df_clean['Altitude (ft)'] / 1000.0 * 0.011))
    df_clean['SL_Total'] = df_clean['Total (yds)'] / (1 + (df_clean['Altitude (ft)'] / 1000.0 * 0.011))
    return df_clean

# --- HELPER CALCS ---
def calculate_optimal_carry(club_speed, loft, benchmark="Scratch"):
    if benchmark == "Tour Pro":
        x_points = [9, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        y_points = [2.90, 2.75, 2.60, 2.50, 2.40, 2.30, 2.20, 2.10, 2.00, 1.85, 1.70]
    else: # Scratch
        x_points = [9, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        y_points = [2.75, 2.65, 2.55, 2.48, 2.40, 2.32, 2.20, 2.05, 1.85, 1.65, 1.50]
    safe_loft = max(5, min(loft, 64))
    efficiency_factor = np.interp(safe_loft, x_points, y_points)
    return club_speed * efficiency_factor

def get_smart_max(series, df_subset):
    valid = df_subset.loc[series.index]
    clean = valid[(valid['Smash'] <= 1.58) & (valid['Smash'] >= 1.0) & (valid['Spin (rpm)'] > 500) & (valid['Height (ft)'] > 8)]
    if clean.empty: return series.max()
    col_to_use = 'Norm_Carry' if 'Norm_Carry' in clean.columns else 'SL_Carry'
    return clean.loc[clean[col_to_use].idxmax(), col_to_use]

def filter_outliers(df):
    mask_physics = ((df['Smash'] >= 1.0) & (df['Smash'] <= 1.58) & (df['Spin (rpm)'] > 500) & (df['Height (ft)'] > 5))
    df_phys = df[mask_physics].copy()
    dropped = len(df) - len(df_phys)
    if not df_phys.empty:
        groups = df_phys.groupby('club')['SL_Carry']
        Q1 = groups.transform(lambda x: x.quantile(0.25))
        Q3 = groups.transform(lambda x: x.quantile(0.75))
        IQR = Q3 - Q1
        mask_iqr = (df_phys['SL_Carry'] >= (Q1 - 1.5 * IQR)) & (df_phys['SL_Carry'] <= (Q3 + 3.0 * IQR))
        df_final = df_phys[mask_iqr].copy()
        dropped += (len(df_phys) - len(df_final))
        return df_final, dropped
    return df_phys, dropped

def check_range(club_name, value, metric_idx, handicap):
    current_bag = st.session_state['my_bag']
    c_lower = str(club_name).lower()
    user_loft = current_bag.get(club_name, 30.0)
    launch_help = 0 if handicap < 5 else 2.0
    if 'driver' in c_lower:
        aoa = (-2.0, 5.0); launch = (10.0 + launch_help, 16.0 + launch_help); spin = (1800, 2800)
    elif 'wood' in c_lower or 'hybrid' in c_lower:
        aoa = (-4.0, 1.0); launch = (user_loft * 0.7 - 2.0, user_loft * 0.7 + 2.0); spin = (user_loft * 180, user_loft * 250)
    else:
        aoa = (-6.0, -1.0); launch = (user_loft * 0.5 - 2.0, user_loft * 0.5 + 2.0); spin = (user_loft * 180, user_loft * 220)
    ranges = [aoa, launch, spin]
    min_v, max_v = ranges[metric_idx]
    if min_v <= value <= max_v: return "Optimal ✅", "normal"
    elif value < min_v: return f"{value - min_v:.1f} (Low) ⚠️", "inverse"
    else: return f"+{value - max_v:.1f} (High) ⚠️", "inverse"

# --- MAIN APP LOGIC ---

# 1. AUTHENTICATION CHECK
if st.session_state['user_id'] is None:
    col1, col2 = st.columns(2)
    col1.title("⛳ FS Pro Login")
    
    tab_login, tab_reg = col1.tabs(["Login", "Register"])
    
    with tab_login:
        l_user = st.text_input("Username")
        l_pass = st.text_input("Password", type="password")
        if st.button("Log In"):
            uid = login_user(l_user, l_pass)
            if uid:
                st.session_state['user_id'] = uid
                st.session_state['username'] = l_user
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab_reg:
        r_user = st.text_input("New Username")
        r_pass = st.text_input("New Password", type="password")
        if st.button("Sign Up"):
            if register_user(r_user, r_pass):
                st.success("Account created! Please log in.")
    st.stop() 

# 2. LOAD DATA (Including Prefs)
if st.session_state['master_df'].empty:
    df_load, bag_load, prefs_load = load_user_data(st.session_state['user_id'])
    st.session_state['master_df'] = df_load
    st.session_state['my_bag'] = bag_load
    st.session_state['my_prefs'] = prefs_load

# 3. GLOBAL VARS
master_df = st.session_state['master_df']
my_bag = st.session_state['my_bag']
my_prefs = st.session_state['my_prefs']
active_user = st.session_state['username']

# --- SIDEBAR ---
with st.sidebar:
    st.write(f"👤 **{active_user}**")
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()
    
    st.markdown("---")
    st.header("1. Timeframe")
    date_filter = st.selectbox("Select Range:", ["All Time", "Last Session", "Last 3 Sessions", "Last 5 Sessions", "Last 30 Days", "Year to Date"])
    
    st.markdown("---")
    st.header("2. Data Manager")
    
    with st.expander("📂 Add New Session"):
        up_file = st.file_uploader("Upload CSVs", accept_multiple_files=True)
        import_date = st.date_input("Session Date")
        if st.button("➕ Add to Database") and up_file:
            new_rows_count = 0
            for f in up_file:
                try:
                    raw = pd.read_csv(f)
                    clean = clean_mevo_data(raw, f.name, import_date)
                    clean['user_id'] = st.session_state['user_id'] 
                    
                    # Upload to Database
                    clean.to_sql('shots', engine, if_exists='append', index=False)
                    new_rows_count += len(clean)
                except Exception as e:
                    st.error(f"Error uploading {f.name}: {e}")
            
            if new_rows_count > 0:
                st.success(f"Uploaded {new_rows_count} shots!")
                st.session_state['master_df'], _, _ = load_user_data(st.session_state['user_id'])
                st.rerun()

    with st.expander("🗑️ Database Actions"):
        if st.button("DELETE ALL MY DATA"):
            with engine.connect() as conn:
                conn.execute(text("DELETE FROM shots WHERE user_id = :uid"), {"uid": st.session_state['user_id']})
                conn.commit()
            st.session_state['master_df'] = pd.DataFrame()
            st.rerun()

    st.markdown("---")
    with st.expander("⚙️ Settings"):
        env_mode = st.radio("Filter Mode:", ["All", "Outdoor Only", "Indoor Only"], index=0)
        
        # HANDICAP LOGIC (With Saving)
        new_handicap = st.number_input("Handicap", 0, 54, value=my_prefs['handicap'])
        if new_handicap != my_prefs['handicap']:
            save_user_pref(st.session_state['user_id'], 'handicap', new_handicap)
            st.session_state['my_prefs']['handicap'] = new_handicap
            st.toast("Handicap Saved!")
            
        new_temp = st.slider("Temp (°F)", 30, 110, value=my_prefs['temp'])
        if new_temp != my_prefs['temp']:
            save_user_pref(st.session_state['user_id'], 'temp', new_temp)
            st.session_state['my_prefs']['temp'] = new_temp

        new_alt = st.number_input("Altitude (ft)", 0, 10000, value=my_prefs['altitude'])
        if new_alt != my_prefs['altitude']:
            save_user_pref(st.session_state['user_id'], 'altitude', new_alt)
            st.session_state['my_prefs']['altitude'] = new_alt
            
        ball_opts = ["Premium (100%)", "Economy (98%)", "Range (90%)"]
        try:
            ball_idx = ball_opts.index(my_prefs.get('ball', "Premium (100%)"))
        except:
            ball_idx = 0
            
        new_ball = st.selectbox("Ball", ball_opts, index=ball_idx)
        if new_ball != my_prefs.get('ball'):
            save_user_pref(st.session_state['user_id'], 'ball', new_ball)
            st.session_state['my_prefs']['ball'] = new_ball

        smash_limit = st.slider("Max Smash Cap", 1.40, 1.60, 1.52)
        outlier_mode = st.checkbox("Auto-Clean Outliers", True)
        
    with st.expander("🎒 Bag Setup"):
        club_sel = st.selectbox("Club", CLUB_SORT_ORDER)
        curr_loft = my_bag.get(club_sel, 30.0)
        new_loft = st.number_input(f"{club_sel} Loft", value=float(curr_loft), step=0.5)
        if st.button("Save Loft"):
            save_bag_loft(st.session_state['user_id'], club_sel, new_loft)
            st.session_state['my_bag'][club_sel] = new_loft
            st.toast("Saved!")

# --- 4. VISUALIZATION ---
# [VISUALIZATION LOGIC SAME AS BEFORE, using local vars for prefs]
handicap = my_prefs['handicap']
temp = my_prefs['temp']
alt = my_prefs['altitude']
ball = my_prefs.get('ball', "Premium (100%)")

if not master_df.empty:
    st.title(f"⛳ Analytics: {active_user}")
    
    # Filter Logic
    filtered_df = master_df.copy()
    if 'Date' in filtered_df.columns:
        filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
        if date_filter == "Last Session":
            last = filtered_df['Date'].max()
            filtered_df = filtered_df[filtered_df['Date'] == last]
        elif date_filter == "Last 3 Sessions":
            dates = sorted(filtered_df['Date'].unique(), reverse=True)[:3]
            filtered_df = filtered_df[filtered_df['Date'].isin(dates)]
        elif date_filter == "Last 5 Sessions":
            dates = sorted(filtered_df['Date'].unique(), reverse=True)[:5]
            filtered_df = filtered_df[filtered_df['Date'].isin(dates)]
        elif date_filter == "Last 30 Days":
            cut = pd.Timestamp.now() - timedelta(days=30)
            filtered_df = filtered_df[filtered_df['Date'] >= cut]
        elif date_filter == "Year to Date":
            cut = pd.Timestamp(pd.Timestamp.now().year, 1, 1)
            filtered_df = filtered_df[filtered_df['Date'] >= cut]
            
    if filtered_df.empty:
        st.warning("No data found for filter.")
        st.stop()

    if env_mode == "Outdoor Only" and 'Mode' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Mode'].str.contains("Outdoor", case=False, na=False)]
    elif env_mode == "Indoor Only" and 'Mode' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Mode'].str.contains("Indoor", case=False, na=False)]
    
    filtered_df = filtered_df[filtered_df['Smash'] <= smash_limit].copy()

    if outlier_mode:
        filtered_df, dropped = filter_outliers(filtered_df)
        if dropped > 0: st.toast(f"Cleaned {dropped} outliers", icon="🧹")

    # Normalization
    t_fac = 1 + ((temp - 70) * 0.001)
    a_fac = 1 + (alt / 1000.0 * 0.011)
    b_fac = {"Premium (100%)":1.0, "Economy (98%)":0.98, "Range (90%)":0.90}[ball]
    total_fac = t_fac * a_fac * b_fac
    
    filtered_df['Norm_Carry'] = filtered_df['SL_Carry'] * total_fac
    filtered_df['Norm_Total'] = filtered_df['SL_Total'] * total_fac

    # TABS
    tabs = st.tabs(["🏠 Dashboard", "🎒 My Bag", "🎯 Accuracy", "📈 Trends", "🔬 Mechanics", "⚔️ Compare"])

    with tabs[0]:
        st.subheader(f"Performance: {date_filter}")
        tot = len(filtered_df)
        sess = filtered_df['Date'].nunique()
        fav = filtered_df['club'].mode()[0] if tot > 0 else "-"
        drivs = filtered_df[filtered_df['club']=='Driver']
        best = get_smart_max(drivs['Norm_Carry'], drivs) if not drivs.empty else 0
        
        c1, c2, c3, c4 = st.columns(4)
        def hero(col, lbl, val, sub):
            col.markdown(f"""<div class="hero-card"><div class="hero-title">{lbl}</div><div class="hero-metric">{val}</div><div class="hero-sub">{sub}</div></div>""", unsafe_allow_html=True)
        hero(c1, "Volume", tot, f"{sess} Sessions")
        hero(c2, "Best Drive", f"{best:.0f}y", f"Normalized")
        hero(c3, "Favorite", fav, "Most Used")
        
        i7 = filtered_df[filtered_df['club'] == '7 Iron']
        if not i7.empty:
            disp = i7['Lateral_Clean'].std() * 2
            hero(c4, "7i Dispersion", f"±{disp:.1f}y", "95% Conf.")
        else: hero(c4, "7i Dispersion", "-", "No Data")

    with tabs[1]:
        st.subheader("🎒 Stock Yardages")
        stats = filtered_df.groupby('club').agg({'Norm_Carry': 'mean', 'Norm_Total': 'mean', 'Ball (mph)': 'mean', 'club': 'count'}).rename(columns={'club': 'Count'})
        ranges = filtered_df.groupby('club')['Norm_Carry'].quantile([0.20, 0.80]).unstack()
        bag = stats.join(ranges)
        bag['Sort'] = bag.index.map(lambda x: CLUB_SORT_ORDER.index(x) if x in CLUB_SORT_ORDER else 99)
        bag = bag.sort_values('Sort')
        
        st.write("---")
        cols = st.columns(4)
        for i, (name, row) in enumerate(bag.iterrows()):
            with cols[i%4]:
                st.markdown(f"""
                <div style="background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #444; margin-bottom: 10px;">
                    <h3 style="margin:0; color: #4DD0E1;">{name}</h3>
                    <h2 style="margin:0; font-size: 32px; color: #FFF;">{row['Norm_Carry']:.0f}<span style="font-size:16px; color:#888"> yds</span></h2>
                    <div style="font-size: 14px; color: #00E5FF; margin-bottom: 5px;">Range: {row[0.2]:.0f} - {row[0.8]:.0f}</div>
                    <hr style="border-color: #444; margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #888;">
                        <span>{row['Ball (mph)']:.0f} mph</span><span>Tot: {row['Norm_Total']:.0f}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

    with tabs[2]:
        st.subheader("🎯 Accuracy & Dispersion")
        avail = [c for c in CLUB_SORT_ORDER if c in filtered_df['club'].unique()]
        if avail:
            tgt_club = st.selectbox("Analyze Club:", avail, key="acc_club_select")
            subset = filtered_df[filtered_df['club'] == tgt_club]
            if not subset.empty:
                avg_c = subset['Norm_Carry'].mean()
                lat_std = subset['Lateral_Clean'].std()
                long_std = subset['Norm_Carry'].std()
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Avg Carry", f"{avg_c:.1f}")
                c2.metric("Dispersion", f"±{lat_std*2:.1f}y", "Width (95%)")
                c3.metric("Depth", f"±{long_std*2:.1f}y", "Long/Short (95%)")
                c4.metric("Shots", len(subset))

                c_ch1, c_ch2 = st.columns([3, 1])
                with c_ch1:
                    fig = px.scatter(subset, x='Lateral_Clean', y='Norm_Carry', color='Smash', hover_data=['Date', 'Spin (rpm)'], title=f"Dispersion: {tgt_club}")
                    fig.add_shape(type="circle", x0=-lat_std*2, y0=avg_c-long_std*2, x1=lat_std*2, y1=avg_c+long_std*2, line=dict(color="#FF1744", width=3, dash="dot"), fillcolor="rgba(255, 23, 68, 0.1)")
                    fig.add_shape(type="circle", x0=-lat_std, y0=avg_c-long_std, x1=lat_std, y1=avg_c+long_std, line=dict(color="#00E676", width=2, dash="solid"), fillcolor="rgba(0, 230, 118, 0.1)")
                    fig.add_vline(x=0, line_color="white", opacity=0.2); fig.add_hline(y=avg_c, line_color="white", opacity=0.2)
                    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                with c_ch2:
                    st.info(f"🎯 **Strategy:**\n\n**Red Circle (95%):**\nYou need **{lat_std*4:.0f} yards** of total width to be safe.\n\n**Green Circle (68%):**\nYour 'Stock Shot' lands in a **{lat_std*2:.0f} yard** wide window.")
        else:
            st.warning("No data for accuracy.")

    with tabs[3]:
        st.subheader("📈 Trends")
        if avail:
            tr_club = st.selectbox("Club:", avail, key="tr_club")
            metric = st.selectbox("Metric:", ["Ball (mph)", "Norm_Carry", "Lateral_Clean"])
            tr_data = filtered_df[filtered_df['club'] == tr_club].groupby('Date')[metric].agg(['mean', 'std']).reset_index()
            if len(tr_data) > 1:
                fig_tr = go.Figure()
                fig_tr.add_trace(go.Scatter(x=tr_data['Date'], y=tr_data['mean'], mode='lines+markers', name='Avg', line=dict(color='#00E676')))
                fig_tr.add_trace(go.Scatter(x=tr_data['Date'], y=tr_data['mean']+tr_data['std'], mode='lines', line=dict(width=0), showlegend=False))
                fig_tr.add_trace(go.Scatter(x=tr_data['Date'], y=tr_data['mean']-tr_data['std'], mode='lines', fill='tonexty', fillcolor='rgba(0,230,118,0.2)', line=dict(width=0), showlegend=False))
                fig_tr.update_layout(template="plotly_dark", hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_tr, use_container_width=True)
            else: st.info("Need 2+ sessions.")

    with tabs[4]:
        st.subheader("🔬 Mechanics")
        if avail:
            mech_club = st.selectbox("Club:", avail, key="mech_club")
            mech_data = filtered_df[filtered_df['club'] == mech_club]
            if not mech_data.empty:
                c1, c2, c3 = st.columns(3)
                if 'AOA (°)' in mech_data.columns:
                    v = mech_data['AOA (°)'].mean()
                    s, c = check_range(mech_club, v, 0, handicap)
                    c1.metric("AoA", f"{v:.1f}°", s, delta_color=c)
                if 'Spin (rpm)' in mech_data.columns:
                    v = mech_data['Spin (rpm)'].mean()
                    s, c = check_range(mech_club, v, 2, handicap)
                    c2.metric("Spin", f"{v:.0f}", s, delta_color=c)
                if 'Club (mph)' in mech_data.columns:
                    with st.expander("🚀 Efficiency Lab", expanded=True):
                        bench = st.radio("Benchmark:", ["Scratch", "Tour Pro"], horizontal=True)
                        spd = mech_data['Club (mph)'].mean()
                        cry = mech_data['Norm_Carry'].mean()
                        loft = my_bag.get(mech_club, 30.0)
                        pot = calculate_optimal_carry(spd, loft, bench)
                        eff = (cry / pot) * 100 if pot > 0 else 0
                        st.metric("Efficiency", f"{eff:.1f}%", f"{cry - pot:.1f}y Gap")
                        st.markdown(f"""<div class="eff-container"><div class="eff-bar-fill" style="width: {min(eff, 100)}%;"></div></div>""", unsafe_allow_html=True)

    with tabs[5]:
        st.subheader("⚔️ Compare")
        if avail:
            comp_club = st.selectbox("Club:", avail, key='comp_club')
            c_data = filtered_df[filtered_df['club'] == comp_club].copy()
            c_data['Label'] = c_data['Date'].dt.strftime('%m-%d') + " " + c_data['Session']
            unq = c_data['Label'].unique()
            if len(unq) >= 2:
                c1, c2 = st.columns(2)
                s1 = c1.selectbox("Session A", unq, 0)
                s2 = c2.selectbox("Session B", unq, 1)
                d1 = c_data[c_data['Label'] == s1]
                d2 = c_data[c_data['Label'] == s2]
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=d1['Norm_Carry'], name='A', opacity=0.75))
                fig.add_trace(go.Histogram(x=d2['Norm_Carry'], name='B', opacity=0.75))
                fig.update_layout(barmode='overlay', template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("Need 2+ sessions.")

else:
    st.markdown("<h1 style='text-align: center; color: #4DD0E1;'>FS Pro Analytics</h1>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.info("🆕 **New User?**\nUse Sidebar > Data Manager > Add Session")
    c2.warning("👋 **Welcome!**\nUpload your first Mevo session CSV in the sidebar.")
