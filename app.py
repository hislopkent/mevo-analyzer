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

# --- CSS STYLING (THEME ENFORCER) ---
st.markdown("""
<style>
    /* 1. GLOBAL TEXT & BACKGROUND */
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stText { color: #FAFAFA !important; }
    
    /* 2. FIX TABS (Invisible Text Fix) */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1E222B;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #FAFAFA; /* Unselected Tab Text */
    }
    .stTabs [aria-selected="true"] {
        background-color: #262730;
        color: #4DD0E1 !important; /* Selected Tab Text */
        border-bottom: 2px solid #4DD0E1;
    }

    /* 3. FIX INPUT FIELDS (White on White Fix) */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] div {
        color: #FAFAFA !important;
        background-color: #262730 !important;
        border-color: #444 !important;
    }
    .stSelectbox svg { fill: #FAFAFA !important; }
    
    /* 4. FIX EXPANDERS */
    .streamlit-expanderHeader {
        background-color: #262730 !important;
        color: #FAFAFA !important;
        border-radius: 5px;
    }
    
    /* 5. FIX BUTTONS */
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
    
    /* 6. METRIC CARDS */
    .hero-card {
        background: linear-gradient(145deg, #1E222B, #262730);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid #444;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .hero-metric { font-size: 36px; font-weight: 800; color: #FAFAFA; margin: 0; }
    .hero-title { font-size: 14px; text-transform: uppercase; color: #B0B3B8; margin-bottom: 10px; }
    .hero-sub { font-size: 12px; color: #00E5FF; margin-top: 5px; }

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
            
            # Initialize default prefs
            uid_query = text("SELECT id FROM users WHERE username = :u")
            uid = conn.execute(uid_query, {"u": username}).fetchone()[0]
            conn.execute(text("INSERT INTO user_prefs (user_id) VALUES (:uid)"), {"uid": uid})
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
    
    # 3. Load User Prefs
    prefs = {'handicap': 15, 'temp': 75, 'altitude': 500, 'ball': 'Premium (100%)'}
    try:
        query = text("SELECT handicap, temp, altitude, ball FROM user_prefs WHERE user_id = :uid")
        with engine.connect() as conn:
            result = conn.execute(query, {"uid": user_id}).fetchone()
            if result:
                prefs = {'handicap': result[0], 'temp': result[1], 'altitude': result[2], 'ball': result[3]}
    except: pass
        
    return df, bag, prefs

def save_user_pref(user_id, col, val):
    try:
        query = text(f"INSERT INTO user_prefs (user_id, {col}) VALUES (:uid, :val) ON CONFLICT (user_id) DO UPDATE SET {col} = :val")
        with engine.connect() as conn:
            conn.execute(query, {"uid": user_id, "val": val})
            conn.commit()
    except Exception as e: st.error(f"Save Error: {e}")

def save_bag_loft(user_id, club, loft):
    with engine.connect() as conn:
        sql = text("INSERT INTO bag_settings (user_id, club_name, loft) VALUES (:uid, :c, :l) ON CONFLICT (user_id, club_name) DO UPDATE SET loft = EXCLUDED.loft;")
        conn.execute(sql, {"uid": user_id, "c": club, "l": loft})
        conn.commit()

# --- SESSION MANAGEMENT FUNCTIONS ---
def rename_session(user_id, old_name, new_name):
    try:
        with engine.connect() as conn:
            sql = text('UPDATE shots SET "Session" = :new WHERE "Session" = :old AND user_id = :uid')
            result = conn.execute(sql, {"new": new_name, "old": old_name, "uid": user_id})
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Rename Failed: {e}")
        return False

def delete_session(user_id, session_name):
    try:
        with engine.connect() as conn:
            sql = text('DELETE FROM shots WHERE "Session" = :sess AND user_id = :uid')
            conn.execute(sql, {"sess": session_name, "uid": user_id})
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Delete Failed: {e}")
        return False

# --- CONSTANTS ---
DEFAULT_LOFTS = {
    'Driver': 10.5, '3 Wood': 15.0, '5 Wood': 18.0, 'Hybrid': 21.0, 
    '3 Iron': 21.0, '4 Iron': 24.0, '5 Iron': 27.0, '6 Iron': 30.0, 
    '7 Iron': 34.0, '8 Iron': 38.0, '9 Iron': 42.0, 'PW': 46.0, 
    'GW': 50.0, 'SW': 54.0, 'LW': 58.0
}
CLUB_SORT_ORDER = ['Driver', '3 Wood', '5 Wood', '7 Wood', 'Hybrid', '2 Iron', '3 Iron', '4 Iron', '5 Iron', '6 Iron', '7 Iron', '8 Iron', '9 Iron', 'PW', 'GW', 'SW', 'LW']

# --- SESSION STATE ---
if 'user_id' not in st.session_state: st.session_state['user_id'] = None
if 'username' not in st.session_state: st.session_state['username'] = None
if 'master_df' not in st.session_state: st.session_state['master_df'] = pd.DataFrame()
if 'my_bag' not in st.session_state: st.session_state['my_bag'] = DEFAULT_LOFTS.copy()
if 'my_prefs' not in st.session_state: st.session_state['my_prefs'] = {'handicap': 15, 'temp': 75, 'altitude': 500, 'ball': 'Premium (100%)'}

# --- CLEANING HELPER (AUTO-NAMING UPDATED) ---
@st.cache_data
def clean_mevo_data(df, filename, selected_date):
    df_clean = df[df['Shot'].astype(str).str.isdigit()].copy()
    
    # --- AUTO-NAMING LOGIC ---
    # 1. Determine Date String
    if 'Date' in df_clean.columns:
        date_obj = pd.to_datetime(df_clean['Date'], errors='coerce').max()
        if pd.isnull(date_obj): date_obj = pd.to_datetime(selected_date)
    else:
        date_obj = pd.to_datetime(selected_date)
    date_str = date_obj.strftime('%Y-%m-%d')
    
    # 2. Determine Mode (Indoor/Outdoor)
    mode_str = "Unknown"
    if 'Mode' in df_clean.columns:
        mode_val = df_clean['Mode'].iloc[0] if not df_clean.empty else ""
        if "Indoor" in str(mode_val): mode_str = "Indoor"
        elif "Outdoor" in str(mode_val): mode_str = "Outdoor"
    
    # 3. Final Name
    session_name = f"{date_str} {mode_str}"
    df_clean['Session'] = session_name
    # -------------------------

    df_clean['Date'] = date_obj # Standardize Date Column
    
    # Lateral Clean
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
        x_points, y_points = [9, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], [2.90, 2.75, 2.60, 2.50, 2.40, 2.30, 2.20, 2.10, 2.00, 1.85, 1.70]
    else:
        x_points, y_points = [9, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], [2.75, 2.65, 2.55, 2.48, 2.40, 2.32, 2.20, 2.05, 1.85, 1.65, 1.50]
    safe_loft = max(5, min(loft, 64))
    return club_speed * np.interp(safe_loft, x_points, y_points)

def get_smart_max(series, df_subset):
    valid = df_subset.loc[series.index]
    clean = valid[(valid['Smash'] <= 1.58) & (valid['Smash'] >= 1.0) & (valid['Spin (rpm)'] > 500) & (valid['Height (ft)'] > 8)]
    if clean.empty: return series.max()
    col_to_use = 'Norm_Carry' if 'Norm_Carry' in clean.columns else 'SL_Carry'
    return clean.loc[clean[col_to_use].idxmax(), col_to_use]

def filter_outliers(df):
    mask_phys = ((df['Smash'] >= 1.0) & (df['Smash'] <= 1.58) & (df['Spin (rpm)'] > 500) & (df['Height (ft)'] > 5))
    df_phys = df[mask_phys].copy()
    dropped = len(df) - len(df_phys)
    if not df_phys.empty:
        groups = df_phys.groupby('club')['SL_Carry']
        Q1, Q3 = groups.transform(lambda x: x.quantile(0.25)), groups.transform(lambda x: x.quantile(0.75))
        mask_iqr = (df_phys['SL_Carry'] >= (Q1 - 1.5 * (Q3 - Q1))) & (df_phys['SL_Carry'] <= (Q3 + 3.0 * (Q3 - Q1)))
        df_final = df_phys[mask_iqr].copy()
        dropped += (len(df_phys) - len(df_final))
        return df_final, dropped
    return df_phys, dropped

def check_range(club_name, value, metric_idx, handicap):
    current_bag = st.session_state['my_bag']
    c_lower, user_loft = str(club_name).lower(), current_bag.get(club_name, 30.0)
    launch_help = 0 if handicap < 5 else 2.0
    if 'driver' in c_lower: ranges = [(-2.0, 5.0), (10.0+launch_help, 16.0+launch_help), (1800, 2800)]
    elif 'wood' in c_lower or 'hybrid' in c_lower: ranges = [(-4.0, 1.0), (user_loft*0.7-2, user_loft*0.7+2), (user_loft*180, user_loft*250)]
    else: ranges = [(-6.0, -1.0), (user_loft*0.5-2, user_loft*0.5+2), (user_loft*180, user_loft*220)]
    min_v, max_v = ranges[metric_idx]
    if min_v <= value <= max_v: return "Optimal ✅", "normal"
    elif value < min_v: return f"{value - min_v:.1f} (Low) ⚠️", "inverse"
    else: return f"+{value - max_v:.1f} (High) ⚠️", "inverse"

# --- MAIN APP ---

# 1. LOGIN
if st.session_state['user_id'] is None:
    col1, col2 = st.columns(2)
    col1.title("⛳ FS Pro Login")
    tab_login, tab_reg = col1.tabs(["Login", "Register"])
    with tab_login:
        l_user, l_pass = st.text_input("Username"), st.text_input("Password", type="password")
        if st.button("Log In"):
            uid = login_user(l_user, l_pass)
            if uid:
                st.session_state['user_id'] = uid
                st.session_state['username'] = l_user
                st.rerun()
            else: st.error("Invalid credentials")
    with tab_reg:
        r_user, r_pass = st.text_input("New Username"), st.text_input("New Password", type="password")
        if st.button("Sign Up"):
            if register_user(r_user, r_pass): st.success("Account created! Please log in.")
    st.stop()

# 2. LOAD
if st.session_state['master_df'].empty:
    df_load, bag_load, prefs_load = load_user_data(st.session_state['user_id'])
    st.session_state['master_df'], st.session_state['my_bag'], st.session_state['my_prefs'] = df_load, bag_load, prefs_load

# 3. VARS
master_df, my_bag, my_prefs, active_user = st.session_state['master_df'], st.session_state['my_bag'], st.session_state['my_prefs'], st.session_state['username']

# --- SIDEBAR ---
with st.sidebar:
    st.write(f"👤 **{active_user}**")
    if st.button("Logout"): st.session_state.clear(); st.rerun()
    st.markdown("---")
    
    st.header("1. Data Manager")
    
    # IMPORT TAB
    with st.expander("📂 Import Sessions"):
        up_file = st.file_uploader("Upload CSVs", accept_multiple_files=True)
        import_date = st.date_input("Session Date", value=pd.Timestamp.now())
        if st.button("➕ Add to Database") and up_file:
            count = 0
            for f in up_file:
                try:
                    raw = pd.read_csv(f)
                    clean = clean_mevo_data(raw, f.name, import_date)
                    clean['user_id'] = st.session_state['user_id']
                    clean.to_sql('shots', engine, if_exists='append', index=False)
                    count += len(clean)
                except Exception as e: st.error(f"Error {f.name}: {e}")
            if count > 0:
                st.success(f"Added {count} shots.")
                st.session_state['master_df'], _, _ = load_user_data(st.session_state['user_id'])
                st.rerun()

    # MANAGE SESSIONS TAB (RENAME / DELETE)
    with st.expander("✏️ Manage Sessions"):
        if not master_df.empty:
            sessions = sorted(master_df['Session'].unique(), reverse=True)
            man_tab1, man_tab2 = st.tabs(["Rename", "Delete"])
            
            with man_tab1:
                target_sess = st.selectbox("Select Session:", sessions, key="ren_target")
                new_sess_name = st.text_input("New Name:", value=target_sess)
                if st.button("Update Name"):
                    if rename_session(st.session_state['user_id'], target_sess, new_sess_name):
                        st.success("Updated!")
                        st.session_state['master_df'], _, _ = load_user_data(st.session_state['user_id'])
                        st.rerun()

            with man_tab2:
                del_target = st.selectbox("Select Session:", sessions, key="del_target")
                st.warning(f"Delete '{del_target}'?")
                if st.button("🗑️ Confirm Delete", type="primary"):
                    if delete_session(st.session_state['user_id'], del_target):
                        st.success("Deleted!")
                        st.session_state['master_df'], _, _ = load_user_data(st.session_state['user_id'])
                        st.rerun()
        else:
            st.info("No sessions found.")

    st.markdown("---")
    # SETTINGS
    with st.expander("⚙️ Settings"):
        env_mode = st.radio("Filter Mode:", ["All", "Outdoor Only", "Indoor Only"], index=0)
        
        # Handicap Save
        nh = st.number_input("Handicap", 0, 54, value=my_prefs['handicap'])
        if nh != my_prefs['handicap']:
            save_user_pref(st.session_state['user_id'], 'handicap', nh)
            st.session_state['my_prefs']['handicap'] = nh
            
        # Temp Save
        nt = st.slider("Temp (°F)", 30, 110, value=my_prefs['temp'])
        if nt != my_prefs['temp']:
            save_user_pref(st.session_state['user_id'], 'temp', nt)
            st.session_state['my_prefs']['temp'] = nt
        
        # Altitude Save
        na = st.number_input("Altitude (ft)", 0, 10000, value=my_prefs['altitude'])
        if na != my_prefs['altitude']:
            save_user_pref(st.session_state['user_id'], 'altitude', na)
            st.session_state['my_prefs']['altitude'] = na
            
        # Ball Save
        b_opts = ["Premium (100%)", "Economy (98%)", "Range (90%)"]
        nb = st.selectbox("Ball", b_opts, index=b_opts.index(my_prefs.get('ball', "Premium (100%)")))
        if nb != my_prefs.get('ball'):
            save_user_pref(st.session_state['user_id'], 'ball', nb)
            st.session_state['my_prefs']['ball'] = nb

        smash_limit = st.slider("Max Smash Cap", 1.40, 1.60, 1.52)
        outlier_mode = st.checkbox("Auto-Clean Outliers", True)

    with st.expander("🎒 Bag Setup"):
        c_sel = st.selectbox("Club", CLUB_SORT_ORDER)
        curr = my_bag.get(c_sel, 30.0)
        n_loft = st.number_input(f"{c_sel} Loft", value=float(curr), step=0.5)
        if st.button("Save Loft"):
            save_bag_loft(st.session_state['user_id'], c_sel, n_loft)
            st.session_state['my_bag'][c_sel] = n_loft
            st.toast("Saved!")
    
    st.markdown("---")
    st.header("2. Timeframe")
    date_filter = st.selectbox("Select Range:", ["All Time", "Last Session", "Last 3 Sessions", "Last 5 Sessions", "Last 30 Days", "Year to Date"])

# --- 4. DASHBOARD ---
handicap, temp, alt, ball = my_prefs['handicap'], my_prefs['temp'], my_prefs['altitude'], my_prefs.get('ball', "Premium (100%)")

if not master_df.empty:
    st.title(f"⛳ Analytics: {active_user}")
    
    # Filter Logic
    df_f = master_df.copy()
    if date_filter == "Last Session": df_f = df_f[df_f['Session'] == df_f['Session'].unique()[0]] # Assumes sorted load? Better to use date
    elif date_filter == "Last 3 Sessions": df_f = df_f[df_f['Session'].isin(df_f['Session'].unique()[:3])]
    elif date_filter == "Last 5 Sessions": df_f = df_f[df_f['Session'].isin(df_f['Session'].unique()[:5])]
    elif date_filter == "Last 30 Days": df_f = df_f[df_f['Date'] >= (pd.Timestamp.now() - timedelta(days=30))]
    elif date_filter == "Year to Date": df_f = df_f[df_f['Date'] >= pd.Timestamp(pd.Timestamp.now().year, 1, 1)]

    # Note: Using Session ID sort is safer than Date for uniqueness if multi-session same day
    
    if env_mode == "Outdoor Only" and 'Mode' in df_f.columns: df_f = df_f[df_f['Mode'].str.contains("Outdoor", case=False, na=False)]
    elif env_mode == "Indoor Only" and 'Mode' in df_f.columns: df_f = df_f[df_f['Mode'].str.contains("Indoor", case=False, na=False)]
    
    df_f = df_f[df_f['Smash'] <= smash_limit].copy()
    if outlier_mode:
        df_f, dropped = filter_outliers(df_f)
        if dropped > 0: st.toast(f"Cleaned {dropped} outliers", icon="🧹")

    # Normalization
    tf, af = 1 + ((temp - 70) * 0.001), 1 + (alt / 1000.0 * 0.011)
    bf = {"Premium (100%)":1.0, "Economy (98%)":0.98, "Range (90%)":0.90}[ball]
    df_f['Norm_Carry'] = df_f['SL_Carry'] * tf * af * bf
    df_f['Norm_Total'] = df_f['SL_Total'] * tf * af * bf

    # TABS
    t1, t2, t3, t4, t5, t6 = st.tabs(["🏠 Dashboard", "🎒 My Bag", "🎯 Accuracy", "📈 Trends", "🔬 Mechanics", "⚔️ Compare"])

    with t1:
        st.subheader(f"Performance: {date_filter}")
        tot, sess = len(df_f), df_f['Session'].nunique()
        fav = df_f['club'].mode()[0] if tot > 0 else "-"
        drivs = df_f[df_f['club']=='Driver']
        best = get_smart_max(drivs['Norm_Carry'], drivs) if not drivs.empty else 0
        
        c1, c2, c3, c4 = st.columns(4)
        def hero(col, lbl, val, sub): col.markdown(f"""<div class="hero-card"><div class="hero-metric">{val}</div><div class="hero-title">{lbl}</div><div class="hero-sub">{sub}</div></div>""", unsafe_allow_html=True)
        hero(c1, "Volume", tot, f"{sess} Sessions")
        hero(c2, "Best Drive", f"{best:.0f}y", f"Normalized")
        hero(c3, "Favorite", fav, "Most Used")
        
        i7 = df_f[df_f['club'] == '7 Iron']
        disp = f"±{i7['Lateral_Clean'].std() * 2:.1f}y" if not i7.empty else "-"
        hero(c4, "7i Dispersion", disp, "95% Conf.")

    with t2:
        st.subheader("🎒 Stock Yardages")
        stats = df_f.groupby('club').agg({'Norm_Carry': 'mean', 'Norm_Total': 'mean', 'Ball (mph)': 'mean', 'club': 'count'}).rename(columns={'club': 'Count'})
        ranges = df_f.groupby('club')['Norm_Carry'].quantile([0.20, 0.80]).unstack()
        bag = stats.join(ranges)
        bag['Sort'] = bag.index.map(lambda x: CLUB_SORT_ORDER.index(x) if x in CLUB_SORT_ORDER else 99)
        bag = bag.sort_values('Sort')
        
        cols = st.columns(4)
        for i, (name, row) in enumerate(bag.iterrows()):
            with cols[i%4]:
                st.markdown(f"""
                <div style="background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #444; margin-bottom: 10px;">
                    <h3 style="margin:0; color: #4DD0E1;">{name}</h3>
                    <h2 style="margin:0; font-size: 32px; color: #FFF;">{row['Norm_Carry']:.0f}<span style="font-size:16px; color:#888"> yds</span></h2>
                    <div style="font-size: 14px; color: #00E5FF; margin-bottom: 5px;">Range: {row[0.2]:.0f} - {row[0.8]:.0f}</div>
                    <div style="font-size: 12px; color: #888;">{row['Ball (mph)']:.0f} mph | Tot: {row['Norm_Total']:.0f}</div>
                </div>""", unsafe_allow_html=True)

    with t3:
        avail = [c for c in CLUB_SORT_ORDER if c in df_f['club'].unique()]
        if avail:
            tgt = st.selectbox("Club:", avail, key="acc_c")
            sub = df_f[df_f['club'] == tgt]
            if not sub.empty:
                ac, ls, los = sub['Norm_Carry'].mean(), sub['Lateral_Clean'].std(), sub['Norm_Carry'].std()
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Carry", f"{ac:.1f}")
                c2.metric("Dispersion", f"±{ls*2:.1f}y")
                c3.metric("Depth", f"±{los*2:.1f}y")
                
                fig = px.scatter(sub, x='Lateral_Clean', y='Norm_Carry', color='Smash', title=f"Dispersion: {tgt}")
                fig.add_shape(type="circle", x0=-ls*2, y0=ac-los*2, x1=ls*2, y1=ac+los*2, line=dict(color="#FF1744", width=3, dash="dot"))
                fig.add_vline(x=0, line_color="white", opacity=0.2); fig.add_hline(y=ac, line_color="white", opacity=0.2)
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

    with t4:
        if avail:
            tc = st.selectbox("Club:", avail, key="tr_c")
            tm = st.selectbox("Metric:", ["Ball (mph)", "Norm_Carry", "Lateral_Clean"])
            td = df_f[df_f['club'] == tc].groupby('Date')[tm].agg(['mean', 'std']).reset_index()
            if len(td) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=td['Date'], y=td['mean'], mode='lines+markers', line=dict(color='#00E676')))
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

    with t5:
        if avail:
            mc = st.selectbox("Club:", avail, key="mc_c")
            md = df_f[df_f['club'] == mc]
            if not md.empty:
                c1, c2, c3 = st.columns(3)
                if 'AOA (°)' in md.columns: 
                    v = md['AOA (°)'].mean()
                    s, c = check_range(mc, v, 0, handicap)
                    c1.metric("AoA", f"{v:.1f}°", s, delta_color=c)
                if 'Spin (rpm)' in md.columns:
                    v = md['Spin (rpm)'].mean()
                    s, c = check_range(mc, v, 2, handicap)
                    c2.metric("Spin", f"{v:.0f}", s, delta_color=c)
                if 'Club (mph)' in md.columns:
                    spd, cry = md['Club (mph)'].mean(), md['Norm_Carry'].mean()
                    pot = calculate_optimal_carry(spd, my_bag.get(mc, 30.0))
                    eff = (cry / pot) * 100 if pot > 0 else 0
                    c3.metric("Efficiency", f"{eff:.1f}%", f"Gap: {cry-pot:.1f}y")

    with t6:
        if avail:
            cc = st.selectbox("Club:", avail, key='comp_c')
            cd = df_f[df_f['club'] == cc].copy()
            unq = cd['Session'].unique()
            if len(unq) >= 2:
                c1, c2 = st.columns(2)
                s1, s2 = c1.selectbox("A", unq, 0), c2.selectbox("B", unq, 1)
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=cd[cd['Session']==s1]['Norm_Carry'], name='A', opacity=0.75))
                fig.add_trace(go.Histogram(x=cd[cd['Session']==s2]['Norm_Carry'], name='B', opacity=0.75))
                fig.update_layout(barmode='overlay', template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
else:
    st.title(f"⛳ Analytics: {active_user}")
    st.info("No data yet. Upload a session in the Sidebar!")
