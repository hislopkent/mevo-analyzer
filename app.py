import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE SETUP ---
st.set_page_config(page_title="Homegrown FS Pro Analytics", layout="wide", page_icon="⛳")

# Custom CSS
st.markdown("""
<style>
    /* 1. FORCE MAIN DARK THEME */
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    
    /* 2. SIDEBAR GLOBAL STYLES */
    section[data-testid="stSidebar"] {
        background-color: #12151d;
        border-right: 1px solid #333;
    }
    
    /* Force ALL Sidebar Text White */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio div[role='radiogroup'] label div {
        color: #FAFAFA !important;
    }

    /* 3. FIX PLAYER CONFIG INPUTS */
    section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] select {
        background-color: #262730 !important;
        color: #FAFAFA !important;
        border: 1px solid #444 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSliderTickBarMin"],
    section[data-testid="stSidebar"] div[data-testid="stSliderTickBarMax"],
    section[data-testid="stSidebar"] div[data-testid="stThumbValue"] {
        color: #FAFAFA !important;
    }

    /* 4. FIX FILE UPLOADER */
    section[data-testid="stSidebar"] [data-testid='stFileUploader'] section {
        background-color: #262730 !important;
        border: 1px dashed #4DD0E1 !important;
    }
    section[data-testid="stSidebar"] [data-testid='stFileUploader'] div,
    section[data-testid="stSidebar"] [data-testid='stFileUploader'] span,
    section[data-testid="stSidebar"] [data-testid='stFileUploader'] small {
        color: #FAFAFA !important;
    }
    section[data-testid="stSidebar"] [data-testid='stFileUploader'] button {
        background-color: #1E222B !important;
        color: #4DD0E1 !important;
        border: 1px solid #4DD0E1 !important;
    }

    /* 5. DASHBOARD CARDS */
    .hero-card {
        background: linear-gradient(145deg, #1E222B, #262730);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid #444;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        height: 100%;
        transition: transform 0.2s;
    }
    .hero-card:hover {
        transform: translateY(-5px);
        border-color: #4DD0E1;
    }
    .hero-title {
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #B0B3B8;
        margin-bottom: 10px;
    }
    .hero-metric {
        font-size: 36px;
        font-weight: 800;
        color: #FAFAFA;
        margin: 0;
    }
    .hero-sub {
        font-size: 12px;
        color: #00E5FF;
        margin-top: 5px;
    }

    /* 6. TAB VISIBILITY FIXES */
    div[data-testid="stTabs"] button[aria-selected="false"] {
        color: #B0B3B8 !important;
        font-weight: 500;
    }
    div[data-testid="stTabs"] button[aria-selected="false"]:hover {
        color: #4DD0E1 !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #FAFAFA !important;
        border-top-color: #4DD0E1 !important;
    }

    /* 7. GENERAL UI */
    div[data-testid="stMetricLabel"] label { color: #B0B3B8 !important; }
    div[data-testid="stMetricValue"] { color: #4DD0E1 !important; }
    .stSelectbox label, .stNumberInput label { color: #FAFAFA !important; }
    
    /* 8. EXPANDERS */
    div[data-testid="stExpander"] {
        background-color: #1E222B !important;
        border: 1px solid #444;
        color: #FAFAFA !important;
    }
    div[data-testid="stExpander"] summary p { color: #FAFAFA !important; font-weight: 600; }
    div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] p { color: #E0E0E0 !important; }

    /* Custom Coach Box & SG Box */
    .coach-box {
        background-color: #262730;
        border-left: 4px solid #FF4081;
        padding: 15px;
        border-radius: 5px;
        margin-top: 15px;
    }
    .sg-box {
        background-color: #262730;
        border: 1px solid #444;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
    }
    .feature-card { background-color: #1E222B; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #333; }
    
    /* Summary Stats (Sidebar) */
    .stat-card-container {
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
    }
    .stat-card {
        background: linear-gradient(145deg, #1E222B, #262730);
        border-radius: 12px;
        padding: 15px;
        flex: 1;
        text-align: center;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stat-value {
        font-size: 28px;
        font-weight: 700;
        color: #4DD0E1;
        margin: 0;
    }
    .stat-label {
        font-size: 14px;
        color: #B0B3B8;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. INITIALIZE MULTI-USER SESSION STATE ---

DEFAULT_LOFTS = {'Driver': 10.5, '3 Wood': 15.0, '5 Wood': 18.0, 'Hybrid': 21.0, '3 Iron': 21.0, '4 Iron': 24.0, '5 Iron': 27.0, '6 Iron': 30.0, '7 Iron': 34.0, '8 Iron': 38.0, '9 Iron': 42.0, 'PW': 46.0, 'GW': 50.0, 'SW': 54.0, 'LW': 58.0}
CLUB_SORT_ORDER = ['Driver', '3 Wood', '5 Wood', '7 Wood', 'Hybrid', '2 Iron', '3 Iron', '4 Iron', '5 Iron', '6 Iron', '7 Iron', '8 Iron', '9 Iron', 'PW', 'GW', 'SW', 'LW']

if 'profiles' not in st.session_state:
    st.session_state['profiles'] = {
        'Default Golfer': {
            'df': pd.DataFrame(), 
            'bag': DEFAULT_LOFTS.copy()
        }
    }

if 'active_user' not in st.session_state:
    st.session_state['active_user'] = 'Default Golfer'

# Shortcut references to ACTIVE data
active_user = st.session_state['active_user']
master_df = st.session_state['profiles'][active_user]['df']
my_bag = st.session_state['profiles'][active_user]['bag']

# --- 2. HELPERS ---
def get_smart_max(series, df_subset):
    """Calculates max value filtering out physics-defying outliers."""
    valid = df_subset.loc[series.index]
    clean = valid[
        (valid['Smash'] <= 1.58) & 
        (valid['Smash'] >= 1.0) &
        (valid['Spin (rpm)'] > 500) & 
        (valid['Height (ft)'] > 8)
    ]
    if clean.empty: 
        return series.max()
    col_to_use = 'Norm_Carry' if 'Norm_Carry' in clean.columns else 'SL_Carry'
    return clean.loc[clean[col_to_use].idxmax(), col_to_use]

def get_dynamic_ranges(club_name, handicap):
    c_lower = str(club_name).lower()
    tolerance = handicap * 0.1
    launch_help = 0 if handicap < 5 else (1.0 if handicap < 15 else 2.0)
    user_loft = my_bag.get(club_name, 30.0)

    if 'driver' in c_lower:
        aoa = (-2.0 - (tolerance*0.2), 5.0 + (tolerance*0.2)) 
        launch = (10.0 + launch_help, 16.0 + launch_help + (tolerance*0.2))
        spin = (1800, 2800 + (handicap * 40))
    elif 'wood' in c_lower or 'hybrid' in c_lower:
        aoa = (-4.0, 1.0 + tolerance)
        l_center = user_loft * 0.7 
        launch = (l_center - 2.0, l_center + 2.0 + launch_help)
        spin = (user_loft * 200, user_loft * 280)
    else:
        target_aoa = -1.0 - (user_loft / 12.0)
        aoa = (target_aoa - 2.0 - tolerance, -0.5)
        l_min = (user_loft * 0.45) - 1.0
        l_max = (user_loft * 0.55) + 1.0 + launch_help
        launch = (l_min - tolerance, l_max + tolerance)
        s_base = user_loft * 210
        spin = (s_base - 1000 - (tolerance*100), s_base + 1000 + (tolerance*100))
    return aoa, launch, spin

def clean_mevo_data(df, filename, selected_date):
    df_clean = df[df['Shot'].astype(str).str.isdigit()].copy()
    df_clean['Session'] = filename.replace('.csv', '')
    
    if 'Date' in df_clean.columns:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce').fillna(pd.to_datetime(selected_date))
    else:
        df_clean['Date'] = pd.to_datetime(selected_date)
    
    def parse_lr(val):
        if pd.isna(val): return 0.0
        s_val = str(val).strip()
        if 'L' in s_val: return -float(s_val.replace('L','').strip())
        if 'R' in s_val: return float(s_val.replace('R','').strip())
        try: return float(s_val)
        except: return 0.0

    dir_cols = ['Lateral (yds)', 'Swing H (°)', 'Launch H (°)', 'Spin Axis (°)', 'Club Path (°)', 'FTP (°)']
    for col in dir_cols:
        if col in df_clean.columns:
            clean_col_name = col.replace(' (yds)', '').replace(' (°)', '') + '_Clean'
            if 'Lateral' in col: df_clean['Lateral_Clean'] = df_clean[col].apply(parse_lr)
            else: df_clean[clean_col_name] = df_clean[col].apply(parse_lr)

    numeric_cols = ['Carry (yds)', 'Total (yds)', 'Ball (mph)', 'Club (mph)', 'Smash', 'Spin (rpm)', 'Height (ft)', 'AOA (°)', 'Launch V (°)']
    for col in numeric_cols:
        if col in df_clean.columns: df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    if 'Altitude (ft)' not in df_clean.columns: df_clean['Altitude (ft)'] = 0.0
    else:
        df_clean['Altitude (ft)'] = df_clean['Altitude (ft)'].astype(str).str.replace(' ft','').str.replace(',','')
        df_clean['Altitude (ft)'] = pd.to_numeric(df_clean['Altitude (ft)'], errors='coerce').fillna(0.0)

    df_clean['SL_Carry'] = df_clean['Carry (yds)'] / (1 + (df_clean['Altitude (ft)'] / 1000.0 * 0.011))
    df_clean['SL_Total'] = df_clean['Total (yds)'] / (1 + (df_clean['Altitude (ft)'] / 1000.0 * 0.011))
    return df_clean

def filter_outliers(df):
    df_filtered = pd.DataFrame()
    outlier_count = 0
    for club in df['club'].unique():
        club_data = df[df['club'] == club].copy()
        
        valid_physics = club_data[
            (club_data['Smash'] <= 1.58) & (club_data['Smash'] >= 1.0) &
            (club_data['Spin (rpm)'] > 500) & (club_data['Height (ft)'] > 8)
        ]
        dropped_physics = len(club_data) - len(valid_physics)
        if len(valid_physics) < 5:
            df_filtered = pd.concat([df_filtered, valid_physics])
            outlier_count += dropped_physics
            continue

        q1 = valid_physics['SL_Carry'].quantile(0.25)
        q3 = valid_physics['SL_Carry'].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - (1.5 * iqr) 
        upper = q3 + (3.0 * iqr)
        final_valid = valid_physics[(valid_physics['SL_Carry'] >= lower) & (valid_physics['SL_Carry'] <= upper)]
        outlier_count += (dropped_physics + (len(valid_physics) - len(final_valid)))
        df_filtered = pd.concat([df_filtered, final_valid])
        
    return df_filtered, outlier_count

def check_range(club_name, value, metric_idx, handicap):
    ranges = get_dynamic_ranges(club_name, handicap) 
    min_v, max_v = ranges[metric_idx]
    if min_v <= value <= max_v: return "Optimal ✅", "normal"
    elif value < min_v: return f"{value - min_v:.1f} (Low) ⚠️", "inverse"
    else: return f"+{value - max_v:.1f} (High) ⚠️", "inverse"

def get_coach_tip(metric_name, status, club):
    if "Optimal" in status: return None
    is_driver = "driver" in str(club).lower()
    if metric_name == "AoA":
        if "Low" in status and is_driver: return "To hit up on Driver, try tilting your trailing shoulder down at address."
        if "High" in status and not is_driver: return "To hit down on irons, ensure your weight shifts to the lead side before impact."
    if metric_name == "Launch":
        if "Low" in status: return "Launch is low. Check ball position (move forward) or if you are delofting the club."
        if "High" in status: return "Launch is high. You might be scooping. Try to keep hands ahead of the ball."
    if metric_name == "Spin":
        if "Low" in status: return "Spin is dangerously low. Strikes might be high on the face."
        if "High" in status: return "Spin is too high. Strikes might be low on the face or you are cutting across it."
    return None

def style_fig(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def calculate_sg_off_tee(row):
    dist_remaining = 400 - row['Norm_Total']
    if dist_remaining < 0: dist_remaining = 10
    abs_lat = abs(row['Lateral_Clean'])
    if abs_lat < 15: lie_penalty = 0 
    elif abs_lat < 30: lie_penalty = 0.3 
    else: lie_penalty = 1.1 
    strokes_from_dist = (dist_remaining / 100) + 2.0 
    sg = 4.10 - (1 + strokes_from_dist + lie_penalty)
    return sg

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("1. User Profile")
    
    profiles = list(st.session_state['profiles'].keys())
    selected_profile = st.selectbox("Active Golfer:", profiles, index=profiles.index(st.session_state['active_user']))
    
    if selected_profile != st.session_state['active_user']:
        st.session_state['active_user'] = selected_profile
        st.rerun()
        
    with st.expander("👤 Manage Profile"):
        new_name_input = st.text_input("Rename Current Profile:", value=active_user)
        if st.button("💾 Rename"):
            if new_name_input and new_name_input != active_user:
                if new_name_input in st.session_state['profiles']:
                    st.error("Name already exists!")
                else:
                    st.session_state['profiles'][new_name_input] = st.session_state['profiles'].pop(active_user)
                    st.session_state['active_user'] = new_name_input
                    st.success(f"Renamed to {new_name_input}")
                    st.rerun()
    
    new_create_name = st.text_input("New Profile Name", key="create_new_prof")
    if st.button("➕ Create New Profile", use_container_width=True):
        if new_create_name and new_create_name not in st.session_state['profiles']:
            st.session_state['profiles'][new_create_name] = {'df': pd.DataFrame(), 'bag': DEFAULT_LOFTS.copy()}
            st.success(f"Created {new_create_name}")
            st.rerun()

    st.markdown("---")
    st.header("2. Database Manager")
    with st.expander(f"📂 Manage Data: {active_user}", expanded=False):
        db_file = st.file_uploader("Restore 'mevo_db.csv'", type='csv', key='db_uploader')
        if db_file:
            if st.button("🔄 Restore Database", use_container_width=True):
                try:
                    restored = pd.read_csv(db_file)
                    if 'Date' in restored.columns: restored['Date'] = pd.to_datetime(restored['Date'])
                    
                    st.session_state['profiles'][active_user]['df'] = restored
                    if 'Ref Loft' in restored.columns:
                        latest_lofts = restored.drop_duplicates('club', keep='last').set_index('club')['Ref Loft'].to_dict()
                        st.session_state['profiles'][active_user]['bag'].update(latest_lofts)
                    
                    st.success(f"Restored data for {active_user}!")
                    st.rerun()
                except Exception as e: st.error(f"Error: {e}")
        
        if not master_df.empty:
            csv_data = master_df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Save Database", csv_data, f"{active_user}_mevo_db.csv", "text/csv", use_container_width=True)
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state['profiles'][active_user]['df'] = pd.DataFrame()
                st.rerun()

    st.header("3. Add Session")
    import_date = st.date_input("Date of Session")
    uploaded_files = st.file_uploader("Upload CSVs", accept_multiple_files=True, type='csv', key=f"uploader_{import_date}")
    
    if st.button("➕ Add to Database", use_container_width=True):
        if uploaded_files:
            new_data = []
            for f in uploaded_files:
                try:
                    current_filenames = master_df['Session'].unique() if not master_df.empty else []
                    if f.name.replace('.csv', '') in current_filenames:
                        st.toast(f"Skipped duplicate: {f.name}", icon="⚠️")
                        continue
                        
                    raw = pd.read_csv(f)
                    clean = clean_mevo_data(raw, f.name, import_date)
                    clean['Ref Loft'] = clean['club'].map(my_bag)
                    new_data.append(clean)
                except Exception as e: st.error(f"Error {f.name}: {e}")
            if new_data:
                batch_df = pd.concat(new_data, ignore_index=True)
                batch_df['Ref Loft'] = batch_df['Ref Loft'].fillna(30.0)
                st.session_state['profiles'][active_user]['df'] = pd.concat([master_df, batch_df], ignore_index=True)
                st.success(f"Added {len(batch_df)} shots to {active_user}!")
                st.rerun()

    st.markdown("---")
    
    # --- MY BAG CONFIG ---
    st.header("4. My Bag Setup")
    with st.expander(f"⚙️ Lofts: {active_user}"):
        selected_club = st.selectbox("Choose Club:", CLUB_SORT_ORDER, index=0)
        current_loft = my_bag.get(selected_club, DEFAULT_LOFTS.get(selected_club, 30.0))
        new_loft = st.number_input(f"Loft for {selected_club} (°)", value=float(current_loft), step=0.5, format="%.1f")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save", type="primary", use_container_width=True):
                st.session_state['profiles'][active_user]['bag'][selected_club] = new_loft
                st.toast(f"Saved for {active_user}", icon="✅")
        with c2:
            if st.button("🔄 Reset", type="secondary", use_container_width=True):
                st.session_state['profiles'][active_user]['bag'] = DEFAULT_LOFTS.copy()
                st.rerun()
            
        st.markdown("---")
        st.caption("Current Configuration:")
        bag_df = pd.DataFrame(list(my_bag.items()), columns=['Club', 'Loft'])
        bag_df['SortIndex'] = bag_df['Club'].apply(lambda x: CLUB_SORT_ORDER.index(x) if x in CLUB_SORT_ORDER else 99)
        bag_df = bag_df.sort_values('SortIndex').drop(columns=['SortIndex'])
        st.dataframe(bag_df, hide_index=True, width="stretch", height=200)
            
    # --- ENVIRONMENT CONFIG ---
    st.markdown("---")
    st.header("5. Environment")
    with st.expander("🌤️ Normalization", expanded=True):
        sim_temp = st.slider("Temperature (°F)", 30, 110, 75, help="Adjust distances to this temp.")
        play_alt = st.number_input("Altitude (ft)", value=0, step=500, help="Adjust for course altitude.")
        ball_type = st.selectbox("Ball Type", ["Premium (100%)", "Economy (98%)", "Range - Hard (95%)", "Range - Limited (85%)"])
        
        temp_factor = 1 + ((sim_temp - 70) * 0.001) 
        alt_factor = 1 + (play_alt / 1000.0 * 0.011) 
        ball_map = {"Premium (100%)": 1.0, "Economy (98%)": 0.98, "Range - Hard (95%)": 0.95, "Range - Limited (85%)": 0.85}
        ball_factor = ball_map[ball_type]
        
        total_norm_factor = temp_factor * alt_factor * ball_factor
        
        st.caption(f"**Total Adjustment: {total_norm_factor:.3f}x**")
        if total_norm_factor > 1:
            st.success(f"Projecting: +{((total_norm_factor-1)*100):.1f}% Gain")
        else:
            st.error(f"Projecting: {((total_norm_factor-1)*100):.1f}% Loss")

    # --- PLAYER CONFIG ---
    st.markdown("---")
    st.header("6. Player Config")
    env_mode = st.radio("Filter:", ["All", "Outdoor Only", "Indoor Only"], index=0)
    handicap = st.number_input("Handicap", 0, 54, 15)
    smash_cap = st.slider("Max Smash Cap", 1.40, 1.65, 1.52, 0.01)
    remove_bad_shots = st.checkbox("Auto-Clean Outliers", value=True)

    # SUMMARY STATS (SIDEBAR)
    if not master_df.empty:
        st.markdown("---")
        tot_shots = len(master_df)
        tot_sess = master_df['Date'].nunique()
        st.markdown(f"""
        <div class="stat-card-container">
            <div class="stat-card"><p class="stat-value">{tot_shots}</p><p class="stat-label">Shots</p></div>
            <div class="stat-card"><p class="stat-value" style="color: #FF4081;">{tot_sess}</p><p class="stat-label">Sessions</p></div>
        </div>
        """, unsafe_allow_html=True)

# --- 4. MAIN APP LOGIC ---
if not master_df.empty:
    st.title(f"⛳ Analytics: {active_user}")
    
    filtered_df = master_df.copy()
    if env_mode == "Outdoor Only" and 'Mode' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Mode'].str.contains("Outdoor", case=False, na=False)]
    elif env_mode == "Indoor Only" and 'Mode' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Mode'].str.contains("Indoor", case=False, na=False)]
    
    filtered_df = filtered_df[filtered_df['Smash'] <= smash_cap]

    if remove_bad_shots:
        filtered_df, dropped_count = filter_outliers(filtered_df)
        if dropped_count > 0: st.toast(f"Cleaned {dropped_count} outliers", icon="🧹")

    # APPLY NORMALIZATION
    filtered_df['Norm_Carry'] = filtered_df['SL_Carry'] * total_norm_factor
    filtered_df['Norm_Total'] = filtered_df['SL_Total'] * total_norm_factor

    # --- TABS ---
    tab_home, tab_bag, tab_acc, tab_gap, tab_sg, tab_target, tab_time, tab_mech, tab_comp, tab_faq = st.tabs(["🏠 Home", "🎒 My Bag", "🎯 Accuracy", "📏 Gapping", "🏆 Strokes Gained", "🎯 Target", "📈 Timeline", "🔬 Mechanics", "⚔️ Compare", "❓ FAQ"])

    # ================= TAB: HOME DASHBOARD =================
    with tab_home:
        total_shots = len(filtered_df)
        total_sessions = filtered_df['Date'].nunique()
        
        driver_df = filtered_df[filtered_df['club'] == 'Driver']
        if not driver_df.empty:
            longest_drive = get_smart_max(driver_df['Norm_Carry'], driver_df)
            fastest_ball = driver_df['Ball (mph)'].max()
        else:
            longest_drive = 0
            fastest_ball = filtered_df['Ball (mph)'].max() if not filtered_df.empty else 0
            
        if not filtered_df.empty:
            fav_club = filtered_df['club'].mode()[0]
            fav_club_count = len(filtered_df[filtered_df['club'] == fav_club])
        else:
            fav_club = "-"
            fav_club_count = 0

        c_h1, c_h2, c_h3, c_h4 = st.columns(4)
        def render_hero(col, title, value, sub):
            col.markdown(f"""<div class="hero-card"><div class="hero-title">{title}</div><div class="hero-metric">{value}</div><div class="hero-sub">{sub}</div></div>""", unsafe_allow_html=True)

        render_hero(c_h1, "Longest Drive", f"{longest_drive:.0f}<span style='font-size:20px'>y</span>", f"@{sim_temp}°F / {ball_type.split()[0]}")
        render_hero(c_h2, "Ball Speed Record", f"{fastest_ball:.0f}<span style='font-size:20px'>mph</span>", "All-Time Max")
        render_hero(c_h3, "Total Volume", f"{total_shots}", f"Across {total_sessions} Sessions")
        render_hero(c_h4, "Favorite Club", f"{fav_club}", f"{fav_club_count} Shots Recorded")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 Recent Activity")
        if not filtered_df.empty:
            activity = filtered_df.groupby('Date').size().reset_index(name='Shots')
            fig_act = px.bar(activity, x='Date', y='Shots', title="")
            fig_act.update_traces(marker_color='#4DD0E1')
            fig_act.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig_act, use_container_width=True)

    # ================= TAB: MY BAG =================
    with tab_bag:
        st.subheader(f"🎒 My Bag & Yardages (Normalized to {sim_temp}°F)")
        bag_data = []
        for club in filtered_df['club'].unique():
            subset = filtered_df[filtered_df['club'] == club]
            s_max = get_smart_max(subset['Norm_Carry'], subset)
            
            # Stock Range (20th - 80th Percentile)
            p20 = subset['Norm_Carry'].quantile(0.20)
            p80 = subset['Norm_Carry'].quantile(0.80)
            
            bag_data.append({
                'Club': club, 'Norm_Carry': subset['Norm_Carry'].mean(), 'Norm_Total': subset['Norm_Total'].mean(),
                'Ball Speed': subset['Ball (mph)'].mean(), 'Max Carry': s_max, 'Count': len(subset),
                'Range_Min': p20, 'Range_Max': p80
            })
        
        bag_stats = pd.DataFrame(bag_data).set_index('Club')
        bag_stats['SortIndex'] = bag_stats.index.map(lambda x: CLUB_SORT_ORDER.index(x) if x in CLUB_SORT_ORDER else 99)
        bag_stats = bag_stats.sort_values('SortIndex')
        
        st.write("---")
        cols = st.columns(4)
        for i, (index, row) in enumerate(bag_stats.iterrows()):
            with cols[i % 4]:
                st.markdown(f"""
                <div style="background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #444; margin-bottom: 10px;">
                    <h3 style="margin:0; color: #4DD0E1;">{index}</h3>
                    <h2 style="margin:0; font-size: 32px; color: #FFF;">{row['Norm_Carry']:.0f}<span style="font-size:16px; color:#888"> yds</span></h2>
                    <div style="font-size: 14px; color: #00E5FF; margin-bottom: 5px; font-weight: 500;">Range: {row['Range_Min']:.0f} - {row['Range_Max']:.0f}</div>
                    <hr style="border-color: #444; margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #888;">
                        <span>Speed: {row['Ball Speed']:.0f}</span>
                        <span style="color: #FFD700;">Pot: {row['Max Carry']:.0f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ================= TAB: ACCURACY =================
    with tab_acc:
        if len(filtered_df) > 0:
            avail_clubs = [c for c in CLUB_SORT_ORDER if c in filtered_df['club'].unique()]
            selected_club = st.selectbox("Select Club", avail_clubs + [c for c in filtered_df['club'].unique() if c not in avail_clubs], key='t1_club')
            subset = filtered_df[filtered_df['club'] == selected_club]
            
            if len(subset) > 0:
                is_scoring = any(x in str(selected_club).lower() for x in ['8','9','p','w','s','l','g'])
                target_val = 5.0 + (handicap * 0.4) if is_scoring else 15.0 + (handicap * 0.8)
                lat_mean = subset['Lateral_Clean'].mean()
                tendency_dir = "Right ➡️" if lat_mean > 0 else "Left ⬅️"
                
                c1, c2, c3, c4 = st.columns(4)
                on_target = len(subset[abs(subset['Lateral_Clean']) <= target_val]) / len(subset) * 100
                c1.metric("Accuracy Score", f"{on_target:.0f}%", f"Target: ±{target_val:.1f}y")
                c2.metric("Tendency", f"{abs(lat_mean):.1f}y", tendency_dir)
                c3.metric("Avg Carry", f"{subset['Norm_Carry'].mean():.1f}")
                c4.metric("Ball Speed", f"{subset['Ball (mph)'].mean():.1f}")

                c_chart1, c_chart2 = st.columns([3, 1])
                with c_chart1:
                    subset['Shape'] = np.where(subset['Lateral_Clean'] > 0, 'Fade (R)', 'Draw (L)')
                    fig = px.scatter(subset, x='Lateral_Clean', y='Norm_Carry', color='Shape',
                        color_discrete_map={'Fade (R)': '#00E5FF', 'Draw (L)': '#FF4081'},
                        hover_data=['Date', 'Smash', 'Spin (rpm)'], title=f"Dispersion: {selected_club}")
                    fig.add_shape(type="rect", x0=-target_val, y0=subset['Norm_Carry'].min()-10, x1=target_val, y1=subset['Norm_Carry'].max()+10,
                        line_color="#00E676", fillcolor="#00E676", opacity=0.1)
                    fig.add_vline(x=0, line_color="white", opacity=0.2)
                    st.plotly_chart(style_fig(fig), use_container_width=True)
                with c_chart2:
                    counts = subset['Shot Type'].value_counts().reset_index()
                    counts.columns = ['Type', 'Count']
                    fig_pie = px.pie(counts, values='Count', names='Type', hole=0.5, color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(style_fig(fig_pie), use_container_width=True)
        else: st.info("No data available.")

    # ================= TAB: GAPPING =================
    with tab_gap:
        if len(filtered_df) > 0:
            st.subheader("🎒 Bag Gapping (Normalized)")
            with st.expander("ℹ️ How to read this chart?"):
                st.markdown("""
                * **The Box:** Represents your "Consistency Zone" (middle 50% of shots). A smaller box is better.
                * **The Line inside:** Your Median (most typical) carry distance.
                * **The Whiskers (Lines):** Your absolute range (Longest vs Shortest), excluding outliers.
                """)
            filtered_df['SortIndex'] = filtered_df['club'].map(lambda x: CLUB_SORT_ORDER.index(x) if x in CLUB_SORT_ORDER else 99)
            fig = px.box(filtered_df.sort_values('SortIndex'), x='club', y='Norm_Carry', color='club', points="all")
            st.plotly_chart(style_fig(fig), use_container_width=True)

    # ================= TAB: STROKES GAINED =================
    with tab_sg:
        st.subheader("🏆 Strokes Gained Calculator (Estimator)")
        sg_mode = st.radio("Mode:", ["Off The Tee (Driver)", "Approach (Irons)"], horizontal=True)
        
        if sg_mode == "Off The Tee (Driver)":
            driver_data = filtered_df[filtered_df['club'] == 'Driver'].copy()
            if len(driver_data) > 0:
                driver_data['SG_OTT'] = driver_data.apply(calculate_sg_off_tee, axis=1)
                avg_sg = driver_data['SG_OTT'].mean()
                c_sg1, c_sg2 = st.columns(2)
                c_sg1.markdown(f"""<div class="sg-box"><h3 style="color:#4DD0E1">SG: Off The Tee</h3><h1 style="color:#FFF">{avg_sg:+.2f}</h1><p style="color:#BBB">vs 15 HCP Baseline</p></div>""", unsafe_allow_html=True)
                fig_sg = px.histogram(driver_data, x="SG_OTT", nbins=20, title="Distribution of Driver Performance")
                fig_sg.add_vline(x=0, line_color="white", annotation_text="Baseline")
                st.plotly_chart(style_fig(fig_sg), use_container_width=True)
            else: st.info("No Driver data found.")
        else:
            st.info("ℹ️ SG: Approach compares your consistency to a Scratch Golfer's dispersion.")
            iron_clubs = [c for c in filtered_df['club'].unique() if "Iron" in c or "Wedge" in c]
            if iron_clubs:
                sel_iron = st.selectbox("Select Iron:", iron_clubs)
                iron_data = filtered_df[filtered_df['club'] == sel_iron]
                avg_dist = iron_data['Norm_Carry'].mean()
                dist_tol = avg_dist * 0.05
                lat_tol = avg_dist * np.tan(np.radians(4))
                good_shots = iron_data[(abs(iron_data['Norm_Carry'] - avg_dist) < dist_tol) & (abs(iron_data['Lateral_Clean']) < lat_tol)]
                score = len(good_shots) / len(iron_data) * 100
                c_app1, c_app2 = st.columns(2)
                c_app1.metric("Scratch Consistency", f"{score:.0f}%", "Shots inside Scratch zone")
                fig_app = px.scatter(iron_data, x="Lateral_Clean", y="Norm_Carry", title=f"{sel_iron} vs Scratch Zone")
                fig_app.add_shape(type="circle", x0=-lat_tol, y0=avg_dist-dist_tol, x1=lat_tol, y1=avg_dist+dist_tol, line_color="#00E676", fillcolor="#00E676", opacity=0.2)
                st.plotly_chart(style_fig(fig_app), use_container_width=True)
            else: st.warning("No Iron/Wedge data found.")

    # ================= TAB: TARGET MODE =================
    with tab_target:
        st.subheader("🎯 Target Practice Challenge")
        c_tgt1, c_tgt2, c_tgt3 = st.columns(3)
        available_sessions = filtered_df['Session'].unique()
        with c_tgt1: 
            tgt_session = st.selectbox("1. Select Session", available_sessions)
        
        session_data = filtered_df[filtered_df['Session'] == tgt_session]
        available_clubs_sess = session_data['club'].unique()
        with c_tgt2:
            tgt_club = st.selectbox("2. Select Club", available_clubs_sess)
            
        with c_tgt3:
            tgt_dist = st.number_input("3. Target Distance (yds)", value=150, step=5)
            
        target_subset = session_data[session_data['club'] == tgt_club].copy()
        
        if not target_subset.empty:
            target_subset['Dist_Err'] = abs(target_subset['Norm_Carry'] - tgt_dist)
            target_subset['Lat_Err'] = abs(target_subset['Lateral_Clean'])
            target_subset['Total_Err'] = target_subset['Dist_Err'] + target_subset['Lat_Err']
            target_subset['Score'] = np.maximum(0, 100 - target_subset['Total_Err'])
            
            avg_score = target_subset['Score'].mean()
            best_shot = target_subset['Score'].max()
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Session Score", f"{avg_score:.0f} / 100")
            m2.metric("Best Shot", f"{best_shot:.0f} / 100")
            m3.metric("Shots Scored", len(target_subset))
            
            fig_tgt = go.Figure()
            for r, color in zip([10, 20, 30], ['green', 'yellow', 'red']):
                fig_tgt.add_shape(type="circle", x0=-r, y0=tgt_dist-r, x1=r, y1=tgt_dist+r, line_color=color, opacity=0.3)
            
            fig_tgt.add_trace(go.Scatter(
                x=target_subset['Lateral_Clean'], y=target_subset['Norm_Carry'],
                mode='markers', marker=dict(size=12, color=target_subset['Score'], colorscale='RdYlGn', showscale=True),
                text=target_subset['Score'].apply(lambda x: f"Score: {x:.0f}"), hoverinfo='text+x+y'
            ))
            fig_tgt.add_trace(go.Scatter(x=[0], y=[tgt_dist], mode='markers', marker=dict(symbol='cross', size=15, color='white'), name='Target'))
            fig_tgt.update_layout(title=f"Target: {tgt_dist}y | Club: {tgt_club}", xaxis_title="Left <-> Right (yds)", yaxis_title="Carry Distance (yds)", yaxis=dict(range=[tgt_dist-50, tgt_dist+50]), xaxis=dict(range=[-40, 40]), showlegend=False, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=600)
            st.plotly_chart(fig_tgt, use_container_width=True)
            
            st.caption("Detailed Shot Scoring:")
            st.dataframe(target_subset[['Score', 'Norm_Carry', 'Lateral_Clean', 'Dist_Err', 'Lat_Err']].sort_values('Score', ascending=False).style.format("{:.1f}"), width="stretch")
        else:
            st.info("No shots found for this club in this session.")

    # ================= TAB: TIMELINE =================
    with tab_time:
        if len(filtered_df) > 0:
            st.subheader("📈 Timeline")
            c_t1, c_t2 = st.columns(2)
            avail_clubs = [c for c in CLUB_SORT_ORDER if c in filtered_df['club'].unique()]
            with c_t1: t_club = st.selectbox("Club", avail_clubs, key='t_club')
            with c_t2: metric = st.selectbox("Metric", ['Ball (mph)', 'Norm_Carry', 'Club (mph)', 'Smash'])
            
            if 'Date' in filtered_df.columns:
                club_data = filtered_df[filtered_df['club'] == t_club]
                trend = club_data.groupby('Date')[metric].agg(['mean', 'std']).reset_index().sort_values('Date')
                trend['std'] = trend['std'].fillna(0)
                trend['upper'] = trend['mean'] + trend['std']
                trend['lower'] = trend['mean'] - trend['std']
                
                if len(trend) > 1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=trend['Date'], y=trend['upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                    fig.add_trace(go.Scatter(x=trend['Date'], y=trend['lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 230, 118, 0.2)', showlegend=False, hoverinfo='skip'))
                    fig.add_trace(go.Scatter(x=trend['Date'], y=trend['mean'], mode='lines+markers', name='Average', line=dict(color='#00E676', width=3), marker=dict(size=8, color='#00E676')))
                    fig.update_layout(title=f"{t_club} Progress: Average ± Consistency", yaxis_title=metric, hovermode="x unified", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("ℹ️ The shaded green area represents your consistency (Standard Deviation). A narrower band means more consistent performance.")
                else: 
                    st.info(f"Not enough sessions to show a trend for {t_club} yet.")
            else: 
                st.warning("No Date info.")

    # ================= TAB: MECHANICS =================
    with tab_mech:
        if len(filtered_df) > 0:
            st.subheader("🔬 Swing Mechanics")
            c_sel1, c_sel2 = st.columns([2,1])
            avail_clubs = [c for c in CLUB_SORT_ORDER if c in filtered_df['club'].unique()]
            with c_sel1: mech_club = st.selectbox("Analyze Club", avail_clubs, key='m_club')
            with c_sel2: 
                curr_loft = my_bag.get(mech_club, 30.0)
                st.metric("Bag Loft", f"{curr_loft}°")

            mech_data = filtered_df[filtered_df['club'] == mech_club]
            
            col_m1, col_m2, col_m3 = st.columns(3)
            def display_mech_metric(col, label, key, idx):
                if key in mech_data.columns:
                    val = mech_data[key].mean()
                    status, color = check_range(mech_club, val, idx, handicap)
                    col.metric(label, f"{val:.1f}", status, delta_color=color)
                    tip = get_coach_tip(label.split()[0], status, mech_club)
                    if tip: st.markdown(f"<div class='coach-box'>💡 <b>Coach:</b> {tip}</div>", unsafe_allow_html=True)

            with col_m1: display_mech_metric(col_m1, "AoA (°)", 'AOA (°)', 0)
            with col_m2: display_mech_metric(col_m2, "Launch (°)", 'Launch V (°)', 1)
            with col_m3: display_mech_metric(col_m3, "Spin (rpm)", 'Spin (rpm)', 2)
            
            if 'Smash' in mech_data.columns:
                smash_val = mech_data['Smash'].mean()
                if smash_val < 1.30 and "driver" in str(mech_club).lower():
                    st.markdown(f"<div class='coach-box'>💡 <b>Coach:</b> Low Smash Factor ({smash_val:.2f}). You might be striking the heel or toe.</div>", unsafe_allow_html=True)

            st.markdown("---")
            col_chart_m1, col_chart_m2 = st.columns([2, 1])
            with col_chart_m1:
                if 'Height (ft)' in mech_data.columns:
                    st.markdown("#### ✈️ Trajectory Window")
                    fig_traj = px.scatter(mech_data, x='Norm_Carry', y='Height (ft)', color='Session')
                    fig_traj.add_shape(type="rect", x0=mech_data['Norm_Carry'].min(), y0=80, x1=mech_data['Norm_Carry'].max(), y1=110, line=dict(color="Gold", width=0), fillcolor="Gold", opacity=0.1)
                    st.plotly_chart(style_fig(fig_traj), use_container_width=True)
            with col_chart_m2:
                if 'Club Path_Clean' in mech_data.columns and 'FTP_Clean' in mech_data.columns:
                    st.markdown("#### ↩️ Shape Control")
                    fig_path = px.scatter(mech_data, x='Club Path_Clean', y='FTP_Clean', color='Lateral_Clean', color_continuous_scale='RdBu_r')
                    fig_path.add_hline(y=0, line_color="white", opacity=0.2)
                    fig_path.add_vline(x=0, line_color="white", opacity=0.2)
                    st.plotly_chart(style_fig(fig_path), use_container_width=True)

    # ================= TAB: COMPARISON =================
    with tab_comp:
        st.subheader("⚔️ Comparison Lab")
        avail_clubs = [c for c in CLUB_SORT_ORDER if c in filtered_df['club'].unique()]
        comp_club = st.selectbox("Select Club to Compare", avail_clubs, key='c_club')
        
        club_data = filtered_df[filtered_df['club'] == comp_club].copy()
        if 'Date' in club_data.columns: club_data['SessionLabel'] = club_data['Date'].dt.strftime('%Y-%m-%d') + ": " + club_data['Session']
        else: club_data['SessionLabel'] = club_data['Session']
            
        unique_sessions = club_data['SessionLabel'].unique()
        if len(unique_sessions) >= 2:
            c1, c2 = st.columns(2)
            with c1: sess_a = st.selectbox("Session A", unique_sessions, index=0)
            with c2: sess_b = st.selectbox("Session B", unique_sessions, index=1)
                
            if sess_a != sess_b:
                data_a = club_data[club_data['SessionLabel'] == sess_a]
                data_b = club_data[club_data['SessionLabel'] == sess_b]
                
                # Determine Winner
                a_carry = data_a['Norm_Carry'].mean()
                b_carry = data_b['Norm_Carry'].mean()
                a_acc = data_a['Lateral_Clean'].abs().mean()
                b_acc = data_b['Lateral_Clean'].abs().mean()
                
                win_carry = "🏆 Session B" if b_carry > a_carry else "🏆 Session A"
                win_acc = "🏆 Session B" if b_acc < a_acc else "🏆 Session A"
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Carry Winner", win_carry, f"Diff: {abs(b_carry - a_carry):.1f}y")
                m2.metric("Accuracy Winner", win_acc, f"Diff: {abs(b_acc - a_acc):.1f}y")
                m3.metric("Consistency (Std Dev)", f"A: {data_a['Norm_Carry'].std():.1f} vs B: {data_b['Norm_Carry'].std():.1f}", delta_color="off")
                
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=data_a['Norm_Carry'], name='Session A', opacity=0.75, marker_color='#FF4081'))
                fig_hist.add_trace(go.Histogram(x=data_b['Norm_Carry'], name='Session B', opacity=0.75, marker_color='#00E5FF'))
                fig_hist.update_layout(barmode='overlay', title=f"Carry Distance Distribution")
                st.plotly_chart(style_fig(fig_hist), use_container_width=True)
            else: st.warning("Select different sessions.")
        else: st.warning("Need 2+ sessions.")

    # ================= TAB: FAQ =================
    with tab_faq:
        st.subheader("❓ FAQ & Help")
        with st.expander("🧹 What is 'Auto-Clean Outliers'?", expanded=False):
            st.markdown("We use the **IQR method** combined with physics checks (Smash/Spin) to strip out misreads and duffs.")
        with st.expander("🌊 How does 'Sea Level' Normalization work?", expanded=False):
            st.markdown("We apply a **1.1% per 1,000 ft** correction to simulate Sea Level performance.")
        with st.expander("⚙️ Why do I need to set 'My Bag' lofts?", expanded=False):
            st.markdown("Setting accurate lofts allows the 'Mechanics' tab to give you specific advice on Launch Angle and Spin.")
        with st.expander("📊 Data Dictionary: What do these numbers mean?", expanded=False):
            st.markdown("""
            * **Smash Factor:** Efficiency (Ball Speed ÷ Club Speed).
            * **Spin Axis:** Positive (+) = Fade/Slice. Negative (-) = Draw/Hook.
            * **AoA:** Hitting Up (+) or Down (-).
            """)

else:
    # --- WELCOME SCREEN ---
    st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <h1 style="font-size: 60px; font-weight: 700; background: -webkit-linear-gradient(45deg, #00E5FF, #FF4081); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Homegrown FS Pro Analytics
        </h1>
        <p style="font-size: 20px; color: #B0B3B8;">Turn your <b>Mevo+ and Mevo Gen 2</b> data into Tour-level insights.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.info("🆕 **New User?**")
        st.markdown("**Start Fresh:** Open **Sidebar > My Bag Setup** to configure lofts, then **Add Session**.")
    with c2:
        st.warning("💾 **Returning User?**")
        st.markdown("**Continue:** Open **Sidebar > Database Manager** and upload your `mevo_db.csv`.")

    st.markdown("---")
    
    # Feature Grid
    c_f1, c_f2, c_f3 = st.columns(3)
    with c_f1: st.markdown('<div class="feature-card"><h3>🎯 Precision Targeting</h3><p style="color:#888">Visualize dispersion with dynamic target lanes.</p></div>', unsafe_allow_html=True)
    with c_f2: st.markdown('<div class="feature-card"><h3>🎒 Smart Gapping</h3><p style="color:#888">Altitude-adjusted ranges for every club.</p></div>', unsafe_allow_html=True)
    with c_f3: st.markdown('<div class="feature-card"><h3>📈 Deep Trends</h3><p style="color:#888">Track consistency scores and gains.</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("❓ FAQ & Help")
    with st.expander("🧹 What is 'Auto-Clean Outliers'?", expanded=False):
        st.markdown("We use the **IQR method** to strip out misreads and duffs.")
    with st.expander("🌊 How does 'Sea Level' Normalization work?", expanded=False):
        st.markdown("We apply a **1.1% per 1,000 ft** correction to simulate Sea Level performance.")
