import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

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
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {
        color: #FAFAFA !important;
    }
    section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] select {
        background-color: #262730 !important; color: #FAFAFA !important; border: 1px solid #444 !important;
    }

    /* 3. METRIC CARDS */
    .metric-card {
        background: linear-gradient(145deg, #1E222B, #262730);
        border: 1px solid #444;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-val { font-size: 28px; font-weight: 700; color: #4DD0E1; margin: 0; }
    .metric-lbl { font-size: 13px; text-transform: uppercase; color: #B0B3B8; letter-spacing: 1px; }
    .metric-sub { font-size: 11px; color: #FF4081; margin-top: 4px; }

    /* 4. DASHBOARD CARDS (Hero) */
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
    .hero-card:hover { transform: translateY(-5px); border-color: #4DD0E1; }
    .hero-title { font-size: 14px; text-transform: uppercase; letter-spacing: 1.5px; color: #B0B3B8; margin-bottom: 10px; }
    .hero-metric { font-size: 36px; font-weight: 800; color: #FAFAFA; margin: 0; }
    .hero-sub { font-size: 12px; color: #00E5FF; margin-top: 5px; }

    /* 5. TABS & GENERAL */
    div[data-testid="stTabs"] button[aria-selected="false"] { color: #B0B3B8 !important; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: #FAFAFA !important; border-top-color: #4DD0E1 !important; }
    div[data-testid="stExpander"] { background-color: #1E222B !important; border: 1px solid #444; }
    div[data-testid="stMetricLabel"] label { color: #B0B3B8 !important; }
    div[data-testid="stMetricValue"] { color: #4DD0E1 !important; }
    
    /* 6. EFFICIENCY BAR */
    .eff-container { background-color: #333; border-radius: 10px; padding: 5px; margin-top: 10px; }
    .eff-bar-fill { height: 10px; background: linear-gradient(90deg, #FF4081, #00E5FF); border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- 1. SESSION STATE & DEFAULTS ---

DEFAULT_LOFTS = {
    'Driver': 10.5, '3 Wood': 15.0, '5 Wood': 18.0, 'Hybrid': 21.0, 
    '3 Iron': 21.0, '4 Iron': 24.0, '5 Iron': 27.0, '6 Iron': 30.0, 
    '7 Iron': 34.0, '8 Iron': 38.0, '9 Iron': 42.0, 'PW': 46.0, 
    'GW': 50.0, 'SW': 54.0, 'LW': 58.0
}
CLUB_SORT_ORDER = ['Driver', '3 Wood', '5 Wood', '7 Wood', 'Hybrid', '2 Iron', '3 Iron', '4 Iron', '5 Iron', '6 Iron', '7 Iron', '8 Iron', '9 Iron', 'PW', 'GW', 'SW', 'LW']

if 'profiles' not in st.session_state:
    st.session_state['profiles'] = {'Default Golfer': {'df': pd.DataFrame(), 'bag': DEFAULT_LOFTS.copy()}}
if 'active_user' not in st.session_state:
    st.session_state['active_user'] = 'Default Golfer'

active_user = st.session_state['active_user']
master_df = st.session_state['profiles'][active_user]['df']
my_bag = st.session_state['profiles'][active_user]['bag']

# --- 2. PHYSICS & HELPER FUNCTIONS ---

def calculate_optimal_carry(club_speed, loft, benchmark="Scratch"):
    """
    Interpolated Efficiency Curve (Yards/MPH vs Loft).
    """
    if benchmark == "Tour Pro":
        x_points = [9, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        y_points = [2.90, 2.75, 2.60, 2.50, 2.40, 2.30, 2.20, 2.10, 2.00, 1.85, 1.70]
    else: # Scratch / High Amateur (Realistic)
        x_points = [9, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        y_points = [2.75, 2.65, 2.55, 2.48, 2.40, 2.32, 2.20, 2.05, 1.85, 1.65, 1.50]
        
    efficiency_factor = np.interp(loft, x_points, y_points)
    return club_speed * efficiency_factor

def get_smart_max(series, df_subset):
    valid = df_subset.loc[series.index]
    clean = valid[(valid['Smash'] <= 1.58) & (valid['Smash'] >= 1.0) & (valid['Spin (rpm)'] > 500) & (valid['Height (ft)'] > 8)]
    if clean.empty: return series.max()
    col_to_use = 'Norm_Carry' if 'Norm_Carry' in clean.columns else 'SL_Carry'
    return clean.loc[clean[col_to_use].idxmax(), col_to_use]

@st.cache_data
def clean_mevo_data(df, filename, selected_date):
    df_clean = df[df['Shot'].astype(str).str.isdigit()].copy()
    df_clean['Session'] = filename.replace('.csv', '')
    
    # Date Handling
    if 'Date' in df_clean.columns:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce').fillna(pd.to_datetime(selected_date))
    else:
        df_clean['Date'] = pd.to_datetime(selected_date)
    
    # Vectorized Lateral Parsing
    if 'Lateral (yds)' in df_clean.columns:
        lat_str = df_clean['Lateral (yds)'].astype(str).str.upper()
        is_left = lat_str.str.contains('L')
        nums = lat_str.str.extract(r'(\d+\.?\d*)')[0].astype(float).fillna(0.0)
        df_clean['Lateral_Clean'] = np.where(is_left, -nums, nums)
    else:
        df_clean['Lateral_Clean'] = 0.0

    # Numeric Conversion
    cols = ['Carry (yds)', 'Total (yds)', 'Ball (mph)', 'Club (mph)', 'Smash', 'Spin (rpm)', 'Height (ft)', 'AOA (°)', 'Launch V (°)']
    for c in cols:
        if c in df_clean.columns: df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')
    
    # Altitude Normalization (Base)
    if 'Altitude (ft)' in df_clean.columns:
        df_clean['Altitude (ft)'] = pd.to_numeric(df_clean['Altitude (ft)'].astype(str).str.replace(',','').str.replace(' ft',''), errors='coerce').fillna(0.0)
    else:
        df_clean['Altitude (ft)'] = 0.0

    df_clean['SL_Carry'] = df_clean['Carry (yds)'] / (1 + (df_clean['Altitude (ft)'] / 1000.0 * 0.011))
    df_clean['SL_Total'] = df_clean['Total (yds)'] / (1 + (df_clean['Altitude (ft)'] / 1000.0 * 0.011))
    
    return df_clean

@st.cache_data
def filter_dataset(df, mode, smash_limit):
    """
    Centralized Filter: Physics -> Smash Cap -> IQR
    """
    # 1. Physics Check
    mask = (
        (df['Smash'] <= 1.58) & (df['Smash'] >= 1.0) &
        (df['Spin (rpm)'] > 500) & (df['Height (ft)'] > 5) &
        (df['Smash'] <= smash_limit) # User setting
    )
    df_clean = df[mask].copy()
    
    # 2. IQR Check (Per Club)
    if not df_clean.empty and mode == "Auto-Clean":
        groups = df_clean.groupby('club')['SL_Carry']
        Q1 = groups.transform(lambda x: x.quantile(0.25))
        Q3 = groups.transform(lambda x: x.quantile(0.75))
        IQR = Q3 - Q1
        mask_iqr = (df_clean['SL_Carry'] >= (Q1 - 1.5 * IQR)) & (df_clean['SL_Carry'] <= (Q3 + 3.0 * IQR))
        df_clean = df_clean[mask_iqr].copy()
        
    return df_clean

def check_range(club_name, value, metric_idx, handicap):
    # Retrieve current bag from session state safely
    current_bag = st.session_state['profiles'][st.session_state['active_user']]['bag']
    c_lower = str(club_name).lower()
    user_loft = current_bag.get(club_name, 30.0)
    tolerance = handicap * 0.1
    launch_help = 0 if handicap < 5 else (1.0 if handicap < 15 else 2.0)

    # Simplified Dynamic Ranges
    if 'driver' in c_lower:
        aoa = (-2.0, 5.0) 
        launch = (10.0 + launch_help, 16.0 + launch_help)
        spin = (1800, 2800)
    elif 'wood' in c_lower or 'hybrid' in c_lower:
        aoa = (-4.0, 1.0)
        launch = (user_loft * 0.7 - 2.0, user_loft * 0.7 + 2.0)
        spin = (user_loft * 180, user_loft * 250)
    else:
        # Irons
        aoa = (-6.0, -1.0)
        launch = (user_loft * 0.5 - 2.0, user_loft * 0.5 + 2.0)
        spin = (user_loft * 180, user_loft * 220)
        
    ranges = [aoa, launch, spin]
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
    return None

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("1. User Profile")
    profs = list(st.session_state['profiles'].keys())
    curr_prof = st.selectbox("Active Golfer:", profs, index=profs.index(active_user))
    if curr_prof != active_user:
        st.session_state['active_user'] = curr_prof
        st.rerun()
        
    with st.expander("👤 Manage Profile"):
        new_name = st.text_input("Rename:", value=active_user)
        if st.button("💾 Save Name") and new_name != active_user:
            st.session_state['profiles'][new_name] = st.session_state['profiles'].pop(active_user)
            st.session_state['active_user'] = new_name
            st.rerun()
        new_prof_create = st.text_input("New Profile:", key="new_prof_create")
        if st.button("➕ Create"):
            st.session_state['profiles'][new_prof_create] = {'df': pd.DataFrame(), 'bag': DEFAULT_LOFTS.copy()}
            st.rerun()
    
    st.markdown("---")
    
    # --- DATE FILTERING (NEW) ---
    st.header("2. Timeframe")
    date_filter = st.selectbox(
        "Select Range:", 
        ["All Time", "Last Session", "Last 3 Sessions", "Last 5 Sessions", "Last 30 Days", "Year to Date"]
    )
    
    st.markdown("---")
    st.header("3. Data Manager")
    with st.expander("📂 Import / Export"):
        up_file = st.file_uploader("Upload CSVs", accept_multiple_files=True)
        import_date = st.date_input("Session Date")
        if st.button("➕ Add Session") and up_file:
            new_list = []
            for f in up_file:
                try:
                    # Dup check
                    if not master_df.empty and f.name.replace('.csv','') in master_df['Session'].unique(): continue
                    raw = pd.read_csv(f)
                    clean = clean_mevo_data(raw, f.name, import_date)
                    clean['Ref Loft'] = clean['club'].map(my_bag)
                    new_list.append(clean)
                except: pass
            if new_list:
                st.session_state['profiles'][active_user]['df'] = pd.concat([master_df, pd.concat(new_list)], ignore_index=True)
                st.rerun()
        
        if not master_df.empty:
            st.download_button("💾 Backup Data", master_df.to_csv(index=False), f"{active_user}_db.csv")
            db_restore = st.file_uploader("Restore Backup", type='csv')
            if db_restore and st.button("🔄 Restore"):
                restored = pd.read_csv(db_restore)
                st.session_state['profiles'][active_user]['df'] = restored
                st.rerun()
            if st.button("🗑️ Reset All"):
                st.session_state['profiles'][active_user]['df'] = pd.DataFrame()
                st.rerun()

    # --- SETTINGS ---
    st.markdown("---")
    with st.expander("⚙️ Settings & Normalization"):
        st.caption("Normalization")
        temp = st.slider("Temp (°F)", 30, 110, 75)
        alt = st.number_input("Altitude (ft)", 0, 10000, 0, 500)
        ball = st.selectbox("Ball", ["Premium (100%)", "Economy (98%)", "Range (90%)"])
        
        st.caption("Filters")
        smash_limit = st.slider("Max Smash Cap", 1.40, 1.60, 1.52)
        outlier_mode = st.checkbox("Auto-Clean Outliers", True)
        
    # --- MY BAG CONFIG (Mini) ---
    with st.expander("🎒 Bag Setup"):
        club_sel = st.selectbox("Club", CLUB_SORT_ORDER)
        curr_loft = my_bag.get(club_sel, 30.0)
        new_loft = st.number_input(f"{club_sel} Loft", value=float(curr_loft), step=0.5)
        if st.button("Save Loft"):
            st.session_state['profiles'][active_user]['bag'][club_sel] = new_loft
            st.toast("Saved!")

# --- 4. MAIN LOGIC ---
if not master_df.empty:
    st.title(f"⛳ Analytics: {active_user}")
    
    # 1. APPLY DATE FILTER
    df_view = master_df.copy()
    if 'Date' in df_view.columns:
        df_view['Date'] = pd.to_datetime(df_view['Date'])
        
        if date_filter == "Last Session":
            last_date = df_view['Date'].max()
            df_view = df_view[df_view['Date'] == last_date]
        elif date_filter == "Last 3 Sessions":
            dates = sorted(df_view['Date'].unique(), reverse=True)[:3]
            df_view = df_view[df_view['Date'].isin(dates)]
        elif date_filter == "Last 5 Sessions":
            dates = sorted(df_view['Date'].unique(), reverse=True)[:5]
            df_view = df_view[df_view['Date'].isin(dates)]
        elif date_filter == "Last 30 Days":
            cutoff = pd.Timestamp.now() - timedelta(days=30)
            df_view = df_view[df_view['Date'] >= cutoff]
        elif date_filter == "Year to Date":
            cutoff = pd.Timestamp(pd.Timestamp.now().year, 1, 1)
            df_view = df_view[df_view['Date'] >= cutoff]

    if df_view.empty:
        st.warning(f"No data found for filter: {date_filter}")
        st.stop()

    # 2. APPLY CLEANING & NORMALIZATION
    # Calculate factors
    t_fac = 1 + ((temp - 70) * 0.001)
    a_fac = 1 + (alt / 1000.0 * 0.011)
    b_fac = {"Premium (100%)":1.0, "Economy (98%)":0.98, "Range (90%)":0.90}[ball]
    total_fac = t_fac * a_fac * b_fac
    
    df_view = filter_dataset(df_view, "Auto-Clean" if outlier_mode else "Raw", smash_limit)
    
    # Create Norm Columns
    df_view['Norm_Carry'] = df_view['SL_Carry'] * total_fac
    df_view['Norm_Total'] = df_view['SL_Total'] * total_fac

    # TABS
    tabs = st.tabs(["🏠 Dashboard", "🎒 My Bag", "🎯 Accuracy", "📈 Trends", "🔬 Mechanics", "⚔️ Compare", "❓ FAQ"])

    # --- TAB 1: DASHBOARD ---
    with tabs[0]:
        st.subheader(f"Performance Summary: {date_filter}")
        
        # High Level Metrics
        tot = len(df_view)
        sess = df_view['Date'].nunique()
        fav = df_view['club'].mode()[0] if tot > 0 else "-"
        
        # Best Drive (in this period)
        drivs = df_view[df_view['club']=='Driver']
        best_drive = get_smart_max(drivs['Norm_Carry'], drivs) if not drivs.empty else 0
        
        c1, c2, c3, c4 = st.columns(4)
        def render_hero(col, title, value, sub):
            col.markdown(f"""<div class="hero-card"><div class="hero-title">{title}</div><div class="hero-metric">{value}</div><div class="hero-sub">{sub}</div></div>""", unsafe_allow_html=True)
            
        render_hero(c1, "Volume", tot, f"{sess} Sessions")
        render_hero(c2, "Best Drive", f"{best_drive:.0f}y", f"Normalized @ {temp}°F")
        render_hero(c3, "Favorite Club", fav, "Most Swings")
        
        # Dispersion Metric (DECADE style - Average 7i lateral error)
        i7 = df_view[df_view['club'] == '7 Iron']
        if not i7.empty:
            disp = i7['Lateral_Clean'].std() * 2 # 2 std devs = 95% confidence width
            render_hero(c4, "7-Iron Dispersion", f"±{disp:.1f}y", "95% Confidence Width")
        else:
            render_hero(c4, "7-Iron Dispersion", "-", "No Data")

    # --- TAB 2: MY BAG ---
    with tabs[1]:
        st.subheader("🎒 Stock Yardages (Normalized)")
        
        # Aggregation
        stats = df_view.groupby('club').agg({
            'Norm_Carry': 'mean',
            'Norm_Total': 'mean',
            'Ball (mph)': 'mean',
            'club': 'count'
        }).rename(columns={'club': 'Count'})
        
        # Ranges
        ranges = df_view.groupby('club')['Norm_Carry'].quantile([0.20, 0.80]).unstack()
        
        # Join
        bag_view = stats.join(ranges)
        bag_view['SortIndex'] = bag_view.index.map(lambda x: CLUB_SORT_ORDER.index(x) if x in CLUB_SORT_ORDER else 99)
        bag_view = bag_view.sort_values('SortIndex')
        
        st.write("---")
        cols = st.columns(4)
        for i, (club_name, row) in enumerate(bag_view.iterrows()):
            with cols[i % 4]:
                st.markdown(f"""
                <div style="background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #444; margin-bottom: 10px;">
                    <h3 style="margin:0; color: #4DD0E1;">{club_name}</h3>
                    <h2 style="margin:0; font-size: 32px; color: #FFF;">{row['Norm_Carry']:.0f}<span style="font-size:16px; color:#888"> yds</span></h2>
                    <div style="font-size: 14px; color: #00E5FF; margin-bottom: 5px; font-weight: 500;">Range: {row[0.2]:.0f} - {row[0.8]:.0f}</div>
                    <hr style="border-color: #444; margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #888;">
                        <span>Speed: {row['Ball (mph)']:.0f}</span>
                        <span style="color: #FFD700;">Tot: {row['Norm_Total']:.0f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # --- TAB 3: ACCURACY (DECADE STYLE) ---
    with tabs[2]:
        st.subheader("🎯 Accuracy & Dispersion Analysis")
        avail = [c for c in CLUB_SORT_ORDER if c in df_view['club'].unique()]
        tgt_club = st.selectbox("Analyze Club:", avail, key="acc_club")
        
        subset = df_view[df_view['club'] == tgt_club]
        
        if not subset.empty:
            avg_carry = subset['Norm_Carry'].mean()
            lat_std = subset['Lateral_Clean'].std()
            long_std = subset['Norm_Carry'].std()
            
            # Confidence Ellipse (Shotgun Pattern)
            fig = px.scatter(subset, x="Lateral_Clean", y="Norm_Carry", 
                             color="Smash", title=f"{tgt_club}: Dispersion Pattern",
                             hover_data=['Ball (mph)', 'Spin (rpm)'])
            
            # Add 1-Sigma (68%) and 2-Sigma (95%) Ellipses
            fig.add_shape(type="circle",
                x0=-lat_std*2, y0=avg_carry-long_std*2, x1=lat_std*2, y1=avg_carry+long_std*2,
                line_color="red", opacity=0.3, line_dash="dot"
            )
            
            fig.add_vline(x=0, line_color="white", opacity=0.1)
            fig.add_hline(y=avg_carry, line_color="white", opacity=0.1, annotation_text="Avg Carry")
            
            fig.update_layout(template="plotly_dark", xaxis_title="Lateral (yds)", yaxis_title="Carry (yds)",
                              yaxis=dict(scaleanchor="x", scaleratio=1)) # 1:1 Aspect Ratio is crucial
            st.plotly_chart(fig, use_container_width=True)
            
            # Strategy Guide
            st.info(f"""
            ℹ️ **Strategy Guide for {tgt_club}:**
            To keep **95%** of your shots in play (Red Circle), you need a window of **{lat_std*4:.0f} yards wide** ({lat_std*2:.0f} yds left/right).
            Your distance control varies by **±{long_std*2:.0f} yards** front-to-back.
            """)

    # --- TAB 4: TRENDS ---
    with tabs[3]:
        st.subheader("📈 Consistency Trends")
        tr_club = st.selectbox("Track Progress:", avail, key="tr_club")
        metric = st.selectbox("Metric:", ["Ball (mph)", "Norm_Carry", "Lateral_Clean"])
        
        tr_data = df_view[df_view['club'] == tr_club].groupby('Date')[metric].agg(['mean', 'std']).reset_index()
        
        if len(tr_data) > 1:
            fig_tr = go.Figure()
            # Mean Line
            fig_tr.add_trace(go.Scatter(x=tr_data['Date'], y=tr_data['mean'], mode='lines+markers', name='Average', line=dict(color='#00E676')))
            # Consistency Band (Std Dev)
            fig_tr.add_trace(go.Scatter(x=tr_data['Date'], y=tr_data['mean']+tr_data['std'], mode='lines', line=dict(width=0), showlegend=False))
            fig_tr.add_trace(go.Scatter(x=tr_data['Date'], y=tr_data['mean']-tr_data['std'], mode='lines', fill='tonexty', fillcolor='rgba(0,230,118,0.2)', line=dict(width=0), showlegend=False))
            
            fig_tr.update_layout(template="plotly_dark", title=f"{tr_club}: {metric} (Shaded = Consistency)", hovermode="x unified")
            st.plotly_chart(fig_tr, use_container_width=True)
        else:
            st.info("Need more than 1 session in the selected timeframe to show a trend.")

    # --- TAB 5: MECHANICS (EFFICIENCY) ---
    with tabs[4]:
        st.subheader("🔬 Mechanics & Efficiency")
        mech_club = st.selectbox("Analyze Club:", avail, key="mech_club")
        mech_data = df_view[df_view['club'] == mech_club]
        
        if not mech_data.empty:
            # 1. Physics Data
            col_m1, col_m2, col_m3 = st.columns(3)
            # Use safe checking
            if 'AOA (°)' in mech_data.columns:
                val = mech_data['AOA (°)'].mean()
                status, color = check_range(mech_club, val, 0, handicap)
                col_m1.metric("AoA (°)", f"{val:.1f}", status, delta_color=color)
                tip = get_coach_tip("AoA", status, mech_club)
                if tip: st.markdown(f"<div class='coach-box'>💡 <b>Coach:</b> {tip}</div>", unsafe_allow_html=True)

            if 'Spin (rpm)' in mech_data.columns:
                val = mech_data['Spin (rpm)'].mean()
                status, color = check_range(mech_club, val, 2, handicap)
                col_m2.metric("Spin (rpm)", f"{val:.0f}", status, delta_color=color)
                tip = get_coach_tip("Spin", status, mech_club)
                if tip: st.markdown(f"<div class='coach-box'>💡 <b>Coach:</b> {tip}</div>", unsafe_allow_html=True)

            # 2. Efficiency Lab
            if 'Club (mph)' in mech_data.columns and 'Norm_Carry' in mech_data.columns:
                with st.expander("🚀 Efficiency Lab", expanded=True):
                    avg_speed = mech_data['Club (mph)'].mean()
                    avg_carry = mech_data['Norm_Carry'].mean()
                    curr_loft = my_bag.get(mech_club, 30.0)
                    
                    potential = calculate_optimal_carry(avg_speed, curr_loft)
                    eff_pct = (avg_carry / potential) * 100 if potential > 0 else 0
                    
                    st.metric("Efficiency Rating", f"{eff_pct:.1f}%", f"{avg_carry - potential:.1f} yds Gap")
                    st.markdown(f"""<div class="eff-container"><div class="eff-bar-fill" style="width: {min(eff_pct, 100)}%;"></div></div>""", unsafe_allow_html=True)
                    
                    # 3. Consistency Score (New)
                    # Score based on (Mean / Std Dev). Higher is better.
                    cons_score = (avg_carry / mech_data['Norm_Carry'].std()) * 2
                    cons_score = min(100, max(0, cons_score))
                    st.metric("Consistency Score", f"{cons_score:.0f}/100", "Based on shot grouping")

    # --- TAB 6: COMPARE ---
    with tabs[5]:
        st.subheader("⚔️ Session Comparison")
        comp_club = st.selectbox("Select Club:", avail, key='c_club')
        club_data = df_view[df_view['club'] == comp_club].copy()
        
        # Create labels
        club_data['Label'] = club_data['Date'].dt.strftime('%Y-%m-%d') + " - " + club_data['Session']
        unique_sessions = club_data['Label'].unique()
        
        if len(unique_sessions) >= 2:
            c1, c2 = st.columns(2)
            s_a = c1.selectbox("Session A", unique_sessions, index=0)
            s_b = c2.selectbox("Session B", unique_sessions, index=1)
            
            if s_a != s_b:
                d_a = club_data[club_data['Label'] == s_a]
                d_b = club_data[club_data['Label'] == s_b]
                
                # Compare Metrics
                m1, m2 = st.columns(2)
                diff_carry = d_b['Norm_Carry'].mean() - d_a['Norm_Carry'].mean()
                diff_acc = abs(d_a['Lateral_Clean'].mean()) - abs(d_b['Lateral_Clean'].mean()) # Positive means B is closer to 0
                
                m1.metric("Carry Difference", f"{diff_carry:+.1f}y", "Session B vs A")
                m2.metric("Accuracy Improvement", f"{diff_acc:+.1f}y", "Closer to center line")
                
                # Histogram
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=d_a['Norm_Carry'], name='Session A', opacity=0.75))
                fig_hist.add_trace(go.Histogram(x=d_b['Norm_Carry'], name='Session B', opacity=0.75))
                fig_hist.update_layout(barmode='overlay', title="Carry Distribution", template="plotly_dark")
                st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Need at least 2 sessions to compare.")

    # --- TAB 7: FAQ ---
    with tabs[6]:
        st.header("❓ Frequently Asked Questions")
        with st.expander("🧹 What does 'Auto-Clean Outliers' do?"):
            st.write("It uses the IQR (Interquartile Range) method to remove shots that are statistically improbable for your club, plus physics checks (e.g., 500rpm spin on a driver is an error).")
        with st.expander("🌤️ How does Normalization work?"):
            st.write("We adjust your carry distance based on the Temperature and Altitude sliders in the sidebar. This lets you compare winter sessions vs summer sessions fairly.")
        with st.expander("🎯 What is the 'Shotgun Pattern'?"):
            st.write("The red circle in the Accuracy tab represents your 95% confidence interval. Strategy experts (like DECADE) suggest aiming this circle away from hazards, rather than aiming at the flag.")

else:
    # --- WELCOME SCREEN (RESTORED) ---
    st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <h1 style="font-size: 60px; font-weight: 700; background: -webkit-linear-gradient(45deg, #00E5FF, #FF4081); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Homegrown FS Pro Analytics
        </h1>
        <p style="font-size: 20px; color: #B0B3B8;">Turn your <b>Mevo+</b> data into Tour-level strategy.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.info("🆕 **New User?**")
        st.markdown("1. Open **Sidebar > 🎒 Bag Setup** to set your lofts.\n2. Use **Sidebar > 📂 Import** to add your first session.")
    with c2:
        st.warning("💾 **Returning User?**")
        st.markdown("Use **Sidebar > 📂 Import > Restore Backup** to load your `mevo_db.csv`.")

    st.markdown("---")
    c_f1, c_f2, c_f3 = st.columns(3)
    with c_f1: st.markdown('<div class="feature-card"><h3>🎯 Dispersion Cones</h3><p style="color:#888">Visualize your 95% miss pattern.</p></div>', unsafe_allow_html=True)
    with c_f2: st.markdown('<div class="feature-card"><h3>🚀 Efficiency Lab</h3><p style="color:#888">Are you maximizing your swing speed?</p></div>', unsafe_allow_html=True)
    with c_f3: st.markdown('<div class="feature-card"><h3>📈 Consistency Trends</h3><p style="color:#888">Track how your grouping tightens over time.</p></div>', unsafe_allow_html=True)
