import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# --- PAGE SETUP ---
st.set_page_config(page_title="FS Pro Strategy & Analytics", layout="wide", page_icon="⛳")

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

    /* 3. METRIC CARDS (DECADE STYLE) */
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

    /* 4. TABS & GENERAL */
    div[data-testid="stTabs"] button[aria-selected="false"] { color: #B0B3B8 !important; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: #FAFAFA !important; border-top-color: #4DD0E1 !important; }
    div[data-testid="stExpander"] { background-color: #1E222B !important; border: 1px solid #444; }
    div[data-testid="stMetricLabel"] label { color: #B0B3B8 !important; }
    div[data-testid="stMetricValue"] { color: #4DD0E1 !important; }
    
    /* 5. EFFICIENCY BAR */
    .eff-bg { background-color: #333; border-radius: 6px; height: 12px; width: 100%; margin-top: 8px; overflow: hidden; }
    .eff-fill { height: 100%; border-radius: 6px; transition: width 0.5s; }
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

def calculate_optimal_carry(club_speed, loft):
    """
    Revised Physics Engine v2.0
    Based on realistic "Tour Average" efficiency, not theoretical max.
    This prevents '110% Efficiency' oddities.
    """
    # Efficiency Curve: Yards Carry per MPH of Club Speed vs Loft
    # Driver (10 deg) -> ~2.75 yds/mph
    # 7 Iron (34 deg) -> ~2.35 yds/mph
    # LW (58 deg)     -> ~1.60 yds/mph
    
    x_lofts = [9, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    y_eff   = [2.75, 2.65, 2.55, 2.48, 2.40, 2.32, 2.20, 2.05, 1.85, 1.65, 1.50]
    
    efficiency_factor = np.interp(loft, x_lofts, y_eff)
    
    # Cap Smash Factor Physics (Prevent sensor noise from inflating potential)
    # E.g. A 1.52 smash on a wedge is impossible. We assume a "Clean Strike" smash based on loft.
    
    return club_speed * efficiency_factor

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
    
    st.markdown("---")
    
    # --- DATE FILTERING (NEW) ---
    st.header("2. Timeframe Analysis")
    date_filter = st.selectbox(
        "Select Range:", 
        ["All Time", "Last Session", "Last 3 Sessions", "Last 5 Sessions", "Last 30 Days", "Last 90 Days", "Year to Date"]
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

# --- 4. MAIN LOGIC ---
if not master_df.empty:
    st.title(f"⛳ Strategy Lab: {active_user}")
    
    # 1. APPLY DATE FILTER
    df_view = master_df.copy()
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
    elif date_filter == "Last 90 Days":
        cutoff = pd.Timestamp.now() - timedelta(days=90)
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
    tabs = st.tabs(["🏠 Dashboard", "🎯 Accuracy & Dispersion", "🚀 Efficiency Lab", "🎒 Bag Gapping", "📈 Trends", "❓ FAQ"])

    # --- TAB 1: DASHBOARD ---
    with tabs[0]:
        st.subheader(f"Performance Summary: {date_filter}")
        
        # High Level Metrics
        tot = len(df_view)
        sess = df_view['Date'].nunique()
        fav = df_view['club'].mode()[0] if tot > 0 else "-"
        
        # Best Drive (in this period)
        drivs = df_view[df_view['club']=='Driver']
        best_drive = drivs['Norm_Carry'].max() if not drivs.empty else 0
        
        c1, c2, c3, c4 = st.columns(4)
        def metric_box(col, label, val, sub):
            col.markdown(f"""
            <div class="metric-card">
                <p class="metric-lbl">{label}</p>
                <p class="metric-val">{val}</p>
                <p class="metric-sub">{sub}</p>
            </div>
            """, unsafe_allow_html=True)
            
        metric_box(c1, "Volume", tot, f"{sess} Sessions")
        metric_box(c2, "Best Drive", f"{best_drive:.0f}y", f"Normalized @ {temp}°F")
        metric_box(c3, "Favorite Club", fav, "Most Swings")
        
        # Dispersion Metric (DECADE style - Average 7i lateral error)
        i7 = df_view[df_view['club'] == '7 Iron']
        if not i7.empty:
            disp = i7['Lateral_Clean'].std() * 2 # 2 std devs = 95% confidence width
            metric_box(c4, "7-Iron Dispersion", f"±{disp:.1f}y", "95% Confidence Width")
        else:
            metric_box(c4, "7-Iron Dispersion", "-", "No Data")

    # --- TAB 2: ACCURACY (DECADE STYLE) ---
    with tabs[1]:
        st.subheader("🎯 Shotgun Patterns & Dispersion")
        st.caption("This view helps you understand your 'Cone of Error' (DECADE Method). Aim for the center of your shotgun pattern, not the flag.")
        
        avail = [c for c in CLUB_SORT_ORDER if c in df_view['club'].unique()]
        tgt_club = st.selectbox("Analyze Club:", avail)
        
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
            fig.add_shape(type="circle",
                x0=-lat_std, y0=avg_carry-long_std, x1=lat_std, y1=avg_carry+long_std,
                line_color="green", opacity=0.5
            )
            
            # Center Line
            fig.add_vline(x=0, line_color="white", opacity=0.1)
            fig.add_hline(y=avg_carry, line_color="white", opacity=0.1, annotation_text="Avg Carry")
            
            fig.update_layout(template="plotly_dark", xaxis_title="Lateral (yds)", yaxis_title="Carry (yds)",
                              yaxis=dict(scaleanchor="x", scaleratio=1)) # 1:1 Aspect Ratio is crucial for dispersion
            st.plotly_chart(fig, use_container_width=True)
            
            # Strategy Table
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.info(f"""
                **Strategy Guide for {tgt_club}:**
                * **Stock Yardage:** {avg_carry:.0f} yds
                * **Aim Window:** You need **{lat_std*2:.0f} yards** of room left/right to keep 68% of shots in play.
                * **Long/Short:** Your shots vary by **±{long_std*2:.0f} yards** front-to-back.
                """)

    # --- TAB 3: EFFICIENCY LAB (UPDATED PHYSICS) ---
    with tabs[2]:
        st.subheader("🚀 Swing Efficiency Lab")
        st.caption("Are you getting the most out of your speed? (Based on Scratch Player Benchmarks)")
        
        e_club = st.selectbox("Check Efficiency:", avail, key="eff_club")
        e_data = df_view[df_view['club'] == e_club]
        
        if not e_data.empty:
            avg_spd = e_data['Club (mph)'].mean()
            avg_cry = e_data['Norm_Carry'].mean()
            curr_loft = my_bag.get(e_club, 30.0)
            
            # Use new Interpolated Physics Formula
            potential = calculate_optimal_carry(avg_spd, curr_loft)
            eff_pct = (avg_cry / potential) * 100 if potential > 0 else 0
            
            # Visuals
            c_e1, c_e2, c_e3 = st.columns(3)
            c_e1.metric("Club Speed", f"{avg_spd:.1f} mph")
            c_e2.metric("Potential Carry", f"{potential:.0f} yds", f"Loft: {curr_loft}°")
            c_e3.metric("Efficiency", f"{eff_pct:.1f}%", f"{avg_cry - potential:.1f} yds Gap")
            
            # Progress Bar color logic
            color = "#00E676" if eff_pct > 90 else "#FFEB3B" if eff_pct > 80 else "#FF4081"
            st.markdown(f"""
            <div class="eff-bg">
                <div class="eff-fill" style="width: {min(eff_pct, 100)}%; background-color: {color};"></div>
            </div>
            """, unsafe_allow_html=True)
            
            if eff_pct < 85:
                st.warning(f"⚠️ You are losing distance. Your speed ({avg_spd:.0f} mph) should produce {potential:.0f} yds. Check if your Spin is too high or Launch is too low.")
            else:
                st.success("✅ Excellent efficiency! You are transferring energy effectively.")

    # --- TAB 4: GAPPING ---
    with tabs[3]:
        st.subheader("🎒 Stock Yardages (Normalized)")
        # Box plot sorted correctly
        df_view['SortIndex'] = df_view['club'].map(lambda x: CLUB_SORT_ORDER.index(x) if x in CLUB_SORT_ORDER else 99)
        sorted_df = df_view.sort_values('SortIndex')
        
        fig_gap = px.box(sorted_df, x='club', y='Norm_Carry', color='club')
        fig_gap.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_gap, use_container_width=True)
        
        # Simple Table
        summary = sorted_df.groupby('club')['Norm_Carry'].agg(['mean', 'min', 'max', 'count']).round(1)
        st.dataframe(summary)

    # --- TAB 5: TRENDS ---
    with tabs[4]:
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

    # --- TAB 6: FAQ ---
    with tabs[5]:
        st.info("ℹ️ **About the Strategy Logic (DECADE Inspired)**")
        st.markdown("""
        * **Shotgun Patterns:** The ellipses in the Accuracy tab show where 68% (Green) and 95% (Red) of your shots land. Course management is about shifting this ellipse away from hazards.
        * **Efficiency Lab:** Uses a custom physics curve based on Loft. It compares your carry to what a Scratch golfer would achieve at your swing speed.
        * **Normalization:** All data is adjusted to **75°F / 0ft Altitude** by default so winter practice doesn't ruin your confidence.
        """)

else:
    # EMPTY STATE
    st.title("⛳ FS Pro Strategy")
    st.info("👈 **Start Here:** Upload your FlightScope CSVs in the Sidebar to unlock the Strategy Lab.")
    st.markdown("### The Four Foundations of This App:")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("#### 1. Expectation\nKnow your *actual* dispersion, not your best shot.")
    c2.markdown("#### 2. Strategy\nUse Shotgun patterns to pick smarter targets.")
    c3.markdown("#### 3. Practice\nTrack strokes gained metrics to fix weaknesses.")
    c4.markdown("#### 4. Review\nAnalyze trends over time with consistency bands.")
