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
    section[data-testid="stSidebar"] input {
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

    /* 5. REDESIGNED STATS BOX */
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

    /* Custom Coach Box */
    .coach-box {
        background-color: #262730;
        border-left: 4px solid #FF4081;
        padding: 15px;
        border-radius: 5px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. INITIALIZE SESSION STATE ---
if 'master_df' not in st.session_state:
    st.session_state['master_df'] = pd.DataFrame()

DEFAULT_LOFTS = {'Driver': 10.5, '3 Wood': 15.0, '5 Wood': 18.0, 'Hybrid': 21.0, '3 Iron': 21.0, '4 Iron': 24.0, '5 Iron': 27.0, '6 Iron': 30.0, '7 Iron': 34.0, '8 Iron': 38.0, '9 Iron': 42.0, 'PW': 46.0, 'GW': 50.0, 'SW': 54.0, 'LW': 58.0}
CLUB_SORT_ORDER = ['Driver', '3 Wood', '5 Wood', '7 Wood', 'Hybrid', '2 Iron', '3 Iron', '4 Iron', '5 Iron', '6 Iron', '7 Iron', '8 Iron', '9 Iron', 'PW', 'GW', 'SW', 'LW']

if 'my_bag' not in st.session_state:
    st.session_state['my_bag'] = DEFAULT_LOFTS.copy()

# --- 2. HELPERS ---
def get_dynamic_ranges(club_name, handicap):
    c_lower = str(club_name).lower()
    tolerance = handicap * 0.1
    launch_help = 0 if handicap < 5 else (1.0 if handicap < 15 else 2.0)
    user_loft = st.session_state['my_bag'].get(club_name, 30.0)

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

    # Normalize
    df_clean['SL_Carry'] = df_clean['Carry (yds)'] / (1 + (df_clean['Altitude (ft)'] / 1000.0 * 0.011))
    df_clean['SL_Total'] = df_clean['Total (yds)'] / (1 + (df_clean['Altitude (ft)'] / 1000.0 * 0.011))
    return df_clean

def filter_outliers(df):
    df_filtered = pd.DataFrame()
    outlier_count = 0
    
    for club in df['club'].unique():
        club_data = df[df['club'] == club].copy()
        
        # --- STAGE 1: PHYSICS SANITY CHECK ---
        valid_physics = club_data[
            (club_data['Smash'] <= 1.58) & 
            (club_data['Smash'] >= 1.0) &
            (club_data['Spin (rpm)'] > 500) & 
            (club_data['Height (ft)'] > 8)
        ]
        
        dropped_physics = len(club_data) - len(valid_physics)
        
        if len(valid_physics) < 5:
            df_filtered = pd.concat([df_filtered, valid_physics])
            outlier_count += dropped_physics
            continue

        # --- STAGE 2: STATISTICAL IQR CHECK (Distance) ---
        q1 = valid_physics['Carry (yds)'].quantile(0.25)
        q3 = valid_physics['Carry (yds)'].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - (1.5 * iqr) 
        upper = q3 + (3.0 * iqr) # Allow more upside variance for good shots
        
        final_valid = valid_physics[
            (valid_physics['Carry (yds)'] >= lower) & 
            (valid_physics['Carry (yds)'] <= upper)
        ]
        
        dropped_stat = len(valid_physics) - len(final_valid)
        outlier_count += (dropped_physics + dropped_stat)
        
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
        if "Low" in status: return "Spin is dangerously low (ball will drop). Check for high strikes on the face."
        if "High" in status: return "Spin is too high (ballooning). Check for low face strikes or excessive cut spin."
    return None

def style_fig(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("1. Database Manager")
    with st.expander("📂 Load / Save History", expanded=False):
        db_file = st.file_uploader("Restore 'mevo_db.csv'", type='csv', key='db_uploader')
        if db_file:
            if st.button("🔄 Restore Database"):
                try:
                    restored = pd.read_csv(db_file)
                    if 'Date' in restored.columns: restored['Date'] = pd.to_datetime(restored['Date'])
                    if 'Ref Loft' in restored.columns:
                        latest_lofts = restored.drop_duplicates('club', keep='last').set_index('club')['Ref Loft'].to_dict()
                        st.session_state['my_bag'].update(latest_lofts)
                        st.toast("Bag Lofts Restored!", icon="🎒")
                    st.session_state['master_df'] = restored
                    st.success(f"Restored {len(restored)} shots!")
                    st.rerun()
                except Exception as e: st.error(f"Error: {e}")
        
        if not st.session_state['master_df'].empty:
            csv_data = st.session_state['master_df'].to_csv(index=False).encode('utf-8')
            st.download_button("💾 Save Database", csv_data, "mevo_db.csv", "text/csv")
            if st.button("🗑️ Clear All"):
                st.session_state['master_df'] = pd.DataFrame()
                st.rerun()

    st.header("2. Add Session")
    import_date = st.date_input("Date of Session")
    uploaded_files = st.file_uploader("Upload CSVs", accept_multiple_files=True, type='csv', key=f"uploader_{import_date}")
    
    if st.button("➕ Add to Database"):
        if uploaded_files:
            new_data = []
            for f in uploaded_files:
                try:
                    raw = pd.read_csv(f)
                    clean = clean_mevo_data(raw, f.name, import_date)
                    clean['Ref Loft'] = clean['club'].map(st.session_state['my_bag'])
                    new_data.append(clean)
                except Exception as e: st.error(f"Error {f.name}: {e}")
            if new_data:
                batch_df = pd.concat(new_data, ignore_index=True)
                batch_df['Ref Loft'] = batch_df['Ref Loft'].fillna(30.0)
                st.session_state['master_df'] = pd.concat([st.session_state['master_df'], batch_df], ignore_index=True)
                st.success(f"Added {len(batch_df)} shots!")
                st.rerun()

    st.markdown("---")
    
    # --- MY BAG CONFIG ---
    st.header("3. My Bag Setup")
    with st.expander("⚙️ Configure Club Lofts"):
        selected_club = st.selectbox("Choose Club:", CLUB_SORT_ORDER, index=0)
        current_loft = st.session_state['my_bag'].get(selected_club, DEFAULT_LOFTS.get(selected_club, 30.0))
        new_loft = st.number_input(f"Loft for {selected_club} (°)", value=float(current_loft), step=0.5, format="%.1f")
        
        if st.button("💾 Save Loft Change", type="primary", use_container_width=True):
            st.session_state['my_bag'][selected_club] = new_loft
            st.toast(f"Saved: {selected_club} @ {new_loft}°", icon="✅")
            
        st.markdown("---")
        st.caption("Current Configuration:")
        bag_df = pd.DataFrame(list(st.session_state['my_bag'].items()), columns=['Club', 'Loft'])
        bag_df['SortIndex'] = bag_df['Club'].apply(lambda x: CLUB_SORT_ORDER.index(x) if x in CLUB_SORT_ORDER else 99)
        bag_df = bag_df.sort_values('SortIndex').drop(columns=['SortIndex'])
        st.dataframe(bag_df, hide_index=True, use_container_width=True, height=200)
            
    # --- PLAYER CONFIG ---
    st.markdown("---")
    st.header("4. Player Config")
    env_mode = st.radio("Filter:", ["All", "Outdoor Only", "Indoor Only"], index=0)
    handicap = st.number_input("Handicap", 0, 54, 15)
    smash_cap = st.slider("Max Smash Cap", 1.40, 1.65, 1.52, 0.01)
    remove_bad_shots = st.checkbox("Auto-Clean Outliers", value=True)

    # SUMMARY STATS
    if not st.session_state['master_df'].empty:
        st.markdown("---")
        tot_shots = len(st.session_state['master_df'])
        tot_sess = st.session_state['master_df']['Date'].nunique()
        st.markdown(f"""
        <div class="stat-card-container">
            <div class="stat-card"><p class="stat-value">{tot_shots}</p><p class="stat-label">Shots</p></div>
            <div class="stat-card"><p class="stat-value" style="color: #FF4081;">{tot_sess}</p><p class="stat-label">Sessions</p></div>
        </div>
        """, unsafe_allow_html=True)

# --- 4. MAIN APP LOGIC ---
master_df = st.session_state['master_df']

if not master_df.empty:
    st.title("⛳ Homegrown FS Pro Analytics")
    
    filtered_df = master_df.copy()
    if env_mode == "Outdoor Only" and 'Mode' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Mode'].str.contains("Outdoor", case=False, na=False)]
    elif env_mode == "Indoor Only" and 'Mode' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Mode'].str.contains("Indoor", case=False, na=False)]
    
    filtered_df = filtered_df[filtered_df['Smash'] <= smash_cap]

    if remove_bad_shots:
        filtered_df, dropped_count = filter_outliers(filtered_df)
        if dropped_count > 0: st.toast(f"Cleaned {dropped_count} outliers", icon="🧹")

    st.caption(f"Analyzing {len(filtered_df)} shots")

    # --- TABS ---
    tab_bag, tab_acc, tab_gap, tab_time, tab_mech, tab_comp, tab_faq = st.tabs(["🎒 My Bag", "🎯 Accuracy", "📏 Gapping", "📈 Timeline", "🔬 Mechanics", "⚔️ Compare", "❓ FAQ"])

    # ================= TAB: MY BAG =================
    with tab_bag:
        st.subheader("🎒 My Bag & Yardages")
        col_set1, col_set2 = st.columns([1, 3])
        with col_set1:
            play_alt = st.number_input("⛰️ Play Altitude (ft)", value=0, step=500, help="Adjust for course altitude.")
            alt_factor = 1 + (play_alt / 1000.0 * 0.011)
            if play_alt > 0: st.caption(f"Boost: +{((alt_factor-1)*100):.1f}%")

        # Smart Max Calculation (Matches Filter Logic)
        def get_smart_max(series, df_subset):
            # We must use the indices from the series to filter the subset correctly
            valid = df_subset.loc[series.index]
            # Apply same physics checks as filter_outliers
            clean = valid[
                (valid['Smash'] <= 1.58) & (valid['Smash'] >= 1.0) &
                (valid['Spin (rpm)'] > 500) & (valid['Height (ft)'] > 8)
            ]
            if clean.empty: return series.max()
            return clean['SL_Carry'].max()

        # Aggregation
        bag_data = []
        for club in filtered_df['club'].unique():
            subset = filtered_df[filtered_df['club'] == club]
            s_max = get_smart_max(subset['SL_Carry'], subset)
            bag_data.append({
                'Club': club,
                'SL_Carry': subset['SL_Carry'].mean(),
                'SL_Total': subset['SL_Total'].mean(),
                'Ball Speed': subset['Ball (mph)'].mean(),
                'Max Carry': s_max,
                'Count': len(subset)
            })
        
        bag_stats = pd.DataFrame(bag_data).set_index('Club')
        bag_stats['SortIndex'] = bag_stats.index.map(lambda x: CLUB_SORT_ORDER.index(x) if x in CLUB_SORT_ORDER else 99)
        bag_stats = bag_stats.sort_values('SortIndex')
        
        bag_stats['Adj. Carry'] = bag_stats['SL_Carry'] * alt_factor
        bag_stats['Adj. Total'] = bag_stats['SL_Total'] * alt_factor
        bag_stats['Adj. Max'] = bag_stats['Max Carry'] * alt_factor
        
        st.write("---")
        cols = st.columns(4)
        for i, (index, row) in enumerate(bag_stats.iterrows()):
            with cols[i % 4]:
                st.markdown(f"""
                <div style="background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #444; margin-bottom: 10px;">
                    <h3 style="margin:0; color: #4DD0E1;">{index}</h3>
                    <h2 style="margin:0; font-size: 32px; color: #FFF;">{row['Adj. Carry']:.0f}<span style="font-size:16px; color:#888"> yds</span></h2>
                    <p style="margin:0; color: #BBB;">Total: <b>{row['Adj. Total']:.0f}</b> <span style="font-size:12px; color:#555">(n={int(row['Count'])})</span></p>
                    <hr style="border-color: #444; margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #888;">
                        <span>Speed: {row['Ball Speed']:.0f}</span>
                        <span style="color: #FFD700;">Pot: {row['Adj. Max']:.0f}</span>
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
                # 3-TIER TARGET LOGIC
                c_name = str(selected_club).lower()
                
                # 1. TEE SHOTS (Driver, Woods, Hybrids)
                if 'driver' in c_name or 'wood' in c_name or 'hybrid' in c_name:
                    target_val = 15.0 + (handicap * 0.5) # Wide Fairway
                    target_type = "Fairway Lane"
                # 2. APPROACH (Irons 2-7)
                elif any(x in c_name for x in ['2','3','4','5','6','7']):
                    target_val = 10.0 + (handicap * 0.4) # Green Width
                    target_type = "Approach Lane"
                # 3. SCORING (8,9, Wedges)
                else:
                    target_val = 5.0 + (handicap * 0.25) # Pin Seeking
                    target_type = "Pin Radius"
                
                # New Tendency Metric
                lat_mean = subset['Lateral_Clean'].mean()
                tendency_dir = "Right ➡️" if lat_mean > 0 else "Left ⬅️"
                
                c1, c2, c3, c4 = st.columns(4)
                on_target = len(subset[abs(subset['Lateral_Clean']) <= target_val]) / len(subset) * 100
                c1.metric("Accuracy Score", f"{on_target:.0f}%", f"{target_type}: ±{target_val:.1f}y")
                c2.metric("Tendency", f"{abs(lat_mean):.1f}y", tendency_dir)
                c3.metric("Avg Carry", f"{subset['Carry (yds)'].mean():.1f}")
                c4.metric("Ball Speed", f"{subset['Ball (mph)'].mean():.1f}")

                c_chart1, c_chart2 = st.columns([3, 1])
                with c_chart1:
                    subset['Shape'] = np.where(subset['Lateral_Clean'] > 0, 'Fade (R)', 'Draw (L)')
                    fig = px.scatter(subset, x='Lateral_Clean', y='Carry (yds)', color='Shape',
                        color_discrete_map={'Fade (R)': '#00E5FF', 'Draw (L)': '#FF4081'},
                        hover_data=['Date'], title=f"Dispersion: {selected_club}")
                    fig.add_shape(type="rect", x0=-target_val, y0=subset['Carry (yds)'].min()-10, x1=target_val, y1=subset['Carry (yds)'].max()+10,
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
            st.subheader("🎒 Bag Gapping")
            with st.expander("ℹ️ How to read this chart?"):
                st.markdown("""
                * **The Box:** Represents your "Consistency Zone" (middle 50% of shots). A smaller box is better.
                * **The Line inside:** Your Median (most typical) carry distance.
                * **The Whiskers (Lines):** Your absolute range (Longest vs Shortest), excluding outliers.
                """)
            filtered_df['SortIndex'] = filtered_df['club'].map(lambda x: CLUB_SORT_ORDER.index(x) if x in CLUB_SORT_ORDER else 99)
            fig = px.box(filtered_df.sort_values('SortIndex'), x='club', y='Carry (yds)', color='club', points="all")
            st.plotly_chart(style_fig(fig), use_container_width=True)

    # ================= TAB: TIMELINE =================
    with tab_time:
        if len(filtered_df) > 0:
            st.subheader("📈 Timeline")
            c_t1, c_t2 = st.columns(2)
            avail_clubs = [c for c in CLUB_SORT_ORDER if c in filtered_df['club'].unique()]
            with c_t1: t_club = st.selectbox("Club", avail_clubs, key='t_club')
            with c_t2: metric = st.selectbox("Metric", ['Ball (mph)', 'Carry (yds)', 'Club (mph)', 'Smash'])
            
            if 'Date' in filtered_df.columns:
                trend = filtered_df[filtered_df['club'] == t_club].groupby('Date')[metric].mean().reset_index().sort_values('Date')
                fig = px.line(trend, x='Date', y=metric, markers=True, title=f"{t_club} Progress")
                fig.update_traces(line_color='#00E676', line_width=4)
                st.plotly_chart(style_fig(fig), use_container_width=True)
            else: st.warning("No Date info.")

    # ================= TAB: MECHANICS =================
    with tab_mech:
        if len(filtered_df) > 0:
            st.subheader("🔬 Swing Mechanics")
            c_sel1, c_sel2 = st.columns([2,1])
            avail_clubs = [c for c in CLUB_SORT_ORDER if c in filtered_df['club'].unique()]
            with c_sel1: mech_club = st.selectbox("Analyze Club", avail_clubs, key='m_club')
            with c_sel2: 
                curr_loft = st.session_state['my_bag'].get(mech_club, 30.0)
                st.metric("Bag Loft", f"{curr_loft}°")

            mech_data = filtered_df[filtered_df['club'] == mech_club]
            
            col_m1, col_m2, col_m3 = st.columns(3)
            
            # Helper to display metrics + coach tip
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

            st.markdown("---")
            col_chart_m1, col_chart_m2 = st.columns([2, 1])
            with col_chart_m1:
                if 'Height (ft)' in mech_data.columns:
                    st.markdown("#### ✈️ Trajectory Window")
                    fig_traj = px.scatter(mech_data, x='Carry (yds)', y='Height (ft)', color='Session')
                    fig_traj.add_shape(type="rect", x0=mech_data['Carry (yds)'].min(), y0=80, x1=mech_data['Carry (yds)'].max(), y1=110, line=dict(color="Gold", width=0), fillcolor="Gold", opacity=0.1)
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
                a_carry = data_a['Carry (yds)'].mean()
                b_carry = data_b['Carry (yds)'].mean()
                a_acc = data_a['Lateral_Clean'].abs().mean()
                b_acc = data_b['Lateral_Clean'].abs().mean()
                
                win_carry = "🏆 Session B" if b_carry > a_carry else "🏆 Session A"
                win_acc = "🏆 Session B" if b_acc < a_acc else "🏆 Session A"
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Carry Winner", win_carry, f"Diff: {abs(b_carry - a_carry):.1f}y")
                m2.metric("Accuracy Winner", win_acc, f"Diff: {abs(b_acc - a_acc):.1f}y")
                
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=data_a['Carry (yds)'], name='Session A', opacity=0.75, marker_color='#FF4081'))
                fig_hist.add_trace(go.Histogram(x=data_b['Carry (yds)'], name='Session B', opacity=0.75, marker_color='#00E5FF'))
                fig_hist.update_layout(barmode='overlay', title=f"Carry Distance Distribution")
                st.plotly_chart(style_fig(fig_hist), use_container_width=True)
            else: st.warning("Select different sessions.")
        else: st.warning("Need 2+ sessions.")

    # ================= TAB: FAQ =================
    with tab_faq:
        st.subheader("❓ FAQ & Help")
        with st.expander("🧹 What is 'Auto-Clean Outliers'?", expanded=False):
            st.markdown("We use the **IQR method** to strip out misreads and duffs.")
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
