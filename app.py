import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="Mevo+ Pro Analytics", layout="wide", page_icon="⛳")

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #4DD0E1; }
    div[data-testid="stMetricDelta"] { font-size: 16px; }
    h1, h2, h3 { color: #FAFAFA; font-family: 'Helvetica Neue', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.title("⛳ Mevo+ Pro Analytics")

# --- 1. DATABASES ---

# Standard Lofts (Reference only - User can override)
STANDARD_LOFTS = {
    'Driver': 10.5, '3 Wood': 15.0, '5 Wood': 18.0, 'Hybrid': 21.0,
    '3 Iron': 21.0, '4 Iron': 24.0, '5 Iron': 27.0, '6 Iron': 30.0,
    '7 Iron': 34.0, '8 Iron': 38.0, '9 Iron': 42.0, 'PW': 46.0,
    'GW': 50.0, 'SW': 54.0, 'LW': 58.0
}

def get_dynamic_ranges(club_name, handicap, user_loft=None):
    """
    Calculates optimal targets based on Handicap AND Specific Club Loft.
    """
    c_lower = str(club_name).lower()
    
    # 1. Handicap Factors
    tolerance = handicap * 0.1  # Widens the accepted window
    launch_help = 0 if handicap < 5 else (1.0 if handicap < 15 else 2.0) # High hcps need more height

    # 2. Loft Logic
    # If we have a specific user loft, use math to determine launch/spin
    # Heuristic: Launch ~ 0.55 * Loft. Spin ~ Loft * 200.
    
    if user_loft:
        target_loft = user_loft
    else:
        # Fallback to dictionary
        target_loft = STANDARD_LOFTS.get(club_name, 30.0) # Default mid-iron

    # --- DRIVER/WOODS SPECIAL LOGIC ---
    if 'driver' in c_lower:
        # Driver Launch is less dependent on static loft, more on AoA
        aoa = (-2.0 - (tolerance*0.2), 5.0 + (tolerance*0.2)) 
        # Base launch 10-15, adjusted by handicap
        launch = (10.0 + launch_help, 16.0 + launch_help + (tolerance*0.2))
        spin = (1800, 2800 + (handicap * 40))

    elif 'wood' in c_lower or 'hybrid' in c_lower:
        aoa = (-4.0, 1.0 + tolerance)
        # Math-based launch for woods
        l_center = target_loft * 0.7 
        launch = (l_center - 2.0, l_center + 2.0 + launch_help)
        spin = (target_loft * 200, target_loft * 280)

    else:
        # --- IRONS & WEDGES (Math Based on Loft) ---
        
        # AoA: Steeper for higher lofts
        # 20deg club -> -2.0 AoA. 50deg club -> -5.0 AoA.
        # Formula: -1.0 - (Loft / 12)
        target_aoa = -1.0 - (target_loft / 12.0)
        aoa = (target_aoa - 2.0 - tolerance, -0.5) # Cap at -0.5 to prevent flipping

        # Launch: Typically ~45-55% of static loft for irons
        l_min = (target_loft * 0.45) - 1.0
        l_max = (target_loft * 0.55) + 1.0 + launch_help
        launch = (l_min - tolerance, l_max + tolerance)

        # Spin: ~200 rpm per degree of loft is a good rule of thumb (e.g., 34deg -> 6800rpm)
        # Stronger lofts (lower number) will naturally require less spin
        s_base = target_loft * 210
        spin = (s_base - 1000 - (tolerance*100), s_base + 1000 + (tolerance*100))

    return aoa, launch, spin

# --- 2. DATA PROCESSING ---
def clean_mevo_data(df, filename):
    df_clean = df[df['Shot'].astype(str).str.isdigit()].copy()
    df_clean['Session'] = filename.replace('.csv', '')
    
    def parse_lr(val):
        if pd.isna(val): return 0.0
        s_val = str(val).strip()
        if 'L' in s_val: 
            try: return -float(s_val.replace('L','').strip())
            except: return 0.0
        if 'R' in s_val: 
            try: return float(s_val.replace('R','').strip())
            except: return 0.0
        try: return float(s_val)
        except: return 0.0

    dir_cols = ['Lateral (yds)', 'Swing H (°)', 'Launch H (°)', 'Spin Axis (°)', 'Club Path (°)', 'FTP (°)']
    for col in dir_cols:
        if col in df_clean.columns:
            clean_col_name = col.replace(' (yds)', '').replace(' (°)', '') + '_Clean'
            if 'Lateral' in col:
                df_clean['Lateral_Clean'] = df_clean[col].apply(parse_lr)
            else:
                df_clean[clean_col_name] = df_clean[col].apply(parse_lr)

    numeric_cols = ['Carry (yds)', 'Total (yds)', 'Ball (mph)', 'Club (mph)', 'Smash', 'Spin (rpm)', 'Height (ft)', 'AOA (°)', 'Launch V (°)']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean

def filter_outliers(df):
    df_filtered = pd.DataFrame()
    outlier_count = 0
    for club in df['club'].unique():
        club_data = df[df['club'] == club].copy()
        if len(club_data) < 5:
            df_filtered = pd.concat([df_filtered, club_data])
            continue
        q1 = club_data['Carry (yds)'].quantile(0.25)
        q3 = club_data['Carry (yds)'].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - (1.5 * iqr) 
        upper = q3 + (3.0 * iqr) 
        valid = club_data[(club_data['Carry (yds)'] >= lower) & (club_data['Carry (yds)'] <= upper)]
        outlier_count += (len(club_data) - len(valid))
        df_filtered = pd.concat([df_filtered, valid])
    return df_filtered, outlier_count

def check_range(club_name, value, metric_idx, handicap, user_loft):
    """
    Checks if value is within dynamic optimal range.
    metric_idx: 0=AoA, 1=Launch, 2=Spin
    """
    ranges = get_dynamic_ranges(club_name, handicap, user_loft) 
    min_v, max_v = ranges[metric_idx]
    
    if min_v <= value <= max_v:
        return "Optimal ✅", "normal"
    elif value < min_v:
        diff = value - min_v
        return f"{diff:.1f} (Low) ⚠️", "inverse"
    else:
        diff = value - max_v
        return f"+{diff:.1f} (High) ⚠️", "inverse"

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("1. Upload")
    uploaded_files = st.file_uploader("CSV Files", accept_multiple_files=True, type='csv')
    
    st.header("2. Environment")
    env_mode = st.radio("Filter:", ["All", "Outdoor Only", "Indoor Only"], index=0)
    
    st.header("3. Player Profile")
    handicap = st.number_input("Handicap", 0, 54, 15, help="Adjusts target sizes and mechanics tolerances.")
    smash_cap = st.slider("Max Smash Cap", 1.40, 1.65, 1.52, 0.01)
    
    st.header("4. Settings")
    remove_bad_shots = st.checkbox("Auto-Clean Outliers", value=True)

# --- 4. MAIN APP ---
if uploaded_files:
    all_data = []
    for f in uploaded_files:
        try:
            raw = pd.read_csv(f)
            clean = clean_mevo_data(raw, f.name)
            all_data.append(clean)
        except Exception as e:
            st.error(f"Error {f.name}: {e}")

    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        
        # Filters
        if env_mode == "Outdoor Only" and 'Mode' in master_df.columns:
            master_df = master_df[master_df['Mode'].str.contains("Outdoor", case=False, na=False)]
        elif env_mode == "Indoor Only" and 'Mode' in master_df.columns:
            master_df = master_df[master_df['Mode'].str.contains("Indoor", case=False, na=False)]
            
        master_df = master_df[master_df['Smash'] <= smash_cap]
        if remove_bad_shots:
            master_df, _ = filter_outliers(master_df)
            
        st.sidebar.markdown("---")
        st.sidebar.download_button("📥 Download Database", master_df.to_csv(index=False).encode('utf-8'), "mevo_db.csv", "text/csv")

        st.write("---")
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Target & Accuracy", "🎒 Gapping", "📈 Trends", "🔬 Swing Mechanics"])

        def style_fig(fig):
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return fig

        # ================= TAB 1: TARGET ANALYZER =================
        with tab1:
            if not master_df.empty:
                club_order = master_df.groupby('club')['Carry (yds)'].mean().sort_values(ascending=False).index
                selected_club = st.selectbox("Select Club", club_order, key='t1_club')
                subset = master_df[master_df['club'] == selected_club]
                
                if len(subset) > 0:
                    is_scoring = any(x in str(selected_club).lower() for x in ['8','9','p','w','s','l','g'])
                    target_val = 5.0 + (handicap * 0.4) if is_scoring else 15.0 + (handicap * 0.8)
                    target_type = "Green Radius" if is_scoring else "Lane Width"
                    
                    if 'Lateral_Clean' in subset.columns:
                        on_target = subset[abs(subset['Lateral_Clean']) <= target_val]
                        acc_score = (len(on_target) / len(subset)) * 100
                    else:
                        acc_score = 0

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy Score", f"{acc_score:.0f}%", f"{target_type}: ±{target_val:.1f}y")
                    c2.metric("Avg Carry", f"{subset['Carry (yds)'].mean():.1f}")
                    c3.metric("Ball Speed", f"{subset['Ball (mph)'].mean():.1f}")
                    c4.metric("Smash", f"{subset['Smash'].mean():.2f}")

                    c_chart1, c_chart2 = st.columns([3, 1])
                    with c_chart1:
                        if 'Lateral_Clean' in subset.columns:
                            subset['Shape'] = np.where(subset['Lateral_Clean'] > 0, 'Fade (R)', 'Draw (L)')
                            fig = px.scatter(subset, x='Lateral_Clean', y='Carry (yds)', color='Shape',
                                color_discrete_map={'Fade (R)': '#00E5FF', 'Draw (L)': '#FF4081'},
                                title=f"Dispersion: {selected_club}")
                            
                            y_min, y_max = subset['Carry (yds)'].min()-10, subset['Carry (yds)'].max()+10
                            avg_c = subset['Carry (yds)'].mean()
                            
                            if is_scoring: 
                                fig.add_shape(type="circle", x0=-target_val, y0=avg_c-target_val, x1=target_val, y1=avg_c+target_val,
                                    line_color="#00E676", fillcolor="#00E676", opacity=0.2)
                            else: 
                                fig.add_shape(type="rect", x0=-target_val, y0=y_min, x1=target_val, y1=y_max,
                                    line_color="#00E676", fillcolor="#00E676", opacity=0.1)
                            
                            fig.add_vline(x=0, line_color="white", opacity=0.2)
                            fig.update_xaxes(range=[-60, 60], title="Left <---> Right")
                            st.plotly_chart(style_fig(fig), use_container_width=True)

                    with c_chart2:
                        if 'Shot Type' in subset.columns:
                            counts = subset['Shot Type'].value_counts().reset_index()
                            counts.columns = ['Type', 'Count']
                            fig_pie = px.pie(counts, values='Count', names='Type', hole=0.5, color_discrete_sequence=px.colors.qualitative.Pastel)
                            st.plotly_chart(style_fig(fig_pie), use_container_width=True)

        # ================= TAB 2: GAPPING =================
        with tab2:
            if not master_df.empty:
                st.subheader("🎒 Bag Gapping")
                means = master_df.groupby("club")["Carry (yds)"].mean().sort_values(ascending=False)
                fig = px.box(master_df, x='club', y='Carry (yds)', color='club', category_orders={'club': means.index}, points="all")
                st.plotly_chart(style_fig(fig), use_container_width=True)

        # ================= TAB 3: TRENDS =================
        with tab3:
            if not master_df.empty:
                st.subheader("📈 Trends")
                c_t1, c_t2 = st.columns(2)
                with c_t1: t_club = st.selectbox("Club", means.index, key='t_club')
                with c_t2: metric = st.selectbox("Metric", ['Ball (mph)', 'Carry (yds)', 'Club (mph)', 'Smash'])
                
                trend = master_df[master_df['club'] == t_club].groupby('Session')[metric].mean().reset_index().sort_values('Session')
                if len(trend) > 1:
                    fig = px.line(trend, x='Session', y=metric, markers=True, title=f"{t_club} Progress")
                    fig.update_traces(line_color='#00E676', line_width=4)
                    st.plotly_chart(style_fig(fig), use_container_width=True)
                else:
                    st.info("Need multiple sessions for trends.")

        # ================= TAB 4: MECHANICS (SMART LOFT) =================
        with tab4:
            if not master_df.empty:
                st.subheader("🔬 Swing Mechanics & Optimization")
                
                # 1. CLUB & LOFT SELECTION
                c_sel1, c_sel2 = st.columns([2,1])
                with c_sel1:
                    mech_club = st.selectbox("Analyze Club", means.index, key='m_club')
                with c_sel2:
                    # Get standard loft default
                    def_loft = STANDARD_LOFTS.get(mech_club, 34.0)
                    user_loft = st.number_input("Reference Loft (°)", value=def_loft, step=0.5, 
                                               help="Adjust this to match your specific club specs for accurate analysis.")
                
                mech_data = master_df[master_df['club'] == mech_club]
                
                # 2. METRICS ROW
                col_m1, col_m2, col_m3 = st.columns(3)
                
                # AoA
                if 'AOA (°)' in mech_data.columns:
                    val_aoa = mech_data['AOA (°)'].mean()
                    delta_txt, delta_col = check_range(mech_club, val_aoa, 0, handicap, user_loft) 
                    col_m1.metric("Angle of Attack", f"{val_aoa:.1f}°", delta=delta_txt, delta_color=delta_col)
                else:
                    col_m1.metric("Angle of Attack", "N/A")
                
                # Launch
                if 'Launch V (°)' in mech_data.columns:
                    val_launch = mech_data['Launch V (°)'].mean()
                    delta_txt, delta_col = check_range(mech_club, val_launch, 1, handicap, user_loft)
                    col_m2.metric("Launch Angle", f"{val_launch:.1f}°", delta=delta_txt, delta_color=delta_col)
                else:
                    col_m2.metric("Launch Angle", "N/A")
                
                # Spin
                if 'Spin (rpm)' in mech_data.columns:
                    val_spin = mech_data['Spin (rpm)'].mean()
                    delta_txt, delta_col = check_range(mech_club, val_spin, 2, handicap, user_loft)
                    col_m3.metric("Spin Rate", f"{val_spin:.0f} rpm", delta=delta_txt, delta_color=delta_col)
                else:
                    col_m3.metric("Spin Rate", "N/A")

                st.markdown("---")
                
                # 3. VISUALS
                col_chart_m1, col_chart_m2 = st.columns([2, 1])
                
                with col_chart_m1:
                    st.markdown("#### 🚀 Launch vs Spin Optimizer")
                    if 'Launch V (°)' in mech_data.columns and 'Spin (rpm)' in mech_data.columns:
                        fig_opt = px.scatter(mech_data, x='Spin (rpm)', y='Launch V (°)', 
                                           color='Session', size='Carry (yds)', title=f"Optimization: {mech_club}")
                        
                        # Get Dynamic Box for visual based on USER LOFT
                        ranges = get_dynamic_ranges(mech_club, handicap, user_loft)
                        launch_r = ranges[1]
                        spin_r = ranges[2]
                        
                        fig_opt.add_shape(type="rect",
                            x0=spin_r[0], y0=launch_r[0], x1=spin_r[1], y1=launch_r[1],
                            line=dict(color="Gold", width=2, dash="dot"),
                            fillcolor="Gold", opacity=0.1
                        )
                        st.plotly_chart(style_fig(fig_opt), use_container_width=True)

                with col_chart_m2:
                    st.markdown("#### ↩️ Face to Path")
                    if 'Club Path_Clean' in mech_data.columns and 'FTP_Clean' in mech_data.columns:
                        fig_path = px.scatter(mech_data, x='Club Path_Clean', y='FTP_Clean',
                                            color='Lateral_Clean', color_continuous_scale='RdBu_r', title="Shape Control")
                        fig_path.add_hline(y=0, line_color="white", opacity=0.2)
                        fig_path.add_vline(x=0, line_color="white", opacity=0.2)
                        st.plotly_chart(style_fig(fig_path), use_container_width=True)
else:
    st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        <h1>🏌️‍♂️ Ready to Practice?</h1>
        <p style="font-size: 18px; color: #888;">Drag and drop your FlightScope CSV files to unlock pro-level analytics.</p>
    </div>
    """, unsafe_allow_html=True)
