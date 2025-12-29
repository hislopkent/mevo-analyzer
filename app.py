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

# --- 1. OPTIMAL RANGES DATABASE (Tour Averages) ---
# Format: Club: (Min AoA, Max AoA), (Min Launch, Max Launch), (Min Spin, Max Spin)
OPTIMAL_RANGES = {
    'Driver':  ((0.0, 5.0),   (10.0, 16.0), (1800, 2800)),
    '3 Wood':  ((-2.0, 0.0),  (9.0, 13.0),  (3000, 4000)),
    '5 Wood':  ((-3.0, -1.0), (10.0, 14.0), (3500, 4500)),
    'Hybrid':  ((-3.0, -1.0), (11.0, 15.0), (4000, 5000)),
    '3 Iron':  ((-3.0, -1.0), (10.0, 14.0), (4000, 5000)),
    '4 Iron':  ((-4.0, -2.0), (11.0, 15.0), (4500, 5500)),
    '5 Iron':  ((-4.0, -2.0), (12.0, 16.0), (5000, 6000)),
    '6 Iron':  ((-5.0, -3.0), (14.0, 18.0), (5500, 6500)),
    '7 Iron':  ((-5.0, -3.0), (16.0, 20.0), (6000, 7000)),
    '8 Iron':  ((-5.0, -3.0), (18.0, 22.0), (7000, 8000)),
    '9 Iron':  ((-6.0, -4.0), (20.0, 24.0), (8000, 9000)),
    'PW':      ((-6.0, -4.0), (24.0, 28.0), (8500, 9500)),
    'GW':      ((-6.0, -4.0), (26.0, 30.0), (9000, 10000)),
    'SW':      ((-6.0, -4.0), (28.0, 32.0), (9000, 10500)),
    'LW':      ((-6.0, -4.0), (30.0, 35.0), (9000, 11000)),
}

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

    # L/R Columns
    dir_cols = ['Lateral (yds)', 'Swing H (°)', 'Launch H (°)', 'Spin Axis (°)', 'Club Path (°)', 'FTP (°)']
    for col in dir_cols:
        if col in df_clean.columns:
            clean_col_name = col.replace(' (yds)', '').replace(' (°)', '') + '_Clean'
            if 'Lateral' in col:
                df_clean['Lateral_Clean'] = df_clean[col].apply(parse_lr)
            else:
                df_clean[clean_col_name] = df_clean[col].apply(parse_lr)

    # Numeric Columns
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

def check_range(club_name, value, metric_idx):
    """
    Checks if value is within optimal range for club.
    metric_idx: 0=AoA, 1=Launch, 2=Spin
    Returns: (delta_string, delta_color)
    """
    # Normalize club name mapping
    c_map = 'Driver' # Default
    c_lower = str(club_name).lower()
    
    if 'driver' in c_lower: c_map = 'Driver'
    elif '3 wood' in c_lower: c_map = '3 Wood'
    elif '5 wood' in c_lower: c_map = '5 Wood'
    elif 'hybrid' in c_lower: c_map = 'Hybrid'
    elif 'p' in c_lower and 'w' in c_lower: c_map = 'PW'
    elif 'g' in c_lower and 'w' in c_lower: c_map = 'GW'
    elif 's' in c_lower and 'w' in c_lower: c_map = 'SW'
    elif 'l' in c_lower and 'w' in c_lower: c_map = 'LW'
    else:
        # Irons 3-9
        for i in range(3, 10):
            if f"{i}" in c_lower:
                c_map = f"{i} Iron"
                break
    
    if c_map not in OPTIMAL_RANGES:
        return None, "off"
        
    ranges = OPTIMAL_RANGES[c_map][metric_idx]
    min_v, max_v = ranges
    
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
    handicap = st.number_input("Handicap", 0, 54, 15)
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

        # --- TABS ---
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

        # ================= TAB 4: MECHANICS (UPDATED) =================
        with tab4:
            if not master_df.empty:
                st.subheader("🔬 Swing Mechanics & Optimization")
                
                mech_club = st.selectbox("Analyze Club", means.index, key='m_club')
                mech_data = master_df[master_df['club'] == mech_club]
                
                # METRICS ROW
                col_m1, col_m2, col_m3 = st.columns(3)
                
                # 1. AoA
                if 'AOA (°)' in mech_data.columns:
                    val_aoa = mech_data['AOA (°)'].mean()
                    delta_txt, delta_col = check_range(mech_club, val_aoa, 0) # 0 is index for AoA
                    col_m1.metric("Angle of Attack", f"{val_aoa:.1f}°", delta=delta_txt, delta_color=delta_col)
                else:
                    col_m1.metric("Angle of Attack", "N/A")
                
                # 2. Launch
                if 'Launch V (°)' in mech_data.columns:
                    val_launch = mech_data['Launch V (°)'].mean()
                    delta_txt, delta_col = check_range(mech_club, val_launch, 1) # 1 is Launch
                    col_m2.metric("Launch Angle", f"{val_launch:.1f}°", delta=delta_txt, delta_color=delta_col)
                else:
                    col_m2.metric("Launch Angle", "N/A")
                
                # 3. Spin
                if 'Spin (rpm)' in mech_data.columns:
                    val_spin = mech_data['Spin (rpm)'].mean()
                    delta_txt, delta_col = check_range(mech_club, val_spin, 2) # 2 is Spin
                    col_m3.metric("Spin Rate", f"{val_spin:.0f} rpm", delta=delta_txt, delta_color=delta_col)
                else:
                    col_m3.metric("Spin Rate", "N/A")

                st.markdown("---")
                
                # CHARTS
                col_chart_m1, col_chart_m2 = st.columns([2, 1])
                
                with col_chart_m1:
                    st.markdown("#### 🚀 Launch vs Spin Optimizer")
                    if 'Launch V (°)' in mech_data.columns and 'Spin (rpm)' in mech_data.columns:
                        fig_opt = px.scatter(mech_data, x='Spin (rpm)', y='Launch V (°)', 
                                           color='Session', size='Carry (yds)',
                                           title=f"Optimization: {mech_club}")
                        
                        # Add Optimal Box if available
                        # Find range for this club
                        # Reuse mapping logic broadly
                        c_lower = str(mech_club).lower()
                        # Simple lookup for box drawing
                        opt_box = None
                        if 'driver' in c_lower: opt_box = OPTIMAL_RANGES['Driver']
                        elif '7 iron' in c_lower: opt_box = OPTIMAL_RANGES['7 Iron']
                        
                        # If we have a range, draw the box
                        # Note: We just look up generic optimal for the box visual
                        # For robustness, we can just rely on the metrics above, 
                        # but adding the Driver box is nice.
                        if 'driver' in c_lower:
                             fig_opt.add_shape(type="rect",
                                x0=1800, y0=10, x1=2800, y1=16,
                                line=dict(color="Gold", width=2, dash="dot"),
                                fillcolor="Gold", opacity=0.1
                            )
                        
                        st.plotly_chart(style_fig(fig_opt), use_container_width=True)

                with col_chart_m2:
                    st.markdown("#### ↩️ Face to Path")
                    if 'Club Path_Clean' in mech_data.columns and 'FTP_Clean' in mech_data.columns:
                        fig_path = px.scatter(mech_data, x='Club Path_Clean', y='FTP_Clean',
                                            color='Lateral_Clean',
                                            color_continuous_scale='RdBu_r',
                                            title="Shape Control")
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
