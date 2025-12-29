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
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #4DD0E1; }
    h1, h2, h3 { color: #FAFAFA; font-family: 'Helvetica Neue', sans-serif; }
    .feature-card { background-color: #1E222B; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #333; }
    .stat-box { background-color: #1E222B; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 10px; border: 1px solid #444; }
</style>
""", unsafe_allow_html=True)

# --- 1. SESSION STATE ---
if 'master_df' not in st.session_state:
    st.session_state['master_df'] = pd.DataFrame()

# --- 2. DATABASES & HELPERS ---
STANDARD_LOFTS = {
    'Driver': 10.5, '3 Wood': 15.0, '5 Wood': 18.0, 'Hybrid': 21.0,
    '3 Iron': 21.0, '4 Iron': 24.0, '5 Iron': 27.0, '6 Iron': 30.0,
    '7 Iron': 34.0, '8 Iron': 38.0, '9 Iron': 42.0, 'PW': 46.0,
    'GW': 50.0, 'SW': 54.0, 'LW': 58.0
}

def get_dynamic_ranges(club_name, handicap, user_loft=None):
    c_lower = str(club_name).lower()
    tolerance = handicap * 0.1
    launch_help = 0 if handicap < 5 else (1.0 if handicap < 15 else 2.0)
    target_loft = user_loft if user_loft else STANDARD_LOFTS.get(club_name, 30.0)

    if 'driver' in c_lower:
        aoa = (-2.0 - (tolerance*0.2), 5.0 + (tolerance*0.2)) 
        launch = (10.0 + launch_help, 16.0 + launch_help + (tolerance*0.2))
        spin = (1800, 2800 + (handicap * 40))
    elif 'wood' in c_lower or 'hybrid' in c_lower:
        aoa = (-4.0, 1.0 + tolerance)
        l_center = target_loft * 0.7 
        launch = (l_center - 2.0, l_center + 2.0 + launch_help)
        spin = (target_loft * 200, target_loft * 280)
    else:
        target_aoa = -1.0 - (target_loft / 12.0)
        aoa = (target_aoa - 2.0 - tolerance, -0.5)
        l_min = (target_loft * 0.45) - 1.0
        l_max = (target_loft * 0.55) + 1.0 + launch_help
        launch = (l_min - tolerance, l_max + tolerance)
        s_base = target_loft * 210
        spin = (s_base - 1000 - (tolerance*100), s_base + 1000 + (tolerance*100))

    return aoa, launch, spin

def clean_mevo_data(df, filename, selected_date):
    df_clean = df[df['Shot'].astype(str).str.isdigit()].copy()
    df_clean['Session'] = filename.replace('.csv', '')
    df_clean['Date'] = pd.to_datetime(selected_date)
    
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
    
    # --- ALTITUDE NORMALIZATION ---
    # Parse Altitude
    if 'Altitude (ft)' not in df_clean.columns:
        df_clean['Altitude (ft)'] = 0.0
    else:
        df_clean['Altitude (ft)'] = df_clean['Altitude (ft)'].astype(str).str.replace(' ft','').str.replace(',','')
        df_clean['Altitude (ft)'] = pd.to_numeric(df_clean['Altitude (ft)'], errors='coerce').fillna(0.0)

    # Calculate Sea Level Carry (Normalize)
    # Approx 1.1% gain per 1000ft. So SL = Actual / (1 + (Alt/1000 * 0.011))
    df_clean['SL_Carry'] = df_clean['Carry (yds)'] / (1 + (df_clean['Altitude (ft)'] / 1000.0 * 0.011))
    df_clean['SL_Total'] = df_clean['Total (yds)'] / (1 + (df_clean['Altitude (ft)'] / 1000.0 * 0.011))

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
                    new_data.append(clean)
                except Exception as e: st.error(f"Error {f.name}: {e}")
            if new_data:
                batch_df = pd.concat(new_data, ignore_index=True)
                st.session_state['master_df'] = pd.concat([st.session_state['master_df'], batch_df], ignore_index=True)
                st.success(f"Added {len(batch_df)} shots!")
                st.rerun()

    st.markdown("---")
    
    # SUMMARY STATS (Moved here from Trophy Room)
    if not st.session_state['master_df'].empty:
        st.subheader("📊 Statistics")
        tot_shots = len(st.session_state['master_df'])
        tot_sess = st.session_state['master_df']['Date'].nunique()
        
        c_s1, c_s2 = st.columns(2)
        c_s1.markdown(f"<div class='stat-box'><b>Shots</b><br><span style='font-size:20px; color:#4DD0E1'>{tot_shots}</span></div>", unsafe_allow_html=True)
        c_s2.markdown(f"<div class='stat-box'><b>Sessions</b><br><span style='font-size:20px; color:#FF4081'>{tot_sess}</span></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.header("3. Player Config")
    env_mode = st.radio("Filter:", ["All", "Outdoor Only", "Indoor Only"], index=0)
    handicap = st.number_input("Handicap", 0, 54, 15)
    smash_cap = st.slider("Max Smash Cap", 1.40, 1.65, 1.52, 0.01)
    remove_bad_shots = st.checkbox("Auto-Clean Outliers", value=True)

# --- 4. MAIN APP LOGIC ---
master_df = st.session_state['master_df']

if not master_df.empty:
    st.title("⛳ Homegrown FS Pro Analytics")
    
    filtered_df = master_df.copy()
    
    # 1. Filters
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
    tab_bag, tab_acc, tab_gap, tab_time, tab_mech, tab_comp = st.tabs(["🎒 My Bag", "🎯 Accuracy", "📏 Ranges", "📈 Timeline", "🔬 Mechanics", "⚔️ Compare"])

    # ================= TAB: MY BAG (NEW) =================
    with tab_bag:
        st.subheader("🎒 My Bag & Yardages")
        
        # 1. Altitude Adjuster
        col_set1, col_set2 = st.columns([1, 3])
        with col_set1:
            play_alt = st.number_input("⛰️ Play Altitude (ft)", value=0, step=500, 
                                      help="Set this to the altitude of the course you are playing to see adjusted yardages.")
            
            # Altitude Factor: (1 + (Alt/1000 * 0.011))
            alt_factor = 1 + (play_alt / 1000.0 * 0.011)
            if play_alt > 0:
                st.caption(f"Projecting distance boost: +{((alt_factor-1)*100):.1f}%")

        # 2. Calculate Bag Data
        # We group by club and calculate stats based on Sea Level (SL) data, then project
        bag_stats = filtered_df.groupby('club').agg({
            'SL_Carry': 'mean',
            'SL_Total': 'mean',
            'Ball (mph)': 'mean',
            'Carry (yds)': 'max' # Raw Max Carry for Personal Best
        }).rename(columns={'Carry (yds)': 'Max Carry (Raw)'})
        
        # Sort by distance
        bag_stats = bag_stats.sort_values('SL_Carry', ascending=False)
        
        # Apply Projection to Averages
        bag_stats['Adj. Carry'] = bag_stats['SL_Carry'] * alt_factor
        bag_stats['Adj. Total'] = bag_stats['SL_Total'] * alt_factor
        
        # Formatting for Display
        display_bag = pd.DataFrame()
        display_bag['Club'] = bag_stats.index
        display_bag['Avg Carry'] = bag_stats['Adj. Carry'].map('{:.1f}'.format)
        display_bag['Avg Total'] = bag_stats['Adj. Total'].map('{:.1f}'.format)
        display_bag['Ball Speed'] = bag_stats['Ball (mph)'].map('{:.1f}'.format)
        display_bag['Max Carry (Raw)'] = bag_stats['Max Carry (Raw)'].map('{:.1f}'.format)
        
        # 3. Display Grid of Cards
        st.write("---")
        # Use columns to create a grid layout
        cols = st.columns(4)
        for i, (index, row) in enumerate(bag_stats.iterrows()):
            with cols[i % 4]:
                st.markdown(f"""
                <div style="background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #444; margin-bottom: 10px;">
                    <h3 style="margin:0; color: #4DD0E1;">{index}</h3>
                    <h2 style="margin:0; font-size: 32px; color: #FFF;">{row['Adj. Carry']:.0f}<span style="font-size:16px; color:#888"> yds</span></h2>
                    <p style="margin:0; color: #BBB;">Total: <b>{row['Adj. Total']:.0f}</b></p>
                    <hr style="border-color: #444; margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #888;">
                        <span>Speed: {row['Ball (mph)']:.0f} mph</span>
                        <span style="color: #FFD700;">Max: {row['Max Carry (Raw)']:.0f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ================= TAB: ACCURACY =================
    with tab_acc:
        if len(filtered_df) > 0:
            club_order = filtered_df.groupby('club')['Carry (yds)'].mean().sort_values(ascending=False).index
            selected_club = st.selectbox("Select Club", club_order, key='t1_club')
            subset = filtered_df[filtered_df['club'] == selected_club]
            
            if len(subset) > 0:
                is_scoring = any(x in str(selected_club).lower() for x in ['8','9','p','w','s','l','g'])
                target_val = 5.0 + (handicap * 0.4) if is_scoring else 15.0 + (handicap * 0.8)
                target_type = "Green Radius" if is_scoring else "Lane Width"
                
                if 'Lateral_Clean' in subset.columns:
                    on_target = subset[abs(subset['Lateral_Clean']) <= target_val]
                    acc_score = (len(on_target) / len(subset)) * 100
                else: acc_score = 0
                
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
                            hover_data=['Date'], title=f"Dispersion: {selected_club}")
                        
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
        else: st.info("No data available.")

    # ================= TAB: GAPPING =================
    with tab_gap:
        if len(filtered_df) > 0:
            st.subheader("📏 Ranges (Box Plot)")
            means = filtered_df.groupby("club")["Carry (yds)"].mean().sort_values(ascending=False)
            fig = px.box(filtered_df, x='club', y='Carry (yds)', color='club', category_orders={'club': means.index}, points="all")
            st.plotly_chart(style_fig(fig), use_container_width=True)

    # ================= TAB: TIMELINE =================
    with tab_time:
        if len(filtered_df) > 0:
            st.subheader("📈 Timeline")
            c_t1, c_t2 = st.columns(2)
            with c_t1: t_club = st.selectbox("Club", means.index, key='t_club')
            with c_t2: metric = st.selectbox("Metric", ['Ball (mph)', 'Carry (yds)', 'Club (mph)', 'Smash'])
            
            if 'Date' in filtered_df.columns:
                trend = filtered_df[filtered_df['club'] == t_club].groupby('Date')[metric].mean().reset_index().sort_values('Date')
                if len(trend) > 1:
                    fig = px.line(trend, x='Date', y=metric, markers=True, title=f"{t_club} Progress")
                    fig.update_traces(line_color='#00E676', line_width=4)
                    st.plotly_chart(style_fig(fig), use_container_width=True)
                else: st.info("Add data from 2+ dates.")
            else: st.warning("No Date info.")

    # ================= TAB: MECHANICS =================
    with tab_mech:
        if len(filtered_df) > 0:
            st.subheader("🔬 Swing Mechanics")
            c_sel1, c_sel2 = st.columns([2,1])
            with c_sel1: mech_club = st.selectbox("Analyze Club", means.index, key='m_club')
            with c_sel2: 
                def_loft = STANDARD_LOFTS.get(mech_club, 34.0)
                user_loft = st.number_input("Reference Loft (°)", value=def_loft, step=0.5)
            
            mech_data = filtered_df[filtered_df['club'] == mech_club]
            
            col_m1, col_m2, col_m3 = st.columns(3)
            if 'AOA (°)' in mech_data.columns:
                val_aoa = mech_data['AOA (°)'].mean()
                delta_txt, delta_col = check_range(mech_club, val_aoa, 0, handicap, user_loft) 
                col_m1.metric("Angle of Attack", f"{val_aoa:.1f}°", delta=delta_txt, delta_color=delta_col)
            if 'Launch V (°)' in mech_data.columns:
                val_launch = mech_data['Launch V (°)'].mean()
                delta_txt, delta_col = check_range(mech_club, val_launch, 1, handicap, user_loft)
                col_m2.metric("Launch Angle", f"{val_launch:.1f}°", delta=delta_txt, delta_color=delta_col)
            if 'Spin (rpm)' in mech_data.columns:
                val_spin = mech_data['Spin (rpm)'].mean()
                delta_txt, delta_col = check_range(mech_club, val_spin, 2, handicap, user_loft)
                col_m3.metric("Spin Rate", f"{val_spin:.0f} rpm", delta=delta_txt, delta_color=delta_col)

            st.markdown("---")
            col_chart_m1, col_chart_m2 = st.columns([2, 1])
            with col_chart_m1:
                # Trajectory Window
                if 'Height (ft)' in mech_data.columns:
                    st.markdown("#### ✈️ Trajectory Window (Height vs Carry)")
                    fig_traj = px.scatter(mech_data, x='Carry (yds)', y='Height (ft)', color='Session',
                                        title=f"Trajectory Control: {mech_club}")
                    fig_traj.add_shape(type="rect", x0=mech_data['Carry (yds)'].min(), y0=80, 
                                     x1=mech_data['Carry (yds)'].max(), y1=110,
                                     line=dict(color="Gold", width=0), fillcolor="Gold", opacity=0.1)
                    st.plotly_chart(style_fig(fig_traj), use_container_width=True)
                elif 'Launch V (°)' in mech_data.columns and 'Spin (rpm)' in mech_data.columns:
                    fig_opt = px.scatter(mech_data, x='Spin (rpm)', y='Launch V (°)', color='Session', size='Carry (yds)')
                    st.plotly_chart(style_fig(fig_opt), use_container_width=True)

            with col_chart_m2:
                if 'Club Path_Clean' in mech_data.columns and 'FTP_Clean' in mech_data.columns:
                    st.markdown("#### ↩️ Shape Control")
                    fig_path = px.scatter(mech_data, x='Club Path_Clean', y='FTP_Clean',
                                        color='Lateral_Clean', color_continuous_scale='RdBu_r')
                    fig_path.add_hline(y=0, line_color="white", opacity=0.2)
                    fig_path.add_vline(x=0, line_color="white", opacity=0.2)
                    st.plotly_chart(style_fig(fig_path), use_container_width=True)

    # ================= TAB: COMPARISON =================
    with tab_comp:
        st.subheader("⚔️ Comparison Lab")
        comp_club = st.selectbox("Select Club to Compare", means.index, key='c_club')
        club_data = filtered_df[filtered_df['club'] == comp_club].copy()
        
        if 'Date' in club_data.columns:
            club_data['SessionLabel'] = club_data['Date'].dt.strftime('%Y-%m-%d') + ": " + club_data['Session']
        else: club_data['SessionLabel'] = club_data['Session']
            
        unique_sessions = club_data['SessionLabel'].unique()
        
        if len(unique_sessions) >= 2:
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1: sess_a = st.selectbox("Session A (Baseline)", unique_sessions, index=0)
            with col_sel2: sess_b = st.selectbox("Session B (Challenger)", unique_sessions, index=1)
                
            if sess_a != sess_b:
                data_a = club_data[club_data['SessionLabel'] == sess_a]
                data_b = club_data[club_data['SessionLabel'] == sess_b]
                
                m1, m2, m3, m4 = st.columns(4)
                diff_carry = data_b['Carry (yds)'].mean() - data_a['Carry (yds)'].mean()
                m1.metric("Carry Gain", f"{diff_carry:+.1f} yds", delta=diff_carry)
                
                diff_ball = data_b['Ball (mph)'].mean() - data_a['Ball (mph)'].mean()
                m2.metric("Ball Speed", f"{diff_ball:+.1f} mph", delta=diff_ball)
                
                diff_disp = data_b['Lateral_Clean'].abs().mean() - data_a['Lateral_Clean'].abs().mean()
                m3.metric("Dispersion", f"{diff_disp:+.1f} yds", delta=-diff_disp, delta_color="inverse")
                
                diff_smash = data_b['Smash'].mean() - data_a['Smash'].mean()
                m4.metric("Smash Eff.", f"{diff_smash:+.2f}", delta=diff_smash)
                
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=data_a['Carry (yds)'], name='Session A', opacity=0.75, marker_color='#FF4081'))
                fig_hist.add_trace(go.Histogram(x=data_b['Carry (yds)'], name='Session B', opacity=0.75, marker_color='#00E5FF'))
                fig_hist.update_layout(barmode='overlay', title=f"Carry Distance Distribution")
                st.plotly_chart(style_fig(fig_hist), use_container_width=True)
            else: st.warning("Select different sessions.")
        else: st.warning("Need 2+ sessions.")

else:
    # --- HOMEGROWN WELCOME PAGE ---
    st.markdown("""
    <div style="text-align: center; padding: 50px 0;">
        <h1 style="font-size: 60px; font-weight: 700; background: -webkit-linear-gradient(45deg, #00E5FF, #FF4081); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Homegrown FS Pro Analytics
        </h1>
        <p style="font-size: 24px; color: #B0B3B8; margin-top: 10px;">
            Turn your <b>Mevo+ and Mevo Gen 2</b> data into Tour-level insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="feature-card"><h3>🎯 Precision Targeting</h3><p style="color:#888">Visualize dispersion with dynamic target lanes based on your handicap.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="feature-card"><h3>🎒 My Bag & Ranges</h3><p style="color:#888">Project your adjusted carry distances for any altitude.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="feature-card"><h3>📈 Historical Trends</h3><p style="color:#888">Track ball speed, accuracy, and consistency improvements over time.</p></div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("👈 **Get Started:** Open the Sidebar to **Add a New Session** or **Restore your Database**.")
