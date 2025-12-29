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
    .faq-box { background-color: #262730; padding: 20px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #4DD0E1; }
</style>
""", unsafe_allow_html=True)

# --- 1. INITIALIZE SESSION STATE ---
if 'master_df' not in st.session_state:
    st.session_state['master_df'] = pd.DataFrame()

# Default Standard Lofts
DEFAULT_LOFTS = {
    'Driver': 10.5, '3 Wood': 15.0, '5 Wood': 18.0, 'Hybrid': 21.0,
    '3 Iron': 21.0, '4 Iron': 24.0, '5 Iron': 27.0, '6 Iron': 30.0,
    '7 Iron': 34.0, '8 Iron': 38.0, '9 Iron': 42.0, 'PW': 46.0,
    'GW': 50.0, 'SW': 54.0, 'LW': 58.0
}

# Custom Sort Order for Display
CLUB_SORT_ORDER = [
    'Driver', '3 Wood', '5 Wood', '7 Wood', 'Hybrid', '2 Iron', '3 Iron', '4 Iron', 
    '5 Iron', '6 Iron', '7 Iron', '8 Iron', '9 Iron', 'PW', 'GW', 'SW', 'LW'
]

# Initialize My Bag in Session State if not present
if 'my_bag' not in st.session_state:
    st.session_state['my_bag'] = DEFAULT_LOFTS.copy()

# --- 2. DATABASES & HELPERS ---

def get_dynamic_ranges(club_name, handicap):
    c_lower = str(club_name).lower()
    tolerance = handicap * 0.1
    launch_help = 0 if handicap < 5 else (1.0 if handicap < 15 else 2.0)
    
    # LOOKUP USER LOFT from Session State
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
    
    if 'Altitude (ft)' not in df_clean.columns:
        df_clean['Altitude (ft)'] = 0.0
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

def check_range(club_name, value, metric_idx, handicap):
    ranges = get_dynamic_ranges(club_name, handicap) 
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
                    
                    # RESTORE BAG CONFIG
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
        st.info("Set your lofts here. They will be saved when you download your database.")
        
        bag_df = pd.DataFrame(list(st.session_state['my_bag'].items()), columns=['Club', 'Loft'])
        bag_df['SortIndex'] = bag_df['Club'].apply(lambda x: CLUB_SORT_ORDER.index(x) if x in CLUB_SORT_ORDER else 99)
        bag_df = bag_df.sort_values('SortIndex').drop(columns=['SortIndex'])
        
        edited_bag = st.data_editor(
            bag_df, 
            num_rows="dynamic", 
            hide_index=True, 
            key='bag_editor'
        )
        
        # Immediate Sync
        updated_dict = dict(zip(edited_bag['Club'], edited_bag['Loft']))
        if updated_dict != st.session_state['my_bag']:
            st.session_state['my_bag'] = updated_dict
            
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
        c_s1, c_s2 = st.columns(2)
        c_s1.markdown(f"<div class='stat-box'><b>Shots</b><br><span style='font-size:20px; color:#4DD0E1'>{tot_shots}</span></div>", unsafe_allow_html=True)
        c_s2.markdown(f"<div class='stat-box'><b>Sessions</b><br><span style='font-size:20px; color:#FF4081'>{tot_sess}</span></div>", unsafe_allow_html=True)

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
    tab_bag, tab_acc, tab_gap, tab_time, tab_mech, tab_comp, tab_faq = st.tabs(["🎒 My Bag", "🎯 Accuracy", "📏 Ranges", "📈 Timeline", "🔬 Mechanics", "⚔️ Compare", "❓ FAQ"])

    # ================= TAB: MY BAG =================
    with tab_bag:
        st.subheader("🎒 My Bag & Yardages")
        
        col_set1, col_set2 = st.columns([1, 3])
        with col_set1:
            play_alt = st.number_input("⛰️ Play Altitude (ft)", value=0, step=500, 
                                      help="Set this to the altitude of the course you are playing to see adjusted yardages.")
            
            alt_factor = 1 + (play_alt / 1000.0 * 0.011)
            if play_alt > 0:
                st.caption(f"Projecting distance boost: +{((alt_factor-1)*100):.1f}%")

        bag_stats = filtered_df.groupby('club').agg({
            'SL_Carry': 'mean',
            'SL_Total': 'mean',
            'Ball (mph)': 'mean',
            'Carry (yds)': 'max' 
        }).rename(columns={'Carry (yds)': 'Max Carry (Raw)'})
        
        # Sort using the Custom Sort Order
        bag_stats['SortIndex'] = bag_stats.index.map(lambda x: CLUB_SORT_ORDER.index(x) if x in CLUB_SORT_ORDER else 99)
        bag_stats = bag_stats.sort_values('SortIndex').drop(columns=['SortIndex'])
        
        bag_stats['Adj. Carry'] = bag_stats['SL_Carry'] * alt_factor
        bag_stats['Adj. Total'] = bag_stats['SL_Total'] * alt_factor
        
        st.write("---")
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
            avail_clubs = [c for c in CLUB_SORT_ORDER if c in filtered_df['club'].unique()]
            extra_clubs = [c for c in filtered_df['club'].unique() if c not in CLUB_SORT_ORDER]
            final_club_order = avail_clubs + extra_clubs
            
            selected_club = st.selectbox("Select Club", final_club_order, key='t1_club')
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
            st.subheader("🎒 Bag Gapping")
            filtered_df['SortIndex'] = filtered_df['club'].map(lambda x: CLUB_SORT_ORDER.index(x) if x in CLUB_SORT_ORDER else 99)
            filtered_df_sorted = filtered_df.sort_values('SortIndex')
            fig = px.box(filtered_df_sorted, x='club', y='Carry (yds)', color='club', points="all")
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
            avail_clubs = [c for c in CLUB_SORT_ORDER if c in filtered_df['club'].unique()]
            
            with c_sel1: mech_club = st.selectbox("Analyze Club", avail_clubs, key='m_club')
            with c_sel2: 
                curr_loft = st.session_state['my_bag'].get(mech_club, 30.0)
                st.metric("Bag Loft", f"{curr_loft}°")

            mech_data = filtered_df[filtered_df['club'] == mech_club]
            
            col_m1, col_m2, col_m3 = st.columns(3)
            if 'AOA (°)' in mech_data.columns:
                val_aoa = mech_data['AOA (°)'].mean()
                delta_txt, delta_col = check_range(mech_club, val_aoa, 0, handicap) 
                col_m1.metric("Angle of Attack", f"{val_aoa:.1f}°", delta=delta_txt, delta_color=delta_col)
            if 'Launch V (°)' in mech_data.columns:
                val_launch = mech_data['Launch V (°)'].mean()
                delta_txt, delta_col = check_range(mech_club, val_launch, 1, handicap)
                col_m2.metric("Launch Angle", f"{val_launch:.1f}°", delta=delta_txt, delta_color=delta_col)
            if 'Spin (rpm)' in mech_data.columns:
                val_spin = mech_data['Spin (rpm)'].mean()
                delta_txt, delta_col = check_range(mech_club, val_spin, 2, handicap)
                col_m3.metric("Spin Rate", f"{val_spin:.0f} rpm", delta=delta_txt, delta_color=delta_col)

            st.markdown("---")
            col_chart_m1, col_chart_m2 = st.columns([2, 1])
            with col_chart_m1:
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
        avail_clubs = [c for c in CLUB_SORT_ORDER if c in filtered_df['club'].unique()]
        comp_club = st.selectbox("Select Club to Compare", avail_clubs, key='c_club')
        
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

    # ================= TAB: FAQ =================
    with tab_faq:
        st.subheader("❓ FAQ & Help")
        
        with st.expander("🧹 What is 'Auto-Clean Outliers'?", expanded=True):
            st.markdown("""
            **The Problem:** Sometimes the radar misreads a shot (e.g., a 'ghost' 400-yard 7-iron) or you duff one 20 yards. These mess up your averages.
            
            **The Solution:** We use the **IQR (Interquartile Range)** method:
            1. We calculate the middle 50% of your shots.
            2. We remove any shot that is exceedingly far outside that range (statistical anomalies).
            3. This gives you a 'True Average' of your *solid* strikes.
            """)
            
        with st.expander("🌊 How does 'Sea Level' Normalization work?", expanded=True):
            st.markdown("""
            **The Physics:** Golf balls fly further at higher altitudes because the air is thinner (less drag).
            
            **The Math:** We use the `Altitude (ft)` recorded by your Mevo+ for every single shot. 
            * We apply a correction factor of approx **1.1% per 1,000 ft**.
            * **Example:** If you play in Denver (5,280 ft), your ball flies ~6% further than in Florida.
            * The app strips away that 6% bonus so you can compare your "Denver 7-iron" to your "Florida 7-iron" fairly.
            """)
        
        with st.expander("⚙️ Why do I need to set 'My Bag' lofts?", expanded=True):
            st.markdown("""
            **Context:** A '7-Iron' isn't a standard unit of measurement.
            * A modern 'Game Improvement' 7-iron might be **28°**.
            * A traditional 'Blade' 7-iron might be **34°**.
            
            **The App:** To tell you if your Launch Angle is "Optimal" or "Too Low," the app needs to know the loft of the club you are holding. 
            Setting this once ensures the 'Swing Mechanics' tab gives you accurate advice.
            """)

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
    st.info("👈 **Get Started:** Open the Sidebar to **Configure Your Bag** and **Add a New Session**.")
