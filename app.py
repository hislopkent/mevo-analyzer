import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE SETUP ---
st.set_page_config(page_title="Mevo+ Pro Analytics", layout="wide", page_icon="⛳")

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #4DD0E1; }
    h1, h2, h3 { color: #FAFAFA; font-family: 'Helvetica Neue', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.title("⛳ Mevo+ Pro Analytics")

# --- 1. LOGIC & HELPERS ---

def get_target_settings(club_name, handicap):
    """
    Returns the target shape and size based on Club + Handicap.
    - Scoring Irons (8i+) -> Circle (Virtual Green)
    - Woods/Long Irons -> Lane (Fairway)
    """
    club = str(club_name).lower()
    
    # 1. Define Groups
    scoring_clubs = ['8 iron', '9 iron', 'pw', 'gw', 'sw', 'lw', 'wedge', 'sand', 'lob', 'gap']
    
    # 2. Base Tolerance (in yards) for a Scratch Golfer (0 Handicap)
    # 'Lane' = Lateral distance from center (e.g., 20 means +/- 20y wide)
    # 'Circle' = Radius from center
    
    is_scoring = any(x in club for x in scoring_clubs)
    
    if is_scoring:
        # VIRTUAL GREEN (Circle)
        shape = 'circle'
        base_size = 5.0  # 5 yard radius for a pro (very tight)
        expansion = handicap * 0.4  # 20 hcp adds 8 yards -> 13y radius
    else:
        # FAIRWAY LANE (Rectangle)
        shape = 'lane'
        # Tighter dispersion expected for irons vs driver
        if 'driver' in club:
            base_size = 15.0 # +/- 15y (30y wide fairway)
            expansion = handicap * 0.8 # 20 hcp adds 16y -> +/- 31y (62y wide)
        elif 'wood' in club or 'hybrid' in club:
            base_size = 12.0
            expansion = handicap * 0.7
        else:
            # Mid/Long Irons (2i - 7i)
            base_size = 10.0
            expansion = handicap * 0.6
            
    total_tolerance = base_size + expansion
    return shape, total_tolerance

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

    dir_cols = ['Lateral (yds)', 'Swing H (°)', 'Launch H (°)', 'Spin Axis (°)']
    for col in dir_cols:
        if col in df_clean.columns:
            clean_col_name = col.replace(' (yds)', '').replace(' (°)', '') + '_Clean'
            if 'Lateral' in col:
                df_clean['Lateral_Clean'] = df_clean[col].apply(parse_lr)
            else:
                df_clean[clean_col_name] = df_clean[col].apply(parse_lr)

    numeric_cols = ['Carry (yds)', 'Total (yds)', 'Ball (mph)', 'Club (mph)', 'Smash', 'Spin (rpm)', 'Height (ft)']
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

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("1. Upload")
    uploaded_files = st.file_uploader("CSV Files", accept_multiple_files=True, type='csv')
    
    st.header("2. Environment")
    env_mode = st.radio("Filter:", ["All", "Outdoor Only", "Indoor Only"], index=0)
    
    st.header("3. Player Profile")
    handicap = st.number_input("Handicap", 0, 54, 15, help="Determines the size of target zones.")
    smash_cap = st.slider("Max Smash Cap", 1.40, 1.65, 1.52, 0.01)
    
    st.header("4. Settings")
    remove_bad_shots = st.checkbox("Auto-Clean Outliers", value=True)

# --- 3. MAIN APP ---
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
            
        # Download
        st.sidebar.markdown("---")
        st.sidebar.download_button("📥 Download Database", master_df.to_csv(index=False).encode('utf-8'), "mevo_db.csv", "text/csv")

        # --- TABS ---
        st.write("---")
        tab1, tab2, tab3 = st.tabs(["🎯 Target & Accuracy", "🎒 Gapping", "📈 Trends"])

        def style_fig(fig):
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return fig

        # ================= TAB 1: TARGET ANALYZER =================
        with tab1:
            if not master_df.empty:
                # Order clubs
                club_order = master_df.groupby('club')['Carry (yds)'].mean().sort_values(ascending=False).index
                selected_club = st.selectbox("Select Club", club_order)
                subset = master_df[master_df['club'] == selected_club]
                
                if len(subset) > 0:
                    # GET DYNAMIC TARGET
                    target_shape, target_val = get_target_settings(selected_club, handicap)
                    avg_carry = subset['Carry (yds)'].mean()
                    
                    # CALCULATE ACCURACY
                    if 'Lateral_Clean' in subset.columns:
                        if target_shape == 'circle':
                            # Distance from center point (0, avg_carry)
                            # Pythagoras: sqrt(x^2 + (y - center_y)^2)
                            # NOTE: For simple "Green" hit stats, we often just check if it landed 
                            # within the radius of the AVERAGE distance, OR the specific shot distance?
                            # Usually "Virtual Green" assumes you hit it the correct distance. 
                            # Let's calculate distance from the geometric center of the group (Precision)
                            # OR distance from the ideal target line + avg carry?
                            # User asked for "Virtual Green". We'll assume the target is at (0, avg_carry).
                            
                            dist_from_target = np.sqrt(
                                subset['Lateral_Clean']**2 + 
                                (subset['Carry (yds)'] - avg_carry)**2
                            )
                            on_target = subset[dist_from_target <= target_val]
                            target_label = f"Green Radius: {target_val:.1f}y"
                        else:
                            # Lane Logic: Just check Lateral
                            on_target = subset[abs(subset['Lateral_Clean']) <= target_val]
                            target_label = f"Lane Width: ±{target_val:.1f}y"

                        acc_score = (len(on_target) / len(subset)) * 100
                    else:
                        acc_score = 0
                        target_label = "N/A"

                    # METRICS
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy Score", f"{acc_score:.0f}%", target_label)
                    c2.metric("Avg Carry", f"{avg_carry:.1f}")
                    c3.metric("Ball Speed", f"{subset['Ball (mph)'].mean():.1f}")
                    c4.metric("Smash", f"{subset['Smash'].mean():.2f}")

                    # PLOT
                    c_chart1, c_chart2 = st.columns([3, 1])
                    with c_chart1:
                        if 'Lateral_Clean' in subset.columns:
                            # Color by Fade/Draw
                            subset['Shape'] = np.where(subset['Lateral_Clean'] > 0, 'Fade (R)', 'Draw (L)')
                            
                            fig = px.scatter(
                                subset, x='Lateral_Clean', y='Carry (yds)', 
                                color='Shape',
                                color_discrete_map={'Fade (R)': '#00E5FF', 'Draw (L)': '#FF4081'},
                                title=f"Dispersion vs {target_shape.title()} Target"
                            )
                            
                            # DRAW TARGET
                            if target_shape == 'circle':
                                fig.add_shape(type="circle",
                                    xref="x", yref="y",
                                    x0=-target_val, y0=avg_carry - target_val,
                                    x1=target_val, y1=avg_carry + target_val,
                                    line_color="#00E676", fillcolor="#00E676", opacity=0.2
                                )
                            else:
                                # Infinite Lane or bounded by max/min?
                                # Let's draw a box covering the range of shots
                                y_min = subset['Carry (yds)'].min() - 10
                                y_max = subset['Carry (yds)'].max() + 10
                                fig.add_shape(type="rect",
                                    x0=-target_val, y0=y_min,
                                    x1=target_val, y1=y_max,
                                    line_color="#00E676", fillcolor="#00E676", opacity=0.1
                                )
                                fig.add_vline(x=-target_val, line_dash="dash", line_color="#00E676")
                                fig.add_vline(x=target_val, line_dash="dash", line_color="#00E676")

                            fig.add_vline(x=0, line_color="white", opacity=0.2)
                            fig.update_xaxes(range=[-60, 60], title="Left <---> Right")
                            fig.update_yaxes(title="Carry (yds)")
                            fig.update_traces(marker=dict(size=12, line=dict(width=1, color='White')))
                            st.plotly_chart(style_fig(fig), use_container_width=True)

                    with c_chart2:
                        if 'Shot Type' in subset.columns:
                            counts = subset['Shot Type'].value_counts().reset_index()
                            counts.columns = ['Type', 'Count']
                            fig_pie = px.pie(counts, values='Count', names='Type', hole=0.5, 
                                            color_discrete_sequence=px.colors.qualitative.Pastel)
                            fig_pie.update_layout(showlegend=False, annotations=[dict(text='Shape', x=0.5, y=0.5, showarrow=False)])
                            st.plotly_chart(style_fig(fig_pie), use_container_width=True)

        # ================= TAB 2: GAPPING =================
        with tab2:
            if not master_df.empty:
                st.subheader("🎒 Bag Gapping")
                means = master_df.groupby("club")["Carry (yds)"].mean().sort_values(ascending=False)
                fig = px.box(master_df, x='club', y='Carry (yds)', color='club', 
                             category_orders={'club': means.index}, points="all")
                fig.update_layout(showlegend=False)
                st.plotly_chart(style_fig(fig), use_container_width=True)

        # ================= TAB 3: TRENDS =================
        with tab3:
            if not master_df.empty:
                st.subheader("📈 Trends")
                c_t1, c_t2 = st.columns(2)
                with c_t1: t_club = st.selectbox("Club", means.index, key='t_club')
                with c_t2: metric = st.selectbox("Metric", ['Ball (mph)', 'Carry (yds)', 'Smash'])
                
                trend = master_df[master_df['club'] == t_club].groupby('Session')[metric].mean().reset_index().sort_values('Session')
                if len(trend) > 1:
                    fig = px.line(trend, x='Session', y=metric, markers=True, title=f"{t_club} Progress")
                    fig.update_traces(line_color='#00E676', line_width=4)
                    st.plotly_chart(style_fig(fig), use_container_width=True)
                else:
                    st.info("Need multiple sessions for trends.")
