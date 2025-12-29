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

# --- 1. DATA PROCESSING ---
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

    # Numeric Columns (Added AOA, Launch V)
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

# --- 2. SIDEBAR ---
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
                    # Target Logic
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
                            
                            # Draw Target Zone
                            y_min, y_max = subset['Carry (yds)'].min()-10, subset['Carry (yds)'].max()+10
                            avg_c = subset['Carry (yds)'].mean()
                            
                            if is_scoring: # Circle
                                fig.add_shape(type="circle", x0=-target_val, y0=avg_c-target_val, x1=target_val, y1=avg_c+target_val,
                                    line_color="#00E676", fillcolor="#00E676", opacity=0.2)
                            else: # Lane
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

        # ================= TAB 4: MECHANICS (NEW) =================
        with tab4:
            if not master_df.empty:
                st.subheader("🔬 Swing Mechanics & Optimization")
                
                # Club Select
                mech_club = st.selectbox("Analyze Club", means.index, key='m_club')
                mech_data = master_df[master_df['club'] == mech_club]
                
                # 1. OPTIMIZER CHART (Launch vs Spin)
                st.markdown("#### 🚀 Driver/Iron Optimization (Launch vs Spin)")
                col_m1, col_m2 = st.columns([3, 1])
                
                with col_m1:
                    if 'Launch V (°)' in mech_data.columns and 'Spin (rpm)' in mech_data.columns:
                        fig_opt = px.scatter(mech_data, x='Spin (rpm)', y='Launch V (°)', 
                                           color='Session', size='Carry (yds)',
                                           title=f"Launch vs Spin: {mech_club}")
                        
                        # Add "Pro Zone" box (Approximate for Driver)
                        if 'driver' in str(mech_club).lower():
                            fig_opt.add_shape(type="rect",
                                x0=1800, y0=10, x1=2800, y1=16,
                                line=dict(color="Gold", width=2, dash="dot"),
                                fillcolor="Gold", opacity=0.1
                            )
                            fig_opt.add_annotation(x=2300, y=16.5, text="Optimal Driver Zone", showarrow=False, font=dict(color="Gold"))
                        
                        st.plotly_chart(style_fig(fig_opt), use_container_width=True)
                    else:
                        st.warning("Launch/Spin data missing.")
                
                with col_m2:
                    st.markdown("**Key Averages**")
                    if 'AOA (°)' in mech_data.columns:
                        avg_aoa = mech_data['AOA (°)'].mean()
                        st.metric("Angle of Attack", f"{avg_aoa:.1f}°", delta="Positive = Up" if avg_aoa > 0 else "Negative = Down")
                    
                    if 'Launch V (°)' in mech_data.columns:
                        st.metric("Avg Launch", f"{mech_data['Launch V (°)'].mean():.1f}°")
                    
                    if 'Spin (rpm)' in mech_data.columns:
                        st.metric("Avg Spin", f"{mech_data['Spin (rpm)'].mean():.0f} rpm")

                # 2. FACE TO PATH (The Curve)
                st.markdown("---")
                st.markdown("#### ↩️ Shot Shape (Face to Path)")
                
                if 'Club Path_Clean' in mech_data.columns and 'FTP_Clean' in mech_data.columns:
                    # Create a scatter of Path vs Face-to-Path
                    fig_path = px.scatter(mech_data, x='Club Path_Clean', y='FTP_Clean',
                                        color='Lateral_Clean', # Color by result (L/R)
                                        title=f"Swing Path vs Face-to-Path: {mech_club}",
                                        labels={'Club Path_Clean': 'Club Path (Neg=Left / Pos=Right)', 
                                                'FTP_Clean': 'Face to Path (Neg=Closed / Pos=Open)'}
                                        )
                    
                    # Add crosshairs
                    fig_path.add_hline(y=0, line_color="white", opacity=0.2) # Square Face
                    fig_path.add_vline(x=0, line_color="white", opacity=0.2) # Neutral Path
                    
                    # Annotations
                    fig_path.add_annotation(x=5, y=5, text="Push Slice zone", showarrow=False, font=dict(size=10, color="gray"))
                    fig_path.add_annotation(x=-5, y=-5, text="Pull Hook zone", showarrow=False, font=dict(size=10, color="gray"))
                    fig_path.add_annotation(x=-5, y=5, text="FADE ZONE", showarrow=False, font=dict(size=12, color="#00E5FF"))
                    fig_path.add_annotation(x=5, y=-5, text="DRAW ZONE", showarrow=False, font=dict(size=12, color="#FF4081"))
                    
                    st.plotly_chart(style_fig(fig_path), use_container_width=True)
                else:
                    st.info("Club Path or Face-to-Path data not available in these CSVs.")

else:
    st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        <h1>🏌️‍♂️ Ready to Practice?</h1>
        <p style="font-size: 18px; color: #888;">Drag and drop your FlightScope CSV files to unlock pro-level analytics.</p>
    </div>
    """, unsafe_allow_html=True)
