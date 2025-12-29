import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="Mevo+ Pro Analytics", layout="wide", page_icon="⛳")

# Custom CSS for "Pro" Dark Mode
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
    # Filter out summary rows
    df_clean = df[df['Shot'].astype(str).str.isdigit()].copy()
    
    # Session ID
    df_clean['Session'] = filename.replace('.csv', '')
    
    # Helper: Parse "10 L" / "10 R"
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

    # Clean directional columns
    dir_cols = ['Lateral (yds)', 'Swing H (°)', 'Launch H (°)', 'Spin Axis (°)']
    for col in dir_cols:
        if col in df_clean.columns:
            clean_col_name = col.replace(' (yds)', '').replace(' (°)', '') + '_Clean'
            if 'Lateral' in col:
                df_clean['Lateral_Clean'] = df_clean[col].apply(parse_lr)
            else:
                df_clean[clean_col_name] = df_clean[col].apply(parse_lr)

    # Clean numeric columns
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
        
        # Strict low (duffs), loose high (bombs)
        lower_bound = q1 - (1.5 * iqr) 
        upper_bound = q3 + (3.0 * iqr) 
        
        valid_shots = club_data[
            (club_data['Carry (yds)'] >= lower_bound) & 
            (club_data['Carry (yds)'] <= upper_bound)
        ]
        
        outlier_count += (len(club_data) - len(valid_shots))
        df_filtered = pd.concat([df_filtered, valid_shots])
        
    return df_filtered, outlier_count

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("1. Data Uplink")
    uploaded_files = st.file_uploader("Drop CSV Files Here", accept_multiple_files=True, type='csv')
    
    st.header("2. Environment")
    # New: Environment Filter
    env_mode = st.radio("Show Sessions:", ["All", "Outdoor Only", "Indoor Only"], index=0)
    
    st.header("3. Player Config")
    handicap = st.number_input("Handicap", min_value=0, max_value=54, value=15)
    
    # New: Smash Factor Cap
    smash_cap = st.slider("Max Smash Factor", 1.40, 1.65, 1.52, 0.01, 
                         help="Filters out shots above this value to remove radar glitches.")
    
    st.header("4. Filters")
    remove_bad_shots = st.checkbox("Smart Clean (Remove Outliers)", value=True)

# --- 3. MAIN APP LOGIC ---
if uploaded_files:
    all_data = []
    for f in uploaded_files:
        try:
            raw = pd.read_csv(f)
            clean = clean_mevo_data(raw, f.name)
            all_data.append(clean)
        except Exception as e:
            st.error(f"Error: {f.name}: {e}")

    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        
        # 1. Apply Environment Filter
        if env_mode == "Outdoor Only":
            if 'Mode' in master_df.columns:
                master_df = master_df[master_df['Mode'].str.contains("Outdoor", case=False, na=False)]
        elif env_mode == "Indoor Only":
            if 'Mode' in master_df.columns:
                master_df = master_df[master_df['Mode'].str.contains("Indoor", case=False, na=False)]
        
        # 2. Apply Smash Factor Cap
        original_len = len(master_df)
        master_df = master_df[master_df['Smash'] <= smash_cap]
        smash_drops = original_len - len(master_df)
        if smash_drops > 0:
            st.sidebar.warning(f"Removed {smash_drops} shots with Smash > {smash_cap}")
            
        # 3. Apply Outlier Filter
        if remove_bad_shots:
            master_df, dropped_count = filter_outliers(master_df)
            if dropped_count > 0:
                st.toast(f"🧹 Removed {dropped_count} outliers", icon="🗑️")
        
        # CSV Download
        csv_data = master_df.to_csv(index=False).encode('utf-8')
        st.sidebar.markdown("---")
        st.sidebar.download_button("📥 Export Database", csv_data, "mevo_master.csv", "text/csv")

        # --- TABS ---
        st.write("---")
        tab1, tab2, tab3 = st.tabs(["🎯 Shot Analyzer", "📏 Gapping Matrix", "📈 Trend Lines"])

        # CHART THEME
        def style_fig(fig):
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial", size=12, color="#FAFAFA"),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            return fig

        # ================= TAB 1: ANALYZER =================
        with tab1:
            if not master_df.empty:
                club_order = master_df.groupby('club')['Carry (yds)'].mean().sort_values(ascending=False).index
                selected_club = st.selectbox("Select Club", club_order)
                subset = master_df[master_df['club'] == selected_club]
                
                if len(subset) > 0:
                    # Stats Row
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    target_radius = 10 + (handicap * 0.8)
                    
                    if 'Lateral_Clean' in subset.columns:
                        on_target = subset[abs(subset['Lateral_Clean']) <= target_radius]
                        acc_score = (len(on_target) / len(subset)) * 100
                        col1.metric("Accuracy Score", f"{acc_score:.0f}%", f"Target: ±{target_radius:.0f}y")
                    
                    col2.metric("Avg Carry", f"{subset['Carry (yds)'].mean():.1f}")
                    col3.metric("Max Carry", f"{subset['Carry (yds)'].max():.1f}")
                    col4.metric("Ball Speed", f"{subset['Ball (mph)'].mean():.1f}")
                    col5.metric("Smash", f"{subset['Smash'].mean():.2f}")

                    # Charts
                    c_chart1, c_chart2 = st.columns([3, 1])
                    
                    with c_chart1:
                        if 'Lateral_Clean' in subset.columns:
                            # Color logic: Fade (Right) is Cyan, Draw (Left) is Pink
                            # We create a temporary column for color coding
                            subset['Direction'] = np.where(subset['Lateral_Clean'] > 0, 'Fade/Push (R)', 'Draw/Pull (L)')
                            
                            fig_disp = px.scatter(
                                subset, x='Lateral_Clean', y='Carry (yds)', 
                                color='Direction', # Color by L/R
                                color_discrete_map={'Fade/Push (R)': '#00E5FF', 'Draw/Pull (L)': '#FF4081'},
                                hover_data=['Ball (mph)', 'Club (mph)'],
                                title=f"Shot Dispersion: {selected_club} (Cyan = Fade Bias)",
                            )
                            
                            # Virtual Green
                            avg_carry = subset['Carry (yds)'].mean()
                            fig_disp.add_shape(type="rect",
                                x0=-target_radius, y0=avg_carry - target_radius,
                                x1=target_radius, y1=avg_carry + target_radius,
                                line=dict(color="#00E676", width=2, dash="dot"),
                                fillcolor="#00E676", opacity=0.15,
                                layer="below"
                            )
                            
                            fig_disp.add_vline(x=0, line_width=1, line_color="#FAFAFA", opacity=0.2)
                            fig_disp.update_xaxes(title="Left <---> Right", range=[-60, 60], zeroline=False, gridcolor='rgba(255,255,255,0.1)')
                            fig_disp.update_yaxes(title="Carry (yds)", gridcolor='rgba(255,255,255,0.1)')
                            fig_disp.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='White')))
                            
                            st.plotly_chart(style_fig(fig_disp), use_container_width=True)

                    with c_chart2:
                        if 'Shot Type' in subset.columns:
                            type_counts = subset['Shot Type'].value_counts().reset_index()
                            type_counts.columns = ['Type', 'Count']
                            fig_pie = px.pie(type_counts, values='Count', names='Type', hole=0.5, 
                                            color_discrete_sequence=px.colors.qualitative.Pastel)
                            fig_pie.update_layout(showlegend=False, annotations=[dict(text='Shape', x=0.5, y=0.5, font_size=16, showarrow=False)])
                            st.plotly_chart(style_fig(fig_pie), use_container_width=True)
            else:
                st.warning("No data found for the selected environment filter.")

        # ================= TAB 2: GAPPING =================
        with tab2:
            if not master_df.empty:
                st.subheader("🎒 Bag Gapping Matrix")
                club_means = master_df.groupby("club")["Carry (yds)"].mean().sort_values(ascending=False)
                
                fig_gap = px.box(
                    master_df, x='club', y='Carry (yds)', color='club',
                    points="all", 
                    category_orders={'club': club_means.index},
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig_gap.update_layout(showlegend=False)
                fig_gap.update_traces(marker=dict(size=3, opacity=0.5), jitter=0.3)
                st.plotly_chart(style_fig(fig_gap), use_container_width=True)

        # ================= TAB 3: TRENDS =================
        with tab3:
            if not master_df.empty:
                st.subheader("📈 Performance Trends")
                c_t1, c_t2 = st.columns(2)
                with c_t1: trend_club = st.selectbox("Track Club", club_means.index, key='t_club')
                with c_t2: metric = st.selectbox("Track Metric", ['Ball (mph)', 'Carry (yds)', 'Club (mph)', 'Smash'])
                
                trend_data = master_df[master_df['club'] == trend_club].groupby('Session')[metric].mean().reset_index().sort_values('Session')
                
                if len(trend_data) > 1:
                    fig_trend = px.line(trend_data, x='Session', y=metric, markers=True, 
                                       title=f"{metric} Progress: {trend_club}")
                    fig_trend.update_traces(line_color='#00E676', line_width=4, marker=dict(size=10, color='White'))
                    st.plotly_chart(style_fig(fig_trend), use_container_width=True)
                else:
                    st.info("Upload multiple sessions with Date-based filenames to see trends.")

else:
    st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        <h1>🏌️‍♂️ Ready to Practice?</h1>
        <p style="font-size: 18px; color: #888;">Drag and drop your FlightScope CSV files to unlock pro-level analytics.</p>
    </div>
    """, unsafe_allow_html=True)
