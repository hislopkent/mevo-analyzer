import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="My Mevo+ Dashboard", layout="wide")
st.title("⛳ Mevo+ Lifetime Analyzer")

# --- 1. DATA PROCESSING FUNCTIONS ---

def clean_mevo_data(df, filename):
    """
    Cleans raw FlightScope Mevo+ CSV data.
    - Removes summary rows.
    - Parses text directions (e.g., "10.5 L").
    - Cleans units.
    """
    # Filter out summary rows (keep only rows where Shot is a number)
    df_clean = df[df['Shot'].astype(str).str.isdigit()].copy()
    
    # Add Session ID
    df_clean['Session'] = filename.replace('.csv', '')
    
    # Helper: Parse "10 L" / "10 R" to floats
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
            # Create a generic numeric name for checking later
            clean_col_name = col.replace(' (yds)', '').replace(' (°)', '') + '_Clean'
            
            # Map Lateral specifically for charts
            if 'Lateral' in col:
                df_clean['Lateral_Clean'] = df_clean[col].apply(parse_lr)
            else:
                df_clean[clean_col_name] = df_clean[col].apply(parse_lr)

    # Clean standard numeric columns
    numeric_cols = ['Carry (yds)', 'Total (yds)', 'Ball (mph)', 'Club (mph)', 'Smash', 'Spin (rpm)', 'Height (ft)']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean

def filter_outliers(df):
    """
    Removes shots that are statistically improbable (Mishits/Misreads).
    """
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
        
        # Lower bound strict (remove duffs), Upper bound loose (keep bombs)
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
    st.header("1. Upload Data")
    uploaded_files = st.file_uploader("Upload CSV Files", accept_multiple_files=True, type='csv')
    
    st.header("2. Player Profile")
    # Handicap Input for Virtual Green Sizing
    handicap = st.number_input("Your Handicap", min_value=0, max_value=54, value=15, step=1, 
                               help="Used to calculate the size of the 'Virtual Green' target zone.")
    
    st.header("3. Settings")
    remove_bad_shots = st.checkbox("Remove Outliers", value=True, 
                                  help="Automatically removes duffs and misreads.")

# --- 3. MAIN APP LOGIC ---
if uploaded_files:
    # A. LOAD & CLEAN
    all_data = []
    for f in uploaded_files:
        try:
            raw = pd.read_csv(f)
            clean = clean_mevo_data(raw, f.name)
            all_data.append(clean)
        except Exception as e:
            st.error(f"Could not load {f.name}: {e}")

    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        original_count = len(master_df)
        
        # B. APPLY OUTLIER LOGIC
        if remove_bad_shots:
            master_df, dropped_count = filter_outliers(master_df)
            if dropped_count > 0:
                st.success(f"🧹 Cleaned Data: Removed {dropped_count} outliers from {original_count} total shots.")
        
        # C. CSV DOWNLOAD (New Feature)
        csv_data = master_df.to_csv(index=False).encode('utf-8')
        st.sidebar.markdown("---")
        st.sidebar.download_button(
            label="📥 Download Merged Database",
            data=csv_data,
            file_name="my_mevo_lifetime_data.csv",
            mime="text/csv"
        )

        # --- D. DASHBOARD TABS ---
        st.write("---")
        tab1, tab2, tab3 = st.tabs(["🔍 Single Club & Accuracy", "🎒 Bag Gapping", "📈 Trends & Progress"])

        # ==========================================
        # TAB 1: SINGLE CLUB & ACCURACY
        # ==========================================
        with tab1:
            # Sort clubs by distance (Driver first)
            club_order = master_df.groupby('club')['Carry (yds)'].mean().sort_values(ascending=False).index
            selected_club = st.selectbox("Select Club", club_order)
            
            subset = master_df[master_df['club'] == selected_club]
            
            if len(subset) > 0:
                # --- CALCULATE VIRTUAL GREEN SIZE ---
                # Logic: Base 10 yds (Scratch) + (Handicap * 0.8)
                # Examples: Hcp 0 -> 10y radius | Hcp 15 -> 22y radius | Hcp 30 -> 34y radius
                target_radius = 10 + (handicap * 0.8)
                
                # --- TILES ---
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Avg Carry", f"{subset['Carry (yds)'].mean():.1f} yds")
                c2.metric("Max Carry", f"{subset['Carry (yds)'].max():.1f} yds")
                
                # Accuracy Score: % of shots within the handicap-adjusted target zone
                if 'Lateral_Clean' in subset.columns:
                    shots_on_target = subset[abs(subset['Lateral_Clean']) <= target_radius]
                    accuracy_pct = (len(shots_on_target) / len(subset)) * 100
                    c3.metric(f"Virtual Green Hit %", f"{accuracy_pct:.0f}%", 
                              help=f"Percentage of shots landing within {target_radius:.1f} yds of center (Based on Handicap {handicap})")
                else:
                    c3.metric("Ball Speed", f"{subset['Ball (mph)'].mean():.1f} mph")
                    
                c4.metric("Smash Factor", f"{subset['Smash'].mean():.2f}")

                # --- CHARTS ---
                col_chart1, col_chart2 = st.columns([2, 1]) 
                
                # 1. Dispersion Scatter Plot with Virtual Green
                with col_chart1:
                    st.subheader(f"🎯 Dispersion")
                    if 'Lateral_Clean' in subset.columns:
                        fig_disp = px.scatter(
                            subset, x='Lateral_Clean', y='Carry (yds)', 
                            color='Session', 
                            hover_data=['Ball (mph)', 'Shot Type'],
                            range_x=[-60, 60]
                        )
                        
                        # Add The "Virtual Green" Zone
                        avg_carry = subset['Carry (yds)'].mean()
                        fig_disp.add_shape(type="rect",
                            x0=-target_radius, y0=avg_carry - target_radius, # Square green logic
                            x1=target_radius, y1=avg_carry + target_radius,
                            line=dict(color="Green", width=1, dash="dot"),
                            fillcolor="Green", opacity=0.1
                        )
                        
                        # Add Center Line
                        fig_disp.add_vline(x=0, line_dash="solid", line_color="green", opacity=0.3)
                        
                        # Labels
                        fig_disp.update_xaxes(title="Left <--- Target ---> Right")
                        fig_disp.update_layout(showlegend=True)
                        st.plotly_chart(fig_disp, use_container_width=True)
                        
                        st.caption(f"Note: The green box represents your target zone ({target_radius:.1f} yds wide) based on a {handicap} handicap.")
                    else:
                        st.warning("No lateral data available.")

                # 2. Shot Shape Pie Chart
                with col_chart2:
                    st.subheader("Shot Shape")
                    if 'Shot Type' in subset.columns:
                        shape_counts = subset['Shot Type'].value_counts().reset_index()
                        shape_counts.columns = ['Shape', 'Count']
                        
                        fig_pie = px.pie(
                            shape_counts, 
                            values='Count', 
                            names='Shape',
                            hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig_pie.update_layout(legend=dict(orientation="h", y=-0.1))
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.info("No 'Shot Type' data found.")
            else:
                st.write("No data for this club.")

        # ==========================================
        # TAB 2: BAG GAPPING
        # ==========================================
        with tab2:
            st.subheader("📏 Yardage Gapping Matrix")
            # Sort by average carry
            club_means = master_df.groupby("club")["Carry (yds)"].mean().sort_values(ascending=False)
            
            fig_gap = px.box(
                master_df, 
                x='club', 
                y='Carry (yds)',
                color='club',
                category_orders={'club': club_means.index},
                title="Carry Distance Ranges by Club"
            )
            fig_gap.update_layout(showlegend=False)
            st.plotly_chart(fig_gap, use_container_width=True)
            
            with st.expander("View Data Table"):
                summary = master_df.groupby('club')[['Carry (yds)', 'Total (yds)', 'Ball (mph)', 'Smash']].mean().round(1)
                summary = summary.reindex(club_means.index)
                st.dataframe(summary)

        # ==========================================
        # TAB 3: TRENDS
        # ==========================================
        with tab3:
            st.subheader("📅 Progress Over Time")
            
            c_t1, c_t2 = st.columns(2)
            with c_t1:
                trend_club = st.selectbox("Club", club_order, key='trend_club_select')
            with c_t2:
                metric_to_track = st.selectbox("Metric", ['Ball (mph)', 'Carry (yds)', 'Club (mph)', 'Smash', 'Spin (rpm)'])
            
            # Trend Logic
            trend_subset = master_df[master_df['club'] == trend_club].copy()
            daily_avg = trend_subset.groupby('Session')[metric_to_track].mean().reset_index()
            daily_avg = daily_avg.sort_values('Session') 
            
            if len(daily_avg) > 1:
                fig_trend = px.line(
                    daily_avg, 
                    x='Session', 
                    y=metric_to_track, 
                    markers=True,
                    title=f"Avg {metric_to_track}: {trend_club}"
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Not enough sessions uploaded to show a trend line.")

else:
    # --- LANDING PAGE ---
    st.markdown("""
    ### 👋 Welcome to your Mevo+ Analyzer
    1. Export CSVs from FlightScope.
    2. **Rename** them to `YYYY-MM-DD_Desc.csv` (e.g., `2023-10-27_Driver.csv`).
    3. Drag & drop into the sidebar.
    """)
