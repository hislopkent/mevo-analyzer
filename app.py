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
    - Removes summary rows (Avg, Std Dev).
    - Parses text directions (e.g., "10.5 L" -> -10.5).
    - Cleans units (e.g., " ft", " mph").
    """
    # Filter out summary rows (keep only rows where Shot is a number)
    # We convert to string first to handle potential mixed types
    df_clean = df[df['Shot'].astype(str).str.isdigit()].copy()
    
    # Add Session ID based on filename
    # Tip: Rename files to YYYY-MM-DD for better sorting
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

    # Clean directional columns (Left is negative, Right is positive)
    dir_cols = ['Lateral (yds)', 'Swing H (°)', 'Launch H (°)', 'Spin Axis (°)']
    for col in dir_cols:
        if col in df_clean.columns:
            clean_col_name = col.replace(' (yds)', '').replace(' (°)', '') + '_Clean'
            # If it's Lateral, we map it specifically for the dispersion chart
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
    Removes shots that are statistically improbable for each club using IQR.
    - Low End (Mishits): Strict (1.5x IQR below Q1).
    - High End (Misreads): Loose (3.0x IQR above Q3) to keep 'bombs'.
    """
    df_filtered = pd.DataFrame()
    outlier_count = 0
    
    # Process each club separately
    for club in df['club'].unique():
        club_data = df[df['club'] == club].copy()
        
        # Need enough shots for stats
        if len(club_data) < 5:
            df_filtered = pd.concat([df_filtered, club_data])
            continue
            
        # Metric to use for filtering (Carry is best for mishits)
        q1 = club_data['Carry (yds)'].quantile(0.25)
        q3 = club_data['Carry (yds)'].quantile(0.75)
        iqr = q3 - q1
        
        # Define bounds
        lower_bound = q1 - (1.5 * iqr) # Strict on duffs
        upper_bound = q3 + (3.0 * iqr) # Loose on bombs
        
        # Filter
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
    
    st.header("2. Settings")
    remove_bad_shots = st.checkbox("Remove Outliers", value=True, 
                                  help="Automatically removes shots that are way too short (duffs) or impossibly long (misreads).")
    
    st.info("**Tip:** Rename your files to `YYYY-MM-DD_SessionName.csv` before uploading so the 'Trends' tab sorts correctly.")

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
        
        # --- C. DASHBOARD TABS ---
        st.write("---")
        tab1, tab2, tab3 = st.tabs(["🔍 Single Club Analysis", "🎒 Bag Gapping", "📈 Trends & Progress"])

        # ==========================================
        # TAB 1: SINGLE CLUB ANALYSIS
        # ==========================================
        with tab1:
            # Sort clubs by distance (Driver first)
            club_order = master_df.groupby('club')['Carry (yds)'].mean().sort_values(ascending=False).index
            selected_club = st.selectbox("Select Club", club_order)
            
            subset = master_df[master_df['club'] == selected_club]
            
            if len(subset) > 0:
                # Metric Tiles
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Avg Carry", f"{subset['Carry (yds)'].mean():.1f} yds")
                c2.metric("Max Carry", f"{subset['Carry (yds)'].max():.1f} yds")
                c3.metric("Ball Speed", f"{subset['Ball (mph)'].mean():.1f} mph")
                c4.metric("Smash Factor", f"{subset['Smash'].mean():.2f}")

                # Dispersion Plot
                st.subheader(f"🎯 Dispersion: {selected_club}")
                
                # Check if we have lateral data
                if 'Lateral_Clean' in subset.columns:
                    fig_disp = px.scatter(
                        subset, x='Lateral_Clean', y='Carry (yds)', 
                        color='Session', 
                        title=f"{selected_club} Dispersion (Left/Right)",
                        hover_data=['Ball (mph)', 'Spin (rpm)'],
                        range_x=[-50, 50] # Fixed range like a driving range
                    )
                    fig_disp.add_vline(x=0, line_dash="dash", line_color="green", opacity=0.5)
                    fig_disp.update_xaxes(title="Left (yds) <--- Target ---> Right (yds)")
                    st.plotly_chart(fig_disp, use_container_width=True)
                else:
                    st.warning("Lateral data not found in these files.")
            else:
                st.write("No data for this club.")

        # ==========================================
        # TAB 2: BAG GAPPING
        # ==========================================
        with tab2:
            st.subheader("📏 Yardage Gapping Matrix")
            st.write("This chart shows the range of distances for every club in your bag.")
            
            # Prepare data: Sort by average carry so the chart flows from Driver -> Wedge
            club_means = master_df.groupby("club")["Carry (yds)"].mean().sort_values(ascending=False)
            
            fig_gap = px.box(
                master_df, 
                x='club', 
                y='Carry (yds)',
                color='club',
                category_orders={'club': club_means.index},
                title="Carry Distance Ranges by Club (Median + Variability)"
            )
            fig_gap.update_layout(showlegend=False)
            st.plotly_chart(fig_gap, use_container_width=True)
            
            # Data Table
            with st.expander("View Gapping Table"):
                summary = master_df.groupby('club')[['Carry (yds)', 'Total (yds)', 'Ball (mph)', 'Smash']].mean().round(1)
                summary = summary.reindex(club_means.index)
                st.dataframe(summary)

        # ==========================================
        # TAB 3: TRENDS
        # ==========================================
        with tab3:
            st.subheader("📅 Progress Over Time")
            
            # Filters for this tab
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                trend_club = st.selectbox("Club", club_order, key='trend_club_select')
            with col_t2:
                metric_to_track = st.selectbox("Metric", ['Ball (mph)', 'Carry (yds)', 'Club (mph)', 'Smash', 'Spin (rpm)'])
            
            # Filter Data
            trend_subset = master_df[master_df['club'] == trend_club].copy()
            
            # Group by Session to get the average for that day
            # Note: This relies on Session names sorting alphabetically/chronologically
            daily_avg = trend_subset.groupby('Session')[metric_to_track].mean().reset_index()
            daily_avg = daily_avg.sort_values('Session') 
            
            if len(daily_avg) > 1:
                fig_trend = px.line(
                    daily_avg, 
                    x='Session', 
                    y=metric_to_track, 
                    markers=True,
                    title=f"Avg {metric_to_track} Trend: {trend_club}"
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Not enough sessions uploaded to show a trend line. Upload at least 2 files.")

else:
    # --- LANDING PAGE ---
    st.markdown("""
    ### 👋 Welcome to your Mevo+ Analyzer
    This tool allows you to merge multiple FlightScope sessions into one lifetime database.
    
    **How to use:**
    1. Export your CSV files from MyFlightScope.com.
    2. **Recommended:** Rename files to `YYYY-MM-DD_Desc.csv` (e.g., `2023-10-27_Driver.csv`).
    3. Drag and drop them into the **sidebar** on the left.
    """
