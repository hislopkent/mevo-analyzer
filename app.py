import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="My Mevo+ Dashboard", layout="wide")
st.title("⛳ Mevo+ Lifetime Analyzer")

# --- 1. DATA PROCESSING FUNCTIONS ---
def clean_mevo_data(df, filename):
    # Filter out summary rows (Avg, Dev) - keep only numbered shots
    df_clean = df[df['Shot'].apply(lambda x: str(x).isdigit())].copy()
    
    # Add Session ID
    df_clean['Session'] = filename.replace('.csv', '')
    
    # Helper: Parse "10 L" / "10 R" to floats
    def parse_lr(val):
        if pd.isna(val): return 0.0
        s_val = str(val).strip()
        if 'L' in s_val: return -float(s_val.replace('L','').strip())
        if 'R' in s_val: return float(s_val.replace('R','').strip())
        try: return float(s_val)
        except: return 0.0

    # Clean numeric columns (handle "ft", strings, etc if needed)
    numeric_cols = ['Carry (yds)', 'Total (yds)', 'Ball (mph)', 'Club (mph)', 'Smash']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Clean directional columns
    if 'Lateral (yds)' in df_clean.columns:
        df_clean['Lateral_Clean'] = df_clean['Lateral (yds)'].apply(parse_lr)
    
    return df_clean

def filter_outliers(df):
    """
    Removes shots that are statistically improbable for each club.
    Logic: Uses Interquartile Range (IQR).
    - Low End (Mishits): Strict (1.5x IQR below Q1)
    - High End (Misreads): Loose (3.0x IQR above Q3) to keep 'bombs'.
    """
    df_filtered = pd.DataFrame()
    outlier_count = 0
    
    # Process each club separately
    for club in df['club'].unique():
        club_data = df[df['club'] == club].copy()
        
        # We need enough shots to calculate stats
        if len(club_data) < 5:
            df_filtered = pd.concat([df_filtered, club_data])
            continue
            
        # Metric to use for filtering (Carry is best for mishits)
        q1 = club_data['Carry (yds)'].quantile(0.25)
        q3 = club_data['Carry (yds)'].quantile(0.75)
        iqr = q3 - q1
        
        # Define bounds
        lower_bound = q1 - (1.5 * iqr) # Standard strictness for short shots
        upper_bound = q3 + (3.0 * iqr) # Very loose for long shots (keep the bombs!)
        
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
    
    st.header("2. Data Filters")
    remove_bad_shots = st.checkbox("Remove Outliers/Mishits", value=True, 
                                  help="Automatically removes shots that are way too short (duffs) or impossibly long (misreads) based on your history.")

# --- 3. MAIN APP ---
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
                st.info(f"🧹 Cleaned Data: Removed {dropped_count} outliers (duffs/misreads) from {original_count} total shots.")
        
        # --- DASHBOARD UI ---
        
        # Club Selector (sorted by typical distance)
        # Sort clubs by avg carry so they appear in order (Driver -> Wedges)
        sort_order = master_df.groupby('club')['Carry (yds)'].mean().sort_values(ascending=False).index
        selected_club = st.selectbox("Select Club to Analyze", sort_order)
        
        subset = master_df[master_df['club'] == selected_club]
        
        if len(subset) == 0:
            st.warning("No data for this club after filtering.")
        else:
            # 1. TOP STATS TILES
            st.markdown("### 📊 Performance Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Carry", f"{subset['Carry (yds)'].mean():.1f} yds")
            c2.metric("Max Carry", f"{subset['Carry (yds)'].max():.1f} yds")
            c3.metric("Ball Speed", f"{subset['Ball (mph)'].mean():.1f} mph")
            c4.metric("Smash Factor", f"{subset['Smash'].mean():.2f}")

            # 2. DISPERSION CHART (The "Pro" View)
            st.subheader("🎯 Shot Dispersion")
            
            # Create a nice scatter plot
            fig = px.scatter(
                subset, 
                x='Lateral_Clean', 
                y='Carry (yds)', 
                color='Session',
                hover_data=['Ball (mph)', 'Club (mph)', 'Spin (rpm)'],
                title=f"{selected_club}: {len(subset)} Shots",
                height=600
            )
            
            # Add target lines
            fig.add_vline(x=0, line_width=1, line_dash="solid", line_color="green", opacity=0.3)
            fig.add_hline(y=subset['Carry (yds)'].mean(), line_width=1, line_dash="dash", line_color="gray", annotation_text="Avg Carry")
            
            # Fix axes to look like a driving range (0 is center)
            max_lat = max(abs(subset['Lateral_Clean'].min()), abs(subset['Lateral_Clean'].max()), 20)
            fig.update_xaxes(range=[-max_lat*1.2, max_lat*1.2], title="Left (yds)  <--  Target  -->  Right (yds)")
            fig.update_yaxes(title="Carry Distance (yds)")
            
            st.plotly_chart(fig, use_container_width=True)

            # 3. DATA TABLE
            with st.expander("View Raw Data Table"):
                st.dataframe(subset)

else:
    st.markdown("""
    ### 👋 Welcome to your Mevo+ Analyzer
    To get started:
    1. Export your sessions from flightscope.com as CSVs.
    2. Drag and drop them into the sidebar on the left.
    3. Check "Remove Outliers" to automatically clean up your stats.
    """)
