import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os
import re

# --- CONFIGURATION ---
DOWNLOAD_DIR = "/tmp/fs_downloads"
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

# --- 1. DATA CLEANING UTILITY ---
def clean_flightscope_data(df):
    """Cleans FlightScope data for analysis."""
    if 'Shot' in df.columns:
        df = df[~df['Shot'].isin(['Avg', 'Dev', 'Average', 'Deviation'])]

    def parse_directional(val):
        if not isinstance(val, str): return val
        val = val.strip()
        if val.endswith('L'):
            try: return -float(re.sub(r'[^\d\.]', '', val))
            except: return 0.0
        elif val.endswith('R'):
            try: return float(re.sub(r'[^\d\.]', '', val))
            except: return 0.0
        else:
            try:
                clean_val = re.sub(r'[^\d\.-]', '', val)
                return float(clean_val) if clean_val else 0.0
            except: return val

    directional_cols = ['Swing H (°)', 'Lateral (yds)', 'Spin Axis (°)', 
                        'Club Path (°)', 'Launch H (°)', 'FTP (°)', 'FTT (°)']
    
    for col in directional_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_directional)
            
    return df

# --- 2. BROWSER SETUP ---
def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    chrome_options.add_argument("--window-size=1920,1080")
    
    prefs = {
        "download.default_directory": DOWNLOAD_DIR,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False
    }
    chrome_options.add_experimental_option("prefs", prefs)

    service = Service(executable_path="/usr/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

# --- 3. HELPER: LOGIN ROUTINE ---
def login_to_flightscope(driver, username, password):
    """Handles the login process common to both actions."""
    wait = WebDriverWait(driver, 30)
    driver.get("https://myflightscope.com/wp-login.php")
    
    try:
        user_field = wait.until(EC.element_to_be_clickable((By.NAME, "log")))
        pass_field = driver.find_element(By.NAME, "pwd")
        
        user_field.clear()
        user_field.send_keys(username)
        time.sleep(0.5)
        pass_field.clear()
        pass_field.send_keys(password)
        time.sleep(0.5)
        
        submit_btn = wait.until(EC.element_to_be_clickable((
            By.XPATH, "//button[contains(., 'Log In')]"
        )))
        submit_btn.click()
        
        # Wait for redirect
        time.sleep(5)
        if "wp-login" in driver.current_url:
            raise Exception("Login failed. Please check your email and password.")
            
    except Exception as e:
        raise Exception(f"Login Interaction Failed: {e}")

# --- 4. ACTION: FETCH SESSION LIST ---
def fetch_session_list(username, password):
    """Logs in and returns a list of available sessions."""
    driver = get_driver()
    wait = WebDriverWait(driver, 30)
    sessions = []
    
    try:
        login_to_flightscope(driver, username, password)
        
        # Navigate to list
        driver.get("https://myflightscope.com/sessions/#APP=FS_GOLF")
        
        # Wait for rows
        rows = wait.until(EC.presence_of_all_elements_located((
            By.CSS_SELECTOR, "#sessions-datatable table tbody tr"
        )))

        # Parse the top 15 sessions (to save time/space)
        for row in rows[:15]:
            try:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) > 4:
                    date_text = cols[1].text.replace("\n", " ")
                    name_text = cols[2].text
                    
                    link_el = row.find_element(By.TAG_NAME, "a")
                    url = link_el.get_attribute("href")
                    
                    sessions.append({
                        "display": f"{date_text} | {name_text}",
                        "url": url
                    })
            except:
                continue
                
        return sessions

    except Exception as e:
        st.error(f"Error fetching list: {e}")
        return []
    finally:
        driver.quit()

# --- 5. ACTION: DOWNLOAD SPECIFIC SESSION ---
def download_session(username, password, session_url):
    """Logs in, goes directly to the URL, and downloads the CSV."""
    driver = get_driver()
    wait = WebDriverWait(driver, 30)
    
    # Clear directory first
    for f in os.listdir(DOWNLOAD_DIR):
        try: os.remove(os.path.join(DOWNLOAD_DIR, f))
        except: pass

    try:
        login_to_flightscope(driver, username, password)
        
        # Go directly to the specific session
        driver.get(session_url)
        
        # Handle Pagination (Try to show "All")
        time.sleep(3)
        try:
            pagination_select = driver.find_element(By.CSS_SELECTOR, ".v-data-footer__select .v-select")
            pagination_select.click()
            time.sleep(1)
            all_option = driver.find_element(By.XPATH, "//div[contains(@class, 'v-list-item') and .//span[contains(text(), 'All')]]")
            all_option.click()
            time.sleep(3)
        except:
            pass # Continue if pagination fails

        # Click Export
        try:
            export_span = wait.until(EC.element_to_be_clickable((
                By.XPATH, "//span[contains(text(), 'Export Table to CSV')]"
            )))
            driver.execute_script("arguments[0].scrollIntoView();", export_span)
            time.sleep(2)
            export_span.click()
        except:
            raise Exception("Could not find Export button on this session page.")

        time.sleep(10)
        
        files = os.listdir(DOWNLOAD_DIR)
        if not files:
            raise Exception("No file downloaded.")
            
        return max([os.path.join(DOWNLOAD_DIR, f) for f in files], key=os.path.getctime)

    except Exception as e:
        st.error(f"Download Error: {e}")
        return None
    finally:
        driver.quit()

# --- 6. STREAMLIT UI ---
st.set_page_config(page_title="FlightScope Downloader", page_icon="⛳")
st.title("⛳ FlightScope Data Manager")

# Initialize Session State
if "sessions" not in st.session_state:
    st.session_state["sessions"] = []

# Sidebar for Credentials
with st.sidebar:
    st.header("Login Details")
    user = st.text_input("Email")
    pw = st.text_input("Password", type="password")
    
    if st.button("🔄 1. Fetch Session List"):
        if user and pw:
            with st.spinner("Logging in and fetching list..."):
                found_sessions = fetch_session_list(user, pw)
                if found_sessions:
                    st.session_state["sessions"] = found_sessions
                    st.success(f"Found {len(found_sessions)} sessions!")
                else:
                    st.error("No sessions found or login failed.")
        else:
            st.warning("Please enter email and password.")

# Main Area
if st.session_state["sessions"]:
    st.subheader("Select a Session")
    
    # create a dictionary for the selectbox
    options_map = {s["display"]: s["url"] for s in st.session_state["sessions"]}
    
    selected_display = st.selectbox(
        "Available Sessions (Most Recent First)", 
        options=list(options_map.keys())
    )
    
    selected_url = options_map[selected_display]
    
    st.write("---")
    
    if st.button(f"📥 2. Download: {selected_display}"):
        with st.spinner("Starting download process..."):
            csv_path = download_session(user, pw, selected_url)
            
            if csv_path:
                st.success("Download Complete!")
                
                # Processing
                try:
                    df_raw = pd.read_csv(csv_path)
                    df_clean = clean_flightscope_data(df_raw.copy())
                    
                    # Metrics
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Shots", len(df_clean))
                    if 'Carry (yds)' in df_clean.columns:
                        c2.metric("Avg Carry", f"{df_clean['Carry (yds)'].mean():.1f}")
                    if 'Ball (mph)' in df_clean.columns:
                        c3.metric("Avg Ball Speed", f"{df_clean['Ball (mph)'].mean():.1f}")
                    
                    st.dataframe(df_clean.head())
                    
                    # Download Buttons
                    csv_data = df_clean.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Save Cleaned CSV",
                        data=csv_data,
                        file_name="flightscope_clean.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error processing CSV: {e}")
else:
    st.info("👈 Enter your login details on the left and click **Fetch Session List** to start.")
