import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import base64
import auth

# Load environment variables (e.g., API keys)
load_dotenv()

# Initialize DB
auth.init_db()

# Set page configuration
st.set_page_config(page_title="AI Data Analyst", page_icon="🤖", layout="wide")


# Session State Auth Tracking
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'role' not in st.session_state:
    st.session_state['role'] = None
if 'welcome_shown' not in st.session_state:
    st.session_state['welcome_shown'] = False
if 'df' not in st.session_state:
    st.session_state.df = None

@st.cache_data
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache_data
def load_csv(file):
    file.seek(0)
    return pd.read_csv(file)

@st.cache_data
def load_css_file(filepath):
    with open(filepath) as f:
        return f.read()

def set_ui_theme():
    # Inject Background Image
    bg_img_path = r'C:\Users\SURAJ KHILLARE\.gemini\antigravity\brain\38848be9-03cd-4a9a-b4de-16ebacf24bf5\ai_data_analyst_bg_1774992257620.png'
    if os.path.exists(bg_img_path):
        bin_str = get_base64(bg_img_path)
        st.markdown(f'''<style>.stApp {{ background-image: url("data:image/png;base64,{bin_str}"); background-size: cover; background-attachment: fixed; }}</style>''', unsafe_allow_html=True)
    
    # Inject Custom CSS
    if os.path.exists('style.css'):
        css_content = load_css_file('style.css')
        st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)

set_ui_theme()

def login_page():
    # Hide sidebar when on login page
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; color: white;'>Access Platform</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["Login", "Create Account"])
        
        with tab_login:
            with st.form("login_form"):
                user = st.text_input("Username").strip()
                pwd = st.text_input("Password", type="password").strip()
                submitted = st.form_submit_button("Login", use_container_width=True)
                if submitted:
                    success, role = auth.login_user(user, pwd)
                    if success:
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = user
                        st.session_state['role'] = role
                        st.rerun()
                    else:
                        st.error("Invalid Username or Password.")
                        
        with tab_signup:
            with st.form("signup_form"):
                st.markdown("#### Register New Account")
                new_user = st.text_input("Choose Username").strip()
                new_pwd = st.text_input("Choose Password", type="password").strip()
                confirm_pwd = st.text_input("Confirm Password", type="password").strip()
                submitted_signup = st.form_submit_button("Create Account", use_container_width=True)
                
                if submitted_signup:
                    if new_user and new_pwd and confirm_pwd:
                        if new_pwd != confirm_pwd:
                            st.error("Passwords do not match!")
                        else:
                            # Hardcoded "User" role for open signups so they can't make themselves Admins
                            success, msg = auth.add_user(new_user, new_pwd, role="User")

                            if success:
                                st.success(f"Account '{new_user}' created successfully! Please switch to the Login tab.")
                            else:
                                st.error(msg)
                    else:
                        st.warning("Please fill in all details.")

def welcome_popup():
    # Hide sidebar while viewing welcome popup
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        .welcome-box {
            background: rgba(30,30,40, 0.95);
            padding: 3rem;
            border-radius: 15px;
            text-align: center;
            border: 1px solid #00D2FF;
            box-shadow: 0px 0px 20px rgba(0, 210, 255, 0.5);
            margin-top: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="welcome-box">
            <h1 style='color: white;'>Welcome to AI Prediction System</h1>
            <p style='color: #cccccc;'>Get ready to explore, analyze, and build intelligent models seamlessly!</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("🚀 Get Started", use_container_width=True, type="primary"):
            st.session_state['welcome_shown'] = True
            st.rerun()

def admin_panel():
    st.header("Admin Dashboard")
    st.write(f"Logged in as: **{st.session_state['username']}**")
    
    tab1, tab2, tab3 = st.tabs(["Manage Users", "Login History", "Add New User"])
    
    with tab1:
        st.subheader("All Users")
        users_df = auth.get_all_users()
        st.dataframe(users_df, use_container_width=True)
        
        st.markdown("#### Delete User")
        user_to_delete = st.selectbox("Select User to Delete", options=[u for u in users_df['username'].tolist() if u != 'admin'])
        if st.button("Delete User", type="primary"):
            success, msg = auth.delete_user(user_to_delete)
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
                
    with tab2:
        st.subheader("Login History")
        logs_df = auth.get_login_logs()
        st.dataframe(logs_df, use_container_width=True)
        csv = logs_df.to_csv(index=False).encode('utf-8')
        st.download_button("Export Login Logs (CSV)", csv, "login_logs.csv", "text/csv")
        
    with tab3:
        st.subheader("Create New User")
        with st.form("new_user_form"):
            new_user = st.text_input("New Username")
            new_pwd = st.text_input("New Password", type="password")
            new_role = st.selectbox("Role", ["User", "Admin"])
            created = st.form_submit_button("Create User")
            if created:
                if new_user and new_pwd:
                    s, m = auth.add_user(new_user, new_pwd, new_role)
                    if s: st.success(m)
                    else: st.error(m)
                else:
                    st.warning("Please fill all fields.")


def main_ml_app(mode):
    # Main content area based on navigation
    if st.session_state.df is None:
        st.info("Please upload a CSV file from the sidebar to begin.")
    else:
        df = st.session_state.df
        if 'raw_df' not in st.session_state and df is not None:
            st.session_state.raw_df = df.copy()
        
        if mode == "Data Preview":
            st.header("Data Preview")
            st.write("Shape of the dataset:", df.shape)
            st.dataframe(df.head(100)) # Display first 100 rows
            
            st.subheader("Basic Information")
            st.write(df.describe())
            
        elif mode == "Data Preparation":
            st.header("Data Preparation & Cleaning")
            from components import data_prep_interface
            st.session_state.df = data_prep_interface(df)
            
        elif mode == "Auto EDA":
            st.header("Automated Exploratory Data Analysis")
            from components import generate_eda_report
            generate_eda_report(df)
            
        elif mode == "Auto ML Model":
            st.header("Automated Machine Learning")
            from components import run_auto_ml
            run_auto_ml(st.session_state.df)
            
        elif mode == "Anomaly Detection":
            st.header("Automated Anomaly Detection")
            from components import run_anomaly_detection
            run_anomaly_detection(st.session_state.df)
            
        elif mode == "Prediction Pipeline":
            st.header("Automated Prediction Pipeline")
            from components import prediction_pipeline_interface
            prediction_pipeline_interface(st.session_state.df)
            
        elif mode == "Chat with Data":
            st.header("Chat with Data using AI")
            from components import chat_with_data_interface
            chat_with_data_interface(df)


# Main App Routing
if not st.session_state['logged_in']:
    login_page()
elif not st.session_state['welcome_shown']:
    welcome_popup()
else:
    # Sidebar for navigation and options
    with st.sidebar:
        # Title with Animation (Custom HTML) inside sidebar
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="font-size: 1.8rem; margin-bottom: 0;">🤖 AI Data Analyst</h2>
            <p style="color: #00D2FF; font-size: 0.9rem; font-weight: 300;">Insights Platform</p>
        </div>
        """, unsafe_allow_html=True)
    
        st.header(f"Welcome, {st.session_state['username']}!")
        if st.button("Logout", key="logout_btn"):
            st.session_state['logged_in'] = False
            st.session_state['username'] = None
            st.session_state['role'] = None
            st.session_state['welcome_shown'] = False
            st.rerun()
            
        st.markdown("---")
        
        # Admin Panel Only visible to Admin
        tabs_list = ["Home"]
        if st.session_state['role'] == 'Admin':
            tabs_list.append("Admin Dashboard")
            
        app_view = st.sidebar.radio("Go to:", tabs_list)
        
        mode = None
        if app_view == "Home":
            st.header("Upload Data")
            uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
            
            if uploaded_file is not None:
                file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                if st.session_state.get('uploaded_file_id') != file_id:
                    try:
                        df = load_csv(uploaded_file)
                        st.session_state.df = df
                        st.session_state.raw_df = df.copy() # Store original version for reset
                        st.session_state.uploaded_file_id = file_id
                        st.success("File uploaded successfully!")
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
                    
            st.markdown("---")
            st.header("Navigation")
            
            # Define steps with icons and numbers for a "workflow" feel
            workflow_steps = [
                "1. Data Preview 📋",
                "2. Data Preparation 🧹",
                "3. Auto EDA 📊",
                "4. Auto ML Model 🚀",
                "5. Anomaly Detection 🕵️",
                "6. Prediction Pipeline 🔮",
                "7. Chat with Data 💬"
            ]
            
            selected_step = st.sidebar.radio("Select Workflow Step:", workflow_steps, key="nav_radio")
            
            # Map selected step back to mode
            mode_map = {
                "1. Data Preview 📋": "Data Preview",
                "2. Data Preparation 🧹": "Data Preparation",
                "3. Auto EDA 📊": "Auto EDA",
                "4. Auto ML Model 🚀": "Auto ML Model",
                "5. Anomaly Detection 🕵️": "Anomaly Detection",
                "6. Prediction Pipeline 🔮": "Prediction Pipeline",
                "7. Chat with Data 💬": "Chat with Data"
            }
            mode = mode_map[selected_step]

    # Content Display
    if app_view == "Admin Dashboard":
        admin_panel()
    else:
        # We need the large title for Home
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="font-size: 3.5rem; margin-bottom: 0;">🤖 Multi-Agent AI Data Analyst</h1>
            <p style="color: #00D2FF; font-size: 1.2rem; font-weight: 300;">Advanced Machine Learning & Insights Platform</p>
        </div>
        """, unsafe_allow_html=True)
        if mode:
            main_ml_app(mode)
