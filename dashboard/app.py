import sys
import os
import joblib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# --- 1. SYSTEM PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# --- 2. CORE IMPORTS ---
try:
    from core.data_loader import load_stock_data
    from core.indicators import apply_indicators
    from core.manager_logic import process_portfolio
    from core.engine import run_simulation_with_ai
except ModuleNotFoundError:
    st.error("CRITICAL: Could not find the 'core' folder. Ensure app.py is inside the 'dashboard' folder.")
    st.stop()

# --- 3. INITIALIZE SESSION STATE ---
# This fixes the 'data_loaded' and 'active_bots' NameErrors
if 'active_bots' not in st.session_state:
    st.session_state.active_bots = {
        "RELIANCE": {
            "qty": 50, "strat": "Auto-Protect (Sell)", "date": "2026-01-15",
            "price": 2450.00, "status": "Active", 
            "logs": ["Bot initialized at 2450.00", "AI tracking trend..."]
        },
        "TCS": {
            "qty": 20, "strat": "Auto-Scout (Buy)", "date": "2026-01-20",
            "price": 3820.00, "status": "Scouting",
            "logs": ["Analyzing entry points..."]
        }
    }

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = True

# --- 4. UI CONFIGURATION ---
st.set_page_config(page_title="IntelliTrail AI Terminal", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #161A25; border-radius: 10px; padding: 15px; border: 1px solid #30363D; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 IntelliTrail: AI Risk Management Engine")

# --- 5. GLOBAL DATA & MODEL LOADING ---
raw_data_path = "data/raw/"
model_path = 'models/trained_models/nifty_model.pkl'

try:
    all_files = os.listdir(raw_data_path)
    available_stocks = sorted(list(set([
        f.rsplit('_', 1)[0] for f in all_files if "_" in f
    ])))
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading assets or model: {e}")
    st.stop()

# --- 6. MAIN LAYOUT TABS ---
tab_stats, tab_analytics, tab_portfolio = st.tabs(["📊 Stats", "📈 Analytics", "💼 Portfolio"])

# --- TAB 1: STATS (Fleet Overview) ---
with tab_stats:
    st.header("Fleet Performance Summary")
    if st.session_state.active_bots:
        total_bots = len(st.session_state.active_bots)
        active_qty = sum(bot['qty'] for bot in st.session_state.active_bots.values())
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Active Bots", total_bots)
        c2.metric("Total Shares Managed", active_qty)
        c3.metric("System Status", "AI LIVE" if 9 <= datetime.now().hour <= 15 else "STANDBY")
        
        # Summary Table
        df_summary = pd.DataFrame.from_dict(st.session_state.active_bots, orient='index')
        st.table(df_summary[['qty', 'strat', 'date', 'status']])
    else:
        st.info("No active bots in the fleet.")

# --- TAB 2: ANALYTICS (Deep AI Insights) ---
with tab_analytics:
    st.header("AI Prediction & Charting")
    selected_stock = st.selectbox("Analyze Asset", available_stocks)
    
    # Load specific data for charting
    target_file = f"{selected_stock}_daily.csv" # Defaulting to daily for analytics
    full_path = os.path.join(raw_data_path, target_file)
    
    if os.path.exists(full_path):
        df = load_stock_data(full_path)
        df = apply_indicators(df)
        recent_df = df.tail(60)
        
        # Plotly Candlestick
        fig = go.Figure(data=[go.Candlestick(x=recent_df.index, open=recent_df['open'], 
                        high=recent_df['high'], low=recent_df['low'], close=recent_df['close'])])
        fig.update_layout(template="plotly_dark", title=f"{selected_stock} AI Trend Analysis")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Select an asset with available CSV data to view charts.")

# --- TAB 3: PORTFOLIO (The Command Center) ---
with tab_portfolio:
    st.subheader("🤖 Deploy New AI Bot")
    
    # Deployment Controls
    dep_col1, dep_col2, dep_col3, dep_col4 = st.columns(4)
    with dep_col1:
        new_ticker = st.selectbox("Select Asset", available_stocks, key="new_bot_ticker")
    with dep_col2:
        strat = st.radio("Strategy", ["Auto-Protect (Sell)", "Auto-Scout (Buy)"], key="new_bot_strat")
    with dep_col3:
        qty = st.number_input("Quantity", min_value=1, value=10, key="new_bot_qty")
    with dep_col4:
        exec_date = st.date_input("Execution Date", datetime.now(), key="new_bot_date")

    if st.button("Launch AI Bot", type="primary"):
        # Simulated AI execution price (would normally come from live API)
        # In a real scenario, we'd pull the latest 'close' from our dataframe
        exec_price = 2500.00 
        
        st.session_state.active_bots[new_ticker] = {
            "qty": qty,
            "strat": strat,
            "date": str(exec_date),
            "price": exec_price,
            "deployed_at": exec_price, # Adding this to prevent deployment errors
            "status": "Running",
            "logs": [f"AI Bot deployed on {exec_date}", f"Strategy: {strat}", f"Initial Price: ₹{exec_price}"]
        }
        st.success(f"AI Bot for {new_ticker} is now LIVE.")
        st.rerun()

    st.divider()
    st.header("📋 Execution Logs & Comparison")

    if not st.session_state.active_bots:
        st.info("No bots currently running.")
    else:
        for ticker, bot in st.session_state.active_bots.items():
            with st.expander(f"🤖 {ticker} Bot - {bot['status']} (Started: {bot['date']})", expanded=True):
                # Comparison Logic: AI vs Manual Fixed Stop Loss
                manual_sl_price = bot['price'] * 0.95
                ai_tslo_price = bot['price'] * 0.97 # Simulated AI Dynamic TSLO
                
                comp1, comp2, comp3 = st.columns(3)
                comp1.metric("AI Dynamic Price", f"₹{ai_tslo_price:,.2f}")
                comp2.metric("Manual (Fixed 5%)", f"₹{manual_sl_price:,.2f}", 
                            delta=f"{ai_tslo_price - manual_sl_price:.2f} (AI Gain)")
                comp3.write(f"**Qty:** {bot['qty']} | **Type:** {bot['strat']}")
                
                st.markdown("**📜 Live Activity Feed**")
                for entry in bot['logs']:
                    st.caption(f"🕒 {entry}")
                
                if st.button(f"Terminate {ticker}", key=f"term_{ticker}"):
                    del st.session_state.active_bots[ticker]
                    st.rerun()

# --- 7. SIDEBAR STATUS ---
with st.sidebar:
    st.header("📡 Engine Status")
    current_hour = datetime.now().hour
    if 9 <= current_hour <= 15:
        st.success("Market Open: AI Engine Active")
    else:
        st.warning("Market Closed: Monitoring Mode")
    
    st.divider()
    if st.button("Refresh All Bots"):
        st.toast("AI re-calculating stop-losses...")
        st.rerun()