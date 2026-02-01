import sys
import os
import joblib  # <--- ADD THIS LINE HERE

# 1. Get the absolute path to the 'dashboard' folder
current_file_path = os.path.abspath(__file__)
dashboard_dir = os.path.dirname(current_file_path)

# 2. Get the 'IntelliTrail' root directory (the parent of 'dashboard')
project_root = os.path.dirname(dashboard_dir)

# 3. Add project root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now, Python will find the 'core' folder
try:
    import streamlit as st
    import pandas as pd
    from core.data_loader import load_stock_data
    from core.indicators import apply_indicators
    from core.manager_logic import process_portfolio
    from core.engine import run_simulation_with_ai
except ModuleNotFoundError as e:
    st.error(f"Still can't find the core folder. Python is looking here: {sys.path}")
    st.stop()

# Get the absolute path of the current script (app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (IntelliTrail)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path so 'core' can be found
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# --- UI CONFIGURATION ---
st.set_page_config(page_title="IntelliTrail AI Terminal", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #161A25; border-radius: 10px; padding: 15px; border: 1px solid #30363D; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 IntelliTrail: AI Risk Management Engine")

# --- DATA DISCOVERY (The "NIFTY" Filter Fix) ---
raw_data_path = "data/raw/"
all_files = os.listdir(raw_data_path)

# 1. Filter: Only take files that have an underscore AND a valid timeframe suffix
# 2. Extract: Pull the full name (like "NIFTY 50") by splitting from the last underscore
available_stocks = sorted(list(set([
    f.rsplit('_', 1)[0] for f in all_files 
    if "_" in f and any(t in f.lower() for t in ["daily", "minute", "hourly", "weekly", "monthly", "yearly"])
])))

# --- SIDEBAR ---
st.sidebar.header("🛠️ Simulation Controls")

# Ensure "NIFTY 50" is the default if available
default_index = available_stocks.index("NIFTY 50") if "NIFTY 50" in available_stocks else 0
selected_stock = st.sidebar.selectbox("Select Asset", available_stocks, index=default_index)

timeframe_selection = st.sidebar.selectbox("Select Timeframe", ["daily", "minute", "hourly", "weekly", "monthly", "yearly"])
days_to_sim = st.sidebar.slider("Simulation Window (Last X Days)", 10, 200, 60)

# --- LOADING DATA ---
# This reconstructs the EXACT filename: "NIFTY 50_daily.csv"
target_file = f"{selected_stock}_{timeframe_selection}.csv"
file_path = os.path.join(raw_data_path, target_file)

if not os.path.exists(file_path):
    st.error(f"⚠️ Data file not found: {target_file}")
    st.stop()

df = load_stock_data(file_path)
df = apply_indicators(df)
model = joblib.load('models/trained_models/nifty_model.pkl')

# --- RUN SIMULATION ---
# --- RUN SIMULATION ---
# Safety Check: Ensure we don't try to take more rows than exist
available_rows = len(df)
sim_window = min(days_to_sim, available_rows)

test_df = df.tail(sim_window).copy()

if test_df.empty:
    st.warning(f"⚠️ Not enough data in {selected_stock}_{timeframe_selection} to run simulation.")
    st.stop()

# Now iloc[0] is safe
entry_price = test_df.iloc[0]['close']
test_df = df.tail(days_to_sim).copy()
entry_price = test_df.iloc[0]['close']

# Note: Temporarily using run_simulation_with_ai. 
# If you haven't updated engine.py to return sl_history yet, sl_history will be empty.
# For now, let's assume it returns (trades, sl_history)
try:
    results = run_simulation_with_ai(test_df, entry_price, model)
    if isinstance(results, tuple):
        trades, sl_history = results
    else:
        trades, sl_history = results, []
except Exception as e:
    st.error(f"Engine Error: {e}")
    trades, sl_history = [], []

# Use the current price based on the simulation window
# This makes the confidence change when you move the slider!
current_row = test_df.iloc[-1] 
current_price = current_row['close']

features = pd.DataFrame([[current_row['sma_20'], current_row['sma_50'], current_row['rsi'], current_row['atr']]], 
                         columns=['sma_20', 'sma_50', 'rsi', 'atr'])
predicted_price = model.predict(features)[0]

# --- PORTFOLIO TAB ---
with tab_portfolio:
    st.subheader("🚀 Active Bot Fleet")
    
    # Let the user "Deploy" a new bot
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        new_ticker = st.selectbox("Deploy Bot for:", available_stocks, key="deploy_ticker")
    with col_b:
        strat = st.radio("Strategy:", ["SELL (Protect)", "BUY (Scout)"])
    with col_c:
        trigger = st.number_input("Trigger Price:", value=0.0)

    if st.button("Launch Bot"):
        st.success(f"Bot successfully deployed for {new_ticker}!")

if st.button("🔄 Refresh All Bots"):
    results = process_portfolio(st.session_state.active_bots, model)
    
    # Create a Summary DataFrame
    summary_data = []
    for ticker, res in results.items():
        summary_data.append({
            "Ticker": ticker,
            "Current Price": res['current_price'],
            "Status": "🔴 EXIT" if res['trades'] else "🟢 ACTIVE",
            "Latest SL": res['history'][-1] if res['history'] else "N/A"
        })
    
    st.table(pd.DataFrame(summary_data))

# --- MAIN LAYOUT TABS ---
# This line creates the two tabs and assigns them to variables
tab_analysis, tab_portfolio = st.tabs(["📊 Individual Analysis", "💼 Live Portfolio Bots"])

# Now you can use "with tab_analysis" for your charts
with tab_analysis:
    st.subheader(f"Deep Analysis: {selected_stock}")
    # ... (put your existing chart and metric code here) ...

# And then the portfolio logic you added last night works here
with tab_portfolio:
    st.header("Active Strategy Portfolio")
    # ... (your bot cards and portfolio table logic goes here) ...
    
    # Define a mock portfolio (This will later come from a database or CSV)
    portfolio_bots = [
        {"stock": "RELIANCE", "type": "SELL (TSLO)", "target": 2500, "status": "🛡️ Protecting"},
        {"stock": "TCS", "type": "BUY (TSLO)", "target": 3800, "status": "🎯 Scouting Entry"},
        {"stock": "INFY", "type": "SELL (TSLO)", "target": 1600, "status": "🔴 Exit Triggered"}
    ]

    # Create dynamic columns for each bot
    cols = st.columns(len(portfolio_bots))
    for i, bot in enumerate(portfolio_bots):
        with cols[i]:
            st.subheader(bot["stock"])
            st.markdown(f"**Strategy:** {bot['type']}")
            st.metric("Status", bot["status"])
            if st.button(f"View {bot['stock']} Logs", key=f"bot_{i}"):
                st.write(f"Detailed logs for {bot['stock']} bot...")

# Calculate dynamic confidence
prediction_diff = abs(predicted_price - current_price) / current_price
confidence = min(99.9, 70 + (prediction_diff * 5000)) # Increased sensitivity

st.sidebar.subheader("🧠 AI Insights")
st.sidebar.progress(int(confidence) / 100)
st.sidebar.write(f"Directional Confidence: {confidence:.1f}%")

# --- METRICS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Entry Price", f"₹{entry_price:,.2f}")
with col2:
    status = "🔴 EXIT HIT" if trades else "🟢 MONITORING"
    st.metric("Bot Status", status)
with col3:
    if trades:
        profit = trades[0]['exit_price'] - entry_price
        st.metric("Net P/L", f"₹{profit:,.2f}", delta=f"{profit:.2f}")
    else:
        current_pl = current_price - entry_price
        st.metric("Unrealized P/L", f"₹{current_pl:,.2f}", delta=f"{current_pl:.2f}")

# --- MAIN CHART ---
fig = go.Figure()
fig.add_trace(go.Candlestick(x=test_df.index, open=test_df['open'], high=test_df['high'], 
                low=test_df['low'], close=test_df['close'], name=selected_stock))

# SMA
fig.add_trace(go.Scatter(x=test_df.index, y=test_df['sma_20'], line=dict(color='orange', width=1), name="SMA 20"))

# TSLO Line (if sl_history exists)
if sl_history:
    fig.add_trace(go.Scatter(x=test_df.index[:len(sl_history)], y=sl_history, 
                  line=dict(color='#BF5AF2', width=2, dash='dot'), name="AI Dynamic TSLO"))

# Exit Marker
if trades:
    fig.add_trace(go.Scatter(x=[trades[0]['exit_time']], y=[trades[0]['exit_price']],
                  mode='markers', marker=dict(color='red', size=12, symbol='x'), name="Exit Point"))

fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# --- EQUITY CURVE ---
st.subheader("📈 Capital Preservation Curve")
test_df['Static_Equity'] = (test_df['close'] / entry_price)
equity_fig = go.Figure()
equity_fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Static_Equity'], name="Market Performance", line=dict(color='#00CF80')))
equity_fig.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(equity_fig, use_container_width=True)

st.success("Analysis Complete.")

# Temporary debug line at the bottom of app.py
st.sidebar.write("Debug - Active Bots:", st.session_state.active_bots)