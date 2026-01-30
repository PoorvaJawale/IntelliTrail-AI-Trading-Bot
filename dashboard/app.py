import sys
import os

# Adds the parent directory (IntelliTrail) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from core.data_loader import load_stock_data
from core.indicators import apply_indicators
from core.engine import run_simulation_with_ai
import joblib

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