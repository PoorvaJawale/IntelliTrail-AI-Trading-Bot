import sys
import os
import joblib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. SYSTEM PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# --- 2. CORE IMPORTS ---
try:
    from core.data_loader import load_stock_data
    from core.indicators import apply_indicators
    from core.engine import run_simulation_with_ai
    from core.bot_orchestrator import BotOrchestrator
    from supabase import create_client, Client
except ModuleNotFoundError as e:
    st.error(f"CRITICAL: Could not find required modules: {e}")
    st.stop()

# --- 3. INITIALIZE SUPABASE CONNECTION ---
@st.cache_resource
def init_supabase():
    """Initialize Supabase client"""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        st.error("Supabase credentials not found in environment variables")
        st.stop()
    return create_client(url, key)

supabase = init_supabase()

# --- 4. INITIALIZE SESSION STATE ---
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'bot_orchestrator' not in st.session_state:
    st.session_state.bot_orchestrator = None

# --- 5. UI CONFIGURATION ---
st.set_page_config(page_title="IntelliTrail AI Terminal", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #161A25; border-radius: 10px; padding: 15px; border: 1px solid #30363D; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 IntelliTrail: AI Risk Management Engine")

# --- 6. GLOBAL DATA & MODEL LOADING ---
@st.cache_data
def load_global_assets():
    raw_data_path = "data/raw/"
    model_path = 'models/trained_models/nifty_model.pkl'

    try:
        all_files = os.listdir(raw_data_path)
        available_stocks = sorted(list(set([
            f.rsplit('_', 1)[0] for f in all_files if "_" in f
        ])))
        model = joblib.load(model_path)
        return available_stocks, model, raw_data_path
    except Exception as e:
        st.error(f"Error loading assets or model: {e}")
        st.stop()
        return [], None, ""

available_stocks, model, raw_data_path = load_global_assets()
st.session_state.model = model
st.session_state.data_loaded = True

# Initialize Bot Orchestrator
if st.session_state.bot_orchestrator is None:
    st.session_state.bot_orchestrator = BotOrchestrator(model, supabase)

# --- 7. DATABASE HELPER FUNCTIONS ---
@st.cache_data(ttl=5)
def fetch_active_orders():
    """Fetch all orders from database"""
    response = supabase.table('orders').select('*').order('created_at', desc=True).execute()
    return response.data

def fetch_order_details(order_id):
    """Fetch complete order details including predictions and manual levels"""
    order = supabase.table('orders').select('*').eq('id', order_id).single().execute()
    predictions = supabase.table('ai_predictions').select('*').eq('order_id', order_id).execute()
    manual_levels = supabase.table('manual_levels').select('*').eq('order_id', order_id).execute()
    execution_logs = supabase.table('bot_executions').select('*').eq('order_id', order_id).order('timestamp').execute()

    return {
        'order': order.data,
        'predictions': predictions.data,
        'manual_levels': manual_levels.data,
        'logs': execution_logs.data
    }

# --- 8. MAIN LAYOUT TABS ---
tab_stats, tab_analytics, tab_portfolio = st.tabs(["📊 Stats", "📈 Analytics", "💼 Portfolio"])

# --- TAB 1: STATS (Fleet Overview) ---
with tab_stats:
    st.header("Fleet Performance Summary")

    orders = fetch_active_orders()

    if orders:
        total_bots = len(orders)
        total_qty = sum(order['quantity'] for order in orders)
        total_pnl = sum(order.get('pnl', 0) or 0 for order in orders)
        executed_count = len([o for o in orders if o['status'] == 'executed'])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Bots", total_bots)
        c2.metric("Total Shares", total_qty)
        c3.metric("Total P&L", f"₹{total_pnl:,.2f}")
        c4.metric("Executed", f"{executed_count}/{total_bots}")

        st.markdown("---")

        summary_data = []
        for order in orders:
            summary_data.append({
                'Stock': order['ticker'],
                'Qty': order['quantity'],
                'Strategy': order['strategy_type'].upper(),
                'Entry Date': order.get('entry_date', 'N/A'),
                'AI Price': f"₹{order.get('ai_executed_price', 0):,.2f}" if order.get('ai_executed_price') else 'Pending',
                'Status': order['status'].upper(),
                'P&L': f"₹{order.get('pnl', 0):,.2f}" if order.get('pnl') else '₹0.00'
            })

        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Recent Execution Logs")

        for order in orders[:5]:
            with st.expander(f"{order['ticker']} - {order['strategy_type'].upper()} | {order['status'].upper()}"):
                details = fetch_order_details(order['id'])

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Order ID", order['id'][:8])
                    st.metric("Quantity", order['quantity'])
                with col2:
                    st.metric("Created", order['created_at'][:10])
                    st.metric("P&L", f"₹{order.get('pnl', 0):,.2f}")

                if details['logs']:
                    st.markdown("**Execution Steps:**")
                    for log in details['logs']:
                        status_emoji = "✅" if log['step_status'] == 'success' else "⏳" if log['step_status'] == 'in_progress' else "❌"
                        st.caption(f"{status_emoji} {log['execution_step']} - {log['step_status']}")
    else:
        st.info("No orders in the system. Deploy your first AI bot from the Portfolio tab!")


# --- TAB 2: ANALYTICS (Deep AI Insights) ---
# Replace the entire ANALYTICS TAB (tab_analytics) with this enhanced version:

with tab_analytics:
    st.header("🔬 AI Technical & Fundamental Analysis")
    
    col1, col2 = st.columns([2,1])
    with col1:
        selected_stock = st.selectbox("📈 Select Stock", available_stocks)
    with col2:
        # Auto-detect available timeframes
        stock_files = [f for f in os.listdir(raw_data_path) if f.startswith(selected_stock + '_')]
        available_tf = sorted(set([f.split('_')[-1].replace('.csv', '') for f in stock_files]))
        timeframe = st.selectbox("⏰ Timeframe", available_tf or ['daily'])
    
    target_file = f"{selected_stock}_{timeframe}.csv"
    full_path = os.path.join(raw_data_path, target_file)
    
    # Intelligent fallback
    if not os.path.exists(full_path):
        daily_path = os.path.join(raw_data_path, f"{selected_stock}_daily.csv")
        if os.path.exists(daily_path):
            st.info(f"📊 Using daily data (no {timeframe} available)")
            full_path = daily_path
            timeframe = 'daily'
        else:
            st.error(f"❌ No data for {selected_stock}. Add {selected_stock}_daily.csv to data/raw/")
            st.stop()
    
    df = load_stock_data(full_path)
    df = apply_indicators(df)
    recent_df = df.tail(120).copy()
    
    # [Rest of your chart + metrics code exactly as before...]

        
        # === 1. ADVANCED TECHNICAL CHART ===
    fig = go.Figure()
        
        # Candlestick
    fig.add_trace(go.Candlestick(
        x=recent_df.index, open=recent_df['open'], high=recent_df['high'], 
        low=recent_df['low'], close=recent_df['close'], name='Price'
    ))
        
        # Moving Averages
    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['sma_20'], 
                                name='SMA 20', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['sma_50'], 
                                name='SMA 50', line=dict(color='blue', width=2)))
        
        # Bollinger Bands (calculate)
    bb_period = 20
    recent_df['bb_middle'] = recent_df['close'].rolling(bb_period).mean()
    recent_df['bb_std'] = recent_df['close'].rolling(bb_period).std()
    recent_df['bb_upper'] = recent_df['bb_middle'] + (recent_df['bb_std'] * 2)
    recent_df['bb_lower'] = recent_df['bb_middle'] - (recent_df['bb_std'] * 2)
        
    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['bb_upper'], 
                                name='BB Upper', line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['bb_lower'], 
                                name='BB Lower', line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['bb_middle'], 
                                name='BB Middle', line=dict(color='gray')))
        
    fig.update_layout(
            template="plotly_dark", title=f"{selected_stock} - Technical Analysis ({timeframe})",
            height=500, showlegend=True
        )
    st.plotly_chart(fig, use_container_width=True)
        
        # === 2. TECHNICAL SUMMARY ===
    st.subheader("📊 Technical Indicators")
    tech_cols = st.columns(4)
        
        # Trend
    current_price = recent_df['close'].iloc[-1]
    sma_trend = "🟢 Bullish" if current_price > recent_df['sma_20'].iloc[-1] > recent_df['sma_50'].iloc[-1] else "🔴 Bearish"
    tech_cols[0].metric("Trend", sma_trend)
        
        # Volatility (ATR)
    atr_pct = (recent_df['atr'].iloc[-1] / current_price) * 100
    tech_cols[1].metric("Volatility (ATR%)", f"{atr_pct:.2f}%")
        
        # RSI Overbought/Oversold
    rsi_status = "🟢 Oversold" if recent_df['rsi'].iloc[-1] < 30 else "🔴 Overbought" if recent_df['rsi'].iloc[-1] > 70 else "🟡 Neutral"
    tech_cols[2].metric("RSI Status", f"{recent_df['rsi'].iloc[-1]:.1f}", delta=rsi_status)
        
        # BB Position
    bb_position = (current_price - recent_df['bb_lower'].iloc[-1]) / (recent_df['bb_upper'].iloc[-1] - recent_df['bb_lower'].iloc[-1])
    bb_signal = "🟢 Buy Zone" if bb_position < 0.2 else "🔴 Sell Zone" if bb_position > 0.8 else "🟡 Middle"
    tech_cols[3].metric("BB Position", f"{bb_position:.1%}", delta=bb_signal)
        
        # === 3. PERFORMANCE METRICS ===
    st.subheader("📈 Performance Analysis")
    perf_cols = st.columns(3)
        
        # Returns
    total_return = ((current_price / recent_df['close'].iloc[0]) - 1) * 100
    perf_cols[0].metric("Total Return", f"{total_return:.2f}%")
        
        # Volatility (20-day)
    vol_20d = recent_df['close'].pct_change().rolling(20).std() * (252**0.5) * 100
    perf_cols[1].metric("20D Volatility", f"{vol_20d.iloc[-1]:.1f}%")
        
        # Sharpe Ratio (simplified)
    returns = recent_df['close'].pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * (252**0.5)
    perf_cols[2].metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        # === 4. AI PREDICTION ===
    if st.session_state.model is not None:
            st.subheader("🤖 AI Price Prediction")
            features = pd.DataFrame([[recent_df['sma_20'].iloc[-1], recent_df['sma_50'].iloc[-1], 
                                    recent_df['rsi'].iloc[-1], recent_df['atr'].iloc[-1]]], 
                                   columns=['sma_20', 'sma_50', 'rsi', 'atr'])
            predicted = st.session_state.model.predict(features)[0]
            
            pred_cols = st.columns(2)
            pred_cols[0].metric("🎯 AI Next Close", f"₹{predicted:,.2f}")
            pred_cols[1].metric("📊 Confidence", f"{abs(predicted/current_price-1)*100:.1f}% {'↑' if predicted>current_price else '↓'}")
        
        # === 5. FUNDAMENTAL INSIGHTS (from price data) ===
    st.subheader("💹 Fundamental Proxies")
    fund_cols = st.columns(3)
    
        # Price momentum
    momentum_10d = ((current_price / recent_df['close'].iloc[-10]) - 1) * 100
    fund_cols[0].metric("10D Momentum", f"{momentum_10d:.1f}%")
        
        # Volume trend (if available)
    if 'volume' in df.columns:
            vol_trend = ((df['volume'].tail(5).mean() / df['volume'].tail(20).mean()) - 1) * 100
            fund_cols[1].metric("Volume Trend", f"{vol_trend:+.1f}%")
    else:
            fund_cols[1].metric("Volume Trend", "N/A")
        
        # Support/Resistance
    support = recent_df['low'].tail(20).min()
    resistance = recent_df['high'].tail(20).max()
    fund_cols[2].metric("S/R Levels", f"S:₹{support:,.0f} | R:₹{resistance:,.0f}")
        



# --- TAB 3: PORTFOLIO (The Command Center) ---
# Replace the entire PORTFOLIO TAB (tab_portfolio) with this enhanced version:

with tab_portfolio:
    st.subheader("🤖 AI Trading Command Center")
    
    # === DEPLOY NEW BOT ===
    st.markdown("---")
    deploy_col1, deploy_col2, deploy_col3, deploy_col4 = st.columns(4)
    with deploy_col1:
        new_ticker = st.selectbox("Stock", available_stocks, key="deploy_ticker")
    with deploy_col2:
        strategy = st.radio("Strategy", ["Auto-Scout (Buy)", "Auto-Protect (Sell)"], horizontal=True, key="deploy_strategy")
    with deploy_col3:
        quantity = st.number_input("Qty", min_value=1, max_value=1000, value=10, key="deploy_qty")
    with deploy_col4:
        sim_days = st.slider("Sim Days", 5, 60, 30, key="deploy_days")
    
    if st.button("🚀 DEPLOY AI BOT", type="primary", use_container_width=True):
        with st.spinner(f"Deploying AI Bot for {new_ticker}..."):
            strategy_type = 'buy' if 'Buy' in strategy else 'sell'

            order_data = {
                'ticker': new_ticker,
                'strategy_type': strategy_type,
                'quantity': quantity,
                'sim_days': sim_days
            }

            result = st.session_state.bot_orchestrator.execute_bot(order_data)

            if result['success']:
                st.success(f"AI Bot successfully deployed for {new_ticker}!")

                st.markdown("---")
                st.subheader("Execution Summary")

                col1, col2, col3 = st.columns(3)
                col1.metric("Executed Price", f"₹{result['executed_price']:,.2f}")
                col2.metric("Confidence", f"{result['confidence']:.1%}")
                col3.metric("P&L", f"₹{result['pnl']:,.2f}")

                st.markdown("**Manual Comparison Levels:**")
                for level, price in result['manual_levels'].items():
                    st.caption(f"{level}: ₹{price:,.2f}")

                fetch_active_orders.clear()
                st.rerun()
            else:
                st.error(f"Failed to deploy bot: {result.get('error', 'Unknown error')}")
    
    # === ACTIVE POSITIONS ===
    st.markdown("---")
    st.header("📊 Active Positions & AI Analysis")

    active_orders = fetch_active_orders()

    if active_orders:
        for order in active_orders:
            details = fetch_order_details(order['id'])

            with st.expander(
                f"📈 {order['ticker']} | {order['strategy_type'].upper()} | Qty: {order['quantity']} | {order['status'].upper()}",
                expanded=(order['status'] == 'executed')
            ):

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("AI Execution Price", f"₹{order.get('ai_executed_price', 0):,.2f}" if order.get('ai_executed_price') else 'Pending')
                col2.metric("Entry Date", order.get('entry_date', 'N/A'))
                col3.metric("Status", order['status'].upper())
                col4.metric("P&L", f"₹{order.get('pnl', 0):,.2f}")

                if details['manual_levels']:
                    st.markdown("**AI vs Manual Execution Levels**")

                    manual_dict = {ml['level_name']: ml['price'] for ml in details['manual_levels']}

                    exec_data = []
                    ai_price = order.get('ai_executed_price', 0)

                    exec_data.append({
                        'Method': 'AI Dynamic',
                        'Price': f"₹{ai_price:,.2f}",
                        'Advantage': 'Baseline'
                    })

                    for level_name, price in manual_dict.items():
                        advantage = ai_price - price if order['strategy_type'] == 'buy' else price - ai_price
                        exec_data.append({
                            'Method': level_name,
                            'Price': f"₹{price:,.2f}",
                            'Advantage': f"+₹{advantage:+,.2f}"
                        })

                    exec_df = pd.DataFrame(exec_data)
                    st.table(exec_df)

                    prices = [ai_price] + list(manual_dict.values())
                    labels = ['AI'] + list(manual_dict.keys())

                    fig = go.Figure(data=[go.Bar(
                        x=labels,
                        y=prices,
                        marker_color=['#00FF88'] + ['#FF6666'] * len(manual_dict),
                        text=[f"₹{p:,.0f}" for p in prices],
                        textposition='auto'
                    )])
                    fig.update_layout(
                        title="AI vs Manual - Execution Prices",
                        height=400,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if details['predictions']:
                    st.markdown("**AI Prediction Data**")
                    pred = details['predictions'][0]
                    pcol1, pcol2 = st.columns(2)
                    pcol1.metric("Predicted Price", f"₹{pred['predicted_price']:,.2f}")
                    pcol2.metric("Confidence", f"{pred.get('confidence_score', 0):.1%}")

                if details['logs']:
                    st.markdown("**Execution Log**")
                    for log in details['logs']:
                        status_emoji = "✅" if log['step_status'] == 'success' else "⏳" if log['step_status'] == 'in_progress' else "❌"
                        st.caption(f"{status_emoji} {log['execution_step']} - {log['step_status']}")

                if order['status'] != 'cancelled':
                    if st.button(f"Cancel Order", key=f"cancel_{order['id']}"):
                        supabase.table('orders').update({'status': 'cancelled'}).eq('id', order['id']).execute()
                        fetch_active_orders.clear()
                        st.rerun()
    else:
        st.info("No active positions. Deploy your first AI bot above!")


# --- 9. SIDEBAR STATUS ---
with st.sidebar:
    st.header("📡 AI Engine Status")
    current_hour = datetime.now().hour
    if 9 <= current_hour <= 15:
        st.success("Market Open: LIVE Mode")
    else:
        st.success("Backtest Mode Active")

    st.divider()
    st.metric("Available Stocks", len(available_stocks))

    orders = fetch_active_orders()
    active_count = len([o for o in orders if o['status'] != 'cancelled'])
    st.metric("Active Bots", active_count)

    if st.button("Refresh All Data"):
        st.cache_data.clear()
        fetch_active_orders.clear()
        st.rerun()

