import pandas as pd

# 1. THE STATIC VERSION (For Comparison)
def run_simulation(df, start_price, risk_multiplier=2.0):
    current_sl = start_price - (df.iloc[0]['atr'] * risk_multiplier)
    trades = []
    sl_history = []
    
    for timestamp, row in df.iterrows():
        sl_history.append(current_sl)
        
        if row['close'] <= current_sl:
            trades.append({'exit_time': timestamp, 'exit_price': row['close']})
            break
            
        new_sl = row['close'] - (row['atr'] * risk_multiplier)
        if new_sl > current_sl:
            current_sl = new_sl
            
    return trades, sl_history

# 2. THE AI SELL BOT (Protects Existing Shares)
def run_simulation_with_ai(df, start_price, model):
    """
    AI-Powered SELL Bot: Protects profits using a Dynamic TSLO.
    """
    trades = []
    sl_history = []
    current_sl = start_price - (df.iloc[0]['atr'] * 2.0)
    
    for timestamp, row in df.iterrows():
        # AI Logic: Predict price to decide the risk multiplier
        features = pd.DataFrame([[row['sma_20'], row['sma_50'], row['rsi'], row['atr']]], 
                                 columns=['sma_20', 'sma_50', 'rsi', 'atr'])
        predicted_price = model.predict(features)[0]
        
        # Bearish AI = tight stop (1.2x), Bullish AI = loose stop (2.0x)
        multiplier = 1.2 if predicted_price < row['close'] else 2.0
        
        sl_history.append(current_sl)
        
        if row['close'] <= current_sl:
            trades.append({'exit_time': timestamp, 'exit_price': row['close']})
            break 
            
        new_sl = row['close'] - (row['atr'] * multiplier)
        if new_sl > current_sl:
            current_sl = new_sl
            
    return trades, sl_history

# 3. THE BUY BOT (New logic for Buy Orders)
def run_buy_simulation(df, target_price, buy_offset=0.01):
    """
    Optimized Buy Bot: Targets the lower range of daily volatility.
    """
    trades = []
    # Calculate a 'Minimum' target based on the day's predicted low
    # using ATR as a proxy for the expected daily range
    for timestamp, row in df.iterrows():
        # Target the daily 'Low' or a price below the mean
        optimized_entry = row['low'] * (1 - buy_offset) 
        
        if row['low'] <= target_price:
            trades.append({'entry_time': timestamp, 'entry_price': row['low']})
            break
            
    return trades, []