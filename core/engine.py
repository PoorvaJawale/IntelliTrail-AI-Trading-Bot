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
def run_buy_simulation(df, target_price, buy_offset=0.02):
    """
    Trailing Buy Bot: Follows price down and buys on the bounce.
    """
    trades = []
    buy_history = []
    # Initial level: if price is 100, we wait for a bounce to 102
    trailing_buy_level = target_price * (1 + buy_offset)
    
    for timestamp, row in df.iterrows():
        buy_history.append(trailing_buy_level)
        
        # As price drops, we drag our buy-trigger level down
        new_buy_trigger = row['low'] * (1 + buy_offset)
        if new_buy_trigger < trailing_buy_level:
            trailing_buy_level = new_buy_trigger
            
        # If price reverses and crosses ABOVE our trigger, we BUY
        if row['close'] >= trailing_buy_level:
            trades.append({'entry_time': timestamp, 'entry_price': row['close']})
            break
            
    return trades, buy_history