import pandas as pd

# 1. THE STATIC VERSION (for comparison)
def run_simulation(df, start_price, risk_multiplier=2.0):
    current_sl = start_price - (df.iloc[0]['atr'] * risk_multiplier)
    trades = []
    for timestamp, row in df.iterrows():
        if row['close'] <= current_sl:
            trades.append({'exit_time': timestamp, 'exit_price': row['close']})
            break
        new_sl = row['close'] - (row['atr'] * risk_multiplier)
        if new_sl > current_sl:
            current_sl = new_sl
    return trades, current_sl

# 2. THE AI VERSION (the brain)
def run_simulation_with_ai(df, start_price, model):
    trades = []  # Initialize empty list
    sl_history = []
    current_sl = start_price - (df.iloc[0]['atr'] * 2.0)
    
    for timestamp, row in df.iterrows():
        # ... your AI logic for multiplier ...
        features = pd.DataFrame([[row['sma_20'], row['sma_50'], row['rsi'], row['atr']]], 
                                 columns=['sma_20', 'sma_50', 'rsi', 'atr'])
        predicted_price = model.predict(features)[0]
        multiplier = 1.2 if predicted_price < row['close'] else 2.0
        
        sl_history.append(current_sl)
        
        if row['close'] <= current_sl:
            trades.append({'exit_time': timestamp, 'exit_price': row['close']})
            break # Exit the loop when stop loss is hit
            
        new_sl = row['close'] - (row['atr'] * multiplier)
        if new_sl > current_sl:
            current_sl = new_sl
            
    return trades, sl_history