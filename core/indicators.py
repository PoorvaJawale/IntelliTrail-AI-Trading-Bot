import pandas_ta as ta

def apply_indicators(df):
    # Determine the 'warm-up' length based on data size
    # If we have very little data (like Yearly), we use a smaller window
    data_size = len(df)
    sma_period = 20 if data_size > 30 else 5
    atr_period = 14 if data_size > 20 else 3
    
    # 1. ATR - Volatility
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
    
    # 2. SMA - Trend
    df['sma_20'] = ta.sma(df['close'], length=sma_period)
    df['sma_50'] = ta.sma(df['close'], length=min(50, data_size-1))
    
    # 3. RSI
    df['rsi'] = ta.rsi(df['close'], length=min(14, data_size-1))
    
    # For Yearly data, we don't dropna() completely, we backfill 
    # so we can still see the chart even if indicators are missing
    if data_size < 20:
        df.fillna(method='bfill', inplace=True)
    else:
        df.dropna(inplace=True)
    
    return df