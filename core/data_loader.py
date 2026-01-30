import pandas as pd
import os

def load_stock_data(file_path):
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    
    # Standardize column names to lowercase immediately
    df.columns = [col.lower() for col in df.columns]
    
    # Find the date column even if it's named 'timestamp' or 'datetime'
    date_col = None
    for col in ['date', 'timestamp', 'datetime', 'time']:
        if col in df.columns:
            date_col = col
            break
            
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    else:
        # Fallback if no date column is found
        print(f"⚠️ Warning: No date column found in {file_path}")
        
    return df