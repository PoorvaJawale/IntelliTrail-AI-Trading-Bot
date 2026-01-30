from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

def train_and_save_model(df):
    # Features: using SMA and RSI to predict 'Close'
    X = df[['sma_20', 'sma_50', 'rsi', 'atr']]
    y = df['close']
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the model to the models/ folder
    os.makedirs('models/trained_models', exist_ok=True)
    model_path = 'models/trained_models/nifty_model.pkl'
    joblib.dump(model, model_path)
    
    print(f"🧠 AI Model Trained and saved to {model_path}")
    return model