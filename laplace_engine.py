import yfinance as yf
import pandas as pd
import numpy as np
import os
import streamlit as st 
from ta.momentum import RSIIndicator

try:
    import tensorflow as tf
    import joblib
    LIBRARIES_LOADED = True
except ImportError:
    LIBRARIES_LOADED = False
    
# Dosya Yolları
MODEL_PATH = "laplace_lstm_model.h5"
FEATURE_SCALER_PATH = "laplace_feature_scaler.pkl"
PRICE_SCALER_PATH = "laplace_price_scaler.pkl"

@st.cache_resource
def load_ai_assets():
    """Modeli ve İKİ Scaler'ı yükler."""
    model = None
    f_scaler = None
    p_scaler = None
    
    if not LIBRARIES_LOADED:
        return None, None, None
        
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"Hata: Model yüklenemedi: {e}")
            
    if os.path.exists(FEATURE_SCALER_PATH):
        try:
            f_scaler = joblib.load(FEATURE_SCALER_PATH)
        except Exception as e:
            print(f"Hata: Feature Scaler yüklenemedi: {e}")

    if os.path.exists(PRICE_SCALER_PATH):
        try:
            p_scaler = joblib.load(PRICE_SCALER_PATH)
        except Exception as e:
            print(f"Hata: Price Scaler yüklenemedi: {e}")
            
    return model, f_scaler, p_scaler

def get_data(ticker, period="2y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # --- CANLI RSI HESAPLAMA ---
        rsi_ind = RSIIndicator(close=df["Close"], window=14)
        df["RSI"] = rsi_ind.rsi()
        
        # Hacim 0 temizliği ve NaN temizliği
        df["Volume"] = df["Volume"].replace(0, np.nan)
        df.dropna(inplace=True) # RSI yüzünden ilk 14 gün silinir
        
        return df
    except Exception:
        return None

def get_lstm_prediction(df, model, f_scaler, p_scaler, lookback=60):
    if not (model and f_scaler and p_scaler):
        return None, "Model veya Scaler dosyaları eksik/hatalı."

    try:
        # Son 'lookback' kadar veriyi al: Close, Volume, RSI
        raw_data = df[['Close', 'Volume', 'RSI']].values[-lookback:]
        
        if len(raw_data) < lookback:
             return None, f"Yetersiz veri. En az {lookback} gün gerekli."
        
        # 1. Girdileri Scale Et (3 Özellikli Scaler ile)
        scaled_data = f_scaler.transform(raw_data)
        
        # 2. Modelin istediği boyut (1, 60, 3)
        X_input = np.reshape(scaled_data, (1, lookback, 3))
        
        # 3. Tahmin Yap (0-1 arasında bir değer döner)
        prediction_scaled = model.predict(X_input, verbose=0)
        
        # 4. Çıktıyı Geri Çevir (Sadece Fiyat Scaler ile)
        prediction = p_scaler.inverse_transform(prediction_scaled)
        
        return prediction[0][0], None
        
    except Exception as e:
        return None, f"Tahmin hatası: {str(e)}"
