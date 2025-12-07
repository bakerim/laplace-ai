import yfinance as yf
import pandas as pd
import numpy as np
import os
import streamlit as st 

# TensorFlow ve Scaler kütüphaneleri kontrolü
try:
    import tensorflow as tf
    import joblib
    LIBRARIES_LOADED = True
except ImportError:
    LIBRARIES_LOADED = False
    
# Dosya Yolları
MODEL_PATH = "laplace_lstm_model.h5"
SCALER_PATH = "laplace_scaler.pkl"

# -----------------------------------------------------------------------------
# MOTOR FONKSİYONLARI
# -----------------------------------------------------------------------------

@st.cache_resource
def load_ai_assets():
    """Modeli ve Scaler'ı yükler."""
    model = None
    scaler = None
    
    if not LIBRARIES_LOADED:
        return None, None 
        
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"Hata: Model yüklenirken sorun oluştu: {e}")
            
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            print(f"Hata: Scaler yüklenirken sorun oluştu: {e}")
            
    return model, scaler

def get_data(ticker, period="2y", interval="1d"):
    """Yahoo Finance verisi çeker."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        # Multi-index sütun temizliği
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None

def get_lstm_prediction(df, model, scaler, lookback=60):
    """LSTM tahmini yapar."""
    if model is None or scaler is None:
        return None, "Model veya Scaler dosyaları eksik."

    if not LIBRARIES_LOADED:
        return None, "Gerekli kütüphaneler eksik."

    try:
        raw_data = df['Close'].values[-lookback:].reshape(-1, 1)
        
        if len(raw_data) < lookback:
             return None, f"Yetersiz veri. En az {lookback} gün gerekli."
        
        scaled_data = scaler.transform(raw_data)
        X_input = np.reshape(scaled_data, (1, lookback, 1))
        prediction_scaled = model.predict(X_input, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled)
        
        return prediction[0][0], None
        
    except Exception as e:
        return None, str(e)
