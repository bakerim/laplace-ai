import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from ta.momentum import RSIIndicator
import os
import warnings
from sklearn.preprocessing import MinMaxScaler

# --- AYARLAR ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)
TICKER = "AAPL"
LOOKBACK = 60
MODEL_PATH = "laplace_lstm_model.h5"
FEATURE_SCALER_PATH = "laplace_feature_scaler.pkl"
PRICE_SCALER_PATH = "laplace_price_scaler.pkl"
INITIAL_CAPITAL = 10000.0  # Başlangıç Sermayesi (Sanal)
BUY_THRESHOLD = 0.005      # Fiyat artış beklentisi %0.5'ten fazlaysa AL

# --- YARDIMCI FONKSİYONLAR ---

def add_technical_indicators(df):
    """Veriye RSI ve Hacim kontrollerini ekler (Trainer'daki ile aynı)"""
    rsi_indicator = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi_indicator.rsi()
    df["Volume"] = df["Volume"].replace(0, np.nan)
    df.dropna(inplace=True)
    return df

def load_assets():
    """Modeli ve Scalerları yükler"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        f_scaler = joblib.load(FEATURE_SCALER_PATH)
        p_scaler = joblib.load(PRICE_SCALER_PATH)
        return model, f_scaler, p_scaler
    except Exception as e:
        print(f"HATA: Gerekli dosyalar yüklenemedi. Önce trainer.py'yi çalıştırın. Hata: {e}")
        return None, None, None

# --- BACKTEST FONKSİYONU ---

def run_backtest():
    model, f_scaler, p_scaler = load_assets()
    if not all([model, f_scaler, p_scaler]):
        return
    
    print(f"\n🚀 {TICKER} için Geçmiş Test (Backtest) Başlatılıyor...")
    df = yf.download(TICKER, period="2y", interval="1d", progress=False) # Son 2 yıllık veri
    if df.empty:
        print("HATA: Veri indirilemedi.")
        return
        
    df = add_technical_indicators(df)
    
    # Portfolio takip değişkenleri
    cash = INITIAL_CAPITAL
    shares = 0
    total_trades = 0
    profitable_trades = 0
    
    # Veri setini tahmin için hazırlama (Close, Volume, RSI)
    dataset = df[['
