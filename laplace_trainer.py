import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from ta.momentum import RSIIndicator
import warnings
import os

# Ayarlar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)

TICKER = "AAPL"
LOOKBACK = 60
EPOCHS = 10  # Veri arttığı için tur sayısını biraz artırdık
BATCH_SIZE = 32
MODEL_NAME = "laplace_lstm_model.h5"
FEATURE_SCALER_NAME = "laplace_feature_scaler.pkl" # Tüm veriler için
PRICE_SCALER_NAME = "laplace_price_scaler.pkl"     # Sadece fiyat için

def add_technical_indicators(df):
    """Veriye RSI ve Hacim kontrollerini ekler"""
    # RSI Hesapla
    rsi_indicator = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi_indicator.rsi()
    
    # Hacim zaten var ama 0 olan yerleri temizleyelim
    df["Volume"] = df["Volume"].replace(0, np.nan)
    
    # NaN (Boş) verileri temizle (RSI ilk 14 gün boş gelir)
    df.dropna(inplace=True)
    return df

def create_and_train_model():
    print(f"📡 {TICKER} için GELİŞMİŞ veriler indiriliyor...")
    df = yf.download(TICKER, period="5y", interval="1d", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # İndikatörleri Ekle
    print("🧪 Teknik indikatörler (RSI, Hacim) hesaplanıyor...")
    df = add_technical_indicators(df)
    
    # Kullanacağımız Özellikler: Close, Volume, RSI
    dataset = df[['Close', 'Volume', 'RSI']].values
    
    # Ölçeklendirme (Scaling)
    print("⚖️ Veriler ölçeklendiriliyor...")
    
    # 1. Genel Scaler (Modelin Girdisi İçin: Close, Vol, RSI)
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = feature_scaler.fit_transform(dataset)
    
    # 2. Fiyat Scaler (Sadece Çıktıyı Geri Çevirmek İçin: Close)
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    price_scaler.fit(df[['Close']].values)
    
    # Eğitim Verisi Hazırlama
    x_train, y_train = [], []
    for i in range(LOOKBACK, len(scaled_data)):
        x_train.append(scaled_data[i-LOOKBACK:i]) # Tüm özellikler (3 adet)
        y_train.append(scaled_data[i, 0])          # Hedef sadece Close (0. indeks)
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    print(f"🧠 Model inşa ediliyor (Girdi Şekli: {x_train.shape})...")
    
    model = Sequential()
    # Input shape artık (60, 3) oldu -> (60 gün, 3 özellik)
    model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
    
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1)) # Tek çıktı: Fiyat
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print(f"🔥 Eğitim başladı ({EPOCHS} Epoch)...")
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    
    print("💾 Dosyalar kaydediliyor...")
    model.save(MODEL_NAME)
    joblib.dump(feature_scaler, FEATURE_SCALER_NAME)
    joblib.dump(price_scaler, PRICE_SCALER_NAME)
    
    print(f"✅ BAŞARILI! Model ve Yeni Scalerlar hazır.")

if __name__ == "__main__":
    create_and_train_model()
