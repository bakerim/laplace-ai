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

# --- YENİ EKLENTİ: Drive Modülünü Çağır ---
from laplace_drive_loader import load_data_from_drive

# Ayarlar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)

# TICKER artık dosya ismine bağlı olacak, ama kaydederken yine genel isim kullanıyoruz
LOOKBACK = 60
EPOCHS = 15  # Büyük veri için epoch sayısını biraz daha artırdım!
BATCH_SIZE = 32
MODEL_NAME = "laplace_lstm_model.h5"
FEATURE_SCALER_NAME = "laplace_feature_scaler.pkl"
PRICE_SCALER_NAME = "laplace_price_scaler.pkl"

def add_technical_indicators(df):
    """Veriye RSI ve Hacim kontrollerini ekler"""
    # Veri setinde 'Close' ve 'Volume' olduğundan emin olalım
    if 'Close' not in df.columns or 'Volume' not in df.columns:
        print("⚠️ HATA: CSV dosyasında 'Close' veya 'Volume' sütunu bulunamadı.")
        return None

    rsi_indicator = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi_indicator.rsi()
    df["Volume"] = df["Volume"].replace(0, np.nan)
    df.dropna(inplace=True)
    return df

def create_and_train_model():
    print(f"📡 Veri Kaynağı: GOOGLE DRIVE")
    
    # 1. Veriyi Drive'dan Çek
    df = load_data_from_drive()
    
    if df is None:
        print("❌ Eğitim iptal edildi. Veri yok.")
        return

    # 2. İndikatörleri Ekle
    print("🧪 Teknik indikatörler hesaplanıyor...")
    df = add_technical_indicators(df)
    if df is None: return
    
    # Kullanacağımız Özellikler
    dataset = df[['Close', 'Volume', 'RSI']].values
    
    print("⚖️ Veriler ölçeklendiriliyor...")
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = feature_scaler.fit_transform(dataset)
    
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    price_scaler.fit(df[['Close']].values)
    
    # Eğitim Verisi Hazırlama
    x_train, y_train = [], []
    for i in range(LOOKBACK, len(scaled_data)):
        x_train.append(scaled_data[i-LOOKBACK:i])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    print(f"🧠 Model inşa ediliyor (Veri Boyutu: {x_train.shape[0]} satır)...")
    
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(units=100, return_sequences=True)) # Birim sayısını 50'den 100'e çıkardık (Daha büyük beyin)
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=50))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print(f"🔥 Eğitim başladı ({EPOCHS} Epoch)...")
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    
    print("💾 Dosyalar kaydediliyor...")
    model.save(MODEL_NAME)
    joblib.dump(feature_scaler, FEATURE_SCALER_NAME)
    joblib.dump(price_scaler, PRICE_SCALER_NAME)
    
    print(f"✅ BAŞARILI! Model Drive verisiyle eğitildi.")

if __name__ == "__main__":
    create_and_train_model()
