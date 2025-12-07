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
from laplace_drive_loader import load_data_from_drive

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)

LOOKBACK = 60
EPOCHS = 20  # Return tahmini daha hassastır, epoch'u artırdık
BATCH_SIZE = 32
MODEL_NAME = "laplace_lstm_model.h5"
FEATURE_SCALER_NAME = "laplace_feature_scaler.pkl"
# Fiyat scaler'ına artık ihtiyacımız yok çünkü oran tahmin ediyoruz
# Ama geri dönüşüm için son fiyatı bilmemiz gerekecek.

def prepare_data(df):
    """Veriyi Log Return formatına çevirir"""
    
    # 1. Logaritmik Getiri Hesapla (Asıl Hedefimiz Bu!)
    # Log Return, fiyatın % değişimini normalize eder.
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. İndikatörler
    rsi_indicator = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi_indicator.rsi()
    df["Volume"] = df["Volume"].replace(0, np.nan)
    
    # İlk satırlar NaN olacağı için temizle
    df.dropna(inplace=True)
    
    return df

def create_and_train_model():
    print(f"📡 Veri Kaynağı: GOOGLE DRIVE (Mod: LOG RETURN)")
    
    df = load_data_from_drive()
    if df is None: return

    print("🧪 Veriler 'Logaritmik Getiri' formatına dönüştürülüyor...")
    df = prepare_data(df)
    
    # Girdiler: Log_Ret, Volume, RSI
    # Hedef (Target): Log_Ret (Bir sonraki günün değişimi)
    dataset = df[['Log_Ret', 'Volume', 'RSI']].values
    
    print("⚖️ Veriler ölçeklendiriliyor (-1 ile 1 arası)...")
    # Return'ler negatif olabileceği için -1, 1 aralığı daha iyidir
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = feature_scaler.fit_transform(dataset)
    
    x_train, y_train = [], []
    for i in range(LOOKBACK, len(scaled_data)):
        x_train.append(scaled_data[i-LOOKBACK:i])
        y_train.append(scaled_data[i, 0]) # Hedef: Log_Ret (0. indeks)
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    print(f"🧠 Model inşa ediliyor (Veri: {x_train.shape[0]} gün)...")
    
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
    
    # Daha derin ve karmaşık bir ağ (Return tahmini zordur)
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1)) # Çıktı: Tahmini Log Return
    
    # Huber Loss, ani fiyat hareketlerine (outliers) karşı daha dayanıklıdır
    model.compile(optimizer='adam', loss='huber')
    
    print(f"🔥 Eğitim başladı ({EPOCHS} Epoch)...")
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    
    print("💾 Dosyalar kaydediliyor...")
    model.save(MODEL_NAME)
    joblib.dump(feature_scaler, FEATURE_SCALER_NAME)
    
    print(f"✅ BAŞARILI! v3.1 (Log Return Modeli) Hazır.")

if __name__ == "__main__":
    create_and_train_model()