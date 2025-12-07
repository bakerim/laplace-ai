import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# Gereksiz uyarıları filtreleyelim (Temiz ekran için)
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # TensorFlow bilgi mesajlarını gizle
warnings.simplefilter(action='ignore', category=FutureWarning)

# -----------------------------------------------------------------------------
# AYARLAR
# -----------------------------------------------------------------------------
TICKER = "AAPL"
LOOKBACK = 60
EPOCHS = 5
BATCH_SIZE = 32
MODEL_NAME = "laplace_lstm_model.h5"
SCALER_NAME = "laplace_scaler.pkl"

def create_and_train_model():
    print(f"📡 {TICKER} için veriler indiriliyor...")
    
    # Veri İndirme
    df = yf.download(TICKER, period="5y", interval="1d", progress=False)
    
    # Multi-index temizliği
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    data = df['Close'].values.reshape(-1, 1)
    
    print("⚖️ Veriler ölçeklendiriliyor...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Eğitim Verisi Hazırlama
    x_train, y_train = [], []
    for i in range(LOOKBACK, len(scaled_data)):
        x_train.append(scaled_data[i-LOOKBACK:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    print("🧠 Model inşa ediliyor (Modern Mimari)...")
    
    # --- MODERNİZASYON BURADA YAPILDI ---
    model = Sequential()
    
    # 1. Yeni 'Input' katmanı (Uyarıyı susturur)
    model.add(Input(shape=(x_train.shape[1], 1)))
    
    # 2. LSTM Katmanları
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    # -------------------------------------
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print(f"🔥 Eğitim başladı ({EPOCHS} Epoch)...")
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    
    print("💾 Dosyalar kaydediliyor...")
    model.save(MODEL_NAME) # .h5 formatında ısrarcıyız çünkü Engine bunu okuyor
    joblib.dump(scaler, SCALER_NAME)
    
    print(f"✅ BAŞARILI! '{MODEL_NAME}' ve '{SCALER_NAME}' güncellendi.")

if __name__ == "__main__":
    create_and_train_model()
