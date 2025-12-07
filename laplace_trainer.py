import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -----------------------------------------------------------------------------
# AYARLAR
# -----------------------------------------------------------------------------
TICKER = "AAPL"          # Eğitimi bu hisse üzerinden yapacağız (Genel bir model için)
LOOKBACK = 60            # Geçmiş kaç güne bakarak tahmin yapacak?
EPOCHS = 5               # Eğitim tur sayısı (Test için 5 yeterli, kalite için 20+ yapılabilir)
BATCH_SIZE = 32
MODEL_NAME = "laplace_lstm_model.h5"
SCALER_NAME = "laplace_scaler.pkl"

def create_and_train_model():
    print(f"📡 {TICKER} için veriler indiriliyor...")
    # 1. Veri İndirme (Son 5 yıl)
    df = yf.download(TICKER, period="5y", interval="1d", progress=False)
    
    # Multi-index sütun temizliği (yfinance uyumluluğu)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    data = df['Close'].values.reshape(-1, 1)
    
    # 2. Veriyi Ölçeklendirme (0 ile 1 arasına sıkıştırma)
    print("⚖️ Veriler ölçeklendiriliyor...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 3. Eğitim Verisi Hazırlama (X: Geçmiş 60 gün, y: 61. gün)
    x_train, y_train = [], []
    for i in range(LOOKBACK, len(scaled_data)):
        x_train.append(scaled_data[i-LOOKBACK:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # 4. LSTM Modelini Kurma
    print("🧠 Model inşa ediliyor...")
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1)) # Tahmin edilen fiyat
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 5. Modeli Eğitme
    print(f"🔥 Eğitim başladı ({EPOCHS} Epoch)...")
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    
    # 6. Dosyaları Kaydetme
    print("💾 Dosyalar kaydediliyor...")
    model.save(MODEL_NAME)
    joblib.dump(scaler, SCALER_NAME)
    
    print(f"✅ BAŞARILI! '{MODEL_NAME}' ve '{SCALER_NAME}' oluşturuldu.")

if __name__ == "__main__":
    create_and_train_model()
