import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
import os
from ta.momentum import RSIIndicator

# Drive yükleyicisini çağırıyoruz (Yahoo yerine)
from laplace_drive_loader import load_data_from_drive

# Ayarlar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
LOOKBACK = 60
MODEL_PATH = "laplace_lstm_model.h5"
FEATURE_SCALER_PATH = "laplace_feature_scaler.pkl"
PRICE_SCALER_PATH = "laplace_price_scaler.pkl"

def diagnose():
    print(f"🩺 BTC-USD Hastası Muayene Ediliyor (Veri Kaynağı: Drive)...")
    
    # 1. Dosyaları Yükle
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        f_scaler = joblib.load(FEATURE_SCALER_PATH)
        p_scaler = joblib.load(PRICE_SCALER_PATH)
    except Exception as e:
        print(f"❌ Dosya Hatası: {e}")
        return

    # 2. Veriyi Drive'dan Çek (Yahoo yerine)
    df = load_data_from_drive()
    
    if df is None:
        print("❌ Veri çekilemedi.")
        return

    # 3. İndikatörleri Hesapla (Eğitimdekiyle aynı mantık)
    rsi_indicator = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi_indicator.rsi()
    df["Volume"] = df["Volume"].replace(0, np.nan)
    df.dropna(inplace=True)
    
    dataset = df[['Close', 'Volume', 'RSI']].values
    
    print("\n🔍 SON 10 GÜNLÜK TAHMİN ANALİZİ (Model Donmuş mu?):")
    print(f"{'TARİH':<12} | {'GERÇEK':<10} | {'TAHMİN':<10} | {'FARK (%)':<10}")
    print("-" * 55)

    # Son 10 günü test et
    total_error = 0
    count = 0
    
    # Döngü aralığını veri setinin sonuna göre ayarla
    start_index = len(dataset) - 10
    end_index = len(dataset)

    for i in range(start_index, end_index):
        # Girdi verisi (önceki 60 gün)
        current_data = dataset[i-LOOKBACK:i]
        
        # Scale et
        scaled_data = f_scaler.transform(current_data)
        X_input = np.reshape(scaled_data, (1, LOOKBACK, 3))
        
        # Tahmin
        pred_scaled = model.predict(X_input, verbose=0)
        pred_price = p_scaler.inverse_transform(pred_scaled)[0][0]
        
        # Gerçek Değer (Kıyaslama için o günün kapanışı)
        real_price = dataset[i][0] 
        
        # Fark Hesabı
        diff = (pred_price - real_price) / real_price * 100
        total_error += abs(diff)
        count += 1
        
        date_str = df.index[i].strftime('%Y-%m-%d')
        print(f"{date_str:<12} | {real_price:<10.2f} | {pred_price:<10.2f} | {diff:+.4f}%")

    print("-" * 55)
    print(f"Ortalama Sapma: %{total_error/count:.4f}")

if __name__ == "__main__":
    diagnose()