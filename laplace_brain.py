import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- LAPLACE BEYİN MOTORU V2.0 (TEMİZ VERİ) ---

DATA_FILE = 'laplace_FINAL_TRAINING_SET.csv'
DATA_PATH = os.path.join("laplace_dataset", DATA_FILE)

# Veri setini LSTM için uygun hale getirme (Zaman Serisi Dönüşümü)
def create_sequences(data, sequence_length):
    """Veriyi, LSTM'in anlayacağı N günlük serilere dönüştürür."""
    X, y = [], []
    # 'target' sütunu en sağda olduğu için sadece sayısal X özelliklerini alıyoruz.
    num_features = data.shape[1] - 1 

    for i in range(len(data) - sequence_length):
        # Geçmiş N günün sayısal verisi (X)
        X.append(data.iloc[i:(i + sequence_length), :num_features].values) 
        # N+1. günün hedefi (y)
        y.append(data.iloc[i + sequence_length]['target'])
        
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Derin Öğrenme Modelini Oluşturur."""
    model = Sequential()
    # 1. Katman: 50 nöronlu LSTM (Sequence öğrenme)
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2)) 
    
    # 2. Katman: 50 nöronlu LSTM
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    
    # Çıkış Katmanı: Binary tahmin (Yükselir/Düşer)
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("📐 LAPLACE BEYİN PROTOKOLÜ BAŞLATILDI.")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ HATA: Eğitim verisi bulunamadı: {DATA_FILE}")
        print("Lütfen önce laplace_fusion.py'yi çalıştırın.")
        return

    # Veri Yükleme ve Temizlik
    df = pd.read_csv(DATA_PATH)
    
    # --- FIX 1: Ticker Sütununu Hariç Tutma ---
    # Final CSV'de 'date' ve 'ticker' (veya 'Ticker') sütunları metin olduğu için bunları ölçekleme dışı bırakıyoruz.
    EXCLUDE_COLS = ['date', 'Date', 'ticker', 'Ticker'] 
    
    # Sadece sayısal ve hedef (target) sütunlarını seç
    features = [col for col in df.columns if col not in EXCLUDE_COLS and col != 'target']
    
    # Ölçekleme için sadece sayısal özellikleri ve hedefi al
    data = df[features + ['target']]
    
    # Veri Normalizasyonu (LSTM için Zorunlu)
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Yalnızca özellik sütunlarını ölçekle (target'ı hariç tut)
    scaled_features = scaler.fit_transform(data.drop(columns=['target'])) 
    
    # Ölçeklenmiş veriye hedef sütununu ekle
    scaled_df = pd.DataFrame(scaled_features, columns=data.drop(columns=['target']).columns)
    scaled_df['target'] = data['target'].values # Hedefi ölçeklemeden geri ekle

    # LSTM Dizileri Oluşturma
    SEQUENCE_LENGTH = 60 # Geçmiş 60 güne bakarak tahmin et
    X, y = create_sequences(scaled_df, SEQUENCE_LENGTH)
    
    # Veriyi eğitim ve test setlerine bölme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"✅ Veri Hazırlığı Tamamlandı.")
    print(f"   - Eğitim Örneği Sayısı: {len(X_train)}")
    print(f"   - Özellik Sayısı: {X_train.shape[2]} (Sadece Sayısal)")

    # Modeli Oluşturma
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    
    # Modeli Eğitme
    print("\n--- YAPAY SİNİR AĞI EĞİTİLİYOR (BU ZAMAN ALACAK) ---")
    
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    # Modeli Kaydetme
    MODEL_NAME = 'laplace_lstm_model.keras'
    model.save(os.path.join(DATA_DIR, MODEL_NAME))
    
    # Sonucu raporlama
    accuracy = history.history['val_accuracy'][-1]
    print("\n" + "="*50)
    print(f"🏁 EĞİTİM TAMAMLANDI!")
    print(f"   - Model Başarısı (Test): %{round(accuracy * 100, 2)}")
    print("="*50)


if __name__ == "__main__":
    main()
