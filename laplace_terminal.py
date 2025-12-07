import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
import json
import time
import numpy as np
# --- YENİ EKLENENLER (LSTM İÇİN) ---
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model 
import os 
import warnings
warnings.filterwarnings('ignore', category=FutureWarning) 

# --- GEREKLİ SABİTLER ---
SEQUENCE_LENGTH = 60
DATA_DIR = 'laplace_dataset'
MODEL_PATH = os.path.join(DATA_DIR, 'laplace_lstm_model.keras')
TRAINING_DATA_PATH = os.path.join(DATA_DIR, 'laplace_FINAL_TRAINING_SET.csv')

# --- GLOBAL MODEL VE ÖLÇEKLEYİCİ YÜKLEME ---
@st.cache_resource
def load_laplace_resources():
    """Modelleri ve Global Ölçekleyiciyi yükler/eğitir."""
    try:
        model = load_model(MODEL_PATH)
        df_train = pd.read_csv(TRAINING_DATA_PATH)
        
        # Ticker, Date ve target hariç tüm sayısal sütunları seç
        EXCLUDE_COLS = ['date', 'Date', 'ticker', 'Ticker', 'target'] 
        features = [col for col in df_train.columns if col not in EXCLUDE_COLS]
        
        # Scaler'ı sadece eğitimde kullandığımız özelliklere fit et.
        global_scaler = MinMaxScaler(feature_range=(0, 1))
        global_scaler.fit(df_train[features])
        
        return model, global_scaler, features
    
    except Exception as e:
        return None, None, None

LSTM_MODEL, GLOBAL_SCALER, FEATURE_COLS = load_laplace_resources()

# --- LAPLACE: SÜRÜM 2.2 (GÜVENLİ ÇALIŞMA) ---
st.set_page_config(page_title="LAPLACE: Neural Terminal V2.2", page_icon="📐", layout="wide")

# --- API KONTROL (AYNI) ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except:
    pass

# --- İZLEME LİSTESİ (AYNI) ---
WATCHLIST = [
    'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'AMD', 'PLTR',
    'AI', 'SMCI', 'ARM', 'PANW', 'ORCL', 'ADBE', 'JPM'
]
WATCHLIST.sort()

# --- CSS: LAPLACE KARANLIK TEMA (AYNI) ---
# ...

# --- YARDIMCI: RSI/MACD/BB HESAPLA ---
def calculate_indicators(df):
    import ta 
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd_indicator = ta.trend.MACD(df['close'])
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['market_sentiment'] = 0.5 
    
    # Yeni eklenen: ATR ve BB'yi eklemeyi unutmuştuk, şimdi ekliyoruz.
    # Miner'da olan tüm özellikleri eklemeliyiz:
    bb_indicator = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bb_indicator.bollinger_hband()
    df['bb_lower'] = bb_indicator.bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    
    df.dropna(inplace=True)
    return df

# --- LSTM PREDICTION MOTORU (GÜVENLİK KONTROLÜ EKLENDİ) ---
def get_lstm_prediction(history_df, model, scaler, features_list):
    if model is None or scaler is None:
        return "MODEL YÜKLENEMEDİ"

    # --- KRİTİK KONTROL (Hata veren yer burasıydı) ---
    missing_cols = [col for col in features_list if col not in history_df.columns]
    
    if missing_cols:
        return f"EKSİK VERİ: Model, {missing_cols} sütununu canlı veride bulamıyor."
    
    if len(history_df) < SEQUENCE_LENGTH:
        return "VERİ YETERSİZ"
    # --- KONTROL BİTTİ ---

    # Gerekli sütunları seç (Ölçekleyiciyi eğittiğimiz sütunlar)
    data_for_scaling = history_df[features_list].copy()

    # Veriyi GLOBAL SCALER ile dönüştür (fit etmeden sadece transform ediyoruz)
    scaled_data = scaler.transform(data_for_scaling) 
    
    # Tahmin için sadece son N günü (60 günü) kullanıyoruz
    X_test = scaled_data[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, scaled_data.shape[1])

    # Tahmini al
    prediction = model.predict(X_test, verbose=0)
    
    # Sonucu Yüzdeye Çevir
    prediction_score = prediction[0][0] * 100 
    
    if prediction_score > 50:
        return f"Yükseliş Olasılığı: %{prediction_score:.2f}"
    else:
        return f"Düşüş Olasılığı: %{100 - prediction_score:.2f}"

# --- MOTOR FONKSİYONLARI (AYNI) ---
# --- laplace_terminal.py içinde get_market_data fonksiyonunu bul ---

@st.cache_data(ttl=600)
def get_market_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Daha uzun geçmiş veri çekiyoruz (LSTM için 60 gün lazım)
        hist = stock.history(period="6mo") 
        if hist.empty: return None, None
        
        # Hata veren satır: if yok, sadece atama var.
        hist.columns = [col.lower() for col in hist.columns] 
        
        # Göstergeleri hesapla
        hist = calculate_indicators(hist)
        
        current_price = hist['close'].iloc[-1]
        summary = {"price": current_price, "rsi": hist['rsi'].iloc[-1]}
        return summary, hist
    except: return None, None



# --- ARAYÜZ AKIŞI ---
st.title("📐 LAPLACE V2.2 (GÜVENLİ ÇALIŞMA)")

if LSTM_MODEL is None or GLOBAL_SCALER is None:
    st.error("⚠️ LSTM Modeli yüklenemedi. Eğitim tamamlandı mı ve tüm dosyalar GitHub'da mı?")
    st.stop() # Hata varsa burada durur

# ----------------------------------------------------
# KRİTİK EKLENECEK ALAN (Buton ve Selectbox Tanımları)
# ----------------------------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    ticker = st.selectbox("Varlık Seçimi", WATCHLIST)
with col2:
    analyze_btn = st.button("HESAPLA ⚡", use_container_width=True, type="primary") # <--- NameError'ı çözen satır
# --- laplace_terminal.py (Ana Akış) ---

# ... (st.title ve st.selectbox kodları)

if analyze_btn: # <-- 1. BLOK BAŞLANGICI
    with st.spinner("Laplace Motorları Çalışıyor..."): # <-- 2. BLOK BAŞLANGICI
        
        # ... (Veri çekme ve tahmin kodları)
        lstm_result = get_lstm_prediction(history_df, LSTM_MODEL, GLOBAL_SCALER, FEATURE_COLS) 
        # lstm_result burada tanımlanır.

        # ... (Gemini kodu)

        # --- EKRAN ÇIKTILARI (BU KISIM MUTLAKA İÇERİDE OLMALI) ---
        st.markdown("### 📈 Teknik & Yapay Zeka Görüşü")
        
        col_lstm, col_gemini = st.columns([1, 2])
        
        with col_lstm: 
            # st.metric gibi st. komutları burada başlar.
            st.metric(label="Yükseliş Olasılığı", value=lstm_result.split(':')[-1].strip())
            # ...
        # ...
            
        with col_gemini:
            # Gemini Analiz Kartı
            st.subheader("Gemini Analizi")
            st.json(gemini_result)
            
        # --- FIX BİTTİ ---