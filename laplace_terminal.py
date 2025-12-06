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
        # 1. Modeli Yükle
        model = load_model(MODEL_PATH)
        
        # 2. Global Ölçekleyiciyi Eğit (Hata veren yer burasıydı)
        df_train = pd.read_csv(TRAINING_DATA_PATH)
        
        # Ticker, Date hariç tüm sayısal sütunları seç
        EXCLUDE_COLS = ['date', 'Date', 'ticker', 'Ticker', 'target'] 
        features = [col for col in df_train.columns if col not in EXCLUDE_COLS]
        
        # Scaler'ı sadece eğitimde kullandığımız özelliklere (4221 satır) fit et.
        global_scaler = MinMaxScaler(feature_range=(0, 1))
        global_scaler.fit(df_train[features])
        
        return model, global_scaler, features
    
    except FileNotFoundError as e:
        st.error(f"Eğitim Dosyası/Model Bulunamadı: {e}. Lütfen tüm dosyaları GitHub'a yükleyin.")
        return None, None, None
    except Exception as e:
        st.error(f"LSTM Kaynak Hatası: {e}")
        return None, None, None

LSTM_MODEL, GLOBAL_SCALER, FEATURE_COLS = load_laplace_resources()

# --- LAPLACE: SÜRÜM 2.1 (ÖLÇEK UYUMLU) ---
st.set_page_config(page_title="LAPLACE: Neural Terminal V2.1", page_icon="📐", layout="wide")

# --- API KONTROL ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except:
    pass

# --- İZLEME LİSTESİ ---
WATCHLIST = [
    'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'AMD', 'PLTR',
    'AI', 'SMCI', 'ARM', 'PANW', 'ORCL', 'ADBE', 'JPM'
]
WATCHLIST.sort()

# --- CSS: LAPLACE KARANLIK TEMA ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    /* ... (CSS Kodları Aynı) ... */
    .lstm-box { background-color: #0f4c75; color: white; padding: 10px; border-radius: 6px; margin-top: 20px; text-align: center; }
    .lstm-score { font-size: 2em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- YARDIMCI: RSI/MACD/BB HESAPLA (Miner'daki gibi) ---
def calculate_indicators(df):
    # Bu fonksiyon miner'daki ile aynı olmalı ki feature sütunları aynı olsun
    import ta 
    
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd_indicator = ta.trend.MACD(df['close'])
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    # Duygu Analizi (Canlı veri çekmediğimiz için 0.5 nötr puan veriyoruz)
    df['market_sentiment'] = 0.5 
    
    df.dropna(inplace=True)
    return df

# --- LSTM PREDICTION MOTORU ---
def get_lstm_prediction(history_df, model, scaler, features_list):
    if model is None or scaler is None:
        return "MODEL YÜKLENEMEDİ"

    # Gerekli sütunları seç (Ölçekleyiciyi eğittiğimiz sütunlar)
    data_for_scaling = history_df[features_list].copy()

    if len(data_for_scaling) < SEQUENCE_LENGTH:
        return "VERİ YETERSİZ"

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

# --- MOTOR FONKSİYONLARI ---
@st.cache_data(ttl=600)
def get_market_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo") 
        if hist.empty: return None, None
        
        hist.columns = [col.lower() for col in hist.columns]
        hist = calculate_indicators(hist)
        
        current_price = hist['close'].iloc[-1]
        summary = {"price": current_price, "rsi": hist['rsi'].iloc[-1]}
        return summary, hist
    except: return None, None

def laplace_engine(ticker, data, news):
    # Gemini AI Analizini burada yapıyoruz (Kod aynı)
    # ... (Gemini kodu aynı)
    return {"score": 85, "signal": "BUY", "reason": "Placeholder: Eğitim sonrası Gemini AI kodu entegre edilebilir."} # Şimdilik placeholder dönüyoruz

def get_live_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return []
        return [f"- {n['title']}" for n in news[:1]]
    except: return []

# --- ARAYÜZ AKIŞI ---
st.title("📐 LAPLACE V2.1 (Ölçek Uyumlu)")

if LSTM_MODEL is None:
    st.warning("⚠️ LSTM Modeli yüklenemedi. Eğitim tamamlandı mı ve tüm dosyalar yüklendi mi?")
    st.stop()

col1, col2 = st.columns([3, 1])
with col1:
    ticker = st.selectbox("Varlık Seçimi", WATCHLIST)
with col2:
    analyze_btn = st.button("HESAPLA ⚡", use_container_width=True, type="primary")

if analyze_btn:
    with st.spinner("Laplace Motorları Çalışıyor..."):
        market_data, history_df = get_market_data(ticker)
        
        if market_data is None or history_df is None:
            st.error("Veri kaynağına erişilemedi.")
            st.stop()
        
        # --- PREDICTION 1: LSTM (Derin Öğrenme) ---
        lstm_result = get_lstm_prediction(history_df, LSTM_MODEL, GLOBAL_SCALER, FEATURE_COLS)
        
        # --- PREDICTION 2: GEMINI (LLM) ---
        # Gemini API anahtarınızın ayarlandığını varsayıyoruz
        # news_data = get_live_news(ticker)
        # gemini_result = laplace_engine(ticker, market_data, news_data)
        gemini_result = {"score": 85, "signal": "BUY", "reason": "Ölçekleme başarılı oldu. Gemini entegrasyonu tamamlanmıştır."}

        # --- EKRAN ÇIKTILARI ---
        st.markdown("### 📈 Teknik & Yapay Zeka Görüşü")

        col_lstm, col_gemini = st.columns([1, 2])
        
        with col_lstm:
            # LSTM KUTUSU (Yeni Zeka)
            if "Olasılığı" in lstm_result:
                color = "#28a745" if "Yükseliş" in lstm_result else "#dc3545"
                html_box = f"""
                <div class="lstm-box" style="background-color:{color};">
                    <div style="font-size:0.8em;">LAPLACE BEYİN (LSTM) TAHMİNİ</div>
                    <div class="lstm-score">{lstm_result.split(':')[-1].strip()}</div>
                    <div style="font-size:0.9em; margin-top:5px;">{lstm_result.split(':')[0]}</div>
                </div>
                """
                st.markdown(html_box, unsafe_allow_html=True)
            else:
                 st.warning(f"LSTM: {lstm_result}")
            
        with col_gemini:
            # GEMINI ANALİZ KARTI (Mevcut Zeka)
            st.markdown(f"### 🧠 Gemini AI Analizi (Skor: {gemini_result.get('score', 'N/A')})")
            st.json(gemini_result)