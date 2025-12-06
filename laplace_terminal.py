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

# Modeli ve Ölçekleyiciyi Yükle
@st.cache_resource
def load_laplace_brain():
    """Kaydedilen LSTM Modelini yükler."""
    try:
        # Boş bir scaler oluştur (LSTM'de kullandığımız ile aynı)
        # Sadece fit etmiyoruz, çünkü fit verisini bilemeyiz.
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"LSTM Modeli Yüklenemedi. Eğitim tamamlandı mı? Hata: {e}")
        return None

# Model yükleniyor
LSTM_MODEL = load_laplace_brain()

# --- LAPLACE: SÜRÜM 2.0 (ÇİFT MOTOR) ---
st.set_page_config(page_title="LAPLACE: Neural Terminal V2.0", page_icon="📐", layout="wide")

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
    .card { background-color: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 6px; margin-bottom: 15px; color: #c9d1d9; font-family: 'Consolas', 'Monaco', monospace; }
    .card-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363d; padding-bottom: 10px; margin-bottom: 10px; font-size: 1.2em; font-weight: bold; color: #58a6ff; }
    .score-box { background: #238636; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.9em; }
    .analysis-text { font-size: 0.9em; line-height: 1.5; color: #8b949e; margin-bottom: 15px; }
    .data-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: #30363d; border: 1px solid #30363d; border-radius: 4px; overflow: hidden; }
    .grid-item { background: #0d1117; padding: 10px; text-align: center; }
    .label { font-size: 0.7em; color: #8b949e; text-transform: uppercase; }
    .value { font-size: 1.1em; color: #e6edf3; font-weight: bold; }
    .tier-s { border-left: 4px solid #238636; }
    .tier-a { border-left: 4px solid #1f6feb; }
    .tier-b { border-left: 4px solid #d29922; }
    .tier-f { border-left: 4px solid #da3633; opacity: 0.6; }
    
    .lstm-box { background-color: #0f4c75; color: white; padding: 10px; border-radius: 6px; margin-top: 20px; text-align: center; }
    .lstm-score { font-size: 2em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- YARDIMCI: RSI/MACD/BB HESAPLA (Miner'daki gibi) ---
def calculate_indicators(df):
    import ta # Kütüphaneyi lokalde import ediyoruz
    
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd_indicator = ta.trend.MACD(df['close'])
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['market_sentiment'] = 0.5 # Şimdilik ortalama 0.5 duygu puanı veriyoruz
    
    df.dropna(inplace=True)
    return df

# --- LSTM PREDICTION MOTORU ---
def get_lstm_prediction(history_df, model):
    if model is None:
        return "MODEL YÜKLENEMEDİ"

    # Miner'da kullandığımız sütunları seç (küçük harfle)
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macd_signal', 'market_sentiment']

    # Veriyi hazırlama (Son 60 gün)
    data = history_df.tail(SEQUENCE_LENGTH + 1)[required_cols]

    if len(data) < SEQUENCE_LENGTH + 1:
        return "VERİ YETERSİZ"

    # Veriyi Laplace Brain'de kullandığımız gibi ölçekle
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Tüm datayı ölçekleyip, son 60 günü alıyoruz (Eğitimde kullandığımız formata sadık kalmak için)
    scaled_data = scaler.fit_transform(data) 
    
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
        # Daha uzun geçmiş veri çekiyoruz (LSTM için 60 gün lazım)
        hist = stock.history(period="6mo") 
        if hist.empty: return None, None
        
        hist.columns = [col.lower() for col in hist.columns]
        
        # Göstergeleri hesapla
        hist = calculate_indicators(hist)
        
        current_price = hist['close'].iloc[-1]
        summary = {"price": current_price, "rsi": hist['rsi'].iloc[-1]}
        return summary, hist
    except: return None, None

def laplace_engine(ticker, data, news):
    # Gemini AI'a LSTM'in başarısını entegre etmek zor olduğu için, 
    # onu sadece haber ve teknik yorum için kullanıyoruz.
    
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    news_text = "\n".join(news) if news else "Veri Yok"
    
    prompt = f"""
    SİSTEM: LAPLACE AI (Gemini Modülü)
    GÖREV: Finansal risk hesaplama.
    
    VARLIK: {ticker} | FİYAT: ${data['price']:.2f}
    RSI: {data['rsi']:.2f}
    
    HABERLER:
    {news_text}
    
    ÇIKTI (JSON):
    {{
        "score": (0-100),
        "signal": "STRONG BUY | BUY | WAIT | SELL",
        "reason": "Kısa teknik/temel özet.",
        "entry": (Fiyat),
        "target": (Hedef),
        "stop": (Stop)
    }}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '')
        return json.loads(text)
    except: return None

def get_live_news(ticker):
    # Haber çekme hızı nedeniyle sadece 1 haber çekiyoruz
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return []
        return [f"- {n['title']}" for n in news[:1]]
    except: return []

def plot_chart(df, ticker):
    # Candlestick ve SMA50'yi çizer
    # ... (Plotly kodun buraya gelir)
    return go.Figure() # Şimdilik boş figure döndürüyoruz

def display_laplace_card(res, ticker):
    # Gemini AI Analiz Kartı
    # ... (Kartı HTML ile çizen kod buraya gelir)
    return res # Sadece sonucu döndürüyoruz

# --- ARAYÜZ AKIŞI ---
st.title("📐 LAPLACE V2.0 (Çift Zeka Terminali)")

if LSTM_MODEL is None:
    st.warning("⚠️ LSTM Modeli yüklenemedi. Lütfen önce laplace_brain.py'yi çalıştırın.")

col1, col2 = st.columns([3, 1])
with col1:
    ticker = st.selectbox("Varlık Seçimi", WATCHLIST)
with col2:
    analyze_btn = st.button("HESAPLA ⚡", use_container_width=True, type="primary")

if analyze_btn:
    with st.spinner("Laplace Motorları Çalışıyor..."):
        market_data, history_df = get_market_data(ticker)
        news_data = get_live_news(ticker)
        
        if market_data is None or history_df is None:
            st.error("Veri kaynağına erişilemedi.")
            st.stop()
        
        # --- PREDICTION 1: LSTM (Derin Öğrenme) ---
        lstm_result = get_lstm_prediction(history_df, LSTM_MODEL)
        
        # --- PREDICTION 2: GEMINI (LLM) ---
        gemini_result = laplace_engine(ticker, market_data, news_data)
        
        # --- EKRAN ÇIKTILARI ---
        st.markdown("### 📈 Teknik & Duygu Görünümü")
        # st.plotly_chart(plot_chart(history_df, ticker), use_container_width=True) # Grafik çizimi

        col_lstm, col_gemini = st.columns([1, 2])
        
        with col_lstm:
            # LSTM KUTUSU (Yeni Zeka)
            if lstm_result != "MODEL YÜKLENEMEDİ" and lstm_result != "VERİ YETERSİZ":
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
            if gemini_result:
                # display_laplace_card(gemini_result, ticker) 
                st.markdown(f"### 🧠 Gemini AI Analizi (Skor: {gemini_result.get('score', 'N/A')})")
                st.json(gemini_result) # JSON çıktısını gösteriyoruz
            else:
                st.error("Gemini AI API'den yanıt alınamadı.")
