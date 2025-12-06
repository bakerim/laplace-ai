import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
import requests
import json
import plotly.graph_objects as go  # <-- İŞTE BU SATIR YENİ

# --- LAPLACE: SÜRÜM 1.1 (GRAFİK MODÜLÜ) ---
st.set_page_config(page_title="LAPLACE: Neural Terminal", page_icon="📐", layout="wide")

# --- API KONTROL ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except:
    pass

# --- İZLEME LİSTESİ ---
WATCHLIST = [
    'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC',
    'PLTR', 'AI', 'SMCI', 'ARM', 'PATH', 'SNOW', 'CRWD', 'PANW', 'ORCL', 'ADBE',
    'COIN', 'MSTR', 'MARA', 'RIOT', 'HOOD', 'PYPL', 'SQ', 'V', 'MA', 'JPM',
    'RIVN', 'LCID', 'NIO', 'FSLR', 'ENPH', 'XOM', 'CVX', 'AVGO', 'MU', 'QCOM'
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
</style>
""", unsafe_allow_html=True)

# --- YARDIMCI: RSI HESAPLA ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- MOTOR FONKSİYONLARI ---
def get_market_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Grafik için son 6 aylık veri
        hist = stock.history(period="6mo")
        if hist.empty: return None, None
        
        # Göstergeler
        hist['SMA50'] = hist['Close'].rolling(50).mean()
        hist['RSI'] = calculate_rsi(hist['Close'])
        
        current_price = hist['Close'].iloc[-1]
        sma50 = hist['SMA50'].iloc[-1] if not pd.isna(hist['SMA50'].iloc[-1]) else current_price
        rsi = hist['RSI'].iloc[-1] if not pd.isna(hist['RSI'].iloc[-1]) else 50
        
        trend = "NÖTR"
        if current_price > sma50: trend = "POZİTİF (SMA50 Üstü)"
        else: trend = "NEGATİF (SMA50 Altı)"
        
        summary = {
            "price": current_price,
            "trend": trend,
            "rsi": rsi,
            "volatility": (hist['High'] - hist['Low']).mean()
        }
        return summary, hist
    except: return None, None

def get_live_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return []
        return [f"- {n['title']}" for n in news[:3]]
    except: return []

# --- GRAFİK ÇİZEN FONKSİYON ---
def plot_chart(df, ticker):
    # Mum Grafiği
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name=ticker)])
    
    # SMA 50 Çizgisi
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA 50', line=dict(color='#FFA500', width=1)))

    fig.update_layout(
        title=f'{ticker} - 6 Aylık Trend Analizi',
        yaxis_title='Fiyat (USD)',
        template='plotly_dark', # Karanlık Tema
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        xaxis_rangeslider_visible=False 
    )
    return fig

def laplace_engine(ticker, data, news):
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    news_text = "\n".join(news) if news else "Veri Yok"
    
    prompt = f"""
    SİSTEM: LAPLACE AI (Probability & Risk Engine)
    GÖREV: Finansal varlık için olasılık ve risk hesabı.
    
    VARLIK: {ticker} | FİYAT: ${data['price']:.2f}
    TREND (SMA50): {data['trend']}
    RSI (14): {data['rsi']:.2f} (30 altı aşırı satım, 70 üstü aşırı alım)
    
    HABER AKIŞI:
    {news_text}
    
    HESAPLAMA PROTOKOLÜ:
    1. RSI Kontrolü: Eğer RSI > 70 ise "Aşırı Alım" riski var, puanı biraz kır.
    2. Trend Kontrolü: Fiyat SMA50'nin altındaysa "Düşüş Trendi" uyarısı ver.
    3. Haber Sentiment: Haberler ile teknik durumu birleştir.
    
    ÇIKTI (JSON):
    {{
        "score": (0-100),
        "signal": "STRONG BUY | BUY | WAIT | SELL",
        "reason": "Teknik (RSI/Trend) ve Temel sentez.",
        "entry": (Fiyat),
        "target": (Hedef),
        "stop": (Stop),
        "term": "X Gün"
    }}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '')
        return json.loads(text)
    except: return None

def display_laplace_card(res, ticker):
    score = res['score']
    if score >= 90: c, sig = "tier-s", "ALPHA"
    elif score >= 75: c, sig = "tier-a", "BETA"
    elif score >= 60: c, sig = "tier-b", "GAMMA"
    else: c, sig = "tier-f", "DELTA (RİSK)"
    
    html = f"""<div class="card {c}"><div class="card-header"><div>{ticker} <span style="font-size:0.6em; color:#888;">{sig}</span></div><div class="score-box">{score}</div></div><div class="analysis-text">{res['reason']}</div><div class="data-grid"><div class="grid-item"><div class="label">SİNYAL</div><div class="value" style="color:#58a6ff;">{res['signal']}</div></div><div class="grid-item"><div class="label">GİRİŞ</div><div class="value">${res['entry']}</div></div><div class="grid-item"><div class="label">HEDEF</div><div class="value">${res['target']}</div></div><div class="grid-item"><div class="label">STOP</div><div class="value" style="color:#da3633;">${res['stop']}</div></div></div></div>"""
    st.markdown(html, unsafe_allow_html=True)

# --- ARAYÜZ AKIŞI ---
st.title("📐 LAPLACE v1.1")
st.caption("Advanced Probability & Market Intelligence System")

col1, col2 = st.columns([3, 1])
with col1:
    ticker = st.selectbox("Varlık Seçimi (US Market)", WATCHLIST)
with col2:
    if st.button("HESAPLA ⚡", use_container_width=True):
        st.session_state['run'] = True

if st.session_state.get('run'):
    with st.spinner("Laplace Motoru Çalışıyor..."):
        # Veriyi çek
        market_data, history_df = get_market_data(ticker)
        news_data = get_live_news(ticker)
        
        if market_data:
            # 1. Analiz Kartı
            result = laplace_engine(ticker, market_data, news_data)
            if result:
                display_laplace_card(result, ticker)
                
                # 2. GRAFİK ALANI (Eski versiyonda burası yoktu!)
                st.markdown("### 📈 Teknik Görünüm")
                chart = plot_chart(history_df, ticker)
                st.plotly_chart(chart, use_container_width=True)
                
                with st.expander("Ham Veri Akışı"):
                    st.write("Teknik Veriler:", market_data)
                    st.write("Haberler:", news_data)
        else:
            st.error("Veri kaynağına erişilemedi.")
    st.session_state['run'] = False
