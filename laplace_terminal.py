import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
import sqlite3
import json
from datetime import datetime

# --- LAPLACE: SÜRÜM 1.3 (HAFIZA & DB) ---
st.set_page_config(page_title="LAPLACE: Neural Terminal", page_icon="📐", layout="wide")

# --- DATABASE KURULUMU (OTOMATİK) ---
def init_db():
    conn = sqlite3.connect('laplace_memory.db')
    c = conn.cursor()
    # Tablo yoksa oluştur
    c.execute('''CREATE TABLE IF NOT EXISTS signals
                 (date TEXT, ticker TEXT, price REAL, rsi REAL, score INTEGER, signal TEXT, reason TEXT)''')
    conn.commit()
    conn.close()

def save_signal(ticker, price, rsi, score, signal, reason):
    conn = sqlite3.connect('laplace_memory.db')
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    c.execute("INSERT INTO signals VALUES (?,?,?,?,?,?,?)", 
              (date_str, ticker, price, rsi, score, signal, reason))
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect('laplace_memory.db')
    df = pd.read_sql_query("SELECT * FROM signals ORDER BY date DESC LIMIT 50", conn)
    conn.close()
    return df

# Veritabanını başlat
init_db()

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
@st.cache_data(ttl=600) 
def get_market_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        if hist.empty: return None, None
        
        hist['SMA50'] = hist['Close'].rolling(50).mean()
        hist['RSI'] = calculate_rsi(hist['Close'])
        
        current_price = hist['Close'].iloc[-1]
        sma50 = hist['SMA50'].iloc[-1] if not pd.isna(hist['SMA50'].iloc[-1]) else current_price
        rsi = hist['RSI'].iloc[-1] if not pd.isna(hist['RSI'].iloc[-1]) else 50
        
        trend = "POZİTİF" if current_price > sma50 else "NEGATİF"
        
        summary = {
            "price": current_price,
            "trend": trend,
            "rsi": rsi,
            "volatility": (hist['High'] - hist['Low']).mean()
        }
        return summary, hist
    except: return None, None

@st.cache_data(ttl=600)
def get_live_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return []
        return [f"- {n['title']}" for n in news[:3]]
    except: return []

def plot_chart(df, ticker):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name=ticker)])
    
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA 50', line=dict(color='#FFA500', width=1)))

    fig.update_layout(
        title=f'{ticker} - 3 Aylık Trend',
        yaxis_title='Fiyat (USD)',
        template='plotly_dark',
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        xaxis_rangeslider_visible=False 
    )
    return fig

def laplace_engine(ticker, data, news):
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    news_text = "\n".join(news) if news else "Veri Yok"
    
    prompt = f"""
    SİSTEM: LAPLACE AI
    VARLIK: {ticker} | FİYAT: ${data['price']:.2f} | RSI: {data['rsi']:.2f}
    HABERLER: {news_text}
    
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

def display_laplace_card(res, ticker):
    score = res['score']
    if score >= 90: c, sig = "tier-s", "ALPHA"
    elif score >= 75: c, sig = "tier-a", "BETA"
    elif score >= 60: c, sig = "tier-b", "GAMMA"
    else: c, sig = "tier-f", "DELTA"
    
    html = f"""<div class="card {c}"><div class="card-header"><div>{ticker} <span style="font-size:0.6em; color:#888;">{sig}</span></div><div class="score-box">{score}</div></div><div class="analysis-text">{res['reason']}</div><div class="data-grid"><div class="grid-item"><div class="label">SİNYAL</div><div class="value" style="color:#58a6ff;">{res['signal']}</div></div><div class="grid-item"><div class="label">GİRİŞ</div><div class="value">${res['entry']}</div></div><div class="grid-item"><div class="label">HEDEF</div><div class="value">${res['target']}</div></div><div class="grid-item"><div class="label">STOP</div><div class="value" style="color:#da3633;">${res['stop']}</div></div></div></div>"""
    st.markdown(html, unsafe_allow_html=True)

# --- ARAYÜZ AKIŞI ---
st.title("📐 LAPLACE v1.3 (Memory)")

# Sekmeler
tab1, tab2 = st.tabs(["⚡ Terminal", "💾 Hafıza Kayıtları"])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.selectbox("Varlık Seçimi", WATCHLIST)
    with col2:
        analyze_btn = st.button("HESAPLA ⚡", use_container_width=True, type="primary")

    if analyze_btn:
        with st.spinner("Laplace Motoru Çalışıyor..."):
            market_data, history_df = get_market_data(ticker)
            news_data = get_live_news(ticker)
            
            if market_data:
                st.markdown("### 📈 Teknik Görünüm")
                chart = plot_chart(history_df, ticker)
                st.plotly_chart(chart, use_container_width=True)
                
                result = laplace_engine(ticker, market_data, news_data)
                
                if result:
                    st.markdown("### 🧠 AI Analizi")
                    display_laplace_card(result, ticker)
                    
                    # --- HAFIZAYA KAYDETME ANI ---
                    save_signal(ticker, market_data['price'], market_data['rsi'], 
                                result['score'], result['signal'], result['reason'])
                    st.toast(f"✅ {ticker} analizi veritabanına kaydedildi!", icon="💾")
                    
            else:
                st.error("Veri kaynağına erişilemedi.")

with tab2:
    st.markdown("### 🗄️ Laplace Veri Seti (Son 50 Sinyal)")
    history = load_history()
    if not history.empty:
        st.dataframe(history, use_container_width=True)
    else:
        st.info("Henüz kayıtlı analiz yok. Terminalden ilk analizini yap!")
