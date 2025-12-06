import yfinance as yf
import pandas as pd
import ta as ta_lib # HATA DÜZELTİLDİ: Artık kurduğun 'ta' kütüphanesini kullanıyor
from newspaper import Article, build
from datetime import datetime, timedelta
import os
import time

# --- LAPLACE: ÇOKLU VERİ MADENCİSİ V2.1 (TA UYUMLU) ---

TICKERS = [
    'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'AMD', 'PLTR'
]

NEWS_SOURCES = [
    'https://finance.yahoo.com/',
    'https://www.cnbc.com/investing/',
    'https://www.marketwatch.com/latest-news',
    'https://www.reuters.com/markets/'
]

DATA_DIR = "laplace_dataset"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def mine_technical_data(ticker):
    """2 Yıllık fiyat ve indikatör verisini çeker."""
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty: return None
        
        df.columns = df.columns.astype(str)
        df.index = df.index.strftime('%Y-%m-%d')
        
        # 3. MATEMATİKSEL HESAPLAMALAR (TA KÜTÜPHANESİ İLE)
        
        # RSI (Göreceli Güç)
        df['RSI'] = ta_lib.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # MACD (Trend Takipçisi)
        macd_indicator = ta_lib.trend.MACD(df['Close'])
        df['MACD'] = macd_indicator.macd()
        df['MACD_SIGNAL'] = macd_indicator.macd_signal()
        
        # Bollinger Bantları (Volatilite)
        bb_indicator = ta_lib.volatility.BollingerBands(df['Close'])
        df['BB_UPPER'] = bb_indicator.bollinger_hband()
        df['BB_LOWER'] = bb_indicator.bollinger_lband()
        
        # HEDEF BELİRLEME (Yarın Yükselir mi? -> 1=Evet)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        df.dropna(inplace=True)
        return df
        
    except Exception as e:
        print(f"❌ Teknik Hata ({ticker}): {e}")
        return None

def mine_news_data():
    """Çoklu kaynaktan güncel haber metinlerini çeker."""
    all_news = []
    
    for url in NEWS_SOURCES:
        try:
            paper = build(url, memoize_articles=False) 
            
            for article in paper.articles:
                if article.url is None: continue

                if article.publish_date and article.publish_date < datetime.now() - timedelta(hours=24):
                     continue

                art = Article(article.url)
                art.download()
                art.parse()
                
                if art.text and len(art.text) > 300:
                    all_news.append({
                        "date": str(art.publish_date),
                        "source": url,
                        "title": art.title,
                        "text": art.text, 
                        "authors": art.authors
                    })
                
                if len(all_news) % 20 == 0 and len(all_news) > 0:
                     print(f"   [-- {len(all_news)} makale indirildi --]")
                     
        except Exception as e:
            print(f"⚠️ Haber kaynağı ({url}) taranamadı: {e}")
            
    return pd.DataFrame(all_news)

def main():
    print("📐 LAPLACE: Veri Madenciliği Protokolü Başlatıldı...")
    
    # --- 1. TEKNİK VERİ MADENCİLİĞİ ---
    print("\n--- TEKNİK (FİYAT) VERİ TOPLANIYOR ---")
    combined_tech_data = []
    
    for ticker in TICKERS:
        data = mine_technical_data(ticker)
        if data is not None:
            data['Ticker'] = ticker
            combined_tech_data.append(data)
            print(f"✅ Teknik Veri Hazır: {ticker}")
        time.sleep(1) # API nezaket süresi (Yükseltildi)

    if combined_tech_data:
        full_tech_dataset = pd.concat(combined_tech_data)
        tech_file = f"{DATA_DIR}/laplace_TECH_DATASET.csv"
        full_tech_dataset.to_csv(tech_file)
        print(f"\n💾 Teknik Veri Toplamı Kaydedildi: {tech_file}")
    
    # --- 2. TEMEL/NLP VERİ MADENCİLİĞİ ---
    print("\n--- HABER (TEMEL) METİN VERİSİ TOPLANIYOR ---")
    news_df = mine_news_data()
    
    if not news_df.empty:
        news_file = f"{DATA_DIR}/laplace_NEWS_DATASET.csv"
        news_df.to_csv(news_file, index=False)
        print(f"\n💾 {len(news_df)} adet Temiz Makale Kaydedildi: {news_file}")
    else:
        print("⚠️ Güncel makale bulunamadı.")
    
    print("\n" + "="*50)
    print("🏁 LAPLACE MİNER TAMAMLANDI.")
    print("="*50)

if __name__ == "__main__":
    main()
