import yfinance as yf
import pandas as pd
import ta 
from newspaper import Article, build
from datetime import datetime, timedelta
import os
import time

# --- LAPLACE: ÇOKLU VERİ MADENCİSİ V3.3 ---

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
    print(f"⛏️  Kazılıyor: {ticker}...", end="")
    try:
        # 1. Son 2 Yılın verisini indir
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty: 
            print(" ❌ Veri Boş")
            return None
        
        # --- PANDAS UYUMLULUK FIX (V3.3) ---
        if isinstance(df.columns, pd.MultiIndex):
            # En güvenli yöntem: İkinci (redundant) seviyeyi atarak sütunları tek seviyeye indir
            df = df.droplevel(1, axis=1)
        
        df.columns = [col.lower() for col in df.columns] # Sütun isimlerini küçük harfe çevir
        df.index = df.index.strftime('%Y-%m-%d')
        # --- FIX BİTTİ ---

        # 3. MATEMATİKSEL HESAPLAMALAR
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        macd_indicator = ta.trend.MACD(df['close'])
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        
        bb_indicator = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()
        
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()

        # HEDEF BELİRLEME (Yarın Yükselir mi? -> 1=Evet)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        df.dropna(inplace=True)
        print(" ✅ Teknik Veri Hazır.")
        return df
        
    except Exception as e:
        print(f" ❌ Kritik Hata: {e}")
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
            pass
            
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
        time.sleep(1) 

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