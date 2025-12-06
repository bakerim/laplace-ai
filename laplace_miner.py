import yfinance as yf
import pandas as pd
import ta 
import os
import time
from datetime import datetime

# --- LAPLACE: DATA MINER v1.0 ---
# Görev: Derin Öğrenme modeli için ham madde (Veri Seti) üretmek.

TICKERS = [
    'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'AMD', 'INTC',
    'PLTR', 'COIN', 'MSTR', 'RIOT', 'HOOD', 'PYPL', 'JPM', 'XOM', 'CVX'
]

DATA_DIR = "laplace_dataset"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def mine_technical_data(ticker):
    print(f"⛏️  Kazılıyor: {ticker}...")
    try:
        # 1. Son 2 Yılın verisini indir (Eğitim için büyük veri lazım)
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        
        if df.empty: return None
        
        # 2. Sütun isimlerini temizle (MultiIndex sorunu için)
        df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
        
        # 3. MATEMATİKSEL HESAPLAMALAR (Feature Engineering)
        # Deep Learning modelinin "gözleri" bu indikatörler olacak.
        
        # RSI (Göreceli Güç)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # MACD (Trend Takipçisi)
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_SIGNAL'] = macd['MACDs_12_26_9']
        
        # Bollinger Bantları (Volatilite)
        bb = ta.bbands(df['Close'], length=20)
        df['BB_UPPER'] = bb['BBU_20_2.0']
        df['BB_LOWER'] = bb['BBL_20_2.0']
        
        # ATR (Ortalama Gerçek Aralık - Volatilite)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # Hacim Osilatörü
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        
        # 4. HEDEF BELİRLEME (Labeling)
        # Modelden neyi tahmin etmesini istiyoruz? 
        # "Yarınki kapanış fiyatı, bugünkünden yüksek mi olacak?" (1 = Evet, 0 = Hayır)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # NaN verileri temizle (İndikatör hesaplarken baştaki günler boş kalır)
        df.dropna(inplace=True)
        
        return df
        
    except Exception as e:
        print(f"❌ Hata ({ticker}): {e}")
        return None

def main():
    print("📐 LAPLACE: Veri Madenciliği Protokolü Başlatıldı...")
    print(f"Hedef: {len(TICKERS)} Varlık | Derinlik: 2 Yıl")
    
    combined_data = []
    
    for ticker in TICKERS:
        data = mine_technical_data(ticker)
        if data is not None:
            # Hangi hisse olduğunu kaydet
            data['Ticker'] = ticker
            
            # CSV olarak kaydet (Her hisse için ayrı)
            file_path = f"{DATA_DIR}/{ticker}_training_data.csv"
            data.to_csv(file_path)
            
            combined_data.append(data)
            print(f"✅ Kaydedildi: {file_path} ({len(data)} satır)")
        
        time.sleep(1) # API nezaket süresi

    # Tüm veriyi tek bir dev dosyada birleştir (Model eğitimi için)
    if combined_data:
        full_dataset = pd.concat(combined_data)
        full_dataset.to_csv("laplace_FULL_DATASET.csv")
        print("\n" + "="*50)
        print(f"🏁 MADENCİLİK TAMAMLANDI.")
        print(f"💾 DEV VERİ SETİ: laplace_FULL_DATASET.csv")
        print(f"📊 Toplam Veri Noktası: {len(full_dataset)} Satır")
        print("="*50)
        print("Şimdi bu veriyi 'laplace_brain.py' ile eğitmek için hazırsın.")

if __name__ == "__main__":
    main()


