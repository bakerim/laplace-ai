import yfinance as yf
import pandas as pd # Pandas kütüphanesi düzgünce içeri aktarıldı.

def generate_data():
    print("Bitcoin verisi indiriliyor...")
    
    # 5 yıllık BTC-USD verisi indiriliyor
    df = yf.download("BTC-USD", period="5y", interval="1d")

    # Sütun isimlerini düzelt (Multi-index temizliği)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Sütun isimlerini temizle (Baş harfleri büyük, boşluklar temizlenir)
    df.columns = [c.strip().title() for c in df.columns]
    
    # CSV olarak kaydet
    df.to_csv("bitcoin_5yillik.csv")
    print(f"✅ 'bitcoin_5yillik.csv' dosyası {len(df)} satır veri ile oluşturuldu!")
    print("Şimdi bu dosyayı Google Drive'daki 'Laplace_Data' klasörünün İÇİNE yükle.")

if __name__ == "__main__":
    generate_data()