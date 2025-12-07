import yfinance as yf

# Bitcoin verisi indirelim (Büyük Veri Testi için ideal)
print("Bitcoin verisi indiriliyor...")
df = yf.download("BTC-USD", period="5y", interval="1d")

# Sütun isimlerini düzelt (Multi-index temizliği)
if isinstance(df.columns, import pandas as pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# CSV olarak kaydet
df.to_csv("bitcoin_5yillik.csv")
print("✅ 'bitcoin_5yillik.csv' dosyası oluşturuldu! Bunu Drive'a yükle.")
