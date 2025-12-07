import yfinance as yf
import pandas as pd

def generate_data():
    print("Ethereum verisi indiriliyor...")
    
    # BTC-USD yerine ETH-USD indiriliyor
    df = yf.download("ETH-USD", period="5y", interval="1d")

    # Sütun isimlerini düzelt
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Dosya adını değiştiriyoruz
    df.to_csv("ethereum_5yillik.csv")
    print(f"✅ 'ethereum_5yillik.csv' dosyası {len(df)} satır veri ile oluşturuldu!")

if __name__ == "__main__":
    generate_data()