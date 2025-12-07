import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from ta.momentum import RSIIndicator
import os
import warnings
from sklearn.preprocessing import MinMaxScaler

# --- AYARLAR ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)
TICKER = "AAPL"
LOOKBACK = 60
MODEL_PATH = "laplace_lstm_model.h5"
FEATURE_SCALER_PATH = "laplace_feature_scaler.pkl"
PRICE_SCALER_PATH = "laplace_price_scaler.pkl"
INITIAL_CAPITAL = 10000.0  # Başlangıç Sermayesi (Sanal)
BUY_THRESHOLD = 0.005      # Fiyat artış beklentisi %0.5'ten fazlaysa AL

# --- YARDIMCI FONKSİYONLAR ---

def add_technical_indicators(df):
    """Veriye RSI ve Hacim kontrollerini ekler (Trainer'daki ile aynı)"""
    rsi_indicator = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi_indicator.rsi()
    df["Volume"] = df["Volume"].replace(0, np.nan)
    df.dropna(inplace=True)
    return df

def load_assets():
    """Modeli ve Scalerları yükler"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        f_scaler = joblib.load(FEATURE_SCALER_PATH)
        p_scaler = joblib.load(PRICE_SCALER_PATH)
        return model, f_scaler, p_scaler
    except Exception as e:
        print(f"HATA: Gerekli dosyalar yüklenemedi. Önce trainer.py'yi çalıştırın. Hata: {e}")
        return None, None, None

# --- BACKTEST FONKSİYONU ---

def run_backtest():
    model, f_scaler, p_scaler = load_assets()
    if not all([model, f_scaler, p_scaler]):
        return
    
    print(f"\n🚀 {TICKER} için Geçmiş Test (Backtest) Başlatılıyor...")
    df = yf.download(TICKER, period="2y", interval="1d", progress=False) # Son 2 yıllık veri
    if df.empty:
        print("HATA: Veri indirilemedi.")
        return
        
    df = add_technical_indicators(df)
    
    # Portfolio takip değişkenleri
    cash = INITIAL_CAPITAL
    shares = 0
    total_trades = 0
    profitable_trades = 0
    
    # Veri setini tahmin için hazırlama (Close, Volume, RSI)
    dataset = df[['Close', 'Volume', 'RSI']].values

    print(f"💰 Başlangıç Sermayesi: {INITIAL_CAPITAL:.2f} USD")
    print(f"⏳ {len(dataset) - LOOKBACK} günlük test verisi bulundu.")
    print("-" * 30)

    for i in range(LOOKBACK, len(dataset) - 1):
        # 1. Tahmin İçin Pencereyi Al
        current_data = dataset[i - LOOKBACK:i]
        
        # 2. Ölçeklendir, Boyutlandır ve Tahmin Yap
        scaled_data = f_scaler.transform(current_data)
        X_input = np.reshape(scaled_data, (1, LOOKBACK, 3))
        prediction_scaled = model.predict(X_input, verbose=0)
        
        # 3. Tahmini Fiyatı Geri Çevir
        predicted_price = p_scaler.inverse_transform(prediction_scaled)[0][0]
        
        # O günkü fiyat
        current_close = df.iloc[i]['Close']
        
        # Bir sonraki günkü gerçek fiyat (Test için bunu kullanacağız)
        next_open = df.iloc[i + 1]['Open'] 
        
        # Tahmin Edilen Yüzdelik Değişim
        predicted_change = (predicted_price - current_close) / current_close

        # --- TİCARET KARARI ---
        
        # KARAR 1: ALIM (BUY)
        if predicted_change > BUY_THRESHOLD and cash > 0:
            # Tüm parayla alım yap
            shares_to_buy = int(cash / next_open)
            if shares_to_buy > 0:
                shares += shares_to_buy
                cash -= shares_to_buy * next_open
                # print(f"ALIM: {df.index[i].strftime('%Y-%m-%d')} | Fiyat: {next_open:.2f} | Pay: {shares_to_buy}")
        
        # KARAR 2: SATIM (SELL) - Karar verme mekanizması: Eğer model düşüş bekliyorsa ve elimizde hisse varsa sat.
        elif predicted_change < 0 and shares > 0:
            # Tüm hisseleri sat
            cash += shares * next_open
            shares = 0
            # print(f"SATIM: {df.index[i].strftime('%Y-%m-%d')} | Fiyat: {next_open:.2f}")

    # --- SONUÇLARIN HESAPLANMASI ---
    
    final_value = cash + (shares * df.iloc[-1]['Close'])
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # Karşılaştırma: Eğer hiç işlem yapmayıp başta alsaydık ne olurdu? (Buy & Hold)
    buy_and_hold_return = (df.iloc[-1]['Close'] - df.iloc[LOOKBACK]['Close']) / df.iloc[LOOKBACK]['Close'] * 100
    
    print("-" * 30)
    print("📈 BACKTEST SONUÇLARI 📉")
    print(f"Başlangıç Tarihi: {df.index[LOOKBACK].strftime('%Y-%m-%d')}")
    print(f"Bitiş Tarihi: {df.index[-1].strftime('%Y-%m-%d')}")
    print("-" * 30)
    print(f"💵 Başlangıç Değeri: {INITIAL_CAPITAL:,.2f} USD")
    print(f"💵 Son Portföy Değeri: {final_value:,.2f} USD")
    print(f"💰 LAPLACE TOPLAM GETİRİ: %{total_return:,.2f}")
    print("-" * 30)
    print(f"📊 Karşılaştırma (Al-Tut): %{buy_and_hold_return:,.2f}")
    
    if total_return > buy_and_hold_return:
        print("🏆 SONUÇ: Laplace, Al-Tut stratejisinden DAHA İYİ performans gösterdi!")
    else:
        print("⚠️ SONUÇ: Laplace, Al-Tut stratejisinin gerisinde kaldı. Model/Strateji Geliştirilmeli.")


if __name__ == "__main__":
    run_backtest()
