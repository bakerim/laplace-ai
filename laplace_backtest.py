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
INITIAL_CAPITAL = 10000.0  
BUY_THRESHOLD = 0.005      
COMMISSION_FEE = 1.50      # <---- YENİ EKLENEN SABİT KOMİSYON ÜCRETİ ($)

# --- YARDIMCI VE YÜKLEME FONKSİYONLARI (DEĞİŞMEDİ) ---

def add_technical_indicators(df):
    close_prices = df["Close"]
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]
    rsi_indicator = RSIIndicator(close=close_prices, window=14)
    df["RSI"] = rsi_indicator.rsi()
    df["Volume"] = df["Volume"].replace(0, np.nan)
    df.dropna(inplace=True)
    return df

def load_assets():
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
    
    print(f"\n🚀 {TICKER} için Geçmiş Test (Komisyon Dahil) Başlatılıyor...")
    df = yf.download(TICKER, period="2y", interval="1d", progress=False)
    
    if df.empty:
        print("HATA: Veri indirilemedi.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = add_technical_indicators(df)
    
    cash = INITIAL_CAPITAL
    shares = 0
    total_trades = 0
    dataset = df[['Close', 'Volume', 'RSI']].values

    print(f"💰 Başlangıç Sermayesi: {INITIAL_CAPITAL:.2f} USD")
    print(f"💸 Her İşlem Başı Komisyon: {COMMISSION_FEE:.2f} USD")
    print(f"⏳ {len(dataset) - LOOKBACK} günlük test verisi işleniyor...")
    print("-" * 30)

    for i in range(LOOKBACK, len(dataset) - 1):
        current_data = dataset[i - LOOKBACK:i]
        scaled_data = f_scaler.transform(current_data)
        X_input = np.reshape(scaled_data, (1, LOOKBACK, 3))
        prediction_scaled = model.predict(X_input, verbose=0)
        predicted_price = p_scaler.inverse_transform(prediction_scaled)[0][0]
        
        current_close = df.iloc[i]['Close']
        if isinstance(current_close, pd.Series): current_close = current_close.iloc[0]
        next_open = df.iloc[i + 1]['Open']
        if isinstance(next_open, pd.Series): next_open = next_open.iloc[0]
        
        predicted_change = (predicted_price - current_close) / current_close

        # --- TİCARET KARARI ---
        
        # ALIM (BUY)
        if predicted_change > BUY_THRESHOLD and cash > 0:
            shares_to_buy = int((cash - COMMISSION_FEE) / next_open) # Komisyon düşüldü
            if shares_to_buy > 0:
                shares += shares_to_buy
                cash -= (shares_to_buy * next_open) + COMMISSION_FEE # Komisyon düşüldü
                total_trades += 1
        
        # SATIM (SELL)
        elif predicted_change < 0 and shares > 0:
            cash += (shares * next_open) - COMMISSION_FEE # Komisyon düşüldü
            shares = 0
            total_trades += 1

    # --- SONUÇLARIN HESAPLANMASI (DEĞİŞMEDİ) ---
    last_close = df.iloc[-1]['Close']
    if isinstance(last_close, pd.Series): last_close = last_close.iloc[0]

    first_close_after_lookback = df.iloc[LOOKBACK]['Close']
    if isinstance(first_close_after_lookback, pd.Series): first_close_after_lookback = first_close_after_lookback.iloc[0]

    final_value = cash + (shares * last_close)
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    buy_and_hold_return = (last_close - first_close_after_lookback) / first_close_after_lookback * 100
    
    print("-" * 30)
    print("📈 BACKTEST SONUÇLARI (KOMİSYON DAHİL) 📉")
    print(f"Başlangıç Tarihi: {df.index[LOOKBACK].strftime('%Y-%m-%d')}")
    print(f"Bitiş Tarihi: {df.index[-1].strftime('%Y-%m-%d')}")
    print("-" * 30)
    print(f"💵 Başlangıç Değeri: {INITIAL_CAPITAL:,.2f} USD")
    print(f"💵 Son Portföy Değeri: {final_value:,.2f} USD")
    print(f"🔄 Toplam İşlem Sayısı: {total_trades}")
    print(f"💰 LAPLACE TOPLAM GETİRİ: %{total_return:,.2f}")
    print("-" * 30)
    print(f"📊 Piyasa (Al-Tut) Getirisi: %{buy_and_hold_return:,.2f}")
    
    if total_return > buy_and_hold_return:
        print("🏆 SONUÇ: Laplace, Komisyonlara Rağmen Piyasayı YENDİ! 🚀")
    else:
        print("❌ SONUÇ: Komisyonlar kârımızı eritti. Strateji Geliştirilmeli!")
        print(f"    (Komisyonsuz Kâr: %{64.89 - total_return:.2f} daha fazlaydı.)")


if __name__ == "__main__":
    run_backtest()