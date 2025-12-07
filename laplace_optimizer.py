import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from ta.momentum import RSIIndicator
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

# --- AYARLAR (Backtest Modülünden Alınmıştır) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)
TICKER = "ETH-USD"
LOOKBACK = 60
MODEL_PATH = "laplace_lstm_model.h5"
FEATURE_SCALER_PATH = "laplace_feature_scaler.pkl"
PRICE_SCALER_PATH = "laplace_price_scaler.pkl"
INITIAL_CAPITAL = 10000.0
COMMISSION_FEE = 1.50

# Test edilecek eşik değerleri (0.1%'den 1.5%'e kadar)
THRESHOLDS = [0.001, 0.002, 0.005, 0.0075, 0.01, 0.015] 

# --- YARDIMCI VE YÜKLEME FONKSİYONLARI (DEĞİŞMEDİ) ---

def add_technical_indicators(df):
    close_prices = df["Close"]
    if isinstance(close_prices, pd.DataFrame): close_prices = close_prices.iloc[:, 0]
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
        print(f"HATA: Dosyalar yüklenemedi. Önce trainer'ı çalıştırın. Hata: {e}")
        return None, None, None

def run_backtest_with_threshold(df, model, f_scaler, p_scaler, threshold):
    """Belirtilen eşik ile tek bir backtest çalıştırır ve getiriyi döner."""
    
    cash = INITIAL_CAPITAL
    shares = 0
    total_trades = 0
    dataset = df[['Close', 'Volume', 'RSI']].values

    for i in range(LOOKBACK, len(dataset) - 1):
        # Tahmin süreci
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
        if predicted_change > threshold and cash > 0:
            shares_to_buy = int((cash - COMMISSION_FEE) / next_open)
            if shares_to_buy > 0:
                shares += shares_to_buy
                cash -= (shares_to_buy * next_open) + COMMISSION_FEE
                total_trades += 1
        
        # SATIM (SELL)
        elif predicted_change < 0 and shares > 0:
            cash += (shares * next_open) - COMMISSION_FEE
            shares = 0
            total_trades += 1

    last_close = df.iloc[-1]['Close']
    if isinstance(last_close, pd.Series): last_close = last_close.iloc[0]
    
    final_value = cash + (shares * last_close)
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    return total_return, total_trades

# --- ANA OPTİMİZASYON FONKSİYONU ---

def optimize_thresholds():
    model, f_scaler, p_scaler = load_assets()
    if not all([model, f_scaler, p_scaler]):
        return
    
    print(f"\n🧠 BTC-USD Optimizasyon Başlatılıyor...")
    
    # Veriyi bir kez çekelim
    df = yf.download(TICKER, period="2y", interval="1d", progress=False)
    if df.empty: return
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = add_technical_indicators(df)

    results = []
    
    # Her eşik için backtesti çalıştır
    for threshold in THRESHOLDS:
        ret, trades = run_backtest_with_threshold(df.copy(), model, f_scaler, p_scaler, threshold)
        results.append({
            'Threshold': threshold * 100,
            'Return': ret,
            'Trades': trades
        })

    # En iyi sonucu bul
    best_result = max(results, key=lambda x: x['Return'])
    
    # Piyasa getirisi (Tekrar hesaplandı)
    last_close = df.iloc[-1]['Close']
    first_close_after_lookback = df.iloc[LOOKBACK]['Close']
    market_return = (last_close - first_close_after_lookback) / first_close_after_lookback * 100
    
    print("\n" + "=" * 50)
    print("🥇 OPTİMİZASYON SONUÇ RAPORU 🥇")
    print(f"Piyasa (Al-Tut) Getirisi: %{market_return:.2f}")
    print("-" * 50)
    
    # Raporlama
    for res in sorted(results, key=lambda x: x['Return'], reverse=True):
        status = " (Piyasayı Yendi!)" if res['Return'] > market_return else ""
        print(f"| Eşik: %{res['Threshold']:.2f} | Getiri: %{res['Return']:.2f} | İşlem: {res['Trades']}{status}")
        
    print("-" * 50)
    print(f"💡 EN İYİ EŞİK: %{best_result['Threshold']:.2f} ile %{best_result['Return']:.2f} getiri.")
    print("=" * 50)
    
    if best_result['Return'] > market_return:
        print("✅ ŞİMDİ BU YENİ EŞİK İLE TEKRAR DENEMELİYİZ.")
    else:
        print("❌ HİÇBİR EŞİK İŞE YARAMADI. MODEL GELİŞTİRİLMELİ.")


if __name__ == "__main__":
    optimize_thresholds()