import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from ta.momentum import RSIIndicator
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
from laplace_drive_loader import load_data_from_drive

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)

LOOKBACK = 60
MODEL_PATH = "laplace_lstm_model.h5"
FEATURE_SCALER_PATH = "laplace_feature_scaler.pkl"
INITIAL_CAPITAL = 10000.0
COMMISSION_FEE = 1.50
# KRİTİK DEĞİŞİKLİK: Eşiği sıfıra indirdik. Pozitif düşündüğü an alacak.
BUY_THRESHOLD = 0.0 

def prepare_data(df):
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    rsi_indicator = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi_indicator.rsi()
    df["Volume"] = df["Volume"].replace(0, np.nan)
    df.dropna(inplace=True)
    return df

def run_backtest():
    print(f"\n🚀 BTC-USD DEBUG BACKTEST (Konuşkan Mod)...")
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        f_scaler = joblib.load(FEATURE_SCALER_PATH)
    except:
        print("❌ Model dosyaları bulunamadı.")
        return

    df = load_data_from_drive()
    if df is None: return
    
    df = prepare_data(df)
    dataset = df[['Log_Ret', 'Volume', 'RSI']].values
    
    cash = INITIAL_CAPITAL
    shares = 0
    total_trades = 0
    
    print(f"⏳ {len(dataset) - LOOKBACK} gün taranıyor. Örnek tahminler aşağıda:")
    print(f"{'GÜN':<5} | {'TAHMİN (%)':<15} | {'KARAR'}")
    print("-" * 40)
    
    for i in range(LOOKBACK, len(dataset) - 1):
        current_data = dataset[i - LOOKBACK:i]
        scaled_data = f_scaler.transform(current_data)
        X_input = np.reshape(scaled_data, (1, LOOKBACK, 3))
        
        pred_scaled = model.predict(X_input, verbose=0)
        
        # Geri dönüşüm (Inverse Transform)
        dummy = np.zeros(shape=(1, 3))
        dummy[0, 0] = pred_scaled[0][0]
        inverse_scaled = f_scaler.inverse_transform(dummy)
        predicted_log_ret = inverse_scaled[0][0]
        
        # Yüzdeye çevir (örn: 0.02 -> %2)
        predicted_change = (np.exp(predicted_log_ret) - 1)
        
        # --- DEBUG ÇIKTISI (Her 50 günde bir ne düşündüğünü yaz) ---
        if i % 50 == 0:
            decision = "NÖTR"
            if predicted_change > BUY_THRESHOLD: decision = "ALIM 🟢"
            elif predicted_change < 0: decision = "SATIM 🔴"
            print(f"{i:<5} | %{predicted_change*100:<14.4f} | {decision}")

        # --- TİCARET KARARI (YENİ KESİRLİ PAY LOGİĞİ) ---
        next_open = df.iloc[i + 1]['Open']
        
        # ALIM (BUY)
        if predicted_change > BUY_THRESHOLD and cash > 0:
            
            # Komisyonu nakitten düş ve kalan tüm parayla alım yap (Kesirli alım)
            buy_amount = cash - COMMISSION_FEE 
            
            if buy_amount > 0 and next_open > 0:
                shares_to_buy = buy_amount / next_open
                
                # Kesirli Pay Alımı
                shares += shares_to_buy
                cash -= buy_amount + COMMISSION_FEE # Toplam harcanan miktar = pay + komisyon
                total_trades += 1
        
        # SATIM (SELL)
        elif predicted_change < 0 and shares > 0:
            # Tüm hisseleri sat
            cash += (shares * next_open) - COMMISSION_FEE
            shares = 0
            total_trades += 1

        # --- TİCARET KARARI ---
        next_open = df.iloc[i + 1]['Open']
        
        if predicted_change > BUY_THRESHOLD and cash > 0:
            shares_to_buy = int((cash - COMMISSION_FEE) / next_open)
            if shares_to_buy > 0:
                shares += shares_to_buy
                cash -= (shares_to_buy * next_open) + COMMISSION_FEE
                total_trades += 1
        
        elif predicted_change < 0 and shares > 0:
            cash += (shares * next_open) - COMMISSION_FEE
            shares = 0
            total_trades += 1

    last_close = df.iloc[-1]['Close']
    final_value = cash + (shares * last_close)
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    start_price = df.iloc[LOOKBACK]['Close']
    market_return = (last_close - start_price) / start_price * 100
    
    print("-" * 30)
    print(f"💵 Son Portföy: {final_value:,.2f} USD")
    print(f"🔄 İşlem Sayısı: {total_trades}")
    print(f"💰 LAPLACE GETİRİSİ: %{total_return:,.2f}")
    print(f"📊 PİYASA GETİRİSİ: %{market_return:,.2f}")

if __name__ == "__main__":
    run_backtest()