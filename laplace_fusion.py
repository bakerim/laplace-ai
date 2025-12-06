import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from datetime import datetime

# --- LAPLACE FÜZYON MOTORU V1.2 (ZAMAN GARANTİLİ) ---
# Görev: Teknik verileri, NLP duygu puanlarıyla birleştirerek eğitilebilir tek bir CSV oluşturmak.

DATA_DIR = "laplace_dataset"

# Yalnızca bu fonksiyonu değiştir (Line 30 ve civarı)
def load_data():
    """Kazılmış Teknik ve Haber verilerini yükler."""
    try:
        # ... (Önceki kod)
        
        # Teknik veri indeksini temizle
        tech_df.index = pd.to_datetime(tech_df.index)
        
        # --- FIX: DATETIMEINDEX UYUMSUZLUĞU ÇÖZÜMÜ ---
        # 1. Zaten DatetimeIndex olduğu için tekrar pd.to_datetime kullanmıyoruz.
        # 2. Sadece indeksteki date bilgisini alıyoruz.
        tech_df.index = tech_df.index.date 
        
        # --- FIX BİTTİ ---

        print(f"✅ Veriler Yüklendi. Teknik: {len(tech_df)} satır. Haber: {len(news_df)} satır.")
        return tech_df, news_df
    except FileNotFoundError:
        print("❌ HATA: Gerekli CSV dosyaları bulunamadı. Lütfen önce laplace_miner.py'yi çalıştırın.")
        exit()

def run_sentiment_analysis(news_df):
    """VADER kullanarak haber metinlerine sayısal duygu puanı verir."""
    
    analyzer = SentimentIntensityAnalyzer()
    
    news_df['sentiment_score'] = news_df['text'].apply(
        lambda x: analyzer.polarity_scores(str(x))['compound']
    )
    
    print("✅ Duygu Analizi Tamamlandı.")
    
    # Günlük Ortalama Duygu Puanını Hesapla
    daily_sentiment = news_df.groupby('date')['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['Date', 'Market_Sentiment']
    
    return daily_sentiment

def merge_and_save(tech_df, daily_sentiment):
    """Teknik ve Duygu verilerini birleştirip kaydeder."""
    
    # Birleştirme için index'i ve sütunu eşitle
    tech_df.rename(columns={'Date': 'Date_Index'}, inplace=True) # İndex adını koru
    
    # Birleştirme (Merging): Basit tarih sütununa göre yap
    final_df = pd.merge(tech_df, daily_sentiment, left_on='Date', right_on='Date', how='left')
    
    # Duygu puanı olmayan günleri Nötr (0) olarak doldur.
    final_df['Market_Sentiment'].fillna(0, inplace=True)
    
    # NaN satırları düşür ve Target sütunu olmayanları temizle
    final_df.dropna(subset=['Target'], inplace=True) 
    
    # Final dosyayı kaydet
    FINAL_FILE = 'laplace_FINAL_TRAINING_SET.csv'
    final_df.to_csv(os.path.join(DATA_DIR, FINAL_FILE), index=False)
    
    print(f"✅ Veri Birleştirme (Füzyon) Tamamlandı.")
    print(f"💾 Nihai Eğitim Seti Kaydedildi: {FINAL_FILE} ({len(final_df)} satır)")
    print(f"📊 Model, {len(final_df.columns)} farklı özelliğe bakarak eğitim alacak.")


if __name__ == "__main__":
    print("📐 LAPLACE FÜZYON PROTOKOLÜ BAŞLATILDI.")
    
    # 1. Verileri Yükle
    tech_data, news_data = load_data()
    
    # 2. Duygu Analizini Çalıştır
    daily_sentiment_scores = run_sentiment_analysis(news_data)
    
    # 3. Birleştir ve Kaydet
    merge_and_save(tech_data, daily_sentiment_scores)
    
    print("\n" + "="*50)
    print("🏁 YAPAY ZEKA EĞİTİMİ İÇİN VERİ HAZIRDIR.")
    print("="*50)

