import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from datetime import datetime

# --- LAPLACE FÜZYON MOTORU V1.1 (ZAMAN UYUMLU) ---
# Görev: Teknik verileri, NLP duygu puanlarıyla birleştirerek eğitilebilir tek bir CSV oluşturmak.

DATA_DIR = "laplace_dataset"

def load_data():
    """Kazılmış Teknik ve Haber verilerini yükler."""
    try:
        # Teknik veriyi yükle
        tech_df = pd.read_csv(os.path.join(DATA_DIR, 'laplace_TECH_DATASET.csv'), index_col=0)
        # Haber verisini yükle
        news_df = pd.read_csv(os.path.join(DATA_DIR, 'laplace_NEWS_DATASET.csv'))
        
        # Tarih sütunlarını datetime formatına çevirirken esnek ol.
        # FIX: ValueError'ı çözmek için format='mixed' ve hataları yoksay ('coerce') kullanılır.
        tech_df.index = pd.to_datetime(tech_df.index)
        news_df['date'] = pd.to_datetime(news_df['date'], format='mixed', errors='coerce').dt.date 
        
        # Tarih hatalarından (NaN) kurtul
        news_df.dropna(subset=['date'], inplace=True)

        print(f"✅ Veriler Yüklendi. Teknik: {len(tech_df)} satır. Haber: {len(news_df)} satır.")
        return tech_df, news_df
    except FileNotFoundError:
        print("❌ HATA: Gerekli CSV dosyaları bulunamadı. Lütfen önce laplace_miner.py'yi çalıştırın.")
        exit()

def run_sentiment_analysis(news_df):
    """VADER kullanarak haber metinlerine sayısal duygu puanı verir."""
    
    # NLP motorunu başlat
    analyzer = SentimentIntensityAnalyzer()
    
    # Duygu puanını hesapla (-1.0 ile +1.0 arası)
    news_df['sentiment_score'] = news_df['text'].apply(
        lambda x: analyzer.polarity_scores(str(x))['compound']
    )
    
    print("✅ Duygu Analizi Tamamlandı.")
    
    # Aynı güne ait tüm haberlerin ortalama duygu puanını hesapla (Genel piyasa duyarlılığı)
    daily_sentiment = news_df.groupby('date')['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['Date', 'Market_Sentiment']
    
    return daily_sentiment

def merge_and_save(tech_df, daily_sentiment):
    """Teknik ve Duygu verilerini birleştirip kaydeder."""
    
    # Teknik veri indeksini Duygu verisinin tarih formatına eşitle (YYYY-MM-DD)
    tech_df.index.name = 'Date'
    
    # Birleştirme (Merging): Teknik veriye Duygu puanını ekle
    # Not: pd.to_datetime(tech_df.index).dt.date yapınca Date index'ini kaybettiği için yeniden to_datetime yapmak gerekebilir
    
    # Teknik veri index'ini (tarih) basit date formatına çevir
    tech_df.reset_index(inplace=True)
    tech_df['Date'] = pd.to_datetime(tech_df['Date']).dt.date

    # Birleştirme: Sadece tarih sütununa göre yap
    final_df = pd.merge(tech_df, daily_sentiment, on='Date', how='left')
    
    # Duygu puanı olmayan günleri (haber çekilemeyen günler) Nötr (0) olarak doldur.
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