import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import time

# --- LAPLACE FÜZYON MOTORU ---
# Görev: Teknik ve Haber verilerini birleştirerek eğitilebilir tek bir CSV oluşturmak.

DATA_DIR = "laplace_dataset"

def load_data():
    """Kazılmış Teknik ve Haber verilerini yükler."""
    try:
        tech_df = pd.read_csv(os.path.join(DATA_DIR, 'laplace_TECH_DATASET.csv'), index_col=0)
        news_df = pd.read_csv(os.path.join(DATA_DIR, 'laplace_NEWS_DATASET.csv'))
        
        # Tarih sütununu datetime formatına çevir
        tech_df.index = pd.to_datetime(tech_df.index)
        news_df['date'] = pd.to_datetime(news_df['date']).dt.date # Saat bilgisini at
        
        print(f"✅ Veriler Yüklendi. Teknik: {len(tech_df)} satır. Haber: {len(news_df)} satır.")
        return tech_df, news_df
    except FileNotFoundError:
        print("❌ HATA: Gerekli CSV dosyaları bulunamadı. Lütfen önce laplace_miner.py'yi çalıştırın.")
        exit()

def run_sentiment_analysis(news_df):
    """VADER kullanarak haber metinlerine sayısal duygu puanı verir."""
    
    # NLP motorunu başlat
    analyzer = SentimentIntensityAnalyzer()
    
    # TextBlob'a benzer şekilde VADER, metinleri tarayıp duygusal yoğunluk (bileşik/compound) puanı verir.
    # Puan -1 (en negatif) ile +1 (en pozitif) arasındadır.
    
    news_df['sentiment_score'] = news_df['text'].apply(
        lambda x: analyzer.polarity_scores(x)['compound']
    )
    
    print("✅ Duygu Analizi Tamamlandı.")
    
    # Aynı gün ve aynı hisse için birden fazla haber varsa, ortalama duygu puanını al.
    
    # Ticker'ı bulmak için basit bir regex kullanıyoruz (Bu kısım ileride gelişebilir)
    # Şimdilik haber metinlerinin içinde hisse isimlerini aramayacağız. Sadece genel piyasa haberlerini baz alacağız.
    
    # Günlük Ortalama Duygu Puanını Hesapla (Tüm piyasa için genel duyarlılık)
    daily_sentiment = news_df.groupby('date')['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['Date', 'Market_Sentiment']
    
    return daily_sentiment

def merge_and_save(tech_df, daily_sentiment):
    """Teknik ve Duygu verilerini birleştirip kaydeder."""
    
    # Teknik veri indeksini Duygu verisinin tarih formatına eşitle
    tech_df.index.name = 'Date'
    
    # Birleştirme (Merging): Teknik veriye Duygu puanını ekle
    final_df = pd.merge(tech_df, daily_sentiment, on='Date', how='left')
    
    # Duygu puanı olmayan günleri (haber çekilemeyen günler) Nötr (0) olarak doldur.
    final_df['Market_Sentiment'].fillna(0, inplace=True)
    
    # NaN satırları düşür ve Target sütunu olmayanları temizle
    final_df.dropna(subset=['Target'], inplace=True) 
    
    # Final dosyayı kaydet
    FINAL_FILE = 'laplace_FINAL_TRAINING_SET.csv'
    final_df.to_csv(os.path.join(DATA_DIR, FINAL_FILE))
    
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
