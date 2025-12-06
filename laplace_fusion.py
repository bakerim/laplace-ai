import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from datetime import datetime

# --- FIX: NLTK ARAMA YOLU GARANTİSİ ---
# Bu blok, 'vader_lexicon' dosyasını mevcut dizine indirerek LookupError hatasını çözer.
try:
    # Analyzer'ı başlatmayı dene
    SentimentIntensityAnalyzer()
except LookupError:
    # Hata varsa, NLTK'yı mevcut dizine indir.
    print("--- NLTK Veri Eksik, İndiriliyor... (Bu sadece bir kere olur) ---")
    nltk.download('vader_lexicon', quiet=True)
    print("--- NLTK Veri İndirme Tamamlandı. ---")

# --- LAPLACE FÜZYON MOTORU V1.4 ---
# Görev: Teknik verileri, NLP duygu puanlarıyla birleştirerek eğitilebilir tek bir CSV oluşturmak.

DATA_DIR = "laplace_dataset"

def load_data():
    """Kazılmış Teknik ve Haber verilerini yükler."""
    tech_df = pd.DataFrame()
    news_df = pd.DataFrame()

    try:
        tech_df = pd.read_csv(os.path.join(DATA_DIR, 'laplace_TECH_DATASET.csv'), index_col=0)
        news_df = pd.read_csv(os.path.join(DATA_DIR, 'laplace_NEWS_DATASET.csv'))
        
        # 1. Haber Tarihini Temizle
        news_df['date'] = pd.to_datetime(news_df['date'], format='mixed', errors='coerce', utc=True)
        news_df.dropna(subset=['date'], inplace=True)
        news_df['date'] = news_df['date'].dt.normalize().dt.date
        
        # 2. Teknik Veri İndeksini Temizle
        tech_df.index = pd.to_datetime(tech_df.index)
        tech_df.index = tech_df.index.date 

        print(f"✅ Veriler Yüklendi. Teknik: {len(tech_df)} satır. Haber: {len(news_df)} satır.")
        return tech_df, news_df
    
    except FileNotFoundError:
        print("❌ HATA: Gerekli CSV dosyaları bulunamadı. Lütfen önce laplace_miner.py'yi çalıştırın.")
        exit()
    except Exception as e:
        print(f"❌ KRİTİK VERİ HATASI: {e}")
        return pd.DataFrame(), pd.DataFrame()

def run_sentiment_analysis(news_df):
    """VADER kullanarak haber metinlerine sayısal duygu puanı verir."""
    
    if news_df.empty:
        print("⚠️ Duygu Analizi İçin Haber Verisi Yok.")
        return pd.DataFrame()

    # NLTK fix'i sayesinde bu satır artık çalışmalı
    analyzer = SentimentIntensityAnalyzer() 
    
    # Duygu puanını hesapla
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
    
    if tech_df.empty:
        print("⚠️ Birleştirme İçin Teknik Veri Yok.")
        return

    # Teknik veri index'ini sütuna çevirip birleştirme için hazırlar
    tech_df['Date'] = tech_df.index
    
    # Birleştirme: Basit tarih sütununa göre yap
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
