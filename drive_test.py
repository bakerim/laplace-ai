import os.path
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# --- AYARLAR ---
# İndirdiğin ve ismini değiştirdiğin anahtar dosyası
SERVICE_ACCOUNT_FILE = 'laplace-secret.json'
# Google Drive Yetki Alanı
SCOPES = ['https://www.googleapis.com/auth/drive']

def test_drive_connection():
    print("📡 Google Drive bağlantısı deneniyor...")

    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        print(f"❌ HATA: '{SERVICE_ACCOUNT_FILE}' dosyası bulunamadı!")
        print("Lütfen JSON anahtar dosyasını proje klasörüne yüklediğinden emin ol.")
        return

    try:
        # 1. Kimlik Doğrulama
        creds = Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        
        # 2. Drive Servisini Başlat
        service = build('drive', 'v3', credentials=creds)

        # 3. Dosyaları Listele (Paylaşılanları ara)
        print("📂 Drive dosyaları taranıyor...")
        results = service.files().list(
            pageSize=10, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            print("⚠️ Klasör boş veya 'laplace-bot' ile henüz bir şey paylaşmadın.")
        else:
            print("✅ BAŞARILI! Drive'a Erişildi. Bulunan Dosyalar:")
            for item in items:
                print(f"   📄 {item['name']} (ID: {item['id']})")

    except Exception as e:
        print(f"❌ BAĞLANTI HATASI: {e}")

if __name__ == '__main__':
    test_drive_connection()
