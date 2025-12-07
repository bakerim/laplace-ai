import os.path
import pandas as pd
import io
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# --- AYARLAR ---
SERVICE_ACCOUNT_FILE = 'laplace-secret.json'
# Senin loglardan aldığımız Klasör ID'si
FOLDER_ID = '1I3f90ThXY8HAuH4HWc9Zgp2-fxjjSA3g' 
SCOPES = ['https://www.googleapis.com/auth/drive']

def get_drive_service():
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        print(f"❌ HATA: '{SERVICE_ACCOUNT_FILE}' bulunamadı.")
        return None
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def load_data_from_drive():
    service = get_drive_service()
    if not service: return None

    print(f"📂 Drive Klasörü (ID: {FOLDER_ID}) taranıyor...")
    
    # Klasörün İÇİNDEKİ csv dosyalarını ara
    query = f"'{FOLDER_ID}' in parents and name contains '.csv' and trashed = false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])

    if not items:
        print("⚠️ Klasörde .csv dosyası bulunamadı!")
        print("Lütfen CSV dosyasını Drive'daki 'Laplace_Data' klasörünün İÇİNE taşıyın.")
        return None

    file_to_download = items[0]
    file_id = file_to_download['id']
    file_name = file_to_download['name']
    
    print(f"⬇️ İndiriliyor: {file_name}...")
    
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    
    done = False
    while done is False:
        status, done = downloader.next_chunk()

    fh.seek(0)
    try:
        df = pd.read_csv(fh)
        print(f"✅ BAŞARILI! {len(df)} satır veri yüklendi.")
        
        # Sütun temizliği
        df.columns = [c.strip().title() for c in df.columns]
        
        # Tarih sütunu ayarı (Genelde Date olur)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
        return df
    except Exception as e:
        print(f"❌ CSV Okuma Hatası: {e}")
        return None
