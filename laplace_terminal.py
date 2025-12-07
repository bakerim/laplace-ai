import streamlit as st
import plotly.graph_objects as go
from laplace_engine import get_data, load_ai_assets, get_lstm_prediction, LIBRARIES_LOADED

# -----------------------------------------------------------------------------
# ARAYÜZ VE AKIŞ
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Laplace Terminal v2.2",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("📐 LAPLACE v2.2 (GÜVENLİ ÇALIŞMA)")
    st.markdown("---")

    # Yan Panel
    with st.sidebar:
        st.header("Varlık Seçimi")
        ticker = st.text_input("Sembol (Örn: AAPL, THYAO.IS)", value="AAPL").upper()
        
        st.markdown("### Sistem Durumu")
        
        lstm_model, global_scaler = load_ai_assets()
        
        if LIBRARIES_LOADED:
            st.success("🧠 Kütüphaneler: AKTİF")
            if lstm_model:
                st.info("✅ Model Dosyası Bulundu")
            else:
                st.warning("⚠️ Model Dosyası Yok")
        else:
             st.error("❌ TensorFlow/Joblib Eksik")

        if st.button("HESAPLA ⚡", type="primary"):
            run_analysis = True
        else:
            run_analysis = False

    # Ana Ekran
    if run_analysis:
        with st.spinner(f"{ticker} verileri analiz ediliyor..."):
            
            df = get_data(ticker)
            
            if df is not None:
                current_price = df['Close'].iloc[-1]
                
                # Grafik
                fig = go.Figure(data=[go.Candlestick(x=df.index,
                                open=df['Open'], high=df['High'],
                                low=df['Low'], close=df['Close'])])
                fig.update_layout(title=f"{ticker} Fiyat Grafiği", height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Sonuçlar
                col1, col2, col3 = st.columns(3)
                col1.metric("Son Fiyat", f"{current_price:.2f}")
                
                prediction, error = get_lstm_prediction(df, lstm_model, global_scaler)
                    
                if error:
                    st.warning(f"Tahmin Notu: {error}")
                else:
                    col2.metric("Laplace Tahmini", f"{prediction:.2f}")
                    diff_percent = ((prediction - current_price) / current_price) * 100
                    col3.metric("Değişim Beklentisi", f"%{diff_percent:.2f}")
                    st.info(f"Yapay zeka tahmini: **{prediction:.2f}**")
            else:
                st.error("Veri çekilemedi.")
    else:
        st.info("İşlem yapmak için soldaki butonu kullanın.")

if __name__ == "__main__":
    main()
