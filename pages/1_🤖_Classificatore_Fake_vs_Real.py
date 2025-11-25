# pages/1_🤖_Classificatore_Fake_vs_Real.py
import streamlit as st
from PIL import Image
from src.ml_inference import predict_lbp, predict_cnn
import io

st.set_page_config(page_title="Classificatore Fake vs Real", layout="wide")
st.title("🤖 Demo: Classificatore Fake vs Real (LBP vs CNN)")
st.markdown("Carica un'immagine per vedere la classificazione di due modelli diversi.")

uploaded_file = st.file_uploader("Scegli un'immagine di un volto...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Mostra l'immagine caricata
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Immagine Caricata", width=400)
    st.markdown("---")
    
    col1, col2 = st.columns(2)

    # --- Colonna 1: Modello LBP (Machine Learning Classico) ---
    with col1:
        st.header("Modello 1: LBP + SVM")
        st.markdown("""
        Questo modello usa **Local Binary Patterns (LBP)** per estrarre le feature della texture
        e un **Support Vector Machine (SVM)** per classificare.
        *([Ispirato da: `fake_real LBP.zip`])*
        """)
        
        if st.button("Esegui Classificazione LBP"):
            with st.spinner("Analisi LBP in corso..."):
                try:
                    risultato_lbp = predict_lbp(image)
                    if "REALE" in risultato_lbp:
                        st.success(f"**Risultato LBP:** {risultato_lbp}")
                    elif "FAKE" in risultato_lbp:
                        st.warning(f"**Risultato LBP:** {risultato_lbp}")
                    else:
                        st.error(risultato_lbp)
                except Exception as e:
                    st.error(f"Errore durante l'analisi LBP: {e}")

    # --- Colonna 2: Modello CNN (Deep Learning) ---
    with col2:
        st.header("Modello 2: CNN (Deep Learning)")
        st.markdown("""
        Questo modello usa una **Convolutional Neural Network (CNN)** addestrata
        per classificare l'immagine.
        *([Ispirato da: `fake_real CNN.zip`])*
        """)
        
        if st.button("Esegui Classificazione CNN"):
            with st.spinner("Analisi CNN in corso..."):
                try:
                    risultato_cnn = predict_cnn(image)
                    if "REALE" in risultato_cnn:
                        st.success(f"**Risultato CNN:** {risultato_cnn}")
                    else:
                        st.warning(f"**Risultato CNN:** {risultato_cnn}")
                except Exception as e:
                    st.error(f"Errore durante l'analisi CNN: {e}")