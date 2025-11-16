# app.py
import streamlit as st
from PIL import Image
from src.inference import load_model, predict_cnn # Importiamo la logica

# Configurazione della pagina Streamlit
st.set_page_config(
    page_title="AI Portfolio: Demo Interattive",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("👨‍💻 Portfolio di AI Engineering & MLOps")
st.markdown("---")

# 1. Caricamento dei Modelli all'avvio
try:
    cnn_model = load_model("models/modello_cnn.keras")
    st.sidebar.success("Modello CNN Caricato con Successo!")
except Exception as e:
    st.sidebar.error(f"Errore nel caricamento del modello CNN: {e}")
    cnn_model = None

# --- SIDEBAR: Navigazione tra i Progetti ---
project = st.sidebar.selectbox(
    "Seleziona un Progetto da Mostrare:",
    ["Classificazione Real vs Fake (CNN)", "Face Morphing Demo (WIP)", "Solutore DFS (8 Regine/Sudoku)"]
)

st.sidebar.markdown("---")
st.sidebar.header("Il Mio Stack Tecnologico")
st.sidebar.write("- **Backend:** Python, FastAPI, Flask")
st.sidebar.write("- **MLOps:** Docker, Kubernetes (concetti), GitHub, HuggingFace Spaces")
st.sidebar.write("- **Frameworks:** TensorFlow, Keras, Scikit-learn, OpenCV")


# --- CORPO PRINCIPALE: Contenuto del Progetto Selezionato ---

if project == "Classificazione Real vs Fake (CNN)":
    st.header("1. Classificazione di Volti Reali vs. Generati dall'AI (CNN)")
    st.markdown("Mostra la capacità di **Deployment di Modelli di Deep Learning (MLOps)**.")
    
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Carica un'immagine di volto...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Immagine Caricata', use_column_width=True)
            
            if st.button("Esegui Classificazione"):
                if cnn_model:
                    with st.spinner('Esecuzione inferenza...'):
                        label, confidence = predict_cnn(image, cnn_model)
                    
                    st.subheader("Risultato dell'Inferenza:")
                    if label == "Real Face":
                        st.success(f"**Volto Reale** con Confidenza: {confidence:.2f}")
                    else:
                        st.warning(f"**Volto AI-Generated** con Confidenza: {confidence:.2f}")
                else:
                    st.error("Modello non disponibile. Controlla il caricamento in sidebar.")

    with col2:
        st.subheader("Dettagli del Progetto")
        st.markdown("""
        Questo progetto dimostra l'uso di una **Convolutional Neural Network (CNN)** per distinguere tra immagini di volti reali e immagini generate da GAN o Diffusion Models.
        - **Tecnologie:** TensorFlow/Keras, OpenCV, Streamlit.
        - **MLOps/Deployment:** Modello serializzato (.keras) e servito tramite una piattaforma di MLOps leggera (HuggingFace Spaces).
        """)
        st.link_button("Vai al Codice su GitHub", "URL_DEL_TUO_REPO_QUI")

# Aggiungi qui la logica per gli altri progetti (Face Morphing, DFS, ecc.)
# ...

elif project == "Face Morphing Demo (WIP)":
    st.header("2. Face Morphing con OpenCV (In Lavorazione)")
    st.info("Questa sezione è in lavorazione. Qui verrà mostrata la capacità di elaborare immagini e manipolare punti chiave (Keypoints) con OpenCV.")
    
# ...