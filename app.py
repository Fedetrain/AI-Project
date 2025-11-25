# app.py
import streamlit as st

st.set_page_config(
    page_title="Portfolio AI - Gem Progetti",
    page_icon="👨‍💻",
    layout="wide"
)

st.title("Benvenuto nel Mio Portfolio di AI Engineering")
st.subheader("Naviga tra le pagine per vedere le demo dei miei progetti.")

st.markdown("""
Questa è una raccolta di progetti che dimostrano le mie competenze in:
- **Deep Learning & MLOps:** Deployment di modelli CNN (TensorFlow/Keras).
- **Machine Learning Classico:** Classificatori basati su feature (Scikit-learn, LBP).
- **Computer Vision:** Manipolazione di immagini, feature detection e morphing (OpenCV, Dlib).
- **Sviluppo Algoritmico:** Solutori basati su ricerca (es. DFS per Sudoku, 9 Regine).
- **Full-Stack & Backend:** Creazione di interfacce interattive (Streamlit) e logica backend (Python).

**Piattaforma:** Questa demo è deployata su **HuggingFace Spaces** e utilizza **Streamlit** per l'interfaccia, con un backend Python modulare.

Usa il menu a sinistra per selezionare un progetto!
""")

st.sidebar.success("Seleziona una demo qui sopra.")