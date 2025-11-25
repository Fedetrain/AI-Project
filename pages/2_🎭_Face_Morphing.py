# pages/2_🎭_Face_Morphing.py
import streamlit as st
from PIL import Image
# Importiamo la nuova funzione per il morphing a catena
from src.face_morphing_logic import generate_chained_morph_gif 

st.set_page_config(page_title="Face Morphing", layout="wide")
st.title("🎭 Demo: Face Morphing (Multi-Image)")
st.markdown("Carica **due o più** immagini (con un volto ciascuna) per generare un morphing a catena tra di loro.")
st.markdown("*([Ispirato da: `face morphing.zip`])*")

# Avviso per il file mancante
st.info("""
**Nota:** Questa demo richiede il file `shape_predictor_68_face_landmarks.dat` di Dlib. 
Mi sono assicurato di includerlo nel path `models/cv/` del repository per il deployment.
""")

# --- Modifica: Singolo Uploader per Multipli File ---
uploaded_files = st.file_uploader(
    "Carica Immagini (Minimo 2 richieste)", 
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True # CHIAVE: Accetta file multipli
)

image_list = []
if uploaded_files:
    st.subheader("Immagini Caricate:")
    # Mostra le anteprime in colonne (massimo 5 colonne)
    cols = st.columns(min(len(uploaded_files), 5)) 
    
    for i, file in enumerate(uploaded_files):
        try:
            img = Image.open(file)
            image_list.append(img)
            # Usa un indice modulare per distribuire le immagini tra le colonne disponibili
            with cols[i % len(cols)]: 
                st.image(img, caption=f"Immagine {i+1}", use_column_width=True)
        except Exception:
            st.warning(f"Impossibile caricare l'immagine {file.name}.")

st.markdown("---")
numero_step = st.slider("Numero di fotogrammi intermedi per ogni transizione", min_value=10, max_value=60, value=30)

# --- Modifica: Verifica minimo 2 immagini e chiama la nuova logica ---
if len(image_list) >= 2:
    if st.button("Genera Morphing a Catena 🚀", use_container_width=True):
        with st.spinner("Calcolo in corso... (potrebbe richiedere diversi minuti a seconda del numero di immagini)"):
            try:
                # Chiamiamo la nuova logica passando la lista di oggetti PIL aperti
                gif_bytes = generate_chained_morph_gif(image_list, numero_step=numero_step)
                
                st.subheader("Risultato del Morphing a Catena")
                st.image(gif_bytes, caption="Morphing GIF")
                
                st.download_button(
                    label="Scarica GIF",
                    data=gif_bytes,
                    file_name="morphing_catena.gif",
                    mime="image/gif"
                )
                
            except Exception as e:
                st.error(f"Morphing fallito: {e}")
                st.error("Assicurati che in tutte le immagini caricate ci sia un volto chiaro e frontale. Un fallimento può interrompere l'intera catena.")
else:
    if uploaded_files:
        st.warning("Carica almeno due immagini per iniziare il morphing a catena.")