# src/ml_inference.py
import streamlit as st
import tensorflow as tf
import pickle
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# --- Gestione Modelli ---

@st.cache_resource
def load_keras_model(path):
    """Carica un modello Keras."""
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Errore caricamento modello Keras ({path}): {e}")
        return None

@st.cache_resource
def load_pickle_model(path):
    """Carica un modello pickle (es. scikit-learn)."""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Errore caricamento modello Pickle ({path}): {e}")
        return None

# --- Logica Progetto 1: Fake vs Real (LBP) ---
# Ispirato da: fake_real LBP.zip/ClassificaImg.py

@st.cache_resource
def load_face_cascade():
    """Carica il classificatore Haar per il rilevamento volti."""
    # Nota: il path è relativo alla root del progetto
    cascade_path = 'models/cv/haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier()
    if not face_cascade.load(cv2.samples.findFile(cascade_path)):
        st.error("Errore critico: Impossibile caricare haarcascade.")
        return None
    return face_cascade

def rileva_e_ritaglia_volto(img_array, fare_resize=True):
    #
    face_cascade = load_face_cascade()
    if face_cascade is None:
        return None
        
    img_grigia = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    facce = face_cascade.detectMultiScale(img_grigia, scaleFactor=1.1, minNeighbors=5)

    if len(facce) > 1:
        facce = [max(facce, key=lambda rect: rect[2] * rect[3])]


    x, y, w, h = facce[0]
    volto = img_array[y:y + h, x:x + w]
    
    if fare_resize:
        volto = cv2.resize(volto, (150, 150))
    return volto

def estrai_caratteristiche_lbp(img_grigia):
    #
    RAGGIO = 1
    PUNTI_P8 = 256
    METODO_P8 = 'uniform'
    
    lbp = local_binary_pattern(img_grigia, PUNTI_P8, RAGGIO, METODO_P8)
    hist, _ = np.histogram(lbp, bins=np.arange(0, PUNTI_P8 + 3), range=(0, PUNTI_P8 + 2), density=True)
    return hist

def predict_lbp(image_pil):
    """Esegue la predizione LBP su un'immagine PIL."""
    model = load_pickle_model("models/lbp/miglior_modello.pkl") #
    scaler = load_pickle_model("models/lbp/scaler.pkl") #
    
    if model is None or scaler is None:
        return "Errore: Modello LBP o Scaler non caricati."

    # Converti PIL Image in array OpenCV
    img_array = np.array(image_pil.convert('RGB'))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    volto_ritagliato = rileva_e_ritaglia_volto(img_array, fare_resize=True)
    
    if volto_ritagliato is None:
        return "Nessun volto singolo rilevato."

    img_grigia = cv2.cvtColor(volto_ritagliato, cv2.COLOR_BGR2GRAY)
    hist = estrai_caratteristiche_lbp(img_grigia)
    
    features = hist.reshape(1, -1)
    features = scaler.transform(features)
    pred = model.predict(features)
    
    
    return "FAKE (Generata da AI)" if pred[0] == 1 else "REALE"


# --- Logica Progetto 2: Fake vs Real (CNN) ---
# Ispirato da: fake_real CNN.zip/assignment3.py

def preprocess_cnn_image(image_pil, img_size=(224, 224)):
    """Pre-processa un'immagine PIL per il modello CNN."""
    #
    img = image_pil.convert('RGB').resize(img_size)
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Aggiunge dimensione batch
    return img_array

def predict_cnn(image_pil):
    """Esegue la predizione CNN su un'immagine PIL."""
    model = load_keras_model("models/cnn/modello_cnn.keras") #
    if model is None:
        return "Errore: Modello CNN non caricato."

    processed_img = preprocess_cnn_image(image_pil)
    
    prediction = model.predict(processed_img)[0][0]

    print(prediction)
    label = 1 if prediction < 0.5 else 0
    class_label = "Reale" if label == 1 else "Generata da IA"
    
    return f"{class_label} )"

