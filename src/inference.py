# src/inference.py
import streamlit as st
import tensorflow as tf
import joblib
import numpy as np

# Funzione per il caching: il modello viene caricato una sola volta
@st.cache_resource
def load_model(model_path):
    """Carica un modello Keras o un artefatto scikit-learn."""
    if model_path.endswith('.keras'):
        # Caricamento Modello CNN (es. fake_real CNN)
        model = tf.keras.models.load_model(model_path)
    elif model_path.endswith('.pkl'):
        # Caricamento Modello Scikit-learn (es. fake_real LBP)
        model = joblib.load(model_path)
    else:
        raise ValueError("Formato modello non supportato")
    return model

@st.cache_resource
def load_scaler(scaler_path):
    """Carica un oggetto StandardScaler."""
    return joblib.load(scaler_path)

def predict_cnn(image, model):
    """Esegue la predizione per il modello CNN."""
    # Simula la pre-elaborazione dell'immagine (ridimensionamento, normalizzazione)
    # Adatta questa logica al tuo modello specifico
    img_array = np.array(image.convert("RGB").resize((128, 128))) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Aggiunge dimensione batch
    
    prediction = model.predict(img_array)[0]
    return "AI-Generated" if prediction > 0.5 else "Real Face", float(prediction)

# Aggiungi qui altre funzioni di predizione (es. per il modello LBP)
# ...