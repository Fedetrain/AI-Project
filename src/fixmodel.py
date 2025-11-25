import tensorflow as tf
import os
import sys

# --- Importazione Modulo di Inferenza ---
# DEVI ASSICURARTI che il percorso a src/ml_inference.py sia corretto.
# Questo è NECESSARIO affinché la classe SelfAttention corretta sia in memoria.

# Se questo script è nella root del progetto:
sys.path.append(os.path.join(os.getcwd(), 'src'))
# Importa il modulo che contiene la classe SelfAttention aggiornata
import ml_inference 

# --- Definizione del Modello e Percorsi ---
MODEL_PATH = "my_ai_portfolio/models/cnn/modello_cnn_attention.keras"

# Dobbiamo dire ESPLICITAMENTE a Keras dove trovare la classe SelfAttention 
# al momento del caricamento, poiché il file salvato è vecchio e non la include.
custom_objects = {
    # La chiave deve corrispondere al nome della classe nell'errore: 'SelfAttention'
    "SelfAttention": ml_inference.SelfAttention
}

print(f"Tentativo di caricamento del modello problematico: {MODEL_PATH}")

try:
    # 1. Carica il modello forzando il riconoscimento della classe custom
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects=custom_objects, 
        compile=True # Mantieni la configurazione di compilazione
    )
    print("Modello caricato con successo utilizzando custom_objects.")

    # 2. Risalva il modello
    # Keras ora salverà il modello con la corretta serializzazione (@register_keras_serializable)
    # presente nella definizione della classe caricata da ml_inference.py.
    model.save(MODEL_PATH)
    print("Modello risalvato con successo. Il file è ora corretto.")

    # 3. Verifica finale (Opzionale)
    # Prova a ricaricare SENZA custom_objects per confermare la correzione
    test_model = tf.keras.models.load_model(MODEL_PATH)
    print("Verifica: Il modello ricarica correttamente senza custom_objects.")

except Exception as e:
    print(f"\nERRORE CRITICO: La riparazione è fallita. Errore: {e}")
    print("Assicurati che la classe SelfAttention in src/ml_inference.py sia definita esattamente come segue:")
    print("  - Decoratore: @tf.keras.saving.register_keras_serializable() o @keras.saving.register_keras_serializable()")
    print("  - Metodo: def get_config(self):...")