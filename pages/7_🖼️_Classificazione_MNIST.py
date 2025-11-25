import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from src.mnist_logic import (
    load_mnist_data, train_random_forest_grid, build_mnist_cnn, get_data_augmentation_layer
)

st.set_page_config(page_title="MNIST Classifier", layout="wide")

st.title("🖼️ Classificazione Immagini (MNIST)")
st.markdown("""
Questo progetto affronta la classificazione multiclasse di cifre scritte a mano (0-9).
**Tecniche Dimostrate:**
* **Random Forest** con ottimizzazione iperparametri (GridSearch).
* **Deep Learning (CNN)** con Keras: Convoluzioni, Pooling, Dropout.
* **Data Augmentation**: Rotazioni e traslazioni randomiche.
* **Training Avanzato**: Learning Rate Decay, Early Stopping, Model Checkpoint.
""")

# --- 1. Caricamento Dati ---
st.header("1. Dataset e Augmentation")

if 'mnist_data' not in st.session_state:
    with st.spinner("Caricamento MNIST..."):
        (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_tr_c, X_val_c, X_te_c) = load_mnist_data()
        st.session_state['mnist_data'] = {
            'raw': (X_train, y_train, X_test, y_test),
            'cnn': (X_tr_c, X_val_c, X_te_c, y_train, y_val, y_test)
        }
        st.success("Dataset Caricato!")

data = st.session_state['mnist_data']
X_train_raw, y_train_raw = data['raw'][0], data['raw'][1]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dati Originali")
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_train_raw[i], cmap='gray')
        ax.axis('off')
        ax.set_title(f"Label: {y_train_raw[i]}")
    st.pyplot(fig)

with col2:
    st.subheader("Data Augmentation (Preview)")
    aug_layer = get_data_augmentation_layer()
    # Prendi una immagine, aggiungi dimensione batch, applica aug
    sample_img = np.expand_dims(data['cnn'][0][0], 0) 
    
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        augmented_img = aug_layer(sample_img)
        ax.imshow(augmented_img[0].numpy().astype("float32"), cmap='gray')
        ax.axis('off')
    st.pyplot(fig)
    st.caption("Variazioni randomiche (shift, rotazione) della prima immagine.")

# --- 2. Random Forest ---
st.header("2. Machine Learning Classico: Random Forest")
st.markdown("GridSearchCV per trovare i migliori parametri.")

if st.button("Avvia Training Random Forest"):
    with st.spinner("Addestramento RF in corso (su subset per velocità)..."):
        gs_rf, scaler = train_random_forest_grid(X_train_raw, y_train_raw)
        st.session_state['rf_model'] = gs_rf
        
        st.success(f"Miglior Accuracy (CV): {gs_rf.best_score_:.4f}")
        st.write("Migliori Parametri:", gs_rf.best_params_)
        
        # Test rapido
        X_test_flat = data['raw'][2].reshape(len(data['raw'][2]), -1)
        X_test_scaled = scaler.transform(X_test_flat)
        acc_test = gs_rf.score(X_test_scaled, data['raw'][3])
        st.info(f"Test Set Accuracy: {acc_test:.4f}")

# --- 3. CNN Keras ---
st.header("3. Deep Learning: CNN")
st.markdown("Architettura: Conv2D -> MaxPooling -> GlobalAveragePooling -> Dense.")

epochs = st.slider("Numero di Epoche", 1, 20, 5) # Ridotto per demo live

if st.button("Avvia Training CNN"):
    X_tr_c, X_val_c, X_te_c, y_tr, y_va, y_te = data['cnn']
    
    # Applica augmentation al training set (on the fly sarebbe meglio, qui statico per semplicità demo o usando tf.data)
    # Per performance in streamlit usiamo i dati raw normalizzati, l'augmentation è nel modello se definito come layer
    # Nota: Nel codice build_cnn non ho messo l'augmentation layer dentro per tenerlo flessibile, 
    # ma nel notebook era applicata. Qui passiamo i dati puliti per velocità di esecuzione live.
    
    model = build_mnist_cnn(input_shape=(28, 28, 1))
    
    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    with st.spinner(f"Training CNN per {epochs} epoche..."):
        history = model.fit(
            X_tr_c, y_tr,
            validation_data=(X_val_c, y_va),
            epochs=epochs,
            batch_size=64, # Batch più grande per velocità
            callbacks=[early_stop],
            verbose=0
        )
        
        st.success("Training Completato!")
        
        # Plot History
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax[0].plot(history.history['loss'], label='Train Loss')
        ax[0].plot(history.history['val_loss'], label='Val Loss')
        ax[0].set_title("Loss Curve")
        ax[0].legend()
        
        # Accuracy
        ax[1].plot(history.history['accuracy'], label='Train Acc')
        ax[1].plot(history.history['val_accuracy'], label='Val Acc')
        ax[1].set_title("Accuracy Curve")
        ax[1].legend()
        
        st.pyplot(fig)
        
        # Final Eval
        loss, acc = model.evaluate(X_te_c, y_te, verbose=0)
        st.metric("Test Accuracy Finale", f"{acc:.4f}")