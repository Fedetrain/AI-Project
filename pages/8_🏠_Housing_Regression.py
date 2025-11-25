import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from src.housing_logic import (
    load_housing_data, preprocess_housing, run_lasso_selection, build_housing_nn
)
import keras
st.set_page_config(page_title="Housing Regression", layout="wide")

st.title("🏠 California Housing Regression")
st.markdown("""
Previsione dei prezzi delle case in California.
**Highlights:**
* **Data Cleaning**: Imputazione valori mancanti basata su correlazioni (stanze/letto).
* **Feature Engineering**: Creazione nuove feature (es. `bedrooms_per_household`) e One-Hot Encoding.
* **Feature Selection**: Uso di **Lasso (L1)** per identificare le feature più rilevanti.
* **Modello Ibrido**: Uso di una **Conv1D Neural Network** su dati tabulari.
""")

# --- 1. Caricamento Dati ---
st.header("1. Data Loading & Preprocessing")

@st.cache_data
def get_data():
    return load_housing_data()

df_raw = get_data()

if df_raw is not None:
    st.write("Dataset Originale (Prime 5 righe):")
    st.dataframe(df_raw.head())
    
    # Preprocessing
    df_clean = preprocess_housing(df_raw)
    st.write("Dataset Processato (One-Hot Encoded + Nuove Features):")
    st.dataframe(df_clean.head())
    
    # Correlation Heatmap
    with st.expander("Vedi Matrice di Correlazione"):
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df_clean.corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Split
    X = df_clean.drop(columns=['median_house_value'])
    y = df_clean['median_house_value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling (Importante per NN e Lasso)
    # Nota: nel notebook si usava la standardizzazione.
    # Qui standardizziamo solo per il training dei modelli, mantenendo i dataframe per visualizzazione
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- 2. Feature Selection (Lasso) ---
    st.header("2. Feature Selection con Lasso")
    
    if st.checkbox("Esegui Lasso Analysis"):

        feats=X.columns
        coef = run_lasso_selection(X_train_s, y_train)
        
        # Visualizzazione
        coef_df = pd.DataFrame({'Feature': feats, 'Coefficient': np.abs(coef)})
        coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
        
        st.bar_chart(coef_df.set_index('Feature'))
        
        # Filtro feature (Threshold demo)
        threshold = 100 # Valore arbitrario per demo visiva, o basato su notebook
        selected_feats = coef_df[coef_df['Coefficient'] > threshold]['Feature'].values
        st.write(f"Feature Selezionate (Top importance): {list(selected_feats)}")
        
        # Update X per NN
        # Per semplicità nella demo usiamo tutte le feature o le selezionate.
        # Usiamo quelle selezionate se l'utente vuole, altrimenti tutte.
        use_sel = st.toggle("Usa solo feature selezionate per il training della Rete Neurale?")
        if use_sel:
            # Indici delle feature
            indices = [list(feats).index(f) for f in selected_feats]
            X_train_nn = X_train_s[:, indices]
            X_test_nn = X_test_s[:, indices]
        else:
            X_train_nn = X_train_s
            X_test_nn = X_test_s
    else:
        X_train_nn = X_train_s
        X_test_nn = X_test_s

    # --- 3. Neural Network ---
    st.header("3. Neural Network (Conv1D)")
    st.markdown("L'input viene trasformato in 3D (Batch, Steps, Channels) per essere processato da Conv1D.")
    
    # Reshape per Conv1D: (samples, features, 1)
    X_train_cnn = np.expand_dims(X_train_nn, axis=2)
    X_test_cnn = np.expand_dims(X_test_nn, axis=2)
    
    input_shape = (X_train_cnn.shape[1], 1)
    
    if st.button("Train Neural Network"):
        model = build_housing_nn(input_shape)
        
        # Callback
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        with st.spinner("Addestramento in corso..."):
            history = model.fit(
                X_train_cnn, y_train,
                validation_split=0.2,
                epochs=30, # Epoche ridotte per demo
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
            
            st.success("Training Completato!")
            
            # Metrics
            y_pred = model.predict(X_test_cnn)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            c1, c2 = st.columns(2)
            c1.metric("R2 Score", f"{r2:.4f}")
            c2.metric("RMSE", f"{rmse:.2f}")
            
            # Plots
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(history.history['loss'], label='Train Loss')
            ax[0].plot(history.history['val_loss'], label='Val Loss')
            ax[0].set_title("Loss (MSE)")
            ax[0].legend()
            
            ax[1].scatter(y_test, y_pred, alpha=0.3, s=5)
            ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax[1].set_xlabel("True Values")
            ax[1].set_ylabel("Predictions")
            ax[1].set_title("True vs Predicted")
            
            st.pyplot(fig)

else:
    st.error("Impossibile caricare il dataset da GitHub. Controlla la connessione.")