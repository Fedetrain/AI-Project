import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.regression_logic import (
    generate_synthetic_data, clean_data_kde, 
    train_kernel_ridge, build_and_train_nn
)

st.set_page_config(page_title="Regressione con Outlier & NN", layout="wide")

st.title("📈 Regressione Complessa e Deep Learning")
st.markdown("""
Questo modulo affronta un problema di regressione non lineare su dati rumorosi.
**Workflow:**
1.  **Generazione Dati**: Funzione sinusoidale + rumore + outlier pesanti.
2.  **Pulizia**: Kernel Density Estimation (KDE) per filtrare gli outlier probabilistici.
3.  **Machine Learning Classico**: Kernel Ridge Regression con GridSearchCV.
4.  **Deep Learning**: Rete Neurale Densa (MLP) con Keras (BatchNormalization, Dropout, L2 Reg).
""")

# --- 1. Generazione Dati ---
st.header("1. Generazione e Analisi Dati")

if 'df_reg' not in st.session_state:
    with st.spinner("Generazione dati sintetici..."):
        st.session_state['df_reg'] = generate_synthetic_data(n_samples=2000, n_outliers=50)

df = st.session_state['df_reg']

col1, col2 = st.columns([1, 3])
with col1:
    st.write(df.head())
    if st.button("Rigenera Dati"):
        st.session_state['df_reg'] = generate_synthetic_data()
        st.experimental_rerun()

with col2:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(df['X'], df['y'], s=5, alpha=0.6, label='Dati Grezzi')
    ax.set_title("Dataset con Outliers")
    ax.legend()
    st.pyplot(fig)

# --- 2. Pulizia KDE ---
st.header("2. Pulizia con KDE (Kernel Density Estimation)")

bandwidth = st.slider("Bandwidth KDE", 0.1, 2.0, 0.5)
percentile = st.slider("Soglia Percentile (taglio outlier)", 0, 50, 20)

if st.button("Analizza e Pulisci"):
    with st.spinner("Calcolo densità..."):
        df_cleaned = clean_data_kde(df, bandwidth=bandwidth, percentile=percentile)
        st.session_state['df_cleaned'] = df_cleaned
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(df['X'], df['y'], s=5, c='red', alpha=0.3, label='Outliers Rimossi')
        ax.scatter(df_cleaned['X'], df_cleaned['y'], s=5, c='blue', alpha=0.6, label='Inliers (Clean)')
        ax.set_title("Risultato Pulizia KDE")
        ax.legend()
        st.pyplot(fig)
        st.success(f"Dati ridotti da {len(df)} a {len(df_cleaned)} campioni.")

if 'df_cleaned' in st.session_state:
    df_c = st.session_state['df_cleaned']
    X = df_c['X'].values.reshape(-1, 1)
    y = df_c['y'].values.reshape(-1, 1)

    # --- 3. Training Models ---
    st.header("3. Addestramento e Confronto Modelli")
    
    col_train1, col_train2 = st.columns(2)
    
    # Kernel Ridge
    with col_train1:
        st.subheader("Kernel Ridge")
        st.markdown("GridSearch su parametri Alpha e Gamma (RBF Kernel).")
        if st.button("Train Kernel Ridge"):
            with st.spinner("Training KR..."):
                gs_kr = train_kernel_ridge(X, y)
                st.session_state['model_kr'] = gs_kr
                st.success(f"Best R2: {gs_kr.best_score_:.4f}")
                st.write("Best Params:", gs_kr.best_params_)

    # Neural Network
    with col_train2:
        st.subheader("Neural Network (Keras)")
        st.markdown("Architettura: 5 Layers Dense, L2, BatchNorm, Dropout.")
        epochs = st.number_input("Epoche", 10, 200, 50)
        if st.button("Train NN"):
            with st.spinner("Training NN..."):
                model_nn, history = build_and_train_nn(X, y, epochs=epochs)
                st.session_state['model_nn'] = model_nn
                st.session_state['history_nn'] = history
                
                # Plot Loss
                fig, ax = plt.subplots(figsize=(6,3))
                ax.plot(history.history['loss'], label='Train Loss')
                ax.plot(history.history['val_loss'], label='Val Loss')
                ax.set_title("Curve di Apprendimento")
                ax.legend()
                st.pyplot(fig)
                
                final_r2 = history.history['r2_score'][-1] if 'r2_score' in history.history else 0
                st.success(f"Final Training R2: {final_r2:.4f}")

    # --- 4. Confronto Finale ---
    if 'model_kr' in st.session_state and 'model_nn' in st.session_state:
        st.header("4. Risultati Finali")
        
        X_plot = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
        
        pred_kr = st.session_state['model_kr'].predict(X_plot)
        pred_nn = st.session_state['model_nn'].predict(X_plot)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(df_c['X'], df_c['y'], c='grey', s=5, alpha=0.3, label='Original Clean Data')
        ax.plot(X_plot, pred_kr, c='blue', label='Kernel Ridge', linewidth=2)
        ax.plot(X_plot, pred_nn, c='green', label='Neural Network', linewidth=2, linestyle='--')
        ax.set_title("Confronto Regressori")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)