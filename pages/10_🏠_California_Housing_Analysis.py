# pages/5_🏠_California_Housing_Analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys

# Aggiungi la cartella src al path per importare il modulo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from src.california_housing_analysis import run_california_housing_analysis
except ImportError:
    st.error("Errore: Il modulo 'california_housing_analysis' non è stato trovato. Assicurati che 'src/california_housing_analysis.py' esista.")
    st.stop()

st.set_page_config(
    page_title="California Housing Analysis",
    layout="wide",
)

st.title("🏠 Analisi sul Dataset California Housing")
st.markdown("""
Questa pagina mostra un'analisi completa del dataset California Housing, 
utilizzando tecniche di **EDA**, **Feature Selection**, **Riduzione della Dimensionalità**, 
**Rilevamento Outlier**, **Clustering** e **Modellazione Predittiva** (Regressione LASSO e Classificazione Gradient Boosting).
""")

# Esecuzione dell'analisi e caching dei risultati per Streamlit
@st.cache_data
def get_analysis_results():
    """Funzione con cache per eseguire l'analisi solo una volta."""
    return run_california_housing_analysis()

# Esegui l'analisi e ottieni i risultati
results = get_analysis_results()

st.header("1. Caricamento e Analisi Esplorativa (EDA)")
st.subheader("1.1. Dataset e Correlazione")
st.markdown("Il dataset contiene 8 feature e il valore mediano delle case (MedHouseVal) come target.")

st.pyplot(results['heatmap_fig'])

st.header("2. Analisi di Regressione (LASSO)")
st.markdown(f"**Modello:** Regressione Lasso ($\\alpha = 0.01$) addestrata sulle feature standardizzate.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("2.1. Metriche di Regressione")
    st.info(f"**Root Mean Squared Error (RMSE):** {results['lasso_rmse']:.4f}")
    st.info(f"**R-squared ($R^2$):** {results['lasso_r2']:.4f}")

with col2:
    st.subheader("2.2. Importanza delle Feature (Coefficienti Lasso)")
    st.dataframe(results['lasso_coefficients'].to_frame(name='Coefficiente'))
    st.markdown("""
    I coefficienti indicano l'impatto standardizzato di ciascuna feature sul valore mediano delle case:
    * **`MedInc` (Reddito Mediano):** Ha l'impatto positivo più forte.
    * **`Latitude` e `Longitude` (Posizione):** Hanno gli impatti negativi più forti (dopo la standardizzazione, indicano la correlazione con le aree meno ricche).
    * **`Population`:** Ha un coefficiente pari a 0.00, indicando che Lasso l'ha completamente scartato.
    """)

st.header("3. Riduzione della Dimensionalità e Visualizzazione")

col3, col4 = st.columns(2)
with col3:
    st.subheader("3.1. SelectKBest (3 Feature Migliori)")
    st.pyplot(results['kbest_3d_fig'])
with col4:
    st.subheader("3.2. Analisi delle Componenti Principali (PCA, 3 Componenti)")
    st.pyplot(results['pca_3d_fig'])

st.header("4. Rilevamento Outlier")
st.markdown("Gli outlier sono stati identificati su dati ridotti a 3 dimensioni (SelectKBest).")

col5, col6 = st.columns(2)
with col5:
    st.subheader("4.1. Outlier con KDE (3° Percentile di Densità)")
    st.pyplot(results['kde_outlier_fig'])
with col6:
    st.subheader("4.2. Outlier con LOF (k=30)")
    st.pyplot(results['lof_outlier_fig'])

st.header("5. Analisi di Clustering")

st.subheader("5.1. K-Means: Metodo del Gomito (Elbow Method)")
st.pyplot(results['kmeans_elbow_fig'])
st.markdown("Il plot suggerisce il numero ottimale di cluster per K-Means.")

st.subheader("5.2. Matrici di Confusione per Clustering (True Labels vs Predicted Clusters)")
st.markdown("Nota: Le etichette dei cluster (0, 1, 2) non corrispondono direttamente alle classi binarie reali (0, 1) per il valore mediano delle case, ma mostrano come gli algoritmi raggruppano i dati rispetto alle classi reali.")

col7, col8 = st.columns(2)
with col7:
    st.caption("DBSCAN (Density-Based Spatial Clustering of Applications with Noise)")
    st.pyplot(results['dbscan_cm_plot'])
with col8:
    st.caption("GMM (Gaussian Mixture Model)")
    st.pyplot(results['gm_cm_plot'])

st.header("6. Classificazione (Gradient Boosting)")
st.markdown("Il problema è convertito in **Classificazione Binaria** (Valore Mediano Casa > $250k$).")
st.markdown("È stata eseguita una **Grid Search** per ottimizzare il modello Gradient Boosting.")

st.subheader("6.1. Risultati Grid Search")
st.json(results['gb_best_params'])

st.subheader("6.2. Valutazione del Modello")
st.info(f"**F1 Score (sul Test Set):** {results['gb_f1_score']:.4f}")
st.info(f"**ROC AUC Score:** {results['gb_auc_score']:.4f}")

st.subheader("6.3. Curva ROC")
st.pyplot(results['gb_roc_fig'])

# Opzionale: Aggiungi un'opzione per salvare il modello
# with st.sidebar:
#     if st.button("Salva Modello Gradient Boosting"):
#         model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'gb_california_housing.pkl')
#         try:
#             with open(model_path, 'wb') as f:
#                 pickle.dump(results['gb_model'], f)
#             st.success(f"Modello salvato con successo in {model_path}")
#         except Exception as e:
#             st.error(f"Errore durante il salvataggio del modello: {e}")