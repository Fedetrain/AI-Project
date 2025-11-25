import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, silhouette_score
from src.clustering_logic import (
    detect_outliers_lof, run_dbscan, run_gmm, 
    get_knn_distances, calculate_purity, calculate_gini, calculate_entropy
)
import os 

st.set_page_config(page_title="Advanced Clustering Analysis", layout="wide")

st.title("📊 Analisi Avanzata di Clustering")
st.markdown("""
Questa applicazione dimostra tecniche avanzate di **Clustering** e **Outlier Detection**.
""")

# ====================================================================
# *** MAPPING DEI CLUSTER ***
# ====================================================================

# Definizione del mapping per GMM (Cluster Trovato -> Classe Originale Stimata)
# Ottimizzato secondo l'analisi della matrice fornita in precedenza
mapping_gm = {
    0: 2,       
    1: 7, 
    4: 4,#ok
    3: 3,#
    4: 6,
    5: 1,
    6: 5
}

# Definizione del mapping per DBSCAN (Esempio iniziale - AGGIORNARE DOPO LA PRIMA RUN)
mapping_db = {
    0: 1,  
    1: 4,  
    2: 6,
    3: 3,
    4: 5,
    5: 7,
    6: 2
}
# ====================================================================

# --- 1. Caricamento Dati ---
st.header("1. Caricamento del Dataset")

DATA_PATH = "data/Aggregation.txt"

# Controllo se il file esiste
if os.path.exists(DATA_PATH):
    try:
        # Il file aggregation.txt è tab-separated e non ha header.
        df = pd.read_csv(DATA_PATH, delimiter='\t', names=['DIM1', 'DIM2', 'CLASS'])
        
        # AGGIUNTA DI PULIZIA DATI
        initial_rows = len(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        if len(df) < initial_rows:
            st.warning(f"Rimosse {initial_rows - len(df)} righe contenenti valori mancanti (NaN) o infiniti.")
        
        st.success(f"Dataset '{DATA_PATH}' caricato con successo!")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("Anteprima Dati:", df.head())
            st.write(f"Shape: {df.shape}")
        
        data = df.iloc[:, :-1].values
        target = df.iloc[:, -1].values
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            for cls in np.unique(target):
                mask = df['CLASS'] == cls
                ax.scatter(df.loc[mask, 'DIM1'], df.loc[mask, 'DIM2'], label=f'Class {cls}', s=20)
            ax.set_title("Distribuzione Originale (Ground Truth)")
            ax.legend()
            st.pyplot(fig)

        # --- 2. Outlier Detection (LOF) ---
        st.header("2. Rilevamento Outliers (LOF)")
        
        k_neighbors = st.slider("K Neighbors per LOF", 2, 20, 2)
        
        if st.button("Esegui LOF"):
            pred = detect_outliers_lof(data, n_neighbors=k_neighbors)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            inliers = data[pred == 1]
            outliers = data[pred == -1]
            
            ax.scatter(inliers[:, 0], inliers[:, 1], c='blue', label='Inliers', s=10)
            ax.scatter(outliers[:, 0], outliers[:, 1], c='red', label='Outliers', s=20, marker='x')
            ax.set_title(f"LOF Result (k={k_neighbors})")
            ax.legend()
            st.pyplot(fig)
            
            st.session_state['data_cleaned'] = data[pred == 1]
            st.session_state['target_cleaned'] = target[pred == 1]
            st.success(f"Dati puliti: {len(inliers)} samples (Rimossi {len(outliers)} outliers)")

        if 'data_cleaned' in st.session_state:
            data_cl = st.session_state['data_cleaned']
            target_cl = st.session_state['target_cleaned']

            # --- 3. DBSCAN ---
            st.header("3. Clustering DBSCAN")
            
            with st.expander("Aiuto per la scelta di Epsilon (K-dist plot)"):
                k_dist = st.slider("K per il grafico delle distanze", 2, 15, 4, key='k_dist_dbscan')
                distances = get_knn_distances(data_cl, k_dist)
                fig, ax = plt.subplots()
                ax.plot(distances)
                ax.set_title(f"K-distance Graph (k={k_dist})")
                ax.set_ylabel("Epsilon")
                ax.grid(True)
                st.pyplot(fig)

            col_d1, col_d2 = st.columns(2)
            eps = col_d1.number_input("Epsilon", 0.1, 5.0, 1.9, 0.1)
            min_samples = col_d2.number_input("Min Samples", 2, 20, 12)

            clusters_db = run_dbscan(data_cl, eps, min_samples)
            
            unique_clusters = np.unique(clusters_db)
            st.info(f"Cluster trovati da DBSCAN: {len(unique_clusters) - (1 if -1 in unique_clusters else 0)} (Rumore: {np.sum(clusters_db == -1)} punti)")

            fig, ax = plt.subplots(figsize=(10, 6))
            for cluster in unique_clusters:
                mask = clusters_db == cluster
                label = f"Cluster {cluster}" if cluster != -1 else "Noise"
                color = 'black' if cluster == -1 else None
                ax.scatter(data_cl[mask, 0], data_cl[mask, 1], label=label, c=color, s=20)
            ax.set_title("Risultato DBSCAN")
            ax.legend()
            st.pyplot(fig)
            
            # Analisi Metriche DBSCAN
            if len(unique_clusters) > 1:
                
                # *** IMPLEMENTAZIONE DEL MAPPING TRAMITE DATAFRAME ***
                
                # 1. Combina dati e cluster (solo i dati non-noise, se necessario per i calcoli)
                mask_no_noise = clusters_db != -1
                
                # DataFrame con solo i dati utili e senza noise
                data_dbscan = pd.DataFrame({
                    'target': target_cl[mask_no_noise],
                    'cluster': clusters_db[mask_no_noise]
                })

                if not data_dbscan.empty:
                    # 2. Applica il mapping ai cluster
                    # .map() userà NaN per i cluster non presenti in mapping_db
                    data_dbscan['cluster_mapped'] = data_dbscan['cluster'].map(mapping_db)
                    
                    # 3. Filtra via i NaN (punti di cluster non mappati)
                    data_dbscan.dropna(subset=['cluster_mapped'], inplace=True)

                    if not data_dbscan.empty:
                        # 4. Estrai i dati per la matrice di confusione
                        target_for_cm = data_dbscan['target'].values
                        clusters_for_cm = data_dbscan['cluster_mapped'].values.astype(int) # Converto in int dopo il mapping
                        st.info(f"Mapping DBSCAN applicato su {len(data_dbscan)} campioni.")
                        
                        cm_db = confusion_matrix(target_for_cm, clusters_for_cm)
                        
                        st.subheader("Valutazione DBSCAN")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Purity", f"{calculate_purity(cm_db):.4f}")
                        m2.metric("Gini Index", f"{calculate_gini(cm_db):.4f}")
                        m3.metric("Entropy", f"{calculate_entropy(cm_db):.4f}")
                        try:
                            # Silhouette score usa i cluster NON mappati (perché non è una metrica di classificazione)
                            sil = silhouette_score(data_cl[mask_no_noise], clusters_db[mask_no_noise])
                            m4.metric("Silhouette", f"{sil:.4f}")
                        except Exception as e:
                            m4.metric("Silhouette", f"N/A ({e})")

                        with st.expander("Vedi Matrice di Confusione DBSCAN Mappata"):
                            fig, ax = plt.subplots()
                            row_names = sorted(list(np.unique(target_for_cm)))
                            col_names = sorted(list(np.unique(clusters_for_cm)))
                            
                            sns.heatmap(cm_db, annot=True, fmt='d', cmap='crest', ax=ax,
                                        xticklabels=col_names, yticklabels=row_names)
                            ax.set_xlabel("Classe Stimata (Mappata)")
                            ax.set_ylabel("Classe Originale")
                            st.pyplot(fig)
                    else:
                        st.warning("Nessun punto mappabile è rimasto dopo il filtraggio DBSCAN.")
                else:
                    st.warning("Nessun cluster non-noise trovato da DBSCAN per l'analisi delle metriche.")

            # --- 4. Gaussian Mixture ---
            st.header("4. Gaussian Mixture Models (GMM)")
            n_comp = st.slider("Numero di Componenti", 2, 15, 7)
            
            clusters_gm = run_gmm(data_cl, n_comp)
            
            # Applicazione del Mapping GMM (resta con np.vectorize che è efficiente)
            if len(mapping_gm) == n_comp:
                clusters_gm_mapped = np.vectorize(mapping_gm.get)(clusters_gm)
                st.success("Mapping GMM applicato per la valutazione.")
            else:
                clusters_gm_mapped = clusters_gm
                st.warning(f"Il mapping non è stato applicato. Il numero di componenti GMM ({n_comp}) non corrisponde alla dimensione del mapping ({len(mapping_gm)}).")

            fig, ax = plt.subplots(figsize=(10, 6))
            for cluster in np.unique(clusters_gm):
                mask = clusters_gm == cluster
                ax.scatter(data_cl[mask, 0], data_cl[mask, 1], label=f"Cluster {cluster}", s=20) 
            ax.set_title("Risultato GMM (Cluster Originali)")
            ax.legend()
            st.pyplot(fig)

            # Metriche GMM (usando i cluster mappati)
            cm_gm = confusion_matrix(target_cl, clusters_gm_mapped)
            st.subheader("Valutazione GMM (con Mapping)")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Purity", f"{calculate_purity(cm_gm):.4f}")
            g2.metric("Gini Index", f"{calculate_gini(cm_gm):.4f}")
            g3.metric("Entropy", f"{calculate_entropy(cm_gm):.4f}")
            try:
                sil_gm = silhouette_score(data_cl, clusters_gm)
                g4.metric("Silhouette", f"{sil_gm:.4f}")
            except:
                g4.metric("Silhouette", "N/A (Errore)")

            with st.expander("Vedi Matrice di Confusione GMM Mappata"):
                fig, ax = plt.subplots()
                col_names = sorted(list(np.unique(clusters_gm_mapped)))
                row_names = sorted(list(np.unique(target_cl)))
                sns.heatmap(cm_gm, annot=True, fmt='d', cmap='flare', ax=ax, 
                            xticklabels=col_names, yticklabels=row_names)
                ax.set_xlabel("Classe Stimata (Mappata)")
                ax.set_ylabel("Classe Originale")
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Errore: {e}. Controlla la presenza di caratteri non validi (non numerici) o separatori errati nel file '{DATA_PATH}'.")
else:
    st.error(f"File non trovato: Assicurati che il dataset sia in '{DATA_PATH}' (cartella 'data' e file 'aggregation.txt').")