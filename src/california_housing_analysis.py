# src/california_housing_analysis.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error, r2_score, f1_score, make_scorer, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity, LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans,DBSCAN

# --- 1. Caricamento e Pre-elaborazione ---
def load_and_preprocess_data(test_size=0.25, random_state=42):
    """Carica il dataset, lo pre-elabora e lo divide in set di training/test."""
    housing = fetch_california_housing()
    data = pd.DataFrame(housing.data, columns=housing.feature_names)
    target = housing.target
    
    # Target di classificazione binaria: Valore mediano > $250k
    y_class = (target > 2.5).astype(int)

    # Divisione dati
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, test_size=test_size, shuffle=True, random_state=random_state
    )

    # Standardizzazione
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # Target di classificazione per i set divisi
    y_class_train = (y_train > 2.5).astype(int)
    y_class_test = (y_test > 2.5).astype(int)

    return x_train_scaled, x_test_scaled, y_train, y_test, y_class_train, y_class_test, data, housing.feature_names

# --- 2. Analisi Esplorativa (EDA) ---
def plot_correlation_heatmap(data):
    """Genera e restituisce la figura della heatmap di correlazione."""
    corr_matrix = data.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Heatmap della Matrice di Correlazione")
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()
    return fig

# --- 3. Analisi di Regressione (LASSO) ---
def perform_lasso_regression(x_train, y_train, x_test, y_test, feature_names, alpha=0.01):
    """Esegue la Regressione Lasso e restituisce i coefficienti e le metriche."""
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(x_train, y_train)
    predictions = lasso.predict(x_test)
    
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    coefficients = pd.Series(lasso.coef_, index=feature_names).sort_values(ascending=False)
    
    return coefficients, rmse, r2

# --- 4. Riduzione della Dimensionalità e Visualizzazione ---
def reduce_and_plot_3d(x_train, y_train, method='kbest'):
    """Esegue SelectKBest o PCA e plota in 3D."""
    if method == 'kbest':
        selector = SelectKBest(k=3)
        x_train_3d = selector.fit_transform(x_train, y_train)
        title = "Scatter 3D con SelectKBest"
    elif method == 'pca':
        pca = PCA(n_components=3)
        x_train_3d = pca.fit_transform(x_train)
        title = "Scatter 3D con PCA"
    else:
        raise ValueError("Method must be 'kbest' or 'pca'")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x_train_3d[:, 0], x_train_3d[:, 1], x_train_3d[:, 2], 
                    c=y_train, cmap='coolwarm', alpha=0.7)

    ax.set_xlabel(f"{'Feature 1' if method == 'kbest' else 'PC1'}")
    ax.set_ylabel(f"{'Feature 2' if method == 'kbest' else 'PC2'}")
    ax.set_zlabel(f"{'Feature 3' if method == 'kbest' else 'PC3'}")
    plt.title(title)
    plt.colorbar(sc, label="MedHouseVal")
    plt.tight_layout()
    plt.close(fig)
    
    # Per la valutazione LOF e KDE, restituiamo i dati 3D
    return fig, x_train_3d

# --- 5. Rilevamento Outlier (KDE) ---
def plot_kde_outliers(x_train_3d, y_train, threshold_percentile=3):
    """Identifica e plota gli outlier usando KDE."""
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(x_train_3d)
    log_density = kde.score_samples(x_train_3d)
    
    threshold = np.percentile(log_density, threshold_percentile)
    outlier_kde_mask = log_density < threshold
    inlier_mask = ~outlier_kde_mask

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Inlier
    ax.scatter(x_train_3d[inlier_mask, 0], x_train_3d[inlier_mask, 1], x_train_3d[inlier_mask, 2],
               c='blue', alpha=0.3, label='Inlier')

    # Outlier
    ax.scatter(x_train_3d[outlier_kde_mask, 0], x_train_3d[outlier_kde_mask, 1], x_train_3d[outlier_kde_mask, 2],
               c='red', alpha=0.8, label='Outlier')

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    plt.title("Outlier detection con KDE (3D)")
    plt.legend()
    plt.tight_layout()
    plt.close(fig)
    return fig

# --- 6. Rilevamento Outlier (LOF) ---
def plot_lof_outliers(x_train_3d, n_neighbors=30):
    """Identifica e plota gli outlier usando LOF."""
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    score = lof.fit_predict(x_train_3d)
    
    outlier_mask = (score == -1)
    inlier_mask = (score == 1)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Inlier
    ax.scatter(x_train_3d[inlier_mask, 0], x_train_3d[inlier_mask, 1], x_train_3d[inlier_mask, 2],
               c='blue', alpha=0.3, label='Inlier')

    # Outlier
    ax.scatter(x_train_3d[outlier_mask, 0], x_train_3d[outlier_mask, 1], x_train_3d[outlier_mask, 2],
               c='red', alpha=0.8, label='Outlier')

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    plt.title("Outlier detection con LOF (3D)")
    plt.legend()
    plt.tight_layout()
    plt.close(fig)
    return fig

# --- 7. Clustering (K-Means) ---
def plot_kmeans_elbow(x_scaled):
    """Esegue il metodo del gomito per K-Means e restituisce la figura."""
    wccs = []
    max_k = 10
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(x_scaled)
        wccs.append(kmeans.inertia_)
        
    plt.figure(figsize=(6, 6))
    plt.plot(range(1, max_k + 1), wccs, marker='o')
    plt.title('Elbow Method per K-Means')
    plt.xlabel('Numero di Cluster (K)')
    plt.ylabel('WCCS (Within-Cluster Sum of Squares)')
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()
    return fig

# --- 8. Classificazione (Gradient Boosting) ---
def tune_and_evaluate_gb(x_train, y_class_train, x_test, y_class_test, random_state=42):
    """Esegue GridSearch per Gradient Boosting e valuta il modello."""
    
    # Usiamo un sottoinsieme per velocizzare GridSearch come nel notebook
    np.random.seed(random_state)
    idx = np.random.choice(len(x_train), 2000, replace=False)
    x_train_small = x_train[idx]
    y_train_small = y_class_train[idx]

    boosting = GradientBoostingClassifier(random_state=random_state)
    param_grid = {
        'n_estimators': [100, 300],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
    
    f1_scorer = make_scorer(f1_score)
    grid_search = GridSearchCV(estimator=boosting,
                               param_grid=param_grid,
                               scoring=f1_scorer,
                               cv=5,
                               verbose=0,  # Impostato a 0 per non stampare tutto
                               refit=True)
    
    grid_search.fit(x_train_small, y_train_small)
    
    best_params = grid_search.best_params_
    
    # Predizioni e metriche sul test set completo
    y_pred = grid_search.predict(x_test)
    y_pred_proba = grid_search.predict_proba(x_test)[:, 1]
    
    f1 = f1_score(y_class_test, y_pred)
    auc_score = roc_auc_score(y_class_test, y_pred_proba)
    
    # Plot ROC Curve
    plt.figure(figsize=(6,6))
    fpr, tpr, _ = roc_curve(y_class_test, y_pred_proba)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Gradient Boosting")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    roc_fig = plt.gcf()
    plt.close()

    return best_params, f1, auc_score, roc_fig, grid_search.best_estimator_

# --- 9. Plot Matrice di Confusione Clustering ---
def plot_clustering_confusion(y_true, y_pred, title="Matrice di Confusione (Clustering)"):
    """Plota la matrice di confusione per un risultato di clustering (usando solo le label)."""
    # Nota: Le etichette di clustering (0, 1, 2) non corrispondono necessariamente alle etichette reali (0, 1, 2).
    # La matrice di confusione qui è solo un modo per visualizzare la distribuzione delle predizioni.
    # In un contesto reale si userebbero metriche come ARI o NMI.
    
    # Per DBSCAN, gli outlier (-1) sono una classe, ma ConfusionMatrixDisplay li gestisce come una label.
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, normalize=None, values_format='d')
    ax.set_title(title)
    plt.tight_layout()
    plt.close(fig)
    return fig

# --- 10. Funzione Principale di Esecuzione ---
def run_california_housing_analysis():
    """Esegue tutti i passaggi dell'analisi."""
    
    print("Inizio analisi California Housing...")
    
    # 1. Caricamento e Pre-elaborazione
    x_train_scaled, x_test_scaled, y_train, y_test, y_class_train, y_class_test, data, feature_names = load_and_preprocess_data()
    
    # 2. EDA
    corr_fig = plot_correlation_heatmap(data)
    
    # 3. Analisi di Regressione (LASSO)
    lasso_coeffs, lasso_rmse, lasso_r2 = perform_lasso_regression(
        x_train_scaled, y_train, x_test_scaled, y_test, feature_names
    )
    
    # 4. Riduzione Dimensionalità
    kbest_plot, x_train_kbest_3d = reduce_and_plot_3d(x_train_scaled, y_train, method='kbest')
    pca_plot, x_train_pca_3d = reduce_and_plot_3d(x_train_scaled, y_train, method='pca')
    
    # 5. Outlier (KDE)
    kde_outlier_plot = plot_kde_outliers(x_train_kbest_3d, y_train)
    
    # 6. Outlier (LOF)
    lof_outlier_plot = plot_lof_outliers(x_train_kbest_3d)

    # 7. Clustering (K-Means)
    kmeans_elbow_plot = plot_kmeans_elbow(x_train_scaled)
    
    # 8. Classificazione (Gradient Boosting)
    gb_best_params, gb_f1, gb_auc, gb_roc_fig, gb_model = tune_and_evaluate_gb(
        x_train_scaled, y_class_train, x_test_scaled, y_class_test
    )
    
    # 9. Plot Matrice di Confusione Clustering
    # Necessitiamo di eseguire gli algoritmi sul set completo per la matrice di confusione
    # (assumendo le predizioni y_test)
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.7)
    dbscan_preds_test = dbscan.fit_predict(x_test_scaled)
    dbscan_cm_plot = plot_clustering_confusion(y_class_test, dbscan_preds_test, "Matrice di Confusione (DBSCAN)")
    
    # GMM
    prior = np.ones(shape=(3,)) / 3
    gm = GaussianMixture(n_components=3, weights_init=prior, random_state=42)
    gm.fit(x_train_scaled) # addestramento su training set
    gm_preds_test = gm.predict(x_test_scaled)
    gm_cm_plot = plot_clustering_confusion(y_class_test, gm_preds_test, "Matrice di Confusione (GMM)")


    results = {
        "heatmap_fig": corr_fig,
        "lasso_coefficients": lasso_coeffs,
        "lasso_rmse": lasso_rmse,
        "lasso_r2": lasso_r2,
        "kbest_3d_fig": kbest_plot,
        "pca_3d_fig": pca_plot,
        "kde_outlier_fig": kde_outlier_plot,
        "lof_outlier_fig": lof_outlier_plot,
        "kmeans_elbow_fig": kmeans_elbow_plot,
        "gb_best_params": gb_best_params,
        "gb_f1_score": gb_f1,
        "gb_auc_score": gb_auc,
        "gb_roc_fig": gb_roc_fig,
        "dbscan_cm_plot": dbscan_cm_plot,
        "gm_cm_plot": gm_cm_plot,
        "gb_model": gb_model # Per eventuale salvataggio/deploy
    }
    
    print("Analisi California Housing completata.")
    return results

if __name__ == '__main__':
    # Esempio di esecuzione locale (solo per test)
    results = run_california_housing_analysis()
    
    # Stampa di alcuni risultati
    print("\n--- Risultati Regressione LASSO ---")
    print(f"RMSE: {results['lasso_rmse']:.4f}")
    print(f"R2 Score: {results['lasso_r2']:.4f}")
    print("\nCoeff. Lasso (Top 5):")
    print(results['lasso_coefficients'].head())
    
    print("\n--- Risultati Classificazione Gradient Boosting ---")
    print(f"Best Params: {results['gb_best_params']}")
    print(f"F1 Score: {results['gb_f1_score']:.4f}")
    print(f"ROC AUC: {results['gb_auc_score']:.4f}")
    
    # Mostrare le figure se in un ambiente compatibile (qui non possibile, ma utile per la pagina Streamlit)
    # plt.show()