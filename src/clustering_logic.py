import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, silhouette_score

def calculate_purity(cm):
    """Calcola la purity dalla matrice di confusione."""
    Pj = np.max(cm, axis=0)
    return np.sum(Pj) / np.sum(cm)

def calculate_gini(cm):
    """Calcola l'indice di Gini."""
    M = np.sum(cm, axis=0)
    # Evitiamo divisione per zero
    M = np.where(M == 0, 1, M)
    G = 1 - np.sum((cm / M) ** 2, axis=0)
    return np.dot(G, M) / np.sum(M)

def calculate_entropy(cm):
    """Calcola l'entropia."""
    M = np.sum(cm, axis=0)
    # Evitiamo divisione per zero
    M_safe = np.where(M == 0, 1, M)
    P = cm / M_safe
    P = np.where(P == 0, 1, P)  # log(1) = 0
    E = -np.sum(P * np.log(P), axis=0)
    return np.dot(E, M) / np.sum(M)

def detect_outliers_lof(data, n_neighbors=7):
    """Esegue LOF per rilevare outlier."""
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, metric='euclidean')
    pred = lof.fit_predict(data)
    return pred

def run_dbscan(data, eps, min_samples):
    """Esegue DBSCAN."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data)
    return clusters

def run_gmm(data, n_components):
    """Esegue Gaussian Mixture Model."""
    gm = GaussianMixture(n_components=n_components, weights_init=np.ones(n_components)/n_components, covariance_type='full', random_state=42)
    clusters = gm.fit_predict(data)
    return clusters

def get_knn_distances(data, k):
    """Calcola le distanze per il grafico del gomito (k-dist)."""
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(data)
    distances, _ = nn.kneighbors(data)
    return np.sort(distances[:, k-1], axis=0)