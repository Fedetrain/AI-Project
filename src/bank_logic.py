import pandas as pd
import numpy as np
import requests
import io
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

def load_bank_data():
    """Scarica e estrae il dataset Bank Marketing (ZIP)."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    # Usiamo il file 'bank-full.csv' contenuto nello zip
    df = pd.read_csv(z.open('bank-full.csv'), sep=';')
    return df

def preprocess_bank(df):
    """Codifica e preparazione."""
    df = df.copy()
    
    # Target mapping
    df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)
    y = df['y']
    X = df.drop('y', axis=1)
    
    # Codifica Ordinale per tutto (semplificazione per Feature Selection)
    # Nel progetto reale si userebbe OneHot per alcune, ma per SelectKBest l'ordinale è gestibile
    enc = OrdinalEncoder()
    X_encoded = enc.fit_transform(X)
    X_encoded = pd.DataFrame(X_encoded, columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    return X_train_s, X_test_s, y_train, y_test, list(X.columns)

def select_features(X_train, y_train, feature_names, k=10):
    """Esegue Feature Selection usando ANOVA F-value."""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X_train, y_train)
    
    scores = selector.scores_
    # Crea dataframe importanza
    imp_df = pd.DataFrame({'Feature': feature_names, 'Score': scores})
    imp_df = imp_df.sort_values(by='Score', ascending=False)
    
    return X_new, selector, imp_df