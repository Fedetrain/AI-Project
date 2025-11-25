import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import keras
from keras import Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.regularizers import l2

def generate_synthetic_data(n_samples=2000, n_outliers=50):
    """Genera il dataset sintetico con outlier."""
    np.random.seed(42)
    X = np.random.uniform(-20, 20, n_samples)
    y = 5 * np.sin(0.5 * X) + X + np.random.normal(scale=1.5, size=X.shape)

    X_outliers = np.random.uniform(-20, 20, n_outliers)
    y_outliers = 5 * np.sin(0.5 * X_outliers) + X_outliers + np.random.normal(scale=15, size=X_outliers.shape)

    X_final = np.concatenate([X, X_outliers])
    y_final = np.concatenate([y, y_outliers])
    
    return pd.DataFrame({'X': X_final, 'y': y_final})

def clean_data_kde(df, bandwidth=0.5, percentile=20):
    """Rimuove outlier usando Kernel Density Estimation."""
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(df.values)
    score = kde.score_samples(df.values)
    df_cleaned = df[score > np.percentile(score, percentile)]
    return df_cleaned

def train_kernel_ridge(X, y):
    """Addestra Kernel Ridge con GridSearchCV."""
    param_grid = {
        'alpha': [0.01, 0.1, 1.0],
        'kernel': ['rbf'],
        'gamma': [0.001, 0.01, 0.05]
    }
    gs = GridSearchCV(KernelRidge(), param_grid=param_grid, scoring='r2', cv=5, n_jobs=-1)
    gs.fit(X, y)
    return gs

def build_and_train_nn(X, y, epochs=100, batch_size=32):
    """Costruisce e addestra la Rete Neurale Keras."""
    model = Sequential()
    model.add(Input(shape=(1,)))
    
    # Architettura definita nel notebook
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())

    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.92, beta_2=0.997),
        loss='mse',
        metrics=['r2_score'] # Nota: r2_score è disponibile in TF recenti, altrimenti usare 'mae'
    )
    
    # Per semplicità nel portfolio non usiamo callback complessi che salvano su disco, 
    # ma manteniamo l'EarlyStopping
    callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.01, patience=15, restore_best_weights=True
    )
    
    history = model.fit(
        x=X, y=y, batch_size=batch_size, epochs=epochs, 
        callbacks=[callback], validation_split=0.1, verbose=0
    )
    return model, history