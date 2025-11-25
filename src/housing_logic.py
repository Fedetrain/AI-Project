import pandas as pd
import numpy as np
import io
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from keras import layers

def load_housing_data():
    """Scarica il dataset California Housing (stessa struttura del notebook)."""
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    try:
        s = requests.get(url).content
        df = pd.read_csv(io.StringIO(s.decode('utf-8')))
        return df
    except:
        return None

def preprocess_housing(df):
    """Applica il preprocessing specifico del notebook."""
    data = df.copy()
    
    # 1. Gestione Valori Mancanti (total_bedrooms)
    # Calcolo ratio total_rooms/total_bedrooms
    ratio = data['total_rooms'] / data['total_bedrooms']
    median_ratio = ratio.median()
    data['total_bedrooms'].fillna(data['total_rooms'] / median_ratio, inplace=True)
    
    # 2. Feature Engineering
    data['bedrooms_per_household'] = data['total_bedrooms'] / data['households']
    
    # 3. Drop colonne ridondanti
    data.drop(columns=['total_bedrooms', 'total_rooms', 'households'], inplace=True)
    
    # 4. Encoding Ocean Proximity
    cat_cols = ['ocean_proximity']
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(data[cat_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
    
    # Unione
    data_final = pd.concat([data.drop(columns=cat_cols), encoded_df], axis=1)
    
    return data_final

def run_lasso_selection(X, y):
    """Esegue Lasso per la feature selection."""
    lasso = Lasso(alpha=1.0)
    lasso.fit(X, y)
    return lasso.coef_

def build_housing_nn(input_shape):
    """Costruisce la rete neurale Conv1D per regressione."""
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=[keras.metrics.RootMeanSquaredError(name='root_mean_squared_error')]
    )
    
    return model