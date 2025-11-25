import pandas as pd
import numpy as np
import requests
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.utils import class_weight

def load_adult_data():
    """Scarica il dataset Adult Census."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
               'hours-per-week', 'native-country', 'income']
    
    s = requests.get(url).content
    df = pd.read_csv(io.StringIO(s.decode('utf-8')), names=columns, na_values=' ?', skipinitialspace=True)
    return df

def preprocess_adult(df):
    """Preprocessing avanzato con gestione missing values."""
    df = df.copy()
    
    # Drop missing
    df.dropna(inplace=True)
    
    # Target Encoding
    df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    
    y = df['income'].values
    X = df.drop('income', axis=1)
    
    # Dummies per le categoriche
    X = pd.get_dummies(X, drop_first=True)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Calcolo pesi classi sbilanciate
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(weights))
    
    return X_train_s, X_test_s, y_train, y_test, class_weights

def build_adult_nn(input_dim):
    """MLP robusto per classificazione."""
    import keras
    from keras import layers
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model