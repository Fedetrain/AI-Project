import pandas as pd
import numpy as np
import requests
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import tensorflow as keras
from tensorflow.keras import layers

def load_titanic_data():
    """Scarica il dataset Titanic."""
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    s = requests.get(url).content
    df = pd.read_csv(io.StringIO(s.decode('utf-8')))
    return df

def preprocess_titanic(df):
    """Pipeline di preprocessing scikit-learn."""
    # Feature Engineering basilare
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    
    # Drop colonne inutili
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Definisci colonne
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title']

    # Pipeline Numerica
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline Categorica
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit transform
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    return X_train_proc, X_test_proc, y_train, y_test, preprocessor

def build_titanic_nn(input_dim):
    """Crea la rete neurale Keras per Titanic."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model