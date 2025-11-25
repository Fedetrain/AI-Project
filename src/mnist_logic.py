import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def load_mnist_data():
    """Carica e preprocessa il dataset MNIST."""
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Split validation come nel notebook (10%)
    X_train, X_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.1, random_state=42)
    
    # Normalizzazione e Reshape per CNN
    X_train_cnn = X_train.astype("float32") / 255.0
    X_val_cnn = X_val.astype("float32") / 255.0
    X_test_cnn = x_test.astype("float32") / 255.0
    
    X_train_cnn = np.expand_dims(X_train_cnn, -1)
    X_val_cnn = np.expand_dims(X_val_cnn, -1)
    X_test_cnn = np.expand_dims(X_test_cnn, -1)
    
    return (X_train, y_train), (X_val, y_val), (x_test, y_test), (X_train_cnn, X_val_cnn, X_test_cnn)

def get_data_augmentation_layer():
    """Restituisce il blocco di Data Augmentation."""
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.05), # +/- 10 gradi circa
        layers.RandomTranslation(0.1, 0.1), # +/- 5 pixel circa su 28x28
    ])
    return data_augmentation

def train_random_forest_grid(X_train, y_train):
    """Esegue GridSearchCV su Random Forest (su un subset per velocità nel portfolio)."""
    # Flattening
    X_flat = X_train.reshape(len(X_train), -1)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    
    # Per la demo riduciamo il dataset o i parametri se necessario, 
    # ma qui replichiamo la logica del notebook
    # Usiamo un subset per non bloccare il server streamlit troppo a lungo
    subset_size = 5000
    X_subset = X_scaled[:subset_size]
    y_subset = y_train[:subset_size]

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 20],
        'n_estimators': [50, 100] # Ridotto leggermente per performance live
    }
    
    rf = RandomForestClassifier(random_state=42)
    gs = GridSearchCV(rf, param_grid, scoring='accuracy', cv=3, n_jobs=-1)
    gs.fit(X_subset, y_subset)
    
    return gs, scaler

def build_mnist_cnn(input_shape):
    """Costruisce la CNN definita nel notebook."""
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        # Data Augmentation layer can be included here or applied to dataset
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        layers.Dropout(0.25),
        layers.Dense(10, activation="softmax"),
    ])
    
    # Learning Rate Schedule
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.96
    )
    
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model