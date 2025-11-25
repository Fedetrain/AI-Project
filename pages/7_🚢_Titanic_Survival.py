import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from src.titanic_logic import load_titanic_data, preprocess_titanic, build_titanic_nn
import tensorflow as tf

st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")
st.title("🚢 Titanic: Predizione della Sopravvivenza")
st.markdown("Un classico problema di classificazione binaria affrontato con tecniche moderne.")

# 1. Data Loading
if 'titanic_df' not in st.session_state:
    with st.spinner("Caricamento dati..."):
        df = load_titanic_data()
        st.session_state['titanic_df'] = df
        X_tr, X_te, y_tr, y_te, prep = preprocess_titanic(df)
        st.session_state['titanic_processed'] = (X_tr, X_te, y_tr, y_te)

df = st.session_state['titanic_df']
X_train, X_test, y_train, y_test = st.session_state['titanic_processed']

with st.expander("Esplora il Dataset"):
    st.dataframe(df.head())
    st.write(f"Dimensioni Training Set (Processato): {X_train.shape}")

# 2. Modeling
st.header("Addestramento Modelli")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Machine Learning Classico")
    model_type = st.selectbox("Scegli Algoritmo", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
    
    if st.button("Addestra ML"):
        if model_type == "Logistic Regression":
            clf = LogisticRegression(max_iter=1000)
        elif model_type == "Random Forest":
            clf = RandomForestClassifier(n_estimators=100, max_depth=10)
        else:
            clf = GradientBoostingClassifier()
            
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.success(f"Accuracy {model_type}: {acc:.4f}")
        
        # Feature Importance (se disponibile)
        if hasattr(clf, 'feature_importances_'):
            st.caption("Nota: L'importanza è basata sulle feature one-hot encoded, non sulle colonne originali.")
            # Semplificazione visualizzazione
            st.bar_chart(clf.feature_importances_[:20])

with col2:
    st.subheader("Deep Learning (Keras)")
    epochs = st.slider("Epoche", 10, 100, 30)
    
    if st.button("Addestra Rete Neurale"):
        model = build_titanic_nn(X_train.shape[1])
        
        with st.spinner("Training NN..."):
            history = model.fit(
                X_train.toarray(), y_train, # .toarray() se sparse
                validation_split=0.2,
                epochs=epochs,
                batch_size=32,
                verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
            )
            
        loss, acc = model.evaluate(X_test.toarray(), y_test, verbose=0)
        st.success(f"NN Test Accuracy: {acc:.4f}")
        
        # Plot Curves
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(history.history['accuracy'], label='Train')
        ax.plot(history.history['val_accuracy'], label='Val')
        ax.set_title("Training Accuracy")
        ax.legend()
        st.pyplot(fig)