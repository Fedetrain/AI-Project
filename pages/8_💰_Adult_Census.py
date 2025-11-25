import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from src.adult_logic import load_adult_data, preprocess_adult, build_adult_nn
import tensorflow as tf

st.set_page_config(page_title="Adult Census Income", layout="wide")
st.title("💰 Adult Census: Previsione del Reddito")
st.markdown("""
Predizione se il reddito supera i 50K$ annui. 
**Focus:** Gestione di **classi sbilanciate** e dati categorici sporchi.
""")

# Load Data
@st.cache_data
def get_adult_data():
    df = load_adult_data()
    return preprocess_adult(df)

if st.button("Carica e Processa Dati (può richiedere qualche secondo)"):
    X_train, X_test, y_train, y_test, weights = get_adult_data()
    st.session_state['adult_data'] = (X_train, X_test, y_train, y_test, weights)
    st.success("Dati pronti!")

if 'adult_data' in st.session_state:
    X_train, X_test, y_train, y_test, weights = st.session_state['adult_data']
    
    st.info(f"Ratio Classi (0/1): {weights[0]:.2f} vs {weights[1]:.2f} (Pesi calcolati per bilanciamento)")

    tab1, tab2 = st.tabs(["Random Forest (Classico)", "Neural Network (Bilanciata)"])

    with tab1:
        n_est = st.slider("N Alberi", 50, 300, 100)
        if st.button("Train Random Forest"):
            # Usiamo class_weight='balanced' per gestire lo sbilanciamento
            rf = RandomForestClassifier(n_estimators=n_est, class_weight='balanced', n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            
            st.text("Classification Report:")
            st.code(classification_report(y_test, y_pred))
            
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

    with tab2:
        st.write("Rete Neurale con Dropout e BatchNormalization.")
        if st.button("Train Keras Model"):
            model = build_adult_nn(X_train.shape[1])
            
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=5, mode='max')
            
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=20,
                batch_size=64,
                class_weight=weights, # Applicazione pesi
                callbacks=[early_stop],
                verbose=0
            )
            
            eval_res = model.evaluate(X_test, y_test, verbose=0)
            st.metric("AUC Score", f"{eval_res[1]:.4f}")
            
            fig, ax = plt.subplots(figsize=(8,3))
            ax.plot(history.history['loss'], label='Loss')
            ax.plot(history.history['val_loss'], label='Val Loss')
            ax.legend()
            st.pyplot(fig)