import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from src.bank_logic import load_bank_data, preprocess_bank, select_features

st.set_page_config(page_title="Bank Marketing", layout="wide")
st.title("🏦 Bank Marketing: Sottoscrizione Depositi")
st.markdown("""
Analisi di campagne di marketing diretto.
**Focus:** Importanza delle Feature e Selezione (SelectKBest).
""")

# Load
if 'bank_data' not in st.session_state:
    with st.spinner("Scaricamento e unzip dati..."):
        df = load_bank_data()
        X_tr, X_te, y_tr, y_te, feat_names = preprocess_bank(df)
        st.session_state['bank_data'] = (X_tr, X_te, y_tr, y_te, feat_names)

X_train, X_test, y_train, y_test, feature_names = st.session_state['bank_data']

# Feature Selection Analysis
st.header("1. Analisi Importanza Feature")
if st.checkbox("Esegui SelectKBest (ANOVA)"):
    k_best = st.slider("Quante feature mantenere?", 1, len(feature_names), 5)
    
    X_train_sel, selector, imp_df = select_features(X_train, y_train, feature_names, k=k_best)
    X_test_sel = selector.transform(X_test)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("Top Feature per ANOVA F-value:")
        st.dataframe(imp_df.head(10))
    with col2:
        fig, ax = plt.subplots()
        sns.barplot(data=imp_df.head(10), x='Score', y='Feature', ax=ax, palette='viridis')
        st.pyplot(fig)
    
    st.session_state['selected_data'] = (X_train_sel, X_test_sel)
else:
    # Usa tutto se non selezionato
    st.session_state['selected_data'] = (X_train, X_test)

# Modeling
st.header("2. Predizione con Gradient Boosting")
if st.button("Addestra Modello"):
    X_tr_curr, X_te_curr = st.session_state['selected_data']
    
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_tr_curr, y_train)
    
    probs = model.predict_proba(X_te_curr)[:, 1]
    auc = roc_auc_score(y_test, probs)
    
    st.metric("ROC AUC Score", f"{auc:.4f}")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"GBM (AUC={auc:.2f})")
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC")
    ax.legend()
    st.pyplot(fig)