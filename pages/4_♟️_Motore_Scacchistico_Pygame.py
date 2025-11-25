# pages/4_♟️_Motore_Scacchistico_Pygame.py
import streamlit as st

st.set_page_config(page_title="Motore Scacchi Pygame", layout="wide")
st.title("♟️ Progetto: Motore Scacchistico Pygame con Stockfish")
st.markdown("*([Ispirato da: `scacchi.zip`])*")

st.markdown("""
Questo progetto è un'applicazione desktop **Pygame** completa che implementa
un gioco di scacchi funzionante.

A differenza degli altri progetti, **un'applicazione Pygame non può essere eseguita
in modo interattivo nel browser** tramite Streamlit.

Questa pagina serve a descrivere il progetto e a dimostrare la mia competenza
nello sviluppo di applicazioni complesse.
""")

st.header("Caratteristiche Principali")
st.markdown("""
- **GUI Interattiva:** Costruita con Pygame per il rendering della scacchiera, dei pezzi e la gestione del drag-and-drop.
- **Logica di Gioco:** Implementazione completa delle regole degli scacchi, inclusi movimenti validi, scacco, scacco matto e arrocco (es. `engine.py`, `game.py`).
- **Integrazione AI:** Utilizza una pipe per comunicare con l'engine **Stockfish**, permettendo di giocare contro un avversario AI di altissimo livello.
- **Gestione PGN:** Include utility per leggere e interpretare la notazione PGN.
""")

st.header("Screenshot del Gioco")
st.warning("Azione richiesta: Aggiungi uno screenshot del tuo gioco in `assets/scacchi_screenshot.png` e scommenta la riga qui sotto.")
# st.image("assets/scacchi_screenshot.png", caption="Screenshot dell'applicazione Pygame in esecuzione")

st.header("Esplora il Codice")
st.markdown("Il codice completo di questo progetto è disponibile sul mio repository GitHub.")
st.link_button("Vai al Repository GitHub", "https://github.com/TUO_NOME_UTENTE/NOME_REPO")