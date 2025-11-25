# pages/3_🧠_Solutori_Algoritmici_DFS.py
import streamlit as st
import numpy as np
# Importiamo le classi dal file di logica
from src.dfs_solvers import solve_sudoku, ReginaSolver, HanoiSolver, MissionaryCannibalSolver 

st.set_page_config(page_title="Solutori Algoritmici", layout="wide")
st.title("🧠 Demo: Solutori Algoritmici (Backtracking/BFS)")
st.markdown("Questa pagina mostra la potenza degli algoritmi di ricerca (DFS e BFS) con una **GUI visivamente esplicativa** per problemi classici di AI/Informatica.")

# --- INIEZIONE CSS GLOBALE PER UNA VISUALIZZAZIONE ROBUSTA ---
# Aggiunto box-sizing: border-box; globale per prevenire problemi di layout CSS
GLOBAL_CSS = """
<style> 
* {
    box-sizing: border-box;
}
/* SUDOKU STYLES */
.sudoku-grid { 
    display: grid; 
    grid-template-columns: repeat(9, 1fr); 
    width: 360px; 
    border: 3px solid black; 
    margin-top: 10px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
} 
.sudoku-cell { 
    height: 40px; 
    width: 40px;
    display: flex; 
    align-items: center; 
    justify-content: center; 
    font-size: 1.2em; 
    border: 1px solid #ccc; 
} 
.initial-number { 
    font-weight: bold; 
    color: darkred; 
    background-color: #f0f0f0; 
}
.solved-number { 
    font-weight: bold; 
    color: #333; 
    background-color: #e6f7ff; /* Colore leggero per i numeri risolti */
}

/* HANOI STYLES */
.hanoi-container { 
    display: flex; 
    justify-content: space-around; 
    width: 100%; 
    height: 250px; 
    margin-top: 20px;
    padding: 10px;
    background-color: #f7f7f7;
    border-radius: 10px;
}
.hanoi-peg { 
    width: 30%; 
    height: 100%; 
    border-bottom: 5px solid #6b4d32; 
    display: flex; 
    flex-direction: column-reverse; /* Questo impila i dischi dal basso */
    align-items: center; 
    position: relative; 
}
.hanoi-peg-center { 
    position: absolute; 
    bottom: 5px; 
    width: 5px; 
    height: calc(100% - 30px); 
    background-color: #6b4d32; 
}
.hanoi-disk { 
    height: 30px; 
    border: 1px solid #333; 
    margin-bottom: 2px; 
    border-radius: 5px; 
    z-index: 1; 
}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# --- FUNZIONI DI RENDERING (Usano le classi CSS definite sopra) ---

def render_sudoku_board(board, initial_board):
    """Rende la board del Sudoku usando CSS."""
    grid_cells = ""
    # Verifica se la board è stata risolta (contiene un 9 e non è identica a quella iniziale)
    is_solved_board = np.any(board != 0) and not np.array_equal(board, initial_board) 

    for r in range(9):
        for c in range(9):
            val = board[r, c]
            display_val = str(val) if val != 0 else ""
            
            style = ""
            if (r + 1) % 3 == 0 and r != 8:
                style += " border-bottom: 3px solid black !important;"
            if (c + 1) % 3 == 0 and c != 8:
                style += " border-right: 3px solid black !important;"
            
            class_name = ""
            if val != 0:
                if initial_board[r, c] != 0: 
                    class_name = "initial-number"
                elif is_solved_board: # Aggiungi la classe solved solo se è la board risolta
                    class_name = "solved-number" 

            grid_cells += f'<div class="sudoku-cell {class_name}" style="{style}">{display_val}</div>'

    st.markdown(f'<div class="sudoku-grid">{grid_cells}</div>', unsafe_allow_html=True)


def render_queens_board(solution):
    """Renderizza una singola soluzione N-Regine come una scacchiera con l'emoji della regina."""
    n = len(solution)
    html = f"<div style='display: grid; grid-template-columns: repeat({n}, 40px); border: 2px solid black; margin-bottom: 15px;'>"
    
    for r in range(n):
        q_col = solution[r]
        for c in range(n):
            color = '#f0d9b5' if (r + c) % 2 == 0 else '#b58863'
            content = '♛' if c == q_col else ''
            queen_style = "font-size: 28px; color: black; line-height: 40px;" if content else ""
            
            html += f"<div style='width: 40px; height: 40px; background-color: {color}; display: flex; align-items: center; justify-content: center; {queen_style}'>{content}</div>"
    
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# CORREZIONE QUI
def render_hanoi_state(state, step_index=None):
    """Renderizza lo stato delle Torri di Hanoi con rettangoli."""
    peg_names = ["A", "B", "C"]
    
    pegs_html = ""
    colors = ['#FFC300', '#FF5733', '#C70039', '#3498db', '#2ecc71'] 
    
    for i in range(3):
        peg_html = ""
        for disk in state[i]:
            width = (disk + 1) * 20 
            color = colors[disk % len(colors)] 
            # I dischi sono resi come fratelli diretti di .hanoi-peg per sfruttare flex-direction: column-reverse
            peg_html += f'<div class="hanoi-disk" style="width: {width}px; background-color: {color};"></div>'

        # Rimosso il div wrapper superfluo che impediva la corretta impilatura (flexbox)
        pegs_html += f'<div class="hanoi-peg"><div class="hanoi-peg-center" style="z-index: 0;"></div>{peg_html} <p style="margin-top: 5px; font-weight: bold;">Torre {peg_names[i]}</p></div>'

    header = f"<h3>Passo {step_index}</h3>" if step_index is not None else ""
    st.markdown(header + f'<div class="hanoi-container">{pegs_html}</div>', unsafe_allow_html=True)


def render_missionary_cannibal_state(state):
    """Renderizza lo stato Missionari e Cannibali usando emoji e st.columns per un layout robusto."""

    M = "👨"  # Missionario
    C = "😈"  # Cannibale
    B = "🛶"  # Barca

    M_L, C_L = state["L"]["missionari"], state["L"]["cannibali"]
    M_R, C_R = state["R"]["missionari"], state["R"]["cannibali"]
    boat_pos = state["boat"]

    left_side_content = M * M_L + C * C_L
    right_side_content = M * M_R + C * C_R

    # Definisco lo stato della barca e il colore del box stato
    if boat_pos == "L":
        river_content = f'<span style="font-size: 30px;">{B}</span> 🌊'
        boat_info = "Barca a Sinistra"
        status_color = "#3498db"
    else:
        river_content = f'🌊 <span style="font-size: 30px;">{B}</span>'
        boat_info = "Barca a Destra"
        status_color = "#e74c3c"

    # Uso di st.container per dare un bordo comune all'intera visualizzazione dello stato
    with st.container(border=True):
        # Header (Sinistra / Destra) con HTML semplice
        st.markdown(f'<div style="display: flex; justify-content: space-between; font-weight: bold; margin-bottom: 5px; font-size: 14px; color: #555;">'
                    f'<span>SINISTRA</span><span>DESTRA</span></div>', unsafe_allow_html=True)
        
        # Layout principale con st.columns: Sinistra(4) - Fiume/Barca(2) - Destra(4)
        col_l, col_river, col_r = st.columns([4, 2, 4]) 

        # Contenuto Sinistro
        with col_l:
            st.caption(f"({M_L} M, {C_L} C)")
            st.markdown(f"<div style='text-align: center; font-size: 32px; min-height: 50px;'>{left_side_content}</div>", unsafe_allow_html=True)
            

        # Fiume
        with col_river:
            # HTML per barca e fiume centrati
            st.markdown(f"""
                <div style="text-align: center; margin-top: 5px; color: #1e81b0; font-size: 20px;">
                    {river_content}
                </div>
            """, unsafe_allow_html=True)


        # Contenuto Destro
        with col_r:
            st.caption(f"({M_R} M, {C_R} C)")
            st.markdown(f"<div style='text-align: center; font-size: 32px; min-height: 50px;'>{right_side_content}</div>", unsafe_allow_html=True)
        
        # Striscia di stato finale sotto le colonne
        st.markdown(f"""
            <div style="text-align: center; font-size: 14px; color: white; background-color: {status_color}; padding: 5px; border-radius: 5px; margin-top: 5px;">
                **{boat_info}**
            </div>
        """, unsafe_allow_html=True)


# ====================================================================
st.header("1. Sudoku 9x9 Solver (DFS/Backtracking)")
st.markdown("Algoritmo di ricerca in profondità (DFS) con **backtracking** per trovare la soluzione univoca. I numeri in **rosso scuro** sono i numeri iniziali.")

col_input, col_output = st.columns(2)

board_iniziale = np.array([
    [5,3,0, 0,7,0, 0,0,0], [6,0,0, 1,9,5, 0,0,0], [0,9,8, 0,0,0, 0,6,0],
    [8,0,0, 0,6,0, 0,0,3], [4,0,0, 8,0,3, 0,0,1], [7,0,0, 0,2,0, 0,0,6],
    [0,6,0, 0,0,0, 2,8,0], [0,0,0, 4,1,9, 0,0,5], [0,0,0, 0,8,0, 0,7,9]
], dtype=int)

with col_input:
    st.subheader("Board Iniziale:")
    render_sudoku_board(board_iniziale, board_iniziale)

with col_output:
    if st.button("Risolvi Sudoku"):
        board_risolta = board_iniziale.copy()
        if solve_sudoku(board_risolta):
            st.success("Soluzione Trovata!")
            st.subheader("Soluzione (numeri risolti in azzurro):")
            render_sudoku_board(board_risolta, board_iniziale)
            st.balloons()
        else:
            st.error("Nessuna soluzione trovata per questa board.")

st.markdown("---")


# ====================================================================
st.header("2. Solutore N-Regine (Backtracking)")
st.markdown("Trova tutte le possibili configurazioni uniche per posizionare 8 regine su una scacchiera 8x8. Utilizza la Ricerca in Profondità con **backtracking**.")

if st.button("Trova Soluzioni (N=8 Regine)"):
    solver = ReginaSolver(n=8)
    
    with st.spinner("Ricerca DFS in corso..."):
        soluzioni = list(solver.solve())
    
    st.success(f"Trovate **{solver.num_soluzioni}** soluzioni uniche.")
    
    num_to_display = min(3, len(soluzioni))
    st.subheader(f"Prime {num_to_display} Soluzioni:")
    
    cols = st.columns(num_to_display)

    for i in range(num_to_display):
        with cols[i]:
            st.caption(f"Soluzione #{i+1}")
            render_queens_board(soluzioni[i])

    if len(soluzioni) > num_to_display:
        st.markdown(f"*...e altre {len(soluzioni) - num_to_display} soluzioni sono state trovate in totale.*")

st.markdown("---")

# ====================================================================
st.header("3. Torri di Hanoi (Ricerca in Ampiezza - BFS)")
st.markdown("L'algoritmo **BFS** (Breadth-First Search) trova la sequenza di mosse **ottimale** (percorso minimo) per spostare **3 dischi** dalla Torre A alla Torre C, rispettando la regola che un disco non può mai essere posizionato sopra uno più piccolo.")

hanoi_solver = HanoiSolver(n_dischi=3)
initial_state = hanoi_solver.start

st.subheader("Stato Iniziale:")
render_hanoi_state(initial_state)

if st.button("Risolvi Torri di Hanoi (3 Dischi)"):
    
    with st.spinner("Ricerca BFS in corso..."):
        soluzione_hanoi = hanoi_solver.solve()

    if soluzione_hanoi:
        st.success(f"Soluzione Trovata in **{len(soluzione_hanoi) - 1} mosse** (ottimale):")
        
        with st.expander("Visualizza i passaggi passo per passo"):
            for i, state in enumerate(soluzione_hanoi):
                st.markdown(f"---")
                st.subheader(f"Passo {i}")
                render_hanoi_state(state, step_index=i)
            
    else:
        st.error("Nessuna soluzione trovata.")

st.markdown("---")

# ====================================================================
st.header("4. Missionari e Cannibali (Ricerca in Ampiezza - BFS)")
st.markdown("Algoritmo **BFS** per trovare il percorso più breve per spostare 3 Missionari (👨) e 3 Cannibali (😈) da sinistra a destra, assicurando che i Missionari non siano mai in inferiorità numerica su nessuna sponda.")

mc_solver = MissionaryCannibalSolver()
initial_mc_state = mc_solver.start

st.subheader("Stato Iniziale:")
render_missionary_cannibal_state(initial_mc_state)

if st.button("Risolvi Missionari e Cannibali"):
    
    with st.spinner("Ricerca BFS in corso..."):
        soluzione_mc = mc_solver.solve()

    if soluzione_mc:
        st.success(f"Soluzione Trovata in **{len(soluzione_mc) - 1} mosse** (ottimale):")
        
        with st.expander("Visualizza i passaggi passo per passo"):
            for i, state in enumerate(soluzione_mc):
                st.markdown(f"---")
                st.subheader(f"Passo {i}")
                render_missionary_cannibal_state(state)
            
    else:
        st.error("Nessuna soluzione trovata.")