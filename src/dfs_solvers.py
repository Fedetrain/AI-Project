# src/dfs_solvers.py
import numpy as np
import copy

# --- Logica Sudoku (DFS/Backtracking) ---

def find_empty(board):
    res = np.where(board == 0)
    if len(res[0]) == 0:
        return None
    return res[0][0], res[1][0]

def is_valid(board, num, row, col):
    # Riga
    if num in board[row, :]:
        return False
    # Colonna
    if num in board[:, col]:
        return False
    # Blocco 3×3
    br = (row // 3) * 3
    bc = (col // 3) * 3
    block = board[br:br+3, bc:bc+3]
    if num in block:
        return False
    return True

def solve_sudoku(board):
    """
    Risolve la board del sudoku in-place.
    Ritorna True se una soluzione è trovata, False altrimenti.
    """
    empty = find_empty(board)
    if not empty:
        return True  # Risolto
    row, col = empty

    for num in range(1, 10):
        if is_valid(board, num, row, col):
            board[row, col] = num
            if solve_sudoku(board):
                return True
            board[row, col] = 0  # Backtrack
    return False

# --- Logica N-Regine (DFS/Backtracking) ---

class ReginaSolver:
    def __init__(self, n=8):
        self.n = n
        self.num_soluzioni = 0
        self.soluzioni = []

    def _is_valid(self, state):
        # Controlla attacchi orizzontali e diagonali
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if state[i] == state[j] or abs(state[i] - state[j]) == j - i:
                    return False
        return True

    def _next_state(self, state):
        # Aggiunge la prossima regina in una colonna non usata
        for i in range(self.n): 
            if i not in state:
                state_copy = copy.deepcopy(state)
                state_copy.append(i)
                yield state_copy

    def _dfs(self, state):
        if len(state) == self.n:
            self.num_soluzioni += 1
            self.soluzioni.append(state)
            yield state
        
        for new_state in self._next_state(state):
            if self._is_valid(new_state):
                yield from self._dfs(new_state)

    def solve(self):
        """Generatore che produce tutte le soluzioni."""
        initial_state = []
        return self._dfs(initial_state)

# --- NUOVA LOGICA: Torri di Hanoi (BFS) ---

class HanoiSolver: 
    """
    Risolve il problema delle Torri di Hanoi (3 dischi) usando la Ricerca in Ampiezza (BFS) 
    per trovare il percorso ottimo (minimo numero di mosse).
    """
    def __init__(self, n_dischi=3):
        # Creiamo i dischi in ordine decrescente: [N-1, N-2, ..., 0]
        self.n_dischi = n_dischi
        disks = list(range(n_dischi - 1, -1, -1)) 
        self.start = [disks, [], []]
        self.goal = [[], [], disks]
        self.frontier = [[self.start]] # La frontiera contiene percorsi
        self.visited = {str(self.start)} # Set per tracciare gli stati visitati

    def _is_move_legal(self, source_peg, dest_peg):
        """Controlla se il disco in cima a source_peg può essere spostato in dest_peg."""
        if not source_peg:
            return False 
        
        disk_to_move = source_peg[-1]
        
        if not dest_peg:
            return True # Qualsiasi disco può essere messo su una torre vuota
        
        top_disk_dest = dest_peg[-1]
        
        return disk_to_move < top_disk_dest # Il disco mosso deve essere PIÙ PICCOLO di quello di destinazione

    def next_states(self, state):
        for i in range(3):  # Torre di partenza (Source)
            for k in range(3):  # Torre di destinazione (Dest)
                if i != k:
                    
                    if self._is_move_legal(state[i], state[k]):
                        
                        new_state = copy.deepcopy(state)
                        disco = new_state[i].pop() 
                        new_state[k].append(disco)
                        
                        state_str = str(new_state)
                        if state_str not in self.visited:
                            self.visited.add(state_str)
                            yield new_state

    def solve(self):
        """Esegue la ricerca in ampiezza e ritorna il percorso ottimale."""
        while self.frontier:
            path = self.frontier.pop(0)  
            current_state = path[-1] 
            
            if current_state == self.goal:  
                return path 
            
            # Genera i prossimi stati validi
            for nuovo_stato in self.next_states(current_state):
                self.frontier.append(path + [nuovo_stato])
        
        return None 


# --- NUOVA LOGICA: Missionari e Cannibali (BFS) ---

class MissionaryCannibalSolver:
    """
    Risolve il problema Missionari e Cannibali (3/3) usando la Ricerca in Ampiezza (BFS) 
    per trovare il percorso ottimo (minimo numero di mosse).
    """
    def __init__(self):
        # Stato: {"L": {"missionari": N, "cannibali": N}, "R": {...}, "boat": "L"|"R"}
        self.start = {"L": {"missionari": 3, "cannibali": 3},
                      "R": {"missionari": 0, "cannibali": 0},
                      "boat": "L"}
        self.goal = {"L": {"missionari": 0, "cannibali": 0},
                     "R": {"missionari": 3, "cannibali": 3},
                     "boat": "R"}
        
        # Mosse legali: (1M), (1C), (2M), (2C), (1M 1C)
        self.mosse = [{'missionari': i, 'cannibali': j} for i in range(3) for j in range(3) if i + j > 0 and i + j <= 2]
        self.frontier = [[self.start]]
        self.visited = {str(self.start)} 

    def _is_valid(self, stato):
        """Verifica se lo stato rispetta la regola M >= C (o M=0) su entrambe le sponde."""
        M_L, C_L = stato["L"]["missionari"], stato["L"]["cannibali"]
        M_R, C_R = stato["R"]["missionari"], stato["R"]["cannibali"]
        
        # Missionari <= Cannibali solo se Missionari == 0
        valid_L = (M_L >= C_L or M_L == 0) and (M_L >= 0 and C_L >= 0 and M_L <= 3 and C_L <= 3)
        valid_R = (M_R >= C_R or M_R == 0) and (M_R >= 0 and C_R >= 0 and M_R <= 3 and C_R <= 3)
        
        return valid_L and valid_R

    def next_states(self, current_state):
        posizione_barca = current_state["boat"]
        
        for mossa in self.mosse:
            
            source = posizione_barca
            dest = "R" if source == "L" else "L"

            # 1. Controlla se ci sono abbastanza persone per la mossa
            enough_miss = current_state[source]['missionari'] >= mossa['missionari']
            enough_cann = current_state[source]['cannibali'] >= mossa['cannibali']
            
            if enough_miss and enough_cann:
                
                new_state = copy.deepcopy(current_state)

                # 2. Esegui la mossa
                new_state[source]["cannibali"] -= mossa["cannibali"]
                new_state[source]["missionari"] -= mossa["missionari"]
                new_state[dest]["cannibali"] += mossa["cannibali"]
                new_state[dest]["missionari"] += mossa["missionari"]
                new_state["boat"] = dest
                
                # 3. Verifica validità e non visitato
                state_str = str(new_state)
                if self._is_valid(new_state) and state_str not in self.visited:
                    self.visited.add(state_str)
                    yield new_state

    def solve(self):
        """Esegue la ricerca in ampiezza e ritorna il percorso ottimale."""
        while self.frontier:
            percorso = self.frontier.pop(0)
            
            if percorso[-1] == self.goal:
                return percorso

            for nuovo_stato in self.next_states(percorso[-1]):
                self.frontier.append(percorso + [nuovo_stato])

        return None