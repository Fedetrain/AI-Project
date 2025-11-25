# main.py
import pygame
import sys
import os
from game import Game
from config import STOCKFISH_PATH

def main():
    """Funzione principale per avviare il gioco."""
    
    print("="*60)
    print(f"Avvio Chess Analyzer...")
    print(f"Percorso Stockfish rilevato: '{STOCKFISH_PATH}'")
    
    if not os.path.exists(STOCKFISH_PATH):
        print("⚠️ ATTENZIONE: Stockfish non trovato nel percorso specificato.")
        print("Il gioco si avvierà, ma l'analisi non funzionerà.")
        print("Per risolvere: crea una cartella 'engines' e mettici l'eseguibile di Stockfish.")
    print("="*60)
        
    try:
        game = Game()
        game.run()
    except KeyboardInterrupt:
        print("\nChiusura richiesta dall'utente.")
    except Exception as e:
        print(f"\n🛑 ERRORE CRITICO: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()