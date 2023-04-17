import numpy as np
import sys
sys.path.append('python_agent/src')
from game import Connect4State
from my_model import Connect4Model
from mcts import MCTS, MCTSNode
from parallel_mcts import ParallelMCTS, ParallelMCTSNode

def play_game(model, starting_player=1):
    state = Connect4State()
    state.current_player = starting_player
    mcts = MCTS(state, model, 100)
    # mcts = ParallelMCTS(state, model, 40)
    while not state.is_terminal():
        print(state.get_board())
        action_probs, value = mcts.get_action_probs(state, state.get_valid_moves())

        print(f"AI thinks the position is worth {value} to player {state.current_player}")
        if state.current_player == 1:
            try:
                action = int(input("Enter your move (0-6): "))
                if action not in state.get_valid_moves():
                    print("Invalid move!")
                    continue
            except ValueError:
                print("Invalid move!")
                continue
        else:
            mcts.run()
            print(action_probs, mcts.get_search_policy())
            print(mcts.root)
            print(mcts.root.children)
            action = mcts.get_best_move()
        state = state.simulate(action)
        mcts.set_root(action)


    print(state.get_result())
    print(state.get_board())
    if state.get_result() == 0:
        print("It's a draw!")
    elif state.current_player == 1:
        print("AI wins!")
    else:
        print("You win!")

def main():
    model = Connect4Model('python_agent/src/models/best_model')
    # model = Connect4Model()
    play_game(model)
    play_game(model, -1)

if __name__ == "__main__":
    main()
