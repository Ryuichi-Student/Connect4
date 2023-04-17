import numpy as np
from my_model import Connect4Model
from mcts import MCTS, MCTSNode
from game import Connect4State
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto
from time import sleep
import os


def self_play(model, num_simulations, num_games):
    data = []
    state = Connect4State()

    for _ in tqdm_auto(range(num_games), position=0):
        game_data = []
        state.reset()
        mcts = MCTS(MCTSNode(state), model, num_simulations)

        while not state.is_terminal():
            mcts.run()
            actions = state.get_valid_moves()
            action_probs = mcts.get_action_probs(state, actions)
            action = np.random.choice(actions, p=action_probs)
            game_data.append((state.get_board(), action_probs, state.current_player))
            game_data.append((state.get_board(), action_probs[::-1], state.current_player))
            state = state.simulate(action)

        result = state.get_result()

        data.extend([(board, action_probs, 0) for board, action_probs, player in game_data[:2]])
        data.extend([(board, action_probs, result*player) for board, action_probs, player in game_data[2:]])

        # print(result)
        # print(state.get_board())
        # print("\n".join(str(x) for x in data))
        # sleep(30)

    return data

def train(model, data, epochs, batch_size):
    boards, action_probs, results = zip(*data)
    boards = np.array(boards, dtype=np.float32)
    action_probs = np.array(action_probs, dtype=np.float32)
    results = np.array(results, dtype=np.float32)

    # Add an extra dimension for the channel
    boards = np.expand_dims(boards, axis=-1)

    model.fit([boards], [action_probs, results], epochs=epochs, batch_size=batch_size)


def play_game(model1, model2, num_simulations):
    state = Connect4State()
    mcts1 = MCTS(MCTSNode(state), model1, num_simulations)
    mcts2 = MCTS(MCTSNode(state), model2, num_simulations)
    p1 = True
    while not state.is_terminal():
        actions = state.get_valid_moves()
        if p1:
            mcts1.run()
            action_probs = mcts1.get_action_probs(state, actions)
        else:
            mcts2.run()
            action_probs = mcts2.get_action_probs(state, actions)

        action = actions[np.argmax(action_probs)]
        state = state.simulate(action)
        p1 = not p1
    return state.get_result()


def main():
    num_simulations = 30
    num_games = 100
    epochs = 32
    batch_size = 16
    evaluation_interval = 1  # Evaluate every iteration
    num_evaluation_games = 40
    win_rate_threshold = 0.55

    model = Connect4Model()
    best_model = Connect4Model()
    model_path = "old_models/checkpoint"
    best_model_path = "models/best_model"

    data = []

    if os.path.exists(best_model_path):
        best_model.load_weights(best_model_path)

    for iteration in range(1, 1001):
        print(f"Training iteration {iteration}")

        data = self_play(best_model, num_simulations, num_games)
        print(len(data))
        train(model, data, epochs=epochs, batch_size=batch_size)

        if iteration % evaluation_interval == 0:
            model.save_weights(model_path)
            wins, draws, losses = 0, 0, 0

            for _ in range(num_evaluation_games // 2):
                winner = play_game(model, best_model, num_simulations)
                if winner == 1:
                    wins += 1
                elif winner == -1:
                    losses += 1
                else:
                    draws += 1

                winner = play_game(best_model, model, num_simulations)
                if winner == 1:
                    losses += 1
                elif winner == -1:
                    wins += 1
                else:
                    draws += 1

            win_rate = (wins+draws/2) / num_evaluation_games
            print(f"Win rate: {win_rate}, Wins: {wins}, Draws: {draws}, Losses: {losses}")

            if win_rate > win_rate_threshold:
                print("New model is better, updating best model.")
                best_model.load_weights(model_path)
                best_model.save_weights(best_model_path)


if __name__ == "__main__":
    main()