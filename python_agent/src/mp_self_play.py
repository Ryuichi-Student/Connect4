# Note: There are a lot of micro-optimizations not present, but this is left for readability and ease of use.

import os
import numpy as np
from my_model import Connect4Model
from mcts import MCTS
from game import Connect4State
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto
import multiprocessing as mp
import pickle
from sys import getsizeof
import gc
from parallel_mcts import ParallelMCTS
from tensorflow.keras.backend import clear_session

# Use to profile code.
# import cProfile
# from pstats import Stats
# pr = cProfile.Profile()
# pr.enable()
# pr.disable()
# stats = Stats(pr)
# stats.sort_stats('tottime').print_stats(10)

__config = {
    "load_parameters": {
        # If true, load all data from REUSE_DATA_FROM to START_ITERATION and train on that.
        "TRAIN_FROM_PREVIOUS": False,
        "REUSE_DATA_FROM": 1,
        # If true, load the previous best model.
        "LOAD_FROM_PREVIOUS": True,
        "START_ITERATION": 20,
        "LAST_ITERATION": 101,
        "MODEL_PATH": "python_agent/src/models/checkpoint",
        "BEST_MODEL_PATH": "python_agent/src/models/best_model",
    },
    "self_play_parameters": {
        "num_simulations": 200,
        # Total number of games is num_processes * mp_num_games * episodes_per_process
        "mp_num_games": 3,
        "num_processes": 4,
        "episodes_per_process": 2,
    },
    "evaluation_parameters": {
        "win_rate_threshold": 0.7,
        "evaluation_interval": 3,
        "num_evaluation_games": 9,
        # Number of processes to use to pit the models against each other. A good hallmark is num_processes/2.
        "pit_processes": 3
    },
    "training_parameters": {
        "renew_data_interval": 5,
        "epochs": 60,  # Number of training epochs
        "batch_size": 64,
        # The worst thing to do is to have a learning_rate too high.
        # For example, a lr of 0.01 causes the model to converge to predicting every board value as 1.
        "learning_rate": 0.0001,
    }
}

def _play_game(state, mcts):
    # Make sure the state and mcts are reset, though they should already be.
    mcts.reset(state)

    game_data = []
    length = 0
    while not state.is_terminal():
        length += 1
        # Run num_simulations simulations.
        mcts.run()
        actions = state.get_valid_moves()

        # The value prediction is the model's idea of how likely it is for the CURRENT player to win.
        action_probs, value = mcts.get_action_probs(state, actions)

        # Pick an action based on the search policy.
        search_policy = mcts.get_search_policy()
        action = np.random.choice(actions, p=search_policy)

        mcts.set_root(action)
        state = state.simulate(action)

        game_data.append((state.get_board(), search_policy, value))

        # This should never be called. This will help debugging.
        if length > 42:
            raise Exception("Game is too long. Something went wrong.")
    return state, mcts, game_data, length


def get_reward(result, length, value):
    # Reward longer losses and shorter wins.
    CONSTANTS = {"result": 0.5, "value": 0.5, "length": 0.4/42}
    length_factor = length * CONSTANTS["length"]
    # Use a linear combination of result and value
    reward = result * (1-length_factor) * CONSTANTS["result"] + value * CONSTANTS["value"]
    return reward


# If you do not want multiprocessing, just make a copy of episode() and remove the multiprocessing code.
def episode(args):
    # Use multiple threads? Slightly faster but not nearly as much as processes.
    PARALLELIZE_MCTS = True

    model_weights, num_simulations, num_games = args
    model = Connect4Model()
    model.load_weights(model_weights)
    state = Connect4State()

    data = []

    for _ in tqdm_auto(range(num_games)):
        # Catches any errors which would otherwise cause the program to simply hang.
        try:
            if PARALLELIZE_MCTS:
                mcts = ParallelMCTS(state, model, num_simulations, num_threads=3)
            else:
                mcts = MCTS(state, model, num_simulations)

            state, mcts, game_data, length = _play_game(state, mcts)

            # The person who made the last move wins.
            # We want the model to predict that the last position is a win for the current player.
            result = state.has_winner()
            for board, policy, value in game_data[::-1]:
                reward = get_reward(result, length, value)
                data.append((board, policy, reward))
                data.append((np.flip(board, axis=1), policy[::-1], reward))
                result = -result

            # SUPER IMPORTANT! Without this, RAM usage skyrockets.
            clear_session()
            gc.collect()
        except Exception as e:
            print("ERROR OCCURED. SKIPPING GAME.")
            print("Exception called was ", e)
            pass

    print(f"Data is of size {getsizeof(data)}")
    return data


# With 6 processes, this offers a 5x speedup on my Macbook. Almost linear scaling!
def mp_self_play(model, num_simulations, num_games, num_processes, episodes_per_process, iteration):
    # Save the model weights to a temporary file so that each process can load it.
    model_weights = "temp_weights.h5"
    model.save_weights(model_weights)

    data = []
    for i in range(num_games):
        print(f"Starting batch {i}")
        with mp.Pool(processes=num_processes) as pool:
            total_episodes = num_processes
            results = list(tqdm(pool.imap(episode,
                                          [(model_weights, num_simulations, episodes_per_process) for _ in range(num_processes)]
                                          )
                                , total=total_episodes))

            # Collect the data from all processes
            for res in results:
                data.extend(res)

    os.remove(model_weights)

    print("pickling data")
    pickle.dump(data, open(f"python_agent/data/data{iteration}.p", "wb" ))
    return data


def train(model, data, epochs, batch_size, learning_rate=None):
    # data is of the form (board, action_probs, result)
    boards, action_probs, results = zip(*data)
    boards = np.array(boards, dtype=np.float32)

    # The search policies are the "better" action_probs
    action_probs = np.array(action_probs, dtype=np.float32)
    results = np.array(results, dtype=np.float32)

    model.fit(boards, [action_probs, results], epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)


# A game between two models.
def versus(args):
    model1_weights, model2_weights, num_simulations = args

    model1 = Connect4Model()
    model1.load_weights(model1_weights)
    model2 = Connect4Model()
    model2.load_weights(model2_weights)
    state = Connect4State()

    PARALLELIZE_MCTS = False
    if PARALLELIZE_MCTS:
        mcts1 = ParallelMCTS(state, model1, num_simulations, num_threads=2)
        mcts2 = ParallelMCTS(state, model2, num_simulations, num_threads=2)
    else:
        mcts1 = MCTS(state, model1, num_simulations)
        mcts2 = MCTS(state, model2, num_simulations)

    length = 0

    cur, opp = mcts1, mcts2
    while not state.is_terminal():
        cur.run()
        action = cur.get_best_move()
        mcts1.set_root(action)
        mcts2.set_root(action)
        state = state.simulate(action)
        cur, opp = opp, cur
        if length > 42:
            raise Exception("Something went wrong. Game is too long.")

    return state.get_result()


def pit(model1, model2, num_evaluation_games, num_processes):
    wins, draws, losses = 0, 0, 0

    model1_weights = "temp_weights1.h5"
    model2_weights = "temp_weights2.h5"
    model1.save_weights(model1_weights)
    model2.save_weights(model2_weights)

    with mp.Pool(processes=num_processes) as pool:
        winners = list(tqdm(pool.imap(versus, [(model1_weights, model2_weights, 10)
                                                  for _ in range(num_evaluation_games//2)]), total=num_evaluation_games//2))
        losers = list(tqdm(pool.imap(versus, [(model2_weights, model1_weights, 10)
                                                 for _ in range(num_evaluation_games//2)]), total=num_evaluation_games//2))

    for winner in winners:
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1
    for loser in losers:
        if loser == 1:
            losses += 1
        elif loser == -1:
            wins += 1
        else:
            draws += 1

    win_rate = (wins + draws / 2) / num_evaluation_games
    print(f"Win rate: {win_rate}, Wins: {wins}, Draws: {draws}, Losses: {losses}")
    clear_session()
    gc.collect()
    return win_rate


def main():
    load_config = __config["load_parameters"]
    self_play_config = __config["self_play_parameters"]
    train_config = __config["training_parameters"]
    evaluation_config = __config["evaluation_parameters"]

    model = Connect4Model()
    best_model = Connect4Model()
    model_path = load_config["MODEL_PATH"]
    best_model_path = load_config["BEST_MODEL_PATH"]

    old_data = []

    if load_config["LOAD_FROM_PREVIOUS"]:
        model.load_weights(best_model_path)
        best_model.load_weights(best_model_path)

    if load_config["TRAIN_FROM_PREVIOUS"]:
        previous_data = []
        for i in range(load_config["REUSE_DATA_FROM"], load_config["START_ITERATION"]):
            with open(f"python_agent/data/data{i}.p", "rb" ) as tmp:
                previous_data.extend(pickle.load(tmp))

        train(model, previous_data, epochs=train_config["epochs"],
              batch_size=train_config["batch_size"], learning_rate=train_config["learning_rate"])
        model.save_weights(model_path)

        win_rate = pit(model, best_model, 10)

        if win_rate > evaluation_config["win_rate_threshold"]:
            print("New model is better, updating best model.")
            best_model.load_weights(model_path)
            best_model.save_weights(best_model_path)

    for iteration in range(load_config["START_ITERATION"], load_config["LAST_ITERATION"]):
        print(f"Training iteration {iteration}")

        data = mp_self_play(best_model, self_play_config["num_simulations"],
                            self_play_config["mp_num_games"], self_play_config["num_processes"],
                            self_play_config["episodes_per_process"], iteration)
        old_data.extend(data)

        train(model, old_data, epochs=train_config["epochs"],
              batch_size=train_config["batch_size"], learning_rate=train_config["learning_rate"])

        if iteration % train_config["renew_data_interval"] == 0:
            old_data = []

        if iteration % evaluation_config["evaluation_interval"] == 0:
            model.save_weights(model_path)
            win_rate = pit(model, best_model,
                           evaluation_config["num_evaluation_games"], evaluation_config["pit_processes"])

            if win_rate > evaluation_config["win_rate_threshold"]:
                print("New model is better, updating best model.")
                best_model.load_weights(model_path)
                best_model.save_weights(best_model_path)


if __name__ == "__main__":
    main()