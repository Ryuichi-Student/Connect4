import os
import numpy as np
from my_model import Connect4Model
from mcts import MCTS, MCTSNode
from game import Connect4State
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto
import multiprocessing as mp
import pickle
from sys import getsizeof
import gc
from parallel_mcts import ParallelMCTSNode, ParallelMCTS
from tensorflow.keras.backend import clear_session
import cProfile
from pstats import Stats


# from tensorflow.python.client import device_lib
#
# def get_available_devices():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU' or x.device_type == 'CPU']
#
# print(get_available_devices())
# exit(0)


# 9mins per game for 1 process.
# 3mins per game for 5 processes.
# With just parallelizing MCTS, 1 process is 6mins per game.


# obj file size function from stackoverflow.
def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0
    gc.collect()

    while obj_q:
        sz += sum(map(getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())


    return sz


def _play_game(state, mcts):
    state.reset()
    game_data = []
    length = 0
    while not state.is_terminal():
        length += 1
        mcts.run()
        actions = state.get_valid_moves()
        action_probs, value = mcts.get_action_probs(state, actions)
        search_policy = mcts.get_search_policy()
        action = np.random.choice(actions, p=search_policy)
        mcts.set_root(action)
        game_data.append((state.get_board(), search_policy, value))
        state = state.simulate(action)
        if length > 42:
            raise Exception("Game is too long. Something went wrong.")
    return state, mcts, game_data, length


def episode(args):
    PARALLELIZE_MCTS = False
    model_weights, num_simulations, num_games = args
    model = Connect4Model()
    model.load_weights(model_weights)
    data = []
    state = Connect4State()


    for _ in tqdm_auto(range(num_games)):
        # Ignore errors because of memory leaks.
        try:
            if PARALLELIZE_MCTS:
                mcts = ParallelMCTS(state, model, num_simulations, num_threads=3)
            else:
                mcts = MCTS(state, model, num_simulations)
            # Doubles down as a garbage collector
            # print(f"mcts is of size {get_obj_size(mcts)}")

            state, mcts, game_data, length = _play_game(state, mcts)

            # The person who made the last move wins.
            # We want the model to predict that the last position is a win for the current player.
            result = state.has_winner()
            for board, policy, value in game_data[::-1]:
                # TODO: Average v and q
                data.append((board, policy, result))
                data.append((np.flip(board, axis=1), policy[::-1], result))
                result = -result


            # print(f"Data is of size: {get_obj_size(game_data)}")

            # Reward longer losses and shorter wins.
            # length_factor = length * 0.4/42
            # rewards = [result * (1-length_factor) for result in results]
            # data.extend([(board, action_probs, 0) for board, action_probs, value, player in game_data[:2]])

            # SUPER IMPORTANT! This is what was causing the RAM usage to skyrocket.
            clear_session()
            gc.collect()
        except Exception as e:
            print("ERROR OCCURED. SKIPPING GAME.")
            print("Exception called was ", e)
            pass
    # print(result)
    # print(state.get_board())
    # print("\n".join(str(x) for x in data))
    # sleep(30)
    print(getsizeof(data))
    return data


def mp_self_play(model, num_simulations, num_games, num_processes, episodes_per_process, iteration):
    # With 6 processes, this offers a 5x speedup. Almost linear scaling!
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


def self_play(model, num_simulations, num_games, iteration):
    # Non multiprocessing version
    PARALLELIZE_MCTS = False
    data = []
    state = Connect4State()

    for _ in tqdm(range(num_games)):
        state.reset()
        if PARALLELIZE_MCTS:
            mcts = ParallelMCTS(state, model, num_simulations)
        else:
            mcts = MCTS(state, model, num_simulations)
        # Doubles down as a garbage collector
        print(f"mcts is of size {get_obj_size(mcts)}")
        state, mcts, game_data, length = _play_game(state, mcts)

        result = state.get_result()
        print(f"Data is of size: {get_obj_size(game_data)}")

        # If the result*player is -1, the player lost so the value should be *-1.
        results = [-result * player for _, _, value, player in game_data]
        length_factor = length * 0.4 / 42
        rewards = [result * (1 - length_factor) for result in results]
        # data.extend([(board, action_probs, 0) for board, action_probs, value, player in game_data[:2]])

        # TODO: Find how to average v and q
        data.extend(
            [(board, action_probs, reward) for (board, action_probs, value, player), reward in zip(game_data, rewards)])
        clear_session()
    # save data in pickle format
    pickle.dump(data, open(f"python_agent/data/data{iteration}.p", "wb" ))

    return data


def train(model, data, epochs, batch_size, learning_rate=None):
    boards, action_probs, results = zip(*data)
    boards = np.array(boards, dtype=np.float32)
    action_probs = np.array(action_probs, dtype=np.float32)
    results = np.array(results, dtype=np.float32)

    model.fit(boards, [action_probs, results], epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)


def play_game(args):
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
    p1 = True
    length = 0
    while not state.is_terminal():
        if p1:
            mcts1.run()
            action = mcts1.get_best_move()
        else:
            mcts2.run()
            action = mcts2.get_best_move()
        mcts1.set_root(action)
        mcts2.set_root(action)
        # action = actions[np.argmax(action_probs)]
        state = state.simulate(action)
        p1 = not p1
        if length > 42:
            raise Exception("Something went wrong. Game is too long.")
    return state.get_result()


def pit(model1, model2, num_evaluation_games):
    wins, draws, losses = 0, 0, 0

    model1_weights = "temp_weights1.h5"
    model2_weights = "temp_weights2.h5"
    model1.save_weights(model1_weights)
    model2.save_weights(model2_weights)

    with mp.Pool(processes=4) as pool:
        winners = list(tqdm(pool.imap(play_game, [(model1_weights, model2_weights, 10)
                                                  for _ in range(num_evaluation_games//2)]), total=num_evaluation_games//2))
        losers = list(tqdm(pool.imap(play_game, [(model2_weights, model1_weights, 10)
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
    TRAIN_FROM_PREVIOUS = False
    LOAD_FROM_PREVIOUS = True
    START_ITERATION = 19
    REUSE_DATA_FROM = 19

    # 10-15 mins per game, per process with 150 simulations per move.
    num_simulations = 200
    mp_num_games = 2
    # Would go for 8 but that really taxes the Macbook, which starts heating up.
    # 6 just keeps fans turned on but doesn't heat up.
    num_processes = 4
    episodes_per_process = 2
    # Let's not waste the precious data we spent so long collecting
    epochs = 60
    batch_size = 64
    evaluation_interval = 3  # Evaluate every 3 iterations
    num_evaluation_games = 8
    win_rate_threshold = 0.7
    num_games = 1
    renew_data_interval = 5

    model = Connect4Model()
    best_model = Connect4Model()
    model_path = "python_agent/src/models/checkpoint"
    best_model_path = "python_agent/src/models/best_model"

    old_data = []

    if LOAD_FROM_PREVIOUS:
        model.load_weights(best_model_path)
        best_model.load_weights(best_model_path)

    if START_ITERATION > 1:
        previous_data = []
        for i in range(REUSE_DATA_FROM, START_ITERATION):
            with open(f"python_agent/data/data{i}.p", "rb" ) as tmp:
                previous_data.extend(pickle.load(tmp))
        old_data.extend(previous_data)
        if TRAIN_FROM_PREVIOUS:
            # train(model,model previous_data, epochs=epochs//2, batch_size=batch_size, learning_rate=0.02)
            train(model, previous_data, epochs=50, batch_size=batch_size, learning_rate=0.0001)
            model.save_weights(model_path)

            win_rate = pit(model, best_model, 10)
            print("win rate: ", win_rate)

            if win_rate > win_rate_threshold:
                print("New model is better, updating best model.")
                best_model.load_weights(model_path)
                best_model.save_weights(best_model_path)
            # else:
            #     exit(0)
    for iteration in range(START_ITERATION, 101):
        num_simulations += 1
        print(f"Training iteration {iteration}")
        # Total number of games is num_processes * mp_num_games * episodes_per_process
        # pr = cProfile.Profile()
        # pr.enable()
        data = mp_self_play(best_model, num_simulations, mp_num_games, num_processes, episodes_per_process, iteration)
        # data = self_play(best_model, num_simulations, num_games, iteration)
        # pr.disable()
        # stats = Stats(pr)
        # stats.sort_stats('tottime').print_stats(10)
        # exit(0)
        old_data.extend(data)

        train(model, old_data, epochs=epochs, batch_size=batch_size, learning_rate=0.0001)

        if iteration % renew_data_interval == 0:
            old_data = []

        if iteration % evaluation_interval == 0:
            model.save_weights(model_path)
            win_rate = pit(model, best_model, num_evaluation_games)

            if win_rate > win_rate_threshold:
                print("New model is better, updating best model.")
                best_model.load_weights(model_path)
                best_model.save_weights(best_model_path)



if __name__ == "__main__":
    main()