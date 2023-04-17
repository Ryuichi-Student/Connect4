from pickle import load
import numpy as np
def check_data_from_file(filename):
    with open(filename, 'rb') as f:
        data = load(f)
    # helper(np.array([data[i][0] for i in range(0, len(data), 2)]))
    print([x for x in data[::2] if np.sum(x[0]) == 42])

def helper(inputs):
    current_player_boards = np.where(inputs == 1, 1, 0)
    opponent_boards = np.where(inputs == -1, 1, 0)
    inputs_reshaped = np.stack((current_player_boards, opponent_boards), axis=-1)
    print(current_player_boards[0])
    print(opponent_boards[0])
    print(inputs_reshaped[0])
    print(inputs_reshaped.shape)

if __name__ == '__main__':
    check_data_from_file('python_agent/data/data6.p')
    # check_data_from_file('python_agent/data/data4.p')