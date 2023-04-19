from pickle import load
import numpy as np
def check_data_from_file(filename):
    with open(filename, 'rb') as f:
        data = load(f)
    # helper(np.array([data[i][0] for i in range(0, len(data), 2)]))
    # print(data[::-2])
    find_draws(data[::-2])

def helper(inputs):
    current_player_boards = np.where(inputs == 1, 1, 0)
    opponent_boards = np.where(inputs == -1, 1, 0)
    inputs_reshaped = np.stack((current_player_boards, opponent_boards), axis=-1)
    print(current_player_boards[0])
    print(opponent_boards[0])
    print(inputs_reshaped[0])
    print(inputs_reshaped.shape)

def check_repeats(data):
    first_few_states = [data[0][0], data[1][0], data[2][0]]
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            if np.array_equal(data[i][0], data[j][0]) and not any(np.array_equal(data[i][0], x)
                                                                  for x in first_few_states):
                print(i, j)

def find_draws(data):
    for i in range(len(data)):
        if np.sum(abs(data[i][0])) == 41:
            print(i)
            print(data[i])


if __name__ == '__main__':
    for i in range(1, 11):
        print(i)
        check_data_from_file(f'python_agent/data/data{i}.p')
    # check_data_from_file('python_agent/data/data6.p')
    # check_data_from_file('python_agent/data/data4.p')