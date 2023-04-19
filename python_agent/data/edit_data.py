from pickle import load, dump
from json import load as json_load
import numpy as np


def edit_data_from_file(filename, _function):
    with open(filename, 'rb') as f:
        data = load(f)
    data = _function(data)
    with open(filename, 'wb') as f:
        dump(data, f)


def flip_rewards(data):
    return [(data[i][0], data[i][1], -data[i][2]) for i in range(len(data))]

def flip_boards(data):
    return [(-data[i][0], data[i][1], data[i][2]) for i in range(len(data))]


def json_to_pickle(filename, out, _function=lambda x:x):
    with open(filename, 'r') as f:
        data = json_load(f)
    data = _function(data)
    with open(out, 'wb') as f:
        dump(data, f)


def correct_json(data):
    def string_to_np_array(string):
        array = [0 if x==" " else 1 if x=="X" else -1 for x in string][::-1]
        return np.flip(np.array(array).reshape((6, 7)), axis=1)
    data = [(string_to_np_array(data[i][0]), data[i][1], data[i][2]) for i in range(len(data))]
    return data


def get_symmetries(data):
    data.extend([(np.flip(data[i][0], axis=1), data[i][1][::-1], data[i][2]) for i in range(len(data))])
    return data


if __name__ == '__main__':
    for i in range(1,2):
        edit_data_from_file(f'python_agent/data/data{i}.p', flip_rewards)
    # edit_data_from_file('python_agent/data/data2.p', flip_rewards)
    # edit_data_from_file('python_agent/data/data1.p', flip_rewards)

    # edit_data_from_file('python_agent/data/data2.p', flip_boards)
    # edit_data_from_file('python_agent/data/data1.p', flip_boards)
    # json_to_pickle('python_agent/data/play_20230413-192855.473657.json', 'python_agent/data/data1.p', correct_json)
    # json_to_pickle('python_agent/data/play_20230413-202653.517773.json', 'python_agent/data/data2.p', correct_json)