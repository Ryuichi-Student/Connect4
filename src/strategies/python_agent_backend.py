from flask import Flask, request, jsonify
import numpy as np

import sys
sys.path.append('python_agent/src')
from game import Connect4State
from my_model import Connect4Model
from mcts import MCTS, MCTSNode
from parallel_mcts import ParallelMCTS, ParallelMCTSNode
from flask_cors import CORS


model = Connect4Model('python_agent/src/models/best_model')
state = Connect4State()
# mcts = MCTS(state, model, 200)
mcts = ParallelMCTS(state, model, 100, num_threads=8)


app = Flask(__name__)
CORS(app)


@app.route('/ai/update_state', methods=['POST'])
def update_state():
    global mcts, state
    player_move = request.json['move']
    if player_move == -1:
        state = Connect4State()
        mcts.reset(state)
        return jsonify({'status': 'ok'})
    state = state.simulate(player_move)
    mcts.set_root(player_move)

    return jsonify({'status': 'ok'})


@app.route('/ai/move', methods=['POST'])
def get_move():
    global mcts, state

    # action_probs, value = mcts.get_action_probs(state, state.get_valid_moves())
    mcts.run()
    move = mcts.get_best_move()
    mcts.set_root(move)
    state = state.simulate(move)

    return jsonify({'move': move})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
