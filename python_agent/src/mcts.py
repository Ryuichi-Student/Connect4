import cProfile
from pstats import Stats

import numpy as np
from numba import njit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from time import sleep

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

@njit
def action_probs_numba_helper(actions, policy_probs):
    action_probs = np.zeros(len(actions))
    for i, action in enumerate(actions):
        if action != -1:
            action_probs[i] = policy_probs[action]
        else:
            actions[i] = 0
    action_probs /= np.sum(action_probs)
    return action_probs

import numpy as np
from game import Connect4State

class MCTSNode:
    def __init__(self, state, mcts, P=1.0, parent=None, action=None):
        self.state = state
        self.P = P
        self.parent = parent
        self.action = action
        self.W = 0
        self.N = 0
        self.Q = 0
        self.children = []
        self.valid_actions = state.get_valid_moves()
        self.untried_actions = [state for state in self.valid_actions if state != -1]

        # Get prior probabilities for the actions from the model
        self.prior_probs = None
        self.mcts = mcts
        self.cached_ucb_score = None

    def select(self, c_param):
        if self.is_fully_expanded():
            selected_child = self.best_child(c_param)
            return selected_child
        elif len(self.children) == 0:
            self.expand_all()
        else:
            raise Exception("This should never happen")

    def expand_all(self):
        if self.prior_probs is None:
            self.prior_probs, _ = self.mcts.get_action_probs(self.state, self.valid_actions)

        for action in self.untried_actions.copy():
            next_state = self.state.simulate(action)
            P = self.prior_probs[action]
            self.expand(action, next_state, P)

    def get_ucb_score(self, c_param):
        if self.cached_ucb_score is None:
            if self.N == 0 or self.parent.N == 0:
                self.cached_ucb_score = float('inf')
            else:
                Q = self.W / self.N
                u = (self.P * np.sqrt(self.parent.N)) / (1 + self.N)
                self.cached_ucb_score = Q + c_param * u
        return self.cached_ucb_score

    def backpropagate(self, value):
        current = self
        while current is not None:
            current.N += 1
            current.W += value
            current.cached_ucb_score = None
            current = current.parent
            value = -value

    def expand(self, action, next_state, P):
        child = MCTSNode(next_state, self.mcts, P, parent=self, action=action)
        self.untried_actions.remove(action)
        self.children.append(child)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param):
        if len(self.children) == 0:
            return None
        scores = [(child.get_ucb_score(c_param), child.action, child) for child in self.children]
        best = max(scores)[0]
        best_children = [child for score, _, child in scores if score == best]
        return best_children[np.random.randint(0, len(best_children))]

    def __repr__(self):
        return f"Action: {self.action}, N: {self.N}, W: {self.W}, Q: {self.W/self.N}, P: {self.P}\n"


class MCTS:
    def __init__(self, state, model, num_simulations, c_param=4):
        self.model = model
        self.num_simulations = num_simulations
        self.c_param = c_param
        if isinstance(state, MCTSNode):
            self.root = state
        else:
            self.root = MCTSNode(state, self)
        self.depth_limit = 6

    def run(self):
        for _ in range(self.num_simulations):
            self.run_simulation()

    def get_action_probs(self, state, actions):
        board = state.get_board()
        input_tensor = np.array([board], dtype=np.float32)
        action_probs, value = self.model.predict(input_tensor)

        action_probs = action_probs_numba_helper(actions, action_probs)
        return action_probs, value

    def run_simulation(self):
        node = self.root
        while not node.state.is_terminal():
            child = node.select(self.c_param)
            if child is None:
                break
            node = child

        state = node.state
        if state.is_terminal():
            # print("Terminal state")
            # print(state.get_board())
            # Suppose this is a direct child of the root.
            # If there is a win, we want this child to have a higher value.
            value = state.has_winner()

            # print(np.sum(state.get_board()), root_player, state.current_player)

        else:
            # Suppose this is a direct child of the root.
            # If this is winning for the current player of this state,
            # we want to pick the lowest value possible.
            _, value = self.model.predict(state.get_board())
            value = -value

        # print(state.get_board(), value, np.sum(state.get_board()))

        node.backpropagate(value)

    def get_best_move(self):
        best_child = max(self.root.children, key=lambda child: child.N)
        return best_child.action

    def reset(self, state):
        self.root = MCTSNode(state, self)
        state.reset()

    def get_search_policy(self, temperature=1.1):
        root_node = self.root
        visit_counts = np.array([child.N for child in root_node.children])
        actions = np.array([child.action for child in root_node.children])
        search_policy = np.zeros(7)

        # Apply the temperature to the visit counts
        visit_counts = visit_counts ** (1 / temperature)

        search_policy[actions] = visit_counts

        # Normalize the visit counts to get a probability distribution
        sum_visit_counts = np.sum(search_policy)
        if sum_visit_counts > 0:
            search_policy /= sum_visit_counts
        else:
            raise Exception("No valid moves found")
        return search_policy

    def set_root(self, action):
        for child in self.root.children:
            if child.action == action:
                self.root = child
                break
        else:
            self.root = MCTSNode(self.root.state.simulate(action), self)

#
# class MCTS:
#     def __init__(self, root, model, num_simulations, c_param=4):
#         self.root = root
#         self.model = model
#         self.num_simulations = num_simulations
#         self.c_param = c_param
#
#     def run(self):
#         for _ in range(self.num_simulations):
#
#             node = self.root
#             search_path = [node]
#
#             while node.is_fully_expanded() and not node.state.is_terminal():
#                 node = node.best_child(self.c_param)
#                 search_path.append(node)
#
#             if not node.state.is_terminal():
#                 action = node.untried_actions[np.random.randint(len(node.untried_actions))]
#                 next_state = node.state.simulate(action)
#
#                 # Get prior probability for the action from the model
#                 action_probs, _ = self.get_action_probs(node.state)
#                 prior_prob = action_probs[action]
#
#                 child_node = node.expand(action, next_state, prior_prob)
#                 search_path.append(child_node)
#
#                 result = self.rollout(search_path[-1].state)
#
#             else:
#                 result = node.state.get_result()
#
#             self.backpropagate(search_path, result)
#
#     def rollout(self, state):
#         board = state.get_board()
#         input_tensor = np.array([board], dtype=np.float32)
#         # pr = cProfile.Profile()
#         # pr.enable()
#         _, value = self.model.predict(input_tensor)
#         # pr.disable()
#         # stats = Stats(pr)
#         # stats.sort_stats('tottime').print_stats(10)
#         #
#         # exit(0)
#         return value
#
#     # What a normal rollout would be like.
#     def mcts_rollout(self, state):
#         while not state.is_terminal():
#             actions = state.get_valid_moves()
#             action_probs = self.get_action_probs(state, actions)
#             selected_action = np.random.choice(actions, p=action_probs)
#             state = state.simulate(selected_action)
#         return state.get_result()
#
#     def get_action_probs(self, state, actions):
#         board = state.get_board()
#         input_tensor = np.array([board], dtype=np.float32)
#         policy_probs, value = self.model.predict(input_tensor)
#         return action_probs_numba_helper(actions, policy_probs), value
#
#     def backpropagate(self, search_path, result):
#         for node in search_path:
#             node.update(result)
#             result = -result
#
#     def get_best_move(self):
#         best_child = self.root.best_child(c_param=0)
#         return best_child.action
#
#     def reset(self, state):
#         self.root = MCTSNode(state)
#         state.reset()
#
#     def get_search_policy(mcts):
#         root_node = mcts.root
#         visit_counts = np.array([child.total_visits for child in root_node.children])
#         actions = np.array([child.action for child in root_node.children])
#         search_policy = np.zeros(7)
#         search_policy[actions] = visit_counts
#
#         # Normalize the visit counts to get a probability distribution
#         search_policy /= np.sum(search_policy)
#         return search_policy
#
#     def set_root(self, action):
#         for child in self.root.children:
#             if child.action == action:
#                 self.root = child
#                 break
#         else:
#             self.root = MCTSNode(self.root.state.simulate(action))


