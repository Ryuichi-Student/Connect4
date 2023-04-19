import numpy as np
from numba import njit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

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
        self.mcts = mcts
        self.cached_ucb_score = None

        # For batched predictions, flag for processing the queue
        self.reached = False

    def select(self, c_param):
        # The node has been visited and expanded before. Choose the best child node.
        if self.is_fully_expanded():
            selected_child = self.best_child(c_param)
            return selected_child

        elif self.reached:
            self.mcts.process_prediction_queue()

        # The node has not been visited before. Expand the node.
        elif len(self.children) == 0:
            self.expand_all()
            self.reached = True

        else:
            raise Exception("This should never happen")

    def get_ucb_score(self, c_param):
        if self.cached_ucb_score is None:
            if self.N == 0 or self.parent.N == 0:
                self.cached_ucb_score = float('inf')
            else:
                # The UCB score is the average value of the node plus the exploration bonus.
                self.Q = self.W / self.N
                u = (self.P * np.sqrt(self.parent.N)) / (1 + self.N)
                self.cached_ucb_score = self.Q + c_param * u
        return self.cached_ucb_score

    def backpropagate(self, value):
        current = self
        while current is not None:
            current.N += 1
            current.W += value
            current.cached_ucb_score = None  # Invalidate the cached UCB score
            current = current.parent
            value = -value # The value is always from the perspective of the player on the node.

    def expand_from_queue(self, prior_probs):
        for action in self.untried_actions.copy()[::-1]:
            next_state = self.state.simulate(action)
            P = prior_probs[action]
            self.expand(action, next_state, P)

    def expand_all(self):
        self.mcts.prediction_queue.append(self)
        if len(self.mcts.prediction_queue) >= self.mcts.batch_size:
            self.mcts.process_prediction_queue()
            self.reached = False # We no longer need to flag this for processing.

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
        # You could randomly pick the best child from those of the same score, but picking the first one is fine.
        # best = max(scores)[0]
        # best_children = [child for score, _, child in scores if score == best]
        # return best_children[np.random.randint(0, len(best_children))]
        return max(scores)[2]

    def __repr__(self):
        return f"Action: {self.action}, N: {self.N}, W: {self.W}, Q: {self.W/self.N}, P: {self.P}\n"


class MCTS:
    def __init__(self, state, model, num_simulations, c_param=3, max_batch_size=64):
        self.model = model
        self.num_simulations = num_simulations
        self.c_param = c_param
        if isinstance(state, MCTSNode):
            self.root = state
        else:
            self.root = MCTSNode(state, self)

        # We want to predict in batches to remove the overhead of calling the model.
        self.batch_size = max_batch_size
        self.prediction_queue = []
        self.value = None

    def run(self):
        for _ in range(self.num_simulations):
            self.run_simulation()

        # Process any remaining queued states
        if len(self.prediction_queue) > 0:
            self.process_prediction_queue()

    def run_simulation(self):
        node = self.root
        while not node.state.is_terminal():
            child = node.select(self.c_param)
            if child is None: # Leaf node. We have expanded the leaf now.
                break
            node = child

        state = node.state
        if state.is_terminal():
            # To reason about this, suppose this is a direct child of the root.
            # If there is a win, we want this child to have a higher value.
            value = state.has_winner()
        else:
            # Suppose this is a direct child of the root.
            # If this is winning for the current player of this state,
            # we want to pick the lowest value possible.
            if self.value is None:
                _, value = self.model.predict(state.get_board())
            else:
                value = self.value
            value = -value

        node.backpropagate(value)

    # Get the action probabilities from the model and filter out the invalid actions.
    def get_action_probs(self, state, actions):
        board = state.get_board()
        input_tensor = np.array([board], dtype=np.float32)
        action_probs, value = self.model.predict(input_tensor)

        action_probs = action_probs_numba_helper(actions, action_probs)
        return action_probs, value

    def process_prediction_queue(self):
        batch_states = [node.state.get_board() for node in self.prediction_queue]
        input_tensor = np.array(batch_states, dtype=np.float32)
        batch_probs, values = self.model.batched_predict(input_tensor)

        for node, prior_probs, value in zip(self.prediction_queue, batch_probs, values):
            filtered_probs = action_probs_numba_helper(node.valid_actions, prior_probs)
            node.expand_from_queue(filtered_probs)
            node.value = value

        self.prediction_queue.clear()

    def get_best_move(self):
        # The best move is the one with the highest visit count (not the highest average value).
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

        # Apply the temperature to the visit counts. Higher temperature means more randomness.
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

