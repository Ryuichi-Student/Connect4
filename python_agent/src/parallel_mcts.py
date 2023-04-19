import threading
from mcts import MCTS, MCTSNode
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

class ParallelMCTSNode(MCTSNode):
    def __init__(self, state, mcts, prior_prob=1.0, parent=None, action=None, virtual_loss=1):
        super().__init__(state, mcts, prior_prob, parent, action)
        self.virtual_loss = virtual_loss
        self.lock = threading.Lock()
        # self.reached will be set to True when the node is added to the queue
        # self.visited will be set to True when the node has been processed
        self.visited = False

    def select(self, c_param):
        with self.lock:
            # The node has been visited and expanded before. Choose the best child node.
            if self.is_fully_expanded():
                selected_child = self.best_child(c_param)
                selected_child.add_virtual_loss()
                return selected_child

            elif self.reached:
                with self.mcts.expansion_lock:
                    # Check if the node has been visited while waiting for the lock
                    if self.visited:
                        selected_child = self.best_child(c_param)
                        selected_child.add_virtual_loss()
                        return selected_child
                    self.mcts.process_prediction_queue(self, c_param)

            # The node has not been visited before. Expand the node.
            elif len(self.children) == 0:
                with self.mcts.expansion_lock:
                    # Check if the node has been visited while waiting for the lock
                    if self.visited:
                        selected_child = self.best_child(c_param)
                        selected_child.add_virtual_loss()
                        return selected_child

                    self.expand_all()
                    self.reached = True

            else:
                raise Exception("This should never happen")

    def get_ucb_score(self, c_param):
        if self.N == 0 or self.parent.N == 0:
            return float('inf')
        else:
            exploit = self.Q
            explore = np.sqrt(np.log(self.parent.N) / self.N)
            return exploit + c_param * self.P * explore

    def add_virtual_loss(self):
        with self.lock:
            self.N += self.virtual_loss
            self.W -= self.virtual_loss

    def remove_virtual_loss(self):
        with self.lock:
            self.N -= self.virtual_loss
            self.W += self.virtual_loss

    def backpropagate(self, value):
        with self.lock:
            self.N += 1
            self.W += value
            self.Q = self.W / self.N
        if self.parent is not None:
            self.parent.backpropagate(-value)

    def expand(self, action, next_state, prior_prob):
        child = ParallelMCTSNode(next_state, self.mcts, prior_prob, parent=self, action=action)
        self.untried_actions.remove(action)
        self.children.append(child)
        return child

    def expand_all(self):
        self.mcts.prediction_queue.append(self)
        if len(self.mcts.prediction_queue) >= self.mcts.batch_size:
            self.mcts.process_prediction_queue(self, self.mcts.c_param)
            self.reached = False # We no longer need to flag this for processing.

class ParallelMCTS(MCTS):
    def __init__(self, state, model, num_simulations, c_param=4, num_threads=4):
        self.model = model
        self.expansion_lock = threading.Lock()
        root = ParallelMCTSNode(state, self)
        super().__init__(root, model, num_simulations, c_param)
        self.num_threads = num_threads

    def run(self):
        num_simulations_per_thread = self.num_simulations // self.num_threads
        extra_simulations = self.num_simulations % self.num_threads

        threads = []
        for _ in range(self.num_threads):
            num_simulations = num_simulations_per_thread + (1 if extra_simulations > 0 else 0)
            if extra_simulations > 0:
                extra_simulations -= 1
            t = threading.Thread(target=self.worker, args=(num_simulations,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Process any remaining queued states
        if len(self.prediction_queue) > 0:
            self.process_prediction_queue(self.root, self.c_param)

    def worker(self, num_simulations):
        for _ in range(num_simulations):
            self.run_simulation()

    def run_simulation(self):
        node = self.root
        depth = 0
        while not node.state.is_terminal():
            child = node.select(self.c_param)
            if child is None:
                break
            node = child
            depth += 1
        state = node.state
        if state.is_terminal():
            value = state.has_winner()
        else:
            _, value = self.model.predict(state.get_board())
            value = -value
        node.backpropagate(value)
        node.remove_virtual_loss()

    def process_prediction_queue(self, cur, c_param):
        if len(self.prediction_queue) == 0:
            raise Exception("No nodes in the queue")
        batch_states = [node.state.get_board() for node in self.prediction_queue]
        input_tensor = np.array(batch_states, dtype=np.float32)
        batch_probs, values = self.model.batched_predict(input_tensor, threads=self.num_threads, processes=4)

        for node, prior_probs, value in zip(self.prediction_queue, batch_probs, values):
            filtered_probs = action_probs_numba_helper(node.valid_actions, prior_probs)
            node.expand_from_queue(filtered_probs)
            node.value = value
            node.visited = True

        self.prediction_queue.clear()
        cur.visited = True

    def reset(self, state):
        self.root = ParallelMCTSNode(state, self)
        state.reset()

    def set_root(self, action):
        for child in self.root.children:
            if child.action == action:
                self.root = child
                break
        else:
            self.root = ParallelMCTSNode(self.root.state.simulate(action), self)