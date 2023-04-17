import threading
from mcts import MCTS, MCTSNode
import numpy as np

class ParallelMCTSNode(MCTSNode):
    def __init__(self, state, model, prior_prob=1.0, parent=None, action=None, virtual_loss=1):
        super().__init__(state, model, prior_prob, parent, action)
        self.virtual_loss = virtual_loss
        self.lock = threading.Lock()

    def select(self, c_param):
        with self.lock:
            # The node has been visited and expanded before. Choose the best child node.
            if self.is_fully_expanded():
                selected_child = self.best_child(c_param)
                selected_child.add_virtual_loss()
                return selected_child

            # The node has not been visited before. Expand the node.
            elif len(self.children) == 0:
                self.expand_all()

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

class ParallelMCTS(MCTS):
    def __init__(self, state, model, num_simulations, c_param=4, num_threads=8):
        self.model = model
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

    def worker(self, num_simulations):
        for _ in range(num_simulations):
            self.run_simulation()

    def run_simulation(self):
        node = self.root
        depth = 0
        while not node.state.is_terminal() and depth < self.depth_limit:
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