import numpy as np
from numba import njit

@njit
def check_line(board, row, col, rows, cols):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for dr, dc in directions:
        if check_direction(board, row, col, dr, dc, rows, cols):
            return True
    return False

@njit
def check_direction(board, row, col, dr, dc, rows, cols):
    x = board[row][col]
    for _ in range(4):
        if row < 0 or row >= rows or col < 0 or col >= cols or board[row][col] != x:
            return False
        row += dr
        col += dc
    return True


@njit
def is_full_numba(board, rows, cols):
    for row in range(rows):
        for col in range(cols):
            if board[row][col] == 0:
                return False
    return True


@njit
def has_winner_numba(board, rows, cols):
    for row in range(rows):
        for col in range(cols):
            if board[row][col] != 0 and check_line(board, row, col, rows, cols):
                return True
    return False


class Connect4State:
    def __init__(self, board=None, current_player=1):
        self.rows = 6
        self.cols = 7
        self.board_size = self.rows * self.cols
        if board is None:
            self.board = np.zeros((self.rows, self.cols), dtype=int)
        else:
            self.board = board.copy()
        self.current_player = current_player

    def get_valid_moves(self):
        return [col if self.board[0][col] == 0 else -1 for col in range(self.cols)]

    def is_terminal(self):
        return self.has_winner() or self.is_full()

    def has_winner(self):
        # for row in range(self.rows):
        #     for col in range(self.cols):
        #         if self.board[row][col] != 0 and check_line(self.board, row, col, self.rows, self.cols):
        #             return True
        # return False
        return has_winner_numba(self.board, self.rows, self.cols)

    def is_full(self):
        # return (self.board != 0).all()
        return is_full_numba(self.board, self.rows, self.cols)

    def simulate(self, action):
        row = self.rows - 1 - np.argmax(self.board[::-1, action] == 0)
        new_board = self.board.copy()
        new_board[row, action] = 1
        return Connect4State(-new_board, -self.current_player)

    def get_result(self):
        if self.has_winner():
            return 1 if self.current_player == -1 else -1
        return 0

    def get_board(self):
        return self.board

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
