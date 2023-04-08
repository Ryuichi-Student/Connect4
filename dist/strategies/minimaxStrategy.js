import { makeMove, isWinningMove, isDraw } from "../board.js";
var minimaxStrategy = /** @class */ (function () {
    function minimaxStrategy(maxDepth) {
        if (maxDepth === void 0) { maxDepth = 4; }
        this.maxDepth = maxDepth;
    }
    minimaxStrategy.prototype.chooseColumn = function (board) {
        // Get the best move using the minimax algorithm
        var column = this.minimax(board, this.maxDepth, true, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY).column;
        return column;
    };
    // The minimax algorithm with alpha-beta pruning
    minimaxStrategy.prototype.minimax = function (board, depth, isMaximizingPlayer, alpha, beta) {
        if (depth === 0 || isDraw(board)) {
            return { score: 0, column: -1 };
        }
        var bestScore = isMaximizingPlayer ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;
        var bestColumn = -1;
        for (var col = 0; col < board.cols; col++) {
            // Check if the column is not full
            var row = board.firstAvailableRows[col];
            if (row >= 0) {
                // Make the move for the current player (1 for maximizing, 2 for minimizing)
                makeMove(board, col, isMaximizingPlayer ? 1 : 2);
                // If the move results in a win, return the appropriate score and column
                if (isWinningMove(board, row, col, isMaximizingPlayer ? 1 : 2)) {
                    board.firstAvailableRows[col]++; // Undo the move
                    board.board[row][col] = 0;
                    return { score: isMaximizingPlayer ? 1 : -1, column: col };
                }
                // Recursive minimax call
                var score = this.minimax(board, depth - 1, !isMaximizingPlayer, alpha, beta).score;
                // Update the best score and best column if necessary
                if (isMaximizingPlayer) {
                    if (score > bestScore) {
                        bestScore = score;
                        bestColumn = col;
                    }
                    alpha = Math.max(alpha, bestScore);
                }
                else {
                    if (score < bestScore) {
                        bestScore = score;
                        bestColumn = col;
                    }
                    beta = Math.min(beta, bestScore);
                }
                // Undo the move
                board.firstAvailableRows[col]++;
                board.board[row][col] = 0;
                // Alpha-beta pruning
                if (beta <= alpha) {
                    break;
                }
            }
        }
        return { score: bestScore, column: bestColumn };
    };
    return minimaxStrategy;
}());
export { minimaxStrategy };
