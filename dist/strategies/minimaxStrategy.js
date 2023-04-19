import { makeMove, isWinningMove, isDraw, bitboardUndoMove } from "../board.js";
export class minimaxStrategy {
    constructor(maxDepth = 4) {
        this.maxDepth = maxDepth;
    }
    chooseColumn(board) {
        // Get the best move using the minimax algorithm
        const { score, column } = this.minimax(board, this.maxDepth, true, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY);
        console.log("" + score + column);
        return column;
    }
    chooseRandomColumn(board) {
        // Get an array of available columns
        const availableColumns = [];
        for (let col = 0; col < board.cols; col++) {
            if (board.firstAvailableRows[col] < board.rows) {
                availableColumns.push(col);
            }
        }
        // If there are no available columns, return -1
        if (availableColumns.length === 0) {
            return -1;
        }
        // Randomly pick a column from the available columns
        const randomIndex = Math.floor(Math.random() * availableColumns.length);
        return availableColumns[randomIndex];
    }
    // The minimax algorithm with alpha-beta pruning
    // strategies/minimaxStrategy.ts
    minimax(board, depth, isMaximizingPlayer, alpha, beta) {
        if (depth === 0 || isDraw(board)) {
            return { score: 0, column: this.chooseRandomColumn(board) };
        }
        let bestScore = isMaximizingPlayer ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;
        let bestColumn = -1;
        for (let col = 0; col < board.cols; col++) {
            // Check if the column is not full
            if (board.firstAvailableRows[col] < board.rows) {
                // Make the move for the current player (1 for maximizing, 2 for minimizing)
                const row = makeMove(board, col, isMaximizingPlayer ? 2 : 1);
                // If the move results in a win, return the appropriate score and column
                if (isWinningMove(board, row, col, isMaximizingPlayer ? 2 : 1)) {
                    board.bitboards[isMaximizingPlayer ? 1 : 0] = bitboardUndoMove(board.bitboards[isMaximizingPlayer ? 1 : 0], col, row, board.rows); // Undo the move
                    board.firstAvailableRows[col]--; // Decrement the first available row for the column
                    return { score: isMaximizingPlayer ? 1 : -1, column: col };
                }
                // Recursive minimax call
                const { score } = this.minimax(board, depth - 1, !isMaximizingPlayer, alpha, beta);
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
                board.bitboards[isMaximizingPlayer ? 1 : 0] = bitboardUndoMove(board.bitboards[isMaximizingPlayer ? 1 : 0], col, row, board.rows); // Undo the move
                board.firstAvailableRows[col]--; // Decrement the first available row for the column
                // Alpha-beta pruning
                if (beta < alpha) {
                    break;
                }
            }
        }
        return { score: bestScore, column: bestColumn };
    }
    asyncChooseColumn(board) {
        throw new Error("Method not implemented.");
    }
}
