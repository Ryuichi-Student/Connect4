// strategies/minimaxStrategy.ts
import { GameBoard, Strategy } from "../types.js";
import { makeMove, isWinningMove, isDraw } from "../board.js";

export class minimaxStrategy implements Strategy {
    maxDepth: number;

    constructor(maxDepth: number = 4) {
        this.maxDepth = maxDepth;
    }

    chooseColumn(board: GameBoard): number {
        // Get the best move using the minimax algorithm
        const { column } = this.minimax(board, this.maxDepth, true, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY);

        return column;
    }

    // The minimax algorithm with alpha-beta pruning
    minimax(board: GameBoard, depth: number, isMaximizingPlayer: boolean, alpha: number, beta: number): { score: number; column: number } {
        if (depth === 0 || isDraw(board)) {
            return { score: 0, column: -1 };
        }

        let bestScore = isMaximizingPlayer ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;
        let bestColumn = -1;

        for (let col = 0; col < board.cols; col++) {
            // Check if the column is not full
            const row = board.firstAvailableRows[col];
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
                const { score } = this.minimax(board, depth - 1, !isMaximizingPlayer, alpha, beta);

                // Update the best score and best column if necessary
                if (isMaximizingPlayer) {
                    if (score > bestScore) {
                        bestScore = score;
                        bestColumn = col;
                    }
                    alpha = Math.max(alpha, bestScore);
                } else {
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
    }
}
