import { GameBoard } from "./types.js";

export function createBoard(rows: number, cols: number): GameBoard {
    return new GameBoard(rows, cols);
}

export function makeMove(gameBoard: GameBoard, col: number, player: number): number {
    const row = gameBoard.firstAvailableRows[col];
    if (row >= 0) {
        gameBoard.board[row][col] = player;
        gameBoard.firstAvailableRows[col] -= 1;
    }
    return row;
}

function checkLine(board: number[][], row: number, col: number, rowDelta: number, colDelta: number, player: number): boolean {
    for (let i = 0; i < 4; i++) {
        if (board[row][col] !== player) {
            return false;
        }
        row += rowDelta;
        col += colDelta;
    }
    return true;
}

export function isWinningMove(gameBoard: GameBoard, row: number, col: number, player: number): boolean {
    const directions = [
        { rowDelta: 0, colDelta: 1 },  // horizontal
        { rowDelta: 1, colDelta: 1 },  // diagonal /
        { rowDelta: 1, colDelta: 0 },  // vertical
        { rowDelta: 1, colDelta: -1 }  // diagonal \
    ];

    for (const direction of directions) {
        let rowStart = row - 3 * direction.rowDelta;
        let colStart = col - 3 * direction.colDelta;

        for (let i = 0; i < 4; i++) {
            if (rowStart >= 0 && colStart >= 0 && rowStart + 3 * direction.rowDelta < gameBoard.rows && colStart + 3 * direction.colDelta < gameBoard.cols) {
                if (checkLine(gameBoard.board, rowStart, colStart, direction.rowDelta, direction.colDelta, player)) {
                    return true;
                }
            }
            rowStart += direction.rowDelta;
            colStart += direction.colDelta;
        }
    }
    return false;
}

export function isDraw(gameBoard: GameBoard): boolean {
    return gameBoard.board.every(row => row.every(cell => cell !== 0));
}

