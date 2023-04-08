import { GameBoard } from "./types.js";
export function createBoard(rows, cols) {
    return new GameBoard(rows, cols);
}
export function makeMove(gameBoard, col, player) {
    var row = gameBoard.firstAvailableRows[col];
    if (row >= 0) {
        gameBoard.board[row][col] = player;
        gameBoard.firstAvailableRows[col] -= 1;
    }
    return row;
}
function checkLine(board, row, col, rowDelta, colDelta, player) {
    for (var i = 0; i < 4; i++) {
        if (board[row][col] !== player) {
            return false;
        }
        row += rowDelta;
        col += colDelta;
    }
    return true;
}
export function isWinningMove(gameBoard, row, col, player) {
    var directions = [
        { rowDelta: 0, colDelta: 1 },
        { rowDelta: 1, colDelta: 1 },
        { rowDelta: 1, colDelta: 0 },
        { rowDelta: 1, colDelta: -1 } // diagonal \
    ];
    for (var _i = 0, directions_1 = directions; _i < directions_1.length; _i++) {
        var direction = directions_1[_i];
        var rowStart = row - 3 * direction.rowDelta;
        var colStart = col - 3 * direction.colDelta;
        for (var i = 0; i < 4; i++) {
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
export function isDraw(gameBoard) {
    return gameBoard.board.every(function (row) { return row.every(function (cell) { return cell !== 0; }); });
}
