import { GameBoard } from "./types.js";
export function createBoard(rows, cols) {
    return new GameBoard(rows, cols);
}
function bitboardMakeMove(bitboard, col, row, rows) {
    return bitboard | (1n << BigInt(col * (rows + 1) + row));
}
export function bitboardUndoMove(bitboard, col, row, rows) {
    return bitboard & ~(1n << BigInt(col * (rows + 1) + row));
}
function bitboardIsWinningMove(bitboard) {
    let y = bitboard & (bitboard >> 6n);
    if (y & (y >> 12n)) // check \ diagonal
     {
        return true;
    }
    y = bitboard & (bitboard >> 7n);
    if (y & (y >> 14n)) // check horizontal -
     {
        return true;
    }
    y = bitboard & (bitboard >> 8n);
    if (y & (y >> 16n)) // check / diagonal
     {
        return true;
    }
    y = bitboard & (bitboard >> 1n);
    if (y & (y >> 2n)) // check vertical |
     {
        return true;
    }
    return false;
}
export function makeMove(gameBoard, col, player) {
    const row = gameBoard.firstAvailableRows[col];
    if (row < gameBoard.rows) {
        gameBoard.bitboards[player - 1] = bitboardMakeMove(gameBoard.bitboards[player - 1], col, row, gameBoard.rows);
        gameBoard.firstAvailableRows[col]++;
        return row;
    }
    return -1;
}
export function isWinningMove(gameBoard, row, col, player) {
    return bitboardIsWinningMove(gameBoard.bitboards[player - 1]);
}
export function isDraw(gameBoard) {
    return gameBoard.firstAvailableRows.every(row => row === gameBoard.rows);
}
