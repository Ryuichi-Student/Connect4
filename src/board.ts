import { GameBoard } from "./types.js";

export function createBoard(rows: number, cols: number): GameBoard {
    return new GameBoard(rows, cols);
}

function bitboardMakeMove(bitboard: bigint, col: number, row: number, rows: number): bigint {
    return bitboard | (1n << BigInt(col * (rows+1) + row));
}

export function bitboardUndoMove(bitboard: bigint, col: number, row: number, rows: number): bigint {
    return bitboard & ~(1n << BigInt(col * (rows+1) + row));
}

function bitboardIsWinningMove(bitboard: bigint): boolean {
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
    if (y & (y >> 2n))     // check vertical |
    {
        return true;
    }
    return false;
}


export function makeMove(gameBoard: GameBoard, col: number, player: number): number {
    const row = gameBoard.firstAvailableRows[col];
    if (row < gameBoard.rows) {
        gameBoard.bitboards[player - 1] = bitboardMakeMove(gameBoard.bitboards[player - 1], col, row, gameBoard.rows);
        gameBoard.firstAvailableRows[col]++;

        return row;
    }
    return -1;
}

export function isWinningMove(gameBoard: GameBoard, row: number, col: number, player: number): boolean {
    return bitboardIsWinningMove(gameBoard.bitboards[player - 1]);
}

export function isDraw(gameBoard: GameBoard): boolean {
    return gameBoard.firstAvailableRows.every(row => row === gameBoard.rows);
}
