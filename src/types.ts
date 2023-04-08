// src/board.ts

export class GameBoard {
    board: number[][];
    rows: number;
    cols: number;
    firstAvailableRows: number[];

    constructor(rows: number, cols: number) {
        this.rows = rows;
        this.cols = cols;
        this.board = new Array(rows).fill(null).map(() => new Array(cols).fill(0));
        this.firstAvailableRows = new Array(cols).fill(rows - 1);
    }
}

export interface Strategy {
    chooseColumn(board: GameBoard): number;
}

// The rest of the code in board.ts remains the same
