// src/board.ts

// src/types.ts
export class GameBoard {
    bitboards: [bigint, bigint];
    rows: number;
    cols: number;
    firstAvailableRows: number[];

    constructor(rows: number, cols: number) {
        this.rows = rows;
        this.cols = cols;
        this.bitboards = [0n, 0n];
        this.firstAvailableRows = new Array(cols).fill(0);
    }
}

export interface Strategy {
    chooseColumn(board: GameBoard): number;

    asyncChooseColumn(board: GameBoard): Promise<number>;
}

// The rest of the code in board.ts remains the same
