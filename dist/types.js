// src/board.ts
// src/types.ts
export class GameBoard {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.bitboards = [0n, 0n];
        this.firstAvailableRows = new Array(cols).fill(0);
    }
}
// The rest of the code in board.ts remains the same
