// src/board.ts
var GameBoard = /** @class */ (function () {
    function GameBoard(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.board = new Array(rows).fill(null).map(function () { return new Array(cols).fill(0); });
        this.firstAvailableRows = new Array(cols).fill(rows - 1);
    }
    return GameBoard;
}());
export { GameBoard };
// The rest of the code in board.ts remains the same
