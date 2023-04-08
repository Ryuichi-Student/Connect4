var randomStrategy = /** @class */ (function () {
    function randomStrategy() {
    }
    randomStrategy.prototype.chooseColumn = function (board) {
        // Get an array of available columns
        var availableColumns = [];
        for (var col = 0; col < board.cols; col++) {
            if (board.board[0][col] === 0) {
                availableColumns.push(col);
            }
        }
        // If there are no available columns, return -1
        if (availableColumns.length === 0) {
            return -1;
        }
        // Randomly pick a column from the available columns
        var randomIndex = Math.floor(Math.random() * availableColumns.length);
        return availableColumns[randomIndex];
    };
    return randomStrategy;
}());
export { randomStrategy };
