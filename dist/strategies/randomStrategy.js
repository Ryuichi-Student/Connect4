export class randomStrategy {
    chooseColumn(board) {
        // Get an array of available columns
        const availableColumns = [];
        for (let col = 0; col < board.cols; col++) {
            if (board.firstAvailableRows[col] < board.rows) {
                availableColumns.push(col);
            }
        }
        // If there are no available columns, return -1
        if (availableColumns.length === 0) {
            return -1;
        }
        // Randomly pick a column from the available columns
        const randomIndex = Math.floor(Math.random() * availableColumns.length);
        return availableColumns[randomIndex];
    }
}
