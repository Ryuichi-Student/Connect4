// src/ui.ts
export function createBoardUI(gameBoard, rows, cols) {
    gameBoard.innerHTML = "";
    for (let row = rows - 1; row >= 0; row--) { // Reverse the order of rows
        for (let col = 0; col < cols; col++) {
            const cell = document.createElement('div');
            cell.classList.add('cell');
            cell.dataset.row = row.toString();
            cell.dataset.col = col.toString();
            gameBoard.appendChild(cell);
        }
    }
}
export function updateBoardUI(gameBoard, row, col, player) {
    const cell = gameBoard.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`);
    cell.dataset.player = player.toString();
}
