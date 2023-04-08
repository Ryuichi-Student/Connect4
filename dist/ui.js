// src/ui.ts
export function createBoardUI(gameBoard, rows, cols) {
    gameBoard.innerHTML = "";
    for (var row = 0; row < rows; row++) {
        for (var col = 0; col < cols; col++) {
            var cell = document.createElement('div');
            cell.classList.add('cell');
            cell.dataset.row = row.toString();
            cell.dataset.col = col.toString();
            gameBoard.appendChild(cell);
        }
    }
}
export function updateBoardUI(gameBoard, row, col, player) {
    var cell = gameBoard.querySelector(".cell[data-row=\"".concat(row, "\"][data-col=\"").concat(col, "\"]"));
    cell.dataset.player = player.toString();
}
