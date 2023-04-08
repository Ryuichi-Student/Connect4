// src/ui.ts

export function createBoardUI(
    gameBoard: HTMLElement,
    rows: number,
    cols: number
): void {
    gameBoard.innerHTML = "";
    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            const cell = document.createElement('div');
            cell.classList.add('cell');
            cell.dataset.row = row.toString();
            cell.dataset.col = col.toString();
            gameBoard.appendChild(cell);
        }
    }
}


export function updateBoardUI(gameBoard: HTMLElement, row: number, col: number, player: number): void {
    const cell = gameBoard.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`) as HTMLElement;
    cell.dataset.player = player.toString();
}
