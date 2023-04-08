import { makeMove, isWinningMove, isDraw } from "./board.js";
import { updateBoardUI } from "./ui.js";
import { botMove } from "./bot.js";
import { GameBoard } from "./types.js";

export function handlePlayerMove(gameBoardElement: HTMLElement, gameBoard: GameBoard, row: number, col: number, currentPlayer: number): boolean {
    const newRow = makeMove(gameBoard, col, currentPlayer);
    if (newRow !== -1) {
        updateBoardUI(gameBoardElement, newRow, col, currentPlayer);

        if (isWinningMove(gameBoard, newRow, col, currentPlayer)) {
            return true;
        } else if (isDraw(gameBoard)) {
            return true;
        }
    }
    return false;
}

export function handleBotMove(gameBoardElement: HTMLElement, gameBoard: GameBoard): boolean {
    const col = botMove(gameBoard);
    const row = makeMove(gameBoard, col, 2);
    if (row !== -1) {
        updateBoardUI(gameBoardElement, row, col, 2);

        if (isWinningMove(gameBoard, row, col, 2)) {
            return true;
        } else if (isDraw(gameBoard)) {
            return true;
        }
    }
    return false;
}
