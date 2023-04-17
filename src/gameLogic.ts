import { makeMove, isWinningMove, isDraw } from "./board.js";
import { updateBoardUI } from "./ui.js";
import { botMove } from "./bot.js";
import { GameBoard } from "./types.js";

export function handlePlayerMove(gameBoardElement: HTMLElement, gameBoard: GameBoard, row: number, col: number, currentPlayer: number): {gameEnded: boolean, draw: boolean} {
    const newRow = makeMove(gameBoard, col, currentPlayer);
    if (newRow !== -1) {
        updateBoardUI(gameBoardElement, newRow, col, currentPlayer);

        if (isWinningMove(gameBoard, newRow, col, currentPlayer)) {
            return {gameEnded: true, draw: false};
        } else if (isDraw(gameBoard)) {
            return {gameEnded: true, draw: true};
        }
    }
    // Use the final unused state to show illegal move.
    if (newRow == -1) { return {gameEnded: false, draw: true} }
    return {gameEnded: false, draw: false};
}

export function handleBotMove(gameBoardElement: HTMLElement, gameBoard: GameBoard): {gameEnded: boolean, draw: boolean}  {
    const col = botMove(gameBoard);
    const row = makeMove(gameBoard, col, 2);
    if (row !== -1) {
        updateBoardUI(gameBoardElement, row, col, 2);

        if (isWinningMove(gameBoard, row, col, 2)) {
            return {gameEnded: true, draw: false};
        } else if (isDraw(gameBoard)) {
            return {gameEnded: true, draw: true};
        }
    }
    return {gameEnded: false, draw: false};
}
