import { makeMove, isWinningMove, isDraw } from "./board.js";
import { updateBoardUI } from "./ui.js";
import { botMove } from "./bot.js";
export function handlePlayerMove(gameBoardElement, gameBoard, row, col, currentPlayer) {
    var newRow = makeMove(gameBoard, col, currentPlayer);
    if (newRow !== -1) {
        updateBoardUI(gameBoardElement, newRow, col, currentPlayer);
        if (isWinningMove(gameBoard, newRow, col, currentPlayer)) {
            return true;
        }
        else if (isDraw(gameBoard)) {
            return true;
        }
    }
    return false;
}
export function handleBotMove(gameBoardElement, gameBoard) {
    var col = botMove(gameBoard);
    var row = makeMove(gameBoard, col, 2);
    if (row !== -1) {
        updateBoardUI(gameBoardElement, row, col, 2);
        if (isWinningMove(gameBoard, row, col, 2)) {
            return true;
        }
        else if (isDraw(gameBoard)) {
            return true;
        }
    }
    return false;
}
