// src/main.ts
import { handlePlayerMove, handleBotMove } from "./gameLogic.js";
import { createBoardUI } from "./ui.js";
import { createBoard } from "./board.js";
export var rows = 6;
export var cols = 7;
var currentPlayer = 1;
var gameActive = true;
var board = createBoard(rows, cols);
var gameBoard = document.querySelector(".game-board");
var newGameButton = document.querySelector(".new-game");
var gameMessage = document.querySelector(".game-message");
function updateGameMessage(message) {
    if (gameMessage) {
        gameMessage.textContent = message;
    }
}
if (gameBoard instanceof HTMLElement) {
    createBoardUI(gameBoard, rows, cols);
    gameBoard.addEventListener("click", function (event) {
        var _a, _b;
        if (!gameActive)
            return;
        var target = event.target;
        if (target.classList.contains("cell")) {
            var row = parseInt((_a = target.dataset.row) !== null && _a !== void 0 ? _a : "0", 10);
            var col = parseInt((_b = target.dataset.col) !== null && _b !== void 0 ? _b : "0", 10);
            var gameEnded = handlePlayerMove(gameBoard, board, row, col, currentPlayer);
            if (gameEnded) {
                gameActive = false;
                updateGameMessage("Player wins!");
            }
            else {
                currentPlayer = 2;
                var botWin = handleBotMove(gameBoard, board);
                if (botWin) {
                    gameActive = false;
                    updateGameMessage("Bot wins!");
                }
                else {
                    currentPlayer = 1;
                }
            }
        }
    });
}
function resetGame() {
    gameActive = true;
    currentPlayer = 1;
    for (var row = 0; row < rows; row++) {
        for (var col = 0; col < cols; col++) {
            board.board[row][col] = 0;
            var cell = gameBoard === null || gameBoard === void 0 ? void 0 : gameBoard.querySelector(".cell[data-row=\"".concat(row, "\"][data-col=\"").concat(col, "\"]"));
            delete cell.dataset.player;
        }
    }
    board.firstAvailableRows = new Array(cols).fill(rows - 1);
    updateGameMessage("");
}
if (newGameButton instanceof HTMLElement) {
    newGameButton.addEventListener("click", function () {
        resetGame();
    });
}
