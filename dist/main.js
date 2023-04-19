// src/main.ts
import { handlePlayerMove, handleBotMove, sendMoveToBackend, handleAsyncBotMove } from "./gameLogic.js";
import { createBoardUI } from "./ui.js";
import { createBoard } from "./board.js";
import { asyncBotMove } from "./bot.js";
export const rows = 6;
export const cols = 7;
let currentPlayer = 1;
let gameActive = true;
let board = createBoard(rows, cols);
const requires_backend = true; // set to true if you want to use the Python AI
const gameBoard = document.querySelector(".game-board");
const newGameButton = document.querySelector(".new-game");
const gameMessage = document.querySelector(".game-message");
function updateGameMessage(message) {
    if (gameMessage) {
        gameMessage.textContent = message;
    }
}
if (requires_backend) {
    sendMoveToBackend(-1).catch((error) => {
        console.error('Error while sending the move to the backend:', error);
    });
}
if (gameBoard instanceof HTMLElement) {
    createBoardUI(gameBoard, rows, cols);
    gameBoard.addEventListener("click", (event) => {
        if (!gameActive || gameBoard.classList.contains("disabled"))
            return;
        gameBoard.classList.add("disabled");
        const target = event.target;
        if (target.classList.contains("cell")) {
            const row = parseInt(target.dataset.row ?? "0", 10);
            const col = parseInt(target.dataset.col ?? "0", 10);
            const { gameEnded, draw } = handlePlayerMove(gameBoard, board, row, col, currentPlayer);
            // Special state to check for illegal move
            if (!gameEnded && draw) {
                gameBoard.classList.remove("disabled");
                return;
            }
            if (gameEnded) {
                gameActive = false;
                if (draw) {
                    updateGameMessage("Draw!");
                }
                else {
                    updateGameMessage("Player wins!");
                }
                gameBoard.classList.remove("disabled");
            }
            else {
                currentPlayer = 2;
                if (requires_backend) {
                    sendMoveToBackend(col).catch((error) => {
                        console.error('Error while sending the move to the backend:', error);
                        gameBoard.classList.remove("disabled");
                        return;
                    });
                    // Wait for the backend to respond
                    asyncBotMove(board).then((col) => {
                        const { gameEnded, draw } = handleAsyncBotMove(gameBoard, board, col);
                        if (gameEnded) {
                            gameActive = false;
                            if (draw) {
                                updateGameMessage("Draw!");
                            }
                            else {
                                updateGameMessage("Bot wins!");
                            }
                        }
                        else {
                            currentPlayer = 1;
                        }
                        gameBoard.classList.remove("disabled");
                    });
                }
                else {
                    const { gameEnded, draw } = handleBotMove(gameBoard, board);
                    if (gameEnded) {
                        gameActive = false;
                        if (draw) {
                            updateGameMessage("Draw!");
                        }
                        else {
                            updateGameMessage("Bot wins!");
                        }
                    }
                    else {
                        currentPlayer = 1;
                    }
                    gameBoard.classList.remove("disabled");
                }
            }
        }
    });
}
function resetGame() {
    gameActive = true;
    currentPlayer = 1;
    board.bitboards = [0n, 0n];
    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            const cell = gameBoard?.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`);
            delete cell.dataset.player;
        }
    }
    board.firstAvailableRows = new Array(cols).fill(0);
    updateGameMessage("");
    sendMoveToBackend(-1).catch((error) => {
        console.error('Error while sending the move to the backend:', error);
    });
}
if (newGameButton instanceof HTMLElement) {
    newGameButton.addEventListener("click", () => {
        resetGame();
        gameBoard?.classList.remove("disabled");
    });
}
