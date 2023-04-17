// src/main.ts

import { handlePlayerMove, handleBotMove } from "./gameLogic.js";
import { createBoardUI } from "./ui.js";
import { createBoard } from "./board.js";

export const rows = 6;
export const cols = 7;
let currentPlayer = 1;
let gameActive = true;
let board = createBoard(rows, cols);

const gameBoard = document.querySelector(".game-board");
const newGameButton = document.querySelector(".new-game");
const gameMessage = document.querySelector(".game-message");

function updateGameMessage(message: string) {
    if (gameMessage) {
        gameMessage.textContent = message;
    }
}

if (gameBoard instanceof HTMLElement) {
    createBoardUI(gameBoard, rows, cols);
    gameBoard.addEventListener("click", (event) => {
        if (!gameActive) return;
        const target = event.target as HTMLElement;
        if (target.classList.contains("cell")) {
            const row = parseInt(target.dataset.row ?? "0", 10);
            const col = parseInt(target.dataset.col ?? "0", 10);

            const {gameEnded, draw} = handlePlayerMove(gameBoard, board, row, col, currentPlayer);

            // Special state to check for illegal move
            if (!gameEnded && draw) { return; }

            if (gameEnded) {
                gameActive = false;
                if (draw) { updateGameMessage("Draw!") }
                else { updateGameMessage("Player wins!"); }
            } else {
                currentPlayer = 2;
                const {gameEnded, draw} = handleBotMove(gameBoard, board);

                if (gameEnded) {
                    gameActive = false;
                    if (draw) { updateGameMessage("Draw!") }
                    else { updateGameMessage("Bot wins!"); }
                } else {
                    currentPlayer = 1;
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
            const cell = gameBoard?.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`) as HTMLElement;
            delete cell.dataset.player;
        }
    }
    board.firstAvailableRows = new Array(cols).fill(0);
    updateGameMessage("");
}

if (newGameButton instanceof HTMLElement) {
    newGameButton.addEventListener("click", () => {
        resetGame();
    });
}
