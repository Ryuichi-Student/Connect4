// src/strategies/pythonAIStrategy.ts

import { Strategy, GameBoard } from "../types.js";

export class PythonAIStrategy implements Strategy {
    async chooseColumn(board: GameBoard): Promise<number> {
        const response = await fetch('http://localhost:5000/ai/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ game_state: board }) // You'll need to modify this to match the state representation used by the Python AI
        });
        const data = await response.json();
        return data.move;
    }
}