// src/strategies/pythonAIStrategy.ts

import { Strategy, GameBoard } from "../types.js";

export class PythonAIStrategy implements Strategy {
    async asyncChooseColumn(board: GameBoard): Promise<number> {
        const response = await fetch('http://localhost:5001/ai/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();
        return data.move;
    }

    chooseColumn(board: GameBoard): number {
        throw new Error("Method not implemented.");
    }
}