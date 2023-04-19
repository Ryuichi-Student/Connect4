// src/strategies/pythonAIStrategy.ts
export class PythonAIStrategy {
    async asyncChooseColumn(board) {
        const response = await fetch('http://localhost:5001/ai/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();
        return data.move;
    }
    chooseColumn(board) {
        throw new Error("Method not implemented.");
    }
}
