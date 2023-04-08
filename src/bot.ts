import {GameBoard} from "./types.js";
import * as strategies from "./strategies/strategies.js";

// const STRATEGY = new strategies.randomStrategy();
const STRATEGY = new strategies.minimaxStrategy(4);

export function botMove(board: GameBoard): number {
    const col = STRATEGY.chooseColumn(board)
    // Implement other strategies if needed
    return col;
}