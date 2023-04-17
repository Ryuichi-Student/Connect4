import {GameBoard} from "./types.js";
import * as strategies from "./strategies/strategies.js";

// const STRATEGY = new strategies.randomStrategy();
const STRATEGY = new strategies.minimaxStrategy(6);

export function botMove(board: GameBoard): number {
    console.log(board.firstAvailableRows)
    const col = STRATEGY.chooseColumn(board)
    // Implement other strategies if needed
    return col;
}
