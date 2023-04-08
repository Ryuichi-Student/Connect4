import * as strategies from "./strategies/strategies.js";
// const STRATEGY = new strategies.randomStrategy();
var STRATEGY = new strategies.minimaxStrategy(4);
export function botMove(board) {
    var col = STRATEGY.chooseColumn(board);
    // Implement other strategies if needed
    return col;
}
