import * as strategies from "./strategies/strategies.js";
// const STRATEGY = new strategies.randomStrategy();
// const STRATEGY = new strategies.minimaxStrategy(6);
const STRATEGY = new strategies.PythonAIStrategy();
export function botMove(board) {
    console.log(board.firstAvailableRows);
    return STRATEGY.chooseColumn(board);
}
export async function asyncBotMove(board) {
    console.log(board.firstAvailableRows);
    return await STRATEGY.asyncChooseColumn(board);
}
