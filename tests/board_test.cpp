//
// Created by syys on 2021/2/26.
//

#include "board_test.h"

int main()
{
    Board board;
    board.init(9, 9);
    board.playMoveAssumeLegal(45, P_BLACK);
    board.playMoveAssumeLegal(46, P_WHITE);
    board.print_board(P_BLACK);
    return 0;
}