//
// Created by syys on 2021/2/26.
//

#include "board_test.h"

void test_not_capture_rule()
{
    Board board(9, 9);
    board.playMoveAssumeLegal(5, 4, P_BLACK);
    board.playMoveAssumeLegal(4,5,P_BLACK);
    board.playMoveAssumeLegal(5,6,P_BLACK);
    board.playMoveAssumeLegal(5, 5, P_WHITE);
    board.print_board(P_BLACK);

    std::vector<int> legal_dist_tmp;
    auto legal_num_black = board.get_legal_move_dist(P_BLACK, legal_dist_tmp);
    assert(legal_dist_tmp[Location::getLoc(6, 5, 9)] == 0);
}

void test_not_suicide()
{
    Board board(9, 9);
    board.playMoveAssumeLegal(5, 4, P_BLACK);
    board.playMoveAssumeLegal(4,5,P_BLACK);
    board.playMoveAssumeLegal(5,6,P_BLACK);
    board.playMoveAssumeLegal(6, 5, P_BLACK);
    board.print_board(P_BLACK);

    std::vector<int> legal_dist_tmp;
    auto legal_num_white = board.get_legal_move_dist(P_WHITE, legal_dist_tmp);
    assert(legal_dist_tmp[Location::getLoc(5,5,9)] == 0);
}

int main()
{
    test_not_capture_rule();
    test_not_suicide();
    return 0;
}