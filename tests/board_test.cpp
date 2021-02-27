//
// Created by syys on 2021/2/26.
//

#include "board_test.h"

TEST(test_board, test_not_capture_rule)
{
    Board board(9, 9);
    board.playMoveAssumeLegal(5, 4, P_BLACK);
    board.playMoveAssumeLegal(4,5,P_BLACK);
    board.playMoveAssumeLegal(5,6,P_BLACK);
    board.playMoveAssumeLegal(5, 5, P_WHITE);
    board.print_board(P_BLACK);
    board.print_legal_dist(P_BLACK);

    std::vector<int> legal_dist_tmp;
    auto legal_num_black = board.get_legal_move_dist(P_BLACK, legal_dist_tmp);
    ASSERT_EQ(legal_dist_tmp[Location::getLocNN(6, 5, 9)], 0);
    for (Loc ynn = 0; ynn < 9; ynn++) {
        for (Loc xnn = 0; xnn < 9; xnn++) {
            Loc locnn = Location::getLocNN(xnn,ynn,9);
            std::cout << legal_dist_tmp[locnn] << " ";
        }
        std::cout << std::endl;
    }
}

TEST(test_board, test_not_suicide)
{
    Board board(9, 9);
    board.playMoveAssumeLegal(5, 4, P_BLACK);
    board.playMoveAssumeLegal(4,5,P_BLACK);
    board.playMoveAssumeLegal(5,6,P_BLACK);
    board.playMoveAssumeLegal(6, 5, P_BLACK);
    board.print_board(P_BLACK);
    board.print_legal_dist(P_WHITE);

    std::vector<int> legal_dist_tmp;
    auto legal_num_tmp = board.get_legal_move_dist(P_WHITE, legal_dist_tmp);
    ASSERT_EQ(legal_dist_tmp[Location::getLocNN(5,5,9)], 0);
    for (Loc ynn = 0; ynn < 9; ynn++) {
        for (Loc xnn = 0; xnn < 9; xnn++) {
            Loc locnn = Location::getLocNN(xnn,ynn,9);
            std::cout << legal_dist_tmp[locnn] << " ";
        }
        std::cout << std::endl;
    }
}

void test_legal_dist()
{
    Board board(9, 9);
    board.playMoveAssumeLegal(5, 4, P_BLACK);
    board.playMoveAssumeLegal(6, 4, P_BLACK);
    board.playMoveAssumeLegal(4,5,P_BLACK);
    board.playMoveAssumeLegal(5,6,P_BLACK);
    board.playMoveAssumeLegal(5, 5, P_WHITE);
    board.playMoveAssumeLegal(4, 4, P_WHITE);
    board.playMoveAssumeLegal(4, 3, P_WHITE);
    board.playMoveAssumeLegal(5, 3, P_WHITE);
    board.playMoveAssumeLegal(6, 3, P_WHITE);
    board.playMoveAssumeLegal(7, 3, P_WHITE);
    board.playMoveAssumeLegal(7, 4, P_WHITE);
    board.playMoveAssumeLegal(7, 5, P_WHITE);
    board.playMoveAssumeLegal(6, 6, P_WHITE);
    board.print_board(P_BLACK);
    board.print_legal_dist(P_BLACK);
    board.print_legal_dist(P_WHITE);

}

int main()
{
//    test_legal_dist();
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}