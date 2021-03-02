//
// Created by syys on 2021/3/1.
//

#include "../nogo.h"
#include "../mcts.h"
#include <torch/torch.h>
#include <torch/script.h>
#include "gtest/gtest.h"

TEST(test_copy, test_copy_board)
{
    Size size=COMPILE_MAX_BOARD_LEN;
    Nogo nogo(size);

    nogo.execute_move(Location::getLoc(1,1,size));
    nogo.execute_move(Location::getLoc(1,2,size));
    nogo.execute_move(Location::getLoc(1,3,size));

    Nogo* nogo_ptr = &nogo;
    Nogo game = *nogo_ptr;
    ASSERT_EQ(game.get_n(), nogo_ptr->get_n());
    ASSERT_EQ(game.get_current_color(), nogo_ptr->get_current_color());
    ASSERT_EQ(game.get_board(), nogo_ptr->get_board());
}
