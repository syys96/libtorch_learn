//
// Created by syys on 2021/3/1.
//

#include "torch/script.h"
#include "torch/torch.h"
#include "../nogo.h"
#include "gtest/gtest.h"

TEST(test_torch_tensor, test_torch_tensor_test_reshape)
{
    Size size = 9;
    std::vector<Loc> board_map(size*size, 0);
    Loc x = 2; Loc y = 5;
    Loc loc = Location::getLoc(x, y, size);
    Loc locNN = Location::getLocNN(x, y, size);
    Loc locNNMir = Location::getLocNN(y, x, size);
    board_map[locNN] = 1;
    board_map[locNNMir] = -1;
    auto tensorm = torch::tensor(board_map).toType(torch::kInt).reshape({ size,size }).transpose(0, 1);
    Loc xNN = Location::getXNN(locNN, size);
    Loc yNN = Location::getYNN(locNN, size);
    ASSERT_EQ(x, xNN);
    ASSERT_EQ(y, yNN);

    std::cout << tensorm[xNN][yNN] << std::endl;
    std::cout << tensorm[yNN][xNN] << std::endl;
//    ASSERT_EQ(tensorm[xNN][yNN].item(), 1);
//    ASSERT_EQ(tensorm[yNN][xNN].item(), -1);

}
