#include <torch/torch.h>
#include <iostream>
#include "tests/thread_pool_test.h"
#include "botzone/parase_input.h"

int main() {
//    torch::Tensor tensor = torch::rand({2, 3});
//    std::cout << tensor << std::endl;
//    std::cout << torch::cuda::is_available() << std::endl;

//    test_thread_pool();
    Board main_board(9, 9);
    Player me = parase_jsonString(main_board);
    main_board.print_board(me);
    main_board.print_legal_dist(me);
    main_board.print_legal_dist(getOpp(me));
}