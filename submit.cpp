//
// Created by syys on 2021/3/6.
//

#include "mcts.h"
#include "train.h"

int main()
{
    TimeCounter tc;
    Nogo nogo(COMPILE_MAX_BOARD_LEN);
    nogo.parse_botzone_input();
    static const char* other_nn_path = "./data/model-best.pt";
    uint32_t state_c = 2;
    uint32_t n_thread = 1;
    uint32_t c_puct=5;
    double temp=1e-3;
    uint32_t n_simulate=200;
    double virtual_loss=3;
    PolicyValueNet network_local(other_nn_path, true, state_c,
                                 nogo.get_n(), nogo.get_action_dim());
    tc.start();
    MCTS mcts_train(&network_local, n_thread, c_puct, temp, n_simulate,
                    virtual_loss, nogo.get_action_dim(), true);
    double nn_load_time = tc.end_s();
    mcts_train.reset();
    tc.start();
    Loc move = mcts_train.get_action(&nogo);
    double decison_time = tc.end_s();
    Loc outx = Location::getXNN(move, COMPILE_MAX_BOARD_LEN);
    Loc outy = Location::getYNN(move, COMPILE_MAX_BOARD_LEN);
    std::cout << outx << " " << outy << std::endl;
    std::cout << "nn load time: " << nn_load_time
                << ", decide time: " << decison_time << std::endl;
    return 0;
}