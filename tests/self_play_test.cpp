//
// Created by syys on 2021/3/1.
//

#include "../nogo.h"
#include "../mcts.h"
#include <torch/torch.h>
#include <torch/script.h>

int main()
{
    const char *model_path = "../model/model-checkpoint.pt";
    const char *best_path = "../model/model-best.pt";
    Size size=COMPILE_MAX_BOARD_LEN; uint32_t state_c=2; uint32_t n_thread=1; double lr=4e-3; double c_lr=1;
    double temp=1; uint32_t n_simulate=5;
    uint32_t c_puct=5; double virtual_loss=3;
    Nogo nogo(size);
    PolicyValueNet network(best_path, true, state_c, size, size*size);
    MCTS mcts(&network, n_thread, c_puct, temp, n_simulate, virtual_loss, size*size, true);
    std::vector<at::Tensor> states_local;
    std::vector<at::Tensor> probs_local;
    std::vector<float> values_local;
    uint32_t n_round = 20;
    bool add_noise = true;
    bool show = true;

//    mcts.simulate(&nogo);

    mcts.self_play(&nogo, states_local, probs_local, values_local, temp,
            n_round, add_noise, show);

}