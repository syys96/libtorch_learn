//
// Created by syys on 2021/3/6.
//

#include "policy_value_net.h"
#include "train.h"

int main()
{
    uint32_t state_c=2;
    Size size=COMPILE_MAX_BOARD_LEN;
    PolicyValueNet network(best_path, true, state_c, size, size*size);
    network.save_model_cpu(best_path_cpu);
    return 0;
}