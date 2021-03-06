//
// Created by syys on 2021/2/28.
//

#include "policy_value_net.h"

void PolicyValueNet::save_model_cpu(const char *save_path) {
    this->model->to(torch::kCPU);
    torch::save(this->model, save_path);
    this->model->to(this->device);
}
