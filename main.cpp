#include <torch/torch.h>
#include <iostream>
#include "tests/thread_pool_test.h"

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    std::cout << torch::cuda::is_available() << std::endl;

    test_thread_pool();
}