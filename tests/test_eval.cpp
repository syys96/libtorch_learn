//
// Created by syys on 2021/3/2.
//

//
// Created by syys on 2021/2/28.
//

#include "../train.h"

int main()
{
    Train train;
    train.evaluate(best_path, 50);
    train.run(model_path, best_path);
    return 0;
}