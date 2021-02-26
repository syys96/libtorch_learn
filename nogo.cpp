//
// Created by syys on 2021/2/26.
//

#include "nogo.h"

Nogo::Nogo(Size n, int first_color): n(n), cur_color(first_color)  {
    board.init(n, n);
}

