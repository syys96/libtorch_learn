//
// Created by syys on 2021/2/25.
//

#include "nogo.h"

Nogo::Nogo(Size n) {
    board.init(n, n);
    n_count = 0;
    curr_player = P_BLACK;
}

Size Nogo::get_n() const {
    return board.get_xsize();
}

Size Nogo::get_action_dim() const {
    Size x_size = board.get_xsize();
    return x_size * x_size;
}

bool Nogo::execute_move(Loc move) {
    board.playMove(move, curr_player);
    n_count++;
    curr_player = getOpp(curr_player);
}

int Nogo::get_legal_move(bool* move_legal) {
    auto x_size = board.get_xsize();
    auto y_size = board.get_ysize();
    int legal_num = 0;
    for(int y = 0; y < y_size; y++) {
        for(int x = 0; x < x_size; x++) {
            Loc loc = (x+1) + (y+1)*(x_size+1);
            move_legal[loc] = board.isLegal(loc, curr_player);
            legal_num += (move_legal[loc] ? 1 : 0);
        }
    }
    return legal_num;
}

void Nogo::reset() {
    n_count = 0;
    curr_player = P_BLACK;
    board.reset();
}

char Nogo::get_symbol(Player player) {
    if (player == P_BLACK) return 'X';
    else if (player == P_WHITE) return 'O';
    else return '_';
}

