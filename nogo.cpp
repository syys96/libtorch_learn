//
// Created by syys on 2021/2/26.
//

#include "nogo.h"

Nogo::Nogo(Size n, int first_color): n(n), cur_color(first_color)  {
    board.init(n, n);
}

bool Nogo::has_legal_moves() {
    Num legal_num = board.get_leagal_moves(cur_color);
    return legal_num > 0;
}

void Nogo::execute_move(Nogo::move_type move) {
    if (!board.playMove(move, cur_color)) {
        throw StringError("illegal move!");
    }
    this->cur_color = -this->cur_color;
}

std::vector<int> Nogo::get_game_status() {
    Num curr_legal_moves = board.get_leagal_moves(cur_color);
    if (curr_legal_moves <= 0) {
        return {1, getOpp(cur_color)};
    } else {
        return {0, P_NULL};
    }
}

void Nogo::display() const {
    board.print_board(cur_color);
}

Num Nogo::get_legal_moves(std::vector<int>& legal_dist) {
    return board.get_legal_move_dist(cur_color, legal_dist);
}

