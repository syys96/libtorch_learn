//
// Created by syys on 2021/2/26.
//

#include "nogo.h"

Nogo::Nogo(Size n, Player first_color): n(n), cur_color(first_color)  {
    board.init(n, n);
}

bool Nogo::has_legal_moves() {
    Num legal_num = board.get_leagal_moves(cur_color);
    return legal_num > 0;
}

bool Nogo::execute_move(Nogo::move_type move) {
    if (!board.playMove(move, cur_color)) {
        throw StringError("illegal move when execute move!");
        return false;
    }
    this->cur_color = getOpp(this->cur_color);
    return true;
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

void Nogo::reset() {
    cur_color = P_BLACK;
    board.init(n, n);
}

at::Tensor Nogo::curr_state(bool to_device, torch::Device &device) {
    // 获取状态作为神经网络的输入  [batch channels height width]
    // 当前玩家视角
    Size size = this->n;
    std::vector<int> temp;
    this->board.get_color_NN(temp);
    at::Tensor s, boardm;
    if (to_device)
    {
        boardm = torch::tensor(temp, device).toType(torch::kInt).reshape({ size,size });
        s = torch::zeros({ 1,2,size,size }, device).toType(torch::kFloat);
    }
    else
    {
        boardm = torch::tensor(temp, torch::kInt).reshape({ size,size });
        s = torch::zeros({ 1,2,size,size }, torch::kFloat);
    }
    Player a = this->get_current_color();
    // 当前玩家的棋子
    s[0][0] = boardm.eq(a == P_BLACK ? NN_BLACK : NN_WHITE);
    // 对手玩家的棋子
    s[0][1] = boardm.eq(getOpp(a) == P_BLACK ? NN_BLACK : NN_WHITE);
    return s;
}

char Nogo::get_symbol(Player player) {
    if (player == P_BLACK) return 'X';
    else if (player == P_WHITE) return 'O';
    else return '_';
}

