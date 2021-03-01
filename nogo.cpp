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
        std::cout << "ERRORAAAAAAAA" << std::endl;
        board.print_board(cur_color);
        board.print_legal_dist(cur_color);
        board.print_legal_dist(getOpp(cur_color));
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
    temp.resize(get_action_dim(), 0);
    this->board.get_color_NN(temp);
    at::Tensor s, boardm;
    if (to_device)
    {
        boardm = torch::tensor(temp, device).toType(torch::kInt).reshape({ size,size }).transpose(0, 1);
        s = torch::zeros({ 1,2,size,size }, device).toType(torch::kFloat);
    }
    else
    {
        boardm = torch::tensor(temp, torch::kInt).reshape({ size,size }).transpose(0, 1);
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

int Nogo::start_play(Playerm *player1, Playerm *player2, bool swap, bool show)
{
    // 默认第一个参数为先手(Black)
    if (nullptr == player1 || nullptr == player2) return 0;
    Player idx = swap ? P_WHITE : P_BLACK;	// 交换先后手
    player1->set_player(idx);
    player1->init();
    player2->set_player(getOpp(idx));
    player2->init();
    Playerm * players[2] = { player1,player2 };
    idx = swap ? 1 : 0; // 总是黑方先下，区别在于是player1还是player2下黑子
    if (players[idx]->get_player() != P_BLACK) {
        throw std::runtime_error("play not inited correctly!");
    }
    Loc move;
    this->reset();
    if (cur_color != P_BLACK) {
        throw std::runtime_error("init color is not black in eval!");
    }
    std::vector<int> res(2, 0);
    if (show)
    {
        std::cout << "New game." << std::endl;
        this->display();
    }
    while (0 == res[0])
    {
        if (show)
        {
            std::printf("Player '%c' (example: 0 0):", this->get_symbol(players[idx]->get_player()));
        }
        move = players[idx]->get_action(this);
        if (this->execute_move(Location::LocNN2Loc(move, n)))
        {
            if (show)
            {
                Loc nnx = Location::getXNN(move, n);
                Loc nny = Location::getYNN(move, n);
                std::printf("Player '%c' : %d %d\n", this->get_symbol(players[idx]->get_player()), nnx, nny);
            }
            // 你如何确定这两个player的root节点对应的board是同步的？
            // 在只有一个player执行update的情况下？
            players[idx]->update_with_move(move);
            players[1-idx]->update_with_move(move);

            res = this->get_game_status();
            idx = 1 - idx;
            if (show) this->display();
        }
    }
    // 玩家重置
    player1->update_with_move(-1);
    player2->update_with_move(-1);
    if (show)
    {
        if (0 != res[1]) std::printf("Game end. Winner is Player '%c'.\n", this->get_symbol(res[1]));
        else std::cout << "Game end. Tie." << std::endl;
    }
    return res[1];
}

//Nogo& Nogo::operator=(const Nogo &) {
//    Nogo copy(n, cur_color);
//    copy.board = board;
//    copy.n = n;
//    copy.cur_color = cur_color;
//    return copy;
//}

