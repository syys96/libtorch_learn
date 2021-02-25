//
// Created by syys on 2021/2/25.
//

#ifndef EXAMPLE_APP_NOGO_H
#define EXAMPLE_APP_NOGO_H

#include <cstdint>
#include <vector>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include "board.h"

class Players;

class Nogo
{
public:
    Nogo(Size n);
    void reset();
    int get_legal_move(bool* move_legal);
    bool execute_move(Loc move);
    std::vector<int> get_game_status();
    at::Tensor curr_state(bool to_device, torch::Device &device);
    void display();
    static char get_symbol(Player player);
    Size get_n() const;
    Size get_action_dim() const;
    inline int get_curr_player() const { return this->curr_player; }
    int start_play(Player *player1, Player *player2, bool swap=false, bool show=false);

private:
    Board board;
    uint32_t n_count;
    Player curr_player;
};

class Players
{
public:
    inline Player(int player = 1) :player(player) {}
    inline ~Player() {}
    inline void set_player(int player) { this->player = player; }
    virtual void init() {}
    virtual void update_with_move(int last_move) {}
    inline int get_player() const { return this->player; }
    virtual uint32_t get_action(Gomoku *gomoku, bool explore = false) = 0;
private:
    int player;
};

#endif //EXAMPLE_APP_NOGO_H
