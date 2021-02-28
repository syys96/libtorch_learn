//
// Created by syys on 2021/2/26.
//

#ifndef EXAMPLE_APP_NOGO_H
#define EXAMPLE_APP_NOGO_H

#include <tuple>
#include <vector>
#include "core/global.h"
#include "board.h"
#include <torch/torch.h>
#include <torch/script.h>

// 前向声明
class Playerm;

class Nogo {
public:
    using move_type = Loc;
    using board_type = Board;

    explicit Nogo(Size n, Player first_color=P_BLACK);

    bool has_legal_moves();
    Num get_legal_moves(std::vector<int>& legal_dist);
    bool execute_move(move_type move);
    std::vector<int> get_game_status();
    at::Tensor curr_state(bool to_device, torch::Device &device);
    void display() const;
    static char get_symbol(Player player);
    int start_play(Playerm *player1, Playerm *player2, bool swap=false, bool show=false);

    inline Size get_action_size() const { return this->n * this->n; }
    inline board_type get_board() const { return this->board; }
    inline Player get_current_color() const { return this->cur_color; }
    inline Size get_n() const { return this->n; }
    inline Size get_action_dim() const { return this->n*this->n; }
    void reset();

private:
    board_type board;      // game borad
    Size n;        // board size
    Player cur_color;       // current player's color
};

class Playerm
{
public:
    inline explicit Playerm(Player player = P_BLACK) :player(player) {}
    inline ~Playerm() = default;
    inline void set_player(Player player) { this->player = player; }
    virtual void init() {}
    virtual void update_with_move(Loc last_move) {}
    inline int get_player() const { return this->player; }
    virtual Loc get_action(Nogo *nogo, bool explore = false) = 0;
private:
    Player player;
};

class Human : public Playerm
{
public:
    inline explicit Human(Player player = 1) :Playerm(player) {}
    inline ~Human() = default;
    inline Loc get_action(Nogo *nogo, bool explore = false) override
    {
        Size n = nogo->get_n(), i, j;
        while (true)
        {
            std::cin >> i >> j;
            std::cin.clear();
            if (i >= 0 && i < n && j >= 0 && j < n) break;
            else std::cout << "Illegal input. Reenter : ";
        }
        return Location::getLoc(i,j,n);
    }
};

#endif //EXAMPLE_APP_NOGO_H
