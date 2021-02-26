//
// Created by syys on 2021/2/26.
//

#ifndef EXAMPLE_APP_NOGO_H
#define EXAMPLE_APP_NOGO_H

#include <tuple>
#include <vector>
#include "core/global.h"
#include "board.h"

class Nogo {
public:
    using move_type = Loc;
    using board_type = Board;

    Nogo(Size n, int first_color);

    bool has_legal_moves();
    std::vector<int> get_legal_moves();
    void execute_move(move_type move);
    std::vector<int> get_game_status();
    void display() const;

    inline unsigned int get_action_size() const { return this->n * this->n; }
    inline board_type get_board() const { return this->board; }
    inline move_type get_last_move() const { return this->last_move; }
    inline int get_current_color() const { return this->cur_color; }
    inline unsigned int get_n() const { return this->n; }

private:
    board_type board;      // game borad
    unsigned int n;        // board size
    int cur_color;       // current player's color
};

#endif //EXAMPLE_APP_NOGO_H
