//
// Created by syys on 2021/2/25.
//

#include "board.h"

#define FOREACHADJ(BLOCK) {int ADJOFFSET = -(x_size+1); {BLOCK}; ADJOFFSET = -1; {BLOCK}; ADJOFFSET = 1; {BLOCK}; ADJOFFSET = x_size+1; {BLOCK}};


bool Board::isSuicide(Loc loc, Player pla) const {
    Player opp = getOpp(pla);
    FOREACHADJ(
            Loc adj = loc + ADJOFFSET;

            if(colors[adj] == C_EMPTY)
                return false;
            else if(colors[adj] == pla)
            {
                if(getNumLiberties(adj) > 1)
                    return false;
            }
            else if(colors[adj] == opp)
            {
                if(getNumLiberties(adj) == 1)
                    return false;
            }
    );

    return true;
}

int Board::getNumLiberties(Loc loc) const {
    return chain_data[chain_head[loc]].num_liberties;
}

bool Board::wouldBeCapture(Loc loc, Player pla) const {
    if(colors[loc] != C_EMPTY)
        return false;
    Player opp = getOpp(pla);
    FOREACHADJ(
            Loc adj = loc + ADJOFFSET;
            if(colors[adj] == opp)
            {
                if(getNumLiberties(adj) == 1)
                    return true;
            }
    );

    return false;
}

bool Board::isLegal(Loc loc, Player pla) const {
    if(pla != P_BLACK && pla != P_WHITE)
        return false;
    return  loc >= 0 &&
            loc < BOARD_ARR_SIZE &&
            (colors[loc] == C_EMPTY) &&
            !wouldBeCapture(loc, pla) &&
            !isSuicide(loc, pla);
}

Board::Board() {
    init(9,9);
}

Board::Board(int x, int y) {
    init(x,y);
}
