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
            loc < MAX_ARR_SIZE &&
            (colors[loc] == C_EMPTY) &&
            !wouldBeCapture(loc, pla) &&
            !isSuicide(loc, pla);
}

Board::Board() {
    init(9,9);
}

Board::Board(Size x, Size y) {
    init(x,y);
}

Board::Board(const Board &other) {
    x_size = other.x_size;
    y_size = other.y_size;

    memcpy(colors, other.colors, sizeof(Color)*MAX_ARR_SIZE);
    memcpy(chain_data, other.chain_data, sizeof(ChainData)*MAX_ARR_SIZE);
    memcpy(chain_head, other.chain_head, sizeof(Loc)*MAX_ARR_SIZE);
    memcpy(next_in_chain, other.next_in_chain, sizeof(Loc)*MAX_ARR_SIZE);

    memcpy(adj_offsets, other.adj_offsets, sizeof(short)*8);
}

void Board::init(Size xS, Size yS)
{
    if(xS < 0 || yS < 0 || xS > MAX_LEN || yS > MAX_LEN)
        throw StringError("Board::init - invalid board size");

    x_size = xS;
    y_size = yS;

    for(int i = 0; i < MAX_ARR_SIZE; i++)
        colors[i] = C_WALL;

    for(int y = 0; y < y_size; y++)
    {
        for(int x = 0; x < x_size; x++)
        {
            Loc loc = (x+1) + (y+1)*(x_size+1);
            colors[loc] = C_EMPTY;
            // empty_list.add(loc);
        }
    }

    Location::getAdjacentOffsets(adj_offsets,x_size);
}

void Location::getAdjacentOffsets(Size adj_offsets[8], Size x_size)
{
    adj_offsets[0] = -(x_size+1);
    adj_offsets[1] = -1;
    adj_offsets[2] = 1;
    adj_offsets[3] = (x_size+1);
    adj_offsets[4] = -(x_size+1)-1;
    adj_offsets[5] = -(x_size+1)+1;
    adj_offsets[6] = (x_size+1)-1;
    adj_offsets[7] = (x_size+1)+1;
}