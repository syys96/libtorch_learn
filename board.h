//
// Created by syys on 2021/2/25.
//

#ifndef EXAMPLE_APP_BOARD_H
#define EXAMPLE_APP_BOARD_H

#include <vector>

typedef int8_t Color;
typedef short Loc;
typedef int8_t Player;

static constexpr Player P_BLACK = 1;
static constexpr Player P_WHITE = 2;

static constexpr int BOARD_SIZE = 9;
static constexpr int BOARD_ARR_SIZE = (BOARD_SIZE+1) * (BOARD_SIZE+1);
static constexpr int ACTION_MAX_SIZE = BOARD_SIZE * BOARD_SIZE;

static constexpr Color C_EMPTY = 0;
static constexpr Color C_BLACK = 1;
static constexpr Color C_WHITE = 2;
static constexpr Color C_NULL = 3;
static constexpr int NUM_BOARD_COLORS = 4;

static inline Color getOpp(Color c)
{return c ^ 3;}

class Board
{
public:
    //Constructors---------------------------------
    Board();  //Create Board of size (9,9)
    Board(int x, int y); //Create Board of size (x,y)
    Board(const Board& other);

    //Check if moving here would be a self-capture
    bool isSuicide(Loc loc, Player pla) const;
    //Check if a move at this location would be a capture of an opponent group.
    bool wouldBeCapture(Loc loc, Player pla) const;
    //Gets the number of liberties of the chain at loc. Precondition: location must be black or white.
    int getNumLiberties(Loc loc) const;
    //Check if moving here is legal.
    bool isLegal(Loc loc, Player pla) const;

private:
    //Structs---------------------------------------

    //Tracks a chain/string/group of stones
    struct ChainData {
        Player owner;        //Owner of chain
        short num_locs;      //Number of stones in chain
        short num_liberties; //Number of liberties in chain
    };

    //Data--------------------------------------------

    int x_size;                  //Horizontal size of board
    int y_size;                  //Vertical size of board
    Color colors[BOARD_ARR_SIZE];  //Color of each location on the board.

    //Every chain of stones has one of its stones arbitrarily designated as the head.
    ChainData chain_data[BOARD_ARR_SIZE]; //For each head stone, the chaindata for the chain under that head. Undefined otherwise.
    Loc chain_head[BOARD_ARR_SIZE];       //Where is the head of this chain? Undefined if EMPTY or WALL
    Loc next_in_chain[BOARD_ARR_SIZE];    //Location of next stone in chain. Circular linked list. Undefined if EMPTY or WALL

};

#endif //EXAMPLE_APP_BOARD_H
