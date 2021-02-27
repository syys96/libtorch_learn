//
// Created by syys on 2021/2/25.
//

#ifndef EXAMPLE_APP_BOARD_H
#define EXAMPLE_APP_BOARD_H

#include <vector>
#include "core/global.h"

#ifndef COMPILE_MAX_BOARD_LEN
#define COMPILE_MAX_BOARD_LEN static_cast<Size>(9)
#endif

struct Board;

typedef int8_t Color;
typedef int16_t Loc;
typedef int16_t Size;
typedef int8_t Player;
typedef int16_t Num;

#define FOREACHADJ(BLOCK) {int ADJOFFSET = -(x_size+1); {BLOCK}; ADJOFFSET = -1; {BLOCK}; ADJOFFSET = 1; {BLOCK}; ADJOFFSET = x_size+1; {BLOCK}};
#define ADJ0 static_cast<Loc>(-(x_size+1))
#define ADJ1 static_cast<Loc>(-1)
#define ADJ2 static_cast<Loc>(1)
#define ADJ3 static_cast<Loc>(x_size+1)
#define FOREACHONBOARD(BLOCK) {for(Loc y = 0; y < y_size; y++) { for(Loc x = 0; x < x_size; x++) {Loc loc=Location::getLoc(x, y, x_size); {BLOCK}}}}


//Location of a point on the board
//(x,y) is represented as (x+1) + (y+1)*(x_size+1)
namespace Location
{
    Loc getLoc(Loc x, Loc y, Size x_size);
    Loc getX(Loc loc, int x_size);
    Loc getY(Loc loc, int x_size);
    Loc getLocNN(Loc x, Loc y, Size x_size);
    Loc getXNN(Loc loc, int x_size);
    Loc getYNN(Loc loc, int x_size);

    void getAdjacentOffsets(Size adj_offsets[8], Size x_size);
    bool isAdjacent(Loc loc0, Loc loc1, int x_size);
    Loc getMirrorLoc(Loc loc, int x_size, int y_size);
    Loc getCenterLoc(int x_size, int y_size);
    bool isCentral(Loc loc, int x_size, int y_size);
    int distance(Loc loc0, Loc loc1, int x_size);
    int euclideanDistanceSquared(Loc loc0, Loc loc1, int x_size);

    std::string toString(Loc loc, int x_size, int y_size);
    std::string toString(Loc loc, const Board& b);
    std::string toStringMach(Loc loc, int x_size);
    std::string toStringMach(Loc loc, const Board& b);

    bool tryOfString(const std::string& str, int x_size, int y_size, Loc& result);
    bool tryOfString(const std::string& str, const Board& b, Loc& result);
    Loc ofString(const std::string& str, int x_size, int y_size);
    Loc ofString(const std::string& str, const Board& b);

    std::vector<Loc> parseSequence(const std::string& str, const Board& b);
}

static constexpr Player P_BLACK = 1;
static constexpr Player P_WHITE = 2;
static constexpr Player P_NULL = 0;

static constexpr Color C_EMPTY = 0;
static constexpr Color C_BLACK = 1;
static constexpr Color C_WHITE = 2;
static constexpr Color C_WALL = 3;
static constexpr int NUM_BOARD_COLORS = 4;

static inline Color getOpp(Color c)
{return c ^ 3;}

class Board
{
public:
    static constexpr int MAX_LEN = COMPILE_MAX_BOARD_LEN;
    static constexpr int MAX_ARR_SIZE = (MAX_LEN+2) * (MAX_LEN+2);
    static constexpr int ACTION_MAX_SIZE = MAX_LEN * MAX_LEN;

    //Constructors---------------------------------
    Board();  //Create Board of size (9,9)
    Board(Size x, Size y); //Create Board of size (x,y)
    Board(const Board& other);

    Board& operator=(const Board&) = default;

    //Check if moving here would be a self-capture
    bool isSuicide(Loc loc, Player pla) const;
    //Check if a move at this location would be a capture of an opponent group.
    bool wouldBeCapture(Loc loc, Player pla) const;
    //Gets the number of liberties of the chain at loc. Precondition: location must be black or white.
    int getNumLiberties(Loc loc) const;
    //Check if moving here is legal.
    bool isLegal(Loc loc, Player pla) const;
    //Plays the specified move, assuming it is legal.
    void playMoveAssumeLegal(Loc loc, Player pla);
    // Plays x y
    void playMoveAssumeLegal(Loc x, Loc y, Player pla);
    //Gets the number of empty spaces directly adjacent to this location
    short getNumImmediateLiberties(Loc loc) const;
    // init board
    void init(Size xS, Size yS);
    // return x size
    Size get_xsize() const;
    // return y size;
    Size get_ysize() const;
    //Attempts to play the specified move. Returns true if successful, returns false if the move was illegal.
    bool playMove(Loc loc, Player pla);
    // play x y
    bool playMove(Loc x, Loc y, Player pla);
    // reset board
    void reset();
    // return legal moves
    Num get_leagal_moves(Player player);
    // print board
    void print_board(Player curr_player) const;
    // return legal distribution
    Num get_legal_move_dist(Player player, std::vector<int>& legal_dist);
    // print legal info
    void print_legal_dist(Player pla) const;
private:
    //Structs---------------------------------------

    //Tracks a chain/string/group of stones
    struct ChainData {
        Player owner;        //Owner of chain
        short num_locs;      //Number of stones in chain
        short num_liberties; //Number of liberties in chain
        ChainData() {owner = C_EMPTY; num_locs = 0; num_liberties = 0;}
    };

    //Data--------------------------------------------

    Size x_size;                  //Horizontal size of board
    Size y_size;                  //Vertical size of board
    Color colors[MAX_ARR_SIZE];  //Color of each location on the board.

    // updated with each move
    std::vector<bool> black_legal_dist;
    std::vector<bool> white_legal_dist;
    Num black_legal_moves;
    Num white_legal_moves;

    //Every chain of stones has one of its stones arbitrarily designated as the head.
    ChainData chain_data[MAX_ARR_SIZE]; //For each head stone, the chaindata for the chain under that head. Undefined otherwise.
    Loc chain_head[MAX_ARR_SIZE];       //Where is the head of this chain? Undefined if EMPTY or WALL
    Loc next_in_chain[MAX_ARR_SIZE];    //Location of next stone in chain. Circular linked list. Undefined if EMPTY or WALL

    Size adj_offsets[8]; //Indices 0-3: Offsets to add for adjacent points. Indices 4-7: Offsets for diagonal points. 2 and 3 are +x and +y.

    void mergeChains(Loc loc1, Loc loc2);
    bool isLibertyOf(Loc loc, Loc head) const;
    void update_blank_legality(const std::vector<Loc>& locs);
};

#endif //EXAMPLE_APP_BOARD_H
