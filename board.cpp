//
// Created by syys on 2021/2/25.
//

#include "board.h"

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
    init(COMPILE_MAX_BOARD_LEN, COMPILE_MAX_BOARD_LEN);
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

    black_legal_dist = other.black_legal_dist;
    white_legal_dist = other.white_legal_dist;

    memcpy(adj_offsets, other.adj_offsets, sizeof(Size)*8);
}

void Board::init(Size xS, Size yS)
{
    if(xS < 0 || yS < 0 || xS > MAX_LEN || yS > MAX_LEN)
        throw StringError("Board::init - invalid board size");

    x_size = xS;
    y_size = yS;

    std::fill(colors, colors+MAX_ARR_SIZE, C_WALL);
    white_legal_dist.resize(MAX_ARR_SIZE, false);
    black_legal_dist.resize(MAX_ARR_SIZE, false);
    std::fill(white_legal_dist.begin(), white_legal_dist.end(), false);
    std::fill(black_legal_dist.begin(), black_legal_dist.end(), false);

    FOREACHONBOARD(
            colors[loc] = C_EMPTY;
            black_legal_dist[loc] = true;
            white_legal_dist[loc] = true;
            )

    black_legal_moves = xS * yS;
    white_legal_moves = xS * yS;

    std::fill(chain_head, chain_head+MAX_ARR_SIZE, C_EMPTY);
    std::fill(next_in_chain, next_in_chain+MAX_ARR_SIZE, C_EMPTY);
    std::fill(chain_data, chain_data+MAX_ARR_SIZE, ChainData());

    Location::getAdjacentOffsets(adj_offsets,x_size);
}

void Board::playMoveAssumeLegal(Loc loc, Player pla)
{
    Player opp = getOpp(pla);

    //Add the new stone as an independent group
    colors[loc] = pla;
    chain_data[loc].owner = pla;
    chain_data[loc].num_locs = 1;
    chain_data[loc].num_liberties = getNumImmediateLiberties(loc);
    chain_head[loc] = loc;
    next_in_chain[loc] = loc;

    //Merge with surrounding friendly chains and capture any necessary opp chains
    int num_opps_seen = 0;  //How many opp chains we have seen so far
    Loc opp_heads_seen[4];   //Heads of the opp chains seen so far

    for(int i = 0; i < 4; i++)
    {
        int adj = loc + adj_offsets[i];

        //Friendly chain!
        if(colors[adj] == pla)
        {
            //Already merged?
            if(chain_head[adj] == chain_head[loc])
                continue;

            //Otherwise, eat one liberty and merge them
            chain_data[chain_head[adj]].num_liberties--;
            mergeChains(adj,loc);
        }

        //Opp chain!
        else if(colors[adj] == opp)
        {
            Loc opp_head = chain_head[adj];

            //Have we seen it already?
            bool seen = false;
            for(int j = 0; j<num_opps_seen; j++)
                if(opp_heads_seen[j] == opp_head)
                {seen = true; break;}

            if(seen)
                continue;

            //Not already seen! Eat one liberty from it and mark it as seen
            chain_data[opp_head].num_liberties--;
            opp_heads_seen[num_opps_seen++] = opp_head;

            //Kill it?
            if(getNumLiberties(adj) == 0)
            {
                throw StringError("capture opp stone is illegal in nogo!");
            }
        }
    }

    //Handle suicide
    if(getNumLiberties(loc) == 0) {
        throw StringError("suicide is illegal in nogo!");
    }

    // update blanks
    black_legal_dist[loc] = false;
    white_legal_dist[loc] = false;
    // make move will eat one blank for both players.
    black_legal_moves--;
    white_legal_moves--;
    std::vector<Loc> to_update{loc, static_cast<Loc>(loc+ADJ0),
                               static_cast<Loc>(loc+ADJ1),
                               static_cast<Loc>(loc+ADJ2),
                               static_cast<Loc>(loc+ADJ3)};
    update_blank_legality(to_update);
}

short Board::getNumImmediateLiberties(Loc loc) const {
    short num_libs = 0;
    if(colors[loc + ADJ0] == C_EMPTY) num_libs++;
    if(colors[loc + ADJ1] == C_EMPTY) num_libs++;
    if(colors[loc + ADJ2] == C_EMPTY) num_libs++;
    if(colors[loc + ADJ3] == C_EMPTY) num_libs++;

    return num_libs;
}

void Board::mergeChains(Loc loc1, Loc loc2)
{
    //Find heads
    Loc head1 = chain_head[loc1];
    Loc head2 = chain_head[loc2];

    assert(head1 != head2);
    assert(chain_data[head1].owner == chain_data[head2].owner);

    //Make sure head2 is the smaller chain.
    if(chain_data[head1].num_locs < chain_data[head2].num_locs)
    {
        Loc temp = head1;
        head1 = head2;
        head2 = temp;
    }

    //Iterate through each stone of head2's chain to make it a member of head1's chain.
    //Count new liberties for head1 as we go.
    chain_data[head1].num_locs += chain_data[head2].num_locs;
    int numNewLiberties = 0;
    Loc loc = head2;
    while(true)
    {
        //Any adjacent liberty is a new liberty for head1 if it is not adjacent to a stone of head1
        FOREACHADJ(
                Loc adj = loc + ADJOFFSET;
                if(colors[adj] == C_EMPTY && !isLibertyOf(adj,head1))
                    numNewLiberties++;
        );

        //Now, add this stone to head1.
        chain_head[loc] = head1;

        //If are not back around, we are done.
        if(next_in_chain[loc] != head2)
            loc = next_in_chain[loc];
        else
            break;
    }

    //Add in the liberties
    chain_data[head1].num_liberties += numNewLiberties;

    //We link up (head1 -> next1 -> ... -> last1 -> head1) and (head2 -> next2 -> ... -> last2 -> head2)
    //as: head1 -> head2 -> next2 -> ... -> last2 -> next1 -> ... -> last1 -> head1
    //loc is now last_2
    next_in_chain[loc] = next_in_chain[head1];
    next_in_chain[head1] = head2;
}

bool Board::isLibertyOf(Loc loc, Loc head) const
{
    Loc adj;
    adj = loc + ADJ0;
    if(colors[adj] == colors[head] && chain_head[adj] == head)
        return true;
    adj = loc + ADJ1;
    if(colors[adj] == colors[head] && chain_head[adj] == head)
        return true;
    adj = loc + ADJ2;
    if(colors[adj] == colors[head] && chain_head[adj] == head)
        return true;
    adj = loc + ADJ3;
    if(colors[adj] == colors[head] && chain_head[adj] == head)
        return true;

    return false;
}

Size Board::get_xsize() const {
    return x_size;
}

bool Board::playMove(Loc loc, Player pla) {
    if(isLegal(loc,pla))
    {
        playMoveAssumeLegal(loc,pla);
        return true;
    }
    return false;
}

Size Board::get_ysize() const {
    return y_size;
}

void Board::reset() {
    std::fill(colors, colors+MAX_ARR_SIZE, C_WALL);
    std::fill(white_legal_dist.begin(), white_legal_dist.end(), false);
    std::fill(black_legal_dist.begin(), black_legal_dist.end(), false);

    FOREACHONBOARD(
            colors[loc] = C_EMPTY;
            black_legal_dist[loc] = true;
            white_legal_dist[loc] = true;
            )

    black_legal_moves = x_size * y_size;
    white_legal_moves = x_size * y_size;

    std::fill(chain_head, chain_head+MAX_ARR_SIZE, C_EMPTY);
    std::fill(next_in_chain, next_in_chain+MAX_ARR_SIZE, C_EMPTY);
    std::fill(chain_data, chain_data+MAX_ARR_SIZE, ChainData());
}

void Board::update_blank_legality(const std::vector<Loc> &locs) {

    //Update the blanks near each chain of loc in locs
    int num_chain_seen = 0;  //How many chains we have seen so far
    Loc heads_chain_seen[4];   //Heads of the chains seen so far
    bool blank_seen_dist[MAX_ARR_SIZE];
    std::fill(blank_seen_dist, blank_seen_dist+MAX_ARR_SIZE, false);
    for (auto& loc : locs) {
        if (colors[loc] == C_EMPTY || colors[loc] == C_WALL)
            continue;

        Loc m_head = chain_head[loc];
        //Have we seen this chain already?
        bool seen = false;
        for(int j = 0; j<num_chain_seen; j++)
            if(heads_chain_seen[j] == m_head)
                {seen = true; break;}
        if(seen)
            continue;

        // iter in the chain and update every blank near the stone of this chain
        // which is based on the fact that the legality of blanks not near the stone of this chain
        // will not be affected by the make move action.
        Loc stone_in_chain = m_head;
        do {
            FOREACHADJ(
                    Loc adj = stone_in_chain + ADJOFFSET;
                    // is this loc blank and we haven't updated it?
                    if (colors[adj] == C_EMPTY && !blank_seen_dist[adj]) {
                        bool old_black_legal = black_legal_dist[adj];
                        bool old_white_legal = white_legal_dist[adj];
                        black_legal_dist[adj] = isLegal(adj, P_BLACK);
                        white_legal_dist[adj] = isLegal(adj, P_WHITE);
                        if (old_black_legal && !black_legal_dist[adj]) {
                            black_legal_moves--;
                        }
                        if (!old_black_legal && black_legal_dist[adj]) {
                            black_legal_moves++;
                        }
                        if (old_white_legal && !white_legal_dist[adj]) {
                            white_legal_moves--;
                        }
                        if (!old_white_legal && white_legal_dist[adj]) {
                            white_legal_moves++;
                        }
                        blank_seen_dist[adj] = true;
                    }
                    );
            stone_in_chain = next_in_chain[stone_in_chain];
        } while (stone_in_chain != m_head);
    }
}

Num Board::get_leagal_moves(Player player) {
    return player == P_BLACK ? black_legal_moves : white_legal_moves;
}

void Board::print_board(Player curr_player) const {
    std::cout << "---------------------" << std::endl;
    std::cout << "Board info, X(Black), O(Whitee), -(Empty)" << std::endl;
    std::cout << "Black num: " << black_legal_moves << ", White num: " << white_legal_moves << std::endl;
    std::cout << "Player to move now: " << (curr_player == P_BLACK ? "Black(X)" : "White(O)") << std::endl;
    std::cout << "  ";
    for (Size i = 0; i < x_size; i++) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    for(Size y = 0; y < y_size; y++) {
        std::cout << y << " ";
        for(Size x = 0; x < x_size; x++) {
            Loc loc = (x+1) + (y+1)*(x_size+1);
            std::cout << (colors[loc] == C_BLACK ? "X " :
                            (colors[loc] == C_WHITE ? "O " : "- "));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "---------------------" << std::endl;
}

Num Board::get_legal_move_dist(Player player, std::vector<int>& legal_dist) {
    legal_dist.resize(x_size*y_size, -1);
    const auto& legal_bool = player == P_BLACK ? black_legal_dist : white_legal_dist;
    FOREACHONBOARD(
            Loc locNN = Location::getLocNN(x, y, x_size);
            legal_dist[locNN] = legal_bool[loc] ? 1 : 0;
            )
    return player == P_BLACK ? black_legal_moves : white_legal_moves;
}

bool Board::playMove(Loc x, Loc y, Player pla) {
    Loc loc = Location::getLoc(x, y, x_size);
    if(isLegal(loc,pla))
    {
        playMoveAssumeLegal(loc,pla);
        return true;
    }
    return false;
}

void Board::playMoveAssumeLegal(Loc x, Loc y, Player pla) {
    Loc loc = Location::getLoc(x, y, x_size);
    playMoveAssumeLegal(loc, pla);
}

void Board::print_legal_dist(Player pla) const {
    std::cout << "---------------------" << std::endl;
    std::cout << "Legal info, O(Legal), -(Illegal)" << std::endl;
    std::cout << "Player: " << (pla == P_BLACK ? "Black(X)" : "White(O)") << std::endl;
    std::cout << "Legal num: " << (pla == P_BLACK ? black_legal_moves : white_legal_moves) << std::endl;
    const auto& legal_dist_tmp = (pla == P_BLACK ? black_legal_dist : white_legal_dist);
    std::cout << "  ";
    for (Size i = 0; i < x_size; i++) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    for(Size y = 0; y < y_size; y++) {
        std::cout << y << " ";
        for(Size x = 0; x < x_size; x++) {
            Loc loc = Location::getLoc(x, y, x_size);
            std::cout << (legal_dist_tmp[loc] ? "O " : "- ");
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "---------------------" << std::endl;
}


//LOCATION--------------------------------------------------------------------------------
Loc Location::getLoc(Loc x, Loc y, Size x_size)
{
    return (x+1) + (y+1)*(x_size+1);
}
Loc Location::getX(Loc loc, int x_size)
{
    return (loc % (x_size+1)) - 1;
}
Loc Location::getY(Loc loc, int x_size)
{
    return (loc / (x_size+1)) - 1;
}

Loc Location::getLocNN(Loc x, Loc y, Size x_size)
{
    return (x) + (y)*(x_size);
}
Loc Location::getXNN(Loc loc, int x_size)
{
    return (loc % (x_size));
}
Loc Location::getYNN(Loc loc, int x_size)
{
    return (loc / (x_size));
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