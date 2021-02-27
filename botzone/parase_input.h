//
// Created by syys on 2021/2/27.
//

#ifndef EXAMPLE_APP_PARASE_INPUT_H
#define EXAMPLE_APP_PARASE_INPUT_H

#include "../board.h"
#include "../jsoncpp/json.h"

// 由于json.h 里面包含#include"../jsoncpp.cpp"
// 所以不能把下面函数的实现放到parase_input.cpp里面去
// 否则多次引入jsoncpp.cpp会造成duplicate symbol
// 也不知道zhouhy是怎么想的，真的坑

Player parase_botzone(Board& main_board)
{
    int x,y;
    // 读入JSON
    std::string str;
    std::cout << "begin read getline" << std::endl;
    getline(std::cin,str);
    std::cout << "finish read getline" << std::endl;
//    std::string str = std::string("{\"requests\":[{\"x\":-1,\"y\":-1},{\"x\":1,\"y\":1},{\"x\":0,\"y\":8},{\"x\":8,\"y\":0},{\"x\":8,\"y\":8},{\"x\":0,\"y\":4},{\"x\":7,\"y\":0},{\"x\":7,\"y\":2},{\"x\":8,\"y\":3},{\"x\":3,\"y\":0},{\"x\":0,\"y\":2},{\"x\":2,\"y\":3},{\"x\":6,\"y\":7},{\"x\":3,\"y\":2},{\"x\":3,\"y\":5},{\"x\":1,\"y\":8},{\"x\":1,\"y\":6},{\"x\":5,\"y\":0},{\"x\":2,\"y\":7},{\"x\":4,\"y\":7},{\"x\":5,\"y\":3},{\"x\":1,\"y\":5},{\"x\":4,\"y\":1},{\"x\":5,\"y\":5},{\"x\":6,\"y\":4},{\"x\":8,\"y\":5},{\"x\":7,\"y\":6},{\"x\":6,\"y\":2},{\"x\":3,\"y\":6},{\"x\":2,\"y\":5},{\"x\":6,\"y\":8},{\"x\":8,\"y\":4},{\"x\":1,\"y\":4},{\"x\":2,\"y\":6},{\"x\":8,\"y\":6},{\"x\":7,\"y\":5}],\"responses\":[{\"x\":0,\"y\":0},{\"x\":2,\"y\":0},{\"x\":1,\"y\":7},{\"x\":7,\"y\":1},{\"x\":7,\"y\":7},{\"x\":6,\"y\":0},{\"x\":8,\"y\":2},{\"x\":1,\"y\":2},{\"x\":2,\"y\":1},{\"x\":0,\"y\":3},{\"x\":2,\"y\":2},{\"x\":6,\"y\":6},{\"x\":1,\"y\":0},{\"x\":5,\"y\":8},{\"x\":2,\"y\":8},{\"x\":0,\"y\":6},{\"x\":5,\"y\":1},{\"x\":3,\"y\":7},{\"x\":5,\"y\":2},{\"x\":4,\"y\":3},{\"x\":3,\"y\":4},{\"x\":4,\"y\":4},{\"x\":3,\"y\":1},{\"x\":4,\"y\":2},{\"x\":7,\"y\":4},{\"x\":6,\"y\":5},{\"x\":5,\"y\":6},{\"x\":6,\"y\":3},{\"x\":4,\"y\":5},{\"x\":4,\"y\":8},{\"x\":8,\"y\":7},{\"x\":2,\"y\":4},{\"x\":1,\"y\":3},{\"x\":5,\"y\":7},{\"x\":0,\"y\":5}]}");
    Json::Reader reader;
    Json::Value input;
    reader.parse(str, input);
    // 分析自己收到的输入和自己过往的输出，并恢复状态
    int turnID = input["responses"].size();
    Player me = input["requests"][0]["x"].asInt() == -1 ? P_BLACK : P_WHITE;
    for (int i = 0; i < turnID; i++)
    {
        x=input["requests"][i]["x"].asInt(), y=input["requests"][i]["y"].asInt();
        if (x!=-1) {
            bool islegal = main_board.playMove(x, y, getOpp(me));
            if (!islegal) {
                main_board.print_board(getOpp(me));
                main_board.print_legal_dist(getOpp(me));
                std::cout << "making move at: " << x << ", " << y << std::endl;
                throw std::runtime_error("?????");
            }
        }
        x=input["responses"][i]["x"].asInt(), y=input["responses"][i]["y"].asInt();
        if (x!=-1){
            bool islegal = main_board.playMove(x, y, me);
            if (!islegal) {
                main_board.print_board(me);
                main_board.print_legal_dist(me);
                std::cout << "making move at: " << x << ", " << y << std::endl;
                throw std::runtime_error("?????");
            }
        }
    }
    x=input["requests"][turnID]["x"].asInt(), y=input["requests"][turnID]["y"].asInt();
    if (x!=-1) {
        bool islegal = main_board.playMove(x, y, getOpp(me));
        if (!islegal) {
            main_board.print_board(getOpp(me));
            main_board.print_legal_dist(getOpp(me));
            std::cout << "making move at: " << x << ", " << y << std::endl;
            throw std::runtime_error("?????");
        }
    }
    return me;
}

#endif //EXAMPLE_APP_PARASE_INPUT_H
