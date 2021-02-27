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
    std::string str;
    int x,y;
    // 读入JSON
    getline(std::cin,str);
    //getline(cin, str);
    Json::Reader reader;
    Json::Value input;
    reader.parse(str, input);
    // 分析自己收到的输入和自己过往的输出，并恢复状态
    int turnID = input["responses"].size();
    Player me = input["requests"][0]["x"].asInt() == -1 ? P_BLACK : P_WHITE;
    for (int i = 0; i < turnID; i++)
    {
        x=input["requests"][i]["x"].asInt(), y=input["requests"][i]["y"].asInt();
        if (x!=-1) main_board.playMoveAssumeLegal(x, y, getOpp(me));
        x=input["responses"][i]["x"].asInt(), y=input["responses"][i]["y"].asInt();
        if (x!=-1) main_board.playMoveAssumeLegal(x, y, me);
    }
    x=input["requests"][turnID]["x"].asInt(), y=input["requests"][turnID]["y"].asInt();
    if (x!=-1) main_board.playMoveAssumeLegal(x, y, getOpp(me));
    return me;
}

#endif //EXAMPLE_APP_PARASE_INPUT_H
