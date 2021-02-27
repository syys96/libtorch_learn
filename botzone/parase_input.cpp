//
// Created by syys on 2021/2/27.
//

//#include "parase_input.h"
//
//Player parase_botzone(Board& main_board)
//{
//    std::string str;
//    int x,y;
//    // 读入JSON
//    getline(std::cin,str);
//    //getline(cin, str);
//    Json::Reader reader;
//    Json::Value input;
//    reader.parse(str, input);
//    // 分析自己收到的输入和自己过往的输出，并恢复状态
//    int turnID = input["responses"].size();
//    Player me = input["requests"][0]["x"].asInt() == -1 ? P_BLACK : P_WHITE;
//    for (int i = 0; i < turnID; i++)
//    {
//        x=input["requests"][i]["x"].asInt(), y=input["requests"][i]["y"].asInt();
//        if (x!=-1) main_board.playMoveAssumeLegal(x, y, getOpp(me));
//        x=input["responses"][i]["x"].asInt(), y=input["responses"][i]["y"].asInt();
//        if (x!=-1) main_board.playMoveAssumeLegal(x, y, me);
//    }
//    x=input["requests"][turnID]["x"].asInt(), y=input["requests"][turnID]["y"].asInt();
//    if (x!=-1) main_board.playMoveAssumeLegal(x, y, getOpp(me));
//    return me;
//}