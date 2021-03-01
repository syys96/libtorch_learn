//
// Created by syys on 2021/3/1.
//
#include <cassert>
#include <iostream>

int main()
{
#ifdef NDEBUG
        std::cout << "mmp" << std::endl;
#endif
    assert(false);
    int aa = 2;
    return 0;
}