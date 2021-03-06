#### Learning to play nogo from scratch with neural network and MCTS

##### This project is based on several excellent works:

- https://github.com/yffbit/gomoku

- https://github.com/hijkzzz/alpha-zero-gomoku

- https://github.com/progschj/ThreadPool

- https://github.com/lightvector/KataGo


##### Requirements

- Libtorch, CUDA and CUDNN, with matched version. 10.2 recommended.
- c++ compiler supporting c++14

- cmake 3.0+


##### Build

```
mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release

make
```

##### How to train a neural net

​	build the runtool target with make and run it.

​	runtool will detect the model file automatically and start loop of selfplay, training and evaluation.

