//
// Created by syys on 2021/3/2.
//

#ifndef EXAMPLE_APP_TRAIN_H
#define EXAMPLE_APP_TRAIN_H

#include <deque>
#include <ctime>
#include "mcts.h"

static const char *model_path = "../model/size_9/model-checkpoint.pt";
static const char *best_path = "../model/size_9/model-best.pt";
static constexpr bool self_play_show = false;
static constexpr bool eval_show = false;
static constexpr uint32_t train_epoch = 10;
static constexpr uint32_t eval_fre = 2;
static constexpr uint32_t explore_count = 15;
static constexpr uint32_t check_frequency = 20;
static constexpr uint32_t buffer_num = 20000;

bool file_exists(const char * file);

class TimeCounter
{
public:
    inline void start() { this->s = clock(); }
    inline clock_t end() { this->e = clock(); return this->e - this->s; }
    inline double end_s() { return (double)this->end() / CLOCKS_PER_SEC; }
private:
    clock_t s, e;
};

class Train
{
public:
    Train(Size size=COMPILE_MAX_BOARD_LEN, uint32_t state_c=2, uint32_t n_thread=6, double lr=4e-3, double c_lr=1, double temp=1, uint32_t n_simulate=400,
          uint32_t c_puct=5, double virtual_loss=3, uint32_t buffer_size=buffer_num, uint32_t batch_size=256, uint32_t epochs=train_epoch, double kl_targ=0.02, uint32_t check_freq=check_frequency, uint32_t n_game=2000) :
            nogo(size), network(best_path, true, state_c, size, size*size), mcts(&network, n_thread, c_puct, temp, n_simulate, virtual_loss, size*size, true),
            state_c(state_c), n_thread(n_thread), c_puct(c_puct), virtual_loss(virtual_loss), temp(temp), n_simulate(n_simulate),
            N(buffer_size), lr(lr), c_lr(c_lr), batch_size(batch_size), epochs(epochs), kl_targ(kl_targ), check_freq(check_freq), n_game(n_game),
            optimizer(network.model->parameters(), torch::optim::AdamOptions(lr).weight_decay(1e-4))
    {
        this->states = torch::zeros({ 0,state_c,size,size });
        this->probs = torch::zeros({ 0,size,size });
        this->values = torch::zeros({ 0,1 });
    }
    // 扩充数据
    void augment_data(std::vector<at::Tensor> &states, std::vector<at::Tensor> &probs, std::vector<float> &values);
    void push(const at::Tensor &s, const at::Tensor &p, const at::Tensor &z);
    // 评估
    double evaluate(const char *best_path_local, uint32_t num);
    double eval_best_with(uint32_t num, const char *other_nn_path= nullptr);
    void run(const char *model_path_local, const char *best_path_local);
    std::vector<double> train_step(const std::vector<at::Tensor> &state, const std::vector<at::Tensor> &prob, const std::vector<at::Tensor> &value, const double &lr);
    std::vector<double> train_step(const at::Tensor &state, const at::Tensor &prob, const at::Tensor &value, const double &lr);
private:
    Nogo nogo;
    uint32_t state_c;
    uint32_t n_thread;
    uint32_t c_puct;
    double temp;
    double virtual_loss;
    uint32_t n_simulate;
    PolicyValueNet network;
    MCTS mcts;
    double lr;	// 初始学习速率
    double c_lr;// 学习速率乘数
    uint32_t batch_size;// 每步训练的数据量
    uint32_t epochs;	// 训练多少步
    double kl_targ;		// kl_loss 目标（控制训练速率）
    uint32_t check_freq;// 每隔多少局游戏进行评估
    uint32_t n_game;	// 自我对弈多少局游戏
    //std::deque<at::Tensor> states;
    //std::deque<at::Tensor> probs;
    //std::deque<at::Tensor> values;
    at::Tensor states;
    at::Tensor probs;
    at::Tensor values;
    uint32_t N;	// 容量
    torch::optim::Adam optimizer;
};

#endif //EXAMPLE_APP_TRAIN_H
