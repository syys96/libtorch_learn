// system include headers
#include <torch/script.h>
#include <queue>
#include <cfloat>
#include <cassert>
#include <cmath>
#include <future>
#include <deque>
#include <atomic>
#include <ctime>
#include <numeric>
#include <cstdlib>
#include <condition_variable>
#include <iostream>
#include <torch/torch.h>
#include <thread>
#include <tuple>
#include <functional>
#include <memory>
#include <vector>


// header files

// begin /Users/syys/CLionProjects/torch_example/thread_pool.h
//
// Created by syys on 2021/2/25.
//

#ifndef EXAMPLE_APP_THREAD_POOL_H
#define EXAMPLE_APP_THREAD_POOL_H


//class ThreadPool
//{
//public:
//    using Task = std::function<void ()>;
//
//    explicit ThreadPool(unsigned int n_thread = 4);
//
//    ~ThreadPool();
//
//    int get_idle_num();
//
//    template<class F, class... Args>
//    auto commit(F &&f, Args &&... args)->std::future<decltype(f(args...))>
//    {
//        // 提交任务，返回 std::future
//        if (!this->run) throw std::runtime_error("commit error : ThreadPool is stopped.");
//        // using return_type = typename std::result_of<F(Args...)>::type;
//        using return_type = decltype(f(args...));
//        // packaged_task package the bind function and future
//        auto task_ptr = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
//
//        {
//            std::lock_guard<std::mutex> lock_temp(this->look);
//            this->tasks.emplace([task_ptr]() { (*task_ptr)(); });
//        }
//
//        this->cv.notify_one();
//        return task_ptr->get_future();
//    }
//
//private:
//    // 线程池
//    std::vector<std::thread> pool;
//    // 任务队列
//    std::queue<Task> tasks;
//    // 同步
//    std::mutex look;
//    std::condition_variable cv;
//    // 状态
//    std::atomic_bool run;
//    std::atomic_uint idle_num;
//};

class ThreadPool {
public:
    using task_type = std::function<void()>;

    inline ThreadPool(unsigned short thread_num = 4) {
        this->run.store(true);
        this->idl_thread_num = thread_num;

        for (unsigned int i = 0; i < thread_num; ++i) {
            // thread type implicit conversion
            pool.emplace_back([this] {
                while (this->run) {
                    std::function<void()> task;

                    // get a task
                    {
                        std::unique_lock<std::mutex> lock(this->lock);
                        this->cv.wait(lock, [this] {
                            return this->tasks.size() > 0 || !this->run.load();
                        });

                        // exit
                        if (!this->run.load() && this->tasks.empty())
                            return;

                        // pop task
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    // run a task
                    this->idl_thread_num--;
                    task();
                    this->idl_thread_num++;
                }
            });
        }
    }

    inline ~ThreadPool() {
        // clean thread pool
        this->run.store(false);
        this->cv.notify_all(); // wake all thread

        for (std::thread &thread : pool) {
            thread.join();
        }
    }

    template <class F, class... Args>
    auto commit(F &&f, Args &&... args) -> std::future<decltype(f(args...))> {
        // commit a task, return std::future
        // example: .commit(std::bind(&Dog::sayHello, &dog));

        if (!this->run.load())
            throw std::runtime_error("commit on ThreadPool is stopped.");

        // declare return type
        using return_type = decltype(f(args...));

        // make a shared ptr for packaged_task
        // packaged_task package the bind function and future
        auto task_ptr = std::make_shared<std::packaged_task<return_type()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        {
            std::lock_guard<std::mutex> lock(this->lock);
            tasks.emplace([task_ptr]() { (*task_ptr)(); });
        }

        // wake a thread
        this->cv.notify_one();

        return task_ptr->get_future();
    }

    inline int get_idl_num() { return this->idl_thread_num; }

private:
    std::vector<std::thread> pool; // thead pool
    std::queue<task_type> tasks;   // tasks queue
    std::mutex lock;               // lock for tasks queue
    std::condition_variable cv;    // condition variable for tasks queue

    std::atomic<bool> run;                    // is running
    std::atomic<unsigned int> idl_thread_num; // idle thread number
};

#endif //EXAMPLE_APP_THREAD_POOL_H

// end

// begin /Users/syys/CLionProjects/torch_example/board.h
//
// Created by syys on 2021/2/25.
//

#ifndef EXAMPLE_APP_BOARD_H
#define EXAMPLE_APP_BOARD_H


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
    Loc getX(Loc loc, Size x_size);
    Loc getY(Loc loc, Size x_size);
    Loc getLocNN(Loc x, Loc y, Size x_size);
    Loc getXNN(Loc loc, Size x_size);
    Loc getYNN(Loc loc, Size x_size);
    Loc Loc2LocNN(Loc loc, Size x_size);
    Loc LocNN2Loc(Loc locNN, Size x_size);

    void getAdjacentOffsets(std::vector<Size>& adj_offsets, Size x_size);
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

static constexpr int NN_BLACK = 1;
static constexpr int NN_WHITE = -1;
static constexpr int NN_EMPTY = 0;

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
    bool operator==(const Board& other) const;

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
    // return color info for nn
    void get_color_NN(std::vector<int>& board_nn) const;
private:
    //Structs---------------------------------------

    //Tracks a chain/string/group of stones
    struct ChainData {
        Player owner;        //Owner of chain
        short num_locs;      //Number of stones in chain
        short num_liberties; //Number of liberties in chain
        ChainData() {owner = C_EMPTY; num_locs = 0; num_liberties = 0;}
        bool operator==(const ChainData& other) const { return owner==other.owner
                                                && num_locs== other.num_locs
                                                && num_liberties==other.num_liberties;}
    };

    //Data--------------------------------------------

    Size x_size;                  //Horizontal size of board
    Size y_size;                  //Vertical size of board
    std::vector<Color> colors;    //Color of each location on the board.

    // updated with each move
    std::vector<bool> black_legal_dist;
    std::vector<bool> white_legal_dist;
    Num black_legal_moves;
    Num white_legal_moves;

    //Every chain of stones has one of its stones arbitrarily designated as the head.
    std::vector<ChainData> chain_data; //For each head stone, the chaindata for the chain under that head. Undefined otherwise.
    std::vector<Loc> chain_head;       //Where is the head of this chain? Undefined if EMPTY or WALL
    std::vector<Loc> next_in_chain;    //Location of next stone in chain. Circular linked list. Undefined if EMPTY or WALL

//    ChainData chain_data[MAX_ARR_SIZE];
//    Loc chain_head[MAX_ARR_SIZE];
//    Loc next_in_chain[MAX_ARR_SIZE];

//    Size adj_offsets[8];
    std::vector<Size> adj_offsets;    //Indices 0-3: Offsets to add for adjacent points. Indices 4-7: Offsets for diagonal points. 2 and 3 are +x and +y.

    void mergeChains(Loc loc1, Loc loc2);
    bool isLibertyOf(Loc loc, Loc head) const;
    void update_blank_legality(const std::vector<Loc>& locs);
};

#endif //EXAMPLE_APP_BOARD_H

// end

// begin /Users/syys/CLionProjects/torch_example/nogo.h
//
// Created by syys on 2021/2/26.
//

#ifndef EXAMPLE_APP_NOGO_H
#define EXAMPLE_APP_NOGO_H


// 前向声明
class Playerm;

class Nogo {
public:
    using move_type = Loc;
    using board_type = Board;

    Nogo& operator=(const Nogo&) = default;

    explicit Nogo(Size n, Player first_color=P_BLACK);

    bool has_legal_moves();
    Num get_legal_moves(std::vector<int>& legal_dist);
    bool execute_move(move_type move);
    std::vector<int> get_game_status();
    at::Tensor curr_state(bool to_device, torch::Device &device);
    void display() const;
    static char get_symbol(Player player);
    int start_play(Playerm *player1, Playerm *player2, bool swap=false, bool show=false);

    inline Size get_action_size() const { return this->n * this->n; }
    inline board_type get_board() const { return this->board; }
    inline Player get_current_color() const { return this->cur_color; }
    inline Size get_n() const { return this->n; }
    inline Size get_action_dim() const { return this->n*this->n; }
    void reset();
    void parse_botzone_input();

private:
    board_type board;      // game borad
    Size n;        // board size
    Player cur_color;       // current player's color
};

class Playerm
{
public:
    inline explicit Playerm(Player player = P_BLACK) :player(player) {}
    inline ~Playerm() = default;
    inline void set_player(Player player) { this->player = player; }
    virtual void init() {}
    virtual void update_with_move(Loc last_move) {}
    inline int get_player() const { return this->player; }
    virtual Loc get_action(Nogo *nogo, bool explore = false) = 0;
private:
    Player player;
};

class Human : public Playerm
{
public:
    inline explicit Human(Player player = 1) :Playerm(player) {}
    inline ~Human() = default;
    inline Loc get_action(Nogo *nogo, bool explore = false) override
    {
        Size n = nogo->get_n(), i, j;
        while (true)
        {
            std::cin >> i >> j;
            std::cin.clear();
            if (i >= 0 && i < n && j >= 0 && j < n) break;
            else std::cout << "Illegal input. Reenter : ";
        }
        return Location::getLoc(i,j,n);
    }
};

#endif //EXAMPLE_APP_NOGO_H

// end

// begin /Users/syys/CLionProjects/torch_example/policy_value_net.h
//
// Created by syys on 2021/2/28.
//

#ifndef EXAMPLE_APP_POLICY_VALUE_NET_H
#define EXAMPLE_APP_POLICY_VALUE_NET_H


struct ResidualBlockImpl : torch::nn::Module
{
    ResidualBlockImpl(int64_t in_channels, int64_t out_channels, int kernel_size = 3) :
            conv1(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(1).padding((kernel_size - 1) >> 1)),
            conv2(torch::nn::Conv2dOptions(out_channels, out_channels, kernel_size).stride(1).padding((kernel_size - 1) >> 1)),
            batch_norm1(torch::nn::BatchNorm2dOptions(out_channels)),
            batch_norm2(torch::nn::BatchNorm2dOptions(out_channels))
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("batch_norm1", batch_norm1);
        register_module("batch_norm2", batch_norm2);
    }
    at::Tensor forward(at::Tensor x)
    {
        at::Tensor x_ = x;
        x = batch_norm1(conv1(x)).relu();
        x = torch::add(batch_norm2(conv2(x)), x_);
        return torch::relu(x);
    }
    torch::nn::Conv2d conv1, conv2;
    torch::nn::BatchNorm2d batch_norm1, batch_norm2;
};
TORCH_MODULE(ResidualBlock);

struct NetworkImpl : torch::nn::Module
{
    NetworkImpl(int64_t state_channels, int64_t state_h, int64_t action_dim, int64_t conv_channels, int kernel_size = 3) :
            conv(torch::nn::Conv2dOptions(state_channels, conv_channels, kernel_size).stride(1).padding((kernel_size - 1) >> 1)),
            bn(torch::nn::BatchNorm2dOptions(conv_channels)),
            residual(ResidualBlock(conv_channels, conv_channels, kernel_size)),
            conv_policy(torch::nn::Conv2dOptions(conv_channels, 2, 1).stride(1)),
            bn_policy(torch::nn::BatchNorm2dOptions(2)),
            fc_policy(2 * state_h * state_h, action_dim),
            conv_value(torch::nn::Conv2dOptions(conv_channels, 1, 1).stride(1)),
            bn_value(torch::nn::BatchNorm2dOptions(1)),
            fc1_value(state_h * state_h, 256),
            fc2_value(256, 1)
    {
        register_module("conv", conv);
        register_module("bn", bn);
        register_module("residual", residual);
        register_module("conv_policy", conv_policy);
        register_module("bn_policy", bn_policy);
        register_module("fc_policy", fc_policy);
        register_module("conv_value", conv_value);
        register_module("bn_value", bn_value);
        register_module("fc1_value", fc1_value);
        register_module("fc2_value", fc2_value);
    }

    std::vector<at::Tensor> forward(at::Tensor x)
    {
        at::Tensor action_prob, value;
        x = residual(bn(conv(x)).relu());
        action_prob = fc_policy(bn_policy(conv_policy(x)).relu().flatten(1)).softmax(1);
        x = fc1_value(bn_value(conv_value(x)).relu().flatten(1)).relu();
        value = fc2_value(x).tanh();
        return { action_prob, value };
    }

    torch::nn::Conv2d conv;
    torch::nn::BatchNorm2d bn;
    ResidualBlock residual;
    torch::nn::Conv2d conv_policy;
    torch::nn::BatchNorm2d bn_policy;
    torch::nn::Linear fc_policy;
    torch::nn::Conv2d conv_value;
    torch::nn::BatchNorm2d bn_value;
    torch::nn::Linear fc1_value;
    torch::nn::Linear fc2_value;
};
TORCH_MODULE(Network);

class PolicyValueNet
{
public:
    PolicyValueNet(const char *model_path, bool use_cuda, int32_t state_c, int32_t state_h, int32_t action_dim) :
            model(state_c, state_h, action_dim, 128, 3), device(torch::kCPU)
    {
        if (use_cuda && torch::cuda::is_available()) this->device = torch::Device(torch::kCUDA, 0);
        if (nullptr != model_path)
        {
            FILE *fp = fopen(model_path, "rb");
            if (nullptr != fp)
            {
                fclose(fp);
                fp = nullptr;
                this->load_model(model_path);
            }
        }
        this->model->to(this->device);
        this->model(torch::zeros({ 1,state_c,state_h,state_h }, this->device));
    }
    ~PolicyValueNet() {};
    void save_model_cpu(const char * save_path);
    inline void save_model(const char * save_path) { torch::save(this->model, save_path); }
    inline void load_model(const char * model_path) { torch::load(this->model, model_path); }
    inline std::vector<at::Tensor> predict(const std::vector<at::Tensor> &x) { return this->model(torch::cat(x, 0).to(this->device)); }
    inline std::vector<at::Tensor> predict(const at::Tensor &x) { return this->model(x); }
    inline at::Tensor dirichlet_noise(uint32_t dim, float alpha)
    {
        std::vector<float> dirichlet(dim, alpha);
        // return torch::_sample_dirichlet(torch::tensor(dirichlet, this->device));
        return torch::_sample_dirichlet(torch::tensor(dirichlet));
    }

    Network model;
    torch::Device device;
};

#endif //EXAMPLE_APP_POLICY_VALUE_NET_H

// end

// begin /Users/syys/CLionProjects/torch_example/mcts.h
//
// Created by syys on 2021/2/28.
//

#ifndef EXAMPLE_APP_MCTS_H
#define EXAMPLE_APP_MCTS_H

#ifndef SINGLE_THREAD
#endif // !SINGLE_THREAD

class TreeNode
{
public:
    friend class MCTS; // 友元类，可以访问TreeNode的私有成员

    TreeNode(uint32_t action_dim);
    TreeNode(TreeNode *parent, double P, uint32_t action_dim);
    TreeNode(const TreeNode &node);
    ~TreeNode();
    TreeNode &operator=(const TreeNode &node);
    void copy(const TreeNode &node, TreeNode *parent);
    Loc select(double c_puct, double virtual_loss);
    double get_value(double c_puct, uint32_t sum_N) const;
    bool expand(const at::Tensor &prior, const std::vector<int> &legal_action);
    void backup(double value, double virtual_loss, bool success);
    inline bool is_leaf() const { return this->leaf; }

private:
    TreeNode *parent;
    std::vector<TreeNode *> children;
    bool leaf;	// 是否为叶子节点
#ifndef SINGLE_THREAD
    std::mutex lock; // 扩展时加锁
    std::atomic<int> N;
    std::atomic<double> W;
    std::atomic<double> Q;
#else
    int N;
	double W;
	double Q;
#endif // !SINGLE_THREAD
    double P;
};

class MCTS : public Playerm
{
public:
    MCTS(PolicyValueNet *network, uint32_t n_thread, double c_puct, double temp,
         uint32_t n_simulate, double virtual_loss, uint32_t action_dim, bool add_noise);
    Loc get_action(Nogo *nogo, bool explore = false) override;
    Loc get_action(std::vector<double> action_probs, bool explore = false);
    std::vector<double> get_action_prob(Nogo *nogo);
    inline void init() override { reset(); }
    void update_with_move(Loc last_move) override;
    inline void set_temp(double temp = 1e-3) { this->temp = temp; }
    int self_play(Nogo *nogo, std::vector<at::Tensor> &states, std::vector<at::Tensor> &probs, std::vector<float> &values,
                  double temp = 1, uint32_t n_round = 20, bool add_noise = true, bool show = false);
    void simulate(Nogo * nogo);
    void clear_tree();
    void reset();
private:
    std::unique_ptr<TreeNode> root;
#ifndef SINGLE_THREAD
    std::unique_ptr<ThreadPool> thread_pool;
#endif // !SINGLE_THREAD
    PolicyValueNet *network;

    uint32_t action_dim;
    uint32_t n_simulate;
    double c_puct;
    double virtual_loss;
    uint32_t n_count;   // 落子计数
    double temp;    // 温度参数
    bool add_noise;	// 扩展时是否添加噪声
};

#endif //EXAMPLE_APP_MCTS_H

// end

// begin /Users/syys/CLionProjects/torch_example/train.h
//
// Created by syys on 2021/3/2.
//

#ifndef EXAMPLE_APP_TRAIN_H
#define EXAMPLE_APP_TRAIN_H


static const char *model_path = "../model/size_9/model-checkpoint.pt";
static const char *best_path = "../model/size_9/model-best.pt";
static const char *best_path_cpu = "../model/size_9/model-best-cpu.pt";
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

// end



// source files

// begin /Users/syys/CLionProjects/torch_example/thread_pool.cpp
//
// Created by syys on 2021/2/25.
//



//ThreadPool::~ThreadPool() {
//    this->run = false;
//    // wake 所有正在wait的线程
//    this->cv.notify_all();
//    for (std::thread & t : this->pool) t.join();
//}
//
//
//int ThreadPool::get_idle_num() {
//    return idle_num;
//}
//
//ThreadPool::ThreadPool(unsigned int n_thread): run(true), idle_num(n_thread)
//{
//    if (n_thread < 1) {
//        throw StringError("thread number less than 1 is illegal!");
//    }
//    // 创建线程
//    for (unsigned int i = 0; i < n_thread; i++)
//    {
//        // 匿名函数作为创建线程的参数
//        pool.emplace_back([this]() {
//            while(this->run)
//            {
//                Task task;
//                {
//                    std::unique_lock<std::mutex> lock_temp(this->look);
//                    // wait 之前判断匿名函数的返回值，为true时不wait，为false时wait
//                    // wait 过程中（锁已经释放）需要 notify 才会退出 wait，重新加锁
//                    this->cv.wait(lock_temp, [this]() { return !this->tasks.empty() || !this->run; });
//                    // 禁用并且任务队列为空时，退出线程
//                    if (!this->run && this->tasks.empty()) return;
//
//                    task = std::move(this->tasks.front());
//                    this->tasks.pop();
//                }
//                // 执行
//                this->idle_num--;
//                task();
//                this->idle_num++;
//            }
//        });
//    }
//}


// end

// begin /Users/syys/CLionProjects/torch_example/board.cpp
//
// Created by syys on 2021/2/25.
//


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

//    memcpy(chain_data, other.chain_data, sizeof(ChainData)*MAX_ARR_SIZE);
//    memcpy(chain_head, other.chain_head, sizeof(Loc)*MAX_ARR_SIZE);
//    memcpy(next_in_chain, other.next_in_chain, sizeof(Loc)*MAX_ARR_SIZE);
    chain_data = other.chain_data;
    chain_head = other.chain_head;
    next_in_chain = other.next_in_chain;

    colors = other.colors;
    black_legal_dist = other.black_legal_dist;
    white_legal_dist = other.white_legal_dist;
    black_legal_moves = other.black_legal_moves;
    white_legal_moves = other.white_legal_moves;

    adj_offsets = other.adj_offsets;
//    memcpy(adj_offsets, other.adj_offsets, sizeof(Size)*8);
}

void Board::init(Size xS, Size yS)
{
    if(xS < 0 || yS < 0 || xS > MAX_LEN || yS > MAX_LEN) {
        std::cout << xS << ", " << MAX_LEN << std::endl;
        throw std::runtime_error("Board::init - invalid board size");
    }


    x_size = xS;
    y_size = yS;

//    std::fill(colors, colors+MAX_ARR_SIZE, C_WALL);
    colors.resize(MAX_ARR_SIZE, C_WALL);
    white_legal_dist.resize(MAX_ARR_SIZE, false);
    black_legal_dist.resize(MAX_ARR_SIZE, false);
    std::fill(white_legal_dist.begin(), white_legal_dist.end(), false);
    std::fill(black_legal_dist.begin(), black_legal_dist.end(), false);
    std::fill(colors.begin(), colors.end(), C_WALL);

    FOREACHONBOARD(
            colors[loc] = C_EMPTY;
            black_legal_dist[loc] = true;
            white_legal_dist[loc] = true;
            )

    black_legal_moves = xS * yS;
    white_legal_moves = xS * yS;

    chain_head.resize(MAX_ARR_SIZE, C_EMPTY);
    next_in_chain.resize(MAX_ARR_SIZE, C_EMPTY);
    chain_data.resize(MAX_ARR_SIZE, ChainData());
    std::fill(chain_head.begin(), chain_head.end(), C_EMPTY);
    std::fill(next_in_chain.begin(), next_in_chain.end(), C_EMPTY);
    std::fill(chain_data.begin(), chain_data.end(), ChainData());

    adj_offsets.resize(8, 0);
    Location::getAdjacentOffsets(adj_offsets,x_size);
}

void Board::playMoveAssumeLegal(Loc loc, Player pla)
{
//    std::cout << black_legal_moves << " vs(before) "
//              << std::count(black_legal_dist.begin(),black_legal_dist.end(), true)
//              << std::endl;

    Player opp = getOpp(pla);

    const auto& legal_bool_tmp = pla == P_BLACK ? black_legal_dist : white_legal_dist;
    if (!legal_bool_tmp[loc]) {
        throw std::runtime_error("wtf");
    }
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
                throw std::runtime_error("capture opp stone is illegal in nogo!");
            }
        }
    }

    //Handle suicide
    if(getNumLiberties(loc) == 0) {
        throw std::runtime_error("suicide is illegal in nogo!");
    }

    // update blanks
    bool old_black_tmp = black_legal_dist[loc];
    bool old_white_tmp = white_legal_dist[loc];
    black_legal_dist[loc] = false;
    white_legal_dist[loc] = false;
    // make move will eat one blank for both players.
    if (old_black_tmp) {
        black_legal_moves--;
    }
    if (old_white_tmp) {
        white_legal_moves--;
    }

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
    std::fill(colors.begin(), colors.end(), C_WALL);
    std::fill(white_legal_dist.begin(), white_legal_dist.end(), false);
    std::fill(black_legal_dist.begin(), black_legal_dist.end(), false);

    FOREACHONBOARD(
            colors[loc] = C_EMPTY;
            black_legal_dist[loc] = true;
            white_legal_dist[loc] = true;
            )

    black_legal_moves = x_size * y_size;
    white_legal_moves = x_size * y_size;

    std::fill(chain_head.begin(), chain_head.end(), C_EMPTY);
    std::fill(next_in_chain.begin(), next_in_chain.end(), C_EMPTY);
    std::fill(chain_data.begin(), chain_data.end(), ChainData());
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
    const auto& legal_bool = pla == P_BLACK ? black_legal_dist : white_legal_dist;
    if (isLegal(loc,pla) != legal_bool[loc]) {
        throw std::runtime_error("not equal!");
    }
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

void Board::get_color_NN(std::vector<int> &board_nn) const {
    FOREACHONBOARD(
            Loc locnn = Location::Loc2LocNN(loc, x_size);
            board_nn[locnn] = (colors[loc] == C_BLACK ? NN_BLACK :
                    (colors[loc] == C_WHITE ? NN_WHITE : NN_EMPTY)
                    );
            )
}

bool Board::operator==(const Board &other) const {
    if (x_size != other.x_size || y_size != other.y_size)
        return false;
    if (colors != other.colors)
        return false;
    if (black_legal_dist != other.black_legal_dist || white_legal_dist != other.white_legal_dist)
        return false;
    if (black_legal_moves != other.black_legal_moves || white_legal_moves != other.white_legal_moves)
        return false;
    if (chain_data != other.chain_data || chain_head != other.chain_head || next_in_chain != other.next_in_chain)
        return false;
    if (adj_offsets != other.adj_offsets)
        return false;
    return true;
}


//LOCATION--------------------------------------------------------------------------------
Loc Location::getLoc(Loc x, Loc y, Size x_size)
{
    return (x+1) + (y+1)*(x_size+1);
}
Loc Location::getX(Loc loc, Size x_size)
{
    return (loc % (x_size+1)) - 1;
}
Loc Location::getY(Loc loc, Size x_size)
{
    return (loc / (x_size+1)) - 1;
}

Loc Location::getLocNN(Loc x, Loc y, Size x_size)
{
    return (x) + (y)*(x_size);
}
Loc Location::getXNN(Loc loc, Size x_size)
{
    return (loc % (x_size));
}
Loc Location::getYNN(Loc loc, Size x_size)
{
    return (loc / (x_size));
}

Loc Location::Loc2LocNN(Loc loc, Size x_size)
{
    Loc locx = getX(loc, x_size);
    Loc locy = getY(loc, x_size);
    return getLocNN(locx, locy, x_size);
}

Loc Location::LocNN2Loc(Loc locNN, Size x_size)
{
    Loc xnn = getXNN(locNN, x_size);
    Loc ynn = getYNN(locNN, x_size);
    return getLoc(xnn, ynn, x_size);
}

void Location::getAdjacentOffsets(std::vector<Size>& adj_offsets, Size x_size)
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
// end

// begin /Users/syys/CLionProjects/torch_example/nogo.cpp
//
// Created by syys on 2021/2/26.
//


Nogo::Nogo(Size n, Player first_color): n(n), cur_color(first_color)  {
    board.init(n, n);
}

bool Nogo::has_legal_moves() {
    Num legal_num = board.get_leagal_moves(cur_color);
    return legal_num > 0;
}

bool Nogo::execute_move(Nogo::move_type move) {
    if (!board.playMove(move, cur_color)) {
        std::cout << "ERRORAAAAAAAA" << std::endl;
        board.print_board(cur_color);
        board.print_legal_dist(cur_color);
        board.print_legal_dist(getOpp(cur_color));
        throw std::runtime_error("illegal move when execute move!");
        return false;
    }
    this->cur_color = getOpp(this->cur_color);
    return true;
}

std::vector<int> Nogo::get_game_status() {
    Num curr_legal_moves = board.get_leagal_moves(cur_color);
    if (curr_legal_moves <= 0) {
        return {1, getOpp(cur_color)};
    } else {
        return {0, P_NULL};
    }
}

void Nogo::display() const {
    board.print_board(cur_color);
}

Num Nogo::get_legal_moves(std::vector<int>& legal_dist) {
    return board.get_legal_move_dist(cur_color, legal_dist);
}

void Nogo::reset() {
    cur_color = P_BLACK;
    board.init(n, n);
}

at::Tensor Nogo::curr_state(bool to_device, torch::Device &device) {
    // 获取状态作为神经网络的输入  [batch channels height width]
    // 当前玩家视角
    Size size = this->n;
    std::vector<int> temp;
    temp.resize(get_action_dim(), 0);
    this->board.get_color_NN(temp);
    at::Tensor s, boardm;
    if (to_device)
    {
        boardm = torch::tensor(temp, device).toType(torch::kInt).reshape({ size,size });
        s = torch::zeros({ 1,2,size,size }, device).toType(torch::kFloat);
    }
    else
    {
        boardm = torch::tensor(temp, torch::kInt).reshape({ size,size });
        s = torch::zeros({ 1,2,size,size }, torch::kFloat);
    }
    Player a = this->get_current_color();
    // 当前玩家的棋子
    s[0][0] = boardm.eq(a == P_BLACK ? NN_BLACK : NN_WHITE);
    // 对手玩家的棋子
    s[0][1] = boardm.eq(getOpp(a) == P_BLACK ? NN_BLACK : NN_WHITE);
    return s;
}

char Nogo::get_symbol(Player player) {
    if (player == P_BLACK) return 'X';
    else if (player == P_WHITE) return 'O';
    else return '_';
}

int Nogo::start_play(Playerm *player1, Playerm *player2, bool swap, bool show)
{
    // 默认第一个参数为先手(Black)
    if (nullptr == player1 || nullptr == player2) return 0;
    Player idx = swap ? P_WHITE : P_BLACK;	// 交换先后手
    player1->set_player(idx);
    player1->init();
    player2->set_player(getOpp(idx));
    player2->init();
    Playerm * players[2] = { player1,player2 };
    idx = swap ? 1 : 0; // 总是黑方先下，区别在于是player1还是player2下黑子
    if (players[idx]->get_player() != P_BLACK) {
        throw std::runtime_error("play not inited correctly!");
    }
    Loc move;
    if (cur_color != P_BLACK) {
        throw std::runtime_error("init color is not black in eval!");
    }
    std::vector<int> res(2, 0);
    if (show)
    {
        std::cout << "New game." << std::endl;
        this->display();
    }
    while (0 == res[0])
    {
        if (show)
        {
            std::printf("Player '%c' (example: 0 0):", this->get_symbol(players[idx]->get_player()));
        }
        move = players[idx]->get_action(this);
        if (this->execute_move(Location::LocNN2Loc(move, n)))
        {
            if (show)
            {
                Loc nnx = Location::getXNN(move, n);
                Loc nny = Location::getYNN(move, n);
                std::printf("Player '%c' : %d %d\n", this->get_symbol(players[idx]->get_player()), nnx, nny);
            }
            // 你如何确定这两个player的root节点对应的board是同步的？
            // 在只有一个player执行update的情况下？
            players[idx]->update_with_move(move);
            players[1-idx]->update_with_move(move);

            res = this->get_game_status();
            idx = 1 - idx;
            if (show) this->display();
        }
    }
    // 玩家重置
    player1->update_with_move(-1);
    player2->update_with_move(-1);
    if (show)
    {
        if (0 != res[1]) std::printf("Game end. Winner is Player '%c'.\n", this->get_symbol(res[1]));
        else std::cout << "Game end. Tie." << std::endl;
    }
    return res[1];
}

void Nogo::parse_botzone_input() {
    reset();
    int inx, iny;
    int turnID;
    std::cin >> turnID;
    for (int i = 0; i < turnID; i++) {
        std::cin >> inx >> iny;
        if (inx != -1) {
            execute_move(Location::getLoc(inx, iny, n));
        }
        if (i < turnID - 1) {
            std::cin >> inx >> iny;
            if (inx >= 0) {
                execute_move(Location::getLoc(inx, iny, n));
            }
        }
    }
}

//Nogo& Nogo::operator=(const Nogo &) {
//    Nogo copy(n, cur_color);
//    copy.board = board;
//    copy.n = n;
//    copy.cur_color = cur_color;
//    return copy;
//}


// end

// begin /Users/syys/CLionProjects/torch_example/policy_value_net.cpp
//
// Created by syys on 2021/2/28.
//


void PolicyValueNet::save_model_cpu(const char *save_path) {
    this->model->to(torch::kCPU);
    torch::save(this->model, save_path);
    this->model->to(this->device);
}

// end

// begin /Users/syys/CLionProjects/torch_example/mcts.cpp
//
// Created by syys on 2021/2/28.
//


TreeNode::TreeNode(uint32_t action_dim) :
        parent(nullptr),
        children(action_dim, nullptr),
        leaf(true),
        N(0),
        W(0),
        Q(0),
        P(0) {}

TreeNode::TreeNode(TreeNode *parent, double P, uint32_t action_dim) :
        parent(parent),
        children(action_dim, nullptr),
        leaf(true),
        N(0),
        W(0),
        Q(0),
        P(P) {}

TreeNode::TreeNode(const TreeNode &node)
{
    // 拷贝构造函数
    this->copy(node, node.parent);
}

TreeNode & TreeNode::operator=(const TreeNode &node)
{
    // 赋值操作符
    if (this == &node) return *this;
    this->copy(node, node.parent);
    return *this;
}

void TreeNode::copy(const TreeNode &node, TreeNode *parent)
{
    this->parent = parent;
    this->leaf = node.leaf;
#ifndef SINGLE_THREAD
    this->N.store(node.N.load());
    this->W.store(node.W.load());
    this->Q.store(node.Q.load());
#else
    this->N = node.N;
	this->W = node.W;
	this->Q = node.Q;
#endif // !SINGLE_THREAD
    this->P = node.P;
    uint32_t i, size = this->children.size();
    TreeNode *temp = nullptr;
    for (i = 0; i < size; i++)
    {
        temp = this->children[i];
        if (temp)
        {
            temp->parent = nullptr;
            this->children[i] = nullptr;
            delete temp;
            temp = nullptr;
        }
    }
    this->children = node.children; // 浅拷贝
    for (i = 0; i < size; i++)
    {
        if (node.children[i])
        {
            this->children[i] = new TreeNode(size);
            this->children[i]->copy(*node.children[i], this);
        }
    }
}

TreeNode::~TreeNode()
{
    uint32_t i, size = this->children.size();
    if (!this->leaf)
    {
        TreeNode *node = nullptr;
        for (i = 0; i < size; i++)
        {
            node = this->children[i];
            if (node)
            {
                node->parent = nullptr;
                this->children[i] = nullptr;
                delete node;
                node = nullptr;
            }
        }
    }
    if (this->parent)
    {
        size = this->parent->children.size();
        for (i = 0; i < size; i++)
        {
            if (this->parent->children[i] == this)
            {
                this->parent->children[i] = nullptr;
                break;
            }
        }
        this->parent = nullptr;
    }
}

Loc TreeNode::select(double c_puct, double virtual_loss)
{
    double best_value = -DBL_MAX, value;
    int best_act = -1;
#ifndef SINGLE_THREAD
    uint32_t sum_N = this->N.load();
#else
    uint32_t sum_N = this->N;
#endif // !SINGLE_THREAD
    uint32_t i, size = this->children.size();
    for (i = 0; i < size; i++)
    {
        if (nullptr == this->children[i]) continue;
        value = this->children[i]->get_value(c_puct, sum_N);
        if (value > best_value)
        {
            best_value = value;
            best_act = i;
        }
    }
#ifndef SINGLE_THREAD
    // 添加虚拟损失
    if (best_act >= 0)
    {
        TreeNode * node = this->children[best_act];
        node->N += (int)virtual_loss;
        node->W = node->W - virtual_loss;
        node->Q = node->W / node->N;
    }
#endif // !SINGLE_THREAD
    return best_act;
}

double TreeNode::get_value(double c_puct, uint32_t sum_N) const
{
#ifndef SINGLE_THREAD
    return this->Q.load() + c_puct * this->P * sqrt(sum_N) / (1 + this->N.load());
#else
    return this->Q + c_puct * this->P * sqrt(sum_N) / (1 + this->N);
#endif // !SINGLE_THREAD
}

bool TreeNode::expand(const at::Tensor &prior, const std::vector<int> &legal_action)
{
#ifndef SINGLE_THREAD
    std::lock_guard<std::mutex> temp_lock(this->lock);
#endif // !SINGLE_THREAD
    if (this->leaf)
    {
        uint32_t action_dim = this->children.size(), i = 0;
        for (i = 0; i < action_dim; i++)
        {
            if (legal_action[i] == 1)
            {
                this->children[i] = new TreeNode(this, prior[i].item().toDouble(), action_dim);
            }
        }
        this->leaf = false;
        return true;
    }
    else return false;
}

void TreeNode::backup(double value, double virtual_loss, bool success)
{
    if (this->parent) this->parent->backup(-value, virtual_loss, success);
    else
    {
        // 根节点
        if (success) this->N += 1;
        return;
    }
    // 非根节点
#ifndef SINGLE_THREAD
    if (success)
    {
        // 移除虚拟损失  更新 W N Q
        this->N = this->N - (int)virtual_loss + 1;
        this->W = this->W + virtual_loss + value;
    }
    else
    {
        // 移除虚拟损失  恢复原值
        this->N = this->N - (int)virtual_loss;
        this->W = this->W + virtual_loss;
    }
#else
    this->N += 1;
	this->W += value;
#endif // !SINGLE_THREAD
    this->Q = this->W / this->N;
}

// MCTS
MCTS::MCTS(PolicyValueNet *network, uint32_t n_thread, double c_puct, double temp,
           uint32_t n_simulate, double virtual_loss, uint32_t action_dim, bool add_noise) :
        network(network),
#ifndef SINGLE_THREAD
        thread_pool(new ThreadPool(n_thread)),
#endif // !SINGLE_THREAD
        c_puct(c_puct),
        n_simulate(n_simulate),
        virtual_loss(virtual_loss),
        temp(temp),
        add_noise(add_noise),
        action_dim(action_dim),
        n_count(0),
        root(new TreeNode(nullptr, 1., action_dim))
{
    srand(time(nullptr));
    torch::set_num_threads(n_thread);
}

void MCTS::update_with_move(Loc last_move)
{
    TreeNode *root_local = this->root.get();
    // 为什么你一定要在selfplay时才利用这个特性呢？
    // 搞不懂
    // selfplay==true的条件去掉可以，但一定要保证mcts的update和nogo的execmove时刻对应
    // 这个代码mcts和nogo是分开的，mcts树里面没有任何action和state的信息，真是服了
    if (last_move >= 0 && last_move < root_local->children.size() && root_local->children[last_move] != nullptr)
    {
        // 利用子树 孩子节点作为根节点
        TreeNode *node = root_local->children[last_move];
        root_local->children[last_move] = nullptr;
        node->parent = nullptr;
        this->root.reset(node);
    }
    else this->root = std::make_unique<TreeNode>(nullptr, 1., this->action_dim);
    this->n_count++;
}

uint32_t binary_search(std::vector<double> &values, double target)
{
    uint32_t i = 0, j = values.size() - 1, m;
    // 左开右闭
    // (0,v[0]] (v[0],v[1]] (v[1],v[2]] ... (v[n-2],v[n-1]]  v[n-1]=1
    // 找到 target 落在哪个区间
    // 重复元素应该取第一个
    while (i < j)
    {
        m = (i + j) >> 1;
        if (values[m] >= target) j = m;
        else i = m + 1;
    }
    return i;
}

Loc MCTS::get_action(Nogo *nogo, bool explore)
{
    return this->get_action(this->get_action_prob(nogo), explore);
}

Loc MCTS::get_action(std::vector<double> action_prob, bool explore)
{
    uint32_t n = action_prob.size(), i = 0;
    // srand(time(0));
    /*if (explore)
    {
		double sum = 0;
        // 添加狄利克雷噪声
		at::Tensor noise = this->network->dirichlet_noise(n, 0.3);
		for (i = 0; i < n; i++)
		{
			if (nullptr == this->root->children[i]) action_prob[i] = 0;
			else action_prob[i] = 0.25 * noise[i].item().toDouble() + 0.75 * action_prob[i];
			sum += action_prob[i];
		}
		std::for_each(action_prob.begin(), action_prob.end(), [sum](double &x) { x /= sum; });
    }*/
    // 按权重随机选择
    for (i = 1; i < n; i++) action_prob[i] += action_prob[i-1];
    double p;
    uint32_t count = 0;
    while (true)
    {
        p = (double)(rand() % 1000) / 1000;
        // 二分查找
        i = binary_search(action_prob, p);
        if ((++count) > 2) std::printf("binary search count : %d\n", count);
        if (this->root->children[i]) break;
    }
    return i;
}

std::vector<double> MCTS::get_action_prob(Nogo *nogo)
{
    uint32_t i;
    // 根节点还未扩展，先扩展根节点
    if (this->root->leaf) this->simulate(nogo);
#ifndef SINGLE_THREAD
    std::vector<std::future<void>> futures;
    for (i = 0; i < this->n_simulate; i++)
    {
        // 提交模拟任务到线程池
        auto future = this->thread_pool->commit(std::bind(&MCTS::simulate, this, nogo));
        futures.emplace_back(std::move(future));
    }
    // 等待模拟结束
    for (i = 0; i < this->n_simulate; i++) futures[i].wait();
#else
    for (i = 0; i < this->n_simulate; i++) this->simulate(gomoku);
#endif // !SINGLE_THREAD

    std::vector<double> action_prob(this->action_dim, 0);
    std::vector<TreeNode *> & children = this->root->children;
    double sum = 0;
    uint32_t n, max_n = 0, size = children.size();
    for (i = 0; i < size; i++)
    {
        if (children[i])
        {
#ifndef SINGLE_THREAD
            n = children[i]->N.load();
#else
            n = children[i]->N;
#endif // !SINGLE_THREAD
            action_prob[i] = n;
            sum += n;
            max_n = n > max_n ? n : max_n;
        }
    }
    if (this->temp > 0 && this->temp <= 1e-3 + FLT_EPSILON)
    {
        // 选取次数最多的
        sum = 0;
        for (i = 0; i < action_prob.size(); i++)
        {
            if (abs(action_prob[i] - max_n) <= FLT_EPSILON) action_prob[i] = 1;
            else action_prob[i] = 0;
            sum += action_prob[i];
        }
    }
    else if (abs(this->temp - 1) > FLT_EPSILON)
    {
        sum = 0;
        for (i = 0; i < action_prob.size(); i++)
        {
            action_prob[i] = pow(action_prob[i], 1 / this->temp);
            sum += action_prob[i];
        }
    }
    if (sum <= FLT_EPSILON) std::cout << sum << std::endl;
    // 归一化
    std::for_each(action_prob.begin(), action_prob.end(), [sum](double &x) { x /= sum; });
    return action_prob;
}

void MCTS::simulate(Nogo *nogo)
{
    // 单次模拟
    // 模拟是否成功
    bool success = false;
    TreeNode *node = nullptr, *root = this->root.get();
    uint32_t action = 0, count = 0;
    double value = 0;
    //clock_t ts = clock();
    while (!success)
    {
        // 复制游戏状态
        Nogo game = *nogo;
        node = root;
        while (!node->leaf)
        {
            action = node->select(this->c_puct, this->virtual_loss);
            game.execute_move(Location::LocNN2Loc(action, game.get_n()));
            node = node->children[action];
        }
        // 游戏是否结束
        std::vector<int> res = game.get_game_status();
        if (res[0] == 0)
        {
            // 未结束 扩展 神经网络评估
            at::Tensor s = game.curr_state(true, this->network->device);
            // 输出包含batch维度
            std::vector<at::Tensor> pred = this->network->predict(s);
            value = pred[1][0].item().toDouble();
            // std::cout << value << std::endl;
            // std::cout << pred[0][0] << std::endl;

            std::vector<int> legal_move;
            game.get_legal_moves(legal_move);
            // 扩展
            at::Tensor prior = pred[0][0].to(torch::kCPU);
            if (this->add_noise)
            {
                // 添加狄利克雷噪声
                prior = 0.75 * prior + 0.25 * this->network->dirichlet_noise(game.get_action_dim(), 0.3);
            }
            success = node->expand(prior, legal_move);
//            if (!success)
//            std::cout << "模拟结果: " << success << ", count: " << count << std::endl;
        }
        else
        {
            // 游戏结束 实际价值（以当前玩家为视角）
            Player winner = res[1];
            value = winner == P_NULL ? 0 : (winner == game.get_current_color() ? 1 : -1);
            success = true;
        }
        // 当前状态的前一步动作为对手方落子，价值取反
        if (node != root) node->backup(-value, this->virtual_loss, success);
//        if ((++count) > 1) std::printf("simulation count : %d\n", count);
    }
//    std::printf("simulation : %d\n", clock() - ts);
}

int MCTS::self_play(Nogo *nogo, std::vector<at::Tensor> &states, std::vector<at::Tensor> &probs, std::vector<float> &values,
                    double temp, uint32_t n_round, bool add_noise, bool show)
{
    std::vector<int> res(2, 0);
    Loc move;
    Player idx;

    //通通初始化
    nogo->reset();
    this->reset();
    this->set_temp(temp); // 起始温度参数
    this->add_noise = add_noise;
    if (this->n_count != 0 || this->temp != temp) {
        throw std::runtime_error("self play not inited");
    }

    if (show)
    {
        std::cout << "New game." << std::endl;
        nogo->display();
    }
    std::vector<double> action_prob;
    std::vector<int> players;
    at::Tensor state;
    clock_t ts;
    uint32_t s0 = states.size();
    if (probs.size() != s0 || values.size() != s0)
    {
        states.clear();
        probs.clear();
        values.clear();
        s0 = 0;
    }
    while (0 == res[0])
    {
        idx = nogo->get_current_color();
        // 训练数据缓存不用CUDA
        state = nogo->curr_state(false, this->network->device);
        //ts = clock();
        action_prob = this->get_action_prob(nogo);
        //std::printf("get_action_prob : %d\n", clock() - ts);
        move = this->get_action(action_prob);
        if (show)
        {
            Size xsize = nogo->get_n();
            Loc locx = Location::getXNN(move, xsize);
            Loc locy = Location::getYNN(move, xsize);
            std::printf("Player '%c' : %d %d\n", Nogo::get_symbol(idx), locx, locy);
        }
        if (nogo->execute_move(Location::LocNN2Loc(move, nogo->get_n())))
        {
            this->update_with_move(move);
            states.emplace_back(state);
            probs.emplace_back(torch::tensor(action_prob));
            players.emplace_back(idx);
            res = nogo->get_game_status();
            if (show) nogo->display();
            if (this->n_count >= n_round) this->temp = 1e-3;
        }
    }

    this->reset();
    nogo->reset();

    if (show)
    {
        if (0 != res[1]) std::printf("Game end. Winner is Player '%c'.\n", Nogo::get_symbol(res[1]));
        else std::cout << "Game end. Tie." << std::endl;
    }
    int i, z;
    for (i = s0; i < states.size(); i++)
    {
        z = res[1] == 0 ? 0 : (players[i] == res[1] ? 1 : -1);
        values.emplace_back(z);
    }
    return res[1];
}

void MCTS::clear_tree() {
    this->root = std::make_unique<TreeNode>(nullptr, 1., this->action_dim);
}

void MCTS::reset() {
    clear_tree();
    n_count = 0;
}



// end

// begin /Users/syys/CLionProjects/torch_example/train.cpp
//
// Created by syys on 2021/3/2.
//


bool file_exists(const char * file)
{
    if (nullptr == file) return false;
    FILE *fp = fopen(file, "rb");
    if (nullptr != fp)
    {
        fclose(fp);
        fp = nullptr;
        return true;
    }
    return false;
}

void Train::augment_data(std::vector<at::Tensor> &states, std::vector<at::Tensor> &probs, std::vector<float> &values)
{
    uint32_t i, j, action_dim, state_h, size = states.size();
    if (0 == size || probs.size() != size || values.size() != size) return;
    // [batch channels height width]
    state_h = states[0].size(2);
    action_dim = probs[0].size(0);
    // action_dim = state_h * state_h;
    //at::Tensor s, p, z;
    //for (i = 0; i < size; i++)
    //{
    //	s = states[i];
    //	p = probs[i].reshape({ state_h,state_h });
    //	z = torch::tensor({ {values[i]} });
    //	for (j = 0; j < 4; j++)
    //	{
    //		this->push(s, p.reshape({ 1,action_dim }), z);
    //		// 上下翻转
    //		this->push(s.flip(2), p.flip(0).reshape({ 1,action_dim }), z);
    //		if (j == 3) break;
    //		// 旋转90度
    //		s = s.rot90(1, { 2,3 });
    //		p = p.rot90(1, { 0,1 });
    //	}
    //}

    uint32_t size0 = this->states.size(0) + (size << 3);
    if (size0 > this->N)
    {
        this->states = this->states.slice(0, size0 - this->N);
        this->probs = this->probs.slice(0, size0 - this->N);
        this->values = this->values.slice(0, size0 - this->N);
    }
    at::Tensor state = torch::cat(states, 0);
    at::Tensor prob = torch::stack(probs, 0).reshape({ size,this->nogo.get_n(),this->nogo.get_n() });
    at::Tensor value = torch::tensor(values).reshape({ size,1 });
    at::Tensor state_flip = state.flip(2);
    at::Tensor prob_flip = prob.flip(1);
    this->states = torch::cat({ this->states,state,state.rot90(1,{2,3}),state.rot90(2,{2,3}),state.rot90(3,{2,3}),
                                state_flip,state_flip.rot90(1,{2,3}),state_flip.rot90(2,{2,3}),state_flip.rot90(3,{2,3}) }, 0);
    this->probs = torch::cat({ this->probs,prob,prob.rot90(1,{1,2}),prob.rot90(2,{1,2}),prob.rot90(3,{1,2}),
                               prob_flip,prob_flip.rot90(1,{1,2}),prob_flip.rot90(2,{1,2}),prob_flip.rot90(3,{1,2}) }, 0);
    this->values = torch::cat({ this->values,value,value,value,value,value,value,value,value }, 0);
}

void Train::push(const at::Tensor &s, const at::Tensor &p, const at::Tensor &z)
{
    //while (this->values.size() >= this->N && this->N > 0)
    //{
    //	this->states.pop_front();
    //	this->probs.pop_front();
    //	this->values.pop_front();
    //}
    //this->states.emplace_back(s);
    //std::cout << this->states[this->states.size()-1] << std::endl;
    //this->probs.emplace_back(p);
    //this->values.emplace_back(z);
}

double Train::evaluate(const char *best_path_local, uint32_t num=50)
{
    PolicyValueNet network_local(best_path_local, true, this->state_c,
                                 this->nogo.get_n(), this->nogo.get_action_dim());
    MCTS mcts_train(&network_local, this->n_thread, this->c_puct, this->temp, this->n_simulate,
                    this->virtual_loss, this->nogo.get_action_dim(), true);
    this->mcts.set_temp(1e-3);
    mcts_train.set_temp(1e-3);

    int winner;
    bool swap = false;
    uint32_t i, count1 = 0, count2 = 0;
    for (i = 0; i < num; i++)
    {
        // 原作者这里的代码有问题吧，player1不一定play的是black，swap时就play的是white啊
        this->mcts.reset();
        mcts_train.reset();
        this->nogo.reset();
        winner = this->nogo.start_play(&this->mcts, &mcts_train, swap, eval_show);
        if (winner == P_BLACK) {
            if (swap) {
                count2++;
            } else {
                count1++;
            }
        }
        else if (winner == P_WHITE) {
            if (swap) {
                count1++;
            } else {
                count2++;
            }
        }
        swap = !swap;
        std::cout << "p1 vs p2: " << count1 << ", " << count2 << std::endl;
    }
    double ratio = (count1 + (double)(num - count1 - count2) / 2) / num;
    if (ratio > 0.55) {
        std::cout << "eval passed! save current net to best model!" << std::endl;
        this->network.save_model(best_path_local);
    }
    else {
        std::cout << "eval faild, keep this net not reloaded " <<
                     "and continue train this net until eval pass" << std::endl;
//        this->network.load_model(best_path_local);
    }
    return ratio;
}

void Train::run(const char *model_path_local, const char *best_path_local)
{
    uint32_t i, j, k, size, idx;
    if (!file_exists(best_path_local)) this->network.save_model(best_path_local);
    std::vector<double> res;
    double kl, best_ratio = 0, ratio;
    TimeCounter timer;
    for (i = 0; i < this->n_game; i++)
    {
        timer.start();
        std::vector<at::Tensor> states_local, probs_local, values_;
        std::vector<float> values_local;
        mcts.self_play(&this->nogo, states_local, probs_local, values_local, this->temp,
                explore_count,
                true, self_play_show);
        this->augment_data(states_local, probs_local, values_local);
        size = this->states.size(0);
        std::printf("game %4d/%d : duration=%.3fs  episode=%lu  buffer=%d\n", i+1, this->n_game, timer.end_s(), states_local.size(), size);
        states_local.clear(); probs_local.clear(); values_local.clear(); values_.clear();
        if (size < this->batch_size) continue;
        if ((i + 1) % this->check_freq == 0)
        {
            std::cout << "data collected, start train and eval" << std::endl;
            for (j = 0; j < this->epochs; j++)
            {
                std::cout << "epoch " << j+1 << ", train begin" << std::endl;
                at::Tensor index = torch::randperm(size, torch::Dtype::Long);
                at::Tensor index1;
                k = 0;
                while (k < size)
                {
                    timer.start();
                    index1 = index.slice(0, k, k + this->batch_size);
                    if (k + this->batch_size > size)
                    {
                        // 补齐batch
                        index1 = torch::cat({ index1,index.slice(0, 0, k + this->batch_size - size) }, 0);
                    }
                    res = this->train_step(this->states.index(index1), this->probs.index(index1).reshape({index1.size(0),this->nogo.get_action_dim()}),
                                           this->values.index(index1), this->lr * this->c_lr);
                    kl = res[2];
                    std::printf("train %3d/%d : cross_entropy_loss=%.8f  mse_loss=%.8f  kl=%.8f  R2_old=%.8f  R2_new=%.8f  c_lr=%.5f  duration=%.3fs\n",
                                j+1, this->epochs, res[0], res[1], kl, res[3], res[4], this->c_lr, timer.end_s());
                    k += this->batch_size;
                }
                if ((j+1) % eval_fre == 0) {
                    std::cout << "epoch " << j+1 << ", eval begin" << std::endl;
                    timer.start();
                    ratio = this->evaluate(best_path_local);
                    if (ratio > best_ratio) best_ratio = ratio;
                    std::printf("evaluate : ratio=%.8f  best_ratio=%.8f  duration=%.3fs\n", ratio, best_ratio, timer.end_s());
                }
            }
            this->network.save_model(model_path_local);
            // after eval finish, we nned to start selfplay with curr best model
            this->network.load_model(best_path_local);
        }
    }
}

std::vector<double> Train::train_step(const std::vector<at::Tensor> &state, const std::vector<at::Tensor> &prob, const std::vector<at::Tensor> &value, const double &lr)
{
    at::Tensor s = torch::cat(state, 0);
    at::Tensor p = torch::cat(prob, 0);
    at::Tensor z = torch::cat(value, 0);
    return this->train_step(s, p, z, lr);
}

std::vector<double> Train::train_step(const at::Tensor &state, const at::Tensor &prob, const at::Tensor &value, const double &lr)
{
    at::Tensor s = state.to(this->network.device);
    at::Tensor p = prob.to(this->network.device);
    at::Tensor z = value.to(this->network.device);
    /*auto param_groups = this->optimizer.param_groups();
    uint32_t i, n = param_groups.size();
    for (i = 0; i < n; i++)
    {
        param_groups[i].set_options(std::make_unique<torch::optim::AdamOptions>(torch::optim::AdamOptions(lr)));
    }*/
    this->optimizer.zero_grad();
    std::vector<at::Tensor> res = this->network.model->forward(s);
    at::Tensor loss1 = torch::binary_cross_entropy(res[0], p);
    at::Tensor loss2 = torch::mse_loss(res[1], z);
    at::Tensor loss = loss1 + loss2;
    loss.backward();
    this->optimizer.step();
    std::vector<at::Tensor> res1 = this->network.model->forward(s);
    // 新旧预测值的KL散度
    at::Tensor kl = (res1[0] * ((res1[0] + 1e-10).log() - (res[0] + 1e-10).log())).sum(1).mean();
    at::Tensor z_var = torch::var(z, 0, true, false);
    at::Tensor R2_old = 1 - torch::var(z - res[1], 0, true, false) / z_var;
    at::Tensor R2_new = 1 - torch::var(z - res1[1], 0, true, false) / z_var;
    return { loss1.item().toDouble(),loss2.item().toDouble(),kl.item().toDouble(),R2_old.item().toDouble(),R2_new.item().toDouble() };
}

double Train::eval_best_with(uint32_t num=50, const char *other_nn_path)
{
    PolicyValueNet network_local(other_nn_path, true, this->state_c,
                                 this->nogo.get_n(), this->nogo.get_action_dim());
    MCTS mcts_train(&network_local, this->n_thread, this->c_puct, this->temp, this->n_simulate,
                    this->virtual_loss, this->nogo.get_action_dim(), true);
    this->mcts.set_temp(1e-3);
    mcts_train.set_temp(1e-3);

    int winner;
    bool swap = false;
    uint32_t i, count1 = 0, count2 = 0;
    for (i = 0; i < num; i++)
    {
        // 原作者这里的代码有问题吧，player1不一定play的是black，swap时就play的是white啊
        this->mcts.reset();
        mcts_train.reset();
        this->nogo.reset();
        winner = this->nogo.start_play(&this->mcts, &mcts_train, swap, eval_show);
        if (winner == P_BLACK) {
            if (swap) {
                count2++;
            } else {
                count1++;
            }
        }
        else if (winner == P_WHITE) {
            if (swap) {
                count1++;
            } else {
                count2++;
            }
        }
        swap = !swap;
        std::cout << "p1 vs p2: " << count1 << ", " << count2 << std::endl;
    }
    double ratio = (count1 + (double)(num - count1 - count2) / 2) / num;
    if (ratio > 0.55) {
        std::cout << "best beats random, eval success and train works, your nn has gained intelligence!" << std::endl;
    }
    else {
        std::cout << "eval faild, your trained nn can't even beat random. there must be some bugs!" << std::endl;
    }
    return ratio;
}


// end

// begin /Users/syys/CLionProjects/torch_example/submit.cpp
//
// Created by syys on 2021/3/6.
//


int main()
{
    TimeCounter tc;
    Nogo nogo(COMPILE_MAX_BOARD_LEN);
    nogo.parse_botzone_input();
    static const char* other_nn_path = "./data/model-best.pt";
    uint32_t state_c = 2;
    uint32_t n_thread = 1;
    uint32_t c_puct=5;
    double temp=1e-3;
    uint32_t n_simulate=200;
    double virtual_loss=3;
    PolicyValueNet network_local(other_nn_path, true, state_c,
                                 nogo.get_n(), nogo.get_action_dim());
    tc.start();
    MCTS mcts_train(&network_local, n_thread, c_puct, temp, n_simulate,
                    virtual_loss, nogo.get_action_dim(), true);
    double nn_load_time = tc.end_s();
    mcts_train.reset();
    tc.start();
    Loc move = mcts_train.get_action(&nogo);
    double decison_time = tc.end_s();
    Loc outx = Location::getXNN(move, COMPILE_MAX_BOARD_LEN);
    Loc outy = Location::getYNN(move, COMPILE_MAX_BOARD_LEN);
    std::cout << outx << " " << outy << std::endl;
    std::cout << "nn load time: " << nn_load_time
                << ", decide time: " << decison_time << std::endl;
    return 0;
}
// end

