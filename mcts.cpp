//
// Created by syys on 2021/2/28.
//

#include <cmath>
#include <cfloat>
#include <memory>
#include <numeric>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include "mcts.h"

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
#ifdef _BOTZONE_ONLINE
    TimeCounter tcm;
    tcm.start();
#endif
    for (i = 0; i < this->n_simulate; i++) {
        #ifdef _BOTZONE_ONLINE
        double timedur = tcm.end_s();
        if (timedur > 0.80) {
            break;
        }
        #endif
        this->simulate(nogo);
    }
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


