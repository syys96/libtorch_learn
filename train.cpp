//
// Created by syys on 2021/3/2.
//

#include "train.h"

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

double Train::evaluate(const char *best_path_local, uint32_t num=20)
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
        winner = this->nogo.start_play(&this->mcts, &mcts_train, swap, false);
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
        std::cout << "eval passed!" << std::endl;
        this->network.save_model(best_path_local);
    }
    else {
        std::cout << "eval faild, net back to best checkpoint" << std::endl;
        this->network.load_model(best_path_local);
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
                COMPILE_MAX_BOARD_LEN*COMPILE_MAX_BOARD_LEN/8,
                true, true);
        this->augment_data(states_local, probs_local, values_local);
        size = this->states.size(0);
        std::printf("game %4d/%d : duration=%.3fs  episode=%lu  buffer=%d\n", i, this->n_game, timer.end_s(), states_local.size(), size);
        states_local.clear(); probs_local.clear(); values_local.clear(); values_.clear();
        if (size < this->batch_size) continue;
        //for (k = 0; k < size; k++)
        //{
        //	states.push_back(this->states[k]);
        //	probs.push_back(this->probs[k]);
        //	values_.push_back(this->values[k]);
        //}
        for (j = 0; j < this->epochs; j++)
        {
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
                            j, this->epochs, res[0], res[1], kl, res[3], res[4], this->c_lr, timer.end_s());
                k += this->batch_size;
            }
            //at::Tensor index = torch::randint(size, this->batch_size);
            //states.clear(); probs.clear(); values.clear(); values_.clear();
            //for (k = 0; k < this->batch_size; k++)
            //{
            //	idx = index[k].item().toInt();
            //	states.push_back(this->states[idx]);
            //	probs.push_back(this->probs[idx]);
            //	values_.push_back(this->values[idx]);
            //}
            //res = this->train_step(states, probs, values_, this->lr * this->c_lr);
            //kl = res[2];
            //if (kl > this->kl_targ * 2 && this->c_lr > 0.1) this->c_lr /= 1.5;
            //else if (kl < this->kl_targ / 2 && this->c_lr < 10) this->c_lr *= 1.5;
            //std::printf("train %3d/%d : cross_entropy_loss=%.8f  mse_loss=%.8f  kl=%.8f  R2_old=%.8f  R2_new=%.8f  c_lr=%.5f  duration=%.3fs\n",
            //		j, this->epochs, res[0], res[1], kl, res[3], res[4], this->c_lr, timer.end_s());
        }
        this->network.save_model(model_path_local);
        if ((i + 1) % this->check_freq == 0)
        {
            timer.start();
            ratio = this->evaluate(best_path_local);
            if (ratio > best_ratio) best_ratio = ratio;
            std::printf("evaluate : ratio=%.8f  best_ratio=%.8f  duration=%.3fs\n", ratio, best_ratio, timer.end_s());
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
