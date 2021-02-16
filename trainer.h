#pragma once

#ifndef __TRAINER_H_
#define __TRAINER_H_

#include "Environment/environment.h"
#include <torch/torch.h>
#include "DQN.h"

class trainer
{
    environment env;
    DQN policy_network, target_network;
    // Optimizer
    float gamma = 0.99;
    double epsilon_start = 1.0;
    double epsilon_final = 0.01;
    int64_t epsilon_decay = 30000;
    int64_t priority_levels = 1;
public:
    trainer();

    trainer(int64_t priority_levels_para = 1)
    {
        this->priority_levels = priority_levels_para;
    }
    
    void init_env();
    torch::Tensor state_to_tensor();
    torch::Tensor compute_td_loss();
    void train(int64_t num_epochs);
};

#endif