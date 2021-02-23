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
    int b;
    int num_events = 10000;
public:
    trainer();

    trainer(int64_t priority_levels_para = 1,int b_para = 1000)
    {
        this->priority_levels = priority_levels_para;
        this->b = b_para;
        policy_network = DQN(2*priority_levels_para*b_para+1,1); // First b entry for service, next b for patience, last for num_servers
        target_network = DQN(2*priority_levels_para*b_para+1,1);
    }
    
    void init_env(float service_rate,float patience_rate,float arrival_rate,int mxN);
    torch::Tensor compute_td_loss();
    void train(int64_t num_epochs);
};

#endif