#pragma once

#ifndef __DQN_H_
#define __DQN_H_

#include <torch/torch.h>
#include "GNN/GNN.h"
#include "Environment/environment.h"

struct DQN : torch::nn::Module
{
    DQN(int64_t node_embedding_size, int64_t edge_embedding_size) 
        : Graph(*register_module<GNN>("GNN", std::make_shared<GNN>(node_embedding_size,edge_embedding_size)))
    {}

    torch::Tensor forward(environment env)
    {
        return Graph.forward(env.network,env.node_list);
    }
    GNN Graph;
};

#endif

