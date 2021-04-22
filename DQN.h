#pragma once

#ifndef __DQN_H_
#define __DQN_H_

#include <torch/torch.h>
#include "GNN/GNN.h"
#include "Environment/environment.h"

struct DQN : torch::nn::Module
{
    DQN();
    DQN(c10::IntArrayRef node_embedding_size, int64_t edge_embedding_size) 
        : Graph(*register_module<GNN>("GNN", std::make_shared<GNN>(node_embedding_size,edge_embedding_size)))
    {}

    std::vector<torch::Tensor> forward(std::vector<torch::Tensor> state)
    {
        state[0].sizes();
        // State - service tensor {|V|,p,b}, patience_tensor {p,b} ,C_tensor - {|V|}, Adjacency matrix {|V|,|V|}
        return Graph.forward(state);
    }
    GNN Graph;
};

#endif

