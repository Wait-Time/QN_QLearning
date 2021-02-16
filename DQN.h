#pragma once

#ifndef __DQN_H_
#define __DQN_H_

#include <torch/torch.h>
#include "GNN/GNN.h"
struct DQN : torch::nn::Module
{
    DQN(int64_t node_embedding_size, int64_t edge_embedding_size)
    {
        Graph = GNN(node_embedding_size,edge_embedding_size);
    }

    torch::Tensor forward(torch::Tensor input)//torch::Tensor adj, torch::Tensor wiegths, torch::Tensor nodes) //input graph and action //output scalar q_value
    {
        return Graph.forward(input);
    }
    GNN Graph;

};

#endif

