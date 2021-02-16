#include <torch/torch.h>

struct GNN : torch::nn::Module
{
    GNN();
    GNN(int64_t node_embedding_size,int64_t edge_embedding_size)
    {
        forward_message = fe( 2*node_embedding_size+edge_embedding_size, node_embedding_size );
        backward_message = fe( 2*node_embedding_size+edge_embedding_size, node_embedding_size );
    }
    torch::Tensor forward(torch::Tensor input)
    {
        // adj,weights,nodes vector list. from input tensor
        // GNN architecture
        
    }
    fe forward_message, backward_message;

};

struct fe : torch::nn::Module
{
    fe();
    fe(int64_t in_channels,int64_t out_channels)
    {
        linear1 = torch::nn::Linear(in_channels, 512);
        linear2 = torch::nn::Linear(512, out_channels);
    }
    torch::Tensor forward(torch::Tensor input)
    {
        input = torch::relu(linear1(input));
        input = torch::relu(linear2(input));
        return input;
    }
    torch::nn::Linear linear1, linear2;
};