 #pragma once

#ifndef __GNN_H_
#define __GNN_H_

#include <torch/torch.h>
#include <memory>

struct fe : torch::nn::Module
{
public:
    fe();
    fe(int64_t in_channels,int64_t out_channels) 
        : linear1(register_module("fe.linear1", torch::nn::Linear(in_channels,512) )),
          linear2(register_module("fe.linear2", torch::nn::Linear(512,out_channels)))
    {}
    torch::Tensor forward(torch::Tensor input)
    {
        input = torch::nn::ReLU(linear1(input));
        input = torch::nn::ReLU(linear2(input));
        return input;
    }
    torch::nn::Linear linear1, linear2;
};


struct GNN : torch::nn::Module
{
public:
    GNN();
    GNN(int64_t node_embedding_size,int64_t edge_embedding_size)
        : forward_message(*register_module<fe>("forward_message",std::make_shared<fe>(2*node_embedding_size+edge_embedding_size, node_embedding_size))),
          backward_message(*register_module<fe>("forward_message",std::make_shared<fe>(2*node_embedding_size+edge_embedding_size, node_embedding_size)))
    {}
    
    torch::Tensor forward(std::vector< std::vector< std::pair<int,float> > > network, std::vector<node> node_list)
    {
        int N = network.size() ; 
        std::vector< std::vector< std::pair<int,float> > > reverse_network(N);
        for(int i=0;i<N;i++)
        {
            for(std::pair<int,float> x:network[i])
            {
                reverse_network[x.first].push_back({i,x.second}) ;
            }
        }
       
        std::vector<torch::Tensor> a_list;
        int T = 100; // HYperParameter
        for(int j =0; j<T;j++)
        {
            for(int i=0;i<N;i++)
            {
                torch::Tensor a = torch::zeros({node_list[0].num_priority*node_list[0].service[0].b+1,1});

                for(auto& incoming_node: reverse_network[i])
                {
                    a += forward_message.forward( at::cat( { node_list[i].convert_to_tensor(), node_list[incoming_node.first].convert_to_tensor(), torch::tensor(incoming_node.second).reshape({1}) } ) );
                }
                for(auto& outgoing_node: network[i])
                {
                    a += backward_message.forward( at::cat( { node_list[i].convert_to_tensor(), node_list[outgoing_node.first].convert_to_tensor(), torch::tensor(incoming_node.second).reshape({1}) } ) );
                }
                // auto r_ptr = a.data_ptr<float>();
                // std::vector<float> node_vector{r_ptr,r_ptr+a.size(0)};
                node_list[i].load_vector(a);
                if( j == T-1)
                {
                    a_list.push_back(a);
                }
            }
        }
        // Dimension increase, aggregation step, Graph vector
        // alpha_list 
        
        //Perform action ?

        // Q_value
        
        

    }
    fe forward_message, backward_message;
};

#endif