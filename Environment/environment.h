#pragma once

#ifndef __ENVIRONMENT_H_
#define __ENVIRONMENT_H_

#include "../../simulation/components/queue_graphv2.h"
#include "node.h"
#include "../includes.h"
#include <torch/torch.h>

class environment
{
public:
    std::vector<distribution> arrival;
    int64_t num_priority;
    int64_t num_events;
    std::vector< std::vector< std::pair<int,float> > > network; 
    std::vector<node> node_list;
    std::vector<float> simulated_data;
    int b; // num-quantiles
    
    environment(std::vector<distribution> arrival_par,int64_t num_priority_para, node init_node, int b_para = 1000, int64_t num_events_para=10000): 
    network( { {} } ),node_list( {init_node} )
    {
        this->arrival = arrival_par;
        this-> num_priority = num_priority_para;
        this-> b = b_para;
        this-> num_events = num_events_para;
    }
    environment(std::vector<distribution> arrival_para, std::vector< std::vector< std::pair<int,float> > > network_para, std::vector<node> node_list_para, int64_t num_priority_para,int b_para = 1000  ,int64_t num_events_para=10000)
    {
        this->arrival = arrival_para;
        this->network = network_para;
        this-> node_list = node_list_para;
        this-> num_priority = num_priority_para;
        this-> b = b_para;
        this-> num_events = num_events_para;
    }
    void simulate();
    float reward(std::vector<float> input,float sigma = 1e-4);
    void perform_action(int64_t action,std::vector<float> alpha_list,node new_node);
    void pretty_print_network();
    int64_t get_size();
};

#endif