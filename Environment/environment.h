#pragma once

#ifndef __ENVIRONMENT_H_
#define __ENVIRONMENT_H_

#include "../../simulation/components/queue_graphv2.h"
#include "node.h"

class environment
{
    std::vector<distribution> arrival;
    int64_t num_priority;
    int64_t num_events;
    std::vector< std::vector< std::pair<int,float> > > network; 
    std::vector<node> node_list;
public:
    environment(): network({}),node_list({node()}){}
    environment(std::vector<distribution> arrival_para, std::vector< std::vector< std::pair<int,float> > > network_para, std::vector<node> node_list_para, int64_t num_priority_para,int64_t num_events_para=100000)
    {
        this->arrival = arrival_para;
        this->network = network_para;
        this-> node_list = node_list_para;
        this-> num_priority = num_priority_para;
        this-> num_events = num_events_para;
    }
    void simulate();
    void calculate_reward();
    void perform_action(int64_t action,std::vector<float> prob);
};

#endif