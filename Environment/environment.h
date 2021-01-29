#pragma once

#ifndef __ENVIRONMENT_H_
#define __ENVIRONMENT_H_

#include "../../simulation/components/queue_graph.h"
#include "node.h"

class environment
{
    distribution arrival;
    int64_t num_priority;
    std::vector< std::vector<int> > network;
    std::vector<node> node_list;
public:
    environment(): network({}),node_list({node()}){}
    void simulate();
    void calculate_reward();
    void perform_action(int64_t action);
};

#endif