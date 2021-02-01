#pragma once

#ifndef __NODE_H_
#define __NODE_H_

#include "../distribution.h"
#include "../../simulation/components/queue_graphv2.h"
class node
{
    // Service dist
    // C func server vs Time
    // Patience times
    // Priority levels
    std::vector<distribution> service;
    std::vector<distribution> patience;
    C_type C;
    int mxN;
    int64_t num_priority;
public:
    node();
    node(int mxN_para,int64_t num_priority_para, C_type C_para,std::vector<distribution> service_para, std::vector<distribution> patience_para)
    {
        this->mxN = mxN_para;
        this-> num_priority = num_priority_para;
        this->C = C_para;
        this->service = service_para;
        this->patience = patience_para;
    }
    station convert_to_station();
};

#endif