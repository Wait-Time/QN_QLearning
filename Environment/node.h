#pragma once

#ifndef __NODE_H_
#define __NODE_H_

#include "../distribution.h"

class node
{
    // Service dist
    // C func server vs Time
    // Patience times
    // Priority levels
    distribution service;
    std::vector<distribution> patience;
    C_type C;
    int64_t num_priority;
public:
    node();
    station convert_to_station();
};

#endif