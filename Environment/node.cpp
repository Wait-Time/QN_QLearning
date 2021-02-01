#include "node.h"
#include "../../simulation/components/station.h"

station node::convert_to_station()
{
    event_type_list dept_para,patience_para;
    for(auto&x: service)
    {
        dept_para.push_back(x.sampler());
    }
    for(auto&x: patience)
    {
        patience_para.push_back(x.sampler());
    }
    return station(mxN,C,dept_para,num_priority,patience_para);
}