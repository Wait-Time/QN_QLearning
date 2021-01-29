#include "environment.h"

void environment::perform_action(int64_t action)
{
    if(action == 0)
        return ;
    else
    {
        node_list.push_back(node());
        network.push_back({});
        int64_t i = 0;
        while(action>0)
        {
            if(action%2)
            {
                network[i].push_back((int)node_list.size()-1);
            }
            action/=2;
            i++;
        }
    }
}

void environment::simulate()
{
    std::vector<station> temp;
    for(auto &x:node_list)
    { 
        temp.push_back(x.convert_to_station());
    }
    graph(0,0,network,temp); // Change Graph.h
}