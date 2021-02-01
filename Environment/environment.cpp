#include "environment.h"

void environment::perform_action(int64_t action,std::vector<float> prob)
{
    // if(action == 0)
    //     return ;
    // else
    // {
    //     node_list.push_back(node());
    //     network.push_back({});
    //     int64_t i = 0;
    //     while(action>0)
    //     {
    //         if(action%2)
    //         {
    //             // network[i].push_back( (int)node_list.size()-1 );
    //         }
    //         action/=2;
    //         i++;
    //     }
    // }
}

void environment::simulate()
{
    // Convert back to the simulator code
    std::vector<station> temp;
    for(auto &x:node_list)
    { 
        temp.push_back(x.convert_to_station());
    }
    graphv2 simulator(0,0,network,temp);

    event_type_list arrival_processes;
    for(auto&x: arrival)
    {
        arrival_processes.push_back(x.sampler());
    }

    float least_dep_time = 0;
    int least_station_index = 0;

	std::tie(std::ignore,least_dep_time) = simulator.find_least_dep_time();

    int discrete_events = 0;
    float t = 0;
    int arriving_customer = 0;
    int priority;
    std::vector<float> ta = call_event_type_list(arrival_processes);

    // temp.print_system_status(T(t));
    // temp.logger(t);
    simulator.initialize_CSV("./graph");

    while(discrete_events < num_events)
    {
        std::tie(least_station_index, least_dep_time) = simulator.find_least_dep_time();
        // station_graph.server_updates();
        t = std::min(least_dep_time, *std::min_element(ta.begin(),ta.end()) );

        simulator.server_updates(t);

        if(t ==  *std::min_element(ta.begin(),ta.end()) )
        {
            priority = std::distance(ta.begin(), std::min_element(ta.begin(),ta.end()) );
            simulator.add_customer_to_graph(t,{priority,arriving_customer});
            arriving_customer++;
            ta[priority] = t + arrival_processes[priority](t) ;
        }
        else
            simulator.departure_updates(least_station_index,t);
        
        // simulator.logger(t);
        if(discrete_events%10000==0)
        {
            std::cout<<"Writing to CSV\n";
            simulator.dump_counter_variable_memory("./graph");
        }
        discrete_events++;
        // std::cout<<discrete_events<<endl;
    }
    // std::cout<<"Writing to CSV\n";
    // simulator.write_to_csv("./output/graph");

}