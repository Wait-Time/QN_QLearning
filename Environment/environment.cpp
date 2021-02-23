#include "environment.h"

// float exponentialrv(float lambda)
// {
//     float U = random;
//     return -log(U)/lambda;
// }

void environment::perform_action(int64_t action,std::vector<float> alpha_list,node new_node)// value for every node)
{
    if(action == 0)
        return ;
    else
    {
        node_list.push_back(
           new_node
        ); // What type of node to initialize Will be fed from a global vector also

        network.push_back({});
        int64_t i = 0;
        while(action>0)
        {
            if(action%2)
            {
                if( !network[i].empty() )
                {
                    for(auto&x: network[i])
                    {
                        x.second = x.second/(1.00+alpha_list[i]);
                    }
                    network[i].push_back( { (int)node_list.size()-1, alpha_list[i]/(1.00+alpha_list[i]) } );
                }
                else
                    network[i].push_back( { (int)node_list.size()-1, 1.00 } );
            }
            action/=2;
            i++;
        }
    }
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
        if(discrete_events%1000==0)
        {
            std::cout<<"Writing to CSV\n";
            simulator.dump_counter_variable_memory("./graph");
        }
        discrete_events++;
        // std::cout<<discrete_events<<endl;
    }
    this->simulated_data = read_csv("graph.csv",8);
    // std::cout<<"Writing to CSV\n";
    // simulator.write_to_csv("./output/graph");


}

float environment::reward(std::vector<float> input,float sigma)
{
    // Chi-square fitness test possible
    // Area under curve
    float tot_var = distribution::area_between_dist( distribution(input, this->b), distribution(this->simulated_data,this->b));

    //Dirac Delta Approximated as Normal
    return exp( -1*( pow( tot_var,2 )/(2*sigma) ) )/(sigma*sqrt(2*M_PIf64));
}

void environment::pretty_print_network()
{
    //std::vector< std::vector< std::pair<int,float> > > network;      
    for(int i=0;i<network.size();i++)
    {   std:: cout << '\n' ;
        std::cout <<"NODE " << i << ": " ;
        for(std::pair<int,float> x:network[i]) 
        {
            std::cout << "{" << x.first << "," << x.second << "}"  ;
            if(x!=network[i].back()) 
                std::cout << "," ;
        }std::cout << '\n' ;
    }
}

int64_t environment::get_size()
{
    return network.size();
}

torch::Tensor environment::to_pytorch_tensor()
{
    // arrival dist
    // network
    // node list
    std::vector<torch::Tensor> node_list;
    for(auto&x : node_list)
    {
        node_list.push_back(x.convert_to_tensor());
    }
    at::cat(node_list);
}