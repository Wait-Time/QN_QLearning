#include <torch/torch.h>
#include <iostream>
#include "../simulation/components/station.h"
#include "../simulation/components/queue_graphv2.h"
#include "includes.h"

using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;





class distribution
{
public:
    int b;
    std::vector<float> quantiles;

    distribution();
    distribution(int b_para, std::vector<float> quantiles_para)
    {
        this->b = b_para;
        this->quantiles = quantiles_para;
    }
    

    distribution(event_type sampler,int b_para,int64_t n= 100000)
    {
        this->b = b_para;
        // Generate sorted samples
        // Apply H&L Method
        /*
        n - samples
        p - percentile 0...100 100 groups (percentile) <-> b groups (b quantile)
        METHOD 10 (“H&L-2”):  The  Pth  percentile value is found by taking that value with  #(np + 0.5).  If this is not an integer, take the interpolated value between    and  # .  As an example, if  
        S5 = (1, 2, 3, 4, 5) and  p = .25, then  #(np + 0.5) = #(1.75) and so  Q1 = 1.75.  
        */
        std::vector< std::pair<float,float> > samples;
        for (size_t i = 0; i < n; i++)
        {
            samples.push_back( {sampler(0), 0} );
        }
        std::sort(samples.begin(),samples.end());

        for (size_t i = 0; i < n; i++)
        {
            // std::cout<<samples[i].first <<',';
            samples[i].second = (i+0.5)/n;
        }
        //    0...100 <-> b groups kth b-quantile 

        for (size_t k = 1; k < b; k++)
        {
            // ((j-1)-0.5)/n <-> (j-0.5)/n
            // ((j-1)-0.5)/n <= k*100/b
            // (k*100/b)
            int j = int( 1.0*k*n/b + 1.5 );
            quantiles.push_back( (samples[j-1].first - samples[j].first)/(samples[j-1].second - samples[j].second) * ( 1.0*k/b - samples[j].second ) + samples[j].first );

            // simpler Algo
            // if( (k*b)%n == 0)
            // {
            //     quantiles.push_back( (samples[k*b/n].first + samples[k*b/n+1].first)/2 );
            // }
            // else
            //     quantiles.push_back( samples[int(k*b/n)+1].first );
        }        
    }
    
    distribution(std::vector<float> data,int b_para)
    {
        
        this->b = b_para;

        int n = data.size();
        std::vector< std::pair<float,float> > samples;
        for (size_t i = 0; i < n; i++)
        {
            samples.push_back( { data[i], 0} );
        }

        std::sort(samples.begin(),samples.end());

        for (size_t i = 0; i < n; i++)
        {
            // std::cout<<samples[i].first <<',';
            samples[i].second = (i+0.5)/n;
        }
        //    0...100 <-> b groups kth b-quantile 

        for (size_t k = 1; k < b; k++)
        {
            // ((j-1)-0.5)/n <-> (j-0.5)/n
            // ((j-1)-0.5)/n <= k*100/b
            // (k*100/b)
            int j = int( 1.0*k*n/b + 1.5 );
            quantiles.push_back( (samples[j-1].first - samples[j].first)/(samples[j-1].second - samples[j].second) * ( 1.0*k/b - samples[j].second ) + samples[j].first );

            // simpler Algo
            // if( (k*b)%n == 0)
            // {
            //     quantiles.push_back( (samples[k*b/n].first + samples[k*b/n+1].first)/2 );
            // }
            // else
            //     quantiles.push_back( samples[int(k*b/n)+1].first );
        }        
    }

    void print_quantiles()
    {
        std::cout<<'[';
        for(auto&x: quantiles)
        {
            std::cout<<x <<',';
        }
        std::cout<<']';
    }

    event_type sampler()
    {
        int b_para = this->b;
        std::vector<float> quantiles_para = this->quantiles;
        return [b_para,quantiles_para](float t) -> float { float U = random; return quantiles_para[int(b_para*U)]; };
    }

    static float area_between_dist(distribution A,distribution B)
    {
        assert(A.b==B.b);
        float res = 0;
        for (size_t i = 0; i < A.b; i++)
        {
            res += abs(A.quantiles[i] - B.quantiles[i]);
        }
        res /= A.b;
        return res;
    }
    torch::Tensor convert_to_tensor()
    {
        int len = quantiles.size();
        auto options = torch::TensorOptions().dtype(torch::kFloat64);
        return torch::from_blob(quantiles.data(), {1,len}, options);
        // return torch::tensor(quantiles.data());
    }
};



class node
{
public:
    // Service dist
    // C func server vs Time
    // Patience times
    // Priority levels
    std::vector<distribution> service;
    std::vector<distribution> patience;
    C_type C;
    int mxN;
    int64_t num_priority;
    node();
    node(int mxN_para,int64_t num_priority_para, C_type C_para,std::vector<distribution> service_para, std::vector<distribution> patience_para)
    {
        this->mxN = mxN_para;
        this-> num_priority = num_priority_para;
        this->C = C_para;
        this->service = service_para;
        this->patience = patience_para;
    }
    station convert_to_station()
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

    std::vector<torch::Tensor> convert_to_tensor()
    {
        std::vector<torch::Tensor> services,patiences;
        for(distribution& temp: this->service)
        {
            services.push_back(temp.convert_to_tensor()); // size b
        }
        for(distribution &temp: this->patience)
        {
            patiences.push_back(temp.convert_to_tensor());
        }
        int b = this->service[0].b-1;
        int p = this->num_priority;
        // std::cout<< at::cat(services).sizes();
        torch::Tensor service_tensor = at::cat(services).reshape({1,p,b});
        torch::Tensor patience_tensor = at::cat(patiences).reshape({1,p,b});
        // tensor.push_back(torch::tensor(this->mxN).reshape({1}));
        // auto options = torch::TensorOptions().dtype(torch::kFloat64);
        int c = this->C(0);
        std::vector<torch::Tensor> node_vector = 
        {
            at::cat({
                service_tensor,
                patience_tensor
            },0),
            torch::tensor(c,torch::TensorOptions().dtype(torch::kInt32)).reshape({1})
            // torch::from_blob(&c,{1},torch::TensorOptions().dtype(torch::kInt32))
        };
        return node_vector; // node vector
    }

    void load_vector(std::vector<torch::Tensor> node_vector) // dim = {2,p,b}
    {
        torch::Tensor distribution_data = node_vector[0];
        torch::Tensor C_tensor = node_vector[1];

        int b = service[0].b;

        assert( (service[0].b - 1) == distribution_data.size(2) );
        assert( (num_priority) == distribution_data.size(1) );

        auto services_tensor = distribution_data.index({0,Ellipsis});
        auto patience_tensor = distribution_data.index({1,Ellipsis});
        
        std::vector<distribution> service_temp;
        std::vector<distribution> patience_temp;

        for(int i =0; i<this->num_priority;i++)
        {
            auto r_ptr = services_tensor.index({i,Ellipsis}).data_ptr<float>();
            std::vector<float> quantile{r_ptr,r_ptr+b-1};
            service_temp.push_back(distribution(b,quantile));
        }
        this->service = service_temp;

        for(int i =0; i<this->num_priority;i++)
        {
            auto r_ptr = patience_tensor.index({i,Ellipsis}).data_ptr<float>();
            std::vector<float> quantile{r_ptr,r_ptr+b-1};
            patience_temp.push_back(distribution(b,quantile));
        }
        this->patience = patience_temp;
        auto r_ptr = C_tensor.data_ptr<int>();
        int data = *(r_ptr+1);
        if( data <= this->mxN )
            this->C = [data](float t)-> int{ return data;};
        else
            this->C = [this](float t)-> int{ return this->mxN;};
    }
};

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
    
    environment();

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
    
    void simulate()
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

    float reward(std::vector<float> input,float sigma = 1e-4)
    {
        // Chi-square fitness test possible
        // Area under curve
        float tot_var = distribution::area_between_dist( distribution(input, this->b), distribution(this->simulated_data,this->b));

        //Dirac Delta Approximated as Normal
        return exp( -1*( pow( tot_var,2 )/(2*sigma) ) )/(sigma*sqrt(2*M_PIf64));
    }
    
    void perform_action(int64_t action,std::vector<float> alpha_list,node new_node)
    {
        assert(alpha_list.size() == node_list.size());
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
    
    void pretty_print_network()
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
    
    int64_t get_size()
    {
        return network.size();
    }

    std::vector<torch::Tensor> state_to_tensor()
    {
        std::vector<torch::Tensor> distribution_vector;
        std::vector<torch::Tensor> C_vector;
        for(auto node: node_list)
        {
            std::vector<torch::Tensor> node_vector = node.convert_to_tensor();
            distribution_vector.push_back(node_vector[0].reshape({1,node_vector[0].size(0),node_vector[0].size(1),node_vector[0].size(2)})); // {1,2,p,b}
            C_vector.push_back( node_vector[1] );
        }
        torch::Tensor distribution_vector_tensor =  at::cat(distribution_vector); // {|V|,2,p,b}
        torch::Tensor C_vector_tensor =  at::cat(C_vector); // |V|

        // {|V|,|V|} Adjacency matrix
        // std::vector< std::vector< std::pair<int,float> > > network; 
        int V = node_list.size();
        std::vector<torch::Tensor> adjacency_matrix;
        bool is_empty = true;
        for(auto& connected_nodes: network)
        {
            is_empty = false;

            std::vector<float> adjacency_row(V,0.0);

            for(auto& connected_node:connected_nodes)
            {
                int index = connected_node.first;
                float prob = connected_node.second;
                if(prob > 0.0)
                    adjacency_row[index] = prob;
            }
            // std::cout<<"ROW:"<<adjacency_row<<endl;
            adjacency_matrix.push_back( torch::tensor(adjacency_row, torch::TensorOptions().dtype(torch::kFloat64)).reshape({1,V}) ); //is_sparse?
        }
        torch::Tensor adjacency_matrix_tensor;
        if(is_empty)
            adjacency_matrix_tensor = torch::tensor( 0.0,torch::TensorOptions().dtype(torch::kFloat64)).reshape({1,1});
        else
            adjacency_matrix_tensor = at::cat(adjacency_matrix); // {|V|,|V|}

        return {distribution_vector_tensor,C_vector_tensor,adjacency_matrix_tensor};
    }
};


int main()
{
    
}