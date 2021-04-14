#include <torch/torch.h>
#include <iostream>
#include "../simulation/components/station.h"
#include "../simulation/components/queue_graphv2.h"
#include "includes.h"


using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;

float exponentialrv(float lambda)
{
    float U = random;
    return -log(U)/lambda;
}


struct fe : torch::nn::Module
{
public:
    fe();
    fe(c10::IntArrayRef node_embedding_size, int64_t edge_embedding_size) 
        : 
        // conv1(torch::nn::Conv2dOptions(2*node_embedding_size[0],3,{1,50}).stride({1,3}).padding({2,0})),
        conv1(torch::nn::Conv2dOptions(3,10,{node_embedding_size[0],1}).padding({node_embedding_size[0],0})),
        conv2(torch::nn::Conv2dOptions(10,100,{node_embedding_size[0],node_embedding_size[1]/2+1})),
        conv3(torch::nn::Conv2dOptions(100,50,{node_embedding_size[0],node_embedding_size[1]/4}).stride({1,node_embedding_size[1]/16})),
        linear1(torch::nn::LinearOptions(753,50))
    {
        conv1->to(torch::kFloat64);
        conv2->to(torch::kFloat64);
        conv3->to(torch::kFloat64);
        linear1->to(torch::kFloat64);
    }
    torch::Tensor forward(std::vector<torch::Tensor> input)
    {
        int p = input[0].size(0);
        int b = input[0].size(1);
        // std::cout<<input[2].flatten() <<endl;

        torch::Tensor serv_u = input[0].reshape({1,1,p,b}); // {1,p,b} 
        torch::Tensor serv_v = input[1].reshape({1,1,p,b}); // {1,p,b} 
        torch::Tensor pat = input[4].reshape({1,1,p,b}); // {1,p,b} 
        
        // -> {2*p*b+2*p*b+1}
        // conv1->forward( torch::cat({hu,hv},0) );
        torch::Tensor x;
        x = torch::cat({serv_u,serv_v,pat},1); // {1,3,p,b}

        x = conv1->forward(x);
        x = conv2->forward(x);
        x = conv3->forward(x);
        x = x.flatten();

        torch::Tensor c_ = input[3].to(torch::kFloat64).reshape({2});// not float not differentiable
        torch::Tensor puv = input[2].reshape({1});
        x = at::cat({c_,puv,x}); // {2+1+50*5}

        x = linear1->forward(x);
        return x;
    }
    torch::nn::Linear linear1;
    torch::nn::Conv2d conv1,conv2,conv3;
};


struct GNN : torch::nn::Module
{
public:
    GNN();
    GNN(c10::IntArrayRef node_embedding_size, int64_t edge_embedding_size)
        : forward_message(*register_module<fe>("forward_message", std::make_shared<fe>(node_embedding_size,edge_embedding_size))),
          backward_message(*register_module<fe>("backward_message", std::make_shared<fe>(node_embedding_size,edge_embedding_size))),
          service(torch::nn::LinearOptions(50,node_embedding_size[0]*node_embedding_size[1])),
          alpha(torch::nn::LinearOptions(50,1)),
          servers(torch::nn::LinearOptions(50,1)),
          gated(torch::nn::LinearOptions(50,1)),
          patience(torch::nn::LinearOptions(50,node_embedding_size[0]*node_embedding_size[1])),
          Q_value(torch::nn::LinearOptions(50,1))
    {
        forward_message.to(torch::kFloat64);
        backward_message.to(torch::kFloat64);
        service->to(torch::kFloat64);
        servers->to(torch::kFloat64);
        alpha->to(torch::kFloat64);
        gated->to(torch::kFloat64);
        patience->to(torch::kFloat64);
        Q_value->to(torch::kFloat64);

    }
    
    std::vector<torch::Tensor> forward(std::vector<torch::Tensor> input)
    {
        // Dimension increase, aggregation step, Graph vector

        // State - service tensor {|V|,p,b}, patience_tensor {p,b} ,C_tensor - {|V|}, Adjacency matrix {|V|,|V|}
        
        torch::Tensor distribution_tensor = input[0];
        int V = distribution_tensor.size(0);
        int p = distribution_tensor.size(1);
        int b = distribution_tensor.size(2);
        torch::Tensor patience_tensor = input[1];
        torch::Tensor C_tensor = input[2];
        torch::Tensor adjacency_matrix = input[3];

        float* ptr = (float*)adjacency_matrix.data_ptr();

        torch::Tensor a_V = torch::zeros({V,50},torch::TensorOptions().dtype(torch::kFloat64));

        for( int i = 0 ; i < adjacency_matrix.size(0) ; i++ )
        {
            for( int j = 0 ; j < adjacency_matrix.size(1) ; j++ )
            {
                if(*ptr > 0)
                {
                    // i->j is connected
                    // j<-i back message
                    std::cout<<"i,j ->" <<i <<',' <<j <<endl;

                    torch::Tensor temp = forward_message.forward({
                        distribution_tensor.index({i,Ellipsis}),
                        distribution_tensor.index({j,Ellipsis}),
                        adjacency_matrix.index({i,j}),
                        at::cat({C_tensor.index({i}).reshape({1}),C_tensor.index({j}).reshape({1})}),
                        patience_tensor
                    });

                    a_V.index({i,Ellipsis}) += temp;

                    temp = backward_message.forward({
                        distribution_tensor.index({i,Ellipsis}),
                        distribution_tensor.index({j,Ellipsis}),
                        adjacency_matrix.index({i,j}),
                        at::cat({C_tensor.index({i}).reshape({1}),C_tensor.index({j}).reshape({1})}),
                        patience_tensor
                    });

                    a_V.index({j,Ellipsis}) += temp;
                }
                ptr++;
            }
        }
        std::cout<<a_V<<endl;
        torch::Tensor distribution_tensor_distort = torch::zeros(input[0].sizes(),torch::TensorOptions().dtype(torch::kFloat64));
        torch::Tensor patience_tensor_distort = torch::zeros(input[1].sizes(),torch::TensorOptions().dtype(torch::kFloat64));
        torch::Tensor C_tensor_distort = torch::zeros(input[2].sizes(),torch::TensorOptions().dtype(torch::kFloat64));
        torch::Tensor alpha_list = torch::zeros({V},torch::TensorOptions().dtype(torch::kFloat64));
        torch::Tensor node_weights = torch::zeros({V},torch::TensorOptions().dtype(torch::kFloat64));

        for(int i = 0; i < V;i++)
        {
            torch::Tensor h_v = a_V.index({i,Ellipsis});
            distribution_tensor_distort.index({i,Ellipsis}) = torch::tanh(service->forward(h_v)).reshape({p,b});
    
            C_tensor_distort.index({i}) = *((float*)torch::tanh(servers->forward(h_v)).data_ptr());

            alpha_list.index({i}) = *(float*)torch::sigmoid(alpha->forward(h_v)).data_ptr();

            node_weights.index({i}) = *(float*)torch::sigmoid(gated->forward(h_v)).data_ptr();
            
        }
        ptr = (float*)node_weights.data_ptr();

        torch::Tensor graph_tensor = torch::zeros({50},torch::TensorOptions().dtype(torch::kFloat64));
        for(int i=0;i < V;i++ )
        {
            torch::Tensor h_v = a_V.index({i,Ellipsis});
            graph_tensor += *ptr*h_v;
            ptr++;
        }
        
        patience_tensor_distort = torch::tanh(patience->forward(graph_tensor));
        torch::Tensor Q = Q_value->forward(graph_tensor);


        // alpha_list 
        
        //Perform action ?

        // Q_value
        
        return {distribution_tensor_distort,patience_tensor_distort.reshape({p,b}),C_tensor_distort,alpha_list,Q};
    }
    fe forward_message, backward_message;
    torch::nn::Linear service,patience,servers,alpha,Q_value,gated;
};



struct DQN : torch::nn::Module
{
    DQN();
    DQN(c10::IntArrayRef node_embedding_size, int64_t edge_embedding_size) 
        : Graph(*register_module<GNN>("GNN", std::make_shared<GNN>(node_embedding_size,edge_embedding_size)))
    {}

    std::vector<torch::Tensor> forward(std::vector<torch::Tensor> state)
    {
        state[0].sizes();
        // State - service tensor {|V|,p,b}, patience_tensor {p,b} ,C_tensor - {|V|}, Adjacency matrix {|V|,|V|}
        return Graph.forward(state);
        // return Graph.forward(env.network,env.node_list);
    }
    GNN Graph;
};



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
    C_type C;
    int mxN;
    int64_t num_priority;

    node();
    node(int mxN_para,int64_t num_priority_para, C_type C_para,std::vector<distribution> service_para)
    {
        this->mxN = mxN_para;
        this-> num_priority = num_priority_para;
        this->C = C_para;
        this->service = service_para;
    }
    station convert_to_station()
    {
        event_type_list dept_para,patience_para;
        for(auto&x: service)
        {
            dept_para.push_back(x.sampler());
        }
        return station(mxN,C,dept_para,num_priority,0);
    }

    std::vector<torch::Tensor> convert_to_tensor()
    {
        std::vector<torch::Tensor> services,patiences;
        for(distribution& temp: this->service)
        {
            services.push_back(temp.convert_to_tensor()); // size b
        }
        int b = this->service[0].b-1;
        int p = this->num_priority;
        // std::cout<< at::cat(services).sizes();
        torch::Tensor service_tensor = at::cat(services).reshape({p,b});
        // tensor.push_back(torch::tensor(this->mxN).reshape({1}));
        // auto options = torch::TensorOptions().dtype(torch::kFloat64);
        int c = this->C(0);
        std::vector<torch::Tensor> node_vector = 
        {
            service_tensor,
            torch::tensor(c,torch::TensorOptions().dtype(torch::kInt32)).reshape({1})
            // torch::from_blob(&c,{1},torch::TensorOptions().dtype(torch::kInt32))
        };
        return node_vector; // node vector
    }

    void load_vector(std::vector<torch::Tensor> node_vector) // dim = {p,b} + {1}
    {
        torch::Tensor services_tensor = node_vector[0];
        torch::Tensor C_tensor = node_vector[1];

        int b = service[0].b;

        assert( (service[0].b - 1) == services_tensor.size(1) );
        assert( (num_priority) == services_tensor.size(0) );
        
        std::vector<distribution> service_temp;

        for(int i =0; i<this->num_priority;i++)
        {
            auto r_ptr = services_tensor.index({i,Ellipsis}).data_ptr<float>();
            std::vector<float> quantile{r_ptr,r_ptr+b-1};
            service_temp.push_back(distribution(b,quantile));
        }
        this->service = service_temp;

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
    std::vector<distribution> patience;
    bool virtual_;
    
    environment();

    environment(std::vector<distribution> arrival_par,int64_t num_priority_para, node init_node, std::vector<distribution> patience_para, bool virtual_ = false,int b_para = 1000, int64_t num_events_para=10000): 
    network( { {} } ),node_list( {init_node} )
    {
        this->arrival = arrival_par;
        this-> num_priority = num_priority_para;
        this-> b = b_para;
        this-> num_events = num_events_para;
        this-> patience = patience_para;
        this-> virtual_ = virtual_;
    }
    environment(std::vector<distribution> arrival_para, std::vector< std::vector< std::pair<int,float> > > network_para, std::vector<node> node_list_para,std::vector<distribution> patience_para, int64_t num_priority_para,bool virtual_ = false,int b_para = 1000  ,int64_t num_events_para=10000)
    {
        this->arrival = arrival_para;
        this->network = network_para;
        this-> node_list = node_list_para;
        this-> num_priority = num_priority_para;
        this-> b = b_para;
        this-> num_events = num_events_para;
        this-> patience = patience_para;
        this-> virtual_ = virtual_;
    }
    
    void simulate()
    {
        // Convert back to the simulator code
        std::vector<station> station_list;
        for(auto &x:node_list)
        { 
            station_list.push_back(x.convert_to_station());
        }
        event_type_list patience_times;
        for(auto&x: patience)
        {
            patience_times.push_back(x.sampler());

        }
        graphv2 simulator(0,0,network,station_list,patience_times);

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
                if(virtual_)
                    simulator.add_customer_to_graph_vir(t,{priority,arriving_customer},true,arrival_processes,ta);
                else
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
            distribution_vector.push_back(node_vector[0].reshape({1,node_vector[0].size(0),node_vector[0].size(1)})); // {1,p,b}
            C_vector.push_back( node_vector[1] );
        }
        std::vector<torch::Tensor> patience_vector;
        for(auto patience_dist:patience)
        {
            patience_vector.push_back( patience_dist.convert_to_tensor() ); 
        }
        torch::Tensor distribution_vector_tensor =  at::cat(distribution_vector); // {|V|,p,b}
        torch::Tensor C_vector_tensor =  at::cat(C_vector); // |V|
        torch::Tensor patience_tensor = at::cat(patience_vector); // {p,b}
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

        return {distribution_vector_tensor,patience_tensor,C_vector_tensor,adjacency_matrix_tensor};
    }
};

class trainer
{
    environment env;
    DQN policy_network, target_network;
    // Optimizer
    float gamma = 0.99;
    double epsilon_start = 1.0;
    double epsilon_final = 0.01;
    int64_t epsilon_decay = 30000;
    int64_t priority_levels = 1;
    int b;
    int num_events = 10000;
public:
    trainer();

    trainer(environment env_init,int64_t priority_levels_para = 1,int b_para = 1000)
        : policy_network({priority_levels_para,b_para-1},1),
          target_network({priority_levels_para,b_para-1},1),
          b(b_para),
          priority_levels(priority_levels_para),
          env(env_init)
    {}
    
    void init_env(float service_rate,float patience_rate,float arrival_rate,int mxN)
    {
        std::vector<distribution> arrival = {};
        arrival.assign( this->priority_levels, distribution( [arrival_rate](float t)-> float { return exponentialrv(arrival_rate); }, this->b )  );
        this->env = environment(arrival,this->priority_levels,

            node(mxN, this->priority_levels, [mxN](float t)-> int{ return mxN;},
                    std::vector<distribution>(this->priority_levels,distribution( [service_rate](float t)-> float { return exponentialrv(service_rate); }, this->b ) )
                    ),

            std::vector<distribution>(this->priority_levels,distribution( [patience_rate](float t)-> float { return exponentialrv(patience_rate); }, this->b )),
            true,
            this->b,
            this->num_events
        );
    }

    torch::Tensor compute_td_loss();

    void train(int64_t num_epochs)
    {
        // this->init_env(1,1,1,10); // is Environment empty????
        this->env.get_size();

        this->policy_network.forward(this->env.state_to_tensor());

        // start training

        for(int i =1; i<=num_epochs;i++)
        {
            double epsilon; //= decay
            auto r = random;
            // convert state to torch tensor **
            // select action
            // policy_network.forward()  return a new node, Q_values, alpha_list
            if( r<= epsilon )
            {
                env.perform_action();
            }
            else
            {
                // select action with q_max
                policy_network.forward(this->env.state_to_tensor()); // -> Returns Q_Value
            }
            //this->env.perform_action(action,alpha_list,new_node);
            //  float reward = this->env.reward(//file read);

            bool done;
            if(done)
            {
                this->init_env(1,1,1,100);
            }
            torch::Tensor loss = this->compute_td_loss();
            
        }

    }
};



int main() 
{
  // trainer.cpp Environment/environment.cpp Environment/node.cpp  ../../nixonz/simulation/components/station.cpp ../../nixonz/simulation/components/tandem.cpp ../../nixonz/simulation/components/queue_graphv2.cpp

  // torch::Tensor tensor = torch::rand({2});
  // torch::Tensor b = torch::rand({2});
  // std::vector<float> list = {1,2};
  // int test = 2;
  // std::cout<<tensor << std::endl;
  // std::vector<torch::Tensor> tensor_list_test = { tensor, b, torch::tensor(test).reshape({1}) };
  // std::cout << at::cat({torch::tensor(list),torch::tensor(test).reshape({1})}) << std::endl;
  // std::cout << temp(tensor) << '\n';
  // std::cout << temp(at::cat({tensor,b})) << '\n';
  // tensor_list_test.push_back( temp(torch::tensor(list)) );
  // std::cout << at::cat(tensor_list_test) << '\n';
  // torch::Tensor result = at::cat(tensor_list_test);
  // auto r_ptr = result.data_ptr<float>();
  // std::vector<float> temp{r_ptr, r_ptr + result.size(0)};

  // for(auto& x:temp)
  // {
  //   std::cout<<x <<',';
  // }
  // std::cout<<'\n';

  // b_max = 100001

  

  
//   distribution exponential(  [](float t)-> float { return exponentialrv(2); }, 100001  );


    int b = 1025;


//   distribution lognormal( read_csv("../lognormal.csv",3), b );
// //   torch::Tensor tensor = exponential.convert_to_tensor();
// //   torch::Tensor tensor = lognormal.convert_to_tensor();



//     // std::vector<distribution> arrival = { 
//     //     distribution( [](float t)-> float { return exponentialrv(0.14); }, b ), 
//     //     distribution( [](float t)-> float { return exponentialrv(0.5); }, b ) 
//     // };

    node station = node(100, 2, [](float t)-> int{ return 100;},
            { distribution( [](float t)-> float { return exponentialrv(0.14); }, b ), distribution( [](float t)-> float { return exponentialrv(0.16); }, b ) }
          );

//     std::vector<torch::Tensor> tensor = station.convert_to_tensor();
// 	// std::cout<<temp[0];
//     // std::cout<<tensor <<endl;

    

//     std::cout<< tensor[0].sizes() <<endl ;  
//     std::cout<<"[0,0,:]: " << tensor[0].index({0,0,Slice()}).mean() <<endl ;
//     std::cout<<"[0,..]: " << tensor[0].index({0,Ellipsis}).sizes() <<endl ;
//     std::cout<<"C: " << tensor[1] <<endl ;
//     std::cout<< tensor[0].std() <<endl ;

//     std::cout<< "===========================" <<endl; 

//     std::cout<< tensor[0].sizes() <<endl ;  
//     tensor[0].index({0,0,Slice()}) = 1.0; // Changes data without loading.
//     std::cout<<"[0,0,:]: " << tensor[0].index({0,0,Slice()}).mean() <<endl ;
//     std::cout<<"[0,..]: " << tensor[0].index({0,Ellipsis}).sizes() <<endl ;
//     std::cout<<"C: " << tensor[1] <<endl ;
//     std::cout<< tensor[0].std() <<endl ;

//     node WORKING = node(100, 2, [](float t)-> int{ return 100;},
//                 { distribution( [](float t)-> float { return exponentialrv(0.5); }, b ), distribution( [](float t)-> float { return exponentialrv(0.5); }, b ) },
//                 { distribution( [](float t)-> float { return exponentialrv(0.5); }, b ), distribution( [](float t)-> float { return exponentialrv(0.5); }, b ) }
//             );

//     WORKING.load_vector(tensor);
//     tensor = WORKING.convert_to_tensor();

//     std::cout<< "===========================" <<endl;
//     std::cout<< tensor[0].sizes() <<endl ;  
//     std::cout<<"[0,0,:]: " << tensor[0].index({0,0,Slice()}).mean() <<endl ;
//     std::cout<<"[0,..]: " << tensor[0].index({0,Ellipsis}).sizes() <<endl ;
//     std::cout<<"C: " << tensor[1] <<endl ;
//     std::cout<< tensor[0].std() <<endl ;

// //  testing Environment
    std::vector<distribution> arrival = { 
            distribution( [](float t)-> float { return exponentialrv(0.14); }, b ), 
            distribution( [](float t)-> float { return exponentialrv(0.5); }, b ) 
        };
    std::vector<distribution> patience = { 
        distribution( [](float t)-> float { return exponentialrv(0.5); }, b ), 
        distribution( [](float t)-> float { return exponentialrv(0.5); }, b ) 
        };
    environment env(arrival,2,station,patience,true,b);

    std::vector<torch::Tensor> tensor = env.state_to_tensor();

    std::cout<< "==============================" <<endl;
    std::cout<< tensor[0].sizes() <<endl ;  // services
    std::cout<< tensor[1].sizes() <<endl; // patience
    std::cout<< tensor[2].sizes() <<endl ; // C_servers
    std::cout<< tensor[3].sizes() <<endl ; // Adjacency matrix
    std::cout<<"[0,0,:]: " << tensor[0].index({0,0,Slice()}).mean() <<endl ;
    std::cout<<"[0,..]: " << tensor[0].index({0,Ellipsis}).sizes() <<endl ;
    std::cout<<"Patience: " << tensor[1].index({0,Ellipsis}).mean() <<endl ;
    std::cout<<"C :" <<tensor[2] <<endl;
    std::cout<< tensor[0].index({0,0,Slice()}).std() <<endl ;

    std::cout<< "Adjacency Matrix" <<endl;
    std::cout<< tensor[3] <<endl;
    std::cout<< "Service tensor" <<endl;
    // std::cout<< tensor[0] <<endl;

    env.perform_action(1,{1},
        node(100, 2, [](float t)-> int{ return 100;},
            { distribution( [](float t)-> float { return exponentialrv(0.14); }, b ), distribution( [](float t)-> float { return exponentialrv(0.16); }, b ) }
          )
    );
    env.perform_action(3,{0.9,1},
        node(1, 2, [](float t)-> int{ return 1;},
            { distribution( [](float t)-> float { return exponentialrv(0.5); }, b ), distribution( [](float t)-> float { return exponentialrv(0.16); }, b ) }
          )
    );

    tensor = env.state_to_tensor();

    std::cout<< "==============================" <<endl;
    std::cout<< tensor[0].sizes() <<endl ;  // services
    std::cout<< tensor[1].sizes() <<endl; // patience
    std::cout<< tensor[2].sizes() <<endl ; // C_servers
    std::cout<< tensor[3].sizes() <<endl ; // Adjacency matrix
    std::cout<<"[0,0,:]: " << tensor[0].index({0,0,Slice()}).mean() <<endl ;
    std::cout<<"[0,..]: " << tensor[0].index({0,Ellipsis}).sizes() <<endl ;
    std::cout<<"Patience: " << tensor[1].index({0,Ellipsis}).mean() <<endl ;
    std::cout<<"C :" <<tensor[2] <<endl;
    std::cout<< tensor[0].index({0,0,Slice()}).std() <<endl ;

    std::cout<< "Adjacency Matrix" <<endl;
    std::cout<< tensor[3] <<endl;
    std::cout<< "Service tensor" <<endl;
    // std::cout<< tensor[0] <<endl;

    torch::Tensor adjacency_matrix = tensor[2];

    // float* ptr = (float*)adjacency_matrix.data_ptr();

    // for( int i = 0 ; i < adjacency_matrix.size(0) ; i++ )
    //     {
    //         for( int j = 0 ; j < adjacency_matrix.size(1) ; j++ )
    //         {
    //             if(*ptr > 0);
    //             // {
    //             //     // std::cout<<*ptr <<',';
    //             // }
    //             std::cout<<*ptr <<',';
    //             ptr++;
    //         }
    //         std::cout<<endl;
    //     }

std::cout<< "==============================" <<endl;
    trainer model(env,2,b);
    model.train(1);
    // CODE
}
// distribution: - <b> dim = {b,1}
// Services: < <b>,<b>,<b>,......p times > dim = {p,b}
// Node: < Service,Patience > <C> {2*p,b} + {1}
// Graph: Nodes: {|V|,p,b,2} + {|V|}, network: {|V|,|V|}
