#include <torch/torch.h>
#include <iostream>
#include "../simulation/components/station.h"
#include "../simulation/components/queue_graphv2.h"
#include "includes.h"


using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;

template <class T>
std::vector<T> convert_tensor(torch::Tensor input)
{
    std::vector<T> output;
    T* ptr = (T*)input.data_ptr();
    input = input.squeeze();
    int N = input.numel();
    for(int i=0;i<N;i++)
    {
        output.push_back(*ptr);
        ptr++;
    }
    return output;
}

float exponentialrv(float lambda)
{
    float U = random;
    return -log(U)/lambda;
}

int rand(int a, int b) {
    return a + rand() % (b - a + 1);
}

struct tree_output : torch::nn::Module
{
public:
    tree_output();
    tree_output(int64_t input_size,int64_t output_size)
    :
     fl(torch::nn::LinearOptions(50,50)),
     fr(torch::nn::LinearOptions(50,50)),
     Q_value(torch::nn::LinearOptions(50,1))
    {
        fl->to(torch::kFloat64);
        fr->to(torch::kFloat64);
        Q_value->to(torch::kFloat64);
    }
    torch::Tensor forward(torch::Tensor input,int64_t depth)
    {
        std::vector<torch::Tensor> leaf_nodes( (1<<(depth)),torch::zeros(50,torch::TensorOptions().dtype(torch::kFloat64)) );
        fl->to(torch::kFloat64);
        fr->to(torch::kFloat64);
        Q_value->to(torch::kFloat64);
        int i = 0;
        split(input,0,depth,leaf_nodes,i);
        std::vector<torch::Tensor> Q_list;
        for(auto& leaf_vector: leaf_nodes)
        {
            Q_list.push_back(torch::relu(Q_value->forward(leaf_vector)));
        }
        return at::cat(Q_list);
    }

    void split(torch::Tensor parent,int cur_depth,int max_depth,std::vector<torch::Tensor>& leaf_nodes,int& i)
    {
        if( cur_depth == max_depth )
        {
            // std::cout<<torch::zeros({1}) <<endl;
            // std::cout<<"I am at leaf"<<parent <<i <<endl;
            leaf_nodes[i] = parent;
            // std::cout<<"I am at leaf"<<parent <<endl;
            i++;
            return;
        }
        // std::cout<<torch::zeros({1}) <<endl;
        // std::cout<<"Hello Iam in" <<endl;
        torch::Tensor left_child = torch::relu(fl->forward(parent));
        torch::Tensor right_child = torch::relu(fr->forward(parent));
        split(left_child,cur_depth+1,max_depth,leaf_nodes,i);
        split(right_child,cur_depth+1,max_depth,leaf_nodes,i);
    }
    torch::nn::Linear fl,fr;
    torch::nn::Linear Q_value;
};


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
        linear1(torch::nn::LinearOptions(50*3*5+3 ,50))
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
        // std::cout<<"DIM OF LAST CONV" <<x.sizes();
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
          Q_list(*register_module<tree_output>("Q_Values",std::make_shared<tree_output>(50,50))),
          new_service(torch::nn::LinearOptions(50,node_embedding_size[0]*node_embedding_size[1])),
          new_C(torch::nn::LinearOptions(50,1))

    {
        forward_message.to(torch::kFloat64);
        backward_message.to(torch::kFloat64);
        service->to(torch::kFloat64);
        servers->to(torch::kFloat64);
        alpha->to(torch::kFloat64);
        gated->to(torch::kFloat64);
        patience->to(torch::kFloat64);
        Q_list.to(torch::kFloat64);
        new_service->to(torch::kFloat64);
        new_C->to(torch::kFloat64);
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
                    // std::cout<<"i,j ->" <<i <<',' <<j <<endl;

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
        
        // std::cout<<a_V<<endl;
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
        
        patience_tensor_distort = torch::tanh(patience->forward(graph_tensor)).reshape({p,b});
        torch::Tensor Q = Q_list.forward(graph_tensor,V); //{2^n}
        torch::Tensor C_new = new_C->forward(graph_tensor);
        torch::Tensor service_new = torch::relu(new_service->forward(graph_tensor)).reshape({p,b});
        // std::cout<<"Q-list" <<Q <<endl;
        return 
        {
            distribution_tensor_distort,
            patience_tensor_distort,
            C_tensor_distort,
            alpha_list,
            Q,
            C_new,
            service_new
        };
    }
    fe forward_message, backward_message;
    torch::nn::Linear service,patience,servers,alpha,gated,new_service,new_C;
    tree_output Q_list;
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
    }
    GNN Graph;
};



class distribution
{
public:
    int b;
    std::vector<float> quantiles;

    distribution();
    distribution(std::vector<float> quantiles_para)
    {
        this->quantiles = quantiles_para;
        this->b = quantiles_para.size();
    }
    distribution(torch::Tensor quantiles_para) // {b}
    {
        // int p = quantiles_para.size(0);
        int b = quantiles_para.numel(); // check this?
        float* ptr = (float*)quantiles_para.data_ptr();
        std::vector<float> quantiles;
        for(int i =0; i<b ; i++)
        {
            quantiles.push_back(*ptr);
            ptr++;
        }
        sort(quantiles.begin(),quantiles.end());
        this->b = b;
        this->quantiles = quantiles;
        this->b = quantiles.size();
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

        for (size_t k = 0; k < b; k++)
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
        this->b = quantiles.size();
    }
    
    distribution(std::vector<float> data,int b_para)
    {
        
        this->b = b_para;

        int n = data.size();
        std::vector< std::pair<float,float> > samples;
        for (size_t i = 0; i < n; i++)
        {
            if(data[i]>=0)
                samples.push_back( { data[i], 0} );
        }
        n = samples.size();
        std::sort(samples.begin(),samples.end());

        for (size_t i = 0; i < n; i++)
        {
            // std::cout<<samples[i].first <<',';
            samples[i].second = (i+0.5)/n;
        }
        //    0...100 <-> b groups kth b-quantile 

        for (size_t k = 0; k < b; k++)
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
        this->b = quantiles.size();
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

    // A better continous sampler can be constructed
    event_type sampler()
    {
        int b_para = this->b;
        std::vector<float> quantiles_para = this->quantiles;
        return [b_para,quantiles_para](float t) -> float { float U = random; return quantiles_para[int(b_para*U)]; };
    }

    static float area_between_dist(distribution A,distribution B)
    {
        assert(A.quantiles.size()==B.quantiles.size());
        float res = 0;
        for (size_t i = 0; i < A.quantiles.size(); i++)
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
    float calc_epsilon()
    {
        float epsilon = INF;
        for(int i=1;i<quantiles.size();i++)
        {
            if( epsilon > quantiles[i]-quantiles[i-1])
                epsilon = quantiles[i] - quantiles[i-1];

        }
        return epsilon;
    }
    void distort(torch::Tensor distortion) // b
    {
        float epsilon = calc_epsilon();
        std::cout<<"Epsilon:" <<epsilon <<endl;
        std::vector<float> distortion_list = convert_tensor<float>(distortion); // -1 to 1
        for(int i=0;i<quantiles.size();i++)
        {
            quantiles[i] += epsilon*distortion_list[i];
        }
        std::sort(quantiles.begin(),quantiles.end());
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
        int b = this->service[0].b;
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
            // service_temp.push_back(distribution(b,quantile));
            service_temp.push_back(distribution(quantile));
        }
        this->service = service_temp;

        auto r_ptr = C_tensor.data_ptr<int>();
        int data = *(r_ptr+1);
        if( data <= this->mxN )
            this->C = [data](float t)-> int{ return data;};
        else
            this->C = [this](float t)-> int{ return this->mxN;};
    }
    void distort(std::vector<torch::Tensor> distortions)
    {
        torch::Tensor distribution_tensor_distort = distortions[0]; // 1,p,b 
        torch::Tensor C_tensor_distort = distortions[1]; // 1
        int p = this->num_priority;

        for(int i =0;i<p;i++)
        {
            this->service[i].distort( distribution_tensor_distort.index({i,Ellipsis}) );
        }
        float* ptr = (float*)C_tensor_distort.data_ptr();
        float C_dist = *ptr;
        int c = this->C(0);
        int c_ = (c - C_dist*c +1);
        this->C = [c_](float t)-> int{ return c_;};
        this->mxN = c_;
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
        simulator.initialize_CSV("./temp");

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
            if(discrete_events%(num_events/100)==0)
            {
                // std::cout<<"Writing to CSV\n";
                simulator.dump_counter_variable_memory("./temp");
            }
            discrete_events++;
            // std::cout<<discrete_events<<endl;
        }
        this->simulated_data = read_csv("./temp.csv",8);
        // std::cout<<"Writing to CSV\n";
        // simulator.write_to_csv("./output/graph");
    }

    float reward(std::vector<float> input,float sigma = 1e-4)
    {
        // Chi-square fitness test possible
        // Area under curve
        this->simulate();
        distribution input_(input, this->b);
        distribution simulated(this->simulated_data,this->b);
        float tot_var = distribution::area_between_dist( input_, simulated);

        return 1.0/tot_var;

        //Dirac Delta Approximated as Normal
        // return exp( -1*( pow( tot_var,2 )/(2*sigma) ) )/(sigma*sqrt(2*M_PIf64));
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

    void distort(std::vector<torch::Tensor> distortions)
    {
        torch::Tensor distribution_tensor_distort = distortions[0]; // ||V|,p,b 
        torch::Tensor patience_tensor_distort = distortions[1]; // {p,b}
        torch::Tensor C_tensor_distort = distortions[2]; // |V|
        int p = this->num_priority;



        for(int i =0;i<p;i++)
        {
            this->patience[i].distort( patience_tensor_distort.index({i,Ellipsis}) );
        }

        for(int i =0;i<this->get_size();i++)
        {
            this->node_list[i].distort({
                distribution_tensor_distort.index({i,Ellipsis}),
                C_tensor_distort.index({i,Ellipsis})
            });
        }

    }
};

class trainer
{
    environment env;
    environment env_init;
    DQN policy_network, target_network;
    // Optimizer
    torch::optim::Adam dqn_optimizer;
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
        : policy_network({priority_levels_para,b_para},1),
          target_network({priority_levels_para,b_para},1),
          b(b_para),
          priority_levels(priority_levels_para),
          env(env_init),
          env_init(env_init),
          dqn_optimizer(policy_network.parameters(true), 0.001)
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
        this->env_init = this->env;
    }
    void init_env(std::vector<distribution> arrival,std::vector<distribution> patience,node init_node)
    {
        this->env = environment(arrival,this->priority_levels,
            init_node,
            patience,
            true,
            this->b,
            this->num_events
        );
        this->env_init = this->env;
    }

    torch::Tensor compute_td_loss(std::vector<torch::Tensor> state, std::vector<torch::Tensor> new_state, torch::Tensor reward,int action,bool done)
    {
        torch::Tensor q_values = policy_network.forward(state)[4];
        torch::Tensor next_target_q_values = target_network.forward(new_state)[4];
        torch::Tensor next_q_values = policy_network.forward(new_state)[4];

        torch::Tensor q_value = q_values.index({action});
        // torch::Tensor maximum = std::get<1>(next_q_values.max(1));
        std::vector<float> Q_list = convert_tensor<float>(next_q_values);
        int max = max_element(Q_list.begin(),Q_list.end()) - Q_list.begin(); 
        torch::Tensor next_q_value = next_target_q_values.index({max});
        torch::Tensor expected_q_value = reward + gamma*next_q_value*(1-int(done));
        torch::Tensor loss = torch::mse_loss(q_value, expected_q_value);

        dqn_optimizer.zero_grad();
        loss.backward();
        dqn_optimizer.step();

        return loss;
    }

    void loadstatedict(torch::nn::Module& model,torch::nn::Module& target_model) 
    {
        torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
        auto new_params = target_model.named_parameters(); // implement this
        auto params = model.named_parameters(true /*recurse*/);
        auto buffers = model.named_buffers(true /*recurse*/);
        for (auto& val : new_params) 
        {
            auto name = val.key();
            auto* t = params.find(name);
            if (t != nullptr) 
            {
                t->copy_(val.value());
            } 
            else 
            {
                t = buffers.find(name);
                if (t != nullptr) 
                {
                    t->copy_(val.value());
                }
            }
        }
        torch::autograd::GradMode::set_enabled(true);
    }

    void train(int64_t num_epochs,int max_size)
    {
        // this->init_env(1,1,1,10); // is Environment empty????

        // start training

        float episode_reward = 0.0;
        std::vector<float> all_rewards;
        std::vector<torch::Tensor> losses;

        for(int i =1; i<=num_epochs;i++)
        {
            std::cout<<"++++++++++++++++++++++++++++++++==== EPOCH:" <<i <<"=======++++++++++++++++++++"<<endl;
            int V = this->env.get_size();
            std::vector<torch::Tensor> state_tensor = this->env.state_to_tensor();

            std::vector<torch::Tensor> output = this->policy_network.forward(state_tensor);

            torch::Tensor distribution_tensor_distort = output[0];
            torch::Tensor patience_tensor_distort = output[1];
            torch::Tensor C_tensor_distort = output[2];

            torch::Tensor alpha_list = output[3];

            torch::Tensor Q = output[4];

            torch::Tensor C_new = output[5];
            torch::Tensor service_new = output[6];

            std::vector<distribution> new_services;
            float* ptr = (float*)C_new.data_ptr();
            int c_new = ( (int)(*ptr) > 0) ? (int)(*ptr) : 1;
            for(int i =0;i<this->priority_levels;i++)
            {
                new_services.push_back(distribution(service_new.index({i,Ellipsis})));
            }
            // std::cout<<"CNEW:" <<c_new <<endl;
            node new_node(
                c_new,
                this->priority_levels,
                [c_new](float t)-> int{ return c_new;},
                new_services
            );

            double epsilon = epsilon_final + (epsilon_start - epsilon_final) * exp(-1. * i / epsilon_decay); //= decay
            auto r = random;
            int action;
            if( r<= epsilon )
            {
                action = rand() % rand(1,(1<<V) );
                std::cout<<"Random Action was taken.\n";
            }
            else
            {
                // select action with q_max
                std::vector<float> Q_list = convert_tensor<float>(Q);
                action = max_element(Q_list.begin(),Q_list.end()) - Q_list.begin(); 
            }

            // Distort states too
            
            env.distort(
                {
                    distribution_tensor_distort,
                    patience_tensor_distort,
                    C_tensor_distort
                }
            );

            std::cout<<"Action:" <<action;
            env.perform_action(action,convert_tensor<float>(alpha_list),new_node);
            env.pretty_print_network();

            std::vector<torch::Tensor> new_state = env.state_to_tensor();
            
            float reward = this->env.reward(read_csv("../MM1.csv",8));
            all_rewards.push_back(reward);
            episode_reward+= reward;
            bool done = env.get_size() == max_size ;

            torch::Tensor reward_tensor = torch::tensor(reward);
            // torch::Tensor done_tensor = torch::tensor(done);
            // done_tensor = done_tensor.to(torch::kFloat32);
            // torch::Tensor action_tensor_new = torch::tensor(action);
            std::cout<<"Reward: " <<reward_tensor <<endl;

            if(done)
            {
                // this->init_env(env.arrival,env.patience,new_node);
                this->env = this->env_init;
                episode_reward = 0.0;
                std::cout<<"++++++++++++++++++++++++++++++++++=RESET+++++++++++++++++++++++++++++++++++++++\n";
            }

            torch::Tensor loss = compute_td_loss(state_tensor,new_state,reward_tensor,action,done);
            losses.push_back(loss);
            std::cout<<"Loss" <<loss <<endl;

            if (i%10==0)
            {
                std::cout<<episode_reward<<endl;
                loadstatedict(policy_network, target_network);
            }
        }
        std::cout<<"Reward:\n" <<all_rewards<<endl;
        std::cout<<"Loss:\n" <<losses<<endl;

    }
};



int main() 
{
    srand(time(0));

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

    int b = 32;
    // distribution exponential(  [](float t)-> float { return exponentialrv(2); }, b );
    // exponential.print_quantiles();
    // std::cout<<endl;
    // std::cout<<exponential.convert_to_tensor() <<endl;
    // std::cout<<torch::zeros({1}) <<endl;

    // distribution lognormal( read_csv("../lognormal.csv",3), b );
    // lognormal.print_quantiles();
    // std::cout<<endl;
    // std::cout<<lognormal.convert_to_tensor() <<endl;
    // std::cout<<torch::zeros({1}) <<endl;
//   torch::Tensor tensor = exponential.convert_to_tensor();
//   torch::Tensor tensor = lognormal.convert_to_tensor();



    // std::vector<distribution> arrival = { 
    //     distribution( [](float t)-> float { return exponentialrv(0.14); }, b ), 
    //     distribution( [](float t)-> float { return exponentialrv(0.5); }, b ) 
    // };

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
            distribution( [](float t)-> float { return exponentialrv(0.09); }, b ), 
            distribution( [](float t)-> float { return exponentialrv(0.06); }, b ) 
        };
    std::vector<distribution> patience = { 
        distribution( [](float t)-> float { return exponentialrv(0.5); }, b ), 
        distribution( [](float t)-> float { return exponentialrv(0.5); }, b ) 
        };
    environment env(arrival,2,station,patience,true,b,10000);

    std::vector<torch::Tensor> tensor = env.state_to_tensor();

    // std::cout<< "==============================" <<endl;
    // std::cout<< tensor[0].sizes() <<endl ;  // services
    // std::cout<< tensor[1].sizes() <<endl; // patience
    // std::cout<< tensor[2].sizes() <<endl ; // C_servers
    // std::cout<< tensor[3].sizes() <<endl ; // Adjacency matrix
    // std::cout<<"[0,0,:]: " << tensor[0].index({0,0,Slice()}).mean() <<endl ;
    // std::cout<<"[0,..]: " << tensor[0].index({0,Ellipsis}).sizes() <<endl ;
    // std::cout<<"Patience: " << tensor[1].index({0,Ellipsis}).mean() <<endl ;
    // std::cout<<"C :" <<tensor[2] <<endl;
    // std::cout<< tensor[0].index({0,0,Slice()}).std() <<endl ;

    // std::cout<< "Adjacency Matrix" <<endl;
    // std::cout<< tensor[3] <<endl;
    // std::cout<< "Service tensor" <<endl;
    // // std::cout<< tensor[0] <<endl;
    // int reward = env.reward(read_csv("../graph.csv",8));
    // std::cout<<"Reward:" <<reward <<endl;

    // env.perform_action(1,{1},
    //     node(100, 2, [](float t)-> int{ return 100;},
    //         { distribution( [](float t)-> float { return exponentialrv(0.14); }, b ), distribution( [](float t)-> float { return exponentialrv(0.16); }, b ) }
    //       )
    // );
    // env.perform_action(3,{0.9,1},
    //     node(1, 2, [](float t)-> int{ return 1;},
    //         { distribution( [](float t)-> float { return exponentialrv(0.5); }, b ), distribution( [](float t)-> float { return exponentialrv(0.16); }, b ) }
    //       )
    // );

    tensor = env.state_to_tensor();

    // std::cout<< "==============================" <<endl;
    // std::cout<< tensor[0].sizes() <<endl ;  // services
    // std::cout<< tensor[1].sizes() <<endl; // patience
    // std::cout<< tensor[2].sizes() <<endl ; // C_servers
    // std::cout<< tensor[3].sizes() <<endl ; // Adjacency matrix
    // std::cout<<"[0,0,:]: " << tensor[0].index({0,0,Slice()}).mean() <<endl ;
    // std::cout<<"[0,..]: " << tensor[0].index({0,Ellipsis}).sizes() <<endl ;
    // std::cout<<"Patience: " << tensor[1].index({0,Ellipsis}).mean() <<endl ;
    // std::cout<<"C :" <<tensor[2] <<endl;
    // std::cout<< tensor[0].index({0,0,Slice()}).std() <<endl ;

    // std::cout<< "Adjacency Matrix" <<endl;
    // std::cout<< tensor[3] <<endl;
    // std::cout<< "Service tensor" <<endl;
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
    // env.simulate();
    // torch::manual_seed(432);
    trainer model(env,2,b);
    model.train(10000,4);
    // CODE
}
// distribution: - <b> dim = {b,1}
// Services: < <b>,<b>,<b>,......p times > dim = {p,b}
// Node: < Service,Patience > <C> {2*p,b} + {1}
// Graph: Nodes: {|V|,p,b,2} + {|V|}, network: {|V|,|V|}
