#include "trainer.h"

float exponentialrv(float lambda)
{
    float U = random;
    return -log(U)/lambda;
}

void trainer::init_env(float service_rate,float patience_rate,float arrival_rate,int mxN)
{
    std::vector<distribution> arrival = {};
    arrival.assign( this->priority_levels, distribution( [arrival_rate](float t)-> float { return exponentialrv(arrival_rate); }, this->b )  );
    this->env = environment(arrival,this->priority_levels,
        node(mxN, this->priority_levels, [mxN](float t)-> int{ return mxN;},
                std::vector<distribution>(this->priority_levels,distribution( [service_rate](float t)-> float { return exponentialrv(service_rate); }, this->b ) ),
                std::vector<distribution>(this->priority_levels,distribution( [service_rate](float t)-> float { return exponentialrv(service_rate); }, this->b ) )
                ),
        this->b,
        this->num_events
    );
}

void trainer::train(int64_t num_epochs)
{
    this->init_env(1,1,1,1);
    this->env.get_size();
    // [1,2^n-1]

    policy_network.forward( this->env );

    for(int i =1; i<=num_epochs;i++)
    {
        double epsilon; //= decay
        auto r = random;
        // convert state to torch tensor **
        // select action
        // policy_network.forward()  return a new node, Q_values, alpha_list
        if( r<= epsilon )
        {
            env.perform_action()
        }
        else
        {
            // select action with q_max
            policy_network.forward(state_to_tensor()); // -> Returns Q_Value
        }
        //this->env.perform_action(action,alpha_list,new_node);
        //  float reward = this->env.reward(//file read);

        bool done;
        if(done)
        {
            this->init_env();
        }
        torch::Tensor loss = this->compute_td_loss();
        
    }
}