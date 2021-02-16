#include "trainer.h"

float exponentialrv(float lambda)
{
    float U = random;
    return -log(U)/lambda;
}

void trainer::init_env()
{
    std::vector<distribution> arrival = {};
    arrival.assign( this->priority_levels, distribution( [](float t)-> float { return exponentialrv(0.14); }, 1000 )  );
    this->env = environment(arrival,this->priority_levels,
    node(100, 2, [](float t)-> int{ return 100;},
            { distribution( [](float t)-> float { return exponentialrv(0.14); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.16); }, 1000 ) },
            { distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ) }
          )
    );
}

void trainer::train(int64_t num_epochs)
{
    this->init_env();
    this->env.get_size();
    // [1,2^n-1]

    for(int i =1; i<=num_epochs;i++)
    {
        double epsilon; //= decay
        auto r = random;
        // convert state to torch tensor **
        // select action
        if( r<= epsilon )
        {
            // select action at random
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