#include "Environment/environment.h"

float exponentialrv(float lambda)
{
    float U = random;
    return -log(U)/lambda;
}

int main()
{
    std::cout << std::fixed << std::setprecision(precision);
    srand((unsigned)time(NULL));

    std::vector<distribution> arrival = { 
        distribution( [](float t)-> float { return exponentialrv(0.14); }, 1000 ), 
        distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ) 
    };

    // std::vector< std::vector< std::pair<int,float> > > network = { { {1,0.7},{2,0.3} }, {  }, {  } }; 

    // std::vector<node> node_list = {
    //     node(100, 2, [](float t)-> int{ return 100;},
    //         { distribution( [](float t)-> float { return exponentialrv(0.14); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.16); }, 1000 ) },
    //         { distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ) }
    //       ), 
    //     node(100, 2, [](float t)-> int{ return 100;},
    //         { distribution( [](float t)-> float { return exponentialrv(0.14); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.16); }, 1000 ) },
    //         { distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ) }
    //       ), 
    //     node(100, 2, [](float t)-> int{ return 100;},
    //         { distribution( [](float t)-> float { return exponentialrv(0.14); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.16); }, 1000 ) },
    //         { distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ) }
    //       ),   
    // };

    // environment env(arrival,network,node_list,2);
    environment env(arrival,2,
      node(100, 2, [](float t)-> int{ return 100;},
            { distribution( [](float t)-> float { return exponentialrv(0.14); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.16); }, 1000 ) },
            { distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ) }
          )
    );
    // std::cout<<"Hello";
    env.perform_action(1,{1},
        node(100, 2, [](float t)-> int{ return 100;},
            { distribution( [](float t)-> float { return exponentialrv(0.14); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.16); }, 1000 ) },
            { distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ) }
          )
    );
    env.perform_action(1,{0.9},
        node(1, 2, [](float t)-> int{ return 1;},
            { distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.16); }, 1000 ) },
            { distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ) }
          )
    );
    env.pretty_print_network();
    env.simulate();
    std::cout<<"Simulated" <<'\n';
    // std::cout<<"Reward:" <<env.reward( read_csv("../simulation/output/tandem.csv",23) ) <<"\n";
    std::cout<<"Reward:" <<env.reward( read_csv("COPY.csv",8) ) <<"\n";
    // std::vector<float> temp = read_csv("COPY.csv",8);
    // std::cout<<temp[0] <<' ' <<temp[1] <<'\n';
    // std::cout<<temp.size();
  
    return 0;
}
// g++ test.cpp Environment/environment.cpp Environment/node.cpp ../../nixonz/simulation/components/station.cpp  ../../nixonz/simulation/components/tandem.cpp ../../nixonz/simulation/components/queue_graphv2.cpp
