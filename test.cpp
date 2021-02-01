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

    std::vector< std::vector< std::pair<int,float> > > network = { { {1,0.7},{2,0.3} }, {  }, {  } }; 

    std::vector<node> node_list = {
        node(100, 2, [](float t)-> int{ return 100;},
            { distribution( [](float t)-> float { return exponentialrv(0.14); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.16); }, 1000 ) },
            { distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ) }
          ), 
        node(100, 2, [](float t)-> int{ return 100;},
            { distribution( [](float t)-> float { return exponentialrv(0.14); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.16); }, 1000 ) },
            { distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ) }
          ), 
        node(100, 2, [](float t)-> int{ return 100;},
            { distribution( [](float t)-> float { return exponentialrv(0.14); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.16); }, 1000 ) },
            { distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ), distribution( [](float t)-> float { return exponentialrv(0.5); }, 1000 ) }
          ),   
    };

    environment env(arrival,network,node_list,2);
    env.simulate();
    return 0;
}

