#include "node.h"
#include "../../simulation/components/station.h"

station node::convert_to_station()
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

torch::Tensor node::convert_to_tensor()
{
    std::vector<torch::Tensor> tensor;
    for(distribution& temp: this->service)
    {
        tensor.push_back(temp.convert_to_tensor()); // size b
    }
    for(distribution &temp: this->patience)
    {
        tensor.push_back(temp.convert_to_tensor());
    }
    tensor.push_back(torch::tensor(&this->mxN).reshape({1});
    return at::cat(tensor);
}

void node::load_vector(torch::Tensor node_vector)
{
    auto r_ptr = node_vector.data_ptr<float>();
    int b = service[0].b;

    std::vector<distribution> service_temp;
    std::vector<distribution> patience_temp;

    for(int i =0; i<this->num_priority;i++)
    {
        std::vector<float> quantiles{r_ptr+b*i,r_ptr+b*(i+1)};
        service_temp.push_back(quantile);
    }
    this->service = service_temp;

    for(int i =0; i<this->num_priority;i++)
    {
        std::vector<float> quantile{r_ptr+b*this->num_priority+b*i,r_ptr+b*this->num_priority+b*(i+1)};
        patience_temp.push_back(quantile);
    }
    this->patience = patience_temp;
    std::vector<float> C_new{r_ptr+node_vector.size(0)-1,r_ptr+node_vector.size(0)};
    this->mxN = C_new[0];
    this->C = [C_new](float t)-> int{ return C_new;};
}