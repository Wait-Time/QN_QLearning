#include <torch/torch.h>
#include <iostream>

torch::Tensor temp(torch::Tensor a)
{
  return torch::relu(a);
}

int main() 
{
  torch::Tensor tensor = torch::rand({2});
  torch::Tensor b = torch::rand({2});
  std::vector<float> list = {1,2};
  int test = 2;
  std::cout<<tensor << std::endl;
  std::vector<torch::Tensor> tensor_list_test = { tensor, b, torch::tensor(test).reshape({1}) };
  std::cout << at::cat({torch::tensor(list),torch::tensor(test).reshape({1})}) << std::endl;
  std::cout << temp(tensor) << '\n';
  std::cout << temp(at::cat({tensor,b})) << '\n';
  tensor_list_test.push_back( temp(torch::tensor(list)) );
  std::cout << at::cat(tensor_list_test) << '\n';
  torch::Tensor result = at::cat(tensor_list_test);
  auto r_ptr = result.data_ptr<float>();
  std::vector<float> temp{r_ptr, r_ptr + result.size(0)};
  for(auto& x:temp)
  {
    std::cout<<x <<',';
  }
  std::cout<<'\n';
}
