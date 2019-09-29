// Adapted from: https://pytorch.org/tutorials/advanced/cpp_export.html#a-minimal-c-application
//               and https://github.com/pytorch/pytorch/pull/16580/files

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/autograd/profiler.h> 

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
  
  torch::set_num_threads(1);
  
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::randn({1, 3, 224, 224}));

  // Execute the model and turn its output into a tensor.
  at::Tensor output;
  std::cout << "Dry run...\n";
  module.forward(inputs).toTensor();

  
  std::ofstream ss("output.json");
  {
    torch::autograd::profiler::RecordProfile guard(ss);
    for (size_t i = 0; i < 10; ++i) {
	  std::cout << i << '\n';
      output = module.forward(inputs).toTensor();
    }
  }
}