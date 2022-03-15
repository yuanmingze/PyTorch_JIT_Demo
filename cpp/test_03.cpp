#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char *argv[])
{
  std::string model_filepath;
  if (argc == 2)
  {
    model_filepath.assign(argv[1]);
  }
  else if (argc == 1)
  {
    model_filepath.assign("../data/test_03_jit_script.pt");
  }
  else
  {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try
  {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(model_filepath);
  }
  catch (const c10::Error &e)
  {
    std::cerr << "error loading the model\n";
    return -1;
  }

  //
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;

  // create a tensor on device 0
  torch::Tensor tensor0 = torch::ones({1, 28, 28}).cuda();
  inputs.push_back(tensor0);

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output << '\n';
}