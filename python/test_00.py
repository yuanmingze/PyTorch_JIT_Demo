'''
TorchScript. 
Use the trace to serialize the model. 

'''
import configuration as config

import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18().cuda()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224).cuda()

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

traced_script_module.save(config.OUTPUT_DIR + "test_00_jit_trace.pt")

## the expected output from c++
test_input = torch.ones_like(example)
result = model(test_input)

print("The expected result is \n{}".format(result[:, 0:5]))
