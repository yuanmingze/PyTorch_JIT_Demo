'''
TorchScript. 
Use the jit.script to serialize the model. 

'''
import configuration as config
import torch


class MyModule(torch.nn.Module):

    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        if input.sum() > 0:
            output = self.weight.mv(input)
        else:
            output = self.weight + input
        return output



my_module = MyModule(10, 20)
sm = torch.jit.script(my_module)
sm.save(config.OUTPUT_DIR + "test_01_jit_script.pt")

## the expected output from c++
test_input = torch.ones((20))
result = my_module(test_input)

print("The expected result is \n{}".format(result[0:5]))
