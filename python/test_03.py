import configuration as config
import torch
from torch import nn


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(28 * 28, 512),
                                               nn.ReLU(), nn.Linear(512, 512),
                                               nn.ReLU(), nn.Linear(512, 10),
                                               nn.ReLU())

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(config.OUTPUT_DIR + "test_02_training.pth"))

sm = torch.jit.script(model)
model_jit_filepath = config.OUTPUT_DIR + "test_03_jit_script.pt"
sm.save(model_jit_filepath)
print("Output JIT model to {}".format(model_jit_filepath))
