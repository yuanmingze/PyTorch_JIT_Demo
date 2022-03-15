This demo tests the PyTorch JIT.

Two dominant models of execution in deep learning: eager and graph.

**eager execution**: it builds its computational graph (the set of steps needed to perform forward or backwards propagation through the network) at runtime.

**graph execution**: Graph execution pushes the management of the computational graph down to the kernel level (e.g. to a C++ process) by adding an additional compilation step to the process.

## Script Model

- jit.ScriptModule class
- @torch.jit.script_method decorator.

## Trace Model

- torch.jit.trace  for functions
- torch.jit.trace_module  for modules

# Code

Python test

build the cpp file, use the head files and lib files from the python set- folder.

## test_02

## test_03

# Reference:

- PyTorch JIT Training: <https://spell.ml/blog/pytorch-jit-YBmYuBEAACgAiv71>
- <https://pytorch.org/docs/stable/jit.html#mixing-tracing-and-scripting>

- https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/