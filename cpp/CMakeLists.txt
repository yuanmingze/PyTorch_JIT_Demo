cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)
set(CMAKE_CXX_STANDARD 14)

# cuda
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-10.2")
set(CUDNN_LIBRARY_PATH "/usr/local/cudnn/cudnn-linux-x86_64-8.3.1.22_cuda10.2/lib/libcudnn.so")
set(CUDNN_INCLUDE_PATH "/usr/local/cudnn/cudnn-linux-x86_64-8.3.1.22_cuda10.2/include")

# Torch
set(Torch_DIR "/home/engs2369/miniconda3/envs/pytorch_1_8_2/lib/python3.8/site-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)

add_executable(test_00 test_00.cpp)
target_link_libraries(test_00 "${TORCH_LIBRARIES}")

add_executable(test_01 test_01.cpp)
target_link_libraries(test_01 "${TORCH_LIBRARIES}")

add_executable(test_02_script test_02_script.cpp)
target_link_libraries(test_02_script "${TORCH_LIBRARIES}")

add_executable(test_02_trace test_02_trace.cpp)
target_link_libraries(test_02_trace "${TORCH_LIBRARIES}")