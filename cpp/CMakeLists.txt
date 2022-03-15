cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(test_00 test_00.cpp)
target_link_libraries(test_00 "${TORCH_LIBRARIES}")
set_property(TARGET test_00 PROPERTY CXX_STANDARD 14)