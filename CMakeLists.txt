project(kf_code)
cmake_minimum_required(VERSION 3.10)

add_subdirectory(thirdparty/pybind11)

pybind11_add_module(kf_cpp wrappers.cpp)
pybind11_add_module(NPPconv boxFilterNPP.cpp)