cmake_minimum_required(VERSION 3.14)
project(wacoh_pybind)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

include_directories(src)
pybind11_add_module(wacoh_sensor bindings.cpp src/Wacoh_linux.cpp)