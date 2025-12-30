cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(molnex LANGUAGES CXX)

# Build C++ extensions from src/molnex/op
add_subdirectory(src/molnex/op)
