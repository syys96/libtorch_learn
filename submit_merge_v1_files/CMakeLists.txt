cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(submit_merged)

set(CMAKE_PREFIX_PATH ../../libtorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_definitions(-D_BOTZONE_ONLINE)
add_executable(submit_merged submit_merged.cpp)
target_link_libraries(submit_merged "${TORCH_LIBRARIES}")
set_property(TARGET submit_merged PROPERTY CXX_STANDARD 14)