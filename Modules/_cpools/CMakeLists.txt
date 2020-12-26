cmake_minimum_required(VERSION 3.17)
project(_cpools)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_PREFIX_PATH "/home/samuel-moses/.local/lib/python3.6/site-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")



add_executable(_cpools
        src/bottom_pool.cpp
        src/left_pool.cpp
        src/right_pool.cpp
        src/top_pool.cpp)

target_link_libraries(_cpools "${TORCH_LIBRARIES}")
set_property(TARGET _cpools PROPERTY CXX_STANDARD 14)