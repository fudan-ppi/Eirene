cmake_minimum_required(VERSION 3.5.0)

project(GPU-STM)

include(FindCUDA)
find_package(CUDA REQUIRED)


#set(Boost_USE_STATIC_LIBS ON)
#set(Boost_USE_MULTITHREADED ON)
#find_package(Boost 1.62.0 COMPONENTS system thread)
#find_package(Boost)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -mavx -mavx2 -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mavx -mavx2 -fopenmp")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    #message("cuda debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -gencode arch=compute_70,code=sm_70 -O0 -g -G")
    #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -gencode arch=compute_61,code=sm_61 -O0 -g -G")
else() 
    #message("cuda release")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -gencode arch=compute_70,code=sm_70 -O3")
    #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -gencode arch=compute_61,code=sm_61 -O3")
endif()

include_directories(./src ./cub)

#aux_source_directory(./src SOURCE_LIST)
#aux_source_directory(./src/gpu-stm-tree-v2 SOURCE_LIST)
file(GLOB SOURCE_LIST "src/*.cpp" "src/gpu-stm-tree-v*/*.cpp")
file(GLOB GPU_SOURCE_LIST "src/*.cu" "src/gpu-stm-tree-v*/*.cu")


cuda_add_executable(stm_test ${SOURCE_LIST} ${GPU_SOURCE_LIST})


 

