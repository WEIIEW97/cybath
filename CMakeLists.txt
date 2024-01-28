cmake_minimum_required(VERSION 3.5.1)
project(cybathlon LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(onnx_runtime_dir ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-gpu-1.16.3)
#set(fast_deploy_install_dir ${CMAKE_CURRENT_SOURCE_DIR}/fastdeploy-linux-x64-gpu-1.0.7)
#include(${fast_deploy_install_dir}/FastDeploy.cmake)
#message(STATUS ${onnx_runtime_dir})

include(CheckIncludeFileCXX)

file(GLOB_RECURSE sources ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
file(GLOB_RECURSE headers ${CMAKE_CURRENT_LIST_DIR}/src/*.h)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(realsense2 REQUIRED)
find_package(OpenMP)

add_compile_definitions(PROJECT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")

message(STATUS "opencv include dir: ${OpenCV_INCLUDE_DIRS}")

set(mandatory_libs ${OpenCV_LIBS})

if(OpenMP_CXX_FOUND)
    list(APPEND {mandatory_libs} OpenMP::OpenMP_CXX)
endif()

add_executable(${PROJECT_NAME} 
        main.cpp
        ${sources}
)

if(HAVE_TENSORRT_PROVIDER_FACTORY_H)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DHAVE_TENSORRT_PROVIDER_FACTORY_H)
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE -DDEBUG)

target_include_directories(
        ${PROJECT_NAME}
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${realsense2_INCLUDE_DIR}
#        ${FASTDEPLOY_INCS}
        ${onnx_runtime_dir}/include/onnxruntime/core/session/
        ${onnx_runtime_dir}/include/onnxruntime/core/providers/tensorrt/
        ${onnx_runtime_dir}/include
)

CHECK_INCLUDE_FILE_CXX(tensorrt_provider_factory.h HAVE_TENSORRT_PROVIDER_FACTORY_H)

target_link_directories(${PROJECT_NAME}
        PRIVATE
        ${onnx_runtime_dir}/lib
)

target_link_libraries(
        ${PROJECT_NAME}
        ${mandatory_libs}
        ${realsense2_LIBRARY}
#        ${FASTDEPLOY_LIBS}
        libonnxruntime.so
)