cmake_minimum_required(VERSION 3.5.1)
project(find_landmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB sources ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
file(GLOB headers ${CMAKE_CURRENT_LIST_DIR}/src/*.h)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
#find_package(Boost REQUIRED COMPONENTS graph)
find_package(OpenMP)


set(mandatory_libs ${OpenCV_LIBS})

if(OpenMP_CXX_FOUND)
    list(APPEND {mandatory_libs} OpenMP::OpenMP_CXX)
endif()

add_executable(${PROJECT_NAME} 
        main.cpp
        ${headers}
        ${sources}
)

target_include_directories(
        ${PROJECT_NAME}
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
#        ${Boost_INCLUDE_DIRS}
)

target_link_libraries(
        ${PROJECT_NAME}
        ${mandatory_libs}
)