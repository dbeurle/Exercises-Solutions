
cmake_minimum_required(VERSION 3.8)

project(opencl_training LANGUAGES C CXX)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

# Directory structure
set(EXERCISE_DIR "Exercise")
set(EXERCISES_DIR "Exercises")
set(SOLUTIONS_DIR "Solutions")
set(COMMON_CXX_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/${EXERCISES_DIR}/Cpp_common")

add_executable(device_info ${EXERCISES_DIR}/${EXERCISE_DIR}01/Cpp/DeviceInfo.cpp)
target_link_libraries(device_info PRIVATE OpenCL::OpenCL)
target_include_directories(device_info PRIVATE ${COMMON_CXX_INCLUDE_DIR})

add_executable(cxx_vector_add ${EXERCISES_DIR}/${EXERCISE_DIR}03/Cpp/vadd.cpp)
target_link_libraries(cxx_vector_add PRIVATE OpenCL::OpenCL)
target_include_directories(cxx_vector_add PRIVATE ${COMMON_CXX_INCLUDE_DIR})
