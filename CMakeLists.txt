
cmake_minimum_required(VERSION 3.8)

project(opencl_training LANGUAGES C CXX)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

# Directory structure
set(EXERCISE_DIR "Exercise")
set(EXERCISES_DIR "Exercises")
set(SOLUTIONS_DIR "Solutions")
set(COMMON_DIRECTORY "${CMAKE_SOURCE_DIR}/common")

# CL_DEVICE_TYPE_DEFAULT
add_compile_definitions(DEVICE=CL_DEVICE_TYPE_GPU)

# Common library
add_library(matrix_library ${COMMON_DIRECTORY}/matrix_lib.cpp)
target_include_directories(matrix_library PRIVATE ${COMMON_DIRECTORY})

# Exercises
# add_executable(device_info ${EXERCISES_DIR}/${EXERCISE_DIR}01/DeviceInfo.cpp)
# target_link_libraries(device_info PRIVATE OpenCL::OpenCL)
# target_include_directories(device_info PRIVATE ${COMMON_DIRECTORY})
#
# add_executable(vector_add ${EXERCISES_DIR}/${EXERCISE_DIR}03/vadd.cpp)
# target_link_libraries(vector_add PRIVATE OpenCL::OpenCL)
# target_include_directories(vector_add PRIVATE ${COMMON_DIRECTORY})

# Solution
add_executable(vector_add ${SOLUTIONS_DIR}/vector_add/main.cpp)
target_link_libraries(vector_add PRIVATE OpenCL::OpenCL)
target_include_directories(vector_add PRIVATE ${COMMON_DIRECTORY})

add_executable(vector_add_multiple ${SOLUTIONS_DIR}/vector_add_multiple/main.cpp)
target_link_libraries(vector_add_multiple PRIVATE OpenCL::OpenCL)
target_include_directories(vector_add_multiple PRIVATE ${COMMON_DIRECTORY})

add_executable(matrix_multiply_basic ${SOLUTIONS_DIR}/matrix_multiply_basic/main.cpp)
target_link_libraries(matrix_multiply_basic PRIVATE OpenCL::OpenCL matrix_library)
target_include_directories(matrix_multiply_basic PRIVATE ${COMMON_DIRECTORY})

add_executable(matrix_multiply_private ${SOLUTIONS_DIR}/matrix_multiply/main.cpp)
target_link_libraries(matrix_multiply_private PRIVATE OpenCL::OpenCL matrix_library)
target_include_directories(matrix_multiply_private PRIVATE ${COMMON_DIRECTORY})

add_executable(matrix_multiply_blocked ${SOLUTIONS_DIR}/matrix_multiply_blocked/main.cpp)
target_link_libraries(matrix_multiply_blocked PRIVATE OpenCL::OpenCL matrix_library)
target_include_directories(matrix_multiply_blocked PRIVATE ${COMMON_DIRECTORY})

add_executable(pi ${SOLUTIONS_DIR}/${EXERCISE_DIR}09/pi_ocl.cpp)
target_link_libraries(pi PRIVATE OpenCL::OpenCL matrix_library)
target_include_directories(pi PRIVATE ${COMMON_DIRECTORY})

add_executable(game_of_life ${SOLUTIONS_DIR}/${EXERCISE_DIR}13/gameoflife.cpp)
target_link_libraries(game_of_life PRIVATE OpenCL::OpenCL matrix_library)
target_include_directories(game_of_life PRIVATE ${COMMON_DIRECTORY})
