
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

# Exercises
add_executable(device_info ${EXERCISES_DIR}/${EXERCISE_DIR}01/DeviceInfo.cpp)
target_link_libraries(device_info PRIVATE OpenCL::OpenCL)
target_include_directories(device_info PRIVATE ${COMMON_DIRECTORY})

add_executable(cxx_vector_add ${EXERCISES_DIR}/${EXERCISE_DIR}03/vadd.cpp)
target_link_libraries(cxx_vector_add PRIVATE OpenCL::OpenCL)
target_include_directories(cxx_vector_add PRIVATE ${COMMON_DIRECTORY})

add_library(matrix_library ${COMMON_DIRECTORY}/matrix_lib.cpp)
target_include_directories(matrix_library PRIVATE ${COMMON_DIRECTORY})

# Solution
add_executable(cxx_vector_add_chain ${SOLUTIONS_DIR}/${EXERCISE_DIR}04/vadd_chain.cpp)
target_link_libraries(cxx_vector_add_chain PRIVATE OpenCL::OpenCL matrix_library)
target_include_directories(cxx_vector_add_chain PRIVATE ${COMMON_DIRECTORY})

add_executable(cxx_vector_add_abc ${SOLUTIONS_DIR}/${EXERCISE_DIR}05/vadd_abc.cpp)
target_link_libraries(cxx_vector_add_abc PRIVATE OpenCL::OpenCL matrix_library)
target_include_directories(cxx_vector_add_abc PRIVATE ${COMMON_DIRECTORY})

add_executable(cxx_matrix_lib ${SOLUTIONS_DIR}/${EXERCISE_DIR}06/matmul.cpp)
target_link_libraries(cxx_matrix_lib PRIVATE OpenCL::OpenCL matrix_library)
target_include_directories(cxx_matrix_lib PRIVATE ${COMMON_DIRECTORY})

add_executable(cxx_matrix_lib_priv ${SOLUTIONS_DIR}/${EXERCISE_DIR}07/matmul.cpp)
target_link_libraries(cxx_matrix_lib_priv PRIVATE OpenCL::OpenCL matrix_library)
target_include_directories(cxx_matrix_lib_priv PRIVATE ${COMMON_DIRECTORY})

add_executable(cxx_matrix_lib_priv_block ${SOLUTIONS_DIR}/${EXERCISE_DIR}08/matmul.cpp)
target_link_libraries(cxx_matrix_lib_priv_block PRIVATE OpenCL::OpenCL matrix_library)
target_include_directories(cxx_matrix_lib_priv_block PRIVATE ${COMMON_DIRECTORY})

add_executable(cxx_pi ${SOLUTIONS_DIR}/${EXERCISE_DIR}09/pi_ocl.cpp)
target_link_libraries(cxx_pi PRIVATE OpenCL::OpenCL matrix_library)
target_include_directories(cxx_pi PRIVATE ${COMMON_DIRECTORY})

add_executable(cxx_game_of_life ${SOLUTIONS_DIR}/${EXERCISE_DIR}13/gameoflife.cpp)
target_link_libraries(cxx_game_of_life PRIVATE OpenCL::OpenCL matrix_library)
target_include_directories(cxx_game_of_life PRIVATE ${COMMON_DIRECTORY})
