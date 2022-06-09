project(basicMatrixOperations)

cmake_minimum_required(VERSION 3.8.2)

# ======================== MAIN ========================
add_executable(nn ${PROJECT_SOURCE_DIR}/nn.cpp)

target_include_directories(
    nn
    PUBLIC  ${PROJECT_SOURCE_DIR}/include/
    PRIVATE ${PROJECT_SOURCE_DIR}/src/
)

target_sources(
    nn
    PUBLIC  ${PROJECT_SOURCE_DIR}/include/BasicMatrixOperations.hpp
    PRIVATE ${PROJECT_SOURCE_DIR}/src/BasicMatrixOperations.cpp
)

# ======================== CUDA ========================
find_package(CUDA)

if(CUDA_FOUND)

    enable_language(CUDA)

    try_run(
    RUN_RESULT_VAR 
    COMPILE_RESULT_VAR
    ${CMAKE_CURRENT_BINARY_DIR}/ 
    ${PROJECT_SOURCE_DIR}/util/CheckIfCudaDeviceIsPresent.cu
    COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
    RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR)
    if("${COMPILE_RESULT_VAR}" AND ("${RUN_RESULT_VAR}" GREATER 0))
        message("-- Found : ${RUN_RESULT_VAR} cuda capable device(s)")    
        set(CUDA_HAVE_GPU TRUE)
    else()
        message("-- Found cuda packages but there's no cuda capable device")
        set(CUDA_HAVE_GPU FALSE)
    endif()

else()
    message("-- Cuda packages not found")
endif(CUDA_FOUND)


if(CUDA_FOUND AND CUDA_HAVE_GPU)
    add_definitions(-DUSE_GPU_FUNCTIONS=TRUE)

    add_executable(gpu_nn ${PROJECT_SOURCE_DIR}/gpu_nn.cpp)

    target_include_directories(
        gpu_nn
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/
        PRIVATE ${PROJECT_SOURCE_DIR}/src/
    )
    
    target_sources(
        gpu_nn
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/BasicMatrixOperations.hpp
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/BasicMatrixOperations.cuh
        PRIVATE ${PROJECT_SOURCE_DIR}/src/BasicMatrixOperations.cpp
        PRIVATE ${PROJECT_SOURCE_DIR}/src/BasicMatrixOperations.cu
    )
endif()

# ======================== TEST ========================

# ===== MATRIX MULTIPLICATION =====
add_executable(test1 ${PROJECT_SOURCE_DIR}/test/TestMatrixMultiplication.cpp)

target_include_directories(
    test1
    PUBLIC  ${PROJECT_SOURCE_DIR}/include/
    PRIVATE ${PROJECT_SOURCE_DIR}/src/
)

target_sources(
    test1
    PUBLIC  ${PROJECT_SOURCE_DIR}/include/BasicMatrixOperations.hpp
    PRIVATE ${PROJECT_SOURCE_DIR}/src/BasicMatrixOperations.cpp
    PRIVATE ${PROJECT_SOURCE_DIR}/test/HelperFunctions.hpp
    PRIVATE ${PROJECT_SOURCE_DIR}/test/HelperFunctions.cpp
)

enable_testing()

add_test(
    NAME test1
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build/
    COMMAND ./test1
)

# ===== MATRIX TRANSPOSE =====
add_executable(test2 ${PROJECT_SOURCE_DIR}/test/TestMatrixTranspose.cpp)

target_include_directories(
    test2
    PUBLIC  ${PROJECT_SOURCE_DIR}/include/
    PRIVATE ${PROJECT_SOURCE_DIR}/src/
)

target_sources(
    test2
    PUBLIC  ${PROJECT_SOURCE_DIR}/include/BasicMatrixOperations.hpp
    PRIVATE ${PROJECT_SOURCE_DIR}/src/BasicMatrixOperations.cpp
    PRIVATE ${PROJECT_SOURCE_DIR}/test/HelperFunctions.hpp
    PRIVATE ${PROJECT_SOURCE_DIR}/test/HelperFunctions.cpp
)

add_test(
    NAME test2
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build/
    COMMAND ./test2
)

# ============ GPU MATRIX MUL =====================
if(CUDA_FOUND AND CUDA_HAVE_GPU)
    add_executable(gpu_test_1 ${PROJECT_SOURCE_DIR}/test/TestGpuMatrixMultiplication.cu)

    target_include_directories(
        gpu_test_1
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/
        PRIVATE ${PROJECT_SOURCE_DIR}/src/
    )

    target_sources(
        gpu_test_1
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/BasicMatrixOperations.hpp
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/BasicMatrixOperations.cuh
        PRIVATE ${PROJECT_SOURCE_DIR}/src/BasicMatrixOperations.cpp
        PRIVATE ${PROJECT_SOURCE_DIR}/src/BasicMatrixOperations.cu
        PRIVATE ${PROJECT_SOURCE_DIR}/test/HelperFunctions.hpp
        PRIVATE ${PROJECT_SOURCE_DIR}/test/HelperFunctions.cpp
    )

    enable_testing()

    add_test(
        NAME gpu_test_1
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build/
        COMMAND ./gpu_test_1
    )

    add_executable(gpu_test_2 ${PROJECT_SOURCE_DIR}/test/TestGpuMatrixTranspose.cu)

    target_include_directories(
        gpu_test_2
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/
        PRIVATE ${PROJECT_SOURCE_DIR}/src/
    )

    target_sources(
        gpu_test_2
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/BasicMatrixOperations.hpp
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/BasicMatrixOperations.cuh
        PRIVATE ${PROJECT_SOURCE_DIR}/src/BasicMatrixOperations.cpp
        PRIVATE ${PROJECT_SOURCE_DIR}/src/BasicMatrixOperations.cu
        PRIVATE ${PROJECT_SOURCE_DIR}/test/HelperFunctions.hpp
        PRIVATE ${PROJECT_SOURCE_DIR}/test/HelperFunctions.cpp
    )

    add_test(
        NAME gpu_test_2
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build/
        COMMAND ./gpu_test_2
    )

    add_executable(gpu_test_3 ${PROJECT_SOURCE_DIR}/test/TestGpuMatrixArithmetic.cu)

    target_include_directories(
        gpu_test_3
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/
        PRIVATE ${PROJECT_SOURCE_DIR}/src/
    )

    target_sources(
        gpu_test_3
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/BasicMatrixOperations.hpp
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/BasicMatrixOperations.cuh
        PRIVATE ${PROJECT_SOURCE_DIR}/src/BasicMatrixOperations.cpp
        PRIVATE ${PROJECT_SOURCE_DIR}/src/BasicMatrixOperations.cu
        PRIVATE ${PROJECT_SOURCE_DIR}/test/HelperFunctions.hpp
        PRIVATE ${PROJECT_SOURCE_DIR}/test/HelperFunctions.cpp
    )

    add_test(
        NAME gpu_test_3
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build/
        COMMAND ./gpu_test_3
    )
endif()