project(basicMatrixOperations)

cmake_minimum_required(VERSION 3.5)

# main
add_executable(main ${PROJECT_SOURCE_DIR}/main.cpp)

target_include_directories(
    main
    PUBLIC  ${PROJECT_SOURCE_DIR}/include/
    PRIVATE ${PROJECT_SOURCE_DIR}/src/
)

target_sources(
    main
    PUBLIC  ${PROJECT_SOURCE_DIR}/include/BasicMatrixOperations.hpp
    PRIVATE ${PROJECT_SOURCE_DIR}/src/BasicMatrixOperations.cpp
)

# test

#MATRIX MUL
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
    PRIVATE ${PROJECT_SOURCE_DIR}/test/HelperFunctions.cpp
)

enable_testing()

add_test(
    NAME test1
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build/
    COMMAND ./test1
)

#MATRIX TRANSPOSE

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
    PRIVATE ${PROJECT_SOURCE_DIR}/test/HelperFunctions.cpp
)

add_test(
    NAME test2
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build/
    COMMAND ./test2
)