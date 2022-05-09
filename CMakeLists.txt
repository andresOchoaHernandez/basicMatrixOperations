project(basicMatrixOperations)

set(BIN_DIR     ${PROJECT_SOURCE_DIR}/bin/basicMatrixOperations)
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/inc)
set(SOURCE_DIR  ${PROJECT_SOURCE_DIR}/src)
set(TEST_DIR    ${PROJECT_SOURCE_DIR}/test)

set(CMAKE_CXX_STANDARD 11)

add_executable(main ${BIN_DIR})

target_include_directories(
    ${BIN_DIR}
    PUBLIC  ${INCLUDE_DIR}
    PRIVATE ${SOURCE_DIR}
)

target_sources(
    ${BIN_DIR}
    PRIVATE ${SOURCE_DIR}/BasicMatrixOperations.cpp
    PUBLIC  ${INCLUDE_DIR}/BasicMatrixOperations.hpp
)

target_compile_options(${BIN_DIR} "-Wall -fopenmp -03")