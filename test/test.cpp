#include <iostream>
#include "BasicMatrixOperations.hpp"

int main(void)
{
    float *a;
    float *a_transpose;
    int N = 10;
    int M = 10;

    std::cout << "TESTS\n";

    matrixTranspose(a,N,M,a_transpose);

    return 0;
}