#include "BasicMatrixOperations.hpp"

int main(void)
{
    float *a;
    float *a_transpose;
    int N = 10;
    int M = 10;

    matrixTranspose(a,N,M,a_transpose);

    return 0;
}