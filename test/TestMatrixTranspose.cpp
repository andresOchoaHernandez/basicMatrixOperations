#include <iostream>

#include "HelperFunctions.hpp"
#include "BasicMatrixOperations.hpp"

const int N = 5000;
const int M = 2000;

int matrixTransposeTestOne()
{
    const int n = N;
    const int m = M;

    float *A = new float[n*m];
    fill_matrix(A,n,m);

    const int n_t =m;
    const int m_t = n;

    float *A_t = new float[n_t*m_t]();

    if(matrixTranspose(A,n,m,A_t,n_t,m_t) == -1) return -1;
    if(checkIfCorrect(A,n,m,A_t,n_t,m_t) == -1)return -1;

    delete[] A;
    delete[] A_t;

    return 0;
}
/* ============================================== */

int main(void)
{
    if(matrixTransposeTestOne() == -1){
        std::cout << "matrixTransposeTestOne [FAILED]\n";
        return -1;
    }

    return 0;
}