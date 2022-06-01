#include <iostream>

#include "HelperFunctions.hpp"
#include "BasicMatrixOperations.hpp"

const int N = 5;
const int M = 4;

int matrixTransposeTestOne()
{
    Matrix2d A;
    A.rows    = N;
    A.columns = M; 

    A.data = new double[A.rows*A.columns];
    fill_matrix(A);

    Matrix2d A_t;
    A_t.rows    = M;
    A_t.columns = N; 

    A_t.data = new double[A_t.rows*A_t.columns]();

    if(matrixTranspose(A,A_t) == -1) return -1;
    if(checkIfCorrect(A,A_t) == -1)return -1;

    print_matrix(A);
    print_matrix(A_t);

    delete[] A.data;
    delete[] A_t.data;

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