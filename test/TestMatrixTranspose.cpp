#include <iostream>

#include "HelperFunctions.hpp"
#include "BasicMatrixOperations.hpp"

const int N = 5;
const int M = 4;

int matrix_transpose_test_one()
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

    if(matrix_transpose(A,A_t) == -1) return -1;
    if(check_if_correct(A,A_t) == -1)return -1;

    delete[] A.data;
    delete[] A_t.data;

    return 0;
}
/* ============================================== */

int main(void)
{
    if(matrix_transpose_test_one() == -1){
        std::cout << "matrixTransposeTestOne [FAILED]\n";
        return -1;
    }

    return 0;
}