#include <iostream>

#include "BasicMatrixOperations.hpp"
#include "HelperFunctions.hpp"

const int NA = 1024;
const int MA = 1024;
const int NB = MA;
const int MB = 1024;

int matrixMulTestOne()
{
    Matrix2d A;
    A.rows    = NA;
    A.columns = MA;
    A.data = new double[A.rows*A.columns];
    one_initialization(&A);

    Matrix2d B;
    B.rows = NB;
    B.columns = MB;
    B.data = new double[B.rows*B.columns];
    one_initialization(&B);

    Matrix2d C;
    C.rows    = A.rows;
    C.columns = B.columns;
    C.data = new double[C.rows*C.columns];

    int code =  
    matrixMultiplication(&A,&B,&C);

    if(code == -1) return -1;

    for (int i = 0; i < C.rows;i++)
    {
        for(int j = 0;j< C.columns; j++)
        {
            if(C.data[i*C.columns+j] != A.columns) return -1;
        }
    }

    delete[] A.data;
    delete[] B.data;
    delete[] C.data;

    return 0;
}

int main(void)
{

    if(matrixMulTestOne() == -1){
        std::cout << "matrixMulTestOne [FAILED]\n";
        return -1;
    }

    return 0;
}