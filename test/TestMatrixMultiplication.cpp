#include <iostream>

#include "BasicMatrixOperations.hpp"
#include "HelperFunctions.hpp"

const int NA = 1024;
const int MA = 1024;
const int NB = MA;
const int MB = 1024;

int matrixMulTestOne()
{
    const int n_A = NA;
    const int m_A = MA;
    float *A = new float[n_A*m_A];
    one_initialization(A,n_A,m_A);

    const int n_B = NB;
    const int m_B = MB;
    float *B = new float[n_B*m_B];
    one_initialization(B,n_B,m_B);

    const int n_C = n_A;
    const int m_C = m_B;
    float *C = new float[n_C*m_C];

    int code =  
    matrixMultiplication(
        A,n_A,m_A,
        B,n_B,m_B,
        C,n_C,m_C
    );

    if(code == -1) return -1;

    for (int i = 0; i < n_C;i++)
    {
        for(int j = 0;j< m_C; j++)
        {
            if(C[i*m_C+j] != m_A) return -1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;

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