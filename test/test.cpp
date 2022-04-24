#include <iostream>
#include "BasicMatrixOperations.hpp"

void one_initialization(float *A,int n,int m)
{
    for (int i = 0; i < n;i++)
    {
        for(int j = 0;j< m; j++)
        {
            A[i*m+j] = 1;
        }
    }
}

int testOne()
{
    const int n_A = 10;
    const int m_A = 40;
    float *A = new float[n_A*m_A];
    one_initialization(A,n_A,m_A);

    print_matrix(A,n_A,m_A);

    const int n_B = 40;
    const int m_B = 30;
    float *B = new float[n_B*m_B];
    one_initialization(B,n_B,m_B);

    print_matrix(B,n_B,m_B);


    const int n_C = n_A;
    const int m_C = m_B;
    float *C = new float[n_C*m_C];

    int code =  
    matrixMultiplication(
        A,n_A,m_A,
        B,n_B,m_B,
        C,n_C,m_C
    );

    print_matrix(C,n_C,m_C);
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

    if(testOne() == -1) return -1;

    return 0;
}