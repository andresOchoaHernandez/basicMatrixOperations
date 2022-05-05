#include <iostream>
#include <chrono>
#include "BasicMatrixOperations.hpp"

/* ============== HELPER FUNCTIONS ============== */

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

void fill_matrix(float *A,int n,int m)
{
    int value = 1;

    for(int i = 0;i < n; i++)
    {
        for(int j = 0; j < m ; j++)
        {
            A[i*m + j] = value++;
        }
    }
}

int checkIfCorrect(float * A,int n,int m,float * A_transpose,int n_t,int m_t)
{
    for(int i = 0; i < n_t;i++)
    {
        for(int j = 0; j <m_t;j++)
        {
            if(A_transpose[i*m_t + j] != A[j*m + i]) return -1;
        }
    }

    return 0;
}

/* ============================================== */

/* =================== TESTS ==================== */

/* MATRIX MULTIPLICATION */

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

int openMP_matrixMulTestOne()
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
    openMP_matrixMultiplication(
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

/* MATRIX TRANPOSE */

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

int openMP_matrixTransposeTestOne()
{
    const int n = N;
    const int m = M;

    float *A = new float[n*m];
    fill_matrix(A,n,m);

    const int n_t =m;
    const int m_t = n;

    float *A_t = new float[n_t*m_t]();

    if(openMP_matrixTranspose(A,n,m,A_t,n_t,m_t) == -1) return -1;
    if(checkIfCorrect(A,n,m,A_t,n_t,m_t) == -1)return -1;

    delete[] A;
    delete[] A_t;

    return 0;
}
/* ============================================== */

int main(void)
{

    if(matrixMulTestOne() == -1){
        std::cout << "matrixMulTestOne [FAILED]\n";
        return -1;
    }

    if(openMP_matrixMulTestOne() == -1){
        std::cout << "openMP_matrixMulTestOne [FAILED]\n";
        return -1;
    }

    if(matrixTransposeTestOne() == -1){
        std::cout << "matrixTransposeTestOne [FAILED]\n";
        return -1;
    }

    if(openMP_matrixTransposeTestOne() == -1){
        std::cout << "openMP_matrixTransposeTestOne [FAILED]\n";
        return -1;
    }

    return 0;
}