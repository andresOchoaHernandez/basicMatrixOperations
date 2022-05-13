#include "HelperFunctions.hpp"

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