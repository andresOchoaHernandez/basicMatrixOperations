#include "HelperFunctions.hpp"

void one_initialization(Matrix2d &A)
{
    for (int i = 0; i < A.rows;i++)
    {
        for(int j = 0;j< A.columns; j++)
        {
            A.data[i*A.columns+j] = 1;
        }
    }
}

void fill_matrix(Matrix2d &A)
{
    double value = 1;

    for(int i = 0;i < A.rows; i++)
    {
        for(int j = 0; j < A.columns; j++)
        {
            A.data[i*A.columns + j] = value++;
        }
    }
}

int checkIfCorrect(const Matrix2d &A,const Matrix2d &A_t)
{
    for(int i = 0; i < A_t.rows;i++)
    {
        for(int j = 0; j < A_t.columns;j++)
        {
            if(A_t.data[i*A_t.columns + j] != A.data[j*A.columns + i]) return -1;
        }
    }

    return 0;
}