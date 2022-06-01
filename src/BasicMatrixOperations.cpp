#include <iostream>
#include <chrono>

#include "BasicMatrixOperations.hpp"

void print_matrix(Matrix2d* A)
{
    std::cout << "--------------------\n";
    for (int i = 0; i < A->rows;i++)
    {
        for(int j = 0;j < A->columns; j++)
        {
            if(j == A->columns-1)
                std::cout << A->data[i*A->rows+j];
            else
                std::cout << A->data[i*A->rows+j]<<", ";
        }
        std::cout << "\n";
    }
    std::cout << "--------------------\n";
}

int matrixMultiplication(Matrix2d* A,Matrix2d* B,Matrix2d* C)
{
    if( A->columns != B->rows ){std::cerr << "matrix_A columns don't match matrix_B rows\n";return -1;}
    if((A->rows != C->rows) || ( B->columns != C->columns) ){std::cerr << "given matrix_C dimensions\n";return -1;}

    for(int rows = 0; rows < C->rows;rows++)
    {
        for(int columns = 0; columns < C->columns;columns++)
        {
            double sum = 0;
            for(int k = 0; k < A->columns;k++)
            {
                sum += A->data[rows * A->columns + k]*B->data[k* B->columns + columns];
            }
            C->data[rows * C->columns + columns] = sum;
        }
    }

    return 0;
}

int matrixTranspose(Matrix2d* A,Matrix2d* A_t)
{
    if( (A->rows != A_t->columns) || (A->columns != A_t->rows) ){std::cerr << "given matrix's dimensions don't match\n";return -1;}

    for(int i = 0; i < A_t->rows;i++)
    {
        for(int j = 0; j <A_t->columns;j++)
        {
            A_t->data[i*A_t->columns + j] = A->data[j*A->columns + i];
        }
    }

    return 0;
}