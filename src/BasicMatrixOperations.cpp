#include <iostream>
#include <chrono>
//#include <stdexcept>

#include "BasicMatrixOperations.hpp"

void print_matrix(const Matrix2d& A)
{
    std::printf("--------------------\n");
    for (int i = 0; i < A.rows;i++)
    {
        for(int j = 0;j < A.columns; j++)
        {
            std::printf("%f ",A.data[i*A.columns+j]);
        }
        std::printf("\n");
    }
    std::printf("--------------------\n");
}

int matrix_multiplication(const Matrix2d& A,const Matrix2d& B,Matrix2d& C)
{
    if( A.columns != B.rows ){std::cerr << "matrix_A columns don't match matrix_B rows\n";
    return -1;}
    if((A.rows != C.rows) || ( B.columns != C.columns) ){std::cerr << "given matrix_C dimensions\n";}

    for(int rows = 0; rows < C.rows;rows++)
    {
        for(int columns = 0; columns < C.columns;columns++)
        {
            double sum = 0;
            for(int k = 0; k < A.columns;k++)
            {
                sum += A.data[rows * A.columns + k]*B.data[k* B.columns + columns];
            }
            C.data[rows * C.columns + columns] = sum;
        }
    }

    return 0;
}

int matrix_dot_product(const Matrix2d& A, const Matrix2d& B,Matrix2d& C)
{
    if(
        !(            
            A.rows    == B.rows     && 
            A.columns == B.columns  &&
            C.rows    == A.rows     &&
            C.columns == A.columns
         )
      )
    {

        std::cerr << "Given matrices' dimensions don't match:"              << std::endl
                  << "param : A (" << A.rows <<" x " << A.columns << ")"    << std::endl
                  << "param : B ("  << B.rows  <<" x " << B.columns  << ")" << std::endl
                  << "param : C ("  << C.rows  <<" x " << C.columns  << ")" << std::endl;
        return -1;
    }

    for(int i = 0; i < C.rows;i++)
    {
        for(int j = 0;j < C.columns;j++)
        {
            C.data[i*C.columns + j] = A.data[i*C.columns + j] * B.data[i*C.columns + j]; 
        }
    }

    return 1;
}

int scalar_matrix_dot_product(const double scalar, const Matrix2d& A,Matrix2d& C)
{
    if( (A.columns != C.columns) || (A.rows != C.rows) ){std::cerr << "given matrix's dimensions don't match\n";return -1;}

    for(int i = 0; i < A.rows;i++)
    {
        for(int j = 0;j < A.columns;j++)
        {
            C.data[i*C.columns + j] = scalar * A.data[i*A.columns + j]; 
        }
    }

    return 1;
}

int matrix_transpose(const Matrix2d& A,Matrix2d& A_t)
{
    if( (A.rows != A_t.columns) || (A.columns != A_t.rows) ){std::cerr << "given matrix's dimensions don't match\n";return -1;}

    for(int i = 0; i < A_t.rows;i++)
    {
        for(int j = 0; j <A_t.columns;j++)
        {
            A_t.data[i*A_t.columns + j] = A.data[j*A.columns + i];
        }
    }

    return 0;
}

int matrix_sum(const Matrix2d& A,const Matrix2d& B, Matrix2d& C)
{
    if(
        !(            
            A.rows    == B.rows     && 
            A.columns == B.columns  &&
            C.rows    == A.rows     &&
            C.columns == A.columns
         )
      )
    {
        std::cerr << "Given matrices' dimensions don't match:"              << std::endl
                  << "param : A (" << A.rows <<" x " << A.columns << ")"    << std::endl
                  << "param : B ("  << B.rows  <<" x " << B.columns  << ")" << std::endl
                  << "param : C ("  << C.rows  <<" x " << C.columns  << ")" << std::endl;
        return -1;
    }

    for(int i = 0; i < C.rows;i++)
    {
        for(int j = 0;j < C.columns;j++)
        {
            C.data[i*C.columns + j] = A.data[i*C.columns + j] + B.data[i*C.columns + j]; 
        }
    }

    return 1;
}

int matrix_diff(const Matrix2d& A,const Matrix2d& B, Matrix2d& C)
{
    if(
        !(            
            A.rows    == B.rows     && 
            A.columns == B.columns  &&
            C.rows    == A.rows     &&
            C.columns == A.columns
         )
      )
    {
        std::cerr << "Given matrices' dimensions don't match:"              << std::endl
                  << "param : A (" << A.rows <<" x " << A.columns << ")"    << std::endl
                  << "param : B ("  << B.rows  <<" x " << B.columns  << ")" << std::endl
                  << "param : C ("  << C.rows  <<" x " << C.columns  << ")" << std::endl;
        return -1;
    }

    for(int i = 0; i < C.rows;i++)
    {
        for(int j = 0;j < C.columns;j++)
        {
            C.data[i*C.columns + j] = A.data[i*C.columns + j] - B.data[i*C.columns + j]; 
        }
    }

    return 1;
}