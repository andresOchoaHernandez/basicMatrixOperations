#include <iostream>
#include <chrono>

#include "BasicMatrixOperations.hpp"

void print_matrix(float * A,int n,int m)
{
    std::cout << "--------------------\n";
    for (int i = 0; i < n;i++)
    {
        for(int j = 0;j< m; j++)
        {
            if(j == m-1)
                std::cout << A[i*m+j];
            else
                std::cout << A[i*m+j]<<", ";
        }
        std::cout << "\n";
    }
    std::cout << "--------------------\n";
}

int matrixMultiplication(float* matrix_A,int rows_A,int columns_A,
                          float* matrix_B,int rows_B,int columns_B,
                          float* matrix_C,int rows_C,int columns_C)
{


    const auto before = std::chrono::high_resolution_clock::now();

    if( columns_A != rows_B ){std::cerr << "matrix_A columns don't match matrix_B rows\n";return -1;}
    if((rows_A != rows_C) || (columns_B != columns_C) ){std::cerr << "given matrix_C dimensions\n";return -1;}

    for(int rows = 0; rows < rows_C;rows++)
    {
        for(int columns = 0; columns < columns_C;columns++)
        {
            float sum = 0;
            for(int k = 0; k < columns_A;k++)
            {
                sum += matrix_A[rows * columns_A + k]*matrix_B[k* columns_B + columns];
            }
            matrix_C[rows * columns_C + columns] = sum;
        }
    }

    const std::chrono::duration<double, std::milli> duration = std::chrono::high_resolution_clock::now() - before;
    std::cout << "matrixMultiplication took: " << duration.count() << " ms" << std::endl;

    return 0;
}

int matrixTranspose(float * A,int n,int m,float * A_transpose,int n_t,int m_t)
{
    const auto before = std::chrono::high_resolution_clock::now();

    if( (n != m_t) || (m != n_t) ){std::cerr << "given matrix's dimensions don't match\n";return -1;}

    for(int i = 0; i < n_t;i++)
    {
        for(int j = 0; j <m_t;j++)
        {
            A_transpose[i*m_t + j] = A[j*m + i];
        }
    }

    const std::chrono::duration<double, std::milli> duration = std::chrono::high_resolution_clock::now() - before;
    std::cout << "matrixTranspose took: " << duration.count() << " ms" << std::endl;

    return 0;
}