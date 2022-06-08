#include <iostream>
#include <limits>

#include "BasicMatrixOperations.cuh"
#include "HelperFunctions.hpp"

const int NA = 509;
const int MA = 675;
const int NB = MA;
const int MB = 413;

bool are_double_nearly_equal(const double a,const double b)
{
    if(std::abs(a-b) < 0.00001)
    {
        return true;
    }

    return false;
}

int gpu_matrix_mul_test_one()
{
    Matrix2d A;
    A.rows    = NA;
    A.columns = MA;
    A.data = new double[A.rows*A.columns];
    uniform_initialization(A);

    Matrix2d B;
    B.rows = NB;
    B.columns = MB;
    B.data = new double[B.rows*B.columns];
    uniform_initialization(B);

    Matrix2d C_host;
    C_host.rows    = A.rows;
    C_host.columns = B.columns;
    C_host.data = new double[C_host.rows*C_host.columns];

    matrix_multiplication(A,B,C_host);

    Matrix2d C_device;
    C_device.rows    = A.rows;
    C_device.columns = B.columns;
    C_device.data = new double[C_device.rows*C_device.columns];

    gpu_matrix_multiplication(A,B,C_device);

    int correct = 0;

    for(int i = 0; i < C_host.rows; i++)
    {
        for(int j = 0; j < C_host.columns; j++ )
        {
            if(!are_double_nearly_equal(C_host.data[i*C_host.columns + j], C_device.data[i*C_device.columns + j]))
            {
                std::cerr << "found error in position: (" << i << " , " << j << ")" << std::endl;
                correct = -1;
                break;
            }
        }
    }

    delete[] A.data;
    delete[] B.data;
    delete[] C_host.data;
    delete[] C_device.data;

    return correct;
}

int main(void)
{

    if(gpu_matrix_mul_test_one() == -1){
        std::cout << "GpuMatrixMulTestOne [FAILED]\n";
        return -1;
    }

    return 0;
}