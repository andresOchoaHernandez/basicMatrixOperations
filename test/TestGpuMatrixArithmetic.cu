#include <iostream>

#include "HelperFunctions.hpp"
#include "BasicMatrixOperations.cuh"

const int N = 1235;
const int M = 5324;

int check_correctness(const Matrix2d& C_host,const Matrix2d& C_device)
{
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

    return correct;
}

int gpu_matrix_arithmetic_test_one()
{
    Matrix2d A;
    A.rows    = N;
    A.columns = M; 

    A.data = new double[A.rows*A.columns];
    uniform_initialization(A);

    Matrix2d B;
    B.rows    = N;
    B.columns = M; 

    B.data = new double[B.rows*B.columns];
    uniform_initialization(B);

    Matrix2d C_host;
    C_host.rows    = N;
    C_host.columns = M; 

    C_host.data = new double[C_host.rows*C_host.columns];

    Matrix2d C_device;
    C_device.rows    = N;
    C_device.columns = M; 

    C_device.data = new double[C_device.rows*C_device.columns];


    matrix_dot_product(A,B,C_host);
    gpu_matrix_dot_product(A,B,C_device);

    int dot_product = check_correctness(C_host,C_device);
    if( dot_product == -1){std::cerr << "Error in dot product" << std::endl;return -1;};

    zero_initialization(C_host);
    zero_initialization(C_device);

    matrix_sum(A,B,C_host);
    gpu_matrix_sum(A,B,C_device);

    int matrix_sum = check_correctness(C_host,C_device);
    if( matrix_sum == -1){std::cerr << "Error in sum" << std::endl;return -1;};

    zero_initialization(C_host);
    zero_initialization(C_device);

    matrix_diff(A,B,C_host);
    gpu_matrix_diff(A,B,C_device);

    int matrix_diff = check_correctness(C_host,C_device);
    if( matrix_diff == -1){std::cerr << "Error in diff" << std::endl;return -1;};

    delete[] A.data;
    delete[] B.data;
    delete[] C_host.data;
    delete[] C_device.data;

    return 0;
}
/* ============================================== */

int main(void)
{
    if(gpu_matrix_arithmetic_test_one() == -1){
        std::cout << "GpuMatrixArithmeticTestOne [FAILED]\n";
        return -1;
    }

    return 0;
}