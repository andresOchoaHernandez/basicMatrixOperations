#include <iostream>

#include "HelperFunctions.hpp"
#include "BasicMatrixOperations.cuh"

const int N = 100;
const int M = 330;

int gpu_matrix_transpose_test_one()
{
    Matrix2d A;
    A.rows    = N;
    A.columns = M; 

    A.data = new double[A.rows*A.columns];
    fill_matrix(A);

    Matrix2d A_t_host;
    A_t_host.rows    = M;
    A_t_host.columns = N; 

    A_t_host.data = new double[A_t_host.rows*A_t_host.columns]();

    matrix_transpose(A,A_t_host);

    Matrix2d A_t_device;
    A_t_device.rows    = M;
    A_t_device.columns = N; 

    A_t_device.data = new double[A_t_device.rows*A_t_device.columns]();

    gpu_matrix_transpose(A,A_t_device);

    int correct = 0;

    for(int i = 0; i < A_t_host.rows; i++)
    {
        for(int j = 0; j < A_t_host.columns; j++ )
        {
            if(!are_double_nearly_equal(A_t_host.data[i*A_t_host.columns + j], A_t_device.data[i*A_t_device.columns + j]))
            {
                std::cerr << "found error in position: (" << i << " , " << j << ")" << std::endl;
                correct = -1;
                break;
            }
        }
    }

    delete[] A.data;
    delete[] A_t_host.data;
    delete[] A_t_device.data;

    return correct;
}
/* ============================================== */

int main(void)
{
    if(gpu_matrix_transpose_test_one() == -1){
        std::cout << "GpuMatrixTransposeTestOne [FAILED]\n";
        return -1;
    }

    return 0;
}