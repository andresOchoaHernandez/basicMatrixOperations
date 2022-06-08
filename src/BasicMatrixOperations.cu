#include <iostream>

#include "BasicMatrixOperations.cuh"

__global__
void matrix_multiplication_kernel(const double *A,const int A_columns_B_rows,const double *B,double *C,const int C_rows,const int C_columns)
{
    int row    = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= C_rows || column >= C_columns) return;

    double sum = 0;
    for(int k = 0; k < A_columns_B_rows;k++)
    {
        sum += A[row * A_columns_B_rows + k] * B[k * C_columns + column]; 
    }
 
    C[row * C_columns + column] = sum;
}
__host__
int gpu_matrix_multiplication(const Matrix2d& A, const Matrix2d& B,Matrix2d& C)
{
    if( (A.columns != B.rows) || (A.rows != C.rows) || ( B.columns != C.columns) )
    {
        std::cerr << "Given matrices' dimensions don't match:"           << std::endl
                  << "param : A (" << A.rows <<" x " << A.columns << ")" << std::endl
                  << "param : B (" << B.rows <<" x " << B.columns << ")" << std::endl
                  << "param : C (" << C.rows <<" x " << C.columns << ")" << std::endl;
        return -1;
    }

    double *d_A,*d_B,*d_C;

    cudaMalloc(&d_A,static_cast<size_t>(A.rows * A.columns * sizeof(double)));
    cudaMalloc(&d_B,static_cast<size_t>(B.rows * B.columns * sizeof(double)));
    cudaMalloc(&d_C,static_cast<size_t>(C.rows * C.columns * sizeof(double)));

    cudaMemcpy(d_A,A.data,static_cast<size_t>(A.rows * A.columns * sizeof(double)),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B.data,static_cast<size_t>(B.rows * B.columns * sizeof(double)),cudaMemcpyHostToDevice);

    dim3 blockDim(32,32,1);
    dim3 gridDim((C.columns + 32 -1)/32 + 1,(C.rows + 32 -1)/32 + 1,1);

    matrix_multiplication_kernel<<<gridDim,blockDim>>>(d_A,A.columns,d_B,d_C,C.rows,C.columns);

    cudaDeviceSynchronize();

    cudaMemcpy(C.data,d_C,static_cast<size_t>(C.rows * C.columns * sizeof(double)),cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset();

    return 1;
}

__global__
void matrix_transpose_kernel(const double *A,double *A_t,const int A_rows,const int A_columns)
{
    int row    = blockIdx.y * blockDim.y + threadIdx.y; 
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= A_rows || column >= A_columns) return;

    A_t[column*A_rows + row] = A[row*A_columns + column];
}

__host__
int gpu_matrix_transpose(const Matrix2d& A,Matrix2d& A_t)
{
    if( (A.rows != A_t.columns) || (A.columns != A_t.rows) )
    {
        std::cerr << "Given matrices' dimensions don't match:"                 << std::endl
                  << "param A  : (" << A.rows << " x " << A.columns << ")"     << std::endl
                  << "param A_t: (" << A_t.rows << " x " << A_t.columns << ")" << std::endl;
        return -1;
    }

    double *d_A,*d_A_t;

    cudaMalloc(&d_A,static_cast<size_t>(A.rows * A.columns * sizeof(double)));
    cudaMalloc(&d_A_t,static_cast<size_t>(A_t.rows * A_t.columns * sizeof(double)));

    cudaMemcpy(d_A,A.data,static_cast<size_t>(A.rows * A.columns * sizeof(double)),cudaMemcpyHostToDevice);

    dim3 blockDim(32,32,1);
    dim3 gridDim((A.columns + 32 -1)/32 + 1,(A.rows + 32 -1)/32 + 1,1);

    matrix_transpose_kernel<<<gridDim,blockDim>>>(d_A,d_A_t,A.rows,A.columns);

    cudaDeviceSynchronize();

    cudaMemcpy(A_t.data,d_A_t,static_cast<size_t>(A_t.rows * A_t.columns * sizeof(double)),cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_A_t);

    cudaDeviceReset();

    return 1;
}


__global__
void matrix_dot_product_kernel(const double *A, const double *B,double *C, const int C_rows,const int C_columns)
{
    int row    = blockIdx.y * blockDim.y + threadIdx.y; 
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= C_rows || column >= C_columns) return;

    C[row * C_columns + column] = A[row*C_columns+column] * B[row*C_columns+column];
}

__host__
int gpu_matrix_dot_product(const Matrix2d& A, const Matrix2d& B,Matrix2d& C)
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

        std::cerr << "Given matrices' dimensions don't match:"           << std::endl
                  << "param : A (" << A.rows <<" x " << A.columns << ")" << std::endl
                  << "param : B (" << B.rows <<" x " << B.columns << ")" << std::endl
                  << "param : C (" << C.rows <<" x " << C.columns << ")" << std::endl;
        return -1;
    }

    double *d_A,*d_B,*d_C;

    cudaMalloc(&d_A,static_cast<size_t>(A.rows * A.columns * sizeof(double)));
    cudaMalloc(&d_B,static_cast<size_t>(B.rows * B.columns * sizeof(double)));
    cudaMalloc(&d_C,static_cast<size_t>(C.rows * C.columns * sizeof(double)));

    cudaMemcpy(d_A,A.data,static_cast<size_t>(A.rows * A.columns * sizeof(double)),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B.data,static_cast<size_t>(B.rows * B.columns * sizeof(double)),cudaMemcpyHostToDevice);

    dim3 blockDim(32,32,1);
    dim3 gridDim((C.columns + 32 -1)/32 + 1,(C.rows + 32 -1)/32 + 1,1);

    matrix_dot_product_kernel<<<gridDim,blockDim>>>(d_A,d_B,d_C,C.rows,C.columns);

    cudaDeviceSynchronize();

    cudaMemcpy(C.data,d_C,static_cast<size_t>(C.rows * C.columns * sizeof(double)),cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset();

    return 1;
}


__global__
void scalar_matrix_dot_product_kernel(const double scalar, const double *A,double *C, const int C_rows,const int C_columns)
{
    int row    = blockIdx.y * blockDim.y + threadIdx.y; 
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= C_rows || column >= C_columns) return;

    C[row * C_columns + column] = scalar * A[row*C_columns+column];
}

__host__
int gpu_scalar_matrix_dot_product(const double scalar, const Matrix2d& A,Matrix2d& C)
{
    if(!(C.rows    == A.rows && C.columns == A.columns))
    {
        std::cerr << "Given matrices' dimensions don't match:"           << std::endl
                  << "param : A (" << A.rows <<" x " << A.columns << ")" << std::endl
                  << "param : C (" << C.rows <<" x " << C.columns << ")" << std::endl;
        return -1;
    }

    double *d_A,*d_C;

    cudaMalloc(&d_A,static_cast<size_t>(A.rows * A.columns * sizeof(double)));
    cudaMalloc(&d_C,static_cast<size_t>(C.rows * C.columns * sizeof(double)));

    cudaMemcpy(d_A,A.data,static_cast<size_t>(A.rows * A.columns * sizeof(double)),cudaMemcpyHostToDevice);

    dim3 blockDim(32,32,1);
    dim3 gridDim((C.columns + 32 -1)/32 + 1,(C.rows + 32 -1)/32 + 1,1);

    scalar_matrix_dot_product_kernel<<<gridDim,blockDim>>>(scalar,d_A,d_C,C.rows,C.columns);

    cudaDeviceSynchronize();

    cudaMemcpy(C.data,d_C,static_cast<size_t>(C.rows * C.columns * sizeof(double)),cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);

    cudaDeviceReset();

    return 1;
}
__global__
void matrix_sum_kernel(const double *A,const double *B, double *C,const int C_rows,const int C_columns)
{
    int row    = blockIdx.y * blockDim.y + threadIdx.y; 
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= C_rows || column >= C_columns) return;

    C[row * C_columns + column] = A[row*C_columns+column] + B[row*C_columns+column];
}

__host__
int gpu_matrix_sum(const Matrix2d& A,const Matrix2d& B, Matrix2d& C)
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

        std::cerr << "Given matrices' dimensions don't match:"           << std::endl
                  << "param : A (" << A.rows <<" x " << A.columns << ")" << std::endl
                  << "param : B (" << B.rows <<" x " << B.columns << ")" << std::endl
                  << "param : C (" << C.rows <<" x " << C.columns << ")" << std::endl;
        return -1;
    }

    double *d_A,*d_B,*d_C;

    cudaMalloc(&d_A,static_cast<size_t>(A.rows * A.columns * sizeof(double)));
    cudaMalloc(&d_B,static_cast<size_t>(B.rows * B.columns * sizeof(double)));
    cudaMalloc(&d_C,static_cast<size_t>(C.rows * C.columns * sizeof(double)));

    cudaMemcpy(d_A,A.data,static_cast<size_t>(A.rows * A.columns * sizeof(double)),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B.data,static_cast<size_t>(B.rows * B.columns * sizeof(double)),cudaMemcpyHostToDevice);

    dim3 blockDim(32,32,1);
    dim3 gridDim((C.columns + 32 -1)/32 + 1,(C.rows + 32 -1)/32 + 1,1);

    matrix_sum_kernel<<<gridDim,blockDim>>>(d_A,d_B,d_C,C.rows,C.columns);

    cudaDeviceSynchronize();

    cudaMemcpy(C.data,d_C,static_cast<size_t>(C.rows * C.columns * sizeof(double)),cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset();

    return 1;
}
__global__
void matrix_diff_kernel(const double *A,const double *B, double *C,const int C_rows,const int C_columns)
{
    int row    = blockIdx.y * blockDim.y + threadIdx.y; 
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= C_rows || column >= C_columns) return;

    C[row * C_columns + column] = A[row*C_columns+column] - B[row*C_columns+column];
}

__host__
int gpu_matrix_diff(const Matrix2d& A,const Matrix2d& B, Matrix2d& C)
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

        std::cerr << "Given matrices' dimensions don't match:"           << std::endl
                  << "param : A (" << A.rows <<" x " << A.columns << ")" << std::endl
                  << "param : B (" << B.rows <<" x " << B.columns << ")" << std::endl
                  << "param : C (" << C.rows <<" x " << C.columns << ")" << std::endl;
        return -1;
    }

    double *d_A,*d_B,*d_C;

    cudaMalloc(&d_A,static_cast<size_t>(A.rows * A.columns * sizeof(double)));
    cudaMalloc(&d_B,static_cast<size_t>(B.rows * B.columns * sizeof(double)));
    cudaMalloc(&d_C,static_cast<size_t>(C.rows * C.columns * sizeof(double)));

    cudaMemcpy(d_A,A.data,static_cast<size_t>(A.rows * A.columns * sizeof(double)),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B.data,static_cast<size_t>(B.rows * B.columns * sizeof(double)),cudaMemcpyHostToDevice);

    dim3 blockDim(32,32,1);
    dim3 gridDim((C.columns + 32 -1)/32 + 1,(C.rows + 32 -1)/32 + 1,1);

    matrix_diff_kernel<<<gridDim,blockDim>>>(d_A,d_B,d_C,C.rows,C.columns);

    cudaDeviceSynchronize();

    cudaMemcpy(C.data,d_C,static_cast<size_t>(C.rows * C.columns * sizeof(double)),cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset();

    return 1;
}