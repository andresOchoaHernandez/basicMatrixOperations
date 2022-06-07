#pragma once

#include "BasicMatrixOperations.hpp"

int gpu_matrix_multiplication(const Matrix2d& A, const Matrix2d& B,Matrix2d& C);

int gpu_matrix_transpose(const Matrix2d& A,Matrix2d& A_t);

int gpu_matrix_dot_product(const Matrix2d& A, const Matrix2d& B,Matrix2d& C);

int gpu_scalar_matrix_dot_product(const double scalar, const Matrix2d& A,Matrix2d& C);

int gpu_matrix_sum(const Matrix2d& A,const Matrix2d& B, Matrix2d& C);

int gpu_matrix_diff(const Matrix2d& A,const Matrix2d& B, Matrix2d& C);