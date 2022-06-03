#pragma once

struct Matrix2d
{
    double *data;
    int rows;
    int columns;
};

void print_matrix(const Matrix2d& A);

int matrix_multiplication(const Matrix2d& A, const Matrix2d& B,Matrix2d& C);

int matrix_transpose(const Matrix2d& A,Matrix2d& A_t);

int matrix_dot_product(const Matrix2d& A, const Matrix2d& B,Matrix2d& C);

int scalar_matrix_dot_product(const double scalar, const Matrix2d& A,Matrix2d& C);

int matrix_sum(const Matrix2d& A,const Matrix2d& B, Matrix2d& C);

int matrix_diff(const Matrix2d& A,const Matrix2d& B, Matrix2d& C);