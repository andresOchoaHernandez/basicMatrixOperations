#pragma once

struct Matrix2d
{
    double *data;
    int rows;
    int columns;
};

void print_matrix( const Matrix2d& A);

int matrixMultiplication(const Matrix2d& A, const Matrix2d& B,Matrix2d& C);

int matrixTranspose(const Matrix2d& A,Matrix2d& A_t);