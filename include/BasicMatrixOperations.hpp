#pragma once

struct Matrix2d
{
    float *data;
    int rows;
    int columns;
};

void print_matrix(Matrix2d* A);

int matrixMultiplication(Matrix2d* A,Matrix2d* B,Matrix2d* C);

int matrixTranspose(Matrix2d* A,Matrix2d* A_t);