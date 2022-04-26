#pragma once

void print_matrix(float * A,int n,int m);

int matrixMultiplication(float* matrix_A,int rows_A,int columns_A,
                          float* matrix_B,int rows_B,int columns_B,
                          float* matrix_C,int rows_C,int columns_C);

int matrixTranspose(float * A,int n,int m,float * A_transpose,int n_t,int m_t);