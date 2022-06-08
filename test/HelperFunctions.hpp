#pragma once
#include "BasicMatrixOperations.hpp"

void one_initialization(Matrix2d &A);
void zero_initialization(Matrix2d &A);
void fill_matrix(Matrix2d &A);
int check_if_correct(const Matrix2d &A,const Matrix2d &A_t);
void uniform_initialization(Matrix2d& D);
bool are_double_nearly_equal(const double a,const double b);