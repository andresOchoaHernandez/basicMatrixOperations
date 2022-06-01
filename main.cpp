#include <iostream>
#include <limits>
#include <cmath>
#include "BasicMatrixOperations.hpp"

void sigmoid(const Matrix2d &y, Matrix2d &z)
{
    if(y.rows != z.rows || y.columns != z.columns)
    {
        std::cerr << "Given matrices' dimensions don't match:"              << std::endl
                  << "param : y (" << y.rows <<" x " << y.columns << ")"    << std::endl
                  << "param : z ("  << z.rows  <<" x " << z.columns  << ")" << std::endl;
        return;
    }

    for(int _class = 0; _class < y.rows;_class++)
    {
        for(int _example = 0;_example < y.columns;_example++)
        {
            z.data[_class*y.columns + _example] = 1.0 / (1.0 + std::exp(-y.data[_class*y.columns + _example]));
        }
    }
}

void softmax(const Matrix2d &y,Matrix2d &z)
{
    if(y.rows != z.rows || y.columns != z.columns)
    {
        std::cerr << "Given matrices' dimensions don't match:"              << std::endl
                  << "param : y (" << y.rows <<" x " << y.columns << ")"    << std::endl
                  << "param : z ("  << z.rows  <<" x " << z.columns  << ")" << std::endl;
        return;
    }
    
    for(int _example = 0;_example < y.columns;_example++)
    {
        double denominator = 0;
        for(int _class = 0;_class < y.rows; _class++)
        {
            denominator += std::exp(y.data[_class*y.columns + _example]);
        }

        for(int _class = 0;_class < y.rows; _class++)
        {
            z.data[_class*z.columns + _example] = std::exp(y.data[_class*y.columns + _example])/denominator;
        }
    }
}

double avg_cross_entropy(const Matrix2d &outputs, const Matrix2d &labels)
{
    if(outputs.rows != labels.rows || outputs.columns != labels.columns)
    {
        std::cerr << "Given matrices' dimensions don't match:"                             << std::endl
                  << "param : outputs (" << outputs.rows <<" x " << outputs.columns << ")" << std::endl
                  << "param : labels ("  << labels.rows  <<" x " << labels.columns  << ")" << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }

    double *losses = new double[outputs.columns];

    for(int _example = 0; _example < outputs.columns; _example++)
    {
        double sum = 0;
        for(int _class = 0; _class < outputs.rows; _class++)
        {
            sum+= labels.data[_class * outputs.columns + _example] * std::log2(outputs.data[_class * outputs.columns + _example]);
        }
        losses[_example] = -sum;
    }

    double lossSum = 0;
    for(int _example = 0; _example < outputs.columns; _example++)
    {
        lossSum += losses[_example];
    }

    delete[] losses;

    return lossSum / static_cast<double>(outputs.columns);
}

double forward_propagation(
                   const Matrix2d& X,
                   const Matrix2d& labels,
                   const Matrix2d& W1,
                   const Matrix2d& b1,
                   const Matrix2d& W2,
                   const Matrix2d& b2,
                   const Matrix2d& W3,
                   const Matrix2d& b3
                )
{

    /*  FIRST LAYER */
    Matrix2d W1_t;
    W1_t.rows    = W1.columns;
    W1_t.columns = W1.rows;
    W1_t.data = new double[W1_t.rows * W1_t.columns];

    matrixTranspose(W1,W1_t);

    Matrix2d Y1;
    Y1.rows    = W1_t.rows;
    Y1.columns = X.columns;
    Y1.data = new double[Y1.rows * Y1.columns];

    Matrix2d W1_t_x_X;
    W1_t_x_X.rows    = Y1.rows;
    W1_t_x_X.columns = Y1.columns;
    W1_t_x_X.data = new double[W1_t_x_X.rows*W1_t_x_X.columns];

    matrixMultiplication(W1_t,X,W1_t_x_X);

    matrix_sum(W1_t_x_X,b1,Y1);

    Matrix2d Z1;
    Z1.rows    = Y1.rows;
    Z1.columns = Y1.columns;
    Z1.data = new double[Z1.rows * Z1.columns]; 
    
    sigmoid(Y1,Z1);

    print_matrix(Y1);
    print_matrix(Z1);

    /*  SECOND LAYER */
    Matrix2d W2_t;
    W2_t.rows    = W2.columns;
    W2_t.columns = W2.rows;
    W2_t.data = new double[W2_t.rows * W2_t.columns];

    matrixTranspose(W2,W2_t);

    Matrix2d Y2;
    Y2.rows    = W2_t.rows;
    Y2.columns = Z1.columns;
    Y2.data = new double[Y2.rows * Y2.columns];

    Matrix2d W2_t_x_Z1;
    W2_t_x_Z1.rows    = Y2.rows;
    W2_t_x_Z1.columns = Y2.columns;
    W2_t_x_Z1.data = new double[W2_t_x_Z1.rows*W2_t_x_Z1.columns];

    matrixMultiplication(W2_t,Z1,W2_t_x_Z1);

    matrix_sum(W2_t_x_Z1,b2,Y2);

    Matrix2d Z2;
    Z2.rows    = Y2.rows;
    Z2.columns = Y2.columns;
    Z2.data = new double[Z2.rows * Z2.columns]; 
    
    sigmoid(Y2,Z2);

    print_matrix(Y2);
    print_matrix(Z2);

    /*  THIRD LAYER */
    Matrix2d W3_t;
    W3_t.rows    = W3.columns;
    W3_t.columns = W3.rows;
    W3_t.data = new double[W3_t.rows * W3_t.columns];

    matrixTranspose(W3,W3_t);

    Matrix2d Y3;
    Y3.rows    = W3_t.rows;
    Y3.columns = Z2.columns;
    Y3.data = new double[Y3.rows * Y3.columns];

    Matrix2d W3_t_x_Z2;
    W3_t_x_Z2.rows    = Y3.rows;
    W3_t_x_Z2.columns = Y3.columns;
    W3_t_x_Z2.data = new double[W3_t_x_Z2.rows*W3_t_x_Z2.columns];

    matrixMultiplication(W3_t,Z2,W3_t_x_Z2);

    matrix_sum(W3_t_x_Z2,b3,Y3);

    Matrix2d Z3;
    Z3.rows    = Y3.rows;
    Z3.columns = Y3.columns;
    Z3.data = new double[Z3.rows * Z3.columns]; 
    
    softmax(Y3,Z3);

    print_matrix(Y3);
    print_matrix(Z3);

    double loss = avg_cross_entropy(Z3,labels);

    delete[] W1_t.data;
    delete[] Y1.data;
    delete[] W1_t_x_X.data;
    delete[] Z1.data;

    delete[] W2_t.data;
    delete[] Y2.data;
    delete[] W2_t_x_Z1.data;
    delete[] Z2.data;

    delete[] W3_t.data;
    delete[] Y3.data;
    delete[] W3_t_x_Z2.data;
    delete[] Z3.data;

    return loss;
}

int main(void)
{
    Matrix2d X;
    X.rows    = 4;
    X.columns = 3;
    X.data = new double [4*3];

    X.data[0]  = 3;
    X.data[1]  = 6;
    X.data[2]  = 4;
    X.data[3]  = 4;
    X.data[4]  = 1;
    X.data[5]  = 3;
    X.data[6]  = 2;
    X.data[7]  = 3;
    X.data[8]  = 2;
    X.data[9]  = 5;
    X.data[10] = 9;
    X.data[11] = 5;

    Matrix2d W1;
    W1.rows    = 4;
    W1.columns = 3;
    W1.data = new double [4*3];

    W1.data[0]  = 1;
    W1.data[1]  = 1;
    W1.data[2]  = 1;
    W1.data[3]  = 1;
    W1.data[4]  = 1;
    W1.data[5]  = 1;
    W1.data[6]  = 1;
    W1.data[7]  = 1;
    W1.data[8]  = 1;
    W1.data[9]  = 1;
    W1.data[10] = 1;
    W1.data[11] = 1;

    Matrix2d b1;
    b1.rows    = 3;
    b1.columns = 3;
    b1.data = new double [3*3];

    b1.data[0]  = 1;
    b1.data[1]  = 1;
    b1.data[2]  = 1;
    b1.data[3]  = 1;
    b1.data[4]  = 1;
    b1.data[5]  = 1;
    b1.data[6]  = 1;
    b1.data[7]  = 1;
    b1.data[8]  = 1;

    Matrix2d W2;
    W2.rows    = 3;
    W2.columns = 2;
    W2.data = new double [3*2];

    W2.data[0]  = 1;
    W2.data[1]  = 1;
    W2.data[2]  = 1;
    W2.data[3]  = 1;
    W2.data[4]  = 1;
    W2.data[5]  = 1;

    Matrix2d b2;
    b2.rows    = 2;
    b2.columns = 3;
    b2.data = new double [2*3];

    b2.data[0]  = 1;
    b2.data[1]  = 1;
    b2.data[2]  = 1;
    b2.data[3]  = 1;
    b2.data[4]  = 1;
    b2.data[5]  = 1;


    Matrix2d W3;
    W3.rows    = 2;
    W3.columns = 3;
    W3.data = new double [2*3];

    W3.data[0]  = 1;
    W3.data[1]  = 1;
    W3.data[2]  = 1;
    W3.data[3]  = 1;
    W3.data[4]  = 1;
    W3.data[5]  = 1;

    Matrix2d b3;
    b3.rows    = 3;
    b3.columns = 3;
    b3.data = new double [3*3];

    b3.data[0]  = 1;
    b3.data[1]  = 1;
    b3.data[2]  = 1;
    b3.data[3]  = 1;
    b3.data[4]  = 1;
    b3.data[5]  = 1;
    b3.data[6]  = 1;
    b3.data[7]  = 1;
    b3.data[8]  = 1;

    Matrix2d labels;
    labels.rows    = 3;
    labels.columns = 3;
    labels.data = new double [3*3];

    labels.data[0]  = 1;
    labels.data[1]  = 0;
    labels.data[2]  = 0;
    labels.data[3]  = 0;
    labels.data[4]  = 1;
    labels.data[5]  = 0;
    labels.data[6]  = 0;
    labels.data[7]  = 0;
    labels.data[8]  = 1;

    double loss = forward_propagation(X,labels,W1,b1,W2,b2,W3,b3);

    std::printf("Forward propagation result (loss): %f\n",loss);

    delete[] X.data;
    delete[] W1.data;
    delete[] b1.data;

    delete[] W2.data;
    delete[] b2.data;

    delete[] W3.data;
    delete[] b3.data;

    delete[] labels.data;

    return 0;
}