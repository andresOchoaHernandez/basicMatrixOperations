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
    Matrix2d output;
    output.rows    = 3;
    output.columns = 3;
    output.data = new double [3*3];

    output.data[0] =0.3;
    output.data[1] =0.1;
    output.data[2] =0.5;
    output.data[3] =0.2;
    output.data[4] =0.1;
    output.data[5] =0.4;
    output.data[6] =0.5;
    output.data[7] =0.8;
    output.data[8] =0.1;

    Matrix2d labels;
    labels.rows    = 3;
    labels.columns = 3;
    labels.data = new double [3*3];

    labels.data[0] =0.0;
    labels.data[1] =0.0;
    labels.data[2] =0.0;
    labels.data[3] =0.0;
    labels.data[4] =0.0;
    labels.data[5] =0.0;
    labels.data[6] =1.0;
    labels.data[7] =1.0;
    labels.data[8] =1.0;

    double result = avg_cross_entropy(output,labels);

    std:: cout << "calculated cross entropy: " << result << std::endl;

    Matrix2d y;
    y.rows    = 3;
    y.columns = 3;
    y.data = new double[3*3];

    y.data[0] = 1.0;
    y.data[1] = 9.0;
    y.data[2] =-10.0;
    y.data[3] = 10.0;
    y.data[4] = 2.0;
    y.data[5] =-1.0;
    y.data[6] = 5.0;
    y.data[7] = 2.0;
    y.data[8] = 3.0;

    Matrix2d z;
    z.rows    = 3;
    z.columns = 3;
    z.data = new double[3*3];

    softmax(y,z);

    sigmoid(y,z);

    print_matrix(y);
    print_matrix(labels);
    matrix_diff(y,labels,z);
    print_matrix(z);

    delete[] output.data;
    delete[] labels.data; 
    delete[] y.data;
    delete[] z.data;

    return 0;
}