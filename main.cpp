#include <iostream>
#include <limits>
#include <cmath>
#include "BasicMatrixOperations.hpp"

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
    
    print_matrix(y);

    Matrix2d z;
    z.rows    = 3;
    z.columns = 3;
    z.data = new double[3*3];

    softmax(y,z);

    print_matrix(z);

    delete[] output.data;
    delete[] labels.data; 
    delete[] y.data;
    delete[] z.data;

    return 0;
}