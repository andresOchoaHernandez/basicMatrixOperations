#include <iostream>
#include <limits>
#include <cmath>
#include "BasicMatrixOperations.hpp"

float avg_cross_entropy(Matrix2d outputs, Matrix2d labels)
{
    if(outputs.rows != labels.rows || outputs.columns != labels.columns)
    {
        std::cerr << "Given matrices' dimensions don't match:"                             << std::endl
                  << "param : outputs (" << outputs.rows <<" x " << outputs.columns << ")" << std::endl
                  << "param : labels ("  << labels.rows  <<" x " << labels.columns  << ")" << std::endl;
        return std::numeric_limits<float>::quiet_NaN();
    }

    float *losses = new float[outputs.columns];

    for(int _example = 0; _example < outputs.columns; _example++)
    {
        float sum = 0;
        for(int _class = 0; _class < outputs.rows; _class++)
        {
            sum+= labels.data[_class * outputs.columns + _example] * std::log2(outputs.data[_class * outputs.columns + _example]);
        }
        losses[_example] = -sum;
    }

    float lossSum = 0;
    for(int _example = 0; _example < outputs.columns; _example++)
    {
        lossSum += losses[_example];
    }

    delete[] losses;

    return lossSum / static_cast<float>(outputs.columns);
}

int main(void)
{
    Matrix2d output;
    output.rows    = 3;
    output.columns = 3;
    output.data = new float [3*3];

    output.data[0] =0.3f;
    output.data[1] =0.1f;
    output.data[2] =0.5f;
    output.data[3] =0.2f;
    output.data[4] =0.1f;
    output.data[5] =0.4f;
    output.data[6] =0.5f;
    output.data[7] =0.8f;
    output.data[8] =0.1f;

    Matrix2d labels;
    labels.rows    = 3;
    labels.columns = 3;
    labels.data = new float [3*3];

    labels.data[0] =0.0f;
    labels.data[1] =0.0f;
    labels.data[2] =0.0f;
    labels.data[3] =0.0f;
    labels.data[4] =0.0f;
    labels.data[5] =0.0f;
    labels.data[6] =1.0f;
    labels.data[7] =1.0f;
    labels.data[8] =1.0f;

    float result = avg_cross_entropy(output,labels);

    std:: cout << "calculated cross entropy: " << result << std::endl;

    delete[] output.data;
    delete[] labels.data; 

    return 0;
}