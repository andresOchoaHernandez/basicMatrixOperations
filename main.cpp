#include <iostream>
#include <limits>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>

#include "BasicMatrixOperations.hpp"

#ifdef USE_GPU_FUNCTIONS
    #include "BasicMatrixOperations.cuh"
#endif

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
                         Matrix2d& Z1,
                   const Matrix2d& W2,
                   const Matrix2d& b2,
                         Matrix2d& Z2,
                   const Matrix2d& W3,
                   const Matrix2d& b3,
                         Matrix2d& Z3
                )
{    
    /*  FIRST LAYER */
    Matrix2d W1_t;
    W1_t.rows    = W1.columns;
    W1_t.columns = W1.rows;
    W1_t.data = new double[W1_t.rows * W1_t.columns];

    matrix_transpose(W1,W1_t);

    Matrix2d Y1;
    Y1.rows    = W1_t.rows;
    Y1.columns = X.columns;
    Y1.data = new double[Y1.rows * Y1.columns];

    Matrix2d W1_t_x_X;
    W1_t_x_X.rows    = Y1.rows;
    W1_t_x_X.columns = Y1.columns;
    W1_t_x_X.data = new double[W1_t_x_X.rows*W1_t_x_X.columns];

    matrix_multiplication(W1_t,X,W1_t_x_X);

    matrix_sum(W1_t_x_X,b1,Y1);
    
    sigmoid(Y1,Z1);

    /*  SECOND LAYER */
    Matrix2d W2_t;
    W2_t.rows    = W2.columns;
    W2_t.columns = W2.rows;
    W2_t.data = new double[W2_t.rows * W2_t.columns];

    matrix_transpose(W2,W2_t);

    Matrix2d Y2;
    Y2.rows    = W2_t.rows;
    Y2.columns = Z1.columns;
    Y2.data = new double[Y2.rows * Y2.columns];

    Matrix2d W2_t_x_Z1;
    W2_t_x_Z1.rows    = Y2.rows;
    W2_t_x_Z1.columns = Y2.columns;
    W2_t_x_Z1.data = new double[W2_t_x_Z1.rows*W2_t_x_Z1.columns];

    matrix_multiplication(W2_t,Z1,W2_t_x_Z1);

    matrix_sum(W2_t_x_Z1,b2,Y2);
    
    sigmoid(Y2,Z2);

    /*  THIRD LAYER */
    Matrix2d W3_t;
    W3_t.rows    = W3.columns;
    W3_t.columns = W3.rows;
    W3_t.data = new double[W3_t.rows * W3_t.columns];

    matrix_transpose(W3,W3_t);

    Matrix2d Y3;
    Y3.rows    = W3_t.rows;
    Y3.columns = Z2.columns;
    Y3.data = new double[Y3.rows * Y3.columns];

    Matrix2d W3_t_x_Z2;
    W3_t_x_Z2.rows    = Y3.rows;
    W3_t_x_Z2.columns = Y3.columns;
    W3_t_x_Z2.data = new double[W3_t_x_Z2.rows*W3_t_x_Z2.columns];

    matrix_multiplication(W3_t,Z2,W3_t_x_Z2);

    matrix_sum(W3_t_x_Z2,b3,Y3);
    
    softmax(Y3,Z3);

    double loss = avg_cross_entropy(Z3,labels);

    delete[] W1_t.data;
    delete[] Y1.data;
    delete[] W1_t_x_X.data;

    delete[] W2_t.data;
    delete[] Y2.data;
    delete[] W2_t_x_Z1.data;

    delete[] W3_t.data;
    delete[] Y3.data;
    delete[] W3_t_x_Z2.data;

    return loss;
}

void uniform_initializazion(Matrix2d& D)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distr(0,1);

    for(int i = 0; i < D.rows;i++)
    {
        for(int j = 0; j < D.columns;j++)
        {
            D.data[i*D.columns + j] = distr(gen);
        }
    }
}

void one_initializazion(Matrix2d& D)
{
    for(int i = 0; i < D.rows;i++)
    {
        for(int j = 0; j < D.columns;j++)
        {
            D.data[i*D.columns + j] = 1;
        }
    }
}

void gradient_descent(
    const Matrix2d& X,
    const Matrix2d& labels,
    const int iterations,
    const double learning_rate,
    Matrix2d& W1,
    Matrix2d& b1,
    Matrix2d& W2,
    Matrix2d& b2,
    Matrix2d& W3,
    Matrix2d& b3
)
{
    uniform_initializazion(W1);
    uniform_initializazion(b1);
    
    uniform_initializazion(W2);
    uniform_initializazion(b2);
    
    uniform_initializazion(W3);
    uniform_initializazion(b3);

    Matrix2d Z1;
    Z1.rows = b1.rows;
    Z1.columns = b1.columns;
    Z1.data = new double[Z1.rows * Z1.columns];

    Matrix2d Z2;
    Z2.rows = b2.rows;
    Z2.columns = b2.columns;
    Z2.data = new double[Z2.rows * Z2.columns];

    Matrix2d Z3;
    Z3.rows = b3.rows;
    Z3.columns = b3.columns;
    Z3.data = new double[Z3.rows * Z3.columns];

    // cached matrices
    Matrix2d dL_dY3;
    dL_dY3.rows = Z3.rows;
    dL_dY3.columns = Z3.columns;
    dL_dY3.data = new double[dL_dY3.rows*dL_dY3.columns];

    Matrix2d dL_dY2;
    dL_dY2.rows = Z2.rows;
    dL_dY2.columns = Z2.columns;
    dL_dY2.data = new double[dL_dY2.rows*dL_dY2.columns];

    Matrix2d dL_dY1;
    dL_dY1.rows = Z1.rows;
    dL_dY1.columns = Z1.columns;
    dL_dY1.data = new double[dL_dY1.rows*dL_dY1.columns];

    // support matrices for calculating dL_dY2
    Matrix2d one_col_vec_Z2;
    one_col_vec_Z2.rows    = Z2.rows;
    one_col_vec_Z2.columns = Z2.columns;
    one_col_vec_Z2.data = new double[one_col_vec_Z2.rows * one_col_vec_Z2.columns];
    one_initializazion(one_col_vec_Z2);

    Matrix2d diff_1_Z2;
    diff_1_Z2.rows    = Z2.rows;
    diff_1_Z2.columns = Z2.columns;
    diff_1_Z2.data = new double[diff_1_Z2.rows * diff_1_Z2.columns]; 

    Matrix2d Z2_dot_diff_1_Z2;
    Z2_dot_diff_1_Z2.rows    = Z2.rows;
    Z2_dot_diff_1_Z2.columns = Z2.columns;
    Z2_dot_diff_1_Z2.data = new double[Z2_dot_diff_1_Z2.rows * Z2_dot_diff_1_Z2.columns]; 

    Matrix2d W3_mmul_dL_dY3;
    W3_mmul_dL_dY3.rows    = W3.rows;
    W3_mmul_dL_dY3.columns = dL_dY3.columns;
    W3_mmul_dL_dY3.data = new double[W3_mmul_dL_dY3.rows * W3_mmul_dL_dY3.columns];

    // support matrices for calculating dL_dY1
    Matrix2d one_col_vec_Z1;
    one_col_vec_Z1.rows    = Z1.rows;
    one_col_vec_Z1.columns = Z1.columns;
    one_col_vec_Z1.data = new double[one_col_vec_Z1.rows * one_col_vec_Z1.columns];
    one_initializazion(one_col_vec_Z1);

    Matrix2d diff_1_Z1;
    diff_1_Z1.rows    = Z1.rows;
    diff_1_Z1.columns = Z1.columns;
    diff_1_Z1.data = new double[diff_1_Z1.rows * diff_1_Z1.columns]; 

    Matrix2d Z1_dot_diff_1_Z1;
    Z1_dot_diff_1_Z1.rows    = Z1.rows;
    Z1_dot_diff_1_Z1.columns = Z1.columns;
    Z1_dot_diff_1_Z1.data = new double[Z1_dot_diff_1_Z1.rows * Z1_dot_diff_1_Z1.columns];

    Matrix2d W2_mmul_dL_dY2;
    W2_mmul_dL_dY2.rows    = W2.rows;
    W2_mmul_dL_dY2.columns = dL_dY2.columns;
    W2_mmul_dL_dY2.data = new double[W2_mmul_dL_dY2.rows * W2_mmul_dL_dY2.columns];

    // update matrices
    Matrix2d dL_dW3;
    dL_dW3.rows = W3.columns;
    dL_dW3.columns = W3.rows;
    dL_dW3.data = new double[dL_dW3.rows*dL_dW3.columns];

    Matrix2d Z2_t;
    Z2_t.rows = Z2.columns;
    Z2_t.columns = Z2.rows;
    Z2_t.data = new double[Z2_t.rows*Z2_t.columns];

    Matrix2d dL_dW2;
    dL_dW2.rows = W2.columns;
    dL_dW2.columns = W2.rows;
    dL_dW2.data = new double[dL_dW2.rows*dL_dW2.columns];

    Matrix2d Z1_t;
    Z1_t.rows = Z1.columns;
    Z1_t.columns = Z1.rows;
    Z1_t.data = new double[Z1_t.rows*Z1_t.columns];

    Matrix2d dL_dW1;
    dL_dW1.rows = W1.columns;
    dL_dW1.columns = W1.rows;
    dL_dW1.data = new double[dL_dW1.rows*dL_dW1.columns];

    Matrix2d X_t;
    X_t.rows = X.columns;
    X_t.columns = X.rows;
    X_t.data = new double[X_t.rows*X_t.columns];

    Matrix2d learning_rate_dL_dW3;
    learning_rate_dL_dW3.rows = dL_dW3.rows;
    learning_rate_dL_dW3.columns = dL_dW3.columns;
    learning_rate_dL_dW3.data = new double [learning_rate_dL_dW3.columns * learning_rate_dL_dW3.rows];

    Matrix2d learning_rate_dL_dW3_t;
    learning_rate_dL_dW3_t.rows = dL_dW3.columns;
    learning_rate_dL_dW3_t.columns = dL_dW3.rows;
    learning_rate_dL_dW3_t.data = new double [learning_rate_dL_dW3_t.columns * learning_rate_dL_dW3_t.rows];

    Matrix2d learning_rate_dL_dW2;
    learning_rate_dL_dW2.rows = dL_dW2.rows;
    learning_rate_dL_dW2.columns = dL_dW2.columns;
    learning_rate_dL_dW2.data = new double [learning_rate_dL_dW2.columns * learning_rate_dL_dW2.rows];

    Matrix2d learning_rate_dL_dW2_t;
    learning_rate_dL_dW2_t.rows = dL_dW2.columns;
    learning_rate_dL_dW2_t.columns = dL_dW2.rows;
    learning_rate_dL_dW2_t.data = new double [learning_rate_dL_dW2_t.columns * learning_rate_dL_dW2_t.rows];

    Matrix2d learning_rate_dL_dW1;
    learning_rate_dL_dW1.rows = dL_dW1.rows;
    learning_rate_dL_dW1.columns = dL_dW1.columns;
    learning_rate_dL_dW1.data = new double [learning_rate_dL_dW1.columns * learning_rate_dL_dW1.rows];

    Matrix2d learning_rate_dL_dW1_t;
    learning_rate_dL_dW1_t.rows = dL_dW1.columns;
    learning_rate_dL_dW1_t.columns = dL_dW1.rows;
    learning_rate_dL_dW1_t.data = new double [learning_rate_dL_dW1_t.columns * learning_rate_dL_dW1_t.rows];

    Matrix2d learning_rate_dL_dY3;
    learning_rate_dL_dY3.rows = dL_dY3.rows;
    learning_rate_dL_dY3.columns = dL_dY3.columns;
    learning_rate_dL_dY3.data = new double [learning_rate_dL_dY3.columns * learning_rate_dL_dY3.rows];

    Matrix2d learning_rate_dL_dY2;
    learning_rate_dL_dY2.rows = dL_dY2.rows;
    learning_rate_dL_dY2.columns = dL_dY2.columns;
    learning_rate_dL_dY2.data = new double [learning_rate_dL_dY2.columns * learning_rate_dL_dY2.rows];


    Matrix2d learning_rate_dL_dY1;
    learning_rate_dL_dY1.rows = dL_dY1.rows;
    learning_rate_dL_dY1.columns = dL_dY1.columns;
    learning_rate_dL_dY1.data = new double [learning_rate_dL_dY1.columns * learning_rate_dL_dY1.rows];

    double loss = 0;
    for(int iter = 1; iter <= iterations;iter++)
    {
        loss = forward_propagation(X,labels,W1,b1,Z1,W2,b2,Z2,W3,b3,Z3);
        
        std::printf("\r[iteration %i] calculated loss: %f",iter,loss);

        // dL_dY3
        matrix_diff(Z3,labels,dL_dY3);

        // dL_dY2
        matrix_diff(one_col_vec_Z2,Z2,diff_1_Z2);
        matrix_dot_product(Z2,diff_1_Z2,Z2_dot_diff_1_Z2);
        matrix_multiplication(W3,dL_dY3,W3_mmul_dL_dY3);
        matrix_dot_product(W3_mmul_dL_dY3,Z2_dot_diff_1_Z2,dL_dY2); 

        // dL_dY1
        matrix_diff(one_col_vec_Z1,Z1,diff_1_Z1);
        matrix_dot_product(Z1,diff_1_Z1,Z1_dot_diff_1_Z1);
        matrix_multiplication(W2,dL_dY2,W2_mmul_dL_dY2);
        matrix_dot_product(W2_mmul_dL_dY2,Z1_dot_diff_1_Z1,dL_dY1);

        //update W3
        matrix_multiplication(dL_dY3,Z2_t,dL_dW3);
        scalar_matrix_dot_product(learning_rate,dL_dW3,learning_rate_dL_dW3);
        matrix_transpose(learning_rate_dL_dW3,learning_rate_dL_dW3_t);
        matrix_diff(W3,learning_rate_dL_dW3_t,W3);

        //update b3
        scalar_matrix_dot_product(learning_rate,dL_dY3,learning_rate_dL_dY3);
        matrix_diff(b3,learning_rate_dL_dY3,b3);

        //update W2
        matrix_multiplication(dL_dY2,Z1_t,dL_dW2);
        scalar_matrix_dot_product(learning_rate,dL_dW2,learning_rate_dL_dW2);
        matrix_transpose(learning_rate_dL_dW2,learning_rate_dL_dW2_t);
        matrix_diff(W2,learning_rate_dL_dW2_t,W2);

        //update b2
        scalar_matrix_dot_product(learning_rate,dL_dY2,learning_rate_dL_dY2);
        matrix_diff(b2,learning_rate_dL_dY2,b2);

        //update W1
        matrix_multiplication(dL_dY1,X_t,dL_dW1);
        scalar_matrix_dot_product(learning_rate,dL_dW1,learning_rate_dL_dW1);
        matrix_transpose(learning_rate_dL_dW1,learning_rate_dL_dW1_t);
        matrix_diff(W1,learning_rate_dL_dW1_t,W1);

        //update b1
        scalar_matrix_dot_product(learning_rate,dL_dY1,learning_rate_dL_dY1);
        matrix_diff(b1,learning_rate_dL_dY1,b1);

        std::cout << std::flush;
    }

    delete[] Z1.data;
    delete[] Z2.data;
    delete[] Z3.data;

    delete[] dL_dY3.data;
    delete[] dL_dY2.data;
    delete[] dL_dY1.data;

    delete[] one_col_vec_Z2.data;
    delete[] diff_1_Z2.data;
    delete[] Z2_dot_diff_1_Z2.data;
    delete[] W3_mmul_dL_dY3.data;

    delete[] one_col_vec_Z1.data;
    delete[] diff_1_Z1.data;
    delete[] Z1_dot_diff_1_Z1.data;
    delete[] W2_mmul_dL_dY2.data;

    delete[] dL_dW3.data;
    delete[] Z2_t.data;
    delete[] dL_dW2.data;
    delete[] Z1_t.data;
    delete[] dL_dW1.data;
    delete[] X_t.data;

    delete[] learning_rate_dL_dW3.data;
    delete[] learning_rate_dL_dW2.data;
    delete[] learning_rate_dL_dW1.data;

    delete[] learning_rate_dL_dW3_t.data;
    delete[] learning_rate_dL_dW2_t.data;
    delete[] learning_rate_dL_dW1_t.data;

    delete[] learning_rate_dL_dY3.data;
    delete[] learning_rate_dL_dY2.data;
    delete[] learning_rate_dL_dY1.data;
} 

void zero_initializazion(Matrix2d& D)
{
    for(int i = 0; i < D.rows;i++)
    {
        for(int j = 0; j < D.columns;j++)
        {
            D.data[i*D.columns + j] = 0;
        }
    }
}

void one_hot_encode(const double *labelsArray,Matrix2d& labels)
{
    zero_initializazion(labels);

    for(int _example = 0; _example < labels.columns; _example++)
    {
        labels.data[static_cast<int>(labelsArray[_example]) * labels.columns + _example ] = 1;
    }
}

void read_minst_dataset(const std::string& path,Matrix2d& data,Matrix2d& labels)
{
    const int EXAMPLES = data.columns;
    const int FEATURES = data.rows;

    std::fstream input_file;
    input_file.open(path);

    double *labelsArray = new double[EXAMPLES];
    double *imageVector = new double[FEATURES];

    std::string line = "";
    std::string strBuffer;

    for(int i = -1; i < EXAMPLES ; i++)
    {
        line.clear();

        getline(input_file,line);

        // avoid reading the header
        if(i == -1){continue;}

        std::stringstream input_stream(line);
        
        // reading the label
        getline(input_stream,strBuffer,',');
        labelsArray[i] = std::stod(strBuffer);
        strBuffer.clear();

        // reading one image
        for(int j = 0; j < FEATURES; j++)
        {
            getline(input_stream,strBuffer,',');
            imageVector[j] = std::stod(strBuffer);
            strBuffer.clear();
        }
   
        // convert the image into a column vector
        for(int row = 0; row < FEATURES; row++)
        {
            data.data[row*EXAMPLES + i] = imageVector[row];
        }
    }

    one_hot_encode(labelsArray,labels);

    input_file.close();

    delete[] labelsArray;
    delete[] imageVector;
}

void softmax_to_one_hot(Matrix2d& output)
{
    double *arrayOfLabels = new double[output.columns];

    for(int _example = 0; _example < output.columns;_example++)
    {
        int index_of_highest_prob_class = -1;
        double highest_prob = -1.0;

        for(int _class = 0; _class < output.rows; _class++)
        {
            if(output.data[_class*output.columns + _example] > highest_prob)
            {
                highest_prob = output.data[_class*output.columns + _example];
                index_of_highest_prob_class = _class*output.columns;
            }    
        }
        arrayOfLabels[_example] = index_of_highest_prob_class;
    }

    one_hot_encode(arrayOfLabels,output);

    delete[] arrayOfLabels;
}

void predict(
    const Matrix2d& test,
    const Matrix2d& test_labels,
    const Matrix2d& W1,
    const Matrix2d& b1,
    const Matrix2d& W2,
    const Matrix2d& b2,
    const Matrix2d& W3,
    const Matrix2d& b3
)
{
    Matrix2d new_b1;
    new_b1.rows    = b1.rows;
    new_b1.columns = test.columns;
    new_b1.data = new double[new_b1.rows*new_b1.columns];

    for(int i = 0; i < new_b1.rows;i++)
    {
        for(int j = 0; j < new_b1.columns;j++)
        {
            new_b1.data[i*new_b1.columns + j] = b1.data[i*new_b1.columns + j];
        }
    }

    Matrix2d new_b2;
    new_b2.rows    = b2.rows;
    new_b2.columns = test.columns;
    new_b2.data = new double[new_b2.rows*new_b2.columns];

    for(int i = 0; i < new_b2.rows;i++)
    {
        for(int j = 0; j < new_b2.columns;j++)
        {
            new_b2.data[i*new_b2.columns + j] = b2.data[i*new_b2.columns + j];
        }
    }

    Matrix2d new_b3;
    new_b3.rows    = b3.rows;
    new_b3.columns = test.columns;
    new_b3.data = new double[new_b3.rows*new_b3.columns];

    for(int i = 0; i < new_b3.rows;i++)
    {
        for(int j = 0; j < new_b3.columns;j++)
        {
            new_b3.data[i*new_b3.columns + j] = b3.data[i*new_b3.columns + j];
        }
    }

    Matrix2d Z1;
    Z1.rows = new_b1.rows;
    Z1.columns = new_b1.columns;
    Z1.data = new double[Z1.rows * Z1.columns];

    Matrix2d Z2;
    Z2.rows = new_b2.rows;
    Z2.columns = new_b2.columns;
    Z2.data = new double[Z2.rows * Z2.columns];

    Matrix2d Z3;
    Z3.rows = new_b3.rows;
    Z3.columns = new_b3.columns;
    Z3.data = new double[Z3.rows * Z3.columns];

    double loss = forward_propagation(test,test_labels,W1,new_b1,Z1,W2,new_b2,Z2,W3,new_b3,Z3);

    std::printf("Average loss of test_set: %f\n",loss);

    delete[] Z1.data;
    delete[] Z2.data;
    delete[] Z3.data;

    delete[] new_b1.data;
    delete[] new_b2.data;
    delete[] new_b3.data;
}

int main(void)
{
    const int TRAINING_EXAMPLES = 1000;
    const int TEST_EXAMPLES = 100;

    Matrix2d X;
    X.rows    = 784;
    X.columns = TRAINING_EXAMPLES;
    X.data = new double[X.rows*X.columns];

    Matrix2d labels;
    labels.rows = 10;
    labels.columns = X.columns;
    labels.data = new double[labels.rows*labels.columns];

    read_minst_dataset("/home/andres/basicMatrixOperations/mnist_dataset/mnist_train.csv",X,labels);

    Matrix2d W1,b1,W2,b2,W3,b3;

    W1.rows = X.rows;
    W1.columns = 100;
    W1.data = new double[W1.rows*W1.columns];

    b1.rows = W1.columns;
    b1.columns = X.columns;
    b1.data = new double[b1.rows*b1.columns];

    W2.rows = W1.columns;
    W2.columns = 50;
    W2.data = new double[W2.rows*W2.columns];

    b2.rows = W2.columns;
    b2.columns = X.columns;
    b2.data = new double[b2.rows*b2.columns];

    W3.rows = W2.columns;
    W3.columns = 10;
    W3.data = new double[W3.rows*W3.columns];

    b3.rows = W3.columns;
    b3.columns = X.columns;
    b3.data = new double[b3.rows*b3.columns];

    gradient_descent(X,labels,300,0.1,W1,b1,W2,b2,W3,b3);

    std::cout << std::endl;

    Matrix2d test;
    test.rows = 784;
    test.columns = TEST_EXAMPLES;
    test.data = new double[test.rows*test.columns];

    Matrix2d test_labels;
    test_labels.rows = 10;
    test_labels.columns = test.columns;
    test_labels.data = new double[test_labels.rows*test_labels.columns];

    read_minst_dataset("/home/andres/basicMatrixOperations/mnist_dataset/mnist_test.csv",test,test_labels);

    predict(test,test_labels,W1,b1,W2,b2,W3,b3);

    delete[] X.data;
    delete[] labels.data;

    delete[] W1.data;
    delete[] b1.data;
    delete[] W2.data;
    delete[] b2.data;
    delete[] W3.data;
    delete[] b3.data;

    delete[] test.data;
    delete[] test_labels.data;

    return 0;
}