#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

typedef struct LinearRegression {
    double *weights;
    double bias;
    int n_features;
} LinearRegression;

// Initialize the Linear Regression model
void initLinearRegression(LinearRegression *model, int n_features) {
    model->n_features = n_features;
    model->weights = (double *)malloc(n_features * sizeof(double));
    model->bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // Random bias initialization
    for (int i = 0; i < n_features; i++) {
        model->weights[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // Random weights initialization
    }
}

// Predict function for Linear Regression
double predict(LinearRegression *model, double *X) {
    double prediction = model->bias;
    for (int i = 0; i < model->n_features; i++) {
        prediction += X[i] * model->weights[i];
    }
    return prediction;
}

// Free memory of the Linear Regression model
void freeLinearRegression(LinearRegression *model) {
    free(model->weights);
}
