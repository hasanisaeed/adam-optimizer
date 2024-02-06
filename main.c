#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "linear_regression.h"
#include "adam.h"
#include "train.h"

// Function to display the dataset
void showDataset(double **X, double *y, int n_samples, int n_features) {
    printf("Dataset (X and Y):\n");
    for (int i = 0; i < n_samples; i++) {
        printf("Sample %d: ", i+1);
        for (int j = 0; j < n_features; j++) {
            printf("X[%d] = %.2f ", j, X[i][j]);
        }
        printf("Y = %.2f\n", y[i]);
    }
}

int main() {
    srand(time(NULL)); // Seed the random number generator

    int n_features, n_samples, n_epochs;

    // Read number of features, samples, and epochs from input
    printf("Enter number of features: ");
    scanf("%d", &n_features);
    printf("Enter number of samples: ");
    scanf("%d", &n_samples);
    printf("Enter number of epochs: ");
    scanf("%d", &n_epochs);

    // Initialize LinearRegression model and AdamOptimizer
    LinearRegression model;
    AdamOptimizer optimizer;

    initLinearRegression(&model, n_features);
    // Initialize AdamOptimizer with desired parameters
    initAdamOptimizer(&optimizer, 0.001, 0.9, 0.999, 1e-8, n_features);

    // Dynamically allocate memory for input data X (n_samples, n_features)
    double **X = (double **)malloc(n_samples * sizeof(double *));
    for (int i = 0; i < n_samples; i++) {
        X[i] = (double *)malloc(n_features * sizeof(double));
    }

    // Dynamically allocate memory for target data y (n_samples)
    double *y = (double *)malloc(n_samples * sizeof(double));

    // Generate fake data
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            X[i][j] = ((double)rand() / RAND_MAX) * 10.0; // Random feature values between 0 and 10
        }
        y[i] = ((double)rand() / RAND_MAX) * 10.0; // Random target values between 0 and 10
    }

    showDataset(X, y, n_samples, n_features);

    // Train the model
    trainLinearRegression(&model, X, y, n_samples, &optimizer, n_epochs);

    // Free resources
    freeLinearRegression(&model);
    freeAdamOptimizer(&optimizer);

    // Free the fake data resources
    for (int i = 0; i < n_samples; i++) {
        free(X[i]);
    }
    free(X);
    free(y);

    return 0;
}
