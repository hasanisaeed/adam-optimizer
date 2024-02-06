#include <math.h>
#include <stdio.h>
#include "linear_regression.h"

// Training loop for Linear Regression
void trainLinearRegression(LinearRegression *model, double **X, double *y, int n_samples, double learning_rate, int n_epochs) {
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        double dW[model->n_features];
        double db = 0.0;
        for (int i = 0; i < model->n_features; i++) {
            dW[i] = 0.0;
        }
        for (int i = 0; i < n_samples; i++) {
            double pred = predict(model, X[i]);
            double error = pred - y[i];
            for (int j = 0; j < model->n_features; j++) {
                dW[j] += (2.0 / n_samples) * X[i][j] * error;
            }
            db += (2.0 / n_samples) * error;
        }
        for (int i = 0; i < model->n_features; i++) {
            model->weights[i] -= learning_rate * dW[i];
        }
        model->bias -= learning_rate * db;

        printf("Epoch %d: Parameters after update:\n", epoch + 1);
        for (int i = 0; i < model->n_features; i++) {
            printf("Param[%d] = %.4d\n", i, model->weights);
        }
    }
}
