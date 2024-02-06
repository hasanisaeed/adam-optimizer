#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

typedef struct LinearRegression {
    double *weights;
    double bias;
    int n_features;
} LinearRegression;

void initLinearRegression(LinearRegression *model, int n_features);
double predict(LinearRegression *model, double *X);
void freeLinearRegression(LinearRegression *model);

#endif