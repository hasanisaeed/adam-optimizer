#ifndef TRAIN_H
#define TRAIN_H

#include "linear_regression.h"
#include "adam.h"

void trainLinearRegression(LinearRegression *model, double **X, double *y, int n_samples, AdamOptimizer *optimizer, int n_epochs);

#endif