#ifndef ADAM_H
#define ADAM_H

typedef struct AdamOptimizer {
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    double *m;
    double *v;
    int t;
    int num_params;
} AdamOptimizer;

void initAdamOptimizer(AdamOptimizer *optimizer, double learning_rate, double beta1, double beta2, double epsilon, int num_params);
void updateParamsAdamOptimizer(AdamOptimizer *optimizer, double *params, double *grads, int num_params);
void freeAdamOptimizer(AdamOptimizer *optimizer);

#endif