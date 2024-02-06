#include <stdlib.h>
#include <string.h>
#include <math.h>

// structure for the Adam optimizer
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

// Initialize the Adam optimizer
void initAdamOptimizer(AdamOptimizer *optimizer, double learning_rate, double beta1, double beta2, double epsilon, int num_params) {
    optimizer->learning_rate = learning_rate;
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->epsilon = epsilon;
    optimizer->m = (double *)calloc(num_params, sizeof(double));
    optimizer->v = (double *)calloc(num_params, sizeof(double));
    optimizer->t = 0;
    optimizer->num_params = num_params;
}

// Update parameters using Adam optimization
void updateParamsAdamOptimizer(AdamOptimizer *optimizer, double *params, double *grads, int num_params) {
    optimizer->t += 1;
    for (int i = 0; i < num_params; i++) {
        optimizer->m[i] = optimizer->beta1 * optimizer->m[i] + (1.0 - optimizer->beta1) * grads[i];
        optimizer->v[i] = optimizer->beta2 * optimizer->v[i] + (1.0 - optimizer->beta2) * grads[i] * grads[i];

        double m_corrected = optimizer->m[i] / (1.0 - pow(optimizer->beta1, optimizer->t));
        double v_corrected = optimizer->v[i] / (1.0 - pow(optimizer->beta2, optimizer->t));

        params[i] -= optimizer->learning_rate * m_corrected / (sqrt(v_corrected) + optimizer->epsilon);
    }
}

// Free memory allocated for Adam optimizer
void freeAdamOptimizer(AdamOptimizer *optimizer) {
    free(optimizer->m);
    free(optimizer->v);
}
