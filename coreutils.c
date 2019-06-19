#include "coreutils.h"

void calc_dot_creal(float* X0, float* X1, float* result,
            int x0_r, int x0_l, int x1_r, int x1_l) {
    int i = 0;
    int j = 0;
    int k = 0;

    for (i = 0; i < x0_r; ++i) {
        for (j = 0; j < x1_r; ++j) {
            for (k = 0; k < x0_l; ++k) {
                result[i * x0_l + j] += X0[i * x0_l + k] * b[k * x1_l + j];
            }
        }
    }

}