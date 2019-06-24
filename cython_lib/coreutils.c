#include "coreutils.h"
#include <stdio.h>
#include <cblas.h>

void _sdot_float_one(float* X0, float* X1, float* result,
            int x0_r, int x0_l, int x1_r, int x1_l)
{
    int i;
    int j;
    int k;
    long int r_target, x0_target, x1_target;

    /*
    for i in range(x0_r):
        for j in range(x1_l):
            result[i*x1_l+j] = <REAL_t>sdot(&x0_l, &X0[i*x0_l], &ONE, &X1[j*x1_r], &ONE)
    */
    for (i = 0; i < x0_r; ++i) {
        for (j = 0; j < x1_l; ++j) {
            r_target = i * x1_l;
            x0_target = i * x0_l;
            x1_target = j * x1_r;
            /*
            for (k = 0; k < x0_l; ++k) {
                result[r_target + j] += X0[x0_target + k] * X1[x1_target + k];
            }
            */
            //result[r_target + j] =
            cblas_sdot(x0_l, X0, 1, X1, 1);
        }
    }

}

float _fast_dot_four(float* X0, float* X1){
    return X0[0] * X1[0];
}

void _sdot_float_four(float* X0, float* X1, float* result,
            int x0_r, int x0_l, int x1_r, int x1_l)
{
    int i, i1, i2, i3;
    int j, j1, j2, j3;
    int k;

    for (i = 0; i < x0_r; i+=4) {
        i1 = i + 1;
        i2 = i + 2;
        i3 = i + 3;
        for (j = 0; j < x1_l; j+=4) {
            j1 = j + 1;
            j2 = j + 2;
            j3 = j + 3;
            for (k = 0; k < x0_l; k++) {
                result[i*x1_l + j] += X0[i*x0_l + k] * X1[j*x1_r + k];
                result[i*x1_l + j1] += X0[i*x0_l + k] * X1[j1*x1_r + k];
                result[i*x1_l + j2] += X0[i*x0_l + k] * X1[j2*x1_r + k];
                result[i*x1_l + j3] += X0[i*x0_l + k] * X1[j3*x1_r + k];

                result[i1*x1_l + j] += X0[i1*x0_l + k] * X1[j*x1_r + k];
                result[i1*x1_l + j1] += X0[i1*x0_l + k] * X1[j1*x1_r + k];
                result[i1*x1_l + j2] += X0[i1*x0_l + k] * X1[j2*x1_r + k];
                result[i1*x1_l + j3] += X0[i1*x0_l + k] * X1[j3*x1_r + k];

                result[i2*x1_l + j] += X0[i2*x0_l + k] * X1[j*x1_r + k];
                result[i2*x1_l + j1] += X0[i2*x0_l + k] * X1[j1*x1_r + k];
                result[i2*x1_l + j2] += X0[i2*x0_l + k] * X1[j2*x1_r + k];
                result[i2*x1_l + j3] += X0[i2*x0_l + k] * X1[j3*x1_r + k];

                result[i3*x1_l + j] += X0[i3*x0_l + k] * X1[j*x1_r + k];
                result[i3*x1_l + j1] += X0[i3*x0_l + k] * X1[j1*x1_r + k];
                result[i3*x1_l + j2] += X0[i3*x0_l + k] * X1[j2*x1_r + k];
                result[i3*x1_l + j3] += X0[i3*x0_l + k] * X1[j3*x1_r + k];

            }
        }
    }
}

void calc_dot_creal(float* X0, float* X1, float* result,
            int x0_r, int x0_l, int x1_r, int x1_l) {
    if(x0_r % 4 == 0 && x1_l % 4 == 0){
        _sdot_float_four(X0, X1, result,
            x0_r, x0_l, x1_r, x1_l);
    }
    else{
        _sdot_float_one(X0, X1, result,
            x0_r, x0_l, x1_r, x1_l);
    }
}