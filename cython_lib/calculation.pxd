import numpy as np
cimport numpy as np

ctypedef np.float32_t REAL_t

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

cdef extern from "coreutils.h":
     void _calc_dot_creal "calc_dot_creal"(REAL_t* X0, REAL_t* X1, REAL_t* result, int x0_r, int x0_l, int x1_r, int x1_l)

cdef extern from "coreutils.h":
     void _calc_gemm_creal "calc_gemm_creal"(REAL_t* X0, REAL_t* X1, REAL_t* result, int x0_r, int x0_l, int x1_r, int x1_l)
        
# sdot
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil

# sgemm
ctypedef void (*sgemm_ptr) (char *transA, char *transB, int *m, int *n, int *k, 
                            float *alpha, float *a, int *lda, 
                            float *b, int *ldb,
                            float *beta, float *c, int *ldc) nogil