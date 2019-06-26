cimport numpy as np

ctypedef np.float32_t REAL_t

# sgemm
ctypedef void (*sgemm_ptr) (char *transA, char *transB, int *m, int *n, int *k, 
                            float *alpha, float *a, int *lda, 
                            float *b, int *ldb,
                            float *beta, float *c, int *ldc) nogil
