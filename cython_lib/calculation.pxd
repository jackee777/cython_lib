cimport numpy as np

ctypedef np.float32_t REAL_t

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil