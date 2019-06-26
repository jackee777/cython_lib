#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8

from cpython cimport PyCapsule_GetPointer # PyCObject_AsVoidPtr
cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np

try:
    from scipy.linalg.blas import fblas
except ImportError:
    # in scipy > 0.15, fblas function has been removed
    import scipy.linalg.blas as fblas

# sdot
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)
# sgemm
cdef sgemm_ptr sgemm=<sgemm_ptr>PyCObject_AsVoidPtr(fblas.sgemm._cpointer)

cpdef REAL_t cnum_dot_one(np.ndarray X0, np.ndarray X1):
    return np.dot(X0, X1)

cpdef np.ndarray cnum_dot(np.ndarray X0, np.ndarray X1):
    return np.dot(X0, X1)

cdef char NORMAL = 'N'
cdef char TRANSPOSE = 'T'
cdef char C = 'C'
cdef int ONE = 1

cpdef np.ndarray cnum_dot_REAL(np.ndarray[REAL_t, ndim=2] X0, np.ndarray[REAL_t, ndim=2] X1):
    cdef int x0_r = <int>X0.shape[0], x0_l = <int>X0.shape[1]
    cdef int x1_r = <int>X1.shape[0], x1_l = <int>X1.shape[1]
    assert x0_l == x1_r, "x0_l must be equal to x1_l"
    cdef np.ndarray[REAL_t, ndim=2] result = np.zeros((x0_r, x1_l), dtype=X0.dtype)
    
    _calc_dot_creal(<REAL_t *>(np.PyArray_DATA(X0)),
                  <REAL_t *>(np.PyArray_DATA(X1)),
                  <REAL_t *>(np.PyArray_DATA(result)),
                  x0_r, x0_l, x1_r, x1_l)

    """
    cnum_sdot_real(<REAL_t *>(np.PyArray_DATA(X0)),
                  <REAL_t *>(np.PyArray_DATA(X1)),
                  <REAL_t *>(np.PyArray_DATA(result)),
                  x0_r, x0_l, x1_r, x1_l)
    """  

    return result

cpdef np.ndarray cnum_gemm_REAL(np.ndarray[REAL_t, ndim=2] X0, np.ndarray[REAL_t, ndim=2] X1):
    cdef int x0_r = <int>X0.shape[0], x0_l = <int>X0.shape[1]
    cdef int x1_r = <int>X1.shape[0], x1_l = <int>X1.shape[1]
    assert x0_l == x1_r, "x0_l must be equal to x1_l"
    cdef np.ndarray[REAL_t, ndim=2] result = np.zeros((x0_r, x1_l), dtype=X0.dtype)
    
    _calc_gemm_creal(<REAL_t *>(np.PyArray_DATA(X0)),
               <REAL_t *>(np.PyArray_DATA(X1)),
               <REAL_t *>(np.PyArray_DATA(result)),
               x0_r, x0_l, x1_r, x1_l)
    return result

    """
    cdef np.ndarray[REAL_t, ndim=2] result = np.zeros((x1_l, x0_r), dtype=X0.dtype)
    cnum_sgemm_real(<REAL_t *>(np.PyArray_DATA(X0)),
               <REAL_t *>(np.PyArray_DATA(X1)),
               <REAL_t *>(np.PyArray_DATA(result)),
               x0_r, x0_l, x1_r, x1_l)
    return result.T
    """
    
# use sdot(X0, X1) not (X0, X1.T)
cdef void cnum_sdot_real(REAL_t* X0, REAL_t* X1, REAL_t* result, int x0_r, int x0_l, int x1_r, int x1_l) nogil:
    cdef int i, j

    for i in xrange(x0_r):
        for j in xrange(x1_l):
            result[i*x1_l+j] = <REAL_t>sdot(&x0_l, &X0[i*x0_l], &ONE, &X1[j*x1_r], &ONE)

cdef void cnum_sgemm_real(REAL_t* X0, REAL_t* X1, REAL_t* result, int x0_r, int x0_l, int x1_r, int x1_l) nogil:
    cdef float alpha = 1.0
    cdef float beta = 0.0 
    
    # https://developer.apple.com/documentation/accelerate/1513282-cblas_dgemm?language=objc
    # using CblasRowMajor
    sgemm(&C, &NORMAL, &x0_r, &x1_l, &x0_l, 
          &alpha, &X0[0], &x0_l, &X1[0], &x1_r, 
          &beta, &result[0], &x0_r)

