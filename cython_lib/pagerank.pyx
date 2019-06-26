#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8

from cpython cimport PyCapsule_GetPointer
import numpy as np
cimport numpy as np

try:
    from scipy.linalg.blas import fblas
except ImportError:
    # in scipy > 0.15, fblas function has been removed
    import scipy.linalg.blas as fblas

cdef char NORMAL = 'N'
cdef char TRANSPOSE = 'T'
cdef char C = 'C'
cdef int ONE = 1    
    
# sgemm
cdef sgemm_ptr sgemm=<sgemm_ptr>PyCapsule_GetPointer(fblas.sgemm._cpointer, NULL)


cpdef np.ndarray PersonalizedPageRank(np.ndarray[REAL_t, ndim=2] Map,
                          np.ndarray[REAL_t, ndim=1] pr,
                          np.ndarray[REAL_t, ndim=1] v, 
                          alpha, iter_num):
    cdef int map_r = <int>Map.shape[0], map_l = <int>Map.shape[1]
    cdef int pr_r = <int>pr.shape[0], pr_l = <int>pr.shape[1]
    cdef int i
    cdef float beta = 1 - alpha
    
    for i in range(iter_num):
        pr = alpha * np.dot(Map, pr) + beta * v
    
    return pr

"""
cpdef np.ndarray testPPR(np.ndarray[REAL_t, ndim=2] Map,
                          np.ndarray[REAL_t, ndim=1] pr,
                          np.ndarray[REAL_t, ndim=1] v, 
                          alpha, iter_num):
    cdef int map_r = <int>Map.shape[0], map_l = <int>Map.shape[1]
    cdef int pr_r = <int>pr.shape[0], pr_l = <int>pr.shape[1]
    cdef int i
    cdef float beta = 1 - alpha
    
    cython_PPR(<REAL_t *>(np.PyArray_DATA(Map)),
               <REAL_t *>(np.PyArray_DATA(pr)),
               <REAL_t *>(np.PyArray_DATA(v)),
               map_r, map_l, pr_r, pr_l,
               alpha, beta, iter_num)
    
    return v


cdef void cython_PPR(REAL_t* MAP,
                     REAL_t* PR,
                     REAL_t* V, 
                     int map_r, int map_l, int pr_r, int pr_l,
                     REAL_t alpha, REAL_t beta, int iter_num):
    cdef int i
    
    for i in range(iter_num):
        sgemm(&C, &NORMAL, &map_r, &pr_l, &map_l, 
              &alpha, &MAP[0], &map_l, &PR[0], &pr_r, 
              &beta, &V[0], &map_r)
"""
        