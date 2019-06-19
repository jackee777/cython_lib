#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8

import numpy as np
cimport numpy as np

cpdef np.ndarray PageRank(np.ndarray[REAL_t, ndim=2] Map,
                                      np.ndarray[REAL_t, ndim=2] v,
                                      np.ndarray[REAL_t, ndim=2] pr,
                                      int iter_num, REAL_t alpha):
    cdef int size = <int>Map.shape[0]
    for i in range(iter_num):
        pr = alpha * np.dot(Map, pr.T) + (1-alpha) * v
    return pr