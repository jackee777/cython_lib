import cython_lib.pagerank
import numpy as np
from stop_watch.stop_watch import calc_time

Map = np.random.random((10000, 10000)).astype(np.float32)
alpha = 0.85
iteration_num = 10

@calc_time
def calc_py_PPR(i):
    v = np.zeros(Map.shape[0], dtype=np.float32)
    v[i] = 1
    pr = np.zeros(Map.shape[0], dtype=np.float32)
    pr[i] = 1
    for i in range(iteration_num):
        pr = alpha * Map @ pr.T + (1-alpha) * pr
    return pr


@calc_time
def calc_cy_PPR(i):
    v = np.zeros(Map.shape[0], dtype=np.float32)
    v[i] = 1
    pr = np.zeros(Map.shape[0], dtype=np.float32)
    pr[i] = 1
    return cython_lib.pagerank.PersonalizedPageRank(Map, pr.T, v.T, alpha, iteration_num)


"""
@calc_time
def calc_c_PPR(i):
    v = np.zeros(Map.shape[0], dtype=np.float32)
    v[i] = 1
    pr = np.zeros(Map.shape[0], dtype=np.float32)
    pr[i] = 1
    return cython_lib.pagerank.testPPR(Map, pr.T, v.T, alpha, iteration_num)
"""

print(calc_py_PPR(0))
print(calc_cy_PPR(0))
print(calc_py_PPR(0) == calc_cy_PPR(0))
#print(calc_c_PPR(0))
#print(calc_cy_PPR(0) == calc_c_PPR(0))