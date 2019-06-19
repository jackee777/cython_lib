import cython_lib.pagerank
import numpy as np

Map = np.random.random((30, 30))
alpha = 0.85
iteration_num = 10

def calc_PPR(i):
    v = np.zeros(Map.shape[0], dtype=np.float16)
    v[i] = 1
    pr = np.zeros(Map.shape[0], dtype=np.float16)
    pr[i] = 1
    pr = cython_lib.pagerank.PR(Map, v, pr)
    return pr

def calc_PPR(i):
    v = np.zeros(Map.shape[0], dtype=np.float16)
    v[i] = 1
    pr = np.zeros(Map.shape[0], dtype=np.float16)
    pr[i] = 1
    pr = cython_lib.pagerank.PR(Map, v, pr)
    return pr

print(calc_PPR(0))
print(calc_PPR(1))