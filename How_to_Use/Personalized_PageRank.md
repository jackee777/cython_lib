# Caution
This code can only use dtype float32.

# Information
This function gets PageRank against the target index.

## PersonalizedPageRank(Map, pr, v, alpha, iteration_num)
```math
alpha * Map * pr + (1-alpha) * v
```
- Map: 2-D float arrays
- pr: 1-D float arrays
- v: 1-D float arrays
- alpha: float
- iteration_num: float, the number of Random Walk

# Example Code
'cnum_gemm_REAL' and 'cnum_dot_REAL' get the same results.

This is the [example code](https://github.com/jackee777/cython_lib/blob/master/examples/check_PPR.py).
```
import cython_lib.pagerank
import numpy as np

Map = np.random.random((10000, 10000)).astype(np.float32)
alpha = 0.85
iteration_num = 10

def calc_cy_PPR(i):
    v = np.zeros(Map.shape[0], dtype=np.float32)
    v[i] = 1
    pr = np.zeros(Map.shape[0], dtype=np.float32)
    pr[i] = 1
    return cython_lib.pagerank.PersonalizedPageRank(Map, pr.T, v.T, alpha, iteration_num)
    
result = []
for i in range(Map.shape[0]):
    result.append(calc_cy_PPR(i))
```
