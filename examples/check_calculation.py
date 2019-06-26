import matplotlib
import cython_lib.calculation
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "stop_watch"))
from stop_watch.stop_watch import calc_time

iter_num = 10000
X = np.random.random(300).astype(np.float32)
Y = np.random.random(300).astype(np.float32)

mX = np.random.random((120, 300)).astype(np.float32)
mY = np.random.random((120, 300)).astype(np.float32)

@calc_time
def numpy_dot():
    for i in range(iter_num):
        X @ Y.T

@calc_time
def multi_numpy_dot():
    for i in range(iter_num):
        mX @ mY.T

numpy_dot()
multi_numpy_dot()

@calc_time
def cnumpy_dot():
    for i in range(iter_num):
        cython_lib.calculation.cnum_dot_one(X, Y.T)

@calc_time
def multi_cnumpy_dot():
    for i in range(iter_num):
        cython_lib.calculation.cnum_dot(mX, mY.T)


@calc_time
def multi_cnumpy_dot_REAL():
    for i in range(iter_num):
        cython_lib.calculation.cnum_dot_REAL(mX, mY.T)
        
@calc_time
def multi_cnumpy_gemm_REAL():
    for i in range(iter_num):
        cython_lib.calculation.cnum_gemm_REAL(mX, mY.T)

# numpy's dot
cnumpy_dot()
multi_cnumpy_dot()
# cnumpy dot with for
multi_cnumpy_dot_REAL()
# gemm dot
multi_cnumpy_gemm_REAL()

# we can see small errors by underflow
print(mX)
print(mY.T)
#print(mX @ mY)
#print(mX.T @ mY.T)
print(mX @ mY.T)
print(cython_lib.calculation.cnum_gemm_REAL(mX, mY.T))
print(mX @ mY.T == cython_lib.calculation.cnum_gemm_REAL(mX, mY.T))
