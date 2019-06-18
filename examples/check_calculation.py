import cython_lib.calculation
from functools import wraps
import numpy as np
import time

def calc_time(function) :
    @wraps(function)
    def wrapper(*args, **kargs) :
        start = time.perf_counter()
        result = function(*args,**kargs)
        process_time =  time.perf_counter() - start
        print("{} spends {} ms the "
              "calculaton".format(function.__name__, process_time))
        return result
    return wrapper

iter_num = 10000
X = np.random.random(300).astype(np.float32)
Y = np.random.random(300).astype(np.float32)

mX = np.random.random((300, 3)).astype(np.float32)
mY = np.random.random((300, 3)).astype(np.float32)

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
        cython_lib.calculation.cnum_dot_REAL(mX, mY)

cnumpy_dot()
multi_cnumpy_dot()
multi_cnumpy_dot_REAL()

# we can see small errors by underflow
print(mX @ mY.T)
print(cython_lib.calculation.cnum_dot_REAL(mX, mY))
print(mX @ mY.T == cython_lib.calculation.cnum_dot_REAL(mX, mY))
