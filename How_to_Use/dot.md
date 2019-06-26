# Caution
This code can only use dtype float32.

# Information
The way to use them is the same as [numpy.dot](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html).

## cnum_dot_REAL(a, b)
- a: 2-D float arrays
- b: 2-D float arrays

## cnum_gemm_REAL(a, b)
- a: 2-D float arrays
- b: 2-D float arrays

# Example Code
'cnum_gemm_REAL' and 'cnum_dot_REAL' get the same results.

This is the [example code](https://github.com/jackee777/cython_lib/blob/master/examples/check_calculation.py).
```
import cython_lib.calculation
import numpy as np

mX = np.random.random((120, 300)).astype(np.float32)
mY = np.random.random((120, 300)).astype(np.float32)

cython_lib.calculation.cnum_dot_REAL(mX, mY.T)
cython_lib.calculation.cnum_gemm_REAL(mX, mY.T)
```
