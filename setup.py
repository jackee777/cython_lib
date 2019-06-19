from Cython.Distutils import build_ext
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension('cython_lib.calculation',
              #sources=['coreutils.c', 'cython_lib/calculation.pyx'],
              sources=['cython_lib/calculation.pyx'],
              include_dirs=[np.get_include(), 'cython_lib'],
              compiler_directives={'language_level' : "3"}
              ),
    #Extension('cython_lib.pagerank',
    #              sources=['cython_lib/pagerank.pyx'],
    #              include_dirs=[np.get_include(), 'cython_lib'],
    #          )
]

setup(
    name='cython_lib',
    packages=['cython_lib'],
    version='0.0.1',
    ext_modules=cythonize(ext_modules),
    cmdclass={'build_ext': build_ext}
)