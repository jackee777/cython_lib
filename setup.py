from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
from setuptools import setup, Extension

ext_modules = [
    Extension('cython_lib.calculation',
              sources=[
                  'cython_lib/coreutils.c',
                  'cython_lib/calculation.pyx'],
              include_dirs=[np.get_include(), 'opt/OpenBLAS/include'],
              libraries = ['openblas'],
              library_dirs = ['/opt/OpenBLAS/lib'],
              language='c',
              compiler_directives={'language_level' : "3"},
              extra_compile_args=['-O2']
              ),
    Extension('cython_lib.pagerank',
              sources=['cython_lib/pagerank.pyx'],
              include_dirs=[np.get_include()],
              compiler_directives={'language_level' : "3"}
              )
]

setup(
    name='cython_lib',
    packages=['cython_lib'],
    version='0.0.1',
    ext_modules=cythonize(ext_modules),
    cmdclass={'build_ext': build_ext}
)
