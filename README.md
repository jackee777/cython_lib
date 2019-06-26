# cython_lib
This repository is my cython library. Perhaps, if you want to use cython, this repository is useful for you.

Especially, calculation.pyx, calculation.pxd, coreutils.c, and, coreutils.h help you to write codes. In addition to this, you should refer [numpy example](https://github.com/numpy/numpy/blob/master/site.cfg.example) and [scipy example](https://github.com/scipy/scipy/blob/master/site.cfg.example).

# reference pages
[How to Use](https://github.com/jackee777/cython_lib/tree/master/How_to_Use)

- [dot](https://github.com/jackee777/cython_lib/blob/master/How_to_Use/dot.md)
- [Personalized PageRank (PPR)](https://github.com/jackee777/cython_lib/blob/master/How_to_Use/Personalized_PageRank.md)

# install
```
python setup.py install
```

## use openblas
### openblas install Ubuntu
```
sudo apt-get install git python-dev gfortran
git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make FC=gfortran
sudo make PREFIX=/opt/openblas install
```

```
export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH > ~/.bashrc
```

### reference memorandom
And this japanese page https://qiita.com/yukiB/items/bec24db4a5a41db02895 and https://qiita.com/higucheese/items/9ae1cf9611aaadfdc073.
