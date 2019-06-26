"# cython_lib" 

# openblas
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
You should refer numpy https://github.com/numpy/numpy/blob/master/site.cfg.example and sumpy.
And this japanese page https://qiita.com/yukiB/items/bec24db4a5a41db02895.


https://qiita.com/higucheese/items/9ae1cf9611aaadfdc073