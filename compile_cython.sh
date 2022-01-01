cd src/core/cython_func

rm -rf main.cpython-37m-x86_64-linux-gnu.so
rm -rf main.c

rm -rf build

python setup.py build_ext --inplace