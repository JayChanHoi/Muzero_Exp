cd src/core/cython_func
rm -rf mcts_core.cpython-37m-x86_64-linux-gnu.so
rm -rf mcts_core.c
rm -rf build

python setup.py build_ext --inplace