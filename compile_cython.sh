cd src/core/cython_func
rm mcts.cpython-37m-x86_64-linux-gnu.so
rm mcts.c
rm -rf build
python setup.py build_ext --inplace