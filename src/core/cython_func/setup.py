from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='cython_func/mcts_main',
    ext_modules=cythonize("main.pyx")
)
