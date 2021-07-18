from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='mcts_utils',
    ext_modules=cythonize("mcts_utils.pyx")
)