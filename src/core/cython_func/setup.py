from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='mcts_core',
    ext_modules=cythonize("mcts.pyx")
)