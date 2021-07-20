from distutils.core import setup
from Cython.Build import cythonize
# from distutils.extension import Extension
# from Cython.Distutils import build_ext

setup(
    name='mcts_core',
    ext_modules=cythonize("mcts_core.pyx")
)
