from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("cython_func/mcts_core",["cython_func/mcts_core.pyx"]),
    Extension("cython_func/mcts",["cython_func/mcts.pyx"])
]

setup(
    ext_modules=cythonize(extensions)
)
