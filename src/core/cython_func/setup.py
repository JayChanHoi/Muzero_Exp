from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("cython_func/mcts_core",["mcts_core.pyx"]),
    Extension("cython_func/mcts",["mcts.pyx"])
]

setup(
    ext_modules=cythonize(extensions)
)
