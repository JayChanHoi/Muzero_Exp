import ray

from .cython_func.mcts_core import MCTS, CyphonNode

if __name__ == "__main__":
    root = CyphonNode(0)
    print('import cython module success')