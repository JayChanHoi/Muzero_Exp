from .cython_func.mcts_core import CyphonNode
from .cython_func.mcts import MCTS

if __name__ == "__main__":
    root = CyphonNode(0)
    MCTS(1, 1, 1, 1)
    print('import cython module succeed')