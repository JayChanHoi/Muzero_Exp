from itertools import count
import math

import numpy as np

import torch

cdef class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""
    cdef public float maximum
    cdef public float minimum

    def __init__(self, float min_value_bound=0, float max_value_bound=0):
        self.maximum = min_value_bound if min_value_bound != 0 else -float('inf')
        self.minimum = max_value_bound if max_value_bound != 0 else float('inf')

    cpdef void update(self, float value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

        return

    cpdef float normalize(self, float value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        else:
            return value