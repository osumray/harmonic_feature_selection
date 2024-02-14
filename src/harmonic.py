from collections import namedtuple, defaultdict
import numpy as np
from scipy import sparse
from sksparse import cholmod
import topology

class Problem:
    def __init__(self, simplicial_complex, simplex_points, kernel):
        self.simplicial_complex = simplicial_complex
        self.simplex_points = simplex_points
        self.kernel = kernel
        