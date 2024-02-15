import logging
from collections import namedtuple, defaultdict
import numpy as np
from scipy import sparse
import sksparse
import topology
#from log_setup import logging 

logging.basicConfig(level=logging.DEBUG)

def log_sparsity(sp):
    logging.debug('Sparsity %.1f%%', 100 * (1 - sp.nnz / (sp.shape[0] * sp.shape[1])))

class FeatureSelectionProblem:
    def __init__(self, simplicial_complex, simplex_points, kernel, max_alpha = 1e-2):
        self.simplicial_complex = simplicial_complex
        self.simplex_points = simplex_points
        self.kernel = kernel
        self.max_alpha = max_alpha
        self.alpha_step = 10
        logging.info('Starting precomputing matrices')
        self._precompute_kernel_matrices()
        self._precompute_cholesky_factors()
        self._precompute_there_and_back_matrices()
        logging.info('Done precomputing matrices')

    def _compute_kernel(self, target, source):
        return sparse.csc_matrix(self.kernel(self.simplex_points[target], self.simplex_points[source]))

    def _precompute_kernel_matrices(self):
        logging.info('Starting precomputing kernel matrices')
        self._kernel_matrices = defaultdict(dict)
        for simplex in self.simplicial_complex.simplices():
            logging.debug('Kernel matrix for %s', simplex)
            self._kernel_matrices[simplex][simplex] = self._compute_kernel(simplex, simplex)
            log_sparsity(self._kernel_matrices[simplex][simplex])
            for face in self.simplicial_complex.faces(simplex):
                logging.debug('Kernel matrix for %s, %s', simplex, face)
                self._kernel_matrices[simplex][face] = self._compute_kernel(simplex, face)
                self._kernel_matrices[face][simplex] = self._kernel_matrices[simplex][face].conj().T
        logging.info('Done precomputing kernel matrices')

    def _compute_cholesky_factor(self, simplex):
        alpha = 0
        matrix = self._kernel_matrices[simplex][simplex]
        while alpha <= self.max_alpha:
            try: 
                regularised_matrix = matrix + alpha * sparse.eye(matrix.shape[0], format='csc')
                return sksparse.cholmod.cholesky(regularised_matrix)
            except sksparse.cholmod.CholmodNotPositiveDefiniteError:
                if alpha == 0:
                    alpha = 1e-15
                else:
                    alpha *= self.alpha_step
                logging.warn('Cholesky decomposition failed due to not PD, now trying with %.1g regularisation', alpha)
        logging.error('Could not find appropriate regularisation within bounds')
        raise ValueError(f'Kernel matrix is not positive definite')


    def _precompute_cholesky_factors(self):
        logging.info('Starting precomputing Cholesky factors')
        self._cholesky_factors = {}
        for simplex in self.simplicial_complex.simplices():
            logging.debug('Cholesky decomposition for %s', simplex)
            self._cholesky_factors[simplex] = self._compute_cholesky_factor(simplex)
        logging.info('Done precomputing Cholesky factors')



    def _compute_there_and_back(self, left_simplex, right_simplex):
        left_right_projection = self._kernel_matrices[left_simplex][right_simplex]
        right_cholesky_factor = self._cholesky_factors[right_simplex]
        right_left_projection = left_right_projection.conj().T
        there_and_back = left_right_projection @ right_cholesky_factor.solve_A(sparse.csc_matrix(right_left_projection))
        return there_and_back
    
    def _precompute_there_and_back_matrices(self):
        logging.info('Starting precomputing there-and-back matrices')
        self._there_and_back_matrices = defaultdict(dict)
        for simplex in self.simplicial_complex.simplices():
            for face in self.simplicial_complex.faces(simplex):
                logging.debug('There-and-back for %s, %s', simplex, face)
                self._there_and_back_matrices[simplex][face] = self._compute_there_and_back(simplex, face)
                self._there_and_back_matrices[face][simplex] = self._compute_there_and_back(face, simplex)
        logging.info('Done precomputing there-and-back matrices')
