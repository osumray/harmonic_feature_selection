import logging
from collections import namedtuple, defaultdict
import numpy as np
from scipy import sparse
import sksparse
import topology


logging.basicConfig(level=logging.DEBUG)


def log_sparsity(sp):
    logging.debug('Sparsity %.1f%%', 100 * (1 - sp.nnz / (sp.shape[0] * sp.shape[1])))


class FeatureSelectionProblem:
    def __init__(self, simplicial_complex, simplex_points, kernel):
        self.simplicial_complex = simplicial_complex
        self.simplex_points = simplex_points
        self.kernel = kernel
        self._has_precomputed = False

    def _compute_kernel(self, target, source):
        return sparse.csc_matrix(self.kernel(self.simplex_points[target], self.simplex_points[source]))

    def _compute_cholesky_factor(self, simplex):
        try: 
            return sksparse.cholmod.cholesky(self._kernel_matrices[simplex][simplex])
        except sksparse.cholmod.CholmodNotPositiveDefiniteError as error_non_pd:
            logging.error('Failed to compute Cholesky factor, attempting to find smallest eigenvalue')
            try:
                small_e = sparse.linalg.eigsh(self._kernel_matrices[simplex][simplex], k=1,
                                               which='SM', tol=1e-3, maxiter=1e5, return_eigenvectors=False)[0]
                logging.error('Failed to compute Cholesky factor, smallest eigenvalue is around %.3f', small_e)
            except: 
                logging.error('Could not find smallest eigenvalue in %i iterations either', 1e5)
            raise error_non_pd
        

    def precompute_matrices(self):
        logging.info('Starting precomputing matrices')
        self._precompute_kernel_matrices(from_gen, to_gen)
        self._precompute_cholesky_factors(gen)
        self._precompute_there_and_back_matrices(from_gen, to_gen)
        logging.info('Done precomputing matrices')
        self._has_precomputed = True

    def _compute_there_and_back(self, left_simplex, right_simplex):
        left_right_projection = self._kernel_matrices[left_simplex][right_simplex]
        right_cholesky_factor = self._cholesky_factors[right_simplex]
        right_left_projection = self._kernel_matrices[right_simplex][left_simplex]
        there_and_back = left_right_projection @ right_cholesky_factor.solve_A(sparse.csc_matrix(right_left_projection))
        return there_and_back

class OriginalFeatureSelectionProblem(FeatureSelectionProblem):
    def __init__(self, simplicial_complex, simplex_points, kernel, at_simplex):
        self.at_simplex = at_simplex
        super().__init__(simplicial_complex, simplex_points, kernel)

class DualFeatureSelectionProblem(FeatureSelectionProblem):
    def _precompute_kernel_matrices(self):
        logging.info('Starting precomputing kernel matrices')
        self._kernel_matrices = defaultdict(dict)
        for simplex in self.simplicial_complex.simplices():
            logging.debug('Kernel matrix for %s', simplex)
            self._kernel_matrices[simplex][simplex] = self._compute_kernel(simplex, simplex)
            log_sparsity(self._kernel_matrices[simplex][simplex])
            for face in self.simplicial_complex.faces(simplex):
                logging.debug('Kernel matrix for %s, %s', simplex, face)
                log_sparsity(self._kernel_matrices[simplex][simplex])
                self._kernel_matrices[simplex][face] = self._compute_kernel(simplex, face)
                self._kernel_matrices[face][simplex] = sparse.csc_matrix(self._kernel_matrices[simplex][face].H)
        logging.info('Done precomputing kernel matrices')

    def _precompute_cholesky_factors(self):
        logging.info('Starting precomputing Cholesky factors')
        self._cholesky_factors = {}
        for simplex in self.simplicial_complex.simplices():
            logging.debug('Cholesky decomposition for %s', simplex)
            self._cholesky_factors[simplex] = self._compute_cholesky_factor(simplex)
        logging.info('Done precomputing Cholesky factors')
 
    def _precompute_there_and_back_matrices(self):
        logging.info('Starting precomputing there-and-back matrices')
        self._there_and_back_matrices = defaultdict(dict)
        for simplex in self.simplicial_complex.simplices():
            for face in self.simplicial_complex.faces(simplex):
                logging.debug('There-and-back for %s, %s', simplex, face)
                self._there_and_back_matrices[simplex][face] = self._compute_there_and_back(simplex, face)
                self._there_and_back_matrices[face][simplex] = self._compute_there_and_back(face, simplex)
        logging.info('Done precomputing there-and-back matrices')

    def compute_condensed_laplacian(self, at_simplex):
        laplacian_matrix = self._kernel_matrices[at_simplex][at_simplex]
        
