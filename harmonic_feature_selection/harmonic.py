import logging
from collections import namedtuple, defaultdict
from functools import reduce
import numpy as np
from scipy import sparse
import sksparse
from ordered_set import OrderedSet
from . import topology


logging.basicConfig(level=logging.INFO)
#logging.basicConfig(filename='out.log', encoding='utf-8', level=logging.DEBUG)

def sparsity_message(sm):
    logging.debug('Sparsity %.1f%%', 100 * (1 - sm.nnz / (sm.shape[0] * sm.shape[1])))

def spy(log_message, x):
    log_message(x)
    return x

class FeatureSelectionProblem:
    def __init__(self, simplicial_complex, simplex_points, kernel, from_simplices, to_simplices_dict, do_both_there_and_back):
        self.simplicial_complex = simplicial_complex
        self.simplex_points = simplex_points
        self.kernel = kernel
        self.simplex_pairings = []

        self.from_simplices = from_simplices
        self.to_simplices_dict = to_simplices_dict
        self.do_both_there_back = do_both_there_and_back
        self.all_simplices = OrderedSet(self.from_simplices | reduce(lambda x, y : x | y, self.to_simplices_dict.values()))
        self.precompute_matrices()

    def _compute_kernel(self, target, source):
        return sparse.csc_matrix(self.kernel(self.simplex_points[target], self.simplex_points[source]))

    def _precompute_self_kernel_matrices(self):
        logging.info('Starting precomputing self kernel matrices')
        self._self_kernel_matrices = {}
        for simplex in self.all_simplices:
            logging.debug('Kernel matrix for %s', simplex)
            self._self_kernel_matrices[simplex] = spy(sparsity_message, self._compute_kernel(simplex, simplex))
        logging.info('Done precomputing self kernel matrices')

    def _compute_cholesky_factor(self, simplex):
        try: 
            return sksparse.cholmod.cholesky(self._self_kernel_matrices[simplex])
        except sksparse.cholmod.CholmodNotPositiveDefiniteError as error_non_pd:
            logging.error('Failed to compute Cholesky factor, attempting to find smallest eigenvalue')
            try:
                small_e = sparse.linalg.eigsh(self._self_kernel_matrices[simplex], k=1,
                                               which='SM', tol=1e-3, maxiter=1e5, return_eigenvectors=False)[0]
                logging.error('Failed to compute Cholesky factor, smallest eigenvalue is around %.3f', small_e)
            except: 
                logging.error('Could not find smallest eigenvalue in %i iterations either', 1e5)
            raise error_non_pd
    
    def _precompute_cholesky_factors(self):
        logging.info('Starting precomputing Cholesky factors')
        self._cholesky_factors = {}
        for simplex in self.all_simplices:
            logging.debug('Cholesky decomposition for %s', simplex)
            self._cholesky_factors[simplex] = self._compute_cholesky_factor(simplex)
        logging.info('Done precomputing Cholesky factors')

    def _precompute_kernel_matrices(self):
        logging.info('Starting precomputing kernel matrices')
        self._kernel_matrices = defaultdict(dict)
        for from_simplex in self.from_simplices:
            for to_simplex in self.to_simplices_dict[from_simplex]:
                logging.debug('Kernel matrix for %s, %s', from_simplex, to_simplex)
                self._kernel_matrices[from_simplex][to_simplex] = spy(sparsity_message, self._compute_kernel(from_simplex, to_simplex))
                self._kernel_matrices[to_simplex][from_simplex] = sparse.csc_matrix(self._kernel_matrices[from_simplex][to_simplex].H)
        logging.info('Done precomputing kernel matrices')

    def _compute_there_and_back(self, left_simplex, right_simplex):
        left_right_projection = self._kernel_matrices[left_simplex][right_simplex]
        right_cholesky_factor = self._cholesky_factors[right_simplex]
        right_left_projection = self._kernel_matrices[right_simplex][left_simplex]
        there_and_back = left_right_projection @ right_cholesky_factor.solve_A(sparse.csc_matrix(right_left_projection))
        return there_and_back
    
    def _precompute_there_and_back_matrices(self):
        logging.info('Starting precomputing there-and-back matrices')
        self._there_and_back_matrices = defaultdict(dict)
        for from_simplex in self.from_simplices:
            for to_simplex in self.to_simplices_dict[from_simplex]:
                logging.debug('There-and-back for %s, %s', from_simplex, to_simplex)
                self._there_and_back_matrices[from_simplex][to_simplex] = self._compute_there_and_back(from_simplex, to_simplex)
                if self.do_both_there_back:
                    self._there_and_back_matrices[to_simplex][from_simplex] = self._compute_there_and_back(to_simplex, from_simplex)
        logging.info('Done precomputing there-and-back matrices')

    def precompute_matrices(self):
        logging.info('Starting precomputing matrices')
        self._precompute_self_kernel_matrices()
        self._precompute_cholesky_factors()
        self._precompute_kernel_matrices()
        self._precompute_there_and_back_matrices()
        logging.info('Done precomputing matrices')

    def simplex_stalk_dimension(self, simplex):
        return len(self.simplex_points[simplex])
    
    def laplacian(self):
        laplacian_matrix = self._compute_laplacian()
        inner_product = self._compute_inner_product()
        return LaplacianGeneralisedEigenvectorProblem(laplacian_matrix, inner_product)

class OriginalFeatureSelectionProblem(FeatureSelectionProblem):
    def __init__(self, simplicial_complex, simplex_points, kernel, at_simplex):
        self.at_simplex = at_simplex
        from_simplices = set([at_simplex])
        self.all_other_simplices = set(simplicial_complex.simplices()) - from_simplices
        to_simplices = {at_simplex: self.all_other_simplices}
        logging.debug(to_simplices)
        super().__init__(simplicial_complex, simplex_points, kernel, from_simplices, to_simplices, do_both_there_and_back=False)

    def _compute_laplacian(self):
        logging.info('Computing Laplacian matrix')
        degree = len(self.all_other_simplices)
        laplacian_matrix = degree * self._self_kernel_matrices[self.at_simplex]
        for to_simplex in self.all_other_simplices:
            laplacian_matrix -= self._there_and_back_matrices[self.at_simplex][to_simplex]
        logging.info('Finished Laplacian matrix')
        return laplacian_matrix

    def _compute_inner_product(self):
        return self._self_kernel_matrices[self.at_simplex]
    
class DualFeatureSelectionProblem(FeatureSelectionProblem):
    def __init__(self, simplicial_complex, simplex_points, kernel):
        from_simplices = set(simplicial_complex.simplices())
        to_simplices_dict =  {s: set(simplicial_complex.faces(s)) for s in from_simplices}
        super().__init__(simplicial_complex, simplex_points, kernel, from_simplices, to_simplices_dict, do_both_there_and_back=True)

    def _compute_diagonal_block(self, simplex):
        logging.debug(':%s:Computing diagonal block', simplex)
        in_degree = len(self.to_simplices_dict[simplex])
        logging.debug('%s:In degree is %i', simplex, in_degree)
        diagonal_block = 2 * in_degree * self._self_kernel_matrices[simplex]
        for other_simplex in self.all_simplices:
            if other_simplex in self.to_simplices_dict[simplex]:
                logging.debug('%s:%s:Removing there and back', simplex, other_simplex)
                diagonal_block -= self._there_and_back_matrices[simplex][other_simplex]
            elif simplex in self.to_simplices_dict[other_simplex]:
                logging.debug('%s:%s:Adding there and back', simplex, other_simplex)
                diagonal_block += self._there_and_back_matrices[simplex][other_simplex]
            else:
                logging.debug('%s:%s:No contribution', simplex, other_simplex)
        return diagonal_block

    def _check_matrix_shape(self, simplex1, simplex2, nrows, ncols, m):
        if m.shape == (nrows, ncols):
            logging.debug('%s:%s:Shape OK:%s', simplex1, simplex2, m.shape)
        else:
            logging.error('%s:%s:Matrix is wrong shape: Should be %s: Is actually %s', simplex1, simplex2, (nrows, ncols), m.shape)

    def _compute_laplacian(self):
        logging.info('Computing Laplacian matrix')
        laplacian_rows = []
        num_rows = len(self.all_simplices)
        for row_index, row_simplex in enumerate(self.all_simplices):
            dim_row = self.simplex_stalk_dimension(row_simplex)
            logging.debug('Row %i of %i:Simplex %s:Dim row %i', row_index, num_rows, row_simplex, dim_row)
            row = []
            for col_index, col_simplex in enumerate(self.all_simplices):
                dim_col = self.simplex_stalk_dimension(col_simplex)
                logging.debug('%s:%s:Dim row %i:Dim col %i', row_simplex, col_simplex, dim_row, dim_col)
                if row_index == col_index:
                    logging.debug('%s:%s:Add diagonal block matrix shape', row_simplex, col_simplex)
                    matrix = self._compute_diagonal_block(row_simplex)
                elif col_simplex in self.to_simplices_dict[row_simplex]:
                    logging.debug('%s:%s:Add kernel matrix', row_simplex, col_simplex)
                    matrix = - self._kernel_matrices[row_simplex][col_simplex]
                elif row_simplex in self.to_simplices_dict[col_simplex]:
                    logging.debug('%s:%s:Add kernel matrix adjoint', row_simplex, col_simplex)
                    matrix = - self._kernel_matrices[row_simplex][col_simplex]
                else:
                    logging.debug('%s:%s:Add zero matrix', row_simplex, col_simplex)
                    matrix = sparse.csc_matrix((dim_row, dim_col))
                self._check_matrix_shape(row_simplex, col_simplex, dim_row, dim_col, matrix)
                row.append(matrix)
            laplacian_rows.append(row)
        laplacian = sparse.bmat(laplacian_rows, format='csc')
        logging.info('Finished Laplacian matrix')
        return laplacian
    
    def _compute_inner_product(self):
        blocks = [self._self_kernel_matrices[s] for s in self.all_simplices]
        return sparse.block_diag(blocks, format='csc')
    

class CholeskyShiftInverse(sparse.linalg.LinearOperator):
    def __init__(self, sym_sp_matrix, inner_product, shift):
        A = sym_sp_matrix + shift * inner_product
        self.chol_factor = sksparse.cholmod.cholesky(A)
        super().__init__(A.dtype, A.shape)

    def _matvec(self, vec):
        return self.chol_factor.solve_A(vec)
    
EigResult = namedtuple('EigResult', 'evals evecs k tol maxiter shift')

class LaplacianGeneralisedEigenvectorProblem:
    def __init__(self, laplacian_matrix, inner_product):
        self.laplacian_matrix = laplacian_matrix
        self.inner_product = inner_product

    def low_spectrum(self, k, tol, maxiter, shift):
        v0 = np.ones(self.laplacian_matrix.shape[0])
        logging.info('Computing shift inverse')
        OPinv = CholeskyShiftInverse(self.laplacian_matrix, self.inner_product, shift)
        try:
            logging.info('Starting generalised eigenvector problem:k=%i:tol=%f:maxiter=%i:shift=%f', k, tol, maxiter, shift)
            evals, evecs = sparse.linalg.eigsh(
                                                self.laplacian_matrix,
                                                k=k,
                                                M=self.inner_product,
                                                sigma=-shift, 
                                                which='LM', 
                                                v0=v0, 
                                                tol=tol, 
                                                maxiter=maxiter,
                                                OPinv=OPinv
            )
        except sparse.linalg.ArpackNoConvergence as error:
            logging.error(error)
            evals = error.eigenvalues
            evecs = error.eigenvectors    
        logging.info('Finished generalised eigenvector problem')    
        return EigResult(evals, evecs, k, tol, maxiter, shift)