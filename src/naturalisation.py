from collections import namedtuple, defaultdict
import numpy as np
from scipy import sparse
from sksparse import cholmod
import topology


class SheafSubspaceProblem:
    def __init__(self, simplicial_complex, simplex_points, kernel):
        self.simplicial_complex = simplicial_complex
        self.simplex_points = simplex_points
        self.kernel = kernel
    
    def _precompute_submatrices(self):
        seen = set()
        self.up_top_degrees = {}
        self.locally_top_face_covariances = {}
        self.non_locally_top_cholesky_factors = {}
        for dim in reversed(sorted(self.simplicial_complex.dimensions())): # need to not access private
            for simplex in self.simplicial_complex.simplices(dim):
                if simplex not in seen:
                    seen.add(simplex)
                    faces = self.simplicial_complex.faces(simplex)
                    face_covariances = {}
                    for face in faces:
                        if face not in seen:
                            self.up_top_degrees[face] = 0
                            self.non_locally_top_cholesky_factors[face] = self._cholesky_factor(face)
                            seen.add(face)
                        self.up_top_degrees[face] += 1
                        face_covariances[face] = self._covariance(simplex, face)
                    self.locally_top_face_covariances[simplex] = face_covariances
        self.locally_top_simplices = list(self.locally_top_face_covariances.keys())
    
    def _covariance(self, target, source):
        return sparse.csc_matrix(self.kernel(self.simplex_points[target], self.simplex_points[source]))
    
    def _cholesky_factor(self, simplex):
        self_covariance = self._covariance(simplex, simplex)
        return cholmod.cholesky(self_covariance)
    
    @staticmethod
    def _compute_row_col_contrib(left, cholesky_factor, right):
        return left @ cholesky_factor.solve_A(sparse.csc_matrix(right.T))
    
    def simplex_stalk_dimension(self, simplex):
        return len(self.simplex_points[simplex])
    
    def _compute_adjacency(self):
        # Compute lower triangular block matrix
        laplacian_lower_rows = []
        for row_index in range(len(self.locally_top_simplices)):
            row_simplex = self.locally_top_simplices[row_index]
            row_faces = self.locally_top_face_covariances[row_simplex].keys()
            lower_row = []
            for col_index in range(row_index):
                col_simplex = self.locally_top_simplices[col_index]
                col_faces = self.locally_top_face_covariances[col_simplex].keys()
                nrows = self.simplex_stalk_dimension(row_simplex)
                ncols = self.simplex_stalk_dimension(col_simplex)
                entry = sparse.csc_array((nrows, ncols))
                for face in row_faces & col_faces:
                    left = self.locally_top_face_covariances[row_simplex][face]
                    cholesky_factor = self.non_locally_top_cholesky_factors[face]
                    right = self.locally_top_face_covariances[col_simplex][face]
                    degree_inv = 1 / self.up_top_degrees[face]
                    entry += degree_inv * self._compute_row_col_contrib(left, cholesky_factor, right)
                lower_row.append(entry)
            laplacian_lower_rows.append(lower_row)
        # Fill in the rest of the matrix using transposes
        for row_index in range(len(self.locally_top_simplices)):
            row = laplacian_lower_rows[row_index]
            row_simplex = self.locally_top_simplices[row_index]
            for col_index in range(row_index, len(self.locally_top_simplices)):
                col_simplex = self.locally_top_simplices[row_index]
                if col_index == row_index:
                    dim = self.simplex_stalk_dimension(row_simplex)
                    row.append(sparse.csc_matrix((dim, dim)))
                else:
                    row.append(laplacian_lower_rows[col_index][row_index].T)
        adjacency = sparse.bmat(laplacian_lower_rows)
        return adjacency
    
    def _compute_diagonal(self):
        diagonal_blocks = []
        for simplex in self.locally_top_simplices:
            simplex_covariance = self.locally_top_simplex_covariances[simplex]
            nrows = self.simplex_stalk_dimension(simplex)
            entry = sparse.csc_array((nrows, nrows))
            faces = self.locally_top_face_covariances[simplex].keys()
            for face in faces:
                left = self.locally_top_face_covariances[simplex][face]
                cholesky_factor = self.non_locally_top_cholesky_factors[face]
                degree_inv = 1 / self.up_top_degrees[face]
                entry += simplex_covariance - degree_inv * self._compute_row_col_contrib(left, cholesky_factor, left)
            diagonal_blocks.append(entry)
        diagonal = sparse.block_diag(diagonal_blocks)
        return diagonal
    
    def laplacian(self):
        self._precompute_submatrices()
        self.locally_top_simplices = list(self.locally_top_face_covariances.keys())
        self.locally_top_simplex_covariances = {simplex: self._covariance(simplex, simplex) for simplex in self.locally_top_simplices}
        adjacency = self._compute_adjacency()
        diagonal = self._compute_diagonal()
        laplacian = diagonal - adjacency
        return Laplacian(laplacian, self)
                        

class CholeskyShiftInverse(sparse.linalg.LinearOperator):
    def __init__(self, sym_sp_matrix, shift):
        self.sym_sp_matrix = sym_sp_matrix
        self.shift = shift
        self.chol_factor = cholmod.cholesky(self.sym_sp_matrix, beta=shift)
        super().__init__(sym_sp_matrix.dtype, sym_sp_matrix.shape)

    def _matvec(self, vec):
        return self.chol_factor.solve_A(vec)
    

EigResult = namedtuple('EigResult', 'evals evecs k tol maxiter shift')


class Laplacian:
    def __init__(self, laplacian, problem):
        self.laplacian = laplacian
        self.problem = problem
        self.non_locally_top_cofaces = defaultdict(list)
        for top_simplex, faces in self.problem.locally_top_face_covariances.items():
            for face in faces:
                self.non_locally_top_cofaces[face].append(top_simplex)
        
    def low_spectrum(self, k, tol, maxiter, shift):
        v0 = np.ones(self.laplacian.shape[0])
        OPinv = CholeskyShiftInverse(sparse.csc_matrix(self.laplacian), shift)
        inner_product = sparse.block_diag([self.problem.locally_top_simplex_covariances[s] for s in self.problem.locally_top_simplices])
        try:
            evals, evecs = sparse.linalg.eigsh(
                                            self.laplacian,
                                            k=k,
                                            M=inner_product,
                                            sigma=-shift, 
                                            which='LM', 
                                            v0=v0, 
                                            tol=tol, 
                                            maxiter=maxiter,
                                            OPinv=OPinv)
        except sparse.linalg.ArpackNoConvergence as error:
            print(error)
            evals = error.eigenvalues
            evecs = error.eigenvectors
        evecs_dict = self._top_array_to_dict(evecs)
        return EigResult(evals, evecs_dict, k, tol, maxiter, shift)

    def _top_array_to_dict(self, array):
        array_dict = {}
        start_index = 0
        for simplex in self.problem.locally_top_simplices:
            end_index = start_index + self.problem.simplex_stalk_dimension(simplex)
            array_dict[simplex] = array[start_index:end_index]
            start_index = end_index
        return array_dict
    
    def extend_top_array_dict(self, array_dict):
        dim = array_dict[self.problem.locally_top_simplices[0]].shape[1]
        for face, cofaces in self.non_locally_top_cofaces.items():
            array = np.zeros((self.problem.simplex_stalk_dimensions(face), dim))
            cholesky_factor = self.problem.non_locally_top_cholesky_factors[face]
            for coface in cofaces:
                B = self.problem.locally_top_face_covariances[coface][face].T @ array_dict[coface]
                array += 1 / self.problem.up_top_degrees[face] * cholesky_factor.solve_A(B)
            array_dict[face] = array
        return array_dict