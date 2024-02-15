import warnings
from collections import defaultdict
from itertools import chain, combinations
from ordered_set import OrderedSet
from sklearn.neighbors import NearestNeighbors


class SimplicialComplex(object):
    def __init__(self):
        self._simplices = defaultdict(OrderedSet)

    def dimensions(self):
        for i in self._simplices.keys():
            yield i
    
    def simplices(self, dim=None):
        if dim is not None:
            if dim in self._simplices:
                for simplex in self._simplices[dim]:
                    yield simplex
        else:
            for dim in self._simplices:
                for simplex in self._simplices[dim]:
                    yield simplex

    def __iter__(self):
        return self.simplices()

    def index(self, simplex):
        return self._simplices[self.dimension(simplex)].index(simplex)

    def __getitem__(self, dim_index):
        if type(dim_index) is int:
            if dim_index in self._simplices:
                return self._simplices[dim_index]
            else:
                raise IndexError(f'no simplices of dimension {dim_index} exist')
        else:
            dim, index = dim_index
            return self._simplices[dim][index]

    @staticmethod
    def dimension(simplex):
        return len(simplex) - 1

    @staticmethod
    def faces(simplex, codim=None):
        if codim is None:
            return chain.from_iterable(combinations(simplex, r) for r in range(1, len(simplex)))
        else:
            return combinations(simplex, len(simplex) - codim)
    
    @staticmethod
    def isface(face, coface):
        return set(face).issubset(coface)
    
    def cofaces(self, simplex):
        cofaces = []
        for higher_simplex in self.simplices(self.dimension(simplex) + 1):
            if SimplicialComplex.isface(simplex, higher_simplex):
                cofaces.append(higher_simplex)
        return cofaces
    
    @staticmethod
    def are_face_coface(simplex_a, simplex_b):
        face, coface = sorted([simplex_a, simplex_b], key=len)
        return SimplicialComplex.isface(face, coface)
        

    @staticmethod
    def boundary_coeff(face, coface):
        for i, vertex in enumerate(coface):
            if vertex not in face:
                return (-1) ** i

    def _insert_simplex(self, simplex):
        self._simplices[self.dimension(simplex)].add(simplex)

    def has_simplex(self, simplex):
        return simplex in self._simplices[self.dimension(simplex)]
        
    def add_simplex(self, simplex):
        if not self.has_simplex(simplex):
            self._insert_simplex(simplex)
            faces = self.faces(simplex)
            if faces:
                for face in faces:
                    self.add_simplex(face)

    def __len__(self):
        return len(list(self.simplices()))


class Cover(object):
    def __init__(self, data, open_sets):
        self.data = data
        self.open_sets = open_sets # ordered set of frozenset of indices of the data
        if not self._is_cover():
            raise ValueError('open_sets does not cover data')

    def _is_cover(self):
        return len(frozenset.union(*self.open_sets)) == len(self.data)

    def open_set_points(self, open_set):
        return self.data[list(open_set)]

    def __len__(self):
        return len(self.open_sets)
        
        
class BallCover(Cover):
    def __init__(self, data, ball_query, query_points):
        open_sets = OrderedSet()
        for point in query_points:
            indices = ball_query(point)
            open_set = frozenset(indices.ravel())
            open_sets.append(open_set)
        super().__init__(data, open_sets)

    
class KNNCover(BallCover):
    def __init__(self, data, n, landmarks=None):
        self.n = n
        nearest_neighbours = NearestNeighbors(n_neighbors=n)
        nearest_neighbours.fit(data)
        def ball_query(point):
            indices = nearest_neighbours.kneighbors(point.reshape(1, -1), return_distance=False)
            return indices
        super().__init__(data, ball_query, data)


class ProximityCover(BallCover):
    def __init__(self, data, radius, landmarks=None):
        self.radius = radius
        self.landmarks = landmarks or data
        nearest_neighbours = NearestNeighbors(radius=radius)
        nearest_neighbours.fit(data)
        def ball_query(point):
            indices = nearest_neighbours.radius_neighbors(point.reshape(1, -1),
                                                          return_distance=False)[0]
            return indices
        super().__init__(data, ball_query, data)
    

class SimplicialComplexCoverFunctor(object):
    def __init__(self, simplicial_complex, cover):
        self.simplicial_complex = simplicial_complex
        self.cover = cover
        self.simplex_open_set = {
            simplex: frozenset() for simplex in self.simplicial_complex.simplices()
            }

    def simplex_points(self, simplex):
        open_set = self.simplex_open_set[simplex]
        points = self.cover.open_set_points(open_set)
        return points

    
class Nerve(SimplicialComplexCoverFunctor):
    def __init__(self, cover):
        self.cover = cover
        self.simplicial_complex = self._construct_simplicial_complex(self.cover)
        self.simplex_open_set = self._construct_functor(self.simplicial_complex, self.cover)
        
    @staticmethod
    def _construct_simplicial_complex(cover):
        simplicial_complex = SimplicialComplex()
        for point_index in range(len(cover.data)):
            covering_sets_indices = []
            for open_set_index, open_set in enumerate(cover.open_sets):
                if point_index in open_set:
                    covering_sets_indices.append(open_set_index)
            simplex = tuple(covering_sets_indices)
            simplicial_complex.add_simplex(simplex)
        return simplicial_complex

    @staticmethod
    def _construct_functor(simplicial_complex, cover):
        simplex_open_set = {}
        for simplex in simplicial_complex.simplices():
            open_sets = [cover.open_sets[i] for i in simplex]
            intersection = frozenset.intersection(*open_sets)
            simplex_open_set[simplex] = intersection
        return simplex_open_set

    
class UnionSimplicialComplexCoverFunctor(SimplicialComplexCoverFunctor):
    def __init__(self, simplicial_complex, cover, vertices_to_open_set_indices):
        super().__init__(simplicial_complex, cover)
        for simplex in self.simplicial_complex.simplices():
            open_sets = [
                self.cover.open_sets[vertices_to_open_set_indices[vertex]] for vertex in simplex
            ]
            union = frozenset.union(*open_sets)
            self.simplex_open_set[simplex] = union
