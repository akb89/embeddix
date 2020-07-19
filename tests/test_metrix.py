"""Test utils.metrix."""

import numpy as np
import scipy.spatial as spatial
import scipy.sparse as sparse
import embeddix.utils.metrix as metrix


def test_energy():
    singvalues = np.array([0, 1, 2, 3, 4, 5])
    assert metrix.energy(singvalues) == 55
    matrix = np.array([[0, 1, 0, 0, 2], [0, 1, 0, 0, 3]])
    matrix = sparse.csr_matrix(matrix)
    print(type(matrix))
    assert matrix.__class__.__name__ == 'csr_matrix'
    assert metrix.energy(matrix) == 15


def test_similarity():
    matrix1 = np.array([[0, 1, 2, 3], [0, 1, 1, 1]], dtype='f')
    matrix2 = np.array([[1, 0, 0, 0], [0, 1, 2, 3]], dtype='f')
    assert matrix1.shape == (2, 4)
    assert matrix1.shape[1] == matrix2.shape[1] == 4
    simsim = 1 - spatial.distance.cdist(matrix1, matrix2, 'cosine')
    sim_cos = np.diagonal(simsim)
    np.testing.assert_array_almost_equal(metrix.similarity(matrix1, matrix2),
                                         sim_cos)
