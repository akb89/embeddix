"""Test align model."""
import numpy as np

import embeddix


def test_align_model():
    A = np.array([[0], [1], [2], [3], [4], [5]])
    vocab = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'X': 4, 'Z': 5}
    shared = {'D': 0, 'C': 1, 'B': 2, 'A': 3}
    model = embeddix.reduce_dense(A, vocab, shared)
    assert model.shape[0] == 4
    assert model.shape[1] == 1
    assert model[0] == A[3]
    assert model[1] == A[2]
    assert model[2] == A[1]
    assert model[3] == A[0]
