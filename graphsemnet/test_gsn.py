import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from .gsn import rescale, compute_paths, compute_decay, compute_nmph
A = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])
A2 = np.array([
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0]
])
A3 = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]]
)
A4 = np.zeros((4, 4))


def test_compute_paths():
    assert_array_equal(np.eye(A.shape[0]), compute_paths(A, 0))
    assert_array_equal(A, compute_paths(A, 1))
    assert_array_equal(A2, compute_paths(A, 2))
    assert_array_equal(A3, compute_paths(A, 3))
    for i in range(4, 9):
        assert_array_equal(A4, compute_paths(A, i))


def test_compute_decay():
    gamma = 0.
    for i in range(1, 4):
        assert_array_equal(np.zeros(A.shape), compute_decay(gamma, A, i))
    gamma = 0.5
    for i, A_true in enumerate([np.eye(A.shape[0]), A, A2, A3, A4]):
        assert_array_equal(A_true * (gamma**i), compute_decay(gamma, A, i))


def test_rescale():
    array = np.random.randn(10, 1000)
    assert((array.min(), array.max()) != (0., 1.))
    array = rescale(array)
    assert((array.min(), array.max()) == (0., 1.))


def test_compute_nmph():
    min_y, inflection_x, inflection_y, max_y = [-.5, 0.5, 0.3, 0.6]
    nmph = compute_nmph(min_y, inflection_x, inflection_y, max_y)
    # start
    assert_almost_equal(nmph(0.), 0.)
    # inflection
    assert_almost_equal(nmph(inflection_x), inflection_y)
    # minimum should be at inflection_x / 2.
    assert_almost_equal(nmph(inflection_x/2.), min_y)
    # max
    assert_almost_equal(nmph(1.), max_y)
