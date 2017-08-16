"""Graphing Semantic Network module"""
from memoize import memoize
import numpy as np
from sklearn.datasets import make_blobs
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import interp1d


def rescale(array):
    """Rescales an array between 0. and 1."""
    mn = array.min()
    mx = array.max()
    array_ = array.copy()
    array_ -= mn
    array_ /= (mx-mn)
    return array_


# XXX: optimize this
@memoize
def compute_paths(A, depth):
    """Compute paths of length `depth` in a graph with adjacency matrix A

    Arguments
    ---------
    A : numpy array (n_nodes, n_nodes)
        a binary adjacency matrix
    depth : int
        desired length of the paths

    Returns
    -------
    A_ : numpy array (n_nodes, n_nodes)
        an adjacency matrix with A_[i, j] = 1 if there exists a path of
        length `depth`
        connecting nodes `i` and `j`.
    """
    if depth == 0:
        return np.eye(A.shape[0])
    if depth == 1:
        return A
    As = [A]
    # XXX: this loop can be refactored to speed up computation
    for i_depth in range(1, depth):
        B = np.dot(A, As[-1])
        for A_ in As:
            B[A_ > 0.] = 0.
        B[B > 1] = 1.
        np.fill_diagonal(B, 0)
        As.append(B)
    return As[-1]


def compute_decay(gamma, A, depth):
    """
    Compute decay in a graph, i.e. gamma^depth * A^depth

    Arguments
    ---------
    gamma : float
        decay parameter
    A : np.array (n_nodes, n_nodes)
        adjacency matrix of a graph
    depth : int
        depth of the activation

    Returns
    -------
    D : np.array (n_nodes, n_nodes)
        amount of decay for each node in the graph
    """
    return (gamma ** depth) * compute_paths(A, depth)


def simulate_distance_matrix(n_features, n_concepts, n_clusters, seed=4324):
    """
    Simulate `n_concepts` clustered in `n_clusters`, each of which has
    `n_features`.

    Arguments
    ---------
    n_features : int
    n_concepts : int
    n_clusters : int
    seed : int

    Results
    -------
    dist_x : np.array (n_concepts, n_concepts)
        a correlation matrix ordered according to the clusters
    """
    X, y = make_blobs(n_samples=n_concepts * n_clusters, n_features=n_features,
                      centers=n_clusters, random_state=seed)
    dist_x = 1. - pdist(X, metric='correlation')
    y_sort = np.argsort(y)
    dist_x = squareform(dist_x)[y_sort, :][:, y_sort]
    return dist_x


def normalize_distance_matrix(dist):
    """
    Normalizes a distance matrix across columns.

    Arguments
    ---------
    dist : np.array (n_concepts, n_concepts)

    Returns
    -------
    dist_normalized : np.array (n_concepts, n_concepts)
        a normalized version of dist across columns.
    """
    dist_ = dist.copy()
    dist_ = np.apply_along_axis(rescale, 0, dist_)
    dist_ /= dist_.sum(axis=0)
    return dist_


def compute_nmph(min_y, inflection_x, inflection_y, y_max):
    """
    Compute Non Monotonic Plasticity Hypothesis function

    Arguments
    ---------
    min_y : float [-1, 1]
        minimum value of the nmph function
    inflection_x : float [0, 1]
        x-coordinate of the point of inflection
    inflection_y : float [-1, 1]
        y-coordinate of the point of inflection
    y_max : float [-1, 1]
        y-max of nmph when x = 1

    Returns
    -------
    nmph : nmph function [0, 1] -> [-1, 1]
    """
    min_x = inflection_x / 2.
    x = [0, min_x, inflection_x, 1.]
    y = [0, min_y, inflection_y, y_max]
    nmph = interp1d(x, y)
    return nmph
