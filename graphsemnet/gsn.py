"""Graphing Semantic Network module"""
import numpy as np


# XXX: optimize this
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
