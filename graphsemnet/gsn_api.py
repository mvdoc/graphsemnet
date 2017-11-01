import copy
import numpy as np


class GraphOperator(object):
    """Wrapper object for functions that activate and reweight a semantic graph
    """
    def __init__(self, graph, operate_fx, xcal_fx, decay):
        """Create a wrapper object for graph activation and reweighting.

        Arguments
        ---------
        graph : SemanticGraph
        operate_fx : function [graph, activations, xcal_fx, decay] ->
                [SemanticGraph]
            Function describing how to activate and reweight the graph
        xcal_fx : function [0, 1] -> float
            Scaling function for edge reweighting
        decay : float
            Decay coefficient for activation propagation in operate_fx
        """
        self.graph = graph
        self.operate_fx = operate_fx
        self.xcal_fx = xcal_fx
        self.decay = decay

    def activate(self, activations):
        """Call operate_fx with stored arguments and return result."""
        return self.operate_fx(
            self.graph, activations, self.xcal_fx, self.decay
        )

    def activate_replace(self, activations):
        """Replace self.graph with operate_fx result."""
        self.graph = self.activate(activations)
        return self.graph


def operate_recur(graph, activations, xcal, decay):
    """Wrapper for depth-first activation and reweighting."""
    new_activations = propagate_recur(graph, activations, xcal, decay)
    return reweight_recur(graph, new_activations, xcal)


def operate_depth(graph, activations, xcal, decay):
    """Wrapper for breadth-first activation and reweighting."""
    activations = activations[None, :]
    Ws, ACT = spread_activation(graph.adj, activations, xcal, gamma=decay)
    result_graph = copy.deepcopy(graph)
    result_graph.adj = Ws[-1]
    return result_graph


def propagate_recur(graph, activations, xcal, decay=0.8, new_adj=None,
                    debug=False):
    """Fire together?

    activations: 1d vector of activation strengths for each node
    """
    if new_adj is None:
        new_adj = copy.copy(graph.adj)
    else:
        new_adj = copy.copy(new_adj)

    new_adj[:, xcal(activations) != 0] = 0
    downstream_activations = np.clip(
        np.dot(new_adj.T, activations) * decay, 0, 1
    )

    # not necessary?
    downstream_activations[xcal(downstream_activations) == 0] = 0

    if debug:
        print(f"init acts: {activations}")
        print(f"down_acts: {downstream_activations}")

    if np.all(downstream_activations == 0):
        return activations
    else:
        return activations + propagate_recur(
            graph, downstream_activations, xcal,
            decay=decay, new_adj=new_adj, debug=debug
        )


def reweight_recur(graph, activations, xcal, debug=False):
    """Wire together.

    activations: 1d vector of actiation strengths for each node
    """
    # for each node, change all of its input weights according to the xcal
    # modify both weights according to the minimum activation of the nodes
    if debug:
        print(f"input activations: {activations}")

    new_graph = copy.deepcopy(graph)

    for i, act_from in enumerate(activations):
        for j, act_to in enumerate(activations):
            min_act = np.min([act_from, act_to])
            new_graph.adj[i, j] += xcal(min_act)
            new_graph.adj[j, i] += xcal(min_act)

    np.fill_diagonal(new_graph.adj, 0)
    new_graph.adj = np.clip(new_graph.adj, 0, 1)
    return new_graph


def compute_adjacency(W):
    """Given a weight matrix W, return an adjacency matrix"""
    A = W.copy()
    A[A > 0.] = 1.
    return A


def rect(x):
    """ReLU function"""
    return np.clip(x, 0, 1.)


def spread_activation(W0, ACT0, nmph, gamma=0.8, alpha=0.5, lambda_=1, d=3):
    """Spread activation with NMPH on a graph with initial weight W0,
    and activation ACT0

    Arguments
    ---------
    W0 : array (n_nodes, n_nodes)
        initial weights of the graph
    ACT0 : array (1, n_nodes)
        row array of initial activations
    npmh : function [0, 1] -> [-1, 1]
        nmph function, generated with `compute_nmph`
    gamma : float [0, 1]
        decay parameter
    alpha : float [0, 1]
        learning parameter
    lambda_ : float [0, 1]
        decay parameter for past activations
    d : int
        how far the activation is allowed to spread

    Returns
    -------
    Ws : list of arrays (n_nodes, n_nodes)
    ACT : list of arrays
        activations for every depth
    """
    # initialize values
    assert (ACT0.ndim == 2 and (ACT0.shape[0] <= ACT0.shape[1]))
    A = compute_adjacency(W0)
    Ws = [W0]
    ACT = [ACT0]
    dACT = np.zeros(ACT0.shape)

    # loop
    for i in range(1, d):
        # update W
        W_i = rect(Ws[-1] + alpha * nmph(ACT[-1]).T * A)
        Ws.append(W_i)
        # update ACT
        dACT += gamma ** i * np.dot(ACT[-1], np.multiply.reduce(Ws))
        ACT_ = rect((lambda_**i) * ACT[-1] + dACT)
        ACT.append(ACT_)
        #print("Loop {0}: len(Ws): {1}\tlen(ACT): {2}".format(i, len(Ws), len(ACT)))
    return Ws, ACT

