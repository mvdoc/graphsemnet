import copy
import numpy as np


class GraphOperator(object):
    def __init__(self, graph, operate_fx):
        self.graph = graph
        self.operate_fx = operate_fx

    def activate(self, activations, xcal, decay):
        return self.operate_fx(self.graph, activations, xcal, decay)

    def activate_replace(self, activations, xcal, decay):
        self.graph = self.activate(self.graph, activations, xcal, decay)
        return self.graph


def operate_recur(graph, activations, xcal, decay):
    new_activations = propagate_recur(graph, activations, xcal, decay)
    return reweight_recur(graph, new_activations, xcal)


def propagate_recur(graph, activations, xcal, decay=0.8, new_adj=None,
                    debug=False):
    """Fire together?

    init_activations: 1d vector of activation strengths for each node
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
        print(f"init acts: {init_acts}")
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

    activations: 1d vector of actiation strengths for each node, used for
                 weight updating.
    """
    # for each node, change all of its input weights according to the xcal
    # modify both weights according to the minimum activation of the nodes
    if debug:
        print(f"adjust input acts: {activations}")

    new_graph = copy.deepcopy(graph)

    for i, act_from in enumerate(activations):
        for j, act_to in enumerate(activations):
            min_act = np.min([act_from, act_to])
            new_graph.adj[i, j] += xcal(min_act)
            new_graph.adj[j, i] += xcal(min_act)

    np.fill_diagonal(new_graph.adj, 0)
    new_graph.adj = np.clip(new_graph.adj, 0, 1)
    return new_graph
