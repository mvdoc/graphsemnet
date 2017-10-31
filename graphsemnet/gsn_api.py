import copy
import numpy as np


class GraphOperator(object):
    def __init__(self, graph, operate_fx):
        self.graph = graph
        self.operate_fx = operate_fx

    def activate(self, activations, xcal):
        return self.operate_fx(self.graph, activations)


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
    down_acts = np.clip(np.dot(new_adj.T, activations) * decay, 0, 1)
    down_acts[xcal(down_acts) == 0] = 0  # not necessary?

    if debug:
        print(f"init acts: {init_acts}")
        print(f"down_acts: {down_acts}")

    if np.all(down_acts == 0):
        return activations
    else:
        return activations + activate_hebb_recur(
            down_acts, decay=decay, new_adj=new_adj, debug=debug
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
