import copy
import numpy as np
from scipy.spatial.distance import squareform
from scipy.interpolate import interp1d


def get_xcal(dip_center, dip_width, min_adjust, max_adjust):
    dip_half_width = dip_width / 2
    xp = [
        0,
        dip_center - dip_half_width,
        dip_center,
        dip_center + dip_half_width,
        1
    ]
    fp = [
        0,
        0,
        min_adjust,
        0,
        max_adjust
    ]
    return interp1d(xp, fp)


class SemanticGraph(object):
    def __init__(self,
                 adj=None,
                 labels=None,
                 word_dsm=None,
                 xcal_dip_center=.2,
                 xcal_dip_width=.15,
                 xcal_min_adjust=-.05,
                 xcal_max_adjust=.05,
                 directed=True):
        if word_dsm is None:
            self.adj = np.array(adj)
        else:
            if len(word_dsm.shape) == 1:
                word_dsm = squareform(word_dsm)
            word_adj = 1 - word_dsm
            np.fill_diagonal(word_adj, 0)
            self.adj = np.array(word_adj)
        self.labels = labels
        self.set_xcal(
            xcal_dip_center,
            xcal_dip_width,
            xcal_min_adjust,
            xcal_max_adjust
        )
        self.directed = directed

    def set_xcal(self,
                 dip_center=.2,
                 dip_width=.15,
                 min_adjust=-.05,
                 max_adjust=.05):
        self.xcal = get_xcal(
            dip_center,
            dip_width,
            min_adjust,
            max_adjust
        )
        self.xcal_dip_center = dip_center
        self.xcal_dip_width = dip_width
        self.xcal_min_adjust = min_adjust
        self.xcal_max_adjust = max_adjust
        self.lower_threshold = dip_center - (dip_width / 2)

    def activate_hebb(self, init_acts, decay=0.8, new_adj=None, debug=False):
        """Fire together?

        init_activations: 1d vector of activation strengths for each node
        """
        if new_adj is None:
            new_adj = copy.copy(self.adj)
        else:
            new_adj = copy.copy(new_adj)

        pass_thresh = self.xcal_dip_center - (self.xcal_dip_width / 2)
        new_adj[:, init_acts > pass_thresh] = 0

        down_acts = np.clip(np.dot(new_adj.T, init_acts) * decay, 0, 1)

        if debug:
            print(f"init acts: {init_acts}")
            print(f"down_acts: {down_acts}")
        down_acts[down_acts < pass_thresh] = 0

        if np.all(down_acts == 0):
            return init_acts
        else:
            return init_acts + self.activate_hebb(
                down_acts, decay=decay, new_adj=new_adj, debug=debug
            )

    def weight_adjust_hebb(self, acts, debug=False):
        """Wire together.

        acts: 1d vector of actiation strengths for each node, used for weight
              updating.
        """
        # for each node, change all of its input weights according to the xcal
        # modify both weights according to the minimum activation of the nodes
        if debug:
            print(f"adjust input acts: {acts}")
        pass_thresh = self.xcal_dip_center - (self.xcal_dip_width / 2)
        for i, act_from in enumerate(acts):
            for j, act_to in enumerate(acts):
                min_act = np.min([act_from, act_to])
                if min_act > pass_thresh:
                    self.adj[i, j] += self.xcal(min_act)
                    self.adj[j, i] += self.xcal(min_act)

        np.fill_diagonal(self.adj, 0)
        self.adj = np.clip(self.adj, 0, 1)

    def activate(self,
                 word_from,
                 strength=1.0,
                 decay=.8,
                 new_adj=None,
                 debug=False):
        """Spread activation, adjusting synaptic weights."""
        if debug:
            print("{0} at strength {1}".format(word_from, strength))

        # Make a copy of the input adjacency matrix to send downstream
        if new_adj is None:
            new_adj = copy.copy(self.adj)
        else:
            new_adj = copy.copy(new_adj)

        word_from_i = self.labels.index(word_from)
        next_layer_activations = new_adj[word_from_i] * strength * decay
        if debug:
            print(next_layer_activations.round(2))

        # Prevent loops by zeroing out weights back to already-activated nodes
        new_adj[:, word_from_i] = 0

        for word_to_i, activation in enumerate(next_layer_activations):
            if activation > self.lower_threshold:

                # Update edge weights
                word_to = self.labels[word_to_i]
                change = self.xcal(activation)
                if debug:
                    print("Updating edge: {0} -> {1}, + {2}".format(
                        word_from, word_to, change
                    ))
                self.adj[word_from_i, word_to_i] += change
                if not self.directed:
                    self.adj[word_to_i, word_from_i] = self.adj[
                        word_from_i, word_to_i
                    ]

                self.adj = np.clip(self.adj, 0, 1)

                self.activate(
                    self.labels[word_to_i],
                    strength=activation,
                    new_adj=new_adj,
                    debug=debug
                )