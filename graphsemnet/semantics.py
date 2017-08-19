import random
from scipy.spatial.distance import pdist
import numpy as np


def connected_word_list(seed_word, keyed_vectors, n_iterations=5):
    all_words = [seed_word]
    for i in range(n_iterations):
        seed_word = random.sample(all_words, 1)
        nearest_neighbors = list(
            np.array(
                keyed_vectors.most_similar(seed_word)
            )[:, 0]
        )
        all_words += nearest_neighbors
    return all_words


def semantic_dsm(word_list, keyed_vectors):
    vectors = np.array([keyed_vectors.word_vec(word) for word in word_list])
    dsm = np.clip(pdist(vectors, metric='cosine'), 0, 1)
    return dsm


def semantic_dsm_safe(word_list, keyed_vectors):
    vectors = []
    labels = []
    for word in word_list:
        try:
            vectors.append(keyed_vectors.word_vec(word))
        except:
            pass
        else:
            labels.append(word)
    vectors = np.array(vectors)
    matrix = pdist(vectors, metric='cosine')
    return (matrix, labels)
