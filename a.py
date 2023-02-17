import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

a = np.asarray([1,1])
b = np.asarray([-1,-1])

def calc_cosine_distance(node1, node2):
    dis = cosine_similarity(np.expand_dims(node1, axis=0), np.expand_dims(node2, axis=0))[0][0]
    return dis

print(calc_cosine_distance(a,b))