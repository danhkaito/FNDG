import numpy as np

class Node:
    def __init__(self, vetcor):
        self.vector = vetcor
    
    def get_vector(self):
        return self.vector

    def __str__(self):
        return np.array2string(self.vector)