import numpy as np

from  model.KSOM.Node import *


class CNode(Node):
    def __init__(self, num_dimension):
        super().__init__(np.random.normal(0, 1, num_dimension))
        self.PNodes = {}
        # print("One Cnode")
        # print(self.dWeights)
        # print("------------")

    def AdjustWeights(self, target_PNode, LearningRate, Influence):
        self.vector += LearningRate * Influence * (target_PNode.get_vector() - self.vector)

    def addPNode(self, inputNode, idx):
        self.PNodes[idx]= inputNode
        return
    
    def create_edge_subgraph(self, model_ksom, method = 'cosine'):
        edge_list = []
        for node_idx in self.PNodes:
            for node_idx_neighbor in self.PNodes:
                if node_idx_neighbor==node_idx:
                    continue
                else:
                    if method == 'cosine':
                        calc_distance = model_ksom.calc_cosine_distance
                    else:
                        calc_distance = model_ksom.calc_euclid_distance
                    dist_2node = calc_distance(self.PNodes[node_idx], self.PNodes[node_idx_neighbor])
                    if dist_2node < 0.7:
                        continue
                    else:
                        edge_list.append((node_idx, node_idx_neighbor, {'edge_weight': dist_2node, 'edge_type': 1}))
        return edge_list

    def __str__(self):
        true_label = 0
        false_label = 0
        if len(self.PNodes) == 0:
            return ("No corpus")
            
        strnode = ""
        for node in self.PNodes:
            strnode += str(self.PNodes[node])
            if self.PNodes[node].label == True:
                true_label += 1
            else:
                false_label += 1
        return strnode + "SUMMARY: " + str(true_label) + " , " + str(false_label) + "\n"


