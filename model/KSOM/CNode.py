import numpy as np
import scipy
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
        # edge_list = []
        # for node_idx in self.PNodes:
        #     for node_idx_neighbor in self.PNodes:
        #         if node_idx_neighbor==node_idx:
        #             continue
        #         else:
        #             if method == 'cosine':
        #                 calc_distance = model_ksom.calc_cosine_distance
        #             else:
        #                 calc_distance = model_ksom.calc_euclid_distance
        #             dist_2node = calc_distance(self.PNodes[node_idx], self.PNodes[node_idx_neighbor])
        #             if dist_2node > 0.15:
        #                 continue
        #             else:
        #                 edge_list.append((node_idx, node_idx_neighbor, {'edge_weight': dist_2node, 'edge_type': 1}))
        # return edge_list
        length = len(self.PNodes)
        if length == 0:
            print("No have node")
            return np.empty((0,3))

        # print(length)
        nodeidx = {}
        node_arr = [0]*length
        i = 0
        for x in self.PNodes:
            nodeidx[i] = x
            node_arr[i] = self.PNodes[x].get_vector()
            i+=1
        node_arr = np.asarray(node_arr)
        print(node_arr.shape)
        dist = scipy.spatial.distance.pdist(node_arr, metric='euclidean')
        num_edge = length*(length-1)/2
        top_k = int(3*num_edge/4)
        top_idx = np.argpartition(dist, -top_k)[-top_k:]
        row,col = np.triu_indices(length,k=1)
        lst_edge = np.concatenate((row[top_idx].reshape(-1,1), col[top_idx].reshape(-1,1)), axis=1)
        if lst_edge.shape[0]==0:
            print("Not have any edge")
            # print(top_idx)
            return np.empty((0,3))
        mapping_fn = lambda x: nodeidx[x]
        vec_mapfn = np.vectorize(mapping_fn)
        lst_edge = vec_mapfn(lst_edge)
        edge_weight = dist[top_idx]
        edge_att = lambda x: {'edge_weight':x,'edge_type':1}
        vec_edge_att = np.vectorize(edge_att)
        lst_edge_att = vec_edge_att(edge_weight).reshape(-1, 1)
        lst_edge = np.concatenate((lst_edge, lst_edge_att), axis=1)
        return lst_edge


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


