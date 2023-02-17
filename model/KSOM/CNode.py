import numpy as np

from  model.KSOM.Node import *

class CNode(Node):
    def __init__(self, num_dimension):
        super().__init__(np.random.normal(0, 1, num_dimension))
        self.PNodes = []
        # print("One Cnode")
        # print(self.dWeights)
        # print("------------")

    def AdjustWeights(self, target_PNode, LearningRate, Influence):
        self.vector += LearningRate * Influence * (target_PNode.get_vector() - self.vector)

    def addPNode(self, inputNode):
        self.PNodes.append(inputNode)
        return
    
    def print_PNodes(self):
        if len(self.PNodes) == 0:
            print("No corpus")
            return
        for node in self.PNodes:
            print((node.corpus, node.label))
