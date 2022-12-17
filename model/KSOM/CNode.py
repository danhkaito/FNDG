import numpy as np
from model.KSOM.PNode import *
import math
from sklearn.metrics.pairwise import cosine_similarity

class CNode:
    def __init__(self, num_component, *numWeights):
        self.dWeights = [0]*num_component
        for i in range(0, len(numWeights)):
            self.dWeights[i] = np.random.normal(0, 1, numWeights[i])
        self.num_component=num_component
        self.PNodes = {}
        self.represent_vector=None
        self.dWeights=np.array(self.dWeights, dtype=object)
        # print("One Cnode")
        # print(self.dWeights)
        # print("------------")
        # self.idx=None

    def CalculateDistance_PNode2CNode(self, InputNode: PNode, bias):
        dis=0
        for i in range (0, len(bias)):
            temp = self.dWeights[i] - InputNode.getvector(i)
            # print(InputNode.getvector(i))
            sum_sq = np.dot(temp.T, temp)
            # print(sum_sq)
            # print(bias[i])
            # print(math.sqrt(sum_sq))
            dis_temp=math.sqrt(sum_sq/len(temp))*bias[i]
            # print(f"Feature {i+1}: {mth.sqrt(sum_sq)*bias[i]}")
            dis+=dis_temp
            # print(math.sqrt(sum_sq)*np.int(bias[i]))
        return dis

    def CalculateDistance2CNode (self, InputCNode, bias):
        dis=0
        for i in range (0, len(bias)):
            temp=self.dWeights[i] - InputCNode.dWeights[i]
            sum_sq = np.dot(temp.T, temp)
            dis+=math.sqrt(sum_sq/len(temp))*bias[i]
        return dis

    def CalculateCosinePNode2CNode(self, InputNode: PNode, bias):
        dis=0
        for i in range (0, len(bias)):
            temp = cosine_similarity([self.dWeights[i]], [InputNode.getvector(i)])[0][0]
            dis_temp=temp*bias[i]
            dis+=dis_temp
        return dis

    def CalculateCosine2CNode (self, InputCNode, bias):
        dis=0
        for i in range (0, len(bias)):
            temp= cosine_similarity ([self.dWeights[i]] , [InputCNode.dWeights[i]])[0][0]
            dis+=temp*bias[i]
        return dis



    def AdjustWeights(self, target_PNode, LearningRate, Influence):
        for i in range(0, self.num_component):
            self.dWeights[i] += LearningRate * Influence * (target_PNode.getvector(i) - self.dWeights[i])

    def addPNode(self, inputNode, idx):
        self.PNodes[idx]=inputNode
        return
    
    def __str__(self):
      return np.array2string(self.dWeights)
