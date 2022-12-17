import numpy as np
import math

from sklearn.metrics.pairwise import cosine_similarity

class PNode:
    def __init__(self,corpus, num_component, vectors):
        self.corpus = corpus
        self.num_component = num_component
        self.vectors=[0]*num_component
        # self.bias=[0]*num_component
        i=0
        for arg in vectors:
            self.vectors[i]=arg
            i+=1
            # self.bias[i]=arg[1]
        self.vectors=np.array(self.vectors, dtype=object)
    
    def getvector(self, i):
        return self.vectors[i]
    
    def calcdistance2PNode(inputNode1, inputNode2, bias):
        dis=0
        for i in range (0, len(bias)):
            temp = inputNode1.getvector(i) - inputNode2.getvector(i)
            # print(InputNode.getvector(i))
            sum_sq = np.dot(temp.T, temp)
            # print(sum_sq)
            # print(bias[i])
            # print(math.sqrt(sum_sq))
            dis+=math.sqrt(sum_sq/len(temp))*bias[i]
            # print(math.sqrt(sum_sq)*np.int(bias[i]))
        # print(dis)
        return dis

    def calc_cosine2PNode(inputNode1, inputNode2, bias):
        dis=0
        for i in range (0, len(bias)):
            temp = cosine_similarity ([inputNode1.getvector(i)] , [inputNode2.getvector(i)])[0][0]
            # print(InputNode.getvector(i))

            # print(sum_sq)
            # print(bias[i])
            # print(math.sqrt(sum_sq))
            dis+=temp*bias[i]
            # print(math.sqrt(sum_sq)*np.int(bias[i]))
        # print(dis)
        return dis

