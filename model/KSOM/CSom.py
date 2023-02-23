import pickle
from model.KSOM.CNode import *
from  model.KSOM.PNode import *

import math
import numpy as np
from tqdm import tqdm
# import copy
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from matplotlib import pyplot as plt

class CSom:
    def __init__(self, MapSize, train_X, train_Y, numIterations, doc_2_vectorizer, constStartLearningRate=0.5):
        self.MapSize = MapSize
        self.corpus = train_X
        self.labels = train_Y
        self.numIterations = numIterations
        self.dMapRadius = MapSize / 2
        self.dTimeConstant = numIterations / math.log(self.dMapRadius)
        self.dLearningRate = constStartLearningRate
        self.constLearningRate = constStartLearningRate
        # self.doc_2_vectorizer = doc_2_vectorizer
        print("Start TFIDF")
        self.PNodes_content_endcode = doc_2_vectorizer
        print(self.PNodes_content_endcode.shape)
        # for x, y in np.ndindex(self.PNodes_content_endcode.shape):
        #     if self.PNodes_content_endcode[x,y]>1 or self.PNodes_content_endcode[x,y]<0:
        #         print("Nastsss")
        # print(self.PNodes_writingstyle_encode[1])
        # print(np.squeeze(np.asarray(self.PNodes_content_endcode[0])))
        self.PNodes=np.asarray([PNode(corpus=self.corpus[i], label=self.labels[i], vector=np.squeeze(np.asarray(self.PNodes_content_endcode[i]))) for i in range(len(self.corpus))])
        print("Done TFIDF")
        # print(self.PNodes[1])
        Node_content_Dimension = self.PNodes_content_endcode.shape[1]
        self.m_Som = np.asarray([[CNode(Node_content_Dimension) for j in range(MapSize)] for i in range(MapSize)])
    
    def calc_euclid_distance(self, node1, node2):
        temp = node1.get_vector() - node2.get_vector()
        sum_sq = np.dot(temp.T, temp)
        return math.sqrt(sum_sq)

    def calc_cosine_distance(self, node1, node2):
        dis = cosine_similarity(np.expand_dims(node1.get_vector(), axis=0), np.expand_dims(node2.get_vector(), axis=0))[0][0]
        return dis

    def FindBestMatchingNode(self, inputPNode, method):
        LowestDistance = 999999
        if method == 'euclid':
            calc_distance = self.calc_euclid_distance
        else:
            calc_distance = self.calc_cosine_distance
        # SecDistance = 999999
        winner = None
        PNode = inputPNode
        for ix, iy in np.ndindex(self.m_Som.shape):
            dist = calc_distance(self.m_Som[ix, iy], PNode)
            if dist < LowestDistance:
                # if len(self.m_Som[iy, ix].PNodes) > 0:
                #     totalSim = 0
                #     for i in range(len(self.m_Som[iy, ix].PNodes)):
                #         totalSim += int(cosine_similarity(inputPNode, self.m_Som[iy, ix].PNodes[i].vector))
                #     if totalSim / len(self.m_Som[iy, ix].PNodes) < 0.5:
                #         continue
                LowestDistance = dist
                winner = self.m_Som[ix, iy]
                win_x=ix
                win_y=iy
        return winner, win_x, win_y
    
    

    # def CalculateDistance_PNode2CNode(self, inputPNode: PNode, inputCNode: CNode):
    #     return inputCNode.CalculateDistance_PNode2CNode(inputPNode, self.bias)

    # def calcdistance2PNode(self, inputPNode1: PNode, inputPNode2: PNode):
    #     return PNode.calcdistance2PNode(inputPNode1, inputPNode2, self.bias)

    # def CalculateDistance2CNode(self, inputNode1: CNode, inputNode2: CNode):
    #     return inputNode1.CalculateDistance2CNode(inputNode2, self.bias)

    # def CalculateCosinePNode2CNode(self, inputPNode, inputCNode):
    #     return inputCNode.CalculateCosinePNode2CNode(inputPNode, self.bias)

    # def CalculateCosine2CNode(self, inputCNode1, inputCNode2):
    #     return inputCNode1.CalculateCosine2CNode(inputCNode2, self.bias)

    # def CalculateCosine2PNode(self, inputPNode1, inputPNode2):
    #     return PNode.calc_cosine2PNode(inputPNode1, inputPNode2, self.bias)
    
    def Train(self, method):
        print("Start Training " + method)
        
        for i in tqdm(range(self.numIterations)):
            # print(f"Epoch {i}")
            randomPNode = self.PNodes[int(np.random.randint(self.PNodes.shape[0], size=1))]
            WinningNode, grid_x, grid_y = self.FindBestMatchingNode(randomPNode, method)
            dNeighbourhoodRadius = self.dMapRadius * math.exp(-float(i) / self.dTimeConstant)
            WidthSq = dNeighbourhoodRadius * dNeighbourhoodRadius
            for ix, iy in np.ndindex(self.m_Som.shape):
                DistToNodeSq = (ix-grid_x)*(ix-grid_x)+(iy-grid_y)*(iy-grid_y)

                if True:
                    self.dInfluence = math.exp(-(DistToNodeSq) / (2 * WidthSq))
                    self.m_Som[ix, iy].AdjustWeights(randomPNode,
                                                     self.dLearningRate, self.dInfluence)
            self.dLearningRate = self.constLearningRate * math.exp(-float(i) / (self.dTimeConstant))
            # if i%5==0:
            #   with open('ksom.txt', 'w') as f:
            #     for iy, ix in np.ndindex(self.m_Som.shape):
            #       f.write(f"{iy} {ix} {self.m_Som[iy,ix]} \n")

    def map_PNode2CNode_training(self, method):
        self.reset_mapping()
        print("Start Mapping Training")
        for i in tqdm(range(self.PNodes.shape[0])):
            SuitNode, _, _ = self.FindBestMatchingNode(self.PNodes[i], method)
            # print(self.PNodes[i])
            # print(SuitNode)
            SuitNode.addPNode(self.PNodes[i], self.MapSize*self.MapSize+i)
        print('Done Mapping Training')

    def map_PNode2CNode_testing(self, method, lst_PNode):
        print("Start Mapping Whole Graph")
        self.map_PNode2CNode_training(method)
        for i in tqdm(range(lst_PNode.shape[0])):
            SuitNode, _, _ = self.FindBestMatchingNode(lst_PNode[i], method)
            # print(self.PNodes[i])
            # print(SuitNode)
            SuitNode.addPNode(lst_PNode[i], self.MapSize*self.MapSize+self.PNodes.shape[0]+i)
        print("Done mapping Whole Graph")

    
    def reset_mapping(self):
        print("Start Reset mapping")
        for ix, iy in np.ndindex(self.m_Som.shape):
            self.m_Som[ix, iy].PNodes ={}
        print("Done Reset Mapping")

    # def Plot(self):
    #     plt.rcParams["figure.autolayout"] = True
    #     count = 0
    #     data2D = np.arange(self.MapSize ** 2).reshape(self.MapSize, self.MapSize)
    #     for iy, ix in np.ndindex(self.m_Som.shape):
    #         if len(self.m_Som[iy, ix].PNodes) == 0:
    #             data2D[iy, ix] = 0
    #     cmap = plt.cm.get_cmap('Blues').copy()
    #     cmap.set_under('white')
    #     plt.imshow(data2D, cmap=cmap, vmin=0)
    #     plt.colorbar(extend='min', extendrect=True)
    #     plt.show()
    
    def save(self, oup):
        pickle.dump(self, oup)
