import pickle
from model.KSOM.CNode import *
from  model.KSOM.PNode import *

import math
import numpy as np
import copy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt

class CSom:
    def __init__(self, MapSize, feature_post, bias, numComponent, numIterations, doc_2_vectorizer, constStartLearningRate=0.5):
        self.numComponent = numComponent
        self.MapSize = MapSize
        self.corpus = [post[1] for post in feature_post]
        # print(self.corpus[1])
        self.numIterations = numIterations
        self.dMapRadius = MapSize / 2
        self.dTimeConstant = numIterations / math.log(self.dMapRadius)
        self.dLearningRate = constStartLearningRate
        self.constLearningRate = constStartLearningRate
        self.bias = bias
        self.doc_2_vectorizer = doc_2_vectorizer
        print("Start TFIDF")
        self.PNodes_content_endcode = self.doc_2_vectorizer.fit_transform(self.corpus).todense()
        print(self.PNodes_content_endcode.shape)
        # for x, y in np.ndindex(self.PNodes_content_endcode.shape):
        #     if self.PNodes_content_endcode[x,y]>1 or self.PNodes_content_endcode[x,y]<0:
        #         print("Nastsss")
        self.PNodes_writingstyle_encode = np.array([post[2] for post in feature_post])
        # print(self.PNodes_writingstyle_encode[1])
        # print(np.squeeze(np.asarray(self.PNodes_content_endcode[0])))
        self.PNodes=np.asarray([PNode(corpus=self.corpus[i], num_component=self.numComponent, vectors=[np.squeeze(np.asarray(self.PNodes_content_endcode[i])), self.PNodes_writingstyle_encode[i]]) for i in range(len(self.corpus))])
        print("Done TFIDF")
        # print(self.PNodes[1])
        Node_content_Dimension = self.PNodes_content_endcode.shape[1]
        Node_writingstyle_Dimension = self.PNodes_writingstyle_encode.shape[1]
        self.m_Som = np.asarray([[CNode(self.numComponent, Node_content_Dimension, Node_writingstyle_Dimension) for j in range(MapSize)] for i in range(MapSize)])

    def FindBestMatchingNode(self, inputPNode):
        LowestDistance = 999999
        # SecDistance = 999999
        winner = None
        PNode = inputPNode
        for iy, ix in np.ndindex(self.m_Som.shape):
            dist = self.m_Som[iy, ix].CalculateDistance_PNode2CNode(PNode, self.bias)
            if dist < LowestDistance:
                # if len(self.m_Som[iy, ix].PNodes) > 0:
                #     totalSim = 0
                #     for i in range(len(self.m_Som[iy, ix].PNodes)):
                #         totalSim += int(cosine_similarity(inputPNode, self.m_Som[iy, ix].PNodes[i].vector))
                #     if totalSim / len(self.m_Som[iy, ix].PNodes) < 0.5:
                #         continue
                LowestDistance = dist
                winner = self.m_Som[iy, ix]
                win_x=ix
                win_y=iy
        return winner, win_x, win_y

    def CalculateDistance_PNode2CNode(self, inputPNode: PNode, inputCNode: CNode):
        return inputCNode.CalculateDistance_PNode2CNode(inputPNode, self.bias)

    def calcdistance2PNode(self, inputPNode1: PNode, inputPNode2: PNode):
        return PNode.calcdistance2PNode(inputPNode1, inputPNode2, self.bias)

    def CalculateDistance2CNode(self, inputNode1: CNode, inputNode2: CNode):
        return inputNode1.CalculateDistance2CNode(inputNode2, self.bias)

    def CalculateCosinePNode2CNode(self, inputPNode, inputCNode):
        return inputCNode.CalculateCosinePNode2CNode(inputPNode, self.bias)

    def CalculateCosine2CNode(self, inputCNode1, inputCNode2):
        return inputCNode1.CalculateCosine2CNode(inputCNode2, self.bias)

    def CalculateCosine2PNode(self, inputPNode1, inputPNode2):
        return PNode.calc_cosine2PNode(inputPNode1, inputPNode2, self.bias)
    def Train(self):
        print("Start Training")
        for i in range(self.numIterations):
            print(f"Epoch {i}")
            randomPNode = int(np.random.randint(self.PNodes.shape[0], size=1))
            WinningNode, grid_x, grid_y = self.FindBestMatchingNode(self.PNodes[randomPNode])
            dNeighbourhoodRadius = self.dMapRadius * math.exp(-float(i) / self.dTimeConstant)
            WidthSq = dNeighbourhoodRadius * dNeighbourhoodRadius
            for iy, ix in np.ndindex(self.m_Som.shape):
                DistToNodeSq = (iy-grid_y)*(iy-grid_y)+(ix-grid_x)*(ix-grid_x)

                if True:
                    self.dInfluence = math.exp(-(DistToNodeSq) / (2 * WidthSq))
                    self.m_Som[iy, ix].AdjustWeights(self.PNodes[randomPNode],
                                                     self.dLearningRate, self.dInfluence)
            self.dLearningRate = self.constLearningRate * math.exp(-float(i) / (self.dTimeConstant))
            # if i%5==0:
            #   with open('ksom.txt', 'w') as f:
            #     for iy, ix in np.ndindex(self.m_Som.shape):
            #       f.write(f"{iy} {ix} {self.m_Som[iy,ix]} \n")


        # for i in range(self.PNodes.shape[0]):
        #     SuitNode = self.FindBestMatchingNode(self.PNodes[i])
        #     # print(self.PNodes[i])
        #     # print(SuitNode)
        #     SuitNode.addPNode(self.corpus[i], self.PNodes[i])

    def Plot(self):
        plt.rcParams["figure.autolayout"] = True
        count = 0
        data2D = np.arange(self.MapSize ** 2).reshape(self.MapSize, self.MapSize)
        for iy, ix in np.ndindex(self.m_Som.shape):
            if len(self.m_Som[iy, ix].PNodes) == 0:
                data2D[iy, ix] = 0
        cmap = plt.cm.get_cmap('Blues').copy()
        cmap.set_under('white')
        plt.imshow(data2D, cmap=cmap, vmin=0)
        plt.colorbar(extend='min', extendrect=True)
        plt.show()
    
    def save(self, oup):
        pickle.dump(self, oup)

