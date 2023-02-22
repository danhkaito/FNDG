import pickle
import os
from utils import *
import numpy as np

METHOD = 'euclid'
model=load_pickle('./model/KSOM/ksom_model_100k_euclid_idf.ckpt')
model.map_PNode2CNode(METHOD)

# for iy, ix in np.ndindex(model.m_Som.shape):
#     for i in range(len(model.m_Som[iy,ix].PNodes)):
#         print(iy," ",ix," ",model.m_Som[iy,ix].PNodes[i].corpus)

# print(f"Quantization Error {sum_quan_err/len(PNodes_arr)}")

with open("node_ksom.txt", "w+", encoding="utf-8") as f:
    for iy, ix in np.ndindex(model.m_Som.shape):
        cNode = model.m_Som[iy, ix]
        if len(cNode.PNodes) > 0:
            f.write(f"Node: ({iy}, {ix})--------------------\n")
            # cnt=0
            # for i in cNode.PNodes:
            #     f.write(str(i))
            #     cnt+=1
            # print(f"Node KSOM {iy, ix} {cnt}")
            f.write(str(cNode))