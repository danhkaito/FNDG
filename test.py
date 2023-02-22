import sys
import networkx as nx
from model.KSOM.CSom import *
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from utils import *
import numpy as np
import random
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv
from model.model import FakeNewsModel
import data_load
RADIUS_MAP = 15
path_model='./model/model_fakenew.pt'

model_fakenew= torch.load(path_model)

features_content, edge_index, edge_type, labels = data_load.load_data_fakenews(True)
data_test = pd.read_csv('benchmark_data\\Liar\\test.csv')

enc_bert = np.load('./model_save/embeddings_idf_test.npy')
content = data_test.Statement.values
labels = data_test.Label.values
model=load_pickle('./model/KSOM/ksom_model_100k_euclid_idf.ckpt')
PNodes_arr_test=create_pnode(corpus=content, pre_data=enc_bert,labels=labels)
# print(data_test.iloc[0])
TP=0
FN=0
TN=0
FP=0


for i in range(0, len(data_test)):
    idx_start = len(features_content)
    # print(idx_start)

    lst_weight_edge_test=[]
    lst_edge_type = []
    features_content_test = features_content.detach()
    # features_style_test = features_style.detach()

    test_node = PNodes_arr_test[i]
    SuitNode, ix, iy = model.FindBestMatchingNode(PNodes_arr_test[i], 'euclid')
    weight_val= model.calc_euclid_distance(PNodes_arr_test[i], SuitNode)
    # SuitNode.addPNode(PNodes_arr[i], i)
    # lst_weight_edge_test.append([RADIUS_MAP*iy+ix, idx_start, weight_val])
    # lst_edge_type.append(1)
    for node_idx in model.m_Som[iy, ix].PNodes:
        # print(f"Node {iy}, {ix}\n")
        weight_pnode2pnode=model.calc_cosine_distance(model.m_Som[iy, ix].PNodes[node_idx], PNodes_arr_test[i])
        # print(f"{weight_pnode2pnode}\n")
        # if weight_pnode2pnode < 0.7:
        #     continue
        lst_weight_edge_test.append([node_idx, idx_start, weight_pnode2pnode])
        lst_edge_type.append(2)

    
    lst_weight_edge_test_row = torch.from_numpy(np.array([x[0] for x in lst_weight_edge_test])).to(torch.long)
    lst_weight_edge_test_col = torch.from_numpy(np.array([x[1] for x in lst_weight_edge_test])).to(torch.long)



    edge_index_test_no_inv = torch.stack([lst_weight_edge_test_row, lst_weight_edge_test_col], dim=0)
    edge_index_test_inv = torch.stack([lst_weight_edge_test_col, lst_weight_edge_test_row], dim=0)


    edge_type_addition = torch.from_numpy(np.array(lst_edge_type))

    edge_index_test = torch.cat((edge_index, edge_index_test_no_inv,edge_index_test_inv), 1)
    edge_type_test = torch.cat((edge_type, edge_type_addition, edge_type_addition))

    assert len(edge_index_test[0])==len(edge_type_test) 

    features_content_test= torch.cat((features_content_test, torch.from_numpy(np.array([PNodes_arr_test[i].get_vector()])).type(torch.float32)))
    # features_style_test= torch.cat((features_style_test, torch.from_numpy(np.array([PNodes_arr_test[i].getvector(1)])).type(torch.float32)))


    # labels_test = np. concatenate((labels, np.array([1 if data_test[i][-1]==True else 0]).astype(np.int64)))
    
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_fakenew = model_fakenew.to(device)
    features_content_test = features_content_test.to(device)
    # features_style_test = features_style_test.to(device)
    edge_index_test = edge_index_test.to(device)
    edge_type_test = edge_type_test.to(device)




    model_fakenew.eval()
    pred = model_fakenew(features_content_test, edge_index_test, edge_type_test)  
    pred_labels=torch.argmax(pred, dim=-1).detach().cpu()[idx_start].item()
    # print(data_test[i][-1])
    # print(pred_labels[idx_start].item())
    if labels[i]==False and pred_labels==1:
      TP+=1
    elif labels[i]==False and pred_labels==0:
      FN+=1
    elif labels[i]==True and pred_labels==0:
      TN+=1
    else:
      FP+=1

print(TP, FP, FN, TN)
print(f"Accuracy {(TP+TN)/(FP+FN+TP+TN)}")
print(f"Precision {(TP)/(TP+FP)}")
print(f"Recall {(TP)/(TP+FN)}")
