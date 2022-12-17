import sys
import networkx as nx
from model.KSOM.CSom import *
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from utils import *

import random
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
import data_load
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv
from model.model import *


features_content, features_style, edge_index, edge_type, labels = data_load.load_data_fakenews(True)
print(len(edge_type))
print(len(edge_type))

# y = torch.from_numpy(labels).type(torch.long)

print(labels[0])


fake_idx = np.squeeze(np.argwhere(labels == 1))
X_fake_train, X_fake_test = train_test_split(fake_idx, test_size=0.2, random_state=42)
true_idx = np.squeeze(np.argwhere(labels == 0))
X_true_train, X_true_test = train_test_split(true_idx, test_size=1- len(X_fake_train)/len(true_idx), random_state=42)
X_train=np.concatenate((true_idx,fake_idx), axis=None)
X_test=np.concatenate((X_true_test,X_fake_test), axis=None)


print(len(X_fake_train), len(X_true_train))
# a = np.squeeze(np.argwhere(X_train < 15*15))
# print(len(a))
# exit()
############# Train mask ############
# Mask all the ksom node
train_mask = torch.zeros(len(features_content), dtype=torch.bool)
test_mask = torch.zeros(len(features_content), dtype=torch.bool)
train_mask[X_train] = True
test_mask[X_test] = True
# for x in range(0, 15*15):
#     test_mask[x]=True


# for i in range (0, 15*15):
#     train_mask[i] = False


model_fakenew = FakeNewsModel(hidden_channels_1=64, hidden_channels_2=16, num_feature_concat=100, num_content_feature=3000, num_style_feature=2, num_classes=2)
print(model_fakenew)
# Use GPU
print("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
features_content = features_content.to(device)
features_style = features_style.to(device)
edge_index = edge_index.to(device)
edge_type = edge_type.to(device)
labels = labels.to(device)
# train_mask = train_mask.to(device)
model_fakenew = model_fakenew.to(device)

# Initialize Optimizer
learning_rate = 0.001
decay = 5e-4
optimizer = torch.optim.Adam(model_fakenew.parameters(), 
                             lr=learning_rate, 
                             weight_decay=decay)


# Define loss function (CrossEntropyLoss for Classification Problems with 
# probability distributions)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model_fakenew.train()
      optimizer.zero_grad() 
      # Use all data as input, because all nodes have node features
      out = model_fakenew(features_content, features_style, edge_index, edge_type)  
      # Only use nodes with labels available for loss calculation --> mask
      loss = criterion(out[train_mask], labels[train_mask])  
      loss.backward() 
      optimizer.step()
      return loss


@torch.no_grad()
def test():
    model_fakenew.eval()
    out = model_fakenew(features_content, features_style, edge_index, edge_type)
    # Use the class with highest probability.
    pred = out.argmax(dim=1)
    # Check against ground-truth labels.

    # test_correct = pred[train_mask] == labels[train_mask]
    # # Derive ratio of correct predictions.
    # test_acc = int(test_correct.sum()) / int(train_mask.sum())  
#   train_correct = pred[train_mask] == data.y[data.train_mask]  
#   # Derive ratio of correct predictions.
#   train_acc = int(train_correct.sum()) / int(data.train_mask.sum())
    print_class_acc(out[test_mask], labels[test_mask])
    # return test_acc
      

losses = []
for epoch in range(0, 2000):
    loss = train()
    losses.append(loss)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        test()

torch.save(model_fakenew, './model/model_fakenew.pt')
