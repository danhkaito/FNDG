import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
import torch
import utils
model_ksom= utils.load_pickle('./model/KSOM/ksom_model_100k_euclid_idf.ckpt')
# processed_data= utils.load_pickle('../dataset/data_preprocess_imbalance_train')
PNodes_arr = np.copy(model_ksom.PNodes)

RADIUS_MAP = model_ksom.MapSize
EMBED_DIM = 768

node_list = []

embeddings_content = np.empty((RADIUS_MAP*RADIUS_MAP+len(PNodes_arr), EMBED_DIM))
# embeddings_style = np.empty((RADIUS_MAP*RADIUS_MAP+len(processed_data), 2))
# for ix, iy in np.ndindex(model.m_Som.shape):
#     temp=tuple()
#     for w in model.m_Som[ix, iy].dWeights:
#         temp+=(w,)
#         embeddings[16*ix+iy]= np.concatenate(temp, axis=None)

for ix, iy in np.ndindex(model_ksom.m_Som.shape):
    cNode = model_ksom.m_Som[ix, iy]
    t1=np.empty((0,EMBED_DIM))
    # t2=np.empty((0,2))
    if len(cNode.PNodes)>0:
        # print("Go here")
        for i in cNode.PNodes.keys():
            t1=np.append(t1, [cNode.PNodes[i].get_vector()], axis=0)
            # t2= np.append(t2, [cNode.PNodes[i].getvector(1)], axis=0)
        v1=np.mean(t1, axis=0)
        # v2=np.mean(t2, axis=0)
        embeddings_content[RADIUS_MAP*ix+iy]= v1
        # embeddings_style[RADIUS_MAP*ix+iy]= v2
    else:
        embeddings_content[RADIUS_MAP*ix+iy]= np.zeros((1,EMBED_DIM))
        # embeddings_style[RADIUS_MAP*ix+iy]= np.zeros((1,2))

for i in range(PNodes_arr.shape[0]):
    embeddings_content[RADIUS_MAP*RADIUS_MAP+i]=PNodes_arr[i].get_vector()
    # embeddings_style[RADIUS_MAP*RADIUS_MAP+i]=PNodes_arr[i].getvector(1)

embeddings_content=np.array(embeddings_content)
# embeddings_style=np.array(embeddings_style)
# embeddings_content = normalize(embeddings_content)
# embeddings_style = normalize(embeddings_style)
features_content = torch.from_numpy(embeddings_content).type(torch.float32)

def create_node_list(node_list):
    
    for i in range (0, RADIUS_MAP*RADIUS_MAP):
        node_list.append((i, {'x': features_content[i], 'y': 2}))
    for i in range (0, len(PNodes_arr)):
        idx = RADIUS_MAP*RADIUS_MAP+i
        node_list.append((idx, {'x': features_content[idx],'y': int(PNodes_arr[i].label==True)}))

create_node_list(node_list)
print(len(node_list))

def create_edge():
    edge_list = []
    for ix, iy in np.ndindex(model_ksom.m_Som.shape):
        cNode = model_ksom.m_Som[ix, iy]
        edge_list = edge_list + cNode.create_edge_subgraph(model_ksom)
    return edge_list

edge_list = create_edge()

G = nx.Graph()

G.add_nodes_from(node_list)
G.add_edges_from(edge_list)

pyg = from_networkx(G)

edge_inv = torch.cat((pyg.edge_index[1].view(1,-1), pyg.edge_index[0].view(1,-1)), dim =0)
pyg.edge_index =  torch.cat((pyg.edge_index, edge_inv), 1)
pyg.edge_weight = torch.cat((pyg.edge_weight,pyg.edge_weight))
pyg.edge_type = torch.cat((pyg.edge_type, pyg.edge_type))

torch.save(pyg, './data/Fake_or_Real_training.pt')