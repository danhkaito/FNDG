import argparse
import scipy.sparse as sp
import numpy as np
import torch
import ipdb
import utils
import pickle
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx



IMBALANCE_THRESH = 101

# def build_graph_ksom(model, ):
#     graph= nx.Graph()

#     edge_dict={}
#     radius=1
#     edge_weighted_list=[]
#     edge_type_list = {}
#     pos={}
#     for ix, iy in np.ndindex(model.m_Som.shape):
#         idx_cnode=RADIUS_MAP*ix+iy
#         pos[idx_cnode]=(ix, iy)
#         for i in range (-radius,radius+1):
#             for j in range (0, radius+1):
#                 if (i==0 and j==0):
#                     continue
#                 x_idx, y_idx= ix+i, iy+j
#                 if (0<=x_idx and x_idx<RADIUS_MAP and 0<=y_idx and y_idx<RADIUS_MAP):
#                     # print(f"CNode in {ix}, {iy} near to {x_idx}, {y_idx}")
#                     idx_cnode_neighbor=RADIUS_MAP*x_idx+y_idx
#                     if (idx_cnode,idx_cnode_neighbor) not in edge_dict and (idx_cnode_neighbor,idx_cnode) not in edge_dict:
#                         # print("Go here")
#                         weights=model.calc_euclid_distance(model.m_Som[ix, iy], model.m_Som[x_idx, y_idx])
#                         # print("W" + str(weights))
#                         # print("DBG "+str((idx_cnode, idx_cnode_neighbor)))
#                         edge_weighted_list.append([idx_cnode, idx_cnode_neighbor, 1.0])
#                         edge_type_list[(idx_cnode, idx_cnode_neighbor)]=0
#                         edge_dict[(idx_cnode, idx_cnode_neighbor)]=1
#     # graph.add_weighted_edges_from(edge_weighted_list)
#     return graph, pos, edge_weighted_list, edge_type_list


# def build_leaf_node_ksom(model, PNodes_arr, lst_weight_edge, lst_edge_type):
#     global G
#     idx_start=RADIUS_MAP*RADIUS_MAP
#     sum_quan_err=0
#     for i in range(PNodes_arr.shape[0]):
      
#         SuitNode, ix, iy = model.FindBestMatchingNode(PNodes_arr[i], 'euclid')
#         weight_val= model.calc_euclid_distance(PNodes_arr[i], SuitNode)
#         # SuitNode.addPNode(PNodes_arr[i], i)
#         sum_quan_err+=weight_val
#         # lst_weight_edge.append([RADIUS_MAP*iy+ix, idx_start, 1.0])
#         # lst_edge_type[(RADIUS_MAP*iy+ix, idx_start)]=1
#         for node_idx in model.m_Som[iy, ix].PNodes:
#             # print(f"Node {iy}, {ix}\n")
#             weight_pnode2pnode=model.calc_cosine_distance(model.m_Som[iy, ix].PNodes[node_idx], PNodes_arr[i])
#             # print(f"{weight_pnode2pnode}\n")
#             # if weight_pnode2pnode < 0.7:
#             #     continue
#             lst_weight_edge.append([node_idx, idx_start, 1.0])
#             lst_edge_type[(node_idx, idx_start)]=2
#         SuitNode.addPNode(PNodes_arr[i], idx_start)
#         idx_start+=1
#     # print(f"Quantization Error {sum_quan_err/len()}")
#     return lst_weight_edge, lst_edge_type


def load_data_fakenews_training(preload, dataset):

    if preload == False:
        model_ksom= utils.load_pickle('./model/KSOM/ksom_model_100k_euclid_liar_bert.ckpt')
        model_ksom.map_PNode2CNode_training('euclid')
        PNodes_arr = np.copy(model_ksom.PNodes)

        RADIUS_MAP = model_ksom.MapSize
        EMBED_DIM = model_ksom.Node_content_Dimension

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

        
            
        for i in range (0, RADIUS_MAP*RADIUS_MAP):
            node_list.append((i, {'x': features_content[i], 'y': 2, 'train_mask': False, 'test_mask': False}))
        for i in range (0, len(PNodes_arr)):
            idx = RADIUS_MAP*RADIUS_MAP+i
            node_list.append((idx, {'x': features_content[idx],'y': int(PNodes_arr[i].label==True), 'train_mask': True, 'test_mask':False}))


        edge_list = []
        for ix, iy in np.ndindex(model_ksom.m_Som.shape):
            cNode = model_ksom.m_Som[ix, iy]
            edge_list = edge_list + cNode.create_edge_subgraph(model_ksom)

        G = nx.Graph()

        G.add_nodes_from(node_list)
        G.add_edges_from(edge_list)

        pyg = from_networkx(G)

        edge_inv = torch.cat((pyg.edge_index[1].view(1,-1), pyg.edge_index[0].view(1,-1)), dim =0)
        pyg.edge_index =  torch.cat((pyg.edge_index, edge_inv), 1)
        pyg.edge_weight = torch.cat((pyg.edge_weight,pyg.edge_weight))
        pyg.edge_type = torch.cat((pyg.edge_type, pyg.edge_type))

        torch.save(pyg, f'./data/{dataset}_training.pt')
        return pyg
    else:
        pyg = torch.load(f'./data/{dataset}_training.pt')
        return pyg

def load_data_fakenews_testing(preload, dataset):
    if preload == False:

        model_ksom= utils.load_pickle('./model/KSOM/ksom_model_100k_euclid_liar_bert.ckpt')

        testing_node_embed = np.load('./model_save/Liar/data numpy/test_embed.npy')
        testing_node_content = np.load('./model_save/Liar/data numpy/test_sentence.npy', allow_pickle=True)
        testing_node_label = np.load('./model_save/Liar/data numpy/test_label.npy', allow_pickle=True)
        
        PNodes_arr_test=utils.create_pnode(corpus=testing_node_content, pre_data=testing_node_embed,labels=testing_node_label)

        model_ksom.map_PNode2CNode_testing('euclid', PNodes_arr_test)

        PNodes_arr = np.copy(model_ksom.PNodes)

        RADIUS_MAP = model_ksom.MapSize
        EMBED_DIM = model_ksom.Node_content_Dimension
        print(EMBED_DIM)


        embeddings_content = np.empty((RADIUS_MAP*RADIUS_MAP+len(PNodes_arr)+len(PNodes_arr_test), EMBED_DIM))
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

        for i in range(PNodes_arr_test.shape[0]):
            embeddings_content[RADIUS_MAP*RADIUS_MAP+len(PNodes_arr)+i]=PNodes_arr_test[i].get_vector()

        embeddings_content=np.array(embeddings_content)
        # embeddings_style=np.array(embeddings_style)
        # embeddings_content = normalize(embeddings_content)
        # embeddings_style = normalize(embeddings_style)
        features_content = torch.from_numpy(embeddings_content).type(torch.float32)

        
        node_list = []    
        for i in range (0, RADIUS_MAP*RADIUS_MAP):
            node_list.append((i, {'x': features_content[i], 'y': 2, 'test_mask':False}))
        for i in range (0, len(PNodes_arr)):
            idx = RADIUS_MAP*RADIUS_MAP+i
            node_list.append((idx, {'x': features_content[idx],'y': int(PNodes_arr[i].label==True), 'test_mask':False}))
        for i in range (0, len(PNodes_arr_test)):
            idx = RADIUS_MAP*RADIUS_MAP+len(PNodes_arr)+i
            node_list.append((idx, {'x': features_content[idx],'y': int(PNodes_arr_test[i].label==True), 'test_mask':True}))


        edge_list = []
        for ix, iy in np.ndindex(model_ksom.m_Som.shape):
            cNode = model_ksom.m_Som[ix, iy]
            edge_list = edge_list + cNode.create_edge_subgraph(model_ksom)
            

        G = nx.Graph()

        G.add_nodes_from(node_list)
        G.add_edges_from(edge_list)

        pyg = from_networkx(G)

        edge_inv = torch.cat((pyg.edge_index[1].view(1,-1), pyg.edge_index[0].view(1,-1)), dim =0)
        pyg.edge_index =  torch.cat((pyg.edge_index, edge_inv), 1)
        pyg.edge_weight = torch.cat((pyg.edge_weight,pyg.edge_weight))
        pyg.edge_type = torch.cat((pyg.edge_type, pyg.edge_type))

        torch.save(pyg, f'./data/{dataset}_testing.pt')
        return pyg
    else:
        pyg = torch.load(f'./data/{dataset}_testing.pt')
        return pyg

def refine_label_order(labels):
    max_label = labels.max()
    j = 0

    for i in range(labels.max(),0,-1):
        if sum(labels==i) >= IMBALANCE_THRESH and i>j:
            while sum(labels==j) >= IMBALANCE_THRESH and i>j:
                j = j+1
            if i > j:
                head_ind = labels == j
                tail_ind = labels == i
                labels[head_ind] = i
                labels[tail_ind] = j
                j = j+1
            else:
                break
        elif i <= j:
            break

    return labels
        




def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def norm_sparse(adj):#normalize a torch dense tensor for GCN, and change it into sparse.
    adj = adj + torch.eye(adj.shape[0]).to(adj)
    rowsum = torch.sum(adj,1)
    r_inv = 1/rowsum
    r_inv[torch.isinf(r_inv)] = 0.
    new_adj = torch.mul(r_inv.reshape(-1,1), adj)

    indices = torch.nonzero(new_adj).t()
    values = new_adj[indices[0], indices[1]] # modify this based on dimensionality

    return torch.sparse.FloatTensor(indices, values, new_adj.size())

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def find_shown_index(adj, center_ind, steps = 2):
    seen_nodes = {}
    shown_index = []

    if isinstance(center_ind, int):
        center_ind = [center_ind]

    for center in center_ind:
        shown_index.append(center)
        if center not in seen_nodes:
            seen_nodes[center] = 1

    start_point = center_ind
    for step in range(steps):
        new_start_point = []
        candid_point = set(adj[start_point,:].reshape(-1, adj.shape[1]).nonzero()[:,1])
        for i, c_p in enumerate(candid_point):
            if c_p.item() in seen_nodes:
                pass
            else:
                seen_nodes[c_p.item()] = 1
                shown_index.append(c_p.item())
                new_start_point.append(c_p)
        start_point = new_start_point

    return shown_index

