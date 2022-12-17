from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
import torch
import numpy as np
class FakeNewsDataset(InMemoryDataset):
    def __init__(self, embeddings_content,embeddings_style, num_nodes, labels, edge_index, transform=None):
        super(FakeNewsDataset, self).__init__('.', transform, None, None)

        data = Data(edge_index=edge_index)
        
        data.num_nodes = num_nodes
        
        # embedding 
        data.content_feature = torch.from_numpy(embeddings_content).type(torch.float32)
        data.style_feature = torch.from_numpy(embeddings_style).type(torch.float32)
        data.num_content_feature = len(data.content_feature[0])
        data.num_style_feature = len(data.style_feature[0])

        # labels
        fake_idx = np.squeeze(np.argwhere(labels == 1))
        X_fake_train, X_fake_test = train_test_split(fake_idx, test_size=0.2, random_state=42)
        true_idx = np.squeeze(np.argwhere(labels == 0))
        X_true_train, X_true_test = train_test_split(true_idx, test_size=0.2, random_state=42)
        X_train=np.concatenate((true_idx,fake_idx), axis=None)
        X_test=np.concatenate((X_true_test,X_fake_test), axis=None)
        
        # print(len(X_train))
        y = torch.from_numpy(labels).type(torch.long)
        data.y = y.clone().detach()
        
        data.num_classes = 2

        # splitting the data into train, validation and test
        n_nodes = num_nodes
        # create train and test masks for data
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[X_train] = True
        for x in range(0, 100):
            test_mask[x]=True
        data['train_mask'] = train_mask
        data['test_mask'] = test_mask
        self.data, self.slices = self.collate([data])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)