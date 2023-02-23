import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, RGCNConv, GatedGraphConv, GATv2Conv, GINConv

class FakeNewsModel(torch.nn.Module):
    def __init__(self, hidden_channels_1, hidden_channels_2, num_feature_concat, num_content_feature, num_style_feature, num_classes):
        super(FakeNewsModel, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        # self.conv1 = GATv2Conv(num_content_feature, hidden_channels_1, heads=8)
        # self.conv2 = GATv2Conv(hidden_channels_1*8, hidden_channels_2, heads=1)
        # self.conv3 = SAGEConv(hidden_channels_2, hidden_channels_2)
        self.conv1 = SAGEConv(num_content_feature, hidden_channels_1)
        self.conv2 = SAGEConv(hidden_channels_1, hidden_channels_2)
        # self.conv3 = SAGEConv(hidden_channels_2, hidden_channels_2)
        # self.conv4 = SAGEConv(hidden_channels_2, hidden_channels_2)

        self.out = nn.Linear(hidden_channels_2, num_classes)

    def forward(self, x_content, edge_index):
        
        # x_content_enc = self.post_enc(x_content)
        # x_style_content = self.style_enc(x_style)
        # x = torch.cat((x_content_enc, x_style_content),1)
        # # First Message Passing Layer (Transformation)
        x = self.conv1(x_content, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # x = self.conv3(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # x = self.conv4(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # x = self.conv 3(x, edge_index)
        # x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)

        # x = self.conv3(x, edge_index)
        # x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)

        # Output layer 
        x = self.out(x)
        return x


class FakeNewsModelGated(torch.nn.Module):
    def __init__(self, num_layer, hidden_channels_2, num_feature_concat, num_content_feature, num_style_feature, num_classes):
        super(FakeNewsModelGated, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.post_enc = nn.Linear(num_content_feature, num_feature_concat)
        self.style_enc = nn.Linear(num_style_feature, num_feature_concat)
        # self.conv1 = RGCNConv(2*num_feature_concat, hidden_channels_1, 3, num_bases=30)
        # self.conv2 = RGCNConv(hidden_channels_1, hidden_channels_2, 3, num_bases= 30)

        self.gatedgnn = GatedGraphConv(hidden_channels_2, num_layers=num_layer)

        self.out = nn.Linear(hidden_channels_2, num_classes)

    def forward(self, x_content, x_style, edge_index, edge_type):
        
        x_content_enc = self.post_enc(x_content)
        x_style_content = self.style_enc(x_style)
        x = torch.cat((x_content_enc, x_style_content),1)
        # First Message Passing Layer (Transformation)
        # x = self.conv1(x, edge_index, edge_type)
        # x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)

        # # Second Message Passing Layer
        # x = self.conv2(x, edge_index, edge_type)
        # x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)

        x = self.gatedgnn(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.4, training=self.training)

        # Output layer 

        x = self.out(x)
        return x

class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, dim_h):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            nn.Sequential(nn.Linear(768, dim_h),
                       nn.BatchNorm1d(dim_h), nn.ReLU(),
                       nn.Linear(dim_h, dim_h), nn.ReLU()))
        self.conv2 = GINConv(
            nn.Sequential(nn.Linear(dim_h, dim_h), nn.BatchNorm1d(dim_h), nn.ReLU(),
                       nn.Linear(dim_h, dim_h), nn.ReLU()))
        self.lin2 = nn.Linear(dim_h, 2)
    def forward(self, x, edge_index):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        # h3 = self.conv3(h2, edge_index)

        # # Graph-level readout
        # h1 = global_add_pool(h1, batch)
        # h2 = global_add_pool(h2, batch)
        # h3 = global_add_pool(h3, batch)

        # # Concatenate graph embeddings
        # h = torch.cat((h1, h2, h3), dim=1)

        # # Classifier
        h = self.lin2(h2)
        h = F.dropout(h, p=0.5, training=self.training)
        
        return h
