import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, RGCNConv, GatedGraphConv

class FakeNewsModel(torch.nn.Module):
    def __init__(self, hidden_channels_1, hidden_channels_2, num_feature_concat, num_content_feature, num_style_feature, num_classes):
        super(FakeNewsModel, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.post_enc = nn.Linear(num_content_feature, num_feature_concat)
        self.style_enc = nn.Linear(num_style_feature, num_feature_concat)
        self.conv1 = SAGEConv(2*num_feature_concat, hidden_channels_1)
        self.conv2 = SAGEConv(hidden_channels_1, hidden_channels_2)
        self.conv3 = SAGEConv(hidden_channels_2, hidden_channels_2)

        self.out = nn.Linear(hidden_channels_2, num_classes)

    def forward(self, x_content, x_style, edge_index, edge_type):
        
        x_content_enc = self.post_enc(x_content)
        x_style_content = self.style_enc(x_style)
        x = torch.cat((x_content_enc, x_style_content),1)
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # x = self.conv3(x, edge_index)
        # x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

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