import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, RGCNConv, GatedGraphConv, GATv2Conv, GINConv
from transformers import BertModel


class BertClassifier(nn.Module):

    def __init__(self, name_model, num_class=2,  dropout=0.2):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(name_model)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_class)

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)

        return linear_output
    

class fakeBERT(nn.Module):

    def __init__(self, name_model, num_class = 2,  dropout=0.2):

        super(fakeBERT, self).__init__()

        # BERT embed
        self.bert = BertModel.from_pretrained(name_model)
        
        # parallel Conv1D
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(768, 128, 3), # in_chanel = 768 (get_all), out_chanel = 128 (num_filter)
            nn.ReLU(),
            nn.MaxPool1d(5)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(768, 128, 4), # in_chanel = 768 (get_all), out_chanel = 128 (num_filter)
            nn.ReLU(),
            nn.MaxPool1d(5)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv1d(768, 128, 5), # in_chanel = 768 (get_all), out_chanel = 128 (num_filter)
            nn.ReLU(),
            nn.MaxPool1d(5)
        )
        
        # After concatenate
        self.conv_block_4_5 = nn.Sequential(
            nn.Conv1d(128, 128, 5), # in_chanel = 128 (get_all), out_chanel = 128 (num_filter)
            nn.ReLU(),
            nn.MaxPool1d(5),
            
            nn.Conv1d(128, 128, 5), # in_chanel = 128 (get_all), out_chanel = 128 (num_filter)
            nn.ReLU(),
            nn.MaxPool1d(30)
        )
        
        self.dense_block = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.Dropout(dropout),
            nn.Linear(128, num_class)
        )
        
        

    def forward(self, input_id, mask):

        #embed size: [bacth_size, token_length, 768(num_feature)]
        embed, _ = self.bert(input_ids = input_id, attention_mask = mask, return_dict=False)
        
        # Convert embedsize to [bacth_size, 768, token_length] <=> (N, Cin, L) of conv1D
        embed = torch.transpose(embed,1,2).contiguous()
        
        # Parallel conv1D
        out_1 = self.conv_block_1(embed)
        out_2 = self.conv_block_2(embed)
        out_3 = self.conv_block_3(embed)
        
        out_cat = torch.cat((out_1, out_2, out_3), 2)
        
        # After concatenate
        out_4_5 = self.conv_block_4_5(out_cat)
        
        # # Convert to [batch_size, token_length, num_feature] to flatten
        out_4_5 = torch.transpose(out_4_5,1,2).contiguous()
        
        out = self.dense_block(out_4_5)

        return out
    

class BertLSTM(nn.Module):
    def __init__(self , name_model,  n_class = 2, embedding_dim= 768, hidden_dim= 128, n_layers = 1, drop_prob=0.2):
        super(BertLSTM, self).__init__()
        self.n_class = n_class
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.bert = BertModel.from_pretrained(name_model)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.linear = nn.Linear(hidden_dim, n_class)
        
    def forward(self, input_id, mask):
        embeds, _ = self.bert(input_ids = input_id, attention_mask = mask, return_dict=False)

        lstm_out, _ = self.lstm(embeds)

        out = self.linear(lstm_out[:,-1])
        return out


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
