import math
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class GlobalGraphNet(nn.Module):
    def __init__(self, embed_dim=128, cat_len=400, poi_len=38333, user_len=1083, out_dim=128):
        super(GlobalGraphNet, self).__init__()
        self.emb = nn.Embedding(cat_len + poi_len + user_len, (embed_dim - 3) // 2)
        self.cov1 = GCNConv((embed_dim - 3) // 2 * 2 + 3, 64)
        self.cov2 = GCNConv(64, 32)
        self.cov3 = GCNConv(32, 32)
        self.cov4 = GCNConv(32, 32)
        self.cov5 = GCNConv(32, 1)
        self.relu = nn.LeakyReLU()
        self.fc_layer = nn.Sequential(
            nn.Linear(poi_len, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.ReLU()
        )

    def forward(self, inputs):
        feature = inputs.x
        edges = inputs.edge_index
        id_feature = feature[:, 0: 2].int()
        embed_feature = self.emb(id_feature)
        embed_feature = embed_feature.reshape(embed_feature.shape[0], -1)
        feature = torch.cat((embed_feature, feature[:, 2: 5]), dim=1)
        feature = self.relu(self.cov1(feature, edges))
        feature = self.relu(self.cov2(feature, edges))
        t_feature = self.cov3(feature, edges)
        feature = self.relu(t_feature) + t_feature
        t_feature = self.cov4(feature, edges)
        feature = self.relu(t_feature) + t_feature
        feature = self.relu(self.cov5(feature, edges))
        feature = feature.reshape(-1)
        output = self.fc_layer(feature)
        return output


class GlobalDistNet(nn.Module):
    def __init__(self,  embed_dim=128, poi_len=38333, graph_features=898, out_dim=128):
        super(GlobalDistNet, self).__init__()
        self.poi_len = poi_len
        self.emb = nn.Embedding(poi_len, embed_dim)
        self.cov1 = GCNConv(graph_features, 64)
        self.cov2 = GCNConv(64, 32)
        self.cov3 = GCNConv(32, 32)
        self.cov4 = GCNConv(32, 32)
        self.cov5 = GCNConv(32, 1)
        self.relu = nn.LeakyReLU()
        self.fc_layer = nn.Sequential(
            nn.Linear(poi_len, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.ReLU()
        )

    def forward(self, inputs, mask):
        feature = inputs.x
        edges = inputs.edge_index
        poi = torch.masked_select(feature, mask)
        poi = poi.reshape(self.poi_len, -1).int()
        emb_poi = self.emb(poi)
        emb_feature = feature.masked_scatter(mask, emb_poi)
        feature = self.relu(self.cov1(emb_feature, edges))
        feature = self.relu(self.cov2(feature, edges))
        t_feature = self.cov3(feature, edges)
        feature = self.relu(t_feature) + t_feature
        t_feature = self.cov4(feature, edges)
        feature = self.relu(t_feature) + t_feature
        feature = self.relu(self.cov5(feature, edges))
        feature = feature.reshape(-1)
        output = self.fc_layer(feature)
        return output


class UserGraphNet(nn.Module):
    def __init__(self, cat_len=400, poi_len=38333, embed_dim=128, out_dim=128, node_len=714):
        super(UserGraphNet, self).__init__()
        self.emb = nn.Embedding(cat_len + poi_len, (embed_dim - 3) // 2)
        self.cov1 = GCNConv((embed_dim - 3) // 2 * 2 + 3, 32)
        self.cov2 = GCNConv(32, 32)
        self.cov3 = GCNConv(32, 32)
        self.cov4 = GCNConv(32, 1)
        self.relu = nn.LeakyReLU()
        self.fc_layer = nn.Sequential(
            nn.Linear(node_len, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.ReLU()
        )

    def forward(self, feature, edges):
        raw_batch = feature.shape[0]
        id_feature = feature[:, :, 0: 2].int()
        embed_feature = self.emb(id_feature)
        embed_feature = embed_feature.reshape(embed_feature.shape[0], embed_feature.shape[1], -1)
        feature = torch.cat((embed_feature, feature[:, :, 2: 5]), dim=2)
        user_graph_data = self.build_graph_data(feature, edges)
        x, edges, batch = user_graph_data.x, user_graph_data.edge_index, user_graph_data.batch
        feature = self.relu(self.cov1(x, edges))
        t_feature = self.cov2(feature, edges)
        feature = self.relu(t_feature) + t_feature
        t_feature = self.cov3(feature, edges)
        feature = self.relu(t_feature) + t_feature
        feature = self.relu(self.cov4(feature, edges))
        feature = feature.reshape(raw_batch, -1)
        output = self.fc_layer(feature)
        return output

    def build_graph_data(self, feature, edges):
        data_list = [Data(t, j) for t, j in zip(feature, edges)]
        graph_loader = DataLoader(data_list, feature.shape[0])
        for _, user_graph_data in enumerate(graph_loader):
            return user_graph_data


class UserHistoryNet(nn.Module):
    def __init__(self, cat_len=400, poi_len=38333, user_len=1083, embed_dim=128, hidden_size=128, lstm_layers=3):
        super(UserHistoryNet, self).__init__()
        self.emb = nn.Embedding(cat_len + poi_len + user_len, (embed_dim - 7) // 3)
        self.lstm = nn.LSTM(embed_dim, hidden_size, lstm_layers, dropout=0.5, batch_first=True)

    def forward(self, inputs):
        id_feature = inputs[:, :, 0: 3].int()
        embed_feature = self.emb(id_feature)
        embed_feature = embed_feature.reshape(inputs.shape[0], inputs.shape[1], -1)
        inputs = torch.cat((embed_feature, inputs[:, :, 2: 10]), dim=2)
        output = self.lstm(inputs)
        return output[0]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, embed_dim, dropout, tran_head, tran_hid, tran_layers, poi_len):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embed_dim, tran_head, tran_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, tran_layers)
        self.embed_size = embed_dim
        self.decoder_poi = nn.Linear(embed_dim, poi_len)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, user_history_feature, global_graph_feature, global_dist_feature, user_graph_feature):
        inputs = torch.cat([global_graph_feature, global_dist_feature, user_graph_feature, user_history_feature], dim=2)
        x = self.transformer_encoder(inputs)
        out_poi = self.decoder_poi(x)
        return out_poi


class GlobalUserNet(nn.Module):
    def __init__(self):
        super(GlobalUserNet, self).__init__()
        global_graph_net = GlobalGraphNet()
        global_dist_net = GlobalDistNet()
        user_graph_net = UserGraphNet()
        user_history_net = UserHistoryNet()
        transformer = TransformerModel()