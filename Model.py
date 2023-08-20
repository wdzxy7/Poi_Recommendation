import torch
import numpy as np
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader


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
    def __init__(self, poi_len=38333, graph_features=898, embed_dim=128, out_dim=128):
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
    def __init__(self, cat_len=400, poi_len=38333, user_len=1083, embed_dim=128):
        super(UserHistoryNet, self).__init__()
        self.emb = nn.Embedding(3, embed_dim)

    def forward(self, inputs):
        pass


class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        pass

    def forward(self, inputs):
        pass


class GlobalUserNet(nn.Module):
    def __init__(self):
        super(GlobalUserNet, self).__init__()

    def forward(self, inputs):
        pass
