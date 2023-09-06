import math
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Parameter


class GcnUnit(nn.Module):
    def __init__(self, gcn_channel=128):
        super(GcnUnit, self).__init__()
        self.cov1 = GCNConv(gcn_channel, gcn_channel)
        self.relu = nn.LeakyReLU()

    def forward(self, x, edges):
        temp = self.cov1(x, edges)
        x = self.relu(temp)
        x = x + temp
        return x


class GlobalGraphNet(nn.Module):
    def __init__(self, cat_len=400, poi_len=38333, cat_dim=100, poi_dim=300, gcn_channel=64, gcn_layers=5):
        super(GlobalGraphNet, self).__init__()
        self.gcn_channel = gcn_channel
        self.gcn_layers = gcn_layers
        self.cat_emb = nn.Embedding(cat_len, cat_dim)
        self.poi_emb = nn.Embedding(poi_len, poi_dim)
        self.cov_in = GCNConv(cat_dim + poi_dim + 3, gcn_channel)
        self.gcn_net = self.build_gcn_net()
        self.cov_out = GCNConv(gcn_channel, 1)
        self.relu = nn.LeakyReLU()
        self.fc_layer = nn.Sequential(
            nn.Linear(poi_len - 1, 128),
            nn.ReLU(),
            nn.Linear(128, poi_len),
            nn.ReLU()
        )

    def forward(self, inputs):
        feature = inputs.x
        edges = inputs.edge_index
        poi_feature = feature[:, 0: 1].int()
        cat_feature = feature[:, 1: 2].int()
        poi_feature = self.poi_emb(poi_feature)
        cat_feature = self.cat_emb(cat_feature)
        poi_feature = poi_feature.reshape(poi_feature.shape[0], -1)
        cat_feature = cat_feature.reshape(cat_feature.shape[0], -1)
        feature = torch.cat((poi_feature, cat_feature, feature[:, 2: 5]), dim=1)
        feature = self.relu(self.cov_in(feature, edges))
        for i in range(self.gcn_layers):
            feature = self.gcn_net[i](feature, edges)
        feature = self.relu(self.cov_out(feature, edges))
        feature = feature.reshape(-1)
        output = self.fc_layer(feature)
        return output

    def build_gcn_net(self):
        gcn_net = nn.ModuleList()
        for i in range(self.gcn_layers):
            gcn_net.append(GcnUnit(self.gcn_channel))
        return gcn_net


class GlobalDistNet(nn.Module):
    def __init__(self,  cat_dim=128, poi_len=38333, graph_features=544, gcn_channel=128, gcn_layers=4):
        super(GlobalDistNet, self).__init__()
        self.poi_len = poi_len
        self.gcn_layers = gcn_layers
        self.gcn_channel = gcn_channel
        self.emb = nn.Embedding(poi_len, cat_dim)
        self.gcn_net = self.build_gcn_net()
        self.cov_in = GCNConv(graph_features, gcn_channel)
        self.cov_out = GCNConv(gcn_channel, 1)
        self.relu = nn.LeakyReLU()
        self.fc_layer = nn.Sequential(
            nn.Linear(poi_len - 1, 128),
            nn.ReLU(),
            nn.Linear(128, poi_len),
            nn.ReLU()
        )

    def forward(self, inputs, mask):
        feature = inputs.x
        edges = inputs.edge_index
        poi = torch.masked_select(feature, mask)
        poi = poi.reshape(self.poi_len - 1, -1).int()
        emb_poi = self.emb(poi)
        emb_feature = feature.masked_scatter(mask, emb_poi)
        feature = self.relu(self.cov_in(emb_feature, edges))
        for i in range(self.gcn_layers):
            feature = self.gcn_net[i](feature, edges)
        feature = self.relu(self.cov_out(feature, edges))
        feature = feature.reshape(-1)
        output = self.fc_layer(feature)
        return output

    def build_gcn_net(self):
        gcn_net = nn.ModuleList()
        for i in range(self.gcn_layers):
            gcn_net.append(GcnUnit(self.gcn_channel))
        return gcn_net


class UserGraphNet(nn.Module):
    def __init__(self, cat_len=400, poi_len=5099, cat_dim=100, poi_dim=300, gcn_channel=128, gcn_layers=3,
                 node_len=714):
        super(UserGraphNet, self).__init__()
        self.gcn_layers = gcn_layers
        self.gcn_channel = gcn_channel
        self.cat_emb = nn.Embedding(cat_len, cat_dim)
        self.poi_emb = nn.Embedding(poi_len, poi_dim)
        self.gcn_net = self.build_gcn_net()
        self.cov_in = GCNConv(cat_dim + poi_dim + 3, gcn_channel)
        self.cov_out = GCNConv(gcn_channel, 1)
        self.relu = nn.LeakyReLU()
        self.fc_layer = nn.Sequential(
            nn.Linear(node_len, 128),
            nn.ReLU(),
            nn.Linear(128, poi_len),
            nn.ReLU()
        )

    def forward(self, feature, edges):
        raw_batch = feature.shape[0]
        poi_feature = feature[:, :, 0: 1].int()
        cat_feature = feature[:, :, 1: 2].int()
        poi_feature = self.poi_emb(poi_feature)
        cat_feature = self.cat_emb(cat_feature)
        poi_feature = poi_feature.reshape(poi_feature.shape[0], poi_feature.shape[1], -1)
        cat_feature = cat_feature.reshape(cat_feature.shape[0], cat_feature.shape[1], -1)
        feature = torch.cat((poi_feature, cat_feature, feature[:, :, 2: 5]), dim=2)
        user_graph_data = self.build_graph_data(feature, edges)
        x, edges, batch = user_graph_data.x, user_graph_data.edge_index, user_graph_data.batch
        feature = self.relu(self.cov_in(x, edges))
        for i in range(self.gcn_layers):
            feature = self.gcn_net[i](feature, edges)
        feature = self.relu(self.cov_out(feature, edges))
        feature = feature.reshape(raw_batch, -1)
        output = self.fc_layer(feature)
        return output

    def build_graph_data(self, feature, edges):
        data_list = [Data(t, j) for t, j in zip(feature, edges)]
        graph_loader = DataLoader(data_list, feature.shape[0])
        for _, user_graph_data in enumerate(graph_loader):
            return user_graph_data

    def build_gcn_net(self):
        gcn_net = nn.ModuleList()
        for i in range(self.gcn_layers):
            gcn_net.append(GcnUnit(self.gcn_channel))
        return gcn_net


class UserHistoryNet(nn.Module):
    def __init__(self, cat_len=400, poi_len=5099, user_len=1083, embed_size_user=20, embed_size_poi=300, embed_size_cat=100,
                 embed_size_hour=20, hidden_size=128, lstm_layers=3):
        super(UserHistoryNet, self).__init__()
        self.embed_user = nn.Embedding(user_len, embed_size_user)
        self.embed_poi = nn.Embedding(poi_len, embed_size_poi)
        self.embed_cat = nn.Embedding(cat_len, embed_size_cat)
        self.embed_hour = nn.Embedding(24, embed_size_hour)
        self.embed_week = nn.Embedding(7, 7)
        self.lstm_poi = nn.LSTM(embed_size_user + embed_size_poi + embed_size_hour + 7 + 2, hidden_size, lstm_layers, dropout=0.5, batch_first=True)
        self.lstm_cat = nn.LSTM(embed_size_user + embed_size_cat + embed_size_hour + 7 + 2, hidden_size, lstm_layers, dropout=0.5, batch_first=True)
        self.poi_fc = nn.Linear(hidden_size, poi_len)
        self.cat_fc = nn.Linear(hidden_size, poi_len)
        self.out_w_poi = Parameter(torch.Tensor([0.5]).repeat(poi_len), requires_grad=True)
        self.out_w_cat = Parameter(torch.Tensor([0.5]).repeat(poi_len), requires_grad=True)

    def forward(self, inputs):
        poi_feature = torch.cat((inputs[:, :, 0: 2], inputs[:, :, 3:]), dim=2)
        cat_feature = torch.cat((inputs[:, :, 0:1], inputs[:, :, 2:]), dim=2)
        poi_out = self.get_output(poi_feature, self.embed_poi, self.embed_user, self.embed_hour, self.embed_week, self.lstm_poi, self.poi_fc)
        cat_out = self.get_output(cat_feature, self.embed_cat, self.embed_user, self.embed_hour, self.embed_week, self.lstm_cat, self.cat_fc)
        out_w_poi = self.out_w_poi[inputs[:, :, 0: 1].long()]
        out_w_cat = self.out_w_cat[inputs[:, :, 0: 1].long()]
        poi_out = torch.mul(poi_out, out_w_poi)
        cat_out = torch.mul(cat_out, out_w_cat)
        return poi_out + cat_out

    def get_output(self, inputs, embed_id, embed_user, embed_hour, embed_week, lstm, fc):
        b = inputs.shape[0]
        user_feature = inputs[:, :, 0: 1].int()
        id_feature = inputs[:, :, 1: 2].int()
        hour_feature = inputs[:, :, -2: -1].int()
        week_feature = inputs[:, :, -1:].int()
        emb_user = embed_user(user_feature.reshape(b, -1))
        emb_id = embed_id(id_feature.reshape(b, -1))
        emb_hour = embed_hour(hour_feature.reshape(b, -1))
        emb_week = embed_week(week_feature.reshape(b, -1))
        features = torch.cat((emb_user, emb_id, emb_hour, emb_week, inputs[:, :, 2: 4]), dim=2)
        output, _ = lstm(features)
        return fc(output)


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
        self.linear = nn.Linear(poi_len, poi_len)
        self.softmax = nn.Softmax(dim=1)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embed_dim, tran_head, tran_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, tran_layers)
        self.embed_size = embed_dim
        self.decoder_poi = nn.Linear(embed_dim, poi_len)
        self.init_weights()
        self.w_g_graph = Parameter(torch.Tensor([0.33]).repeat(poi_len), requires_grad=True)
        self.w_g_dist = Parameter(torch.Tensor([0.33]).repeat(poi_len), requires_grad=True)
        self.w_u_graph = Parameter(torch.Tensor([0.33]).repeat(poi_len), requires_grad=True)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, user_history_feature, global_graph_feature, global_dist_feature, user_graph_feature, src_mask):
        w_g_graph = self.w_g_graph[global_graph_feature[:, :, 0: 1].long()]
        w_g_dist = self.w_g_dist[global_graph_feature[:, :, 0: 1].long()]
        w_u_graph = self.w_u_graph[global_graph_feature[:, :, 0: 1].long()]
        inputs = torch.mul(global_graph_feature, w_g_graph) + torch.mul(global_dist_feature, w_g_dist) + torch.mul(user_graph_feature, w_u_graph)
        inputs = inputs * math.sqrt(self.embed_size)
        inputs = self.pos_encoder(inputs)
        x = self.transformer_encoder(inputs, src_mask)
        out_poi = self.decoder_poi(x)
        out_poi = out_poi + user_history_feature
        return out_poi


class GlobalUserNet(nn.Module):
    def __init__(self):
        super(GlobalUserNet, self).__init__()
        global_graph_net = GlobalGraphNet()
        global_dist_net = GlobalDistNet()
        user_graph_net = UserGraphNet()
        user_history_net = UserHistoryNet()
        transformer = TransformerModel()