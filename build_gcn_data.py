import os
import glob
import torch
import pickle
import numpy as np
import networkx as nx
from utils import MinMaxNormalization
from torch_geometric.data import Data


def np_to_dict(node_index):
    node = node_index[:, 0].tolist()
    ind = node_index[:, 1].tolist()
    return dict(zip(ind, node))


def get_edge_index(edges):
    source = edges[:, 0]
    dest = edges[:, 1]
    weight = edges[:, -1]
    edge_index = torch.tensor([source, dest], dtype=torch.long)
    return edge_index, weight


def build_global_graph_data():
    node_index = np.loadtxt('./graph/{}/graph_node_id2idx.txt'.format(data_name), dtype=int)
    node_dict = np_to_dict(node_index)
    global_edges = np.loadtxt('./graph/{}/graph_edge.edgelist'.format(data_name), dtype=int)
    edge_index, weight = get_edge_index(global_edges)
    weight = torch.tensor(weight, dtype=torch.long)
    global_graph = nx.read_gpickle('./graph/{}/global_graph.pkl'.format(data_name))
    node_feature = global_graph._node
    features = []
    max_checkin = 0
    for key in node_dict.keys():
        node_id = node_dict[key]
        node = node_feature[node_id]
        feature = [node_id, node['poi_catid'], node['checkin_cnt'], node['latitude'], node['longitude']]
        features.append(feature)
        if node['checkin_cnt'] > max_checkin:
            max_checkin = node['checkin_cnt']
    features = torch.FloatTensor(features)
    features[:, 3: 5] = change_gps(features[:, 3: 5])
    checkins = features[:, 2: 3].numpy()
    mmn = MinMaxNormalization(min_=0, max_=max_checkin)
    mmn.fit(checkins)
    mmn_all_data = [mmn.transform(d) for d in checkins]
    features[:, 2: 3] = torch.FloatTensor(mmn_all_data)
    data = Data(x=features, edge_index=edge_index)
    pickle.dump(data, open('./processed/{}/global_graph_data.pkl'.format(data_name), 'wb'))
    pickle.dump(weight, open('./processed/{}/global_graph_weight_data.pkl'.format(data_name), 'wb'))


def build_user_graph_data():
    users = glob.glob('./graph/{}/user_graph/edges/*.edgelist'.format(data_name))
    user_count = len(users) + 1
    max_checkin_count = 234
    max_count = 0
    for i in range(1, user_count):
        idx_file = './graph/{}/user_graph/id2idx/'.format(data_name) + str(i) + '_node_id2idx.txt'
        edge_file = './graph/{}/user_graph/edges/'.format(data_name) + str(i) + '_edges.edgelist'
        graph_file = './graph/{}/user_graph/graph/'.format(data_name) + str(i) + '_graph.pkl'
        node_index = np.loadtxt(idx_file, dtype=int)
        node_dict = np_to_dict(node_index)
        graph = nx.read_gpickle(graph_file)
        edges = np.loadtxt(edge_file, dtype=int)
        edge_index, weight = get_edge_index(edges)
        weight_list = []
        for j in range(len(weight)):
            weight_list.append([j, weight[j]])
        node_feature = graph._node
        features = []
        for key in node_dict.keys():
            node_id = node_dict[key]
            node = node_feature[node_id]
            feature = [node_id, node['poi_catid'], node['checkin_cnt'], node['latitude'], node['longitude']]
            features.append(feature)
            if node['checkin_cnt'] > max_count:
                max_count = node['checkin_cnt']
        features = torch.FloatTensor(features)
        features[:, 3: 5] = change_gps(features[:, 3: 5])
        checkins = features[:, 2: 3].numpy()
        mmn = MinMaxNormalization(min_=0, max_=max_checkin_count)
        mmn.fit(checkins)
        mmn_all_data = [mmn.transform(d) for d in checkins]
        features[:, 2: 3] = torch.FloatTensor(mmn_all_data)
        data = Data(x=features, edge_index=edge_index)
        pickle.dump(data, open('./processed/{}/users/{}_user_graph_data.pkl'.format(data_name, i), 'wb'))
        pickle.dump(torch.Tensor(weight_list), open('./processed/{}/users/{}_user_graph_weight_data.pkl'.format(data_name, i), 'wb'))
    print('-----------------------')
    print(max_count)


def get_dist_dict(dist):
    dist_dict = {}
    pre_source = -1
    for row in dist:
        source = row[0]
        dest = row[1]
        dis = row[2]
        if pre_source != source:
            pre_source = source
            dist_dict[int(source)] = {int(dest): dis}
        else:
            dist_dict[int(source)][int(dest)] = dis
    return dist_dict


def build_dist_graph_data():
    graph_dist = np.loadtxt('./graph/{}/graph_dist.distlist'.format(data_name))
    dist_dict = get_dist_dict(graph_dist)
    node_index = np.loadtxt('./graph/{}/graph_node_id2idx.txt'.format(data_name), dtype=int)
    node_dict = np_to_dict(node_index)
    global_edges = np.loadtxt('./graph/{}/graph_edge.edgelist'.format(data_name), dtype=int)
    edge_index, weight = get_edge_index(global_edges)
    weight = torch.tensor(weight, dtype=torch.long)
    features = []
    max_len = -1
    for key in node_dict.keys():
        try:
            connect_nodes = dist_dict[key]
            if len(connect_nodes) > max_len:
                max_len = len(connect_nodes)
        except:
            features.append([])
            continue
        feature = []
        for i in connect_nodes.keys():
            feature.append(i)
            feature.append(connect_nodes[i])
        features.append(np.array(feature))
    for i in range(len(features)):
        pad_len = max_len * 2 - len(features[i])
        features[i] = np.pad(features[i], (0, pad_len), 'constant')
    features = torch.FloatTensor(features)
    mask = get_dist_mask(features)
    distance = torch.masked_select(features, ~mask)
    distance = distance.numpy()
    mmn = MinMaxNormalization()
    mmn.fit(distance)
    mmn_all_data = [mmn.transform(d) for d in distance]
    mmn_all_data = torch.FloatTensor(mmn_all_data)
    features = features.masked_scatter(~mask, mmn_all_data)
    data = Data(x=features, edge_index=edge_index)
    pickle.dump(data, open('./processed/{}/global_dist_data.pkl'.format(data_name), 'wb'))
    pickle.dump(weight,  open('./processed/{}/global_dist_weight_data.pkl'.format(data_name), 'wb'))


def change_gps(gps):
    gps = abs(gps * 1000000)
    la = gps[:, 0: 1].int()
    lo = gps[:, 1: 2].int()
    la_sort = la.view(-1).argsort()
    lo_sort = lo.view(-1).argsort()
    gps[:, 0: 1] = la_sort.view(-1, 1)
    gps[:, 1: 2] = lo_sort.view(-1, 1)
    return gps


def get_dist_mask(global_dist):
    mask = []
    for i in range(global_dist.shape[1] // 2):
        mask.append(True)
        mask.append(False)
    mask = torch.Tensor(mask).reshape(1, -1)
    mask = mask.repeat(global_dist.shape[0], 1)
    return mask.bool()


def mkdirs():
    if not os.path.exists('./processed'):
        os.makedirs('./processed')
    if not os.path.exists('./processed/{}'.format(data_name)):
        os.makedirs('./processed/{}'.format(data_name))
    if not os.path.exists('./processed/{}/users'.format(data_name)):
        os.makedirs('./processed/{}/users'.format(data_name))


if __name__ == '__main__':
    data_name = 'NYC'
    mkdirs()
    build_global_graph_data()
    build_user_graph_data()
    build_dist_graph_data()
