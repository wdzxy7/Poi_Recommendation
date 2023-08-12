import glob
import torch
import pickle
import numpy as np
import networkx as nx
from torch_geometric.data import Data


def np_to_dict(node_index):
    node = node_index[:, 0].tolist()
    ind = node_index[:, 1].tolist()
    return dict(zip(ind, node))


def get_edge_index(edges):
    source = edges[:, 0]
    dest = edges[:, 1]
    edge_index = torch.tensor([source, dest], dtype=torch.long)
    return edge_index


def build_global_graph_data():
    node_index = np.loadtxt('./graph/{}/graph_node_id2idx.txt'.format(data_type), dtype=int)
    node_dict = np_to_dict(node_index)
    global_edges = np.loadtxt('./graph/{}/graph_edge.edgelist'.format(data_type), dtype=int)
    edge_index = get_edge_index(global_edges)
    global_graph = nx.read_gpickle('./graph/{}/global_graph.pkl'.format(data_type))
    node_feature = global_graph._node
    features = []
    for key in node_dict.keys():
        node_id = node_dict[key]
        node = node_feature[node_id]
        feature = [node['checkin_cnt'], node['poi_catid'], node['latitude'], node['longitude']]
        features.append(feature)
    features = torch.LongTensor(features)
    data = Data(x=features, edge_index=edge_index)
    pickle.dump(data, open('./processed/{}/global_graph_data.pkl'.format(data_type), 'wb'))


def build_user_graph_data():
    users = glob.glob('./graph/{}/user_graph/edges/*.edgelist'.format(data_type))
    user_count = len(users) + 1
    for i in range(1, user_count):
        idx_file = './graph/{}/user_graph/id2idx/'.format(data_type) + str(i) + '_node_id2idx.txt'
        edge_file = './graph/{}/user_graph/edges/'.format(data_type) + str(i) + '_edges.edgelist'
        graph_file = './graph/{}/user_graph/graph/'.format(data_type) + str(i) + '_graph.pkl'
        node_index = np.loadtxt(idx_file, dtype=int)
        node_dict = np_to_dict(node_index)
        graph = nx.read_gpickle(graph_file)
        edges = np.loadtxt(edge_file, dtype=int)
        edge_index = get_edge_index(edges)
        node_feature = graph._node
        features = []
        for key in node_dict.keys():
            node_id = node_dict[key]
            node = node_feature[node_id]
            feature = [node['checkin_cnt'], node['poi_catid'], node['latitude'], node['longitude']]
            features.append(feature)
        features = torch.LongTensor(features)
        data = Data(x=features, edge_index=edge_index)
        pickle.dump(data, open('./processed/{}/users/{}_user_graph_data.pkl'.format(data_type, i), 'wb'))


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
    graph_dist = np.loadtxt('./graph/{}/graph_dist.distlist'.format(data_type))
    dist_dict = get_dist_dict(graph_dist)
    node_index = np.loadtxt('./graph/{}/graph_node_id2idx.txt'.format(data_type), dtype=int)
    node_dict = np_to_dict(node_index)
    global_edges = np.loadtxt('./graph/{}/graph_edge.edgelist'.format(data_type), dtype=int)
    edge_index = get_edge_index(global_edges)
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
    features = torch.LongTensor(features)
    data = Data(x=features, edge_index=edge_index)
    pickle.dump(data, open('./processed/{}/global_dist_data.pkl'.format(data_type), 'wb'))


if __name__ == '__main__':
    data_type = 'nyc'
    # build_global_graph_data()
    # build_user_graph_data()
    build_dist_graph_data()
