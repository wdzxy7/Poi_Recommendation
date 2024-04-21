import re
import os
import glob
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as data
from build_trajectory_map import pre_process, filter_data, build_trajectory


class MinMaxNormalization(object):
    def __init__(self, min_=None, max_=None):
        self.min = min_
        self.max = max_
        pass

    def fit(self, X):
        if self.min is None:
            self._min = X.min()
            self._max = X.max()
        else:
            self._min = self.min
            self._max = self.max
        print("min:", self._min, "max: ", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X


class PoiDataset(data.Dataset):
    def __init__(self, data_name, data_type='train'):
        self.data_name = data_name
        self.max_graph_node = 0
        self.max_graph_edges = 0
        self.max_graph_weight_len = 0
        if data_name == 'NYC':
            poi_data_path = './processed/NYC/poi_data/'
            self.user_graph_path = './processed/NYC/users'
        elif data_name == 'TKY':
            poi_data_path = './processed/TKY/poi_data/'
            self.user_graph_path = './processed/TKY/users'
        with open(poi_data_path + '{}_data.pkl'.format(data_type), 'rb') as f:
            self.user_poi_data = pickle.load(f)
        with open(poi_data_path + 'poi_neighbor.pkl', 'rb') as f:
            self.poi_neighbor_data = pickle.load(f)
        self.poi_data = []
        self.past_data = []
        self.trajectory_len = []
        self.process_neighbor()
        self.convert_tensor()
        self.user_graph_dict = {}
        self.user_graph_weight_dict = {}
        self.load_user_graph()
        self.pad_graph()
        self.data_len = len(self.poi_data)

    def __getitem__(self, index):
        x = self.poi_data[index][:-1]
        last_poi = x[-1, 1]
        neighbor = self.poi_neighbor_data[int(last_poi)]
        y = self.poi_data[index][1:][:, 1]
        graph = self.user_graph_dict[int(x[0][0])]
        weight = self.user_graph_weight_dict[int(x[0][0])]
        return x, y, self.trajectory_len[index], graph.x, graph.edge_index, weight, neighbor

    def __len__(self):
        return self.data_len

    def convert_tensor(self):
        for line in zip(self.user_poi_data):
            poi = torch.Tensor(line)[0]
            self.poi_data.append(poi)
            self.trajectory_len.append(poi.shape[0] - 1)

    def process_neighbor(self):
        t_dict = dict()
        for key in self.poi_neighbor_data.keys():
            pois = self.poi_neighbor_data[key]
            pois = np.array(pois)
            pois = torch.Tensor(pois[:, 0])
            t_dict[key] = pois
        self.poi_neighbor_data = t_dict

    def pad_graph(self):
        for key in self.user_graph_dict.keys():
            #  padding nodes
            nodes = self.user_graph_dict[key].x.shape[0]
            pad = nn.ZeroPad2d(padding=(0, 0, 0, self.max_graph_node - nodes))
            self.user_graph_dict[key].x = pad(self.user_graph_dict[key].x)
            # padding edges
            edges = self.user_graph_dict[key].edge_index.shape[1]
            pad = nn.ZeroPad2d(padding=(0, self.max_graph_edges - edges, 0, 0))
            self.user_graph_dict[key].edge_index = pad(self.user_graph_dict[key].edge_index)
            #  padding weight
            weight_len = self.user_graph_weight_dict[key].shape[0]
            pad = nn.ZeroPad2d(padding=(0, 0, 0, self.max_graph_edges - weight_len))
            self.user_graph_weight_dict[key] = pad(self.user_graph_weight_dict[key])

    def load_user_graph(self):
        users_graphs = glob.glob(self.user_graph_path + '/*graph_data.pkl')
        for graph_file in users_graphs:
            user = re.findall(r'users/(.*?)_user', graph_file)[0]
            user = int(user)
            with open(graph_file, "rb") as f:
                self.user_graph_dict[user] = pickle.load(f)
                if self.user_graph_dict[user].x.shape[0] > self.max_graph_node:
                    self.max_graph_node = self.user_graph_dict[user].x.shape[0]
                if self.user_graph_dict[user].edge_index.shape[1] > self.max_graph_edges:
                    self.max_graph_edges = self.user_graph_dict[user].edge_index.shape[1]
            weight_file = graph_file.replace('graph_data', 'graph_weight_data')
            with open(weight_file, "rb") as f:
                self.user_graph_weight_dict[user] = pickle.load(f)
        # print(self.max_graph_node)

    def timestamp2vec(self, days):
        ret = []
        for day in days:
            vec = []
            day = int(day)
            if day == 0:
                day = 7
            for i in range(7 - day):
                vec.append(0)
            vec.append(1)
            for i in range(day - 1):
                vec.append(0)
            ret.append(vec)
        return np.asarray(ret)


def normal_data(df):
    mmn = MinMaxNormalization()
    arr = np.array(df[['latitude', 'longitude']])
    mmn.fit(arr)
    mmn_all_data = [mmn.transform(d) for d in arr]
    df[['latitude', 'longitude']] = mmn_all_data
    mmn = MinMaxNormalization(min_=94, max_=411)
    arr = np.array(df[['day']])
    mmn.fit(arr)
    mmn_all_data = [mmn.transform(d) for d in arr]
    df[['day']] = mmn_all_data
    return df


def spilt_data(data_name, rate=0.8, read=False):
    val_rate = (1 - rate) / 2
    if read:
        df = pd.read_csv('./data/{}_filter.csv'.format(data_name))
        df.drop(['cat_name', 'time', 'timezone', 'hour_48', 'day', 'latitude', 'longitude', 'timestamp'], axis=1,
                inplace=True)
    else:
        if data_name == 'NYC':
            data_path = './data/dataset_TSMC2014_NYC.txt'
        elif data_name == 'TKY':
            data_path = './data/dataset_TSMC2014_TKY.txt'
        df = pd.read_table(data_path, header=None, encoding="latin-1")
        df = build_trajectory(df)
        print(len(set(df['poi_id'])), len(set(df['user_id'])), len(set(df['cat_id'])), len(df), len(set(df['trajectory_id'])))
        df = filter_data(df)
        df = normal_data(df)
        df.to_csv('./data/{}_filter.csv'.format(data_name), index_label=False)
        df.drop(['cat_name', 'time', 'timezone', 'hour_48', 'day', 'latitude', 'longitude', 'timestamp'], axis=1, inplace=True)
    print(
        'Flitered data:\ndata_len: {}\t\t poi_len: {}\t\t cat_len: {}\t\t user_len: {}\t\t trajectory_len: {}\t\t '.format(
            len(df), len(set(df['poi_id'])), len(set(df['cat_id'])), len(set(df['user_id'])),
            len(set(df['trajectory_id']))))
    train_data = []
    test_data = []
    val_data = []
    for _, group in df.groupby('user_id'):
        trajectory_len = len(set(group['trajectory_id']))
        train_len = int(trajectory_len * rate)
        test_len, val_len = int(trajectory_len * val_rate * 2), int(trajectory_len * val_rate * 0)
        if train_len == 0:
            train_len = trajectory_len
        train = []
        test = []
        val = []
        count = 0
        for _, tra_group in group.groupby('trajectory_id'):
            if count == 0:
                count = 1
                train_len -= 1
                continue
            if train_len > 0:
                train.append(np.array(tra_group.drop(['trajectory_id'], axis=1)))
                train_len -= 1
            elif val_len > 0:
                val.append(np.array(tra_group.drop(['trajectory_id'], axis=1)))
                val_len -= 1
            elif test_len > 0:
                test.append(np.array(tra_group.drop(['trajectory_id'], axis=1)))
                test_len -= 1
        train_data += train
        val_data += val
        test_data += test
    if not os.path.exists('./processed/{}/poi_data'.format(data_name)):
        os.makedirs('./processed/{}/poi_data'.format(data_name))
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    val_data = np.array(val_data)
    pickle.dump(train_data, open('./processed/{}/poi_data/train_data.pkl'.format(data_name), 'wb'))
    pickle.dump(val_data, open('./processed/{}/poi_data/val_data.pkl'.format(data_name), 'wb'))
    pickle.dump(test_data, open('./processed/{}/poi_data/test_data.pkl'.format(data_name), 'wb'))
    print(len(train_data), len(val_data), len(test_data))


def get_poi_neighbor(data_name, top):
    df = pd.read_csv('./data/{}_filter.csv'.format(data_name))
    df = df[['poi_id', 'latitude', 'longitude']]
    df.drop_duplicates(subset=['poi_id'], inplace=True)
    print(df.head())
    dist_dict = dict()
    for i in range(df.shape[0]):
        poi = int(df.iloc[i]['poi_id'])
        point1 = np.array([df.iloc[i]['latitude'], df.iloc[i]['longitude']])
        dist_dict[poi] = []
        for j in range(df.shape[0]):
            point2 = np.array([df.iloc[j]['latitude'], df.iloc[j]['longitude']])
            dist = np.linalg.norm(point1 - point2) * 1000000
            dist_dict[poi].append([int(df.iloc[j]['poi_id']), dist])
    for key in dist_dict.keys():
        dist_dict[key] = sorted(dist_dict[key], key=lambda x: x[1])
        dist_dict[key] = dist_dict[key][:top]
    pickle.dump(dist_dict, open(os.path.join('./processed/{}/poi_data/poi_neighbor.pkl'.format(data_name)), 'wb'))


if __name__ == '__main__':
    spilt_data('CA', rate=0.8, read=True)
    get_poi_neighbor('CA', 20)
    # dataset = PoiDataset('NYC', 'train')
    # train_loader = data.DataLoader(dataset=dataset, batch_size=32, shuffle=False, collate_fn=lambda x: x)
    # for _, batch_data in enumerate(train_loader, 1):
    #     print(1)
    #     break