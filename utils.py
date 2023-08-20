import os
import glob
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as data
from build_trajectory_map import pre_process


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
        if data_name == 'NYC':
            poi_data_path = './processed/NYC/poi_data/'
            self.user_graph_path = './processed/NYC/users'
        elif data_name == 'TKY':
            poi_data_path = './processed/TKY/poi_data/'
            self.user_graph_path = './processed/TKY/users'
        with open(poi_data_path + '{}_data.pkl'.format(data_type), 'rb') as f:
            self.user_poi_data = pickle.load(f)
        self.user_poi_data = torch.tensor(self.user_poi_data).float()
        self.user_graph_dict = {}
        self.load_user_graph()
        self.pad_graph()
        self.data_len = len(self.user_poi_data)

    def __getitem__(self, index):
        x = self.user_poi_data[index, 0: 20]
        y = self.user_poi_data[index, -1]
        graph = self.user_graph_dict[int(x[0, 0])]
        return x, y[1], graph.x, graph.edge_index

    def __len__(self):
        return self.data_len

    def pad_graph(self):
        for key in self.user_graph_dict.keys():
            l = self.user_graph_dict[key].x.shape[0]
            pad = nn.ZeroPad2d(padding=(0, 0, 0, self.max_graph_node - l))
            self.user_graph_dict[key].x = pad(self.user_graph_dict[key].x)

    def load_user_graph(self):
        users_graphs = glob.glob(self.user_graph_path + '/*.pkl')
        user_count = len(users_graphs)
        for graph, user in zip(users_graphs, range(1, user_count + 1)):
            with open(graph, "rb") as f:
                self.user_graph_dict[user] = pickle.load(f)
                if self.user_graph_dict[user].x.shape[0] > self.max_graph_node:
                    self.max_graph_node = self.user_graph_dict[user].x.shape[0]


def spilt_data(data_name, current_len=20, rate=0.8):
    if data_name == 'NYC':
        data_path = './data/dataset_TSMC2014_NYC.txt'
    elif data_name == 'TKY':
        data_path = './data/dataset_TSMC2014_TKY.txt'
    df = pd.read_table(data_path, encoding='latin-1')
    df.columns = ["user_id", "poi_id", "cat_id", "cat_name", "latitude", "longitude", "timezone", "time"]
    df = pre_process(df)
    df.drop(['cat_name', 'time', 'timezone'], axis=1, inplace=True)
    mmn = MinMaxNormalization()
    arr = np.array(df[['latitude', 'longitude']])
    mmn.fit(arr)
    mmn_all_data = [mmn.transform(d) for d in arr]
    df[['latitude', 'longitude']] = mmn_all_data
    train_data = []
    test_data = []
    for _, group in df.groupby('user_id'):
        max_len = group.shape[0]
        front = 0
        back = current_len + 1
        user_dataset = []
        while back <= max_len:
            slice_data = group[front: back]
            front += 1
            back += 1
            user_dataset.append(slice_data)
        train_len = int(len(user_dataset) * rate)
        test_len = len(user_dataset) - train_len
        train = user_dataset[: train_len]
        test = user_dataset[-test_len:]
        train_data += train
        test_data += test
    if not os.path.exists('./processed/{}/poi_data'.format(data_name)):
        os.makedirs('./processed/{}/poi_data'.format(data_name))
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    pickle.dump(train_data, open('./processed/{}/poi_data/train_data.pkl'.format(data_name), 'wb'))
    pickle.dump(test_data, open('./processed/{}/poi_data/test_data.pkl'.format(data_name), 'wb'))


if __name__ == '__main__':
    spilt_data('NYC')
    data = PoiDataset('NYC', 'train')
