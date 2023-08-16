import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch.utils.data as data
from build_trajectory_map import pre_process


class PoiDataset(data.Dataset):
    def __init__(self, data_name, data_type='train'):
        self.data_name = data_name
        if data_name == 'NYC':
            poi_data_path = './processed/NYC/poi_data/'
            self.user_graph_path = './processed/NYC/users'
        elif data_name == 'TKY':
            poi_data_path = './processed/TKY/poi_data/'
            self.user_graph_path = './processed/TKY/users'
        with open(poi_data_path + '{}_data.pkl'.format(data_type), 'rb') as f:
            self.user_poi_data = pickle.load(f)
        self.user_graph_dict = {}
        self.load_user_graph()

    def __getitem__(self, index):
        x = self.user_poi_data[index, 0: 20]
        y = self.user_poi_data[index, -1]
        graph = self.user_graph_dict[x[0, 0]]
        return x, y, graph

    def __len__(self):
        pass

    def load_user_graph(self):
        users_graphs = glob.glob(self.user_graph_path + '/*.pkl')
        user_count = len(users_graphs)
        for graph, user in zip(users_graphs, range(1, user_count + 1)):
            with open(graph, "rb") as f:
                self.user_graph_dict[user] = pickle.load(f)


def spilt_data(data_name, current_len=20, rate=0.8):
    if data_name == 'NYC':
        data_path = './data/dataset_TSMC2014_NYC.txt'
    elif data_name == 'TKY':
        data_path = './data/dataset_TSMC2014_TKY.txt'
    df = pd.read_table(data_path, encoding='latin-1')
    df.columns = ["user_id", "poi_id", "cat_id", "cat_name", "latitude", "longitude", "timezone", "time"]
    df = pre_process(df)
    df.drop(['cat_name', 'time', 'timezone'], axis=1, inplace=True)
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
    # spilt_data('NYC')
    data = PoiDataset('NYC', 'train')