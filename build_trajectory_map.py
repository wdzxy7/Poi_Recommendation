import math
import os
import pickle
import time
import datetime
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from pandas import DataFrame


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



def pre_process(data):
    timestamp = []
    hour = []
    day = []
    week = []
    hour_48 = []
    deal_time = []
    for i in range(len(data)):
        times = data['time'].values[i]
        try:
            timestamp.append(time.mktime(time.strptime(times, '%a %b %d %H:%M:%S %z %Y')))
            t = datetime.datetime.strptime(times, '%a %b %d %H:%M:%S %z %Y')
        except:
            timestamp.append(time.mktime(time.strptime(times, '%Y-%m-%dT%H:%M:%SZ')))
            t = datetime.datetime.strptime(times, '%Y-%m-%dT%H:%M:%SZ')
        year = int(t.strftime('%Y'))
        day_i = int(t.strftime('%j'))
        week_i = int(t.strftime('%w'))
        hour_i = int(t.strftime('%H'))
        hour_i_48 = hour_i
        if week_i == 0 or week_i == 6:
            hour_i_48 = hour_i + 24
        if year == 2013:
            day_i = day_i + 366
        day.append(day_i)
        hour.append(hour_i)
        hour_48.append(int(hour_i_48))
        week.append(week_i)
        deal_time.append(t)

    data['timestamp'] = timestamp
    data['hour'] = hour
    data['day'] = day
    data['week'] = week
    data['hour_48'] = hour_48
    data['time'] = deal_time
    data.sort_values(by=['user_id', 'timestamp'], inplace=True, ascending=True)
    #################################################################################
    # 2、filter users and POIs
    data['user_id'] = data['user_id'].rank(method='dense').values
    data['user_id'] = data['user_id'].astype(int)
    data['poi_id'] = data['poi_id'].rank(method='dense').values
    data['poi_id'] = data['poi_id'].astype(int)
    try:
        for venueid, group in data.groupby('poi_id'):
            indexs = group.index
            if len(set(group['cat_id'].values)) > 1:
                for i in range(len(group)):
                    data.loc[indexs[i], 'cat_id'] = group.loc[indexs[0]]['cat_id']
        data['cat_id'] = data['cat_id'].rank(method='dense').values
        data['cat_id'] = data['cat_id'].astype(int)
        data['timestamp'] = data['timestamp'].astype(int)
    except:
        pass
    data = data.drop_duplicates()
    return data


def build_global_graph(df):
    G = nx.DiGraph()
    users = list(set(df['user_id'].to_list()))
    loop = tqdm(users)
    for user_id in loop:
        user_df = df[df['user_id'] == user_id]
        # Add Nodes
        for i, row in user_df.iterrows():
            node = row['poi_id']
            if node not in G.nodes():
                if data_name != 'CA':
                    G.add_node(row['poi_id'],
                               checkin_cnt=1,
                               poi_catid=row['cat_id'],
                               latitude=row['latitude'],
                               longitude=row['longitude'])
                else: # CA dataset
                    G.add_node(row['poi_id'],
                               checkin_cnt=1,
                               latitude=row['latitude'],
                               longitude=row['longitude'])
            else:
                G.nodes[node]['checkin_cnt'] += 1
        # Add Edges
        previous_poi_id = 0
        previous_traj_id = 0
        for _, row in user_df.iterrows():
            poi_id = row['poi_id']
            traj_id = row['trajectory_id']
            # No edge for the begin of the seq or different traj
            if (previous_poi_id == 0) or (previous_traj_id != traj_id):
                previous_poi_id = poi_id
                previous_traj_id = traj_id
                pre_row = row
                continue
            # Add edges
            if G.has_edge(previous_poi_id, poi_id):
                G.edges[previous_poi_id, poi_id]['weight'] += 1
            else:  # Add new edge
                G.add_edge(previous_poi_id, poi_id, weight=1, distance=math.sqrt(np.square(pre_row['latitude'] - row['latitude']) + np.square(pre_row['longitude'] - row['longitude'])))
            previous_traj_id = traj_id
            previous_poi_id = poi_id
            pre_row = row
    return G


def build_user_all_graph(df):
    graphs = []
    users = list(set(df['user_id'].to_list()))
    loop = tqdm(users)
    for user_id in loop:
        G = nx.DiGraph()
        user_df = df[df['user_id'] == user_id]
        # Add Nodes
        for i, row in user_df.iterrows():
            node = row['poi_id']
            if node not in G.nodes():
                if data_name != 'CA':
                    G.add_node(row['poi_id'],
                               checkin_cnt=1,
                               poi_catid=row['cat_id'],
                               poi_catname=row['cat_name'],
                               latitude=row['latitude'],
                               longitude=row['longitude'])
                else: # CA dataset
                    G.add_node(row['poi_id'],
                               checkin_cnt=1,
                               latitude=row['latitude'],
                               longitude=row['longitude'])
            else:
                G.nodes[node]['checkin_cnt'] += 1
        # Add edges (Check-in seq)
        previous_poi_id = 0
        previous_traj_id = 0
        for i, row in user_df.iterrows():
            poi_id = row['poi_id']
            traj_id = row['trajectory_id']
            # No edge for the begin of the seq or different traj
            if (previous_poi_id == 0) or (previous_traj_id != traj_id):
                previous_poi_id = poi_id
                previous_traj_id = traj_id
                continue
            # Add edges
            if G.has_edge(previous_poi_id, poi_id):
                G.edges[previous_poi_id, poi_id]['weight'] += 1
            else:  # Add new edge
                G.add_edge(previous_poi_id, poi_id, weight=1)
            previous_traj_id = traj_id
            previous_poi_id = poi_id
        graphs.append(G)
    return graphs, users


def save_graph_to_pickle(G):
    pickle.dump(G, open(os.path.join('./graph/{}/global_graph.pkl'.format(data_name)), 'wb'))


def save_graph_edge(G):
    nodelist = G.nodes()
    node_id2idx = {k: v for v, k in enumerate(nodelist)}

    with open(os.path.join('./graph/{}/graph_node_id2idx.txt'.format(data_name)), 'w') as f:
        for i, node in enumerate(nodelist):
            print(f'{node} {i}', file=f)

    with open(os.path.join('./graph/{}/graph_edge.edgelist'.format(data_name)), 'w') as f:
        for edge in nx.generate_edgelist(G, data=['weight']):
            src_node, dst_node, weight = edge.split(' ')
            print(f'{node_id2idx[int(src_node)]} {node_id2idx[int(dst_node)]} {weight}', file=f)


def save_graph_dist(G):
    nodelist = G.nodes()
    node_id2idx = {k: v for v, k in enumerate(nodelist)}
    with open(os.path.join('./graph/{}/graph_dist.distlist'.format(data_name)), 'w') as f:
        for edge in nx.generate_edgelist(G, data=['distance']):
            src_node, dst_node, dis = edge.split(' ')
            print(f'{node_id2idx[int(src_node)]} {node_id2idx[int(dst_node)]} {dis}', file=f)


def save_users_graph(graphs, users):
    for G, user in zip(graphs, users):
        nodelist = G.nodes()
        node_id2idx = {k: v for v, k in enumerate(nodelist)}

        with open(os.path.join('./graph/{}/user_graph/id2idx/{}_node_id2idx.txt'.format(data_name, user)), 'w') as f:
            for i, node in enumerate(nodelist):
                print(f'{node} {i}', file=f)

        pickle.dump(G, open(os.path.join('./graph/{}/user_graph/graph/{}_graph.pkl'.format(data_name, user)), 'wb'))

        with open(os.path.join('./graph/{}/user_graph/edges/{}_edges.edgelist'.format(data_name, user)), 'w') as f:
            for edge in nx.generate_edgelist(G, data=['weight']):
                src_node, dst_node, weight = edge.split(' ')
                print(f'{node_id2idx[int(src_node)]} {node_id2idx[int(dst_node)]} {weight}', file=f)


def main():
    if data_name == 'NYC':
        data = pd.read_table('./data/dataset_TSMC2014_NYC.txt', header=None, encoding="latin-1")
        data = build_trajectory(data)
    elif data_name == 'TKY':
        data = pd.read_table('./data/dataset_TSMC2014_TKY.txt', header=None, encoding="latin-1")
        data = build_trajectory(data)
    elif data_name == 'CA':
        data = pd.read_table('./data/Gowalla_totalCheckins.txt', header=None, encoding="latin-1")
        data = build_trajectory(data)
    if data_name != 'CA':
        print('Raw data:\ndata_len: {}\t\t poi_len: {}\t\t cat_len: {}\t\t user_len: {}\t\t trajectory_len: {}\t\t '.format(
            len(data), len(set(data['poi_id'])), len(set(data['cat_id'])), len(set(data['user_id'])), len(set(data['trajectory_id']))))
    else:
        print(
            'Raw data:\ndata_len: {}\t\t poi_len: {}\t\tuser_len: {}\t\t trajectory_len: {}\t\t '.format(
             len(data), len(set(data['poi_id'])), len(set(data['user_id'])), len(set(data['trajectory_id']))))
    statistic_data(data)
    data = filter_data(data)
    df = normal_data(data)
    df.to_csv('./data/filter_data/{}_filter.csv'.format(data_name), index_label=False)
    if data_name != 'CA':
        print('Flitered data:\ndata_len: {}\t\t poi_len: {}\t\t cat_len: {}\t\t user_len: {}\t\t trajectory_len: {}\t\t '.format(
            len(data), len(set(data['poi_id'])), len(set(data['cat_id'])), len(set(data['user_id'])),
            len(set(data['trajectory_id']))))
    else:
        print(
            'Flitered data:\ndata_len: {}\t\t poi_len: {}\t\t user_len: {}\t\t trajectory_len: {}\t\t '.format(
             len(data), len(set(data['poi_id'])), len(set(data['user_id'])), len(set(data['trajectory_id']))))
    global_graph = build_global_graph(data)
    save_graph_to_pickle(global_graph)
    save_graph_edge(global_graph)
    save_graph_dist(global_graph)
    user_all_graph, users = build_user_all_graph(data)
    save_users_graph(user_all_graph, users)


def build_trajectory(df):
    try:
        df.columns = ["user_id", "poi_id", "cat_id", "cat_name", "latitude", "longitude", "timezone", "time"]
    except:
        df.columns = ["user_id", "time", "latitude", "longitude", "poi_id"]
    df = pre_process(df)
    df.sort_values(by=['user_id', 'time'], inplace=True, ascending=True)
    trajectory = []
    for venueid, group in df.groupby('user_id'):
        user_id = venueid
        start_time = group['time'].iloc[0]
        trajectory_id = 1
        for _, u in group.iterrows():
            now_time = u['time']
            if now_time - pd.Timedelta(24, "H") <= start_time:
                trajectory.append(str(user_id) + '_' + str(trajectory_id))
            else:
                trajectory_id += 1
                trajectory.append(str(user_id) + '_' + str(trajectory_id))
                start_time = now_time
    df['trajectory_id'] = trajectory
    return df


def filter_data(data):
    res_data = pd.DataFrame([])
    for poi_id, group in data.groupby('poi_id'):
        if group.shape[0] >= 10:
            res_data = res_data.append(group)
    data = res_data.copy()
    res_data = pd.DataFrame([])
    for user_id, group in data.groupby('user_id'):
        if group.shape[0] >= 10:
            res_data = res_data.append(group)
    data = res_data.copy()
    print(len(set(data['user_id'])))
    res_data = pd.DataFrame([])
    for group_id, group in data.groupby('trajectory_id'):
        if group.shape[0] >= 4: # 6 4951
            res_data = res_data.append(group)
    data = res_data.copy()
    print(len(set(data['user_id'])))
    res_data = pd.DataFrame([])
    for user_id, group in data.groupby('user_id'):
        if len(set(group.trajectory_id)) >= 4:
            res_data = res_data.append(group)
    data = res_data.copy()
    print(len(set(data['user_id'])))
    # resort
    data['user_id'] = data['user_id'].rank(method='dense').values
    data['user_id'] = data['user_id'].astype(int)
    data['poi_id'] = data['poi_id'].rank(method='dense').values
    data['poi_id'] = data['poi_id'].astype(int)
    try:
        for venueid, group in data.groupby('poi_id'):
            indexs = group.index
            if len(set(group['cat_id'].values)) > 1:
                for i in range(len(group)):
                    data.loc[indexs[i], 'cat_id'] = group.loc[indexs[0]]['cat_id']
        data['cat_id'] = data['cat_id'].rank(method='dense').values
        data['cat_id'] = data['cat_id'].astype(int)
    except:
        pass
    data = data.drop_duplicates()
    return data


def mkdirs():
    if not os.path.exists('./graph'):
        os.makedirs('./graph')
    if not os.path.exists('./graph/{}'.format(data_name)):
        os.makedirs('./graph/{}'.format(data_name))
    if not os.path.exists('./graph/{}/user_graph'.format(data_name)):
        os.makedirs('./graph/{}/user_graph'.format(data_name))
    if not os.path.exists('./graph/{}/user_graph/edges'.format(data_name)):
        os.makedirs('./graph/{}/user_graph/edges'.format(data_name))
    if not os.path.exists('./graph/{}/user_graph/graph'.format(data_name)):
        os.makedirs('./graph/{}/user_graph/graph'.format(data_name))
    if not os.path.exists('./graph/{}/user_graph/id2idx'.format(data_name)):
        os.makedirs('./graph/{}/user_graph/id2idx'.format(data_name))


def statistic_data(data):
    count = 0
    for _, user in data.groupby('user_id'):
        if len(user) <= 10:
            count += 1
    print(data['user_id'], count)
    count = 0
    for _, user in data.groupby('trajectory_id'):
        if len(user) < 3:
            count += 1
    print(data['trajectory_id'], count)


if __name__ == '__main__':
    data_name = 'CA'
    mkdirs()
    main()