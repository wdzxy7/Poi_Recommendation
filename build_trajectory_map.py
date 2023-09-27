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


def pre_process(data):
    timestamp = []
    hour = []
    day = []
    week = []
    hour_48 = []
    deal_time = []
    for i in range(len(data)):
        times = data['time'].values[i]
        timestamp.append(time.mktime(time.strptime(times, '%a %b %d %H:%M:%S %z %Y')))
        t = datetime.datetime.strptime(times, '%a %b %d %H:%M:%S %z %Y')
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
    for venueid, group in data.groupby('poi_id'):
        indexs = group.index
        if len(set(group['cat_id'].values)) > 1:
            for i in range(len(group)):
                data.loc[indexs[i], 'cat_id'] = group.loc[indexs[0]]['cat_id']

    data = data.drop_duplicates()
    data['cat_id'] = data['cat_id'].rank(method='dense').values
    data['cat_id'] = data['cat_id'].astype(int)
    data['timestamp'] = data['timestamp'].astype(int)
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
                G.add_node(row['poi_id'],
                           checkin_cnt=1,
                           poi_catid=row['cat_id'],
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
                G.add_node(row['poi_id'],
                           checkin_cnt=1,
                           poi_catid=row['cat_id'],
                           poi_catname=row['cat_name'],
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
    if data_name == 'TKY':
        data = pd.read_table('./data/dataset_TSMC2014_TKY.txt', header=None, encoding="latin-1")
        data = build_trajectory(data)
    print(len(set(data['poi_id'])), len(set(data['user_id'])), len(set(data['cat_id'])), len(data))
    data = filter_data(data)
    print(len(set(data['poi_id'])), len(set(data['user_id'])), len(set(data['cat_id'])), len(data))
    global_graph = build_global_graph(data)
    save_graph_to_pickle(global_graph)
    save_graph_edge(global_graph)
    save_graph_dist(global_graph)
    user_all_graph, users = build_user_all_graph(data)
    save_users_graph(user_all_graph, users)


def build_trajectory(df):
    df.columns = ["user_id", "poi_id", "cat_id", "cat_name", "latitude", "longitude", "timezone", "time"]
    df = pre_process(df)
    df.sort_values(by=['user_id', 'time'], inplace=True, ascending=True)
    trajectory = []
    for venueid, group in df.groupby('user_id'):
        user_id = venueid
        start_time = group['time'].iloc[0]
        trajectory_id = 1
        for _, u in group.iterrows():
            now_time = u['time']
            if now_time- pd.Timedelta(10, "H") <= start_time:
                trajectory.append(str(user_id) + '_' + str(trajectory_id))
            else:
                trajectory_id += 1
                trajectory.append(str(user_id) + '_' + str(trajectory_id))
                start_time = now_time
    df['trajectory_id'] = trajectory
    return df


def filter_data(data):
    for poi_id, group in data.groupby('poi_id'):
        if group.shape[0] < 10:
            data.drop(data[(data.poi_id == poi_id)].index, inplace=True)
    for user_id, group in data.groupby('user_id'):
        if group.shape[0] < 50:
            data.drop(data[(data.user_id == user_id)].index, inplace=True)
    for group_id, group in data.groupby('trajectory_id'):
        if group.shape[0] < 3:
            data.drop(data[(data.trajectory_id == group_id)].index, inplace=True)
    for user_id, group in data.groupby('user_id'):
        if len(set(group.trajectory_id)) < 3:
            data.drop(data[(data.user_id == user_id)].index, inplace=True)
    # resort
    data['user_id'] = data['user_id'].rank(method='dense').values
    data['user_id'] = data['user_id'].astype(int)
    data['poi_id'] = data['poi_id'].rank(method='dense').values
    data['poi_id'] = data['poi_id'].astype(int)
    for venueid, group in data.groupby('poi_id'):
        indexs = group.index
        if len(set(group['cat_id'].values)) > 1:
            for i in range(len(group)):
                data.loc[indexs[i], 'cat_id'] = group.loc[indexs[0]]['cat_id']
    data = data.drop_duplicates()
    data['cat_id'] = data['cat_id'].rank(method='dense').values
    data['cat_id'] = data['cat_id'].astype(int)
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


if __name__ == '__main__':
    data_name = 'NYC'
    mkdirs()
    main()