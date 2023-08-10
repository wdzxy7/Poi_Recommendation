import math
import os
import pickle
import time
import datetime
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm


def pre_process(data):
    timestamp = []
    hour = []
    day = []
    week = []
    hour_48 = []
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

    data['timestamp'] = timestamp
    data['hour'] = hour
    data['day'] = day
    data['week'] = week
    data['hour_48'] = hour_48
    data.sort_values(by='timestamp', inplace=True, ascending=True)
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
        pre_row = None
        for _, row in user_df.iterrows():
            if pre_row is None:
                pre_row = row
                continue
            if row['day'] - pre_row['day'] <= 1:
                if G.has_edge(pre_row['poi_id'], row['poi_id']):
                    G.edges[pre_row['poi_id'], row['poi_id']]['weight'] = 1
                else:  # Add new edge
                    G.add_edge(pre_row['poi_id'], row['poi_id'], weight=1, distance=math.sqrt(np.square(pre_row['latitude'] - row['latitude']) + np.square(pre_row['longitude'] - row['longitude'])))
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
        # Add Edges
        pre_row = None
        for _, row in user_df.iterrows():
            if pre_row is None:
                pre_row = row
                continue
            if row['day'] - pre_row['day'] <= 1:
                if G.has_edge(pre_row['poi_id'], row['poi_id']):
                    G.edges[pre_row['poi_id'], row['poi_id']]['weight'] = 1
                else:  # Add new edge
                    G.add_edge(pre_row['poi_id'], row['poi_id'], weight=1)
            pre_row = row
        graphs.append(G)
    return graphs, users


def save_graph_to_pickle(G):
    pickle.dump(G, open(os.path.join('./graph/{}/global_graph.pkl'.format(data_type)), 'wb'))


def save_graph_edge(G):
    nodelist = G.nodes()
    node_id2idx = {k: v for v, k in enumerate(nodelist)}

    with open(os.path.join('./graph/{}/graph_node_id2idx.txt'.format(data_type)), 'w') as f:
        for i, node in enumerate(nodelist):
            print(f'{node} {i}', file=f)

    with open(os.path.join('./graph/{}/graph_edge.edgelist'.format(data_type)), 'w') as f:
        for edge in nx.generate_edgelist(G, data=['weight']):
            src_node, dst_node, weight = edge.split(' ')
            print(f'{node_id2idx[int(src_node)]} {node_id2idx[int(dst_node)]} {weight}', file=f)


def save_graph_dist(G):
    nodelist = G.nodes()
    node_id2idx = {k: v for v, k in enumerate(nodelist)}
    with open(os.path.join('./graph/{}/graph_dist.distlist'.format(data_type)), 'w') as f:
        for edge in nx.generate_edgelist(G, data=['distance']):
            src_node, dst_node, dis = edge.split(' ')
            print(f'{node_id2idx[int(src_node)]} {node_id2idx[int(dst_node)]} {dis}', file=f)


def save_users_graph(graphs, users):
    count = 1
    for G, user in zip(graphs, users):
        nodelist = G.nodes()
        node_id2idx = {k: v for v, k in enumerate(nodelist)}

        with open(os.path.join('./graph/{}/user_graph/id2idx/{}_node_id2idx.txt'.format(data_type, count)), 'w') as f:
            for i, node in enumerate(nodelist):
                print(f'{node} {i}', file=f)

        pickle.dump(G, open(os.path.join('./graph/{}/user_graph/graph/{}_graph.pkl'.format(data_type, user)), 'wb'))

        with open(os.path.join('./graph/{}/user_graph/edges/{}_edges.edgelist'.format(data_type, user)), 'w') as f:
            for edge in nx.generate_edgelist(G, data=['weight']):
                src_node, dst_node, weight = edge.split(' ')
                print(f'{node_id2idx[int(src_node)]} {node_id2idx[int(dst_node)]} {weight}', file=f)
        count += 1


def main():
    data = pd.read_table('./data/dataset_TSMC2014_NYC.txt', encoding='latin-1')
    data.columns = ["user_id", "poi_id", "cat_id", "cat_name", "latitude", "longitude", "timezone", "time"]
    data = pre_process(data)
    data.sort_values(by=['user_id', 'timestamp'], inplace=True)
    global_graph = build_global_graph(data)
    save_graph_to_pickle(global_graph)
    save_graph_edge(global_graph)
    save_graph_dist(global_graph)
    user_all_graph, users = build_user_all_graph(data)
    save_users_graph(user_all_graph, users)


if __name__ == '__main__':
    data_type = 'nyc'
    main()