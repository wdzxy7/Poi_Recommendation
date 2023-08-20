import os
import random
import sys
import torch
import pickle
import logging
import argparse
import torch.nn as nn
from utils import PoiDataset
import torch.utils.data as data
from Model import GlobalGraphNet, GlobalDistNet, UserGraphNet, UserHistoryNet, GlobalUserNet


parser = argparse.ArgumentParser(description='Parameters for my model')
parser.add_argument('--poi_len', type=int, default=38333, help='The length of POI_id,NYC is 38333, TKY is 61858')
parser.add_argument('--user_len', type=int, default=1083, help='The length of users')
parser.add_argument('--cat_len', type=int, default=400, help='The length of category')
parser.add_argument('--node_len', type=int, default=400, help='The length of user graph node')
parser.add_argument('--global_graph_dim', type=int, default=128, help='The embedding dim of GlobalGraphNet')
parser.add_argument('--global_dist_dim', type=int, default=128, help='The embedding dim of GlobalDistNet')
parser.add_argument('--global_dist_features', type=int, default=898, help='The feature sum of global distance graph')
parser.add_argument('--user_graph_dim', type=int, default=128, help='The embedding dim of UserGraphNet')
parser.add_argument('--user_history_dim', type=int, default=128, help='The embedding dim of UserHistoryNet')
parser.add_argument('--epochs', type=int, default=150, help='Epochs of train')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size of dataloader')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of optimizer')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight_decay of optimizer')
parser.add_argument('--data_name', type=str, default='NYC', help='Train data name')
parser.add_argument('--gpu_num', type=int, default=0, help='Choose which GPU to use')
parser.add_argument('--test_num', type=str, default='1', help='Just for test')
parser.add_argument('--seed', type=int, default=666, help='random seed')


def load_data():
    global train_len, test_len
    train_dataset = PoiDataset(data_name, data_type='train')
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = PoiDataset(data_name, data_type='test')
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    train_len = train_dataset.data_len
    test_len = test_dataset.data_len
    return train_loader, test_loader


def load_global_graph():
    with open('./processed/{}/global_graph_data.pkl'.format(data_name), 'rb') as f:
        graph_data = pickle.load(f)
    with open('./processed/{}/global_dist_data.pkl'.format(data_name), 'rb') as f:
        dist_data = pickle.load(f)
    return graph_data, dist_data


def get_dist_mask(global_dist):
    mask = []
    for i in range(global_dist.x.shape[1] // 2):
        mask.append(True)
        mask.append(False)
    mask = torch.Tensor(mask).reshape(1, -1)
    mask = mask.repeat(global_dist.x.shape[0], 1)
    return mask.bool()


def train():
    train_loader, test_loader = load_data()
    global_graph, global_dist = load_global_graph()
    dist_mask = get_dist_mask(global_dist)
    global_graph_model = GlobalGraphNet()
    global_dist_model = GlobalDistNet()
    user_graph_model = UserGraphNet()
    user_history_model = UserHistoryNet()
    global_user_model = GlobalUserNet()
    global_graph_model.to(device)
    global_dist_model.to(device)
    user_graph_model.to(device)
    user_history_model.to(device)
    global_user_model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(list(global_graph_model.parameters()) +
                                 list(global_dist_model.parameters()) +
                                 list(user_graph_model.parameters()) +
                                 list(user_history_model.parameters()) +
                                 list(global_user_model.parameters())
                                 , lr=lr, weight_decay=weight_decay)
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    train_len = len(train_loader)
    test_count = 1
    for i in range(epochs):
        precision_1 = 0
        precision_5 = 0
        precision_10 = 0
        precision_20 = 0
        for _, batch_data in enumerate(train_loader, 1):
            global_graph_model.train()
            global_dist_model.train()
            user_graph_model.train()
            user_history_model.train()
            optimizer.zero_grad()
            history_feature = batch_data[0].to(device)
            y = batch_data[1].to(device)
            user_graph = batch_data[2].to(device)
            user_graph_edges = batch_data[3].to(device)
            global_graph = global_graph.to(device)
            global_dist = global_dist.to(device)
            dist_mask = dist_mask.to(device)
            global_graph_feature = global_graph_model(global_graph)
            global_dist_feature = global_dist_model(global_dist, dist_mask)
            user_graph_feature = user_graph_model(user_graph, user_graph_edges)
            user_history_feature = user_history_model(history_feature)
            global_graph_feature = global_graph_feature.repeat(y.shape[0], 1)
            global_dist_feature = global_dist_feature.repeat(y.shape[0], 1)
            y_pred = global_user_model(user_history_feature, global_graph_feature, global_dist_feature, user_graph_feature)
            y_pred, indices = torch.sort(y_pred, dim=1, descending=True)
            precision_1 += cal_precision(indices, y, 1, train_len)
            precision_5 += cal_precision(indices, y, 5, train_len)
            precision_10 += cal_precision(indices, y, 10, train_len)
            precision_20 += cal_precision(indices, y, 20, train_len)
            loss = criterion(_, _)
            loss.backward()
            optimizer.step()
            sys.stdout.write("\rTRAINDATE:  Epoch:{}\t\t loss:{} res train:{}".format(i, loss.item(), train_len - _))
        test_model(global_graph_model, global_dist_model, user_graph_model, user_history_model, test_loader,
                   global_graph, global_dist)
        test_count += 1
        stepLR.step()


def test_model(global_graph_model, global_dist_model, user_graph_model, user_history_model, test_loader,
               global_graph, global_dist):
    model_path = './model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    global_graph_model.eval()
    global_dist_model.eval()
    user_graph_model.eval()
    user_history_model.eval()
    precision_1 = 0
    precision_5 = 0
    precision_10 = 0
    precision_20 = 0
    with torch.no_grad():
        for _, batch_data in enumerate(test_loader):
            x = batch_data[0]
            y = batch_data[1]
            user_graph = batch_data[2]
            user_graph_edges = batch_data[3]
            global_graph_feature = global_graph_model(global_graph)
            global_dist_feature = global_dist_model(global_dist)
            user_graph_feature = user_graph_model(user_graph, user_graph_edges)
            y_pred = user_history_model(x, global_graph_feature, global_dist_feature, user_graph_feature)
            y_pred, indices = torch.sort(y_pred, dim=1, descending=True)
            precision_1 += cal_precision(indices, y, 1, train_len)
            precision_5 += cal_precision(indices, y, 5, train_len)
            precision_10 += cal_precision(indices, y, 10, train_len)
            precision_20 += cal_precision(indices, y, 20, train_len)


def cal_precision(indices, batch_y, k, count):
    precision = 0
    for i in range(indices.size(0)):
        sort = indices[i]
        if batch_y[i].long() in sort[:k]:
            precision += 1
    return precision / count


if __name__ == '__main__':
    train_len = 0
    test_len = 0
    args = parser.parse_args()
    test_num = args.test_num
    test_key = 'test' + test_num
    #  global parameters
    seed = args.seed
    gpu_num = args.gpu_num
    torch.manual_seed(seed)
    device = torch.device(gpu_num)
    # model parameters
    global_graph_dim = args.global_graph_dim
    global_dist_dim = args.global_dist_dim
    user_graph_dim = args.user_graph_dim
    user_history_dim = args.user_history_dim
    node_len = args.node_len
    # train parameters
    lr = args.lr
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    epochs = args.epochs
    data_name = args.data_name
    print(args)
    train()
