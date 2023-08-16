import os
import random
import sys
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from utils import PoiDataset
import torch.utils.data as data
from Model import GlobalGraphNet, GlobalDistNet, UserGraphNet, UserHistoryNet


parser = argparse.ArgumentParser(description='Parameters for my model')
parser.add_argument('--epochs', type=int, default=150, help='Epochs of train')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size of dataloader')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of optimizer')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight_decay of optimizer')
parser.add_argument('--data_name', type=str, default='NYC', help='Train data name')
parser.add_argument('--gpu_num', type=int, default=0, help='Choose which GPU to use')
parser.add_argument('--test_num', type=str, default='1', help='Just for test')
parser.add_argument('--seed', type=int, default=666, help='random seed')


def load_data():
    train_dataset = PoiDataset(data_name, data_type='train')
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = PoiDataset(data_name, data_type='test')
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train():
    train_loader, test_loader = load_data()
    global_graph_model = GlobalGraphNet()
    global_dist_model = GlobalDistNet()
    user_graph_model = UserGraphNet()
    user_history_model = UserHistoryNet()
    global_graph_model.to(device)
    global_dist_model.to(device)
    user_graph_model.to(device)
    user_history_model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(list(global_graph_model.parameters()) +
                                 list(global_dist_model.parameters()) +
                                 list(user_graph_model.parameters()) +
                                 list(user_history_model.parameters())
                                 , lr=lr, weight_decay=weight_decay)
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    train_len = len(train_loader)
    test_count = 1
    for i in range(epochs):
        for _, batch_data in enumerate(train_loader, 1):
            global_graph_model.train()
            global_dist_model.train()
            user_graph_model.train()
            user_history_model.train()
            optimizer.zero_grad()
            x = batch_data[0]
            y = batch_data[1]
            graph_feature = batch_data[2]
            graph_edges = batch_data[3]

            loss = criterion(_, _)
            loss.backward()
            optimizer.step()
            sys.stdout.write("\rTRAINDATE:  Epoch:{}\t\t loss:{} res train:{}".format(i, loss.item(), train_len - _))
        test_model()
        test_count += 1
        stepLR.step()


def test_model():
    pass


def cal_rmse():
    pass


if __name__ == '__main__':
    args = parser.parse_args()
    test_num = args.test_num
    test_key = 'test' + test_num
    #  global parameters
    seed = args.seed
    gpu_num = args.gpu_num
    torch.manual_seed(seed)
    device = torch.device(gpu_num)
    # model parameters
    # train parameters
    lr = args.lr
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    epochs = args.epochs
    data_name = args.data_name

    print(args)
    train()
