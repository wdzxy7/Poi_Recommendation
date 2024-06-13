import os
import sys
import time
import torch
import pickle
import random
import logging
import argparse
import numpy as np
import torch.nn as nn
from utils import PoiDataset
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from Model import GlobalGraphNet, GlobalDistNet, UserGraphNet, UserHistoryNet, TransformerModel
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


parser = argparse.ArgumentParser(description='Parameters for my model')

parser.add_argument('--poi_len', type=int, default=7791, help='The length of POI_id,NYC is 5099, TKY is 61858')
parser.add_argument('--user_len', type=int, default=2124, help='The length of users')
parser.add_argument('--cat_len', type=int, default=286, help='The length of category')
parser.add_argument('--node_len', type=int, default=334, help='The length of user graph node(debug to see)')
parser.add_argument('--lat_len', type=int, default=7791, help='The length of gps')
parser.add_argument('--long_len', type=int, default=7791, help='The length of gps')

parser.add_argument('--cat_dim', type=int, default=200, help='The embedding dim of poi category')
parser.add_argument('--user_dim', type=int, default=150, help='The embedding dim of poi users')
parser.add_argument('--poi_dim', type=int, default=100, help='The embedding dim of pois')
parser.add_argument('--gps_dim', type=int, default=100, help='The embedding dim of gps')
parser.add_argument('--gcn_channel', type=int, default=128, help='The channels in GCN')

parser.add_argument('--neighbor_size', type=int, default=20, help='The size of neighbor')
parser.add_argument('--graph_out_dim', type=int, default=1024, help='The embedding dim of three graph Conv')
parser.add_argument('--global_graph_layers', type=int, default=5, help='The gcn layers in GlobalGraphNet')
parser.add_argument('--global_dist_features', type=int, default=2294, help='The feature sum of global distance graph(debug to see)')
parser.add_argument('--global_dist_layers', type=int, default=4, help='The gcn layers in GlobalDistNet')
parser.add_argument('--user_graph_layers', type=int, default=3, help='The gcn layers in UserGraphNet')
parser.add_argument('--embed_size_user', type=int, default=150, help='The embedding dim of embed_size_user in UserHistoryNet')  #150
parser.add_argument('--embed_size_poi', type=int, default=100, help='The embedding dim of embed_size_poi in UserHistoryNet')  #100
parser.add_argument('--embed_size_cat', type=int, default=50, help='The embedding dim of embed_size_cat in UserHistoryNet')  #200
parser.add_argument('--embed_size_hour', type=int, default=20, help='The embedding dim of embed_size_hour in UserHistoryNet')
parser.add_argument('--history_out_dim', type=int, default=1024, help='The embedding dim of GRU in UserHistoryNet')
parser.add_argument('--hidden_size', type=int, default=128, help='The hidden size in UserHistoryNet`s LSTM')  #128
parser.add_argument('--lstm_layers', type=int, default=2, help='The layer of LSTM model in UserHistoryNet')  #2
parser.add_argument('--hid_dim', type=int, default=128, help='The dim of previous four model')
parser.add_argument('--dropout', type=float, default=0.5, help='The dropout rate in Transformer')
parser.add_argument('--tran_head', type=int, default=4, help='The number of heads in Transformer')
parser.add_argument('--tran_hid', type=int, default=128, help='The dim in Transformer')
parser.add_argument('--tran_layers', type=int, default=3, help='The layer of Transformer')
parser.add_argument('--epochs', type=int, default=50, help='Epochs of train')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size of dataloader')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of optimizer')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight_decay of optimizer')
parser.add_argument('--lr_scheduler_factor', type=float, default=0.1, help='The decrease rate of ReduceLROnPlateau')
parser.add_argument('--data_name', type=str, default='TKY', help='Train data name')
parser.add_argument('--gpu_num', type=int, default=2, help='Choose which GPU to use')
parser.add_argument('--seed', type=int, default=5566, help='random seed')


def load_data():
    test_dataset = PoiDataset(data_name, data_type='test', robust_rate=-0.2, neighbor_size=neighbor_size)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    return test_loader


def load_global_graph():
    with open('./processed/{}/global_graph_data.pkl'.format(data_name), 'rb') as f:
        graph_data = pickle.load(f)
    with open('./processed/{}/global_dist_data.pkl'.format(data_name), 'rb') as f:
        dist_data = pickle.load(f)
    with open('./processed/{}/global_graph_weight_data.pkl'.format(data_name), 'rb') as f:
        graph_weight_data = pickle.load(f)
    with open('./processed/{}/global_dist_weight_data.pkl'.format(data_name), 'rb') as f:
        dist_weight_data = pickle.load(f)
    return graph_data, graph_weight_data.float(), dist_data, dist_weight_data.float()


def get_dist_mask(global_dist):
    mask = []
    for i in range(global_dist.x.shape[1] // 2):
        mask.append(True)
        mask.append(False)
    mask = torch.Tensor(mask).reshape(1, -1)
    mask = mask.repeat(global_dist.x.shape[0], 1)
    return mask.bool()


def test_model():
    global gg_time, gd_time, ug_time, uh_time, trans_time
    test_loader = load_data()
    with  open(os.path.join('./processed/{}/poi_data/poi_neighbor_{}.pkl'.format(data_name, neighbor_size)), 'rb') as f:
        dis_mess = pickle.load(f)
    global_graph, global_graph_weight, global_dist, global_dist_weight = load_global_graph()
    dist_mask = get_dist_mask(global_dist)
    global_graph_model = GlobalGraphNet(cat_len=cat_len + 1, poi_len=poi_len + 1, cat_dim=cat_dim, poi_dim=poi_dim,
                                        gps_dim=gps_dim,
                                        gcn_channel=gcn_channel, gcn_layers=global_graph_layers,
                                        graph_out_dim=graph_out_dim,
                                        lat_len=lat_len, long_len=long_len)
    global_dist_model = GlobalDistNet(poi_dim=poi_dim // 2, poi_len=poi_len + 1, graph_features=global_dist_features,
                                      gcn_layers=global_dist_layers, graph_out_dim=graph_out_dim)
    user_graph_model = UserGraphNet(cat_len=cat_len + 1, poi_len=poi_len + 1, node_len=node_len, cat_dim=cat_dim,
                                    poi_dim=poi_dim,
                                    gps_dim=gps_dim, gcn_channel=gcn_channel, gcn_layers=user_graph_layers,
                                    graph_out_dim=graph_out_dim,
                                    lat_len=lat_len, long_len=long_len)
    user_history_model = UserHistoryNet(cat_len=cat_len + 1, poi_len=poi_len + 1, user_len=user_len + 1,
                                        embed_size_user=embed_size_user,
                                        embed_size_poi=embed_size_poi, embed_size_cat=embed_size_cat,
                                        embed_size_hour=embed_size_hour,
                                        hidden_size=hid_dim, lstm_layers=lstm_layers, history_out_dim=poi_len + 1)
    transformer = TransformerModel(embed_dim=1024, dropout=dropout, tran_head=tran_head, tran_hid=tran_hid,
                                   tran_layers=tran_layers, poi_len=poi_len + 1)
    model_path = "./pretrain/"
    global_graph_model.load_state_dict(torch.load(model_path + 'global_graph_model_{}_parameter_test.pkl'.format(data_name)), False)
    global_dist_model.load_state_dict(torch.load(model_path + 'global_dist_model_{}_parameter_test.pkl'.format(data_name)), False)
    user_graph_model.load_state_dict(torch.load(model_path + 'user_graph_model_{}_parameter_test.pkl'.format(data_name)), False)
    user_history_model.load_state_dict(torch.load(model_path + 'user_history_model_{}_parameter_test.pkl'.format(data_name)), False)
    transformer.load_state_dict(torch.load(model_path + 'transformer_{}_parameter_test.pkl'.format(data_name)), False)

    global_graph_model.to(device)
    global_dist_model.to(device)
    user_graph_model.to(device)
    user_history_model.to(device)
    transformer.to(device)

    test_batches_top1_acc_list = []
    test_batches_top5_acc_list = []
    test_batches_top10_acc_list = []
    test_batches_top15_acc_list = []
    test_batches_top20_acc_list = []
    test_batches_top30_acc_list = []
    test_batches_mAP1_list = []
    test_batches_mAP5_list = []
    test_batches_mAP10_list = []
    test_batches_mAP15_list = []
    test_batches_mAP20_list = []
    test_batches_mrr_list = []
    loss_list = []

    with torch.no_grad():
        src_mask = transformer.generate_square_subsequent_mask(batch_size).to(device)
        for _, batch_data in enumerate(test_loader, 1):
            b_len = len(batch_data)
            if b_len != batch_size:
                src_mask = transformer.generate_square_subsequent_mask(b_len).to(device)
            history_feature, y, trajectory_len, user_graph, user_graph_edges, user_graph_weight, neighbor = spilt_batch(
                batch_data)
            y = y.to(device)
            history_feature = history_feature.to(device)
            user_graph = user_graph.to(device)
            user_graph_edges = user_graph_edges.to(device)
            global_graph = global_graph.to(device)
            global_graph_weight = global_graph_weight.to(device)
            global_dist = global_dist.to(device)
            global_dist_weight = global_dist_weight.to(device)
            user_graph_weight = user_graph_weight.to(device)
            neighbor = neighbor.to(device)
            t1 = time.time()
            global_graph_feature = global_graph_model(global_graph, global_graph_weight)
            t2 = time.time()
            global_dist_feature, mask_time = global_dist_model(global_dist, dist_mask, global_dist_weight)
            t3 = time.time()
            user_graph_feature = user_graph_model(user_graph, user_graph_edges, user_graph_weight)
            t4 = time.time()
            user_history_feature, embed_neighbor = user_history_model(history_feature, neighbor)
            t5 = time.time()
            global_graph_feature = global_graph_feature.repeat(b_len, user_history_feature.shape[1], 1)
            global_dist_feature = global_dist_feature.repeat(b_len, user_history_feature.shape[1], 1)
            user_graph_feature = user_graph_feature.reshape(b_len, 1, -1).repeat(1, user_history_feature.shape[1], 1)
            embed_neighbor = embed_neighbor.repeat(1, user_history_feature.shape[1], 1)
            t6 = time.time()
            y_pred = transformer(user_history_feature, global_graph_feature, global_dist_feature, user_graph_feature,
                                 src_mask, embed_neighbor)
            t7 = time.time()
            gg_time.append(t2 - t1)
            gd_time.append(t3 - t2)
            ug_time.append(t4 - t3)
            uh_time.append(t5 - t4)
            m_time.append(mask_time)
            trans_time.append(t7 - t6)
            precision_1 = 0
            precision_5 = 0
            precision_10 = 0
            precision_15 = 0
            precision_20 = 0
            precision_30 = 0
            mAP1 = 0
            mAP5 = 0
            mAP10 = 0
            mAP15 = 0
            mAP20 = 0
            mrr = 0
            y_pred = y_pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            for predict, true, tra_len, history in zip(y_pred, y, trajectory_len, history_feature):
                true = true[: tra_len]
                predict = predict[: tra_len, :]
                # user poi cat hour week
                last_trajectory = history[-1: tra_len, :]
                poi_neighbor = neighbor.detach().cpu().numpy()
                precision_1 += top_k_acc_last_timestep(true, predict, k=1)
                precision_5 += top_k_acc_last_timestep(true, predict, k=5)
                precision_10 += top_k_acc_last_timestep(true, predict, k=10)
                precision_15 += top_k_acc_last_timestep(true, predict, k=15)
                precision_20 += top_k_acc_last_timestep(true, predict, k=20)
                precision_30 += top_k_acc_last_timestep(true, predict, k=30)
                mAP1 += mAP_metric_last_timestep(true, predict, k=1)
                mAP5 += mAP_metric_last_timestep(true, predict, k=5)
                mAP10 += mAP_metric_last_timestep(true, predict, k=10)
                mAP15 += mAP_metric_last_timestep(true, predict, k=15)
                mAP20 += mAP_metric_last_timestep(true, predict, k=20)
                mrr += MRR_metric_last_timestep(true, predict)
            test_batches_top1_acc_list.append(precision_1 / y.shape[0])
            test_batches_top5_acc_list.append(precision_5 / y.shape[0])
            test_batches_top10_acc_list.append(precision_10 / y.shape[0])
            test_batches_top15_acc_list.append(precision_15 / y.shape[0])
            test_batches_top20_acc_list.append(precision_20 / y.shape[0])
            test_batches_top30_acc_list.append(precision_30 / y.shape[0])
            test_batches_mAP1_list.append(mAP1 / y.shape[0])
            test_batches_mAP5_list.append(mAP5 / y.shape[0])
            test_batches_mAP10_list.append(mAP10 / y.shape[0])
            test_batches_mAP15_list.append(mAP15 / y.shape[0])
            test_batches_mAP20_list.append(mAP20 / y.shape[0])
            test_batches_mrr_list.append(mrr / y.shape[0])
    mess = (
        "\rTESTING: Epoch:{}\t\t acc30:{} \t\t  precision_1:{}\t\t precision_5:{}\t\t precision_10:{} \t\t precision_15:{} \t\t precision_20:{} "
        "\t\t mAP1:{} \t\t mAP5:{} \t\t mAP10:{} \t\t mAP15:{} \t\t mAP20:{} \t\t mrr:{}.".
        format(1, np.mean(test_batches_top30_acc_list), np.mean(test_batches_top1_acc_list), np.mean(test_batches_top5_acc_list)
               , np.mean(test_batches_top10_acc_list), np.mean(test_batches_top15_acc_list),
               np.mean(test_batches_top20_acc_list), np.mean(test_batches_mAP1_list), np.mean(test_batches_mAP5_list),
               np.mean(test_batches_mAP10_list), np.mean(test_batches_mAP15_list), np.mean(test_batches_mAP20_list),
               np.mean(test_batches_mrr_list)))
    print(mess)

    return None


def top_k_acc_last_timestep(y_true_seq, y_pred_seq, k):
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    top_k_rec = y_pred.argsort()[-k:][::-1]
    idx = np.where(top_k_rec == y_true)[0]
    if len(idx) != 0:
        return 1
    else:
        return 0


def mAP_metric_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-k:][::-1]
    r_idx = np.where(rec_list == y_true)[0]
    if len(r_idx) != 0:
        return 1 / (r_idx[0] + 1)
    else:
        return 0


def MRR_metric_last_timestep(y_true_seq, y_pred_seq):
    """ next poi metrics """
    # Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-len(y_pred):][::-1]
    r_idx = np.where(rec_list == y_true)[0][0]
    return 1 / (r_idx + 1)


def spilt_batch(batch):
    history_feature, y, trajectory_len, user_graph, user_graph_edges, user_graph_weight, neighbor = [], [], [], [], [], [], []
    for i in batch:
        history_feature.append(i[0])
        y.append(i[1])
        trajectory_len.append(i[2])
        user_graph.append(i[3])
        user_graph_edges.append(i[4])
        user_graph_weight.append(i[5])
        neighbor.append(i[6])
    history_feature = pad_sequence(history_feature, batch_first=True, padding_value=0)
    y = pad_sequence(y, batch_first=True, padding_value=0)
    user_graph_weight = pad_sequence(user_graph_weight, batch_first=True, padding_value=0)
    return history_feature, y, trajectory_len, torch.stack(user_graph), torch.stack(user_graph_edges), user_graph_weight, torch.stack(neighbor)


if __name__ == '__main__':
    train_len = 0
    test_len = 0
    args = parser.parse_args()
    test_num = args.gpu_num
    test_key = 'test' + str(test_num)
    #  global parameters
    seed = args.seed
    gpu_num = args.gpu_num
    torch.manual_seed(seed)
    device = torch.device(gpu_num)
    # model parameters
    # Share
    cat_len = args.cat_len
    node_len = args.node_len
    poi_len = args.poi_len
    user_len = args.user_len
    cat_dim = args.cat_dim
    poi_dim = args.poi_dim
    user_dim = args.user_dim
    gcn_channel = args.gcn_channel
    graph_out_dim = args.graph_out_dim
    history_out_dim = args.history_out_dim
    gps_dim = args.gps_dim
    lat_len = args.lat_len
    long_len = args.long_len
    neighbor_size = args.neighbor_size
    # GlobalGraphNet
    global_graph_layers = args.global_graph_layers
    # GlobalDistNet
    global_dist_features = args.global_dist_features
    global_dist_layers = args.global_dist_layers
    # UserGraphNet
    user_graph_layers = args.user_graph_layers
    # UserHistoryNet
    embed_size_user = args.embed_size_user
    embed_size_poi = args.embed_size_poi
    embed_size_cat = args.embed_size_cat
    embed_size_hour = args.embed_size_hour
    hidden_size = args.hidden_size
    lstm_layers = args.lstm_layers
    hid_dim = args.hid_dim
    # Transformer
    dropout = args.dropout
    tran_head = args.tran_head
    tran_hid = args.tran_hid
    tran_layers = args.tran_layers
    # train parameters
    lr = args.lr
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    lr_scheduler_factor = args.lr_scheduler_factor
    epochs = args.epochs
    data_name = args.data_name
    if args.data_name == 'CA':
        embed_size_cat = 0
        cat_dim = 0
        cat_len = 0
    print(args)
    # ----------------------------------------------------------------------------- #
    gg_time = []
    gd_time = []
    ug_time = []
    uh_time = []
    trans_time = []
    m_time = []
    for i in range(10):
        test_model()
    print('time cost: \t\t global_graph:{} \t\t global_distance_graph:{} \t\t user_graph:{} \t\t history_graph:{} '
          '\t\t transformer_graph:{}\t\t mask_time:{}'.format(np.mean(gg_time), np.mean(gd_time), np.mean(ug_time), np.mean(uh_time),
                                             np.mean(trans_time), np.mean(m_time)))