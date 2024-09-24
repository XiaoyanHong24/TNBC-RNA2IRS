from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from model import *
from evaluation import eva
from collections import Counter
from pretrain import LoadDataset
import os
from utils import *
from louvian_test import *
from collections import Counter
import torch
from torch.nn.parameter import Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# torch.cuda.set_device(1)


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class Model(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1):
        super(Model, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(
            args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        sigma = 0.5

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)
        # h = h-torch.max(h,dim=1).values.unsqueeze(dim=1)
        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) -
                                             self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z


class Model_GAT(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1, dropout=0.6, alpha=0.2):
        super(Model_GAT, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        # Assuming args.pretrain_path is defined elsewhere
        # self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GAT for inter information
        self.gnn_1 = GATLayer(n_input, n_enc_1, dropout, alpha)
        self.gnn_2 = GATLayer(n_enc_1, n_enc_2, dropout, alpha)
        self.gnn_3 = GATLayer(n_enc_2, n_enc_3, dropout, alpha)
        self.gnn_4 = GATLayer(n_enc_3, n_z, dropout, alpha)
        self.gnn_5 = GATLayer(n_z, n_clusters, dropout, alpha, concat=False)  # concat=False if not multi-head

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        sigma = 0.5

        # GAT Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z,
                       adj)  # Note: Removed active=False as it depends on GATLayer implementation
        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) -
                                             self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z


class Model_GAT_sparse(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1, dropout=0.6, alpha=0.2):
        super(Model_GAT_sparse, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        # Assuming args.pretrain_path is defined elsewhere
        # self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GAT for inter information
        self.gnn_1 = GATLayer_sparse(n_input, n_enc_1, alpha)
        self.gnn_2 = GATLayer_sparse(n_enc_1, n_enc_2, alpha)
        self.gnn_3 = GATLayer_sparse(n_enc_2, n_enc_3, alpha)
        self.gnn_4 = GATLayer_sparse(n_enc_3, n_z, alpha)
        self.gnn_5 = GATLayer_sparse(n_z, n_clusters, alpha, concat=False)  # concat=False if not multi-head

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # Ensure adj is a sparse tensor for GAT processing
        if not adj.is_sparse:
            raise ValueError("adj must be a sparse tensor.")
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        sigma = 0.5

        # GAT Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z,
                       adj)  # Note: Removed active=False as it depends on GATLayer implementation
        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) -
                                             self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def save_pred_cluster_umap(pred):
    method = spa_km_cluster_center(args.name, n_cluster=args.n_clusters)
    pred_idx = torch.argmax(pred, dim=1).data.cpu().numpy()
    method.adata.obs["pred"] = pred_idx
    method.adata.obs["pred"] = method.adata.obs["pred"].astype('category')
    method.adata.write_h5ad(
        "/".join(list((args.res_path, args.res_name + "_res.h5ad"))))


def save_pred(pred):
    pred_ = pred.data.cpu().numpy()
    f = args.cluster_res_savepath
    # with open(f,"r+") as file:
    with open(f, "w+") as file:  # w覆盖a追加
        file.truncate()
    for i in range(0, pred_.shape[0]):
        # print(pred[i]) 这里是取pre中概率更大的那个作为预测的结果
        maxIndex = 0
        max = pred_[i][0]
        for j in range(1, pred_.shape[1]):
            if pred_[i][j] > max:
                max = pred_[i][j]
                maxIndex = j
        with open(f, "a") as file:
            file.write(str(maxIndex) + "\n")

    np.savetxt(args.predres_save_path, pred_)
    print("Pred write finished in ", args.predres_save_path)


def train_model(dataset):
    model = Model_GAT_sparse(500, 500, 2000, 2000, 500, 500,
                            n_input=args.n_input,
                            n_z=args.n_z,
                            n_clusters=args.n_clusters,
                            v=1).to(device)
    # print(model)

    # optimizer = Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    iters_num = args.model_epoch_num // 3000 if args.model_epoch_num >= 1000 else 2
    step = args.model_epoch_num // iters_num
    lr_step_ls = list(range(step, args.model_epoch_num, step))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_step_ls, gamma=0.95)

    adj = load_graph(args.name, args.k, args.graph_method)
    adj = adj.to(device)

    # cluster parameter initiate
    input_data = torch.Tensor(dataset.x).to(device)
    # y = dataset.y
    with torch.no_grad():
        # _, _, _, _, z = model.ae(data)
        res = model.ae(input_data)
        z = res[-1]

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=200)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    # y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    all_loss = []
    temp_loss = 9e10
    for epoch in range(args.model_epoch_num):
        if epoch % 1 == 0:
            # update_interval
            _, tmp_q, pred, _ = model(input_data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

        x_bar, q, pred, _ = model(input_data, adj)

        pred = torch.clamp(pred, min=1e-10, max=1 - (1e-10))

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')

        re_loss = F.mse_loss(x_bar, input_data)
        loss = 0.5 * kl_loss + 0.5 * ce_loss + re_loss  # 原始结果存储的loss算值
        print('model {} loss: {},lr {}'.format(epoch, loss, optimizer.state_dict()['param_groups'][0]['lr']))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # 加一个保存最新和最优以及对应的loss
        torch.save(model.state_dict(), args.model_path + "_latest.pkl")

        if float(loss) < temp_loss and epoch >= args.model_epoch_num // 3:
            torch.save(model.state_dict(), args.model_path + "_best.pkl")
            temp_loss = loss

        if (epoch + 1) % args.model_epoch_num == 0:
            save_pred(pred)
            # save_pred_cluster_umap(pred)
        all_loss.append(float(loss))
    return all_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='reut')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--graph_method', type=str, default='ncos_no_one')
    parser.add_argument('--model_epoch_num', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n_clusters', default=50, type=int)
    parser.add_argument('--n_z', default=100, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    parser.add_argument('--model_path', type=str, default='pkl')
    parser.add_argument('--cluster_res_savepath', type=str, default='txt')
    parser.add_argument('--predres_save_path', type=str, default='txt')
    parser.add_argument('--cluster_pic_save_path', type=str, default='res')
    parser.add_argument('--res_name', type=str, default='res')
    parser.add_argument('--res_path', type=str, default='res')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    res_dir = "res"

    data_list = ['CAMprep']
    args.model_epoch_num = 50000
    k_range_list = range(1, 15)
    graph_method_ls = ['ncos_no_one', 'heat', 'cos', 'ncos']

    for graph_method in graph_method_ls:
        args.graph_method = graph_method
        for data_type in data_list:
            args.name = data_type
            for k in k_range_list:
                args.k = k
                data_name, data_file_name = got_data_name_from_data_class(
                    args.name)

                args.pretrain_path = "/".join(
                    list((res_dir, data_name + "_res", data_name + "_pre.pkl")))

                args.res_name = "model_{}_cluster{}_graph{}_method_{}".format(
                    data_name, args.n_clusters, args.k, args.graph_method)
                args.res_path = "/".join(list((res_dir, data_name + "_res")))
                args.model_path = "/".join(list((args.res_path, args.res_name)))
                args.cluster_res_savepath = "/".join(
                    list((args.res_path, args.res_name + "_cluster_res.txt")))
                args.predres_save_path = "/".join(
                    list((args.res_path, args.res_name + "_pred_res.txt")))
                args.cluster_pic_save_path = "/".join(
                    list((res_dir, data_name + "_image")))
                data, _, _ = load_data(data_file_name)[1]

                print("load data from {}".format(data_file_name))
                if data.shape[0] < data.shape[1]:
                    data = np.transpose(data)

                args.n_input = min(data.shape)

                dataset = LoadDataset(data)

                loss_list = train_model(dataset)
                loss_fun = loss_save_and_draw(
                    loss_list=loss_list, save_name=args.res_name + "_loss", data_class=data_name)
                loss_fun.save_and_draw()
