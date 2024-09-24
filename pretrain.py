import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation import eva
from utils import *
import os

# torch.cuda.set_device(3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        return x_bar, z


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
            torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretrain_ae(model, dataset, data_name, AePrenum, res_dir="res"):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    # print(model)
    features_save_path = "/".join(list((res_dir, data_name + "_res")))
    if not os.path.exists(features_save_path):
        os.makedirs(features_save_path)
    features_save_path = "/".join(list((features_save_path, data_name + "_pre.pkl")))
    image_features_save_path = "/".join(list((res_dir, data_name + "_image")))
    if not os.path.exists(image_features_save_path):
        os.makedirs(image_features_save_path)
    if os.path.exists(features_save_path):
        model.load_state_dict(torch.load(features_save_path, map_location=device))
    print("PreRes save in {},preloss img save in {}".format(features_save_path, image_features_save_path))

    optimizer = Adam(model.parameters(), lr=5e-4)

    pre_loss = []
    loss_min = float("inf")
    for epoch in range(AePrenum):
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()

            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('ae {} loss: {}'.format(epoch, loss))
            pre_loss.append(float(loss))
            # kmeans = KMeans(n_clusters=4, n_init=20).fit(z.data.cpu().numpy())
            # eva(y, kmeans.labels_, epoch)
        if loss < loss_min:
            torch.save(model.state_dict(), features_save_path)
            loss_min = loss
    return pre_loss


if __name__ == '__main__':
    data_list = ['CAMprep', 'CAMprep_01_loss']
    res_dir = "res"
    pretrain_epochs = 5000

    for data_type in data_list:
        data_name, data_file_name = got_data_name_from_data_class(data_type)

        data, _, _ = load_data(data_file_name)[1]

        print("load data from {},data shape is {}".format(data_file_name, data.shape))
        if data.shape[0] < data.shape[1]:
            data = np.transpose(data)

        input_size = min(data.shape)
        model = AE(n_enc_1=500, n_enc_2=500, n_enc_3=2000,
                   n_dec_1=2000, n_dec_2=500, n_dec_3=500,
                   n_input=input_size, n_z=100).cuda()
        dataset = LoadDataset(data)

        loss_list = pretrain_ae(model, dataset, data_name, AePrenum=pretrain_epochs, res_dir=res_dir)
        loss_fun = loss_save_and_draw(loss_list=loss_list, save_name="{}_pre_loss".format(data_name),
                                      data_class=data_name)
        loss_fun.save_and_draw()
