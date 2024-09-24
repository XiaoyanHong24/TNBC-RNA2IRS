import shutil
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import torch
from torch.utils.data import Dataset
import os
import anndata as ad
from string import ascii_uppercase
from anndata import AnnData
# import SpaGCN as spg
import scanpy as sc
from collections import Counter
from sklearn.metrics import silhouette_score
from matplotlib.pyplot import MultipleLocator
from sklearn.cluster import KMeans
from sklearn.metrics import *
from sklearn import metrics

dataset_dir = "../SetData/"


def csv_to_npy(data_name):
    data_path = dataset_dir + 'InitData/{}'.format(data_name)
    if data_name[-3:] == "csv":
        csv_data = pd.read_csv(data_path)
        data_name = data_name[:-4] + "_npy.npy"
        np.save(data_path, csv_data, allow_pickle=True)
        print("Succeed convert csv file to npy in ", data_path)
    else:
        print("npy file exit!")
    return data_name


def rename_npy(data_name):
    data_path = dataset_dir + 'InitData/{}'.format(data_name)
    if os.path.exists(data_path[:-4] + "_npy.npy"):
        data_name = data_name[:-4] + "_npy.npy"
    elif data_name[-3:] == "csv":
        data_name = csv_to_npy(data_name)
    return data_name


def load_graph(dataset, k, method):
    data_name, data_file_name = got_data_name_from_data_class(dataset)
    if k:
        path = 'graph/{}/{}{}_graph.txt'.format(method, dataset, k)
    else:
        path = 'graph/{}/{}_graph.txt'.format(method, dataset)
    print("load graph from {}".format(path))
    # data = np.loadtxt('data/{}.txt'.format(dataset))
    # dataset = load_data(rename_npy(data_file_name))
    data, _, _ = load_data(data_file_name)[1]
    # data = dataset.x[:, 1:].astype(float)
    if data.shape[0] < data.shape[1]:
        data = np.transpose(data)

    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# class load_data(Dataset):
#     def __init__(self, dataset):
#         # self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
#         # self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)
#         self.x = np.load('data/InitData/{}'.format(dataset),allow_pickle=True)
#
#
#     def __len__(self):
#         return self.x.shape[0]
#
#     def __getitem__(self, idx):
#         return torch.from_numpy(np.array(self.x[idx])),\
#                torch.from_numpy(np.array(idx))

def read_txt_every_line(txt_path):
    txt_tables = []
    f = open(txt_path, "r", encoding='utf-8')
    line = f.readline()  # 读取第一行
    while line == '\n':
        line = f.readline()  # 读取下一行
    while line != '\n' and line != '':
        # txt_data = float(line[:-1])  # 可将字符串变为元组
        txt_data = float(line)  # 可将字符串变为元组
        txt_tables.append(txt_data)  # 列表增加
        line = f.readline()  # 读取下一行
    # print(txt_tables)
    return txt_tables


class loss_save_and_draw:
    def __init__(self, loss_list, save_name, data_class, file_type="a+"):
        self.loss_list = loss_list
        self.save_name = save_name
        self.data_class = data_class
        self.res_dir = data_class + "_res"
        self.img_dir = "/".join(list(("res", data_class + "_image")))
        self.save_path = "/".join(list(("res",
                                        self.res_dir, save_name + ".txt")))
        self.file_type = file_type
        print("Res save in {},img save in {}".format(
            self.save_path, self.img_dir))

    def save_loss(self):
        # new_loss_list = pd.DataFrame(columns=["train_loss"], data=all_loss)
        # new_loss_list.to_csv(args.loss_res_savepath, encoding='gbk')
        if len(self.loss_list) != 0:
            str_n = '\n'
            f = open(self.save_path, self.file_type)
            f.write(str_n)
            if type(self.loss_list[-1]) == float:
                f.write(str_n.join('%s' % index for index in self.loss_list))
            else:
                f.write(str_n.join('%s' % float(index.data)
                                   for index in self.loss_list))
            f.close()
            print("Succeed save loss in ", self.save_path)
        else:
            print("Error:we can't save empty loss!")

    @staticmethod
    def draw_loss_curve_from_list(loss_list, pic_name, pic_path):
        # epochs = range(1, len(loss_list) + 1)
        epochs = range(len(loss_list))
        if not os.path.exists(pic_path):
            os.makedirs(pic_path)
        pic_path = pic_path + '/' + pic_name + '.png'
        plt.title(pic_name)
        plt.plot(epochs, loss_list, 'b-', label=pic_name, linewidth=0.3)
        # plt.legend()
        # note min point
        loss_min = np.argmin(loss_list[len(loss_list) // 10:]) + len(loss_list) // 10
        show_min = '[' + str(float(f'{loss_list[loss_min]:.8f}')) + ']'
        plt.plot(epochs[loss_min], loss_list[loss_min], 'g.')
        plt.annotate(show_min, xy=(loss_min, loss_list[loss_min]), xytext=(loss_min, loss_list[loss_min]),
                     bbox=dict(boxstyle='round,pad=0.3', fc='green', ec='k', lw=1, alpha=0.5))

        plt.xlabel("epoch")
        plt.ylabel("loss")
        # ax = plt.gca()
        # y_major_locator = MultipleLocator(0.5)
        # ax.yaxis.set_major_locator(y_major_locator)
        plt.savefig(pic_path)
        print("Succeed save pic in", pic_path)
        # plt.show()
        plt.clf()

    def draw_loss(self):
        if os.path.exists(self.save_path):
            loss_list = read_txt_every_line(self.save_path)
            loss_save_and_draw.draw_loss_curve_from_list(
                loss_list, self.save_name, self.img_dir)
        else:
            print("Error:we can't find txt file in ", self.save_path)

    def save_and_draw(self):
        # 保存传入的loss list
        loss_save_and_draw.save_loss(self)
        # 画图
        loss_save_and_draw.draw_loss(self)


def get_cluster_centers_by_pred(pred_list, data_np):
    if data_np.device != 'cpu':
        data_np = data_np.cpu()
    features = pd.DataFrame(data_np.detach().numpy(),
                            index=np.arange(0, data_np.shape[0]))
    Group = pd.Series(pred_list, index=np.arange(
        0, features.shape[0]), name="Group")
    Mergefeature = pd.concat([features, Group], axis=1)
    cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
    return cluster_centers


def got_data_name_from_data_class(data_type):
    data_name_dict = {'CAMprep': 'CAM_prep_before_cluster', 'CAMprep_01_loss': 'CAM_prep_before_cluster_loss_01'}
    data_path_dict = {'CAM_prep_before_cluster': 'exp_data_nolog_after_CAMclusterPrep.csv',
                      'CAM_prep_before_cluster_loss_01': 'exp_data_nolog_after_CAMclusterPrep.csv'}
    if data_name_dict.__contains__(data_type) and data_path_dict.__contains__(data_name_dict[data_type]):
        data_name = data_name_dict[data_type]
        data_file_name = data_path_dict[data_name]
    else:
        raise Exception('data type should in {}'.format(data_name_dict.keys()))
    return data_name, data_file_name


class load_data(Dataset):
    def __init__(self, data_file_name, data_dir='ProceedData'):
        self.data_name = data_file_name
        self.data_path = dataset_dir + '{}/{}'.format(data_dir, self.data_name)
        self.npy_data_name = self.data_name[:-4] + "_npy.npy"
        self.npy_data_path = dataset_dir + '{}/{}'.format(data_dir, self.npy_data_name)

    def __len__(self):
        return self.x.shape[0]

    def is_npy_exit(self):
        return os.path.exists(self.npy_data_path)

    def get_name(self):
        csv_data = pd.read_csv(self.data_path, header=None, low_memory=False)
        gene_list = csv_data.iloc[1:, 0].to_list()
        sample_list = csv_data.iloc[0, 1:].to_list()
        return gene_list, sample_list

    def csv2npy(self):
        # 输入的是csv的文件名字
        csv_data = pd.read_csv(self.data_path, header=None)
        np.save(self.npy_data_path, csv_data, allow_pickle=True)
        print("Succeed convert csv file to npy in ", self.npy_data_path)

    def load_npy_data(self):
        # 如果输入的文件名的npy版本存在 ,直接返回其npy文件名字,否则保存npy版本并返回新的npy文件名字
        if not load_data.is_npy_exit(self):
            load_data.csv2npy(self)
        npy_data = np.load(self.npy_data_path, allow_pickle=True)
        return npy_data

    def __getitem__(self, idx):
        self.x = load_data.load_npy_data(self)
        gene_list, sample_list = load_data.get_name(self)
        # return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(np.array(idx))
        return self.x[idx:, idx:].astype(float), gene_list, sample_list


def add_ann_name(data, gene_list, sample_list):
    data.var.index = sample_list
    data.obs.index = gene_list
    return data


def load_calculate_ann_data(init_data_name, init_data, gene_list, sample_list, data_dir='InitData'):
    ann_data_path = dataset_dir + "{}/{}".format(data_dir, init_data_name[:-4] + ".h5ad")
    if not os.path.exists(ann_data_path):
        # data_ann = AnnData(temp, gene_list, sample_list)
        data_ann = AnnData(init_data)
        data_ann = add_ann_name(data_ann, gene_list, sample_list)
        data_ann.write_h5ad(ann_data_path)
    else:
        data_ann = sc.read(ann_data_path)
    return data_ann


class CalculateScore:
    def __init__(self, data_type, data_file_name):
        self.data_type = data_type
        self.data_file_name = data_file_name
        self.res_dir, _ = got_data_name_from_data_class(data_type)
        self.res_path = os.path.join("res", self.res_dir + "_res")
        self.data_path = os.path.join(self.res_path, data_file_name)
        self.n_cluster = int(data_file_name.replace('model_' + self.res_dir + '_cluster', '').split(sep="_")[0])

        res_data = sc.read_h5ad(self.data_path)
        self.data = res_data.X
        self.cluster_clabel = list(map(int, list(res_data.obs['pred'].values)))

    def got_km_res(self):
        kmeans = KMeans(n_clusters=self.n_cluster, n_init=20)
        y_pred = kmeans.fit_predict(self.data)
        return y_pred

    def km_score(self):
        km_pred = CalculateScore.got_km_res(self)
        # print("KM res in data {} distribution is {}".format(self.data_type,km_pred))
        res = [np.mean(silhouette_samples(self.data, km_pred)), metrics.calinski_harabasz_score(self.data, km_pred),
               metrics.davies_bouldin_score(self.data, km_pred)]
        return res

    def sc_score(self):
        return np.mean(silhouette_samples(self.data, self.cluster_clabel))

    def ch_score(self):
        return metrics.calinski_harabasz_score(self.data, self.cluster_clabel)

    def db_score(self):
        return metrics.davies_bouldin_score(self.data, self.cluster_clabel)

    def contrast_km_gcn(self):
        km_res = CalculateScore.km_score(self)
        gcn_res = [CalculateScore.sc_score(self), CalculateScore.ch_score(self), CalculateScore.db_score(self)]
        print("In res file got km and gcn score is \nkm res {}\ngcn res {}".format(km_res, gcn_res))
        return km_res, gcn_res


class load_cluster_res:
    def __init__(self, data_type, res_dir_path="res"):
        self.data_name, self.data_file_name = got_data_name_from_data_class(data_type)
        self.res_dir_path = os.path.join(res_dir_path, self.data_name + "_res")

    def got_cluster_list(self, res_name):
        res_path = "/".join(list((self.res_dir_path, res_name)))
        adata_cluster_res = read_txt_every_line(res_path)
        print(Counter(adata_cluster_res))
        return adata_cluster_res

    def got_cluster_df(self, res_name):
        data_gene_path = "/".join(list(('data', "InitData", self.data_file_name)))
        data_gene = pd.read_csv(data_gene_path).iloc[:, 0]
        cluster_ls = {'gene_name': data_gene,
                      'gene_class': load_cluster_res.got_cluster_list(self, res_name)}
        cluster_df = pd.DataFrame(cluster_ls)
        return cluster_df


def load_right_res(res_num_ls, data_type="ftl"):
    res_dir_name, _ = got_data_name_from_data_class(data_type)
    all_res_file = list(map(lambda x: "model_{}_cluster{}_graph{}".format(res_dir_name, x[0], x[1]), res_num_ls))

    km_res_list = []
    gcn_res_list = []
    for res_name in all_res_file:
        res_name += "_res.h5ad"
        km_t, gcn_t = CalculateScore(data_type, res_name).contrast_km_gcn()
        km_res_list.append(km_t)
        gcn_res_list.append(gcn_t)

    return km_res_list, gcn_res_list


def draw_score_pc(right_num_tuple, data_type="ftl", pic_name="all_score_pic", km_flag=True):
    # km作为标准线，其他的按照类型并标注对应的值
    right_num_tuple.sort(key=lambda x: x[0])
    km_res_list, gcn_res_list = load_right_res(right_num_tuple)

    ss_score = list(map(lambda x: x[0], gcn_res_list))
    ch_score = list(map(lambda x: x[1], gcn_res_list))
    db_score = list(map(lambda x: x[2], gcn_res_list))

    km_ss_score = list(map(lambda x: x[0], km_res_list))
    km_ch_score = list(map(lambda x: x[1], km_res_list))
    km_db_score = list(map(lambda x: x[2], km_res_list))

    res_dict = {"Silhouette_Coefficient_max": ss_score,
                "Calinski_Harabaz_max": ch_score, "Davies_Bouldin_min": db_score}

    len_ls = list(range(len(ss_score)))

    # plt.figure(figsize=(450, 100))
    plt.figure()
    for idx, v in enumerate(res_dict.items()):
        data_name, data_ls = v[0], v[1]
        # plt.subplot(1, 3, idx + 1)
        plt.plot(len_ls, data_ls, label=right_num_tuple, marker='o')
        if km_flag:
            plt.plot(len_ls, list(map(lambda x: x[idx], km_res_list)), label=right_num_tuple, marker='o', linestyle=":")
            plt.legend(['model_res', 'km_res'])
        if data_name.endswith("min"):
            plt.plot(np.argmin(data_ls), data_ls[np.argmin(data_ls)], marker='*')
        else:
            plt.plot(np.argmax(data_ls), data_ls[np.argmax(data_ls)], marker='*')
        plt.ylabel(str(data_name), fontsize=16)
        plt.xlabel('data type', fontsize=16)
        for i in len_ls:
            plt.text(len_ls[i], data_ls[i], str(right_num_tuple[i]), ha='center', va='bottom', fontsize=10)
        # 保存数据
        # plt.gcf()
        res_dir_name, _ = got_data_name_from_data_class(data_type)
        img_path = os.path.join("res", res_dir_name + "_image", pic_name + data_name + ".png")
        print("Save all right res score image in {}".format(img_path))
        plt.savefig(img_path, dpi=500, bbox_inches='tight')
        plt.close()
        plt.show()
        plt.clf()
        plt.close()


def save_right_data_to_csv(right_num, data_type):
    res_dir_name, _ = got_data_name_from_data_class(data_type)
    data_name = "model_{}_cluster{}_graph{}".format(res_dir_name, right_num[0], right_num[1])
    res_path = os.path.join("res", res_dir_name + "_res", data_name + "_res.h5ad")
    if os.path.exists(res_path):
        res_data = sc.read(res_path)
    else:
        raise Exception("No file in {}".format(res_path))


def got_all_num_tuple_list(cluster_list, graph_num_list):
    all_tuple = []
    for cluster_num in cluster_list:
        for graph_num in graph_num_list:
            all_tuple.append((cluster_num, graph_num))
    return all_tuple


class CopyRes:
    def __init__(self, copy_res_dir, aim_dir, data_type):
        self.copy_res_dir = copy_res_dir
        self.aim_dir = aim_dir
        self.data_type = data_type
        self.data_name, _ = got_data_name_from_data_class(data_type)
        self.res_path = "/".join(list((self.copy_res_dir, self.data_name + "_res")))
        self.aim_path = "/".join(list((self.copy_res_dir, self.aim_dir, self.data_name + "_res")))
        self.type_list = {"pred_res": ".txt", "loss": ".txt", "model": ".pkl", "res": ".h5ad",
                          "cluster_res": ".txt", "combine_gene_symbol": ".csv"}
        if not os.path.exists(self.aim_path): os.makedirs(self.aim_path)
        # print("Copy {} data from {} to {}!\n".format(data_type,self.res_path,self.aim_path))

    def copy_type_res(self, copy_type):
        assert copy_type in self.type_list, "Copy type not in type_list!type should in \n{}".format(
            self.type_list.keys())
        for root, dirs, files in os.walk(self.res_path):
            type_file_list = [name for name in files if os.path.splitext(name)[1] == self.type_list[copy_type]
                              and os.path.splitext(name)[0].endswith(copy_type)]
            for file_name in type_file_list:
                file_path = os.path.join(root, file_name)
                new_file_path = os.path.join(self.aim_path, file_name)
                shutil.copyfile(file_path, new_file_path)  # 复制文件
                print("Suceed copy file from {} to {}".format(file_path, new_file_path))

    def convert_h5ad_to_gene_class_df(self, convert_type="res"):
        # 提取对应的h5ad的结果中基因的聚类结果，相当于cluster_combine_gene_symbol
        assert convert_type in self.type_list, "Copy type not in type_list!type should in \n{}".format(
            self.type_list.keys())
        for root, dirs, files in os.walk(self.res_path):
            type_file_list = [name for name in files if os.path.splitext(name)[1] == self.type_list[convert_type]
                              and os.path.splitext(name)[0].endswith(convert_type)]
            for file_name in type_file_list:
                file_path = os.path.join(root, file_name)
                new_file_path = os.path.join(root, os.path.splitext(file_name)[0] + "_combine_gene_symbol.csv")
                ann_data_gene_class = sc.read_h5ad(file_path).obs
                ann_data_gene_class.to_csv(new_file_path, sep=',', index=True, header=True)
                print("Convert file from {} to {}".format(file_path, new_file_path))

    def res_h5ad_convert_and_copy(self):
        CopyRes.convert_h5ad_to_gene_class_df(self)
        CopyRes.copy_type_res(self, copy_type="combine_gene_symbol")


if __name__ == '__main__':
    # dataset = load_data("GEOdata_gene_nolog_filter_npy.npy")
    # dataset = dataset.x[:, 1:].astype(float)
    # print(dataset.__len__())

    # # Got anndata
    # for data_class in ['CAMprep']:
    #     data_name, adata_name = got_data_name_from_data_class(data_class)
    #     print("data type {},load in {}".format(data_class, adata_name))
    #     temp, gene_name, sample_name = load_data(adata_name)[1]
    #     ann_data = load_calculate_ann_data(
    #         adata_name, temp, gene_name, sample_name)
    #     print(ann_data)
    # 根据CAM结果进行绘图
    # right_num_tuple = [(11, 5), (3, 10), (5, 10), (7, 10), (11, 1), (3, 1),
    #                    (4, 1), (11, 3), (4, 3), (5, 3), (7, 3), (11, 5)]
    # draw_score_pc(right_num_tuple, data_type="ftl",km_flag=False)
    # # 画一下所有图的版本
    # cluster_ls = [2,3,4,5,7,11,12,18,19];graph_ls = [1,3,5,10]
    # draw_score_pc(got_all_num_tuple_list(cluster_ls,graph_ls), data_type="ftl",km_flag=False,pic_name="All_res_pic")
    # # plt.plot(np.random.randn(10), np.random.randn(10),marker='o')
    # # plt.savefig("test.jpg")
    # # 把需要的结果存入一个csv文件
    # CopyRes("res", "need_res", "CAMprep").res_h5ad_convert_and_copy()
    CopyRes("res", "need_res", "CAMprep_01_loss").res_h5ad_convert_and_copy()
