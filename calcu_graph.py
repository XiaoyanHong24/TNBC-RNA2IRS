import pandas as pd
import numpy as np
import h5py
import sklearn.preprocessing
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
from utils import *
import os


def construct_graph(features, data_name, topk, method='heat'):
    graph_dir = "graph/"+method
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    fname = graph_dir+'/{}{}_graph.txt'.format(data_name, topk)
    # num = len(label)
    dist = None
    # KNN Graph similarity
    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        features[features > 0] = 1
        features = sklearn.preprocessing.normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)
    elif method == 'ncos_no_one':
        # features[features > 0] = 1
        features = sklearn.preprocessing.normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk+1))[-(topk+1):]
        inds.append(ind)

    f = open(fname, 'w')
    counter = 0
    A = np.zeros_like(dist)
    for i, v in enumerate(inds):
        mutual_knn = False
        for vv in v:
            if vv == i:
                pass
            else:
                # if label[vv] != label[i]:
                #     counter += 1
                f.write('{} {}\n'.format(i, vv))
    f.close()
    # print('error rate: {}'.format(counter / (num * topk)))
    return dist


if __name__ == '__main__':
    data_list = ['CAMprep', 'CAMprep_01_loss']
    for data_type in data_list:
        data_name, data_file_name = got_data_name_from_data_class(data_type)
        dataset, _, _ = load_data(data_file_name, data_dir="ProceedData")[1]

        print("load data from {}".format(data_file_name))

        if dataset.shape[0] < dataset.shape[1]:
            dataset = np.transpose(dataset)

        for topk in range(1, 15):
            construct_graph(dataset, data_type, topk, 'ncos_no_one')
            construct_graph(dataset, data_type, topk, 'heat')
            construct_graph(dataset, data_type, topk, 'cos')
            construct_graph(dataset, data_type, topk, 'ncos')
