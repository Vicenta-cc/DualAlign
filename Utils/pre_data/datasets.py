from torch.utils.data import Dataset
import torch
from sklearn.cluster import KMeans
from torch_geometric.data import Data
import numpy as np
import scipy.sparse as sp
from torch_geometric.transforms import OneHotDegree
import networkx as nx
import os
import sys
from torch_geometric.io import read_txt_array

sys.path.append('..')
from Utils.pre_data import pre_arxiv


class MyDataset(Dataset):
    def __init__(self, feature, labels, domain):
        self.labels = labels
        self.feature = feature
        self.domain = domain

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.feature[idx]
        domain = self.domain[idx]
        sample = {"feature": data, "Class": label, "Domain": domain}
        return sample


def read_edges(edge_fp):
    edges = edge_fp.readlines()
    edges = [tuple(row.split()) for row in edges]
    edges = [(float(row[0]), float(row[1])) for row in edges]
    return edges


def edges_to_adj(edges, num_node):
    edge_source = [int(i[0]) for i in edges]
    edge_target = [int(i[1]) for i in edges]
    data = np.ones(len(edge_source))
    adj = sp.csr_matrix((data, (edge_source, edge_target)),
                        shape=(num_node, num_node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    rows, columns = adj.nonzero()
    edge_index = torch.tensor([rows, columns], dtype=torch.long)
    return adj, edge_index

def edgeidx_to_adj(edge_source, edge_target, num_node):
    data = np.ones(len(edge_source))
    adj = sp.csr_matrix((data, (edge_source, edge_target)),
                        shape=(num_node, num_node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = adj + sp.eye(adj.shape[0])
    return adj

def pileup(num_events, args, datadir):
    graph_list = prepare_pileup.prepare_dataset(num_events, args, datadir)
    return graph_list


def apply_rotation(features, angle):
    rotation_matrix = np.array([
        [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
        [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
    ])
    return np.dot(features, rotation_matrix)

def get_synthetic_source_data(num_nodes, SIGMA, p, q, rotation_angle=0):
    B = [[p, q, q],
         [q, p, q],
         [q, q, p]]


    C0 = np.random.multivariate_normal([-1, 0], np.eye(2) * SIGMA ** 2, num_nodes)
    C1 = np.random.multivariate_normal([1, 0], np.eye(2) * SIGMA ** 2, num_nodes)
    C2 = np.random.multivariate_normal([3, 2], np.eye(2) * SIGMA ** 2, num_nodes)
    features = np.vstack([C0, C1, C2])

    features = apply_rotation(features, rotation_angle)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    node_idex = np.arange(3 * num_nodes)
    n = len(labels)
    adjacency_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            prob = B[labels[i]][labels[j]]
            if np.random.rand() < prob:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    G = nx.from_numpy_array(adjacency_matrix)
    edge_list = list(G.edges)

    features = torch.FloatTensor(features)
    label = torch.LongTensor(labels)

    c0_idx = np.where(labels == 0)[0]
    c1_idx = np.where(labels == 1)[0]
    c2_idx = np.where(labels == 2)[0]

    np.random.shuffle(c0_idx)
    np.random.shuffle(c1_idx)
    np.random.shuffle(c2_idx)

    idx_source_train = np.concatenate((c0_idx[:int(0.6 * len(c0_idx))],
                                       c1_idx[:int(0.6 * len(c1_idx))], c2_idx[:int(0.6 * len(c2_idx))]))
    idx_source_valid = np.concatenate((c0_idx[int(0.6 * len(c0_idx)): int(0.8 * len(c0_idx))],
                                       c1_idx[int(0.6 * len(c1_idx)): int(0.8 * len(c1_idx))],
                                       c2_idx[int(0.6 * len(c2_idx)): int(0.8 * len(c2_idx))]))
    idx_source_test = np.concatenate((c0_idx[int(0.8 * len(c0_idx)):],
                                      c1_idx[int(0.8 * len(c1_idx)):], c2_idx[int(0.8 * len(c2_idx)):]))
    idx_target_valid = np.concatenate((c0_idx[:int(0.2 * len(c0_idx))],
                                       c1_idx[:int(0.2 * len(c1_idx))], c2_idx[:int(0.2 * len(c2_idx))]))
    idx_target_test = np.concatenate((c0_idx[int(0.2 * len(c0_idx)):],
                                      c1_idx[int(0.2 * len(c1_idx)):], c2_idx[int(0.2 * len(c2_idx)):]))
    num_nodes = len(label)
    adj, edge_index = edges_to_adj(edge_list, num_nodes)

    graph = Data(x=features, edge_index=edge_index, y=label)
    graph.source_training_mask = idx_source_train
    graph.source_validation_mask = idx_source_valid
    graph.source_testing_mask = idx_source_test
    graph.target_validation_mask = idx_target_valid
    graph.target_testing_mask = idx_target_test
    graph.source_mask = np.arange(graph.num_nodes)
    graph.target_mask = np.arange(graph.num_nodes)

    graph.adj = adj
    graph.y_hat = label
    graph.num_classes = 3
    graph.edge_weight = torch.ones(graph.num_edges)

    return graph


def get_synthetic_target_data(num_nodes, SIGMA, p, q, rotation_angle=30):
    B = [[p, q, q],
         [q, p, q],
         [q, q, p]]

    C0 = np.random.multivariate_normal([-1, 0], np.eye(2) * SIGMA ** 2, num_nodes)
    C1 = np.random.multivariate_normal([1, 0], np.eye(2) * SIGMA ** 2, num_nodes)
    C2 = np.random.multivariate_normal([3, 2], np.eye(2) * SIGMA ** 2, num_nodes)
    features = np.vstack([C0, C1, C2])

    features = apply_rotation(features, rotation_angle)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)


    node_idex = np.arange(3 * num_nodes)
    n = len(labels)
    adjacency_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            prob = B[labels[i]][labels[j]]
            if np.random.rand() < prob:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    G = nx.from_numpy_array(adjacency_matrix)
    edge_list = list(G.edges)


    features = torch.FloatTensor(features)
    label = torch.LongTensor(labels)

    c0_idx = np.where(labels == 0)[0]
    c1_idx = np.where(labels == 1)[0]
    c2_idx = np.where(labels == 2)[0]

    np.random.shuffle(c0_idx)
    np.random.shuffle(c1_idx)
    np.random.shuffle(c2_idx)

    idx_source_train = np.concatenate((c0_idx[:int(0.6 * len(c0_idx))],
                                       c1_idx[:int(0.6 * len(c1_idx))], c2_idx[:int(0.6 * len(c2_idx))]))
    idx_source_valid = np.concatenate((c0_idx[int(0.6 * len(c0_idx)): int(0.8 * len(c0_idx))],
                                       c1_idx[int(0.6 * len(c1_idx)): int(0.8 * len(c1_idx))],
                                       c2_idx[int(0.6 * len(c2_idx)): int(0.8 * len(c2_idx))]))
    idx_source_test = np.concatenate((c0_idx[int(0.8 * len(c0_idx)):],
                                      c1_idx[int(0.8 * len(c1_idx)):], c2_idx[int(0.8 * len(c2_idx)):]))
    idx_target_valid = np.concatenate((c0_idx[:int(0.2 * len(c0_idx))],
                                       c1_idx[:int(0.2 * len(c1_idx))], c2_idx[:int(0.2 * len(c2_idx))]))
    idx_target_test = np.concatenate((c0_idx[int(0.2 * len(c0_idx)):],
                                      c1_idx[int(0.2 * len(c1_idx)):], c2_idx[int(0.2 * len(c2_idx)):]))
    num_nodes = len(label)
    adj, edge_index = edges_to_adj(edge_list, num_nodes)

    graph = Data(x=features, edge_index=edge_index, y=label)
    graph.source_training_mask = idx_source_train
    graph.source_validation_mask = idx_source_valid
    graph.source_testing_mask = idx_source_test
    graph.target_validation_mask = idx_target_valid
    graph.target_testing_mask = idx_target_test
    graph.source_mask = np.arange(graph.num_nodes)
    graph.target_mask = np.arange(graph.num_nodes)

    graph.adj = adj
    graph.y_hat = label
    graph.num_classes = 3
    graph.edge_weight = torch.ones(graph.num_edges)

    return graph




def prepare_dblp_acm(raw_dir, name):
    docs_path = os.path.join(raw_dir, name, 'raw/{}_docs.txt'.format(name))
    f = open(docs_path, 'rb')
    content_list = []
    for line in f.readlines():
        line = str(line, encoding="utf-8")
        content_list.append(line.split(","))
    x = np.array(content_list, dtype=float)
    x = torch.from_numpy(x).to(torch.float)

    edge_path = os.path.join(raw_dir, name, 'raw/{}_edgelist.txt'.format(name))
    edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long).t()

    num_node = x.size(0)
    data = np.ones(edge_index.size(1))
    adj = sp.csr_matrix((data, (edge_index[0], edge_index[1])),
                        shape=(num_node, num_node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    label_path = os.path.join(raw_dir, name, 'raw/{}_labels.txt'.format(name))
    f = open(label_path, 'rb')
    content_list = []
    for line in f.readlines():
        line = str(line, encoding="utf-8")
        line = line.replace("\r", "").replace("\n", "")
        content_list.append(line)
    y = np.array(content_list, dtype=int)

    num_class = np.unique(y)
    class_index = []
    for i in num_class:
        c_i = np.where(y == i)[0]
        class_index.append(c_i)

    training_mask = np.array([])
    validation_mask = np.array([])
    testing_mask = np.array([])
    tgt_validation_mask = np.array([])
    tgt_testing_mask = np.array([])
    for idx in class_index:
        np.random.shuffle(idx)
        training_mask = np.concatenate((training_mask, idx[0:int(len(idx) * 0.6)]), 0)
        validation_mask = np.concatenate((validation_mask, idx[int(len(idx) * 0.6):int(len(idx) * 0.8)]), 0)
        testing_mask = np.concatenate((testing_mask, idx[int(len(idx) * 0.8):]), 0)
        tgt_validation_mask = np.concatenate((tgt_validation_mask, idx[0:int(len(idx) * 0.2)]), 0)
        tgt_testing_mask = np.concatenate((tgt_testing_mask, idx[int(len(idx) * 0.2):]), 0)

    training_mask = training_mask.astype(int)
    testing_mask = testing_mask.astype(int)
    validation_mask = validation_mask.astype(int)
    y = torch.from_numpy(y).to(torch.int64)
    graph = Data(edge_index=edge_index, x=x, y=y)
    graph.source_training_mask = training_mask
    graph.source_validation_mask = validation_mask
    graph.source_testing_mask = testing_mask
    graph.source_mask = np.concatenate((training_mask, validation_mask, testing_mask), 0)
    graph.target_validation_mask = tgt_validation_mask
    graph.target_testing_mask = tgt_testing_mask
    graph.target_mask = np.concatenate((tgt_validation_mask, tgt_testing_mask), 0)
    graph.adj = adj
    graph.y_hat = y
    graph.num_classes = len(num_class)
    graph.edge_weight = torch.ones(graph.num_edges)

    return graph


def prepare_Arxiv(root, years):
    dataset = pre_arxiv.load_nc_dataset(root, 'ogb-arxiv', years)
    idx = (dataset.test_mask == True).nonzero().view(-1).numpy()
    np.random.shuffle(idx)
    num_training = idx.shape[0]
    adj = edgeidx_to_adj(dataset.graph['edge_index'][0], dataset.graph['edge_index'][1], dataset.graph['num_nodes'])
    edge_index = torch.from_numpy(np.array([adj.nonzero()[0], adj.nonzero()[1]])).long()
    graph = Data(edge_index=edge_index, x=dataset.graph['node_feat'], y=dataset.label.view(-1))
    graph.adj = adj
    graph.source_training_mask = idx[0:int(0.6*num_training)]
    graph.source_validation_mask = idx[int(0.6*num_training):int(0.8*num_training)]
    graph.source_testing_mask = idx[int(0.8*num_training):]
    graph.target_validation_mask = idx[0:int(0.2*num_training)]
    graph.target_testing_mask = idx[int(0.2*num_training):]
    graph.source_mask = idx
    graph.target_mask = idx
    graph.edge_weight = torch.ones(graph.num_edges)
    graph.num_classes = dataset.num_classes
    if torch.unique(graph.y).size(0) < graph.num_classes:
        print("miss classes")
    graph.y_hat = graph.y
    edge_class = np.zeros((graph.num_edges, graph.num_classes, graph.num_classes))
    for idx in range(graph.num_edges):
        i = graph.edge_index[0][idx]
        j = graph.edge_index[1][idx]
        edge_class[idx, graph.y[i], graph.y[j]] = 1
    graph.edge_class = edge_class
    return graph


def prepare_airport(raw_dir,name):
    label_path = os.path.join(raw_dir,name, 'raw/{}_labels.txt'.format(name))
    f = open(label_path, 'rb')
    content_list = []
    for line in f.readlines():
        line = str(line, encoding="utf-8")
        line = line.replace("\r", "").replace("\n", "")
        content_list.append(line)
    y = np.array(content_list, dtype=int)
    #y = torch.from_numpy(y).to(torch.int64)
    num_node = len(y)
    edge_path = os.path.join(raw_dir, name,'raw/{}_edgelist.txt'.format(name))
    edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long).t()
    data = np.ones(edge_index.size(1))
    adj = sp.csr_matrix((data, (edge_index[0], edge_index[1])),
                        shape=(num_node, num_node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)


    num_class = np.unique(y)
    class_index = []
    for i in num_class:
        c_i = np.where(y == i)[0]
        class_index.append(c_i)

    training_mask = np.array([])
    validation_mask = np.array([])
    testing_mask = np.array([])
    tgt_validation_mask = np.array([])
    tgt_testing_mask = np.array([])
    for idx in class_index:
        np.random.shuffle(idx)
        training_mask = np.concatenate((training_mask, idx[0:int(len(idx) * 0.6)]), 0)
        validation_mask = np.concatenate((validation_mask, idx[int(len(idx) * 0.6):int(len(idx) * 0.8)]), 0)
        testing_mask = np.concatenate((testing_mask, idx[int(len(idx) * 0.8):]), 0)
        tgt_validation_mask = np.concatenate((tgt_validation_mask, idx[0:int(len(idx) * 0.2)]), 0)
        tgt_testing_mask = np.concatenate((tgt_testing_mask, idx[int(len(idx) * 0.2):]), 0)

    training_mask = training_mask.astype(int)
    testing_mask = testing_mask.astype(int)
    validation_mask = validation_mask.astype(int)

    y = torch.from_numpy(y).to(torch.int64)

    # Apply OneHotDegree
    graph = Data(edge_index=edge_index, y=y, x=torch.ones((num_node, 1))) 
    one_hot_transform = OneHotDegree(241)
    graph = one_hot_transform(graph)

    graph.source_training_mask = training_mask
    graph.source_validation_mask = validation_mask
    graph.source_testing_mask = testing_mask
    graph.source_mask = np.concatenate((training_mask, validation_mask, testing_mask), 0)
    graph.target_validation_mask = tgt_validation_mask
    graph.target_testing_mask = tgt_testing_mask
    graph.target_mask = np.concatenate((tgt_validation_mask, tgt_testing_mask), 0)
    graph.adj = adj
    graph.y_hat = y
    graph.num_classes = len(num_class)
    graph.edge_weight = torch.ones(graph.num_edges)

    return graph

