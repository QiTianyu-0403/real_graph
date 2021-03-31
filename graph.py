
import torch, argparse
import os.path as osp
import torch_geometric.datasets as geo_data
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
import numpy as np
import random
import scipy.sparse as sp
import networkx as nx

from models import data_to_G,SBM,p_random,p_random_simple,zipf,DCSBM,get_weight
from draw import draw_degree,draw,draw_data_x,draw_degree_fit_power,\
                    draw_degree_fit_line,draw_old,get_ave_degree,draw_network


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_parse():
    # parser for hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data', type=str, default='arxiv',
                        help='{cora, pubmed, citeseer, arxiv}.')
    return parser

def normalize_adj_row(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(adj)
    return mx,rowsum



def to_torch_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(args):
    DATA_ROOT = 'datasets'
    path = osp.join(DATA_ROOT, args.data)
    data = geo_data.Planetoid(path, args.data)[0]

    # data.train_mask = data.train_mask.type(torch.bool)
    # data.val_mask = data.val_mask.type(torch.bool)
    # data.test_mask = data.test_mask.type(torch.bool)
    # expand test_mask to all rest nodes
    # data.test_mask = ~(data.train_mask + data.val_mask)
    # get adjacency matrix
    n = len(data.x)
    adj = sp.csr_matrix((np.ones(data.edge_index.shape[1]), data.edge_index), shape=(n, n))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(adj.shape[0])
    adj,degree = normalize_adj_row(adj)  # symmetric normalization works bad, but why? Test more.
    data.adj = to_torch_sparse(adj)

    return data,degree


def build_data(args):
    """
        for cora:
            Data(adj=[2708, 2708], edge_index=[2, 10556], test_mask=[2708],
                train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])
        for citeseer:
            Data(adj=[3327, 3327], edge_index=[2, 9104], test_mask=[3327],
                train_mask=[3327], val_mask=[3327], x=[3327, 3703], y=[3327])
        for pumbed:
            Data(adj=[19717, 19717], edge_index=[2, 88648], test_mask=[19717],
                train_mask=[19717], val_mask=[19717], x=[19717, 500], y=[19717])

    """
    assert args.data in ['cora', 'pubmed', 'citeseer']
    data,degree = load_data(args)
    n_class = int(data.y.max()) + 1
    n_feat = data.x.size(1)
    return data, n_feat, n_class,degree

def build_OGB_data(args):
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./datasets/')
    print(dataset)
    data = dataset[0]
    n = len(data.x)
    adj = sp.csr_matrix((np.ones(data.edge_index.shape[1]), data.edge_index), shape=(n, n))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(adj.shape[0])
    adj, degree = normalize_adj_row(adj)  # symmetric normalization works bad, but why? Test more.
    data.adj = to_torch_sparse(adj)
    n_class = int(data.y.max()) + 1
    n_feat = data.x.size(1)
    return data, n_feat, n_class,degree


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = get_parse()
    args = parser.parse_args()

    if args.data == 'arxiv':
        data, n_feat, n_class ,degree = build_OGB_data(args)
        data = data.to(device)
    else:
        data, n_feat, n_class ,degree = build_data(args)
        data = data.to(device)
    print(data.edge_index)
    G=data_to_G(data)
    #draw_old(G)
    draw_degree_fit_line(G)

    '''
    power=[]
    for i in range(1,10001):
        power.append(zipf(i,450,1.3))

    random.shuffle(power)
    G=DCSBM(sizes=[500,500], p=p_random_simple(2), theta=power, sparse=True)
    draw_degree_fit_power(G)
    '''

    '''
    #get_weight(data)
    p=p_random_simple(2)
    pin=p[0][0]
    pout=p[0][1]
    get_weight(degree, pin, pout)
    G2 = DCSBM(sizes=[9858, 9859], p=p, theta=get_weight(degree,pin,pout), sparse=True)
    #G2=SBM(sizes=[1354, 1354], p=p, nodelist=None, seed=None, directed=False, selfloops=False, sparse=True)
    get_ave_deree(G2)
    draw_degree_fit_line(G2)
    '''


    #get_ave_degree(G)

    print('hello')



