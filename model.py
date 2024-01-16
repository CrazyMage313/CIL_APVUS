import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from dgl.nn.pytorch.factory import KNNGraph
from layers import GraphAttentionLayer, SpGraphAttentionLayer,balanceSampleLayer
# import utils
from torch.autograd import Variable
# import torch_geometric.nn as gnn
from chainer import Chain
from chainer.functions import contrastive
import chainer.links as L


class SiameseNetwork(Chain):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(in_size=11, out_size=8)
            self.l2 = L.Linear(in_size=8, out_size=2)
            self.mlp = L.Linear(in_size=2, out_size=1)
            self.sigmoid = torch.nn.Sigmoid()


    def forward_once(self, x_data):
        # x_data = self.gat(x_data)
        h = self.l1(x_data)
        return self.l2(h)

    def forward_dist(self, x0, x1):
        y0 = self.forward_once(x0)
        y1 = self.forward_once(x1)
        dist = self.mlp(abs(y0 - y1))

        return dist


    def forward(self, x0, x1, label):
        y0 = self.forward_once(x0)
        y1 = self.forward_once(x1)

        return contrastive(y0, y1, label)

    # def gat(self, x):
    #     dgl = self.cg1(x)
    #     edges = dgl.edges()
    #     adj = sp.coo_matrix((np.ones(2 * len(x)), (edges[0], edges[1])), shape=(len(x), len(x)))
    #     adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    #     adj = torch.FloatTensor(np.array(adj.todense()))
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = F.elu(self.out_att(x, adj))
    #     # return F.log_softmax(x, dim=1)
    #     return x



class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim=[32, 32, 32], with_dropout=False):
        super(Net, self).__init__()
        conv = gnn.GCNConv  # SplineConv  NNConv   GraphConv   SAGEConv
        self.latent_dim = latent_dim
        self.conv_params = nn.ModuleList()
        self.conv_params.append(conv(input_dim, latent_dim[0], cached=False))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(conv(latent_dim[i - 1], latent_dim[i], cached=False))

        latent_dim = sum(latent_dim)

        self.linear1 = nn.Linear(latent_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)

        self.with_dropout = with_dropout

    def forward(self, data):
        data.to(torch.device("cuda"))
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
        cur_message_layer = x
        cat_message_layers = []
        lv = 0
        while lv < len(self.latent_dim):
            cur_message_layer = self.conv_params[lv](cur_message_layer, edge_index)
            cur_message_layer = torch.tanh(cur_message_layer)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)

        batch_idx = torch.unique(batch)
        idx = []
        for i in batch_idx:
            idx.append((batch == i).nonzero()[0].cpu().numpy()[0])

        cur_message_layer = cur_message_layer[idx, :]

        hidden = self.linear1(cur_message_layer)
        self.feature = hidden
        hidden = F.relu(hidden)

        if self.with_dropout:
            hidden = F.dropout(hidden, training=self.training)

        logits = self.linear2(hidden)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return logits, loss, acc, self.feature
        else:
            return logits


class ConGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(ConGAT, self).__init__()
        self.dropout = dropout

        self.cg1 = ConGraph(nfeat, nhid)

        self.balanceSample = balanceSampleLayer(nfeat)

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid+10, dropout=dropout, alpha=alpha, concat=False) ###
        self.mlp = nn.Linear(nhid+10, nclass)

    def forward(self, x1,x2,test=False):
        if test:
            x = self.balanceSample(x1,x2,test)
        else:
            x,label = self.balanceSample(x1,x2,test)
        dgl = self.cg1(x)
        edges = dgl.edges()
        adj = sp.coo_matrix((np.ones(2 * len(x)), (edges[0], edges[1])), shape=(len(x), len(x)))

        # adj = adj + adj.T.multiple(adj.T > adj) - adj.multiple(adj.T > adj)

        adj = normalize_adj(adj + sp.eye(adj.shape[0]))

        adj = torch.FloatTensor(np.array(adj.todense()))

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = self.mlp(x)      ###
        if test:
            return F.log_softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1),label
        # return  x

class Classifier(nn.Module):
    def __init__(self, nhid, nclass, dropout):
        super(Classifier, self).__init__()

        self.mlp = nn.Linear(nhid, nclass)
        # self.mlp = nn.Sequential(nn.Linear(nhid,nhid*5),nn.ReLU(),nn.Dropout(0.1),
        #                         nn.Linear(nhid*5,nclass))
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)
        # def init_weight(m):
        #     if type(m) == nn.Linear:
        #         nn.init.normal_(m.weight,std=0.01,)
        #         #print("weight:",m.weight)
        # self.mlp.apply(init_weight)

    def forward(self, x):
        x = self.mlp(x)

        return F.softmax(x,dim=1)


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

class ConGraph(nn.Module):
    def __init__(self, nfeat, nhid):
        super(ConGraph, self).__init__()

    def forward(self, X):
        kg = KNNGraph(2)
        adj = kg(X)
        return adj

class SpConGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpConGAT, self).__init__()
        self.dropout = dropout
        self.cg1 = ConGraph(nfeat, nhid)
        self.mlp1 = nn.Linear(nclass, 1)
        self.mlp2 = nn.Linear(nclass, 2)
        self.sigmoid = torch.nn.Sigmoid()

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)



    def forward(self, x1, x2):

        x1 = self.gat(x1)
        x2 = self.gat(x2)
        # x = F.cosine_similarity(x1, x2).unsqueeze(-1)
        o1 = torch.abs(x1 - x2)

        x = - self.mlp1(o1)

        # return F.log_softmax(x, dim=1)
        return x

    def single_forward(self, x1):
        x1 = self.gat(x1)
        # x1 = self.mlp2(x1)
        return F.log_softmax(x1, dim = 1)

    def gat(self, x):
        dgl = self.cg1(x)
        edges = dgl.edges()
        adj = sp.coo_matrix((np.ones(2 * len(x)), (edges[0], edges[1])), shape=(len(x), len(x)))
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        return x




