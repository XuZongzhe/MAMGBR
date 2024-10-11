import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import random

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCN, self).__init__()
        self.gcn_msg = fn.copy_u(u='h', out='m')
        self.gcn_reduce = fn.mean(msg='m', out='h')

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(self.gcn_msg, self.gcn_reduce)
        return g.ndata['h']

class GraphGCN(nn.Module):
    def __init__(self, in_dimension, hidden_dimension, out_dimension, dropout=0.5):
        super(GraphGCN, self).__init__()
        self.dropout = dropout
        self.gcn1 = GCN(in_dimension, hidden_dimension)
        self.gcn2 = GCN(hidden_dimension, out_dimension)
        self.masked_adj = None
        self.edge_indices = None
        self.edge_values = None

    def pre_epoch_processing(self, g):
        if self.dropout <= 0:
            self.masked_adj = g.adj()
            return

        adj = g.adj()
        self.edge_indices = adj.indices()
        self.edge_values = torch.ones(self.edge_indices.size(1))

        keep_len = int(self.edge_values.size(0) * (1. - self.dropout))
        keep_idx = torch.tensor(random.sample(range(self.edge_values.size(0)), keep_len)).long()
        self.edge_indices = self.edge_indices[:, keep_idx]
        self.edge_values = self.edge_values[keep_idx]

        keep_len = int(self.edge_values.size(0) * (1. - self.dropout))
        keep_idx = torch.tensor(random.sample(range(self.edge_values.size(0)), keep_len)).long()
        self.edge_indices = self.edge_indices[:, keep_idx].to(g.device)
        self.edge_values = self.edge_values[keep_idx].to(g.device)
        self.masked_adj = torch.sparse_coo_tensor(self.edge_indices, self.edge_values, g.adj().shape, device=g.device)

    def forward(self, g, features):
        self.pre_epoch_processing(g)

        x1 = self.gcn1(g, features)
        x2 = self.gcn2(g, x1)

        weights = F.cosine_similarity(x2, x1, dim=-1).unsqueeze(-1)
        final_embedding = weights * x2 + (1 - weights) * x1

        return final_embedding
