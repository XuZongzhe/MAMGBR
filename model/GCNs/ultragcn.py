import torch
import torch.nn as nn
import dgl.function as fn

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, use_constraint=True):
        super(GCN, self).__init__()
        self.apply_mod = nn.Linear(in_feats, out_feats)
        self.gcn_msg = fn.copy_u(u='h', out='m')
        self.gcn_reduce = fn.mean(msg='m', out='h')
        self.use_constraint = use_constraint

    def forward(self, g, feature):
        if self.use_constraint:
            in_degrees = g.in_degrees().float()
            constraint_matrix = 1.0 / torch.sqrt(in_degrees + 1)
            feature = feature * constraint_matrix.unsqueeze(-1)

        g.ndata['h'] = feature
        g.update_all(self.gcn_msg, self.gcn_reduce)
        h = g.ndata['h']
        h = self.apply_mod(h)
        return h

class GraphGCN(nn.Module):
    def __init__(self, in_dimension, hidden_dimension, out_dimension, use_constraint=True):
        super(GraphGCN, self).__init__()
        self.gcn1 = GCN(in_dimension, hidden_dimension, use_constraint)
        self.gcn2 = GCN(hidden_dimension, out_dimension, use_constraint)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x
