import torch
import torch.nn as nn
import dgl.function as fn

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
    def __init__(self, in_dimension, hidden_dimension, out_dimension):
        super(GraphGCN, self).__init__()
        self.gcn1 = GCN(in_dimension, hidden_dimension)
        self.gcn2 = GCN(hidden_dimension, out_dimension)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x2 = self.gcn2(g, x)
        final_embedding = torch.stack([x, x2], dim=1)
        final_embedding = torch.mean(final_embedding, dim=1)
        return final_embedding
