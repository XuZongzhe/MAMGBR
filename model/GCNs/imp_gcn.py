import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation, groups=3):
        super(GCN, self).__init__()
        self.apply_mod = nn.Linear(in_feats, out_feats)
        self.gcn_msg = fn.copy_u(u='h', out='m')
        self.gcn_reduce = fn.mean(msg='m', out='h')
        self.groups = groups
        self.W_gc = nn.Parameter(torch.randn(in_feats, self.groups))
        self.b_gc = nn.Parameter(torch.randn(1, self.groups))

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(self.gcn_msg, self.gcn_reduce)
        h = g.ndata['h']
        group_embeddings = F.relu(torch.matmul(h, self.W_gc) + self.b_gc)
        top_group, _ = torch.topk(group_embeddings, 1, dim=-1)
        group_mask = torch.eq(group_embeddings, top_group).float()
        group_mask = group_mask.unsqueeze(-1)
        group_mask = group_mask.expand(-1, -1, h.size(1))
        h = h.unsqueeze(1)
        h = h * group_mask
        h = h.sum(dim=1)
        h = self.apply_mod(h)
        return h

class GraphGCN(nn.Module):
    def __init__(self, in_dimension, hidden_dimension, out_dimension, groups=3):
        super(GraphGCN, self).__init__()
        self.gcn1 = GCN(in_dimension, hidden_dimension, F.relu, groups)
        self.gcn2 = GCN(hidden_dimension, out_dimension, F.relu, groups)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x
