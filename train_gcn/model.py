import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv, TAGConv
import torch.nn.functional as F

RELU = F.relu

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')

def make_tagcn(*widths: int, radius: int, use_bias: bool = True, nonlinearity = None):
    """Will make a stack of TAGCNNs `len(widths)-1` high."""
    return nn.Sequential(*(TAGConv(w_in, w_out, k=radius, bias=use_bias, activation=nonlinearity)
                           for (w_in, w_out)
                           in zip(widths, widths[1:])))
