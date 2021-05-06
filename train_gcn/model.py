import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv, TAGConv
from dgl import udf
import torch.nn.functional as F
from typing import Callable


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


class RelationalTAGConv(nn.Module):
    def __init__(self, *, radius: int, width_in: int, use_relu: bool = True, **attribute_output_widths: int):
        """
        Takes `width_in` node attributes coming in, and ouputs sum(output_width for _ in attributes) node attributes,
        where each attribute is a boolean edge attribute defined on all edges.
        """
        super().__init__()
        self.kernels = nn.ModuleDict({attribute: TAGConv(width_in,
                                                         width_out,
                                                         k=radius,
                                                         activation=(nn.ReLU if False else None))
                                      for (attribute, width_out) in attribute_output_widths.items()})

    @staticmethod
    def _relation_edge_predicate(key: str) -> Callable[[udf.EdgeBatch], torch.Tensor]:
        """Needs to return a "predicate" from a list of edges to a list of "include" booleans."""
        def predicate(edges: udf.EdgeBatch) -> torch.Tensor:
            return edges.data[key]
        return predicate

    def forward(self, g, in_feats):
        # A subgraph for each kernel:
        relations = {r: g.edge_subgraph(g.filter_edges(RelationalTAGConv._relation_edge_predicate(r)),
                                        preserve_nodes=True)
                     for r in self.kernels.keys()}
        # The output features of each kernel:
        hs = [k(relations[r], in_feats)
              for (r, k) in self.kernels.items()]  # the order should be deterministic.
        return torch.cat(hs, dim=1)  # This was kinda a guess, but to the best of my ability to inspect what's happening it's correct!
        

