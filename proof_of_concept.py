from dgl import from_networkx
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import TAGConv
from dgl.nn.pytorch.glob import AvgPooling
from dgl.nn.pytorch.utils import Sequential
from itertools import count, cycle, product
import matplotlib.pyplot as plt
from networkx import DiGraph
from random import shuffle
from torch import cat, no_grad, reshape, Tensor
from torch.nn import Module
from torch.nn.functional import cross_entropy
from torch.optim import Adam, Optimizer
from torch.utils.data.sampler import RandomSampler
from typing import Callable, List, Mapping, Sequence, Tuple

from train_gcn.model import RelationalTAGConv

Factory = Callable[[bool, bool], DiGraph]
Criterion = Callable[[Tensor, Tensor], Tensor]  # IDK how to tighten this.

# For demonstration purposes, we'll be asking models to distinguish between four classes along two axies.
# Thus, random guessing will give 25% accuracy, and ability to distinguish one of the axies of differnece will give 50%.
cases = ((0, False, False),
         (1, False, True),
         (2, True, False),
         (3, True, True))

class SyntheticDataset(DGLDataset):
    """Just like our real dataset class, except it gets its data from a factory function instead of reading from disk."""
    def __init__(self, factory: Factory, node_attrs: Sequence[str], edge_attrs: Sequence[str], size: int):
        self.size = (((size - 1) // 4) + 1) * 4  # round up to multiple of 4.
        self.factory = factory
        self.node_attrs = node_attrs
        self.edge_attrs = edge_attrs
        super().__init__(name='synthetic')

    def process(self) -> None:
        self.graphs = []
        self.labels = []
        for (_, (c, arg1, arg2)) in zip(range(self.size), cycle(cases)):
            g_nx = self.factory(arg1, arg2)
            num_nodes = g_nx.number_of_nodes()
            g = from_networkx(g_nx, node_attrs=self.node_attrs, edge_attrs=self.edge_attrs)  # We need the edge data passed in in this way,
            g.ndata['tensor'] = cat([reshape(g.ndata[key], (len(g_nx.nodes), 1))  # but this is the only format in which we need the node data?
                                     for key in self.node_attrs],
                                    dim=1).float()
            self.graphs.append(g)
            self.labels.append(c)

    def __getitem__(self, i) -> Tuple:
        return self.graphs[i], self.labels[i]

    def __len__(self) -> int:
        return len(self.graphs)

def demonstrate_learning(models: Mapping[str, Module],
                         factory: Factory,
                         node_attrs: Sequence[str],
                         edge_attrs: Sequence[str],
                         train_data_len: int = 200,
                         epochs: int = 25,
                         test_data_len: int = 200
                         ) -> None:
    """Train the models and plot their accuracy."""
    accuracies: Mapping[str, List[float]] = {name: [] for name in models.keys()}
    batch_size = next(n for n in range(20, 0, -1) if (train_data_len % n) == 0 and (test_data_len % n) == 0)  # It's just 20.
    training_data = SyntheticDataset(factory=factory,
                                     node_attrs=node_attrs,
                                     edge_attrs=edge_attrs,
                                     size=train_data_len)
    training_loader = GraphDataLoader(training_data,
                                      sampler=RandomSampler(training_data),
                                      batch_size=batch_size,
                                      drop_last=False)
    test_data = SyntheticDataset(factory=factory,
                                 node_attrs=node_attrs,
                                 edge_attrs=edge_attrs,
                                 size=test_data_len)
    test_loader = GraphDataLoader(test_data,
                                  sampler=RandomSampler(test_data),
                                  batch_size=batch_size,
                                  drop_last=False)
    optimizers = {name: Adam(model.parameters(), lr=0.1) for (name, model) in models.items()}
    criterions = {name: cross_entropy for (name, model) in models.items()}
    for _ in range(epochs):
        for (name, model) in models.items():
            accuracies[name].append(epoch(training_loader, test_loader, model, optimizers[name], criterions[name]))
            print('.', end='', flush=True)
    print()
    plt.figure(figsize=(12, 7))
    for (i, (style, (name, accuracy))) in enumerate(zip(cycle(['-', '--', '-.', ':']), accuracies.items())):
        plt.plot(accuracy,
                 label=name,
                 alpha=0.75,
                 linewidth=(12 - (10 * (i / len(accuracies)))),
                 linestyle=style)
    plt.legend(handlelength=6)
    plt.show()

def train(data: GraphDataLoader,
          model: Module,
          optimizer: Optimizer,
          criterion: Criterion
          ) -> None:
    model.train()  # enable learning
    for batched_graph, labels in data:
        optimizer.zero_grad()
        pred = model(batched_graph, batched_graph.ndata['tensor'])
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

def test(data: GraphDataLoader,
         model: Module
         ) -> float:  # returns accuracy
    model.eval()  # disable learning
    with no_grad():  # skip computation of gradients
        def correct(batched_graph, labels) -> int:
            pred = model(batched_graph, batched_graph.ndata['tensor'])
            return (pred.argmax(1) == labels).sum().item()
        num_correct = sum(correct(batched_graph, labels) for batched_graph, labels in data)
        num_tests = sum(len(labels) for batched_graph, labels in data)
        accuracy = num_correct / num_tests
    return accuracy

def epoch(training_data: GraphDataLoader,
          testing_data: GraphDataLoader,
          model: Module,
          optimizer: Optimizer,
          criterion: Criterion
          ) -> float:  # returns the accuracy.
    train(training_data, model, optimizer, criterion)
    return test(testing_data, model)

def the_dumbest_factory(arg1: bool, arg2: bool) -> DiGraph:
    """First axis of differnce: mostly 0s vs mostly 1s. Second: All nodes link to all 0-nodes, or to all 1-nodes."""
    cs = [0, 0, 0, int(arg1), int(arg1), int(arg1), 1, 1, 1]
    g = DiGraph()
    for (n, c) in enumerate(cs):
        g.add_node(n, c=c)
    for n1 in g:
        for n2 in g:
            if int(arg2) == g.nodes[n2]['c']:
                g.add_edge(n1, n2)
    return g

def scratch1(a: bool, b: bool) -> DiGraph:
    c_key, s_key = 'c', 's'
    g = DiGraph()
    for (n, c) in enumerate(range(3)):
        g.add_node(n, **{c_key: c})
    for (n1, n2) in product(g, g):
        if n1 <= n2:
            g.add_edge(n1, n2, **{s_key: int(n1 == n2)})
    return g
def scratch2(a: bool, b: bool) -> DiGraph:
    c_key, s_key = 'c', 's'
    g = DiGraph()
    for (n, c) in enumerate(range(300)):
        g.add_node(n, **{c_key: c})
    for (n1, n2) in product(g, g):
        if n1 <= n2:
            g.add_edge(n1, n2, **{s_key: int(n1 == n2)})
    return g
def scratch3(a: bool, b: bool) -> DiGraph:
    c_key, s_key = 'c' * 1000, 's' * 1000
    g = DiGraph()
    for (n, c) in enumerate(range(3)):
        g.add_node(n, **{c_key: c})
    for (n1, n2) in product(g, g):
        if n1 <= n2:
            g.add_edge(n1, n2, **{s_key: int(n1 == n2)})
    return g
def scratch4(a: bool, b: bool) -> DiGraph:
    c_key, s_key = 'c' * 1000, 's' * 1000
    g = DiGraph()
    for (n, c) in enumerate(range(300)):
        g.add_node(n, **{c_key: c})
    for (n1, n2) in product(g, g):
        if n1 <= n2:
            g.add_edge(n1, n2, **{s_key: int(n1 == n2)})
    return g

def demo1() -> None:
    """A radius-0 TAGConv can distinguish the averate values of nodes, but not what they link to."""
    r0 = Sequential(TAGConv(1, 4, k=0), AvgPooling())
    r1 = Sequential(TAGConv(1, 4, k=1), AvgPooling())
    r2 = Sequential(TAGConv(1, 4, k=2), AvgPooling())
    demonstrate_learning(models={'radius 0': r0,
                                 'radius 1': r1,
                                 'radius 2': r2},
                         factory=the_dumbest_factory,
                         node_attrs=['c'],
                         edge_attrs=[])

def topology_factory(arg1: bool, arg2: bool) -> DiGraph:
    """First axis of differnce: two small chains or one bigger one. Second: open chains or closed loops."""
    g = DiGraph()
    for n in range(7):
        g.add_node(n, c=1)
    if arg1:
        for (n, n_) in ((0, 1), (1, 2),
                        (3, 4), (4, 5), (5, 6)):
            g.add_edge(n, n_)
            if arg2:
                g.add_edge(2, 0)
                g.add_edge(6, 3)
    else:
        for (n, n_) in zip(range(6), count(1)):
            g.add_edge(n, n_)
        if arg2:
            g.add_edge(6, 0)
    return g

def demo2() -> None:
    """What's needed to distinguish these homogeneous graphs with obviously different topology?"""
    r0d4 = Sequential(TAGConv(1, 2, k=0), TAGConv(2, 2, k=0), TAGConv(2, 2, k=0), TAGConv(2, 4, k=0), AvgPooling())
    r1d1 = Sequential(TAGConv(1, 4, k=1), AvgPooling())
    r1d2 = Sequential(TAGConv(1, 2, k=1), TAGConv(2, 4, k=1), AvgPooling())
    r2d1 = Sequential(TAGConv(1, 4, k=2), AvgPooling())
    r2d4 = Sequential(TAGConv(1, 2, k=2), TAGConv(2, 2, k=2), TAGConv(2, 2, k=2), TAGConv(2, 4, k=2), AvgPooling())
    r5d1 = Sequential(TAGConv(1, 4, k=5), AvgPooling())
    r4d3 = Sequential(TAGConv(1, 2, k=4), TAGConv(2, 2, k=4), TAGConv(2, 4, k=4), AvgPooling())
    r8d8 = Sequential(TAGConv(1, 2, k=8), TAGConv(2, 2, k=8), TAGConv(2, 2, k=8), TAGConv(2, 2, k=8),
                      TAGConv(2, 2, k=8), TAGConv(2, 2, k=8), TAGConv(2, 2, k=8), TAGConv(2, 4, k=8), AvgPooling())
    demonstrate_learning(models={'r0d4': r0d4,
                                 'r1d1': r1d1,
                                 'r1d2': r1d2,
                                 'r2d1': r2d1,
                                 'r2d4': r2d4,
                                 'r5d1': r5d1,
                                 'r4d3': r4d3,
                                 'r8d8': r8d8,
                                 },
                         factory=topology_factory,
                         node_attrs=['c'],
                         edge_attrs=[],
                         epochs=35)

def relation_topology_factory(arg1: bool, arg2: bool) -> DiGraph:
    """First axis of differnce: 3-loops vs 4-loops. Second: relation selects largest neighbor, or smallest neighbor."""
    cs = [(2 * n / 12) - 1 for n in range(12)]
    shuffle(cs)
    grouper = [iter(range(12))] * (3 if arg1 else 4)  # I found this pattern in itertools recipies: grouper.
    groups = zip(*grouper)

    def is_selected(n: int, ns: Sequence[int]) -> bool:
        """If arg2, select the smaller of the neighbors, else select the larger."""
        return (n == max(ns, key=lambda _n: cs[_n])) != arg2

    g = DiGraph()
    for (n, c) in enumerate(cs):
        g.add_node(n, c=c)
    for group in groups:
        for (i, n1) in enumerate(group):
            neighbors = (group[(n1 - 1) % len(group)], group[(n1 + 1) % len(group)])
            for n2 in neighbors:
                selected = is_selected(n2, neighbors)
                g.add_edge(n1, n2, everyone=True, selected=selected, not_selected=not selected)
    return g

def demo3(reach: int = 2) -> None:
    """A radius-0 TAGConv can distinguish the averate values of nodes, but not what they link to."""
    nothing = Sequential(RelationalTAGConv(radius=1, width_in=1, everyone=8), TAGConv(8, 4, k=0), AvgPooling())
    no_reach = Sequential(RelationalTAGConv(radius=1, width_in=1, selected=4, not_selected=4), TAGConv(8, 4, k=0), AvgPooling())
    no_relations = Sequential(RelationalTAGConv(radius=reach, width_in=1, everyone=8), TAGConv(8, 4, k=0), AvgPooling())
    should_work = Sequential(RelationalTAGConv(radius=reach, width_in=1, selected=4, not_selected=4), TAGConv(8, 4, k=0), AvgPooling())
    demonstrate_learning(models={'nothin': nothing,
                                 'no reach': no_reach,
                                 'no relations': no_relations,
                                 'should work': should_work},
                         factory=relation_topology_factory,
                         node_attrs=['c'],
                         edge_attrs=['everyone', 'selected', 'not_selected'])


