from dgl import from_networkx
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import TAGConv
from dgl.nn.pytorch.glob import AvgPooling
from dgl.nn.pytorch.utils import Sequential
from itertools import cycle
import matplotlib.pyplot as plt
from networkx import DiGraph
from torch import cat, no_grad, reshape, Tensor
from torch.nn import Module
from torch.nn.functional import cross_entropy
from torch.optim import Adam, Optimizer
from torch.utils.data.sampler import RandomSampler
from typing import Callable, List, Mapping, Sequence, Tuple

Generator = Callable[[bool, bool], DiGraph]
Criterion = Callable[[Tensor, Tensor], Tensor]  # IDK how to tighten this.

classes = ((0, False, False),
           (1, False, True),
           (2, True, False),
           (3, True, True))

class SyntheticDataset(DGLDataset):
    def __init__(self, generator: Generator, node_attrs: Sequence[str], edge_attrs: Sequence[str], size: int):
        self.size = (((size - 1) // 4) + 1) * 4
        self.generator = generator
        self.node_attrs = node_attrs
        self.edge_attrs = edge_attrs
        super().__init__(name='synthetic')

    def process(self) -> None:
        self.graphs = []
        self.labels = []
        for (_, (c, arg1, arg2)) in zip(range(self.size), cycle(classes)):
            g_nx = self.generator(arg1, arg2)
            num_nodes = g_nx.number_of_nodes()
            g = from_networkx(g_nx, node_attrs=self.node_attrs, edge_attrs=self.edge_attrs)
            g.ndata['tensor'] = cat([reshape(g.ndata[key], (len(g_nx.nodes), 1))  # there's gotta be a better way?
                                     for key in self.node_attrs],
                                    dim=1).float()
            # and then somehow I gotta do the same thing for the edges?
            self.graphs.append(g)
            self.labels.append(c)
        
    def __getitem__(self, i) -> Tuple:
        return self.graphs[i], self.labels[i]

    def __len__(self) -> int:
        return len(self.graphs)

def demonstrate_learning(models: Mapping[str, Module],
                         generator: Generator,
                         node_attrs: Sequence[str],
                         edge_attrs: Sequence[str], 
                         train_data_len: int = 1000,
                         epochs: int = 100,
                         test_data_len: int = 1000
                        ) -> None:
    accuracies: Mapping[str, List[float]] = {name: [] for name in models.keys()}
    batch_size = next(n for n in range(20, 0, -1) if (train_data_len % n) == 0 and (test_data_len % n) == 0)  # It's just 20.
    training_data = SyntheticDataset(generator=generator,
                                     node_attrs=node_attrs,
                                     edge_attrs=edge_attrs,
                                     size=train_data_len)
    training_loader = GraphDataLoader(training_data,
                                      sampler=RandomSampler(training_data),
                                      batch_size=batch_size,
                                      drop_last=False)
    test_data = SyntheticDataset(generator=generator,
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
    for (name, accuracy) in accuracies.items():
        plt.plot(accuracy, label=name)
    plt.legend()
    plt.show()

def the_dumbest_generator(arg1: bool, arg2: bool) -> DiGraph:
    nodes = [0, 0, 0, int(arg1), int(arg1), int(arg1), 1, 1, 1]
    g = DiGraph()
    for (n, c) in enumerate(nodes):
        g.add_node(n, c=c)
        #g.add_edge(n, n)
    for n1 in g:
        for n2 in g:
            if int(arg2) == g.nodes[n2]['c']:
                g.add_edge(n1, n2)
    return g

def extra_easy() -> None:
    model1 = Sequential(TAGConv(1, 4, k=0), AvgPooling())
    model2 = Sequential(TAGConv(1, 4, k=1), AvgPooling())
    model3 = Sequential(TAGConv(1, 4, k=2), AvgPooling())
    demonstrate_learning(models={'hopefully_75p': model1,
                                 'hopefully_100p': model2,
                                 'wtf': model3},
                         generator=the_dumbest_generator,
                         node_attrs=['c'],
                         edge_attrs=[], 
                         train_data_len=100,
                         epochs=50,
                         test_data_len=1000)

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
    with no_grad(): # skip computation of gradients
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

