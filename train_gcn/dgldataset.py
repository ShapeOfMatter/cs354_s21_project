import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import networkx as nx
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from typing import Sequence, Tuple

from train_gcn.state_classes import Settings

def true_subgraph(G, nodes_to_keep):
    G_sub = G.copy(as_view=False)
    G_sub.remove_nodes_from([n for n in G if n not in set(nodes_to_keep)])
    return G_sub

class SyntheticDataset(DGLDataset):
    def __init__(self, indices, master_dir, sub_graph_choices):
        self._indices = indices
        self._master_dir = master_dir
        self._sub_graph_choices = sub_graph_choices
        super().__init__(name='synthetic')

    def process(self):
        '''
        This whole function needs to be re-worked such that it can work with new data,
        and so that we can be confident if it works or not.
        '''
        self.graphs = []
        self.labels = []
        
        for label in ["med", "scl"]:
            print("starting ", label)
            for i in tqdm(self._indices):
                '''
                DO NOT REMOVE
                We arent using this now but this example(from tutorial) would work for when we include additional graph features we get using RDS estimators during sampling once that is implemented (stuff like estimated num_nodes, estimated mean degree, estimated transitivity, etc.)
                properties = pd.read_csv('./graph_properties.csv')
        
                # Create a graph for each graph ID from the edges table.
                # First process the properties table into two dictionaries with graph IDs as keys.
                # The label and number of nodes are values.
                label_dict = {}
                num_nodes_dict = {}
                for _, row in properties.iterrows():
                    label_dict[row['graph_id']] = row['label']
                    num_nodes_dict[row['graph_id']] = row['num_nodes']
                '''
                for j in self._sub_graph_choices:
                    path = f"{self._master_dir}/{label}/graph_{i}/sample_{j}.pkl"
                    G_nx = nx.read_gpickle(path)
                    for node in G_nx.nodes:
                        #TODO: The first possibility in this OR should never happen, but it does.That means there is a bug in graph2samp.py where some people dont get assigned 'recruiter'
                        if ('recruiter' not in G_nx.nodes[node].keys()) or G_nx.nodes[node]['recruiter'] == "None" :
                            G_nx.nodes[node]['is_seed'] = 1
                        else:
                            G_nx.nodes[node]['is_seed'] = 0

                    G_nx = true_subgraph(G_nx, max(nx.connected_components(G_nx), key=len) )
                    
                    num_nodes = G_nx.number_of_nodes()
                    directed_G_nx = G_nx.to_directed()  # Is the graph [supposed to be] directed?
                    #TODO: Implement "dist_from_seed" attr and include here:
                    g = dgl.from_networkx(directed_G_nx, node_attrs=["true_degree", "is_seed"])
                    # This does nothing?
                    [e for e in (g.edges()[1],g.edges()[0])]
                    #TODO: Double check that the above does indeed get read into dgl.graph as having bi-drected edges
                    g.ndata['attr'] = torch.cat((torch.reshape(g.ndata['true_degree'], (len(G_nx.nodes), 1)),
                                                 torch.reshape(g.ndata['is_seed'], (len(G_nx.nodes), 1))), 1)
                    self.graphs.append(g)
                    self.labels.append(label)
            print(f"Finished looping on {label}")
         
        # Convert the label list to tensor for saving.
        self.labels = [{'med':0,'scl':1}[l] for l in self.labels] # The labels need to be integers.
        self.labels = torch.LongTensor(self.labels)
        
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    @property
    def indices(self):
        return self._indices
    
    @property
    def master_dir(self):
        return self._master_dir
    
    @property
    def sub_graph_choices(self):
        return self._sub_graph_choices

        
def make_dataloader(use_indices: Sequence[int],  # used to split test/train data
                    sub_graph_choices: Sequence[int],  # typically used to reduce load for demo
                    max_batch_size: int,  # will choose a batch size to evenly divide the data
                    master_dir: str #the directory where the all the samples are stored
                    ) -> GraphDataLoader:
    # Why are reading/processing the dataset here? doesn't that mean we end up doing it twice?
    dataset = SyntheticDataset(indices=use_indices, master_dir=master_dir, sub_graph_choices=sub_graph_choices)
    print("len(dataset): ", len(dataset))
    batch_size = next(n for n in range(max_batch_size, 0, -1) if (len(dataset) % n) == 0)
    print(f'Using batch size {batch_size}.')
    sampler = SubsetRandomSampler(range(len(dataset)))  #TODO: Get a citation that this is a good choice.
    dataloader = GraphDataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=False)
    return dataloader

def get_train_test_dataloaders(settings: Settings) -> Tuple[GraphDataLoader, GraphDataLoader]:
    # This can probably be rebuilt from scratch for the new data.
    test_train_splitter = np.random.default_rng(seed=settings.deterministic_random_seed)
    train_indices = list(test_train_splitter.choice(np.arange(5000),size = 4000))  #numbers from old data
    test_indices = list(np.array(list(set(range(5000)) - set(train_indices))))
    train_dataloader = make_dataloader(master_dir="datasets/samples",
                                       use_indices=train_indices,
                                       sub_graph_choices=settings.sub_graph_choices,
                                       max_batch_size=settings.max_batch_size)
    test_dataloader = make_dataloader(master_dir="datasets/samples",
                                      use_indices=test_indices,
                                      sub_graph_choices=[0],
                                      max_batch_size=settings.max_batch_size)
    return train_dataloader, test_dataloader
    

