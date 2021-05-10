import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from sklearn import preprocessing
import networkx as nx
import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
from typing import List, Sequence, Tuple
import glob
from pathlib import Path
import sys

from train_gcn.state_classes import Settings


class WikiDatasets(DGLDataset):
    def __init__(self, paths: Sequence[str], new_process = False):
        self.paths = paths
        self.new_process = new_process
        self.graphs: List[dgl.DGLGraph] = []
        self.node_attrs = ["true_degree", "distance_to_seed"]
        self.edge_attrs = ['forward_edge', 'backward_edge']
        super().__init__(name = 'WikiDataset')
        
    def process(self) -> None:
        labels = []
        for path in self.paths:
            path_info = path.replace('\\','/').split('/')
            if '..' in path_info:
                path_info.remove('..')
            try:
                try:
                    # Try to read a preprocessed graph.
                    assert(self.new_process == False), 'Raise an error since we are forcing new processing of the graphs'
                    G_nx = nx.read_gpickle(path.replace('samples','processed'))
                    
                except:
                    # Process graph from sample
                    G_nx = nx.read_gpickle(path) 
                    
                    # we need to add self-loops for the vanilla GCN. 
                    to_add = [(node,node) for node in G_nx.nodes]
                    G_nx.add_edges_from(to_add)
                    
                    original_edges = G_nx.edges
                    for edge in original_edges:
                        G_nx.edges[edge]['forward_edge'] = True
                        reverse_edge = edge[1],edge[0]
                        if reverse_edge in G_nx.edges:
                            G_nx.edges[edge]['backward_edge'] = True
                        else:
                            G_nx.edges[edge]['backward_edge'] = False
                            G_nx.add_edge(reverse_edge[0],reverse_edge[1])
                            G_nx.edges[reverse_edge]['forward_edge'] = False
                            G_nx.edges[reverse_edge]['backward_edge'] = True
                    
                    Path('\\'.join(path.replace('samples','processed').split('\\')[0:-1])).mkdir(parents=True, exist_ok=True)
                    nx.write_gpickle(G_nx, path.replace('samples','processed'))
                       
                # Should be digraph.
                assert(str(type(G_nx)) == "<class 'networkx.classes.digraph.DiGraph'>"), 'Graph needs to be digraph'
                
                g = dgl.from_networkx(G_nx, node_attrs=self.node_attrs, edge_attrs=self.edge_attrs)
                
                # Combine all node attributes into a large tensor. 
                node_tensors = [torch.reshape(g.ndata[key],(len(G_nx.nodes),1)) for key in g.ndata.keys()]
                g.ndata['attr'] = torch.cat(node_tensors,1)
                
                edge_tensors = [torch.reshape(g.edata[key],(len(G_nx.edges),1)) for key in g.edata.keys()]
                g.edata['attr'] = torch.cat(edge_tensors,1)
                
                # For relational conv. . Requires |E| length representing class of edge.
                g.edata['encode'] = g.edata['attr'].int()[:,0] * 2 + g.edata['attr'].int()[:,1] - 1
                
                full_label = path_info[2]
                label = full_label.split('.')[0]
                
                self.graphs.append(g)
                labels.append(label)
            except Exception as e:
                print('Could not load graph from file', path)
                raise e
        
        int_labels = preprocessing.LabelEncoder().fit_transform(labels)
        self.labels = torch.LongTensor(int_labels)
        
            
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    def num_labels(self) -> int:
        return len(self.labels.unique())
    

def get_dataloaders(settings: Settings) -> Tuple[GraphDataLoader, GraphDataLoader, GraphDataLoader]: 
    master_dir = settings.master_dir
    
    # Gather paths for each split
    img_paths = glob.glob(master_dir+'/*wiki*/*')
    img_paths_val = glob.glob(master_dir + '/*wiki*' + str(settings.val_year) + '*/*')
    img_paths_train_test = [path for path in img_paths if path not in img_paths_val]  # Are we doing 9k linear scans of a 1k-elem array?
    # Shuffle test and train s.t. taking a slice gives a random sample.
    np.random.shuffle(img_paths_train_test)
    split = round(len(img_paths_train_test) * 0.8)
    img_paths_train = img_paths_train_test[0:split]
    img_paths_test = img_paths_train_test[split:]
    
    if settings.reduce_datasize:
        img_paths_train = img_paths_train[0:1000]
        img_paths_test = img_paths_test[0:1000]
        img_paths_val = img_paths_val[0:1000]
    
    # Create the datasets using the appropriate path.
    train_dataset = WikiDatasets(paths = img_paths_train, new_process=settings.new_process)
    test_dataset = WikiDatasets(paths = img_paths_test, new_process=settings.new_process)
    val_dataset = WikiDatasets(paths = img_paths_val, new_process=settings.new_process)
    
    # Create the dataloaders.
    def batch_size(sample_size: int) -> int:
        divisors = (n for n in range(m, 0, -1) if (sample_size % n) == 0)
        return next(divisors)
    training_loader = GraphDataLoader(train_dataset,
                                      sampler=RandomSampler(train_dataset),
                                      batch_size=batch_size(len(train_dataset)),
                                      drop_last=False)
    testing_loader = GraphDataLoader(test_dataset,
                                     sampler=RandomSampler(test_dataset),
                                     batch_size=batch_size(len(test_dataset)),
                                     drop_last=False)
    validation_loader = GraphDataLoader(val_dataset ,
                                        sampler=RandomSampler(val_dataset),
                                        batch_size=batch_size(len(val_dataset)),
                                        drop_last=False)
    return training_loader, testing_loader, validation_loader

if __name__ == '__main__':
    get_dataloaders(Settings.load('../'+sys.argv[1]))

