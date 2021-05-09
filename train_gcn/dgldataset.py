import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from sklearn import preprocessing
import networkx as nx
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from typing import Sequence, Tuple
import glob
from pathlib import Path


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

                    #G_nx = true_subgraph(G_nx, max(nx.connected_components(G_nx), key=len) )
                    
                    num_nodes = G_nx.number_of_nodes()
                    G_nx = G_nx.to_directed()  # Is the graph [supposed to be] directed?
                    #TODO: Implement "dist_from_seed" attr and include here:
                    edge_bidirection = [1 if G_nx.has_edge(v,u) else 0 for u,v in G_nx.edges]
                    i = 0
                    for edge in G_nx.edges:
                        G_nx.edges[edge]['is_bidirected'] = edge_bidirection[i]
                        i += 1
                    
                    g = dgl.from_networkx(G_nx, node_attrs=["true_degree", "is_seed"], edge_attrs=['is_bidirected'])
                    # This does nothing?
                    #[e for e in (g.edges()[1],g.edges()[0])]
                    #TODO: Double check that the above does indeed get read into dgl.graph as having bi-drected edges
                    g.ndata['attr'] = torch.cat((torch.reshape(g.ndata['true_degree'], (len(G_nx.nodes), 1)),
                                                 torch.reshape(g.ndata['is_seed'], (len(G_nx.nodes), 1))), 1)
                    g.edata['attr'] = torch.reshape(g.edata['is_bidirected'], (len(G_nx.edges),1))
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
    master_dir = "datasets/samples"
    
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
    

class WikiDatasets(DGLDataset):
    def __init__(self, paths, new_process = False):
        self.paths = paths
        self.new_process = new_process
        super().__init__(name = 'WikiDataset')
        
    def process(self):
        self.graphs = []
        self.full_labels = []
        self.years = []
        self.labels = []
        img_paths = self.paths
        for path in img_paths:
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
                    
                    # TODO: Investigate error below. 
                    # DGLError: There are 0-in-degree nodes in the graph, output for those nodes will be invalid. 
                    # This is harmful for some applications, causing silent performance regression. 
                    # Adding self-loop on the input graph by calling `g = dgl.add_self_loop(g)` will resolve the issue. 
                    # Setting ``allow_zero_in_degree`` to be `True` when constructing this module will suppress the 
                    # check and let the code run.
                    
                    # Currently addressed by adding self loop to networkx graphs
                    to_add = [(node,node) for node in G_nx.nodes]
                    G_nx.add_edges_from(to_add)
                    
                    original_edges = G_nx.edges
                    for edge in original_edges:
                        G_nx.edges[edge]['forward'] = True
                        reverse_edge = edge[1],edge[0]
                        if reverse_edge in G_nx.edges:
                            G_nx.edges[edge]['backward'] = True
                        else:
                            G_nx.edges[edge]['backward'] = False
                            G_nx.add_edge(reverse_edge[0],reverse_edge[1])
                            G_nx.edges[reverse_edge]['forward'] = False
                            G_nx.edges[reverse_edge]['backward'] = True
                    
                    # Using the code below denotes every edge as bidirectional
                    #for edge in G_nx.edges:
                    #    G_nx.edges[edge]['forward'] = True if edge in G_nx.in_edges else False
                    #    G_nx.edges[edge]['backward'] = True if edge in G_nx.out_edges else False
                    Path('\\'.join(path.replace('samples','processed').split('\\')[0:-1])).mkdir(parents=True, exist_ok=True)
                    nx.write_gpickle(G_nx, path.replace('samples','processed'))
                
                       
                # Should be digraph.
                assert(str(type(G_nx)) == "<class 'networkx.classes.digraph.DiGraph'>"), 'Graph needs to be digraph'
                
                g = dgl.from_networkx(G_nx, node_attrs=["true_degree","distance_to_seed"], edge_attrs=['forward','backward'])
                
                
                # Combine all node attributes into a large tensor. 
                node_tensors = [torch.reshape(g.ndata[key],(len(G_nx.nodes),1)) for key in g.ndata.keys()]
                g.ndata['attr'] = torch.cat(node_tensors,1)
                
                edge_tensors = [torch.reshape(g.edata[key],(len(G_nx.edges),1)) for key in g.edata.keys()]
                g.edata['attr'] = torch.cat(edge_tensors,1)
                
                
                # For relational conv. . Requires |E| length representing class of edge.
                g.edata['encode'] = g.edata['attr'].int()[:,0] * 2 + g.edata['attr'].int()[:,1] - 1
                
                full_label = path_info[2]
                label = full_label.split('.')[0]
                year = full_label.split('.')[-1][0:4]
                G_nx = nx.read_gpickle(path)
                
                self.graphs.append(g)
                self.full_labels.append(full_label)
                self.years.append(year)
                self.labels.append(label)
            except:
                print('Could not load graph from file',path)
                # Convert the label list to tensor for saving.
        
        int_labels = preprocessing.LabelEncoder().fit_transform(self.labels)
        #onehot_labels = preprocessing.OneHotEncoder(sparse = False).fit_transform(int_labels.reshape(len(int_labels),1))
        self.labels = torch.LongTensor(int_labels)
            
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

def get_dataloaders(master_dir = "datasets\samples", val_year = 2013, batch_size = 5): 
    #TODO: add master_dir, val_year, new_process to settings. 
    
    
    # Gather paths for each split
    img_paths = glob.glob(master_dir+'/*wiki*/*')
    img_paths_val = glob.glob(master_dir+'/*wiki*'+str(val_year)+'*/*')
    img_paths_train_test = [path for path in img_paths if path not in img_paths_val]
    # Shuffle test and train s.t. taking a slice gives a random sample.
    np.random.shuffle(img_paths_train_test)
    split = round(len(img_paths_train_test)* 0.8)
    img_paths_train = img_paths_train_test[0:split]
    img_paths_test = img_paths_train_test[split:]
    
    # Create the datasets using the appropriate path.
    new_process= True
    train_dataset = WikiDatasets(paths = img_paths_train, new_process=new_process)
    test_dataset = WikiDatasets(paths = img_paths_test, new_process=new_process)
    val_dataset = WikiDatasets(paths = img_paths_val, new_process=new_process)
    
    # Create the dataloaders. 
    training_loader = GraphDataLoader(train_dataset,
                                          sampler= SubsetRandomSampler(torch.arange(len(img_paths_train))),
                                          batch_size=batch_size,
                                          drop_last=False)
    testing_loader = GraphDataLoader(test_dataset,
                                          sampler=SubsetRandomSampler(torch.arange(len(img_paths_test))),
                                          batch_size=batch_size,
                                          drop_last=False)
    validation_loader = GraphDataLoader(val_dataset ,
                                          sampler=SubsetRandomSampler(torch.arange(len(img_paths_val))),
                                          batch_size=batch_size,
                                          drop_last=False)
    return training_loader, testing_loader, validation_loader

if __name__ == '__main__':
    get_dataloaders(master_dir = "..\datasets\samples")