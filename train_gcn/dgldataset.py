import dgl
from dgl.data import DGLDataset
import torch
from typing import Sequence

from tqdm import tqdm
import networkx as nx

def true_subgraph(G, nodes_to_keep):
    G_sub = G.copy(as_view=False)
    G_sub.remove_nodes_from([n for n in G if n not in set(nodes_to_keep)])
    return G_sub

class SyntheticDataset(DGLDataset):
    def __init__(self, indices, master_dir, sub_graph_choices):
        #super().__init__(name='synthetic', indices=indices, master_dir=master_dir, sub_graph_choices=sub_graph_choices)
        self._indices = indices
        self._master_dir = master_dir
        self._sub_graph_choices = sub_graph_choices
        super().__init__(name='synthetic')

    def process(self):
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
                        if ('recruiter' not in G_nx.nodes[node].keys()) or G_nx.nodes[node]['recruiter'] == "None" : #TODO: The first possibility in this OR should never happen, but it does.That means there is a bug in graph2samp.py where some people dont get assigned 'recruiter'
                            G_nx.nodes[node]['is_seed'] = 1
                        else:
                            G_nx.nodes[node]['is_seed'] = 0
                            

                    
                    G_nx = true_subgraph(G_nx, max(nx.connected_components(G_nx), key=len) )
                    
                    num_nodes = G_nx.number_of_nodes()
                    directed_G_nx = G_nx.to_directed()                    
                    #directed_G_nx = max(nx.connected_components(directed_G_nx), key=len) 
                    
                    g = dgl.from_networkx(directed_G_nx, node_attrs=["true_degree", "is_seed"]) #TODO: Implement "dist_from_seed" attr and include here
                       
                    [e for e in (g.edges()[1],g.edges()[0])]
                    #TODO: Double check that the above does indeed get read into dgl.graph as having bi-drected edges
                    g.ndata['attr'] = torch.cat((torch.reshape(g.ndata['true_degree'],(len(G_nx.nodes),1)),torch.reshape(g.ndata['is_seed'],(len(G_nx.nodes),1))),1)

                    # Is the graph [supposed to be] directed? handle that.
                    
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

        