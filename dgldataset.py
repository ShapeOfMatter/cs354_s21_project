import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd
import numpy as np

# Class that allows DGL to access data. 
class SyntheticDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='synthetic')
        self.data = self._load()
        
    def _load(self):
        original_med = pd.read_csv('medoriginal.csv')
        original_school = pd.read_csv('schoolnetJeffsNetsoriginal.csv')
        samples_med = pd.read_csv('med10_True.csv')
        samples_school = pd.read_csv('schoolnetJeffsNets10_True.csv')
        data = pd.concat([original_med,original_school,samples_med,samples_school],ignore_index = True)
        return data
    def process(self):
        #data = self.data
        #if not any('sample_num' in s for s in [self.data.columns]):
        #    self.data['sample_num'] = 0
        self.data['graph_id'] = self.data['parent_index'].astype(str) + self.data['sample_num'].astype(str)  + self.data['parent_label']
        edges = self.data[['src','dst','graph_id','parent_label']]
        #properties = pd.read_csv('./graph_properties.csv')
        self.graphs = []
        self.labels = []

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        #label_dict = {}
        #num_nodes_dict = {}
        #for _, row in properties.iterrows():
        #    label_dict[row['graph_id']] = row['label']
         #   num_nodes_dict[row['graph_id']] = row['num_nodes']

        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby('graph_id')

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            all_nodes = sorted(list(set(list(src)+list(dst))))
            # DGL requires nodes labels are less than the number of nodes.
            # mapper function reassigns order
            mapper = dict([(all_nodes[i],i) for i in range(len(all_nodes))])
            #num_nodes = 100
            label = graph_id[-3:]
            label = {'sch':0,'med':1}[label]
            src = np.vectorize(mapper.get)(src)
            dst = np.vectorize(mapper.get)(dst)

            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst))
            g.add_edges(dst, src)
            #g = dgl.add_self_loop(g) # Fix for 0 degree nodes. TODO: find a better solution for this. Causes major slowdown.
            g.ndata['attr'] = torch.reshape(g.out_degrees(), (len(all_nodes),1))
            g.ndata['label'] = torch.reshape(torch.tensor([0]*len(all_nodes)), (len(all_nodes),1))
            self.graphs.append(g)
            self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)
    
    def partition(self,indices):
        self.data = self.data[self.data['parent_index'].isin(indices)]
        
    def select_samples(self,samples):
        self.data = self.data[self.data['sample_num'].isin(samples)]