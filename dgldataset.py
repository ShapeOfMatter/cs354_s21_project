import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd
import numpy as np

class SyntheticDataset(DGLDataset):
    def __init__(self,path):
        self.path = path
        super().__init__(name='synthetic')

    def process(self):
        data = pd.read_csv(self.path)
        data['graph_id'] = data['parent_index'].astype(str) + data['sample_num'].astype(str)  + data['parent_label']
        edges = data[['src','dst','graph_id','parent_label']]
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
            mapper = dict([(all_nodes[i],i) for i in range(len(all_nodes))])
            #num_nodes = 100
            label = graph_id[-3:]
            label = {'sch':0,'map':1}[label]
            src = np.vectorize(mapper.get)(src)
            dst = np.vectorize(mapper.get)(dst)

            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst))
            self.graphs.append(g)
            self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

dataset = SyntheticDataset('schoolnetJeffsNets10_True.csv')
graph, label = dataset[0]
print(graph, label)
