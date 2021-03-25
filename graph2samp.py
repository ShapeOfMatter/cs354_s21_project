#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: samrosenblatt
"""

import networkx as nx
import numpy as np
import random
import netwulf
import sys
import glob
import os
import shutil
import dgl
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from multiprocessing import Pool
from dgl.nn import GraphConv

#these next imports are from python programs I wrote 
from samp_utils import true_subgraph 
from graph_getters import read_our_csv
from dgldataset import SyntheticDataset


def get_rds(G, num_seeds=5, num_coupons=3, samp_size=100, keep_labels = False, only_rds_edges=True):
    
    N=G.number_of_nodes()
    
    
    if keep_labels:
        print("Assuming the current labels are integers from 0 to N-1")
    
    if keep_labels == False:
        # doing sorted ordering so that we know for sure that the same labels always correspond to the same nodes,
        # I dont think we need that but just in case.
        # https://networkx.github.io/documentation/stable/reference/generated/networkx.relabel.convert_node_labels_to_integers.html#networkx.relabel.convert_node_labels_to_integers
        # if the network came with node labels before, make a node attribute "old_lable" and store them there.
        G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='sorted', label_attribute="old_label")
    
    seeds = np.random.choice(N, num_seeds, replace=False)
    
    coupon_holders = []
    for seed in seeds:
        # We want coupon_holders to have num_seeds copies of each seed.
        # This represents how many coupons have yet to be given out and who holds them.
        coupon_holders += num_seeds*[seed]
    
    #coupon_holders.shuffle() #TODO: we assume people give out coupons in a random order. What do other people assume?

    # technically RDS is supposed to be with replacement but in practice it usually isnt.
    # We might want to look into this further
    # https://journals.sagepub.com/doi/pdf/10.1177/0049124108318333?casa_token=H89MMuDxbAAAAAAA:-pxCEqeqN1WzA3Az6aZ1gTh_YRoAIF-4N8OTfugF58uM4ateInh3m733uX27CVkBX_CJCsHAJQ8
    nodes_in_samp = set(seeds)

    
    if only_rds_edges:
        G_samp = nx.Graph()
        # it would probably be faster to add all the true_degrees at the end by doing like degDictG = {n: d for n, d in G.degree()}
        # and then writing that in or sending that back seperately.
        G_samp.add_nodes_from([(seed, {"true_degree": G.degree(seed)}) for seed in seeds])

        # copy any other info from G
        for seed in seeds:
            for key, value in G.nodes[seed].items():
                G_samp.nodes[seed][key] = value #maybe we wanna add a prefix that this is info from before or something?
                #TODO: theres gotta be a faster way than this to copy over the relevant info using some subgraph mechanism
                #G_samp.nodes[seed]['recruiter'] = None
                # seeds have no recruiter. We can make this a string if it messes stuff up to have this None-valued
                G_samp.nodes[seed]['recruiter'] = "None"
               
        
        while len(nodes_in_samp) < samp_size:
            if len(coupon_holders) == 0: #if the referral chains die out
                # add more seeds
                potential_new_seeds = set(list(G.nodes)) - nodes_in_samp #get new seeds from the population not including those you have sampled
                seeds = np.random.choice(list(potential_new_seeds), num_seeds, replace=False)
                # it would probably be faster to add all the true_degrees at the end by doing like degDictG = {n: d for n, d in G.degree()}
                # and then writing that in or sending that baxk seperately
                G_samp.add_nodes_from([(seed, {"true_degree": G.degree(seed)}) for seed in seeds])
                nodes_in_samp = nodes_in_samp | set(seeds) #set union
                for seed in seeds:
                    # we want coupon_holders to have num_seeds copies of each seed.
                    # This represents how many coupons have yet to be given out and who holds them
                    coupon_holders += num_seeds*[seed]
                    for key, value in G.nodes[seed].items():
                        G_samp.nodes[seed][key] = value #maybe we wanna add a prefix that this is info from before or something?
                        #TODO: theres gotta be a faster way than this to copy over the relevant info using some subgraph mechanism
                        #G_samp.nodes[seed]['recruiter'] = None 
                        # seeds have no recruiter. We can make this a string if it messes stuff up to have this None-valued
                        G_samp.nodes[seed]['recruiter'] = "None"
            
            recruiter = np.random.choice(coupon_holders)
            neighbors =  [n for n in G[recruiter]] #We know all of egos neighbors
            
            if 'recruits' not in G_samp.nodes[recruiter].keys():
                G_samp.nodes[recruiter]['recruits'] = []
            
            # So we are going to let them try to add people even if they are already in the sample because it seems thats how its done irl
            
            recruit = np.random.choice(neighbors) #could be slightly faster if i pick the index using randint and then select from that
            if recruit not in nodes_in_samp:
                nodes_in_samp.add(recruit)
                #G_samp.add_node(recruit)
                G_samp.add_edge(recruiter, recruit, edge_type='recruited')
                G_samp.nodes[recruit]['recruits'] = []
                G_samp.nodes[recruit]['recruiter'] = recruiter
                
                G_samp.nodes[recruit]["true_degree"] = len(neighbors)
                G_samp.nodes[recruiter]['recruits'].append(recruit)
                #copy any other info from G
                for key, value in G.nodes[recruit].items():
                    G_samp.nodes[recruit][key] = value #maybe we wanna add a prefix that this is info from before or something?
                    #TODO: theres gotta be a faster way than this to copy over the relevant info using some subgraph mechanism
                
                coupon_holders += num_coupons*[recruit]
            
        #TODO: if coupon_holders is empty and we havent reached the sample size we need to get more seeds,
        # https://www.nature.com/articles/s41598-020-63269-0
            
    if only_rds_edges == False:
        # technically RDS is supposed to be with replacement but in practice it usually isnt.
        # We might want to look into this further
        # https://journals.sagepub.com/doi/pdf/10.1177/0049124108318333?casa_token=H89MMuDxbAAAAAAA:-pxCEqeqN1WzA3Az6aZ1gTh_YRoAIF-4N8OTfugF58uM4ateInh3m733uX27CVkBX_CJCsHAJQ8
        G_samp = nx.Graph()
        # it would probably be faster to add all the true_degrees at the end by doing like degDictG = {n: d for n, d in G.degree()}
        # and then writing that in or sending that baxk seperately
        G_samp.add_nodes_from([(seed, {"true_degree": G.degree(seed)}) for seed in seeds])

        #copy any other info from G

        for seed in seeds:
            for key, value in G.nodes[seed].items():
                G_samp.nodes[seed][key] = value #maybe we wanna add a prefix that this is info from before or something?
                #TODO: theres gotta be a faster way than this to copy over the relevant info using some subgraph mechanism
                #G_samp.nodes[seed]['recruiter'] = None # seeds have no recruiter.
                # We can make this a string if it messes stuff up to have this None-valued
                G_samp.nodes[seed]['recruiter'] = "None"
        
        while len(nodes_in_samp) < samp_size:
            
            if len(coupon_holders) == 0: #if the referral chains die out
                # add more seeds
                potential_new_seeds = set(list(G.nodes)) - nodes_in_samp #get new seeds from the population not including those you have sampled
                seeds = np.random.choice(list(potential_new_seeds), num_seeds, replace=False)
                # it would probably be faster to add all the true_degrees at the end by doing like degDictG = {n: d for n, d in G.degree()}
                # and then writing that in or sending that baxk seperately
                G_samp.add_nodes_from([(seed, {"true_degree": G.degree(seed)}) for seed in seeds])
                nodes_in_samp = nodes_in_samp | set(seeds) #set union

                for seed in seeds:
                    # we want coupon_holders to have num_seeds copies of each seed.
                    # This represents how many coupons have yet to be given out and who holds them
                    coupon_holders += num_seeds*[seed]
                    for key, value in G.nodes[seed].items():
                        G_samp.nodes[seed][key] = value #maybe we wanna add a prefix that this is info from before or something?
                        #TODO: theres gotta be a faster way than this to copy over the relevant info using some subgraph mechanism
                        # seeds have no recruiter. We can make this a string if it messes stuff up to have this None-valued
                        G_samp.nodes[seed]['recruiter'] = "None"

            recruiter = np.random.choice(coupon_holders)
            coupon_holders.remove(recruiter)
            neighbors =  [n for n in G[recruiter]] #We know all of egos neighbors
            
            if 'recruits' not in G_samp.nodes[recruiter]:
                G_samp.nodes[recruiter]['recruits'] = []
            
            # So we are going to let them try to add people even if they are already in the sample because it seems thats how its done irl
            
            recruit = np.random.choice(neighbors) #could be slightly faster if i pick the index using randint and then select from that
            if recruit not in nodes_in_samp:
                nodes_in_samp.add(recruit)
                G_samp.add_edge(recruiter, recruit, edge_type='recruited')
                G_samp.nodes[recruit]['recruits'] = []
                
                G_samp.nodes[recruit]["true_degree"] = len(neighbors)
                G_samp.nodes[recruiter]['recruits'].append(recruit)
                #copy any other info from G
                for key, value in G.nodes[recruit].items():
                    G_samp.nodes[recruit][key] = value #maybe we wanna add a prefix that this is info from before or something?
                    #TODO: theres gotta be a faster way than this to copy over the relevant info using some subgraph mechanism
                
                coupon_holders += num_coupons*[recruit]
    
        # Now we want to add the rest of the edges
        # This is really janky, if we end up using this module instead of some elses this could def be sped up
        G_sub = true_subgraph(G, nodes_in_samp)
        
        edges_to_add = set(G_sub.edges())-set(G_samp.edges())
        
        G_samp.add_edges_from(edges_to_add)
    
    
    return G_samp

def store_net(folderpath, name):
    pass

def graph_sample(path,number_per,only_rds):

  # commented code can be uncommented to have pkl outputs. 
  
  #try:
  #    shutil.rmtree(path+'//outputs'+str(number_per)+'_'+str(only_rds))
  #except:
  #    pass
  #os.mkdir(path+'//outputs'+str(number_per)+'_'+str(only_rds))
  
  
  input_folder = path+'//original//*.csv'
  #output_folder = path+'//outputs'+str(number_per)+'_'+str(only_rds)
  print("ASSUMING YOUR INPUT NETWORKS ARE CSVs OF A SPECIFIC FORMAT")
  print("YOU LIKELY WANT TO CHANGE THIS READING IN PART")
  
  data = pd.DataFrame(columns = ['src','dst','parent_index','sample_num','parent_label'])
  with Pool() as p:
      outputs = p.map(graph_sample_helper,[(number_per,only_rds,g) for g in glob.glob(input_folder)])
  data = pd.concat(outputs,ignore_index = True)
 # for infile in glob.glob(input_folder):
      #filename = infile[len(input_folder):]

  data.to_csv(path+str(number_per)+'_'+str(only_rds)+'.csv')
  
def graph_sample_helper(args):
    number_per = args[0]
    only_rds = args[1]
    infile = args[2]
    data = pd.DataFrame(columns = ['src','dst','parent_index','sample_num','parent_label'])
    # the comments ='V' simply ignores the headers in the csv, otherwise they would be counted as edges
    G = nx.read_edgelist(infile, delimiter=',', nodetype=int, comments='V')
    for i in range(number_per):
          G_samp = get_rds(G, only_rds_edges = only_rds)
          #nx.write_gpickle(G_samp, output_folder+"/"+filename+'_'+str(i)+".pkl") 
          temp_data = pd.DataFrame(nx.to_edgelist(G_samp))
          temp_data[2] =  ''.join([str(c) for c in infile if c.isdigit()])
          temp_data[3] = i
          temp_data[4] = infile[0:3]
          temp_data.columns = ['src','dst','parent_index','sample_num','parent_label']
          data = pd.concat([data,temp_data],ignore_index=True)
    return data
  
def convert_original(path):
    input_folder = path+'//original//*.csv'
    #output_folder = path+'//outputs'+str(number_per)+'_'+str(only_rds)
    print("ASSUMING YOUR INPUT NETWORKS ARE CSVs OF A SPECIFIC FORMAT")
    print("YOU LIKELY WANT TO CHANGE THIS READING IN PART")
    
    data = pd.DataFrame(columns = ['src','dst','parent_index','sample_num','parent_label'])
    with Pool() as p:
        outputs = p.map(convert_original_helper,glob.glob(input_folder))
    data = pd.concat(outputs,ignore_index=True)
    data.to_csv(path+'original.csv')
    
def convert_original_helper(infile):
    # the comments ='V' simply ignores the headers in the csv, otherwise they would be counted as edges
    G = nx.read_edgelist(infile, delimiter=',', nodetype=int, comments='V')
    temp_data = pd.DataFrame(nx.to_edgelist(G))
    temp_data[2] =  ''.join([str(c) for c in infile if c.isdigit()])
    temp_data[3] = 'original'
    temp_data[4] = infile[0:3]
    temp_data.columns = ['src','dst','parent_index','sample_num','parent_label']
    return temp_data

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
    
def main():
    new = False
    if new:
        # Convert networks into DGL format. Original's are entire networks.
        convert_original('schoolnetJeffsNets')
        convert_original('med')
        # Pulls N samples. N = 10 
        graph_sample('schoolnetJeffsNets', 10, False)
        graph_sample('med', 10, False)
        
    # Assign test and train indicies. Note that there are 5000 files in each.
    # These will remain static for all three cases. 
    np.random.seed(seed=None)
    train_indices = np.random.choice(np.arange(5000),size = 4000)
    test_indices = np.array(list(set(range(5000))- set(train_indices)))
    
    # Case where whole graph is used to train
    dataset_train = SyntheticDataset()
    dataset_train.partition(train_indices)
    dataset_train.select_samples([0])
    dataset_train.process()
    
    
    dataset_test = SyntheticDataset()
    dataset_test.partition(test_indices)
    dataset_test.select_samples(list(range(10)))
    dataset_test.process()
    
    train_sampler = SubsetRandomSampler(torch.arange(4000))
    test_sampler = SubsetRandomSampler(torch.arange(1000))
    
    train_dataloader = GraphDataLoader(dataset_train, sampler=train_sampler, batch_size=5, drop_last=False)
    test_dataloader = GraphDataLoader(dataset_test, sampler=test_sampler,  batch_size=5, drop_last=False)
    
    model = GCN(1, 4, 2) #dim of node data, conv filter size, number of classes.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(50):
        for batched_graph, labels in train_dataloader:
            pred = model(batched_graph, batched_graph.ndata['attr'].float())
            loss = F.cross_entropy(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        num_correct = 0
        num_tests = 0
        for batched_graph, labels in test_dataloader:
            pred = model(batched_graph, batched_graph.ndata['attr'].float())
            #print(labels,pred)
            num_correct += (pred.argmax(1) == labels).sum().item()
            num_tests += len(labels)
    
        print('Test accuracy:', num_correct / num_tests)
    
        # TODO, save model every epoch, accuracy, either overwrite or reload old model.
    
    # Case where 1 sample is used to train
    
    # case where 100 samples are used to train

  
if __name__ == "__main__":
    main()
    
    

  
