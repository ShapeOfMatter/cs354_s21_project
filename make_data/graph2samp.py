#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: samrosenblatt
"""

import networkx as nx
import numpy as np
import pandas as pd
import random
import netwulf
import sys
import glob
import os
import shutil
# import dgl
# from dgl.dataloading import GraphDataLoader
# from torch.utils.data.sampler import SubsetRandomSampler
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
from multiprocessing import Pool
#from dgl.nn import GraphConv
from typing import Callable, Sequence
import gzip

#these next imports are from python programs I wrote 
from samp_utils import true_subgraph 
from graph_getters import read_our_csv
#from dgldataset import SyntheticDataset
#from state_classes import AdamTrainingProfile, Settings, TrainingProfile

#Global constants
from pandas._libs.parsers import STR_NA_VALUES
na_vals = STR_NA_VALUES - set(['NaN']) # There is a literal wikipedia page about nan values and this was interpreting that as missing valeus so we explicitly prevent that https://en.wikipedia.org/wiki/NaN


def get_rds(G, num_seeds=1, num_coupons=3, samp_size=100, keep_labels = False, only_rds_edges=True):
    
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

def graph_sample(input_path,outdir,number_per,only_rds,size):
  try:
      shutil.rmtree(outdir)
  except:
      pass
  os.mkdir(outdir)
  input_folder = input_path+'/*.csv.gz'
  print("input folder: ", input_folder)
  infiles = [(i,file, outdir,number_per,only_rds,size) for i,file in enumerate(glob.glob(input_folder))]

  print("ASSUMING YOUR INPUT NETWORKS ARE gzips of CSVs OF A SPECIFIC FORMAT")
  print("YOU LIKELY WANT TO CHANGE THIS READING IN PART")
  with Pool(4) as p: #limit to 4 processes so it doesnt crash my computer again
      p.map(graph_sample_helper,infiles)


def graph_sample_helper(info):
    #print("gettin some help")
    i,infile,outdir,number_per,only_rds,num_nodes = info[0],info[1],info[2],info[3],info[4],info[5]
    #G = nx.read_edgelist(infile, delimiter=',', nodetype=int, comments='V')
      

    df = pd.read_csv(gzip.open(infile, 'rb'), 
                 sep="	",
                na_values = na_vals,
                keep_default_na=False)
    
    #Check there are no missing values
    print("there are ", df.isnull().sum().sum(), " missing values in df created from ", infile)
    
    #create networkx graph from two of the columns in the dataframe
    G = nx.from_pandas_edgelist(df, source='page_id_from', target='page_id_to', create_using=nx.DiGraph())
    print("number of nodes: ", G.number_of_nodes())
    df=None #clear the dataframe once its been read from cuz it takes alot of memory
    subdir = outdir+"/"+infile[-39:-7]
      
    if not os.path.exists(subdir):
        os.makedirs(subdir)
        
    
    for j in range(number_per):
        outpath = subdir+"/sample_"+str(j)
        print("about to make ", outpath)
        G_samp = get_rds(G, only_rds_edges = only_rds,samp_size = num_nodes)
        print("G_sampe  number of nodes: ", G_samp.number_of_nodes())
        
        
        
        nx.write_gpickle(G_samp, outpath+".pkl")
        print(outpath, " should be made by now")

          
input_path = "/Users/samrosenblatt/Documents/UVM/Deep_Learning/cs354_s21_project/datasets/wikidata"
outdir = "/Users/samrosenblatt/Documents/UVM/Deep_Learning/cs354_s21_project/datasets/samples"
number_per = 10
only_rds = False
size = 100
graph_sample(input_path,outdir,number_per,only_rds,size)          