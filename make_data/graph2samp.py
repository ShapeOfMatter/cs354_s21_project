#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import gzip
from multiprocessing import Pool
import networkx as nx
import numpy as np
import os
import pandas as pd
from pandas._libs.parsers import STR_NA_VALUES
import shutil
import traceback

#Global constants
na_vals = STR_NA_VALUES - set(['NaN']) # There is a literal wikipedia page about nan values and this was interpreting that as missing valeus so we explicitly prevent that https://en.wikipedia.org/wiki/NaN

def true_subgraph(G, nodes_to_keep):  # This looks slow...
    G_sub = G.copy(as_view=False)
    G_sub.remove_nodes_from([n for n in G if n not in set(nodes_to_keep)])
    return G_sub

def get_rds(G, num_seeds=1, num_coupons=3, samp_size=100, keep_labels = False):
    N = G.number_of_nodes()
    seeds = np.random.choice(N, num_seeds, replace=False)
    coupon_holders = []
    for seed in seeds:
        # We want coupon_holders to have num_seeds copies of each seed.
        # This represents how many coupons have yet to be given out and who holds them.
        coupon_holders += num_seeds*[seed]
    
    #MAYBE: we assume people give out coupons in a random order. What do other people assume?

    # technically RDS is supposed to be with replacement but in practice it usually isnt.
    # We might want to look into this further
    # https://journals.sagepub.com/doi/pdf/10.1177/0049124108318333?casa_token=H89MMuDxbAAAAAAA:-pxCEqeqN1WzA3Az6aZ1gTh_YRoAIF-4N8OTfugF58uM4ateInh3m733uX27CVkBX_CJCsHAJQ8
    nodes_in_samp = set(seeds)
    
    G_samp = nx.Graph()
    # it would probably be faster to add all the true_degrees at the end by doing like degDictG = {n: d for n, d in G.degree()}
    # and then writing that in or sending that baxk seperately
    G_samp.add_nodes_from([(seed, {"true_degree": G.degree(seed)}) for seed in seeds])

    #copy any other info from G

    for seed in seeds:
        for key, value in G.nodes[seed].items():
            G_samp.nodes[seed][key] = value #maybe we wanna add a prefix that this is info from before or something?
            #TODO: theres gotta be a faster way than this to copy over the relevant info using some subgraph mechanism
            # We can make this a string if it messes stuff up to have this None-valued
            G_samp.nodes[seed]['recruiter'] = "None" # seeds have no recruiter.
    
    while len(nodes_in_samp) < samp_size:
        if len(coupon_holders) == 0: #if the referral chains die out
            # add more seeds
            potential_new_seeds = set(list(G.nodes)) - nodes_in_samp #get new seeds from the population not including those you have sampled
            seeds = np.random.choice(list(potential_new_seeds), num_seeds, replace=False)
            G_samp.add_nodes_from([(seed, {"true_degree": G.degree(seed)}) for seed in seeds])
            nodes_in_samp = nodes_in_samp | set(seeds) #set union

            for seed in seeds:
                coupon_holders += num_seeds*[seed]
                for key, value in G.nodes[seed].items():
                    G_samp.nodes[seed][key] = value #maybe we wanna add a prefix that this is info from before or something?
                    G_samp.nodes[seed]['recruiter'] = "None"

        recruiter = np.random.choice(coupon_holders)
        coupon_holders.remove(recruiter)
        neighbors =  [n for n in G[recruiter]] #We know all of egos neighbors
        
        if 'recruits' not in G_samp.nodes[recruiter]:  # TODO: do this in advance.
            G_samp.nodes[recruiter]['recruits'] = []
        
        # So we are going to let them try to add people even if they are already in the sample because it seems thats how its done irl
        if neighbors:
            recruit = np.random.choice(neighbors)
            if recruit not in nodes_in_samp:
                nodes_in_samp.add(recruit)
                G_samp.add_edge(recruiter, recruit, edge_type='recruited')
                G_samp.nodes[recruit]['recruits'] = []
                
                G_samp.nodes[recruit]["true_degree"] = len(neighbors)  # TODO: THIS IS PROBABLY WRONG?!
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
  with Pool(2, maxtasksperchild=1) as p: #limit to 2 processes so it doesnt crash my computer again
      p.map(graph_sample_helper,infiles)


def graph_sample_helper(info):
    #print("gettin some help")
    print("\n")
    i,infile,outdir,number_per,only_rds,num_nodes = info[0],info[1],info[2],info[3],info[4],info[5]
    #G = nx.read_edgelist(infile, delimiter=',', nodetype=int, comments='V')
      

    df = pd.read_csv(gzip.open(infile, 'rb'), 
                 sep="	",
                na_values = na_vals,
                keep_default_na=False,
                dtype = {'page_id_from':np.int64, 'page_id_to':np.int64},
                usecols = ['page_id_from', 'page_id_to'])
    
    #Check there are no missing values
    print("there are ", df.isnull().sum().sum(), " missing values in df created from ", infile)
    
    #create networkx graph from two of the columns in the dataframe
    G = nx.from_pandas_edgelist(df, source='page_id_from', target='page_id_to', create_using=nx.DiGraph())
    print("number of nodes: ", G.number_of_nodes())
    df=None #clear the dataframe once its been read from cuz it takes alot of memory
    subdir = outdir+"/"+infile[-39:-7]
      
    if not os.path.exists(subdir):
        os.makedirs(subdir)
        
    
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='sorted')

    
    
    for j in range(number_per):
        print("\n")
        outpath = subdir+"/sample_"+str(j)
        print("about to make ", outpath)
        
        try:
            G_samp = get_rds(G, only_rds_edges = only_rds,samp_size = num_nodes)
        except Exception as e:
            #print(e)
            traceback.print_exc()
            
        print("G_sampe  number of nodes: ", G_samp.number_of_nodes())
        
        
        
        nx.write_gpickle(G_samp, outpath+".pkl")
        print(outpath, " should be made by now")

          
input_path = "/Users/samrosenblatt/Documents/UVM/Deep_Learning/cs354_s21_project/datasets/wikidata/dutch_small"
outdir = "/Users/samrosenblatt/Documents/UVM/Deep_Learning/cs354_s21_project/datasets/samples"
number_per = 10
only_rds = False
size = 100
graph_sample(input_path,outdir,number_per,only_rds,size)          
