#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import gzip
from multiprocessing import Pool
import networkx as nx
import numpy as np
import os
import pandas as pd
import shutil
import traceback

def get_rds(G, num_seeds=1, num_coupons=3, samp_size=100, keep_labels = False):
    N = G.number_of_nodes()
    nodes_in_samp = set()
    nodes_not_in_samp = set(range(N))
    coupon_holders = []
    G_samp = nx.Graph()
    #MAYBE: we assume people give out coupons in a random order. What do other people assume?

    def get_seeds():
        # technically RDS is supposed to be with replacement but in practice it usually isnt.
        # We might want to look into this further
        # https://journals.sagepub.com/doi/pdf/10.1177/0049124108318333?casa_token=H89MMuDxbAAAAAAA:-pxCEqeqN1WzA3Az6aZ1gTh_YRoAIF-4N8OTfugF58uM4ateInh3m733uX27CVkBX_CJCsHAJQ8
        return np.random.choice(list(nodes_not_in_samp), num_seeds, replace=False)

    def add_nodes_to_samp(nodes, distance_to_seed):
        nodes_in_samp.update(nodes)
        nodes_not_in_samp.difference_update(nodes)
        # We want coupon_holders to have num_seeds copies of each seed.
        # This represents how many coupons have yet to be given out and who holds them.
        coupon_holders.extend(n for n in nodes for _ in range(num_coupons))
        # it would probably be faster to add all the true_degrees at the end by doing like degDictG = {n: d for n, d in G.degree()}
        # and then writing that in or sending that baxk seperately
        G_samp.add_nodes_from([(n, dict(G.nodes[n], # maybe we wanna add a prefix that this is info from before or something?
                                        true_degree=G.degree(n),
                                        distance_to_seed=distance_to_seed,
                                        recruits=0))
                               for n in nodes])
        
    seeds = get_seeds()
    add_nodes_to_samp(seeds, 0)

    while len(nodes_in_samp) < samp_size:
        if len(coupon_holders) == 0:  # if the referral chains die out add more seeds
            seeds = get_seeds()
            add_nodes_to_samp(seeds, 0)

        recruiter = np.random.choice(coupon_holders)
        coupon_holders.remove(recruiter)
        neighbors = [n for n in G[recruiter]] #We know all of egos neighbors
        
        if neighbors:
            # So we are going to let them try to add people even if they are already in the sample because it seems thats how its done irl
            recruit = np.random.choice(neighbors)
            if recruit not in nodes_in_samp:
                # was previously setting true-degree as degree of parent/recrutier.
                add_nodes_to_samp((recruit,), G_samp.nodes[recruiter]['distance_to_seed'] + 1)
                G_samp.nodes[recruiter]['recruits'] += 1

    # Now we want to add the rest of the edges
    for node in nodes_in_samp:
        neighbors = {n for n in G[node]} & nodes_in_samp
        G_samp.add_edges_from((node, neighbor) for neighbor in neighbors)

    return G_samp

def graph_sample(input_path, outdir, number_per, size):
  try:
      shutil.rmtree(outdir)
  except:
      pass
  os.mkdir(outdir)
  input_folder = input_path + '/*.csv.gz'
  print(f"input folder: {input_folder}")
  infiles = [(i, f, outdir, number_per, size) for (i, f) in enumerate(glob.glob(input_folder))]

  print(f'STARTING POOL WITH {len(infiles)} TASKS')
  with Pool(2, maxtasksperchild=1) as p:
      p.map(graph_sample_helper, infiles)


def graph_sample_helper(info):
    i, infile, outdir, number_per, num_nodes = info
    print(f"POOL-TASK {i} BEGINING.")
    columns_types = {
        'page_id_from': np.int64,
        'page_id_to': np.int64
    }
    df = pd.read_csv(gzip.open(infile, 'rb'),
                     sep="	",
                     keep_default_na=False,
                     dtype = columns_types,
                     usecols = columns_types.keys())
    
    #Check there are no missing values
    print(f"there are {df.isnull().sum().sum()} missing values in df created from {infile}")
    
    #create networkx graph from two of the columns in the dataframe
    _G = nx.from_pandas_edgelist(df, source='page_id_from', target='page_id_to', create_using=nx.DiGraph())
    G = nx.convert_node_labels_to_integers(_G, first_label=0, ordering='sorted')
    print(f"number of nodes: {G.number_of_nodes()}")
    df=None #clear the dataframe once its been read from cuz it takes alot of memory
    
    subdir = outdir + "/" + infile[-39:-7]  # TODO: actually parse the string.
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    
    for j in range(number_per):
        outpath = subdir + "/sample_" + str(j) + ".pkl"
        print(f"About to make {outpath}")
        
        try:
            G_samp = get_rds(G, samp_size = num_nodes)
        except Exception as e:
            traceback.print_exc()
            return
            
        print(f"G_samp ({i},{j}) number of nodes: {G_samp.number_of_nodes()}")
        
        nx.write_gpickle(G_samp, outpath)
        print(f"{outpath} should be made by now.")

 
# TODO: take cli arguments and wrap this in a `main`

input_path = "datasets/wikidata/dutch_small"
outdir = "datasets/samples"
number_per = 10
size = 100
graph_sample(input_path, outdir, number_per, size)          
