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
from tqdm import tqdm
import csv 

def get_rds(G, num_seeds=1, num_coupons=3, samp_size=100, keep_labels = False):
    N = G.number_of_nodes()
    nodes_in_samp = set()
    nodes_not_in_samp = set(range(N))
    coupon_holders = []
    G_samp = nx.DiGraph()
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
                G_samp.add_edge(recruiter, recruit, recruitment_edge=True)
                G_samp.nodes[recruiter]['recruits'] += 1

    # Now we want to add the rest of the edges
    for node in nodes_in_samp:
        neighbors = {n for n in G[node]} & nodes_in_samp #look at all a nodes neighbors but then only keep the ones who are also in the sample (& is intersection)
        
        non_recruitment_edges = []
        for neighbor in neighbors:
            if (node, neighbor) not in G_samp.edges():
                non_recruitment_edges.append( (node, neighbor, {'recruitment_edge':False}) )
                
        #G_samp.add_edges_from((node, neighbor) for neighbor in neighbors)
        G_samp.add_edges_from(non_recruitment_edges)

    return G_samp

def graph_sample_old(input_path, outdir, number_per, size):
    try:
        os.mkdir(outdir)
    except:
        pass
    
    input_folder = input_path + '/*.csv.gz'
    print(f"input folder: {input_folder}")
    infiles = [(i, f, outdir, number_per, size) for (i, f) in enumerate(glob.glob(input_folder))]
    
    
    
    
    
    print(f'STARTING POOL WITH {len(infiles)} TASKS')
    with Pool(2, maxtasksperchild=1) as p:
        p.map(graph_sample_helper, infiles)
  
def hacky_check(infile):
    years_to_avoid = [str(year) for year in range(2013, 2019)]
    for big_year in years_to_avoid:
        if big_year in infile:
            return True
    
      
#try a lighter version without pooling
def graph_sample(input_path, outdir, number_per, size):
    try:
        os.mkdir(outdir)
    except:
        pass
    
    input_folder = input_path + '/*.csv.gz'
    print(f"input folder: {input_folder}")
    #infiles = [(i, f, outdir, number_per, size) for (i, f) in enumerate(glob.glob(input_folder))]
    
    infiles = glob.glob(input_folder)
    
    for infile in tqdm(infiles):
    
        


        
        # if hacky_check(infile):
        #     print("skipping ", infile[-39:-7])
        #     continue
        
        #skip the big files and ones weve already done
        subdir = outdir + "/" + infile[-39:-7]  # TODO: actually parse the string.
        if os.path.exists(subdir) or hacky_check(infile):
            print("skipping ", infile[-39:-7])
        else:
            os.makedirs(subdir)
            print(infile[-39:-7])
            
            G = nx.DiGraph()
        
        
            
            
            with gzip.open(infile, 'rt') as csvfile:
                print(infile[-39:-7], "gzip opened")
                reader = csv.reader(csvfile, delimiter="	", quotechar='|') #did the delimiter copy right?
                reader.__next__() #ignore the header row. Whats the proper way to do this?
                for row in tqdm(reader):
                    G.add_edge(np.uint32(row[0]), np.uint32(row[2]))
            print("Done loading")    
            G = nx.convert_node_labels_to_integers(G, first_label=0) #TODO Try doing this in-place with ns.relabel_nodes(copy=True) or maybe try to relabel the text file before ever making it a network or something. There seems to be alot going on under the hood of relabel that could slow stuff down that we could avoid if we pre-relabel
            print("Done relabeling")
            
            
            for j in tqdm(range(number_per)):
                outpath = subdir + "/sample_" + str(j) + ".pkl"
                #print(f"About to make {outpath}")
                
                try:
                    G_samp = get_rds(G, samp_size = size)
                except Exception as e:
                    traceback.print_exc()
                    print(e)
                    return
                    
                #print(f"G_samp ({i},{j}) number of nodes: {G_samp.number_of_nodes()}")
                
                nx.write_gpickle(G_samp, outpath)
                    
            print("Done with ", infile)
            print("\n\n\n")

    
    
    

def graph_sample_helper(info):
    i, infile, outdir, number_per, num_nodes = info
    try:
        
        print(f"POOL-TASK {i} BEGINING.")
        columns_types = {
            'page_id_from': np.int64,
            'page_id_to': np.int64
        }
        # df = pd.read_csv(gzip.open(infile, 'rb'),
        #                  sep="	",
        #                  keep_default_na=False,
        #                  dtype = columns_types,
        #                  usecols = columns_types.keys())
        
        #Check there are no missing values
        # print(f"there are {df.isnull().sum().sum()} missing values in df created from {infile}")

        G = nx.DiGraph()
        
        with gzip.open(infile, 'rt') as csvfile:
            print("gzip opened")
            reader = csv.reader(csvfile, delimiter="	", quotechar='|') #did the delimiter copy right?
            reader.__next__() #ignore the header row. Whats the proper way to do this?
            for row in tqdm(reader):
                G.add_edge(np.uint32(row[0]), np.uint32(row[2]))
                #print((row[0], row[2]))

        
        #create networkx graph from two of the columns in the dataframe
        _G = nx.from_pandas_edgelist(df, source='page_id_from', target='page_id_to', create_using=nx.DiGraph())
        G = nx.convert_node_labels_to_integers(_G, first_label=0, ordering='sorted')
        print(f"number of nodes: {G.number_of_nodes()}")
        df=None #clear the dataframe once its been read from cuz it takes alot of memory
        
        subdir = outdir + "/" + infile[-39:-7]  # TODO: actually parse the string.
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        
        for j in tqdm(range(number_per)):
            outpath = subdir + "/sample_" + str(j) + ".pkl"
            #print(f"About to make {outpath}")
            
            try:
                G_samp = get_rds(G, samp_size = num_nodes)
            except Exception as e:
                traceback.print_exc()
                return
                
            #print(f"G_samp ({i},{j}) number of nodes: {G_samp.number_of_nodes()}")
            
            nx.write_gpickle(G_samp, outpath)
            #print(f"{outpath} should be made by now.")
        
    except Exception as e:
        traceback.print_exc()
        return
    print(f"POOL-TASK {i} ENDING.")

 
# TODO: take cli arguments and wrap this in a `main`

input_path = "/Users/samrosenblatt/Documents/UVM/Deep_Learning/cs354_s21_project/datasets/wikidata.nosync"
outdir = "/Users/samrosenblatt/Documents/UVM/Deep_Learning/cs354_s21_project/datasets/samples_wEdgeData.nosync"
number_per = 1000
size = 100
graph_sample(input_path, outdir, number_per, size)          
