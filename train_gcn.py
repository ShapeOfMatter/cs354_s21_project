#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import dgl
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from typing import Callable, Sequence
import matplotlib.pyplot as plt
import os
import networkx as nx
import glob
#from tqdm import tqdm # Why'd you get rid of my loader bar :'(

from dgldataset import SyntheticDataset
from state_classes import AdamTrainingProfile, ModelState, Settings, State, TrainingProfile
from graph2samp import graph_sample


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
    
def make_dataloader(#source_csvs: Sequence[str],  # filenames
                    use_indices: Sequence[int],  # typically used to split test/train data
                    sub_graph_choices: Sequence[int],  # typically used to reduce load for demo
                    max_batch_size: int,  # will choose a batch size to evenly divide the data
                    master_dir: str #the directory where the all the samples are stored (in appropriately organized and labeled nested subdirectories)
                    ) -> GraphDataLoader:
    dataset = SyntheticDataset(indices=use_indices, master_dir=master_dir, sub_graph_choices=sub_graph_choices)
    #dataset.process(use_indices=use_indices, master_dir=master_dir, sub_graph_choices=sub_graph_choices) #I am guessing the better way to do this would be to have use_indices and master_dir as SyntheticDataset() attributes initialized in the line above but idk how to write python classes
    
    #NOTE: Based on some print statement stuff i was doing for other reasons I found that it seems like process() gets called upon initialization so we do not need to run it again here
    #dataset.process()
    #dataset.select_samples(sub_graph_choices)
    #dataset.process()
    print("len(dataset): ", len(dataset))
    batch_size = next(n for n in range(max_batch_size, 0, -1) if (len(dataset) % n) == 0)
    print(f'Using batch size {batch_size}.')
    sampler = SubsetRandomSampler(torch.arange(len(dataset)))  #Mako: I may have misunderstood? #Sam: Are we sure len(dataset) is the length of the number of graphs in it?
    dataloader = GraphDataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=False) #If stuff never gets better we should check if we really understand how this is working
    return dataloader

def make_optimizer(profile: TrainingProfile, model: GCN) -> torch.optim.Optimizer:
    if isinstance(profile, AdamTrainingProfile):
        return torch.optim.Adam(model.parameters(),
                                lr=profile.learning_rate,
                                betas=(profile.beta_1, profile.beta_2),
                                eps=profile.epsilon,
                                weight_decay=profile.weight_decay,
                                amsgrad=profile.ams_gradient_variant)
    else:
        raise Exception("We haven't implemented any other optimizers.")

def main(settings: Settings):
    #print(dir(settings))
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Fix for crashing.
    if settings.new_data:
        # Pulls N samples. N = 10 
         
          # Label, data directory, outdirectory, number per, only RDS, size of samples
          graph_sample('med','med','datasets//samples//med', settings.num_samples, False, settings.sample_size)
          graph_sample('scl','schoolnetJeffsNets','datasets//samples//scl', settings.num_samples, False, settings.sample_size)
    
    print("settings.max_batch_size: ", settings.max_batch_size)
    
    # Assign test and train indicies. Note that there are 5000 files in each.
    test_train_splitter = np.random.default_rng(seed=settings.deterministic_random_seed)
    train_indices = test_train_splitter.choice(np.arange(5000),size = 4000)
    test_indices = np.array(list(set(range(5000)) - set(train_indices)))
    train_dataloader = make_dataloader(master_dir="datasets/samples", # using settings seemed simple but some weird stuff was going on so i just hardcoded for now
                                       use_indices=train_indices,
                                       sub_graph_choices=settings.sub_graph_choices,
                                       max_batch_size=settings.max_batch_size)
    test_dataloader = make_dataloader(master_dir="datasets/samples",
                                      use_indices=test_indices,
                                      sub_graph_choices=[0],
                                      max_batch_size=settings.max_batch_size)
    
    model = GCN(2, 4, 2) #dim of node data, conv filter size, number of classes.
    optimizer = make_optimizer(settings.training_profile, model)
    
    if True:
        try:
            rmdir = glob.glob('sub_graph_classifier_*.pt')
            for d in rmdir:
                os.remove(d)
            os.remove('subgraph_training.state')
        except:
            pass
    train_accuracy = []
    test_accuracy = []
    for epoch_number in range(20):
        state = State.load(settings.state_file)
        if state.models:
            # Depending if we're doing stuff in parallele, maybe we shouldn't reload every time?
            use_weights = state.models[0]  # for now there will be only one.
            model.load_state_dict(torch.load(use_weights.saved_in), strict=True)
            print(f'  Loaded model with accuracy {use_weights.accuracy} from {use_weights.saved_in}')
        train_accuracy_epoch, new_accuracy = epoch(f'e_{epoch_number}',
                             training_data=train_dataloader,
                             testing_data=test_dataloader,
                             model=model,
                             optimizer=optimizer,
                             criterion=F.cross_entropy)
        test_accuracy.append(new_accuracy)
        train_accuracy.append(train_accuracy_epoch)
        if new_accuracy >= min((m.accuracy for m in state.models), default=0):
            old_weights = (
                state.models.pop(0)  # or some other index if we're using more than one.
                 if state.models
                 else ModelState(accuracy=0, saved_in=settings.model_name_format.format(settings.sample_size))
            )
            print(f'  Overwriting weights (accuracy={old_weights.accuracy}).')
            torch.save(model.state_dict(), old_weights.saved_in)
            state.models.append(ModelState(accuracy=new_accuracy, saved_in=old_weights.saved_in))
            state.save(settings.state_file)
        else:
            print(f'  New weights (accuracy={new_accuracy}) will be discarded.')
    plt.plot(train_accuracy,label = 'training accuracy')
    plt.plot(test_accuracy,label = 'validation accuracy')
    plt.legend(loc = 'best')
    plt.xlabel('Epoch')
    #plt.ylim(0,1)
    plt.ylabel('Accuracy')
    plt.title('Accuracy with sample size '+ str(settings.sample_size))
    plt.show()
    graphs = [nx.read_gpickle('datasets//samples//med//graph_0//sample_0.pkl'),nx.read_gpickle('datasets//samples//scl//graph_0//sample_0.pkl')]
    titles = ['Colorodo Spring drug users','High school friendships']
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes.flatten()
    #fig.suptitle('Sample networks with sample size '+ str(settings.sample_size))
    
    for i in range(2):
        nx.draw_networkx(graphs[i], ax=ax[i],with_labels = False,alpha = 0.5, node_size = 100)
        #ax[i].set_axis_off()
        ax[i].set_title(titles[i])
    
    plt.show()
        
def epoch(name: str,
          training_data: GraphDataLoader,
          testing_data: GraphDataLoader,
          model: GCN,
          optimizer: torch.optim.Optimizer,
          criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
          ) -> float:  # returns the accuracy.
    
    for batched_graph, labels in training_data:
        pred = model(batched_graph, batched_graph.ndata['attr'].float())
        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test(batched_graph, labels):
        pred = model(batched_graph, batched_graph.ndata['attr'].float())
        return (pred.argmax(1) == labels).sum().item()
    
    num_correct_train = sum(test(batched_graph, labels)
                      for batched_graph, labels in training_data)
    num_tests_train = sum(len(labels)  # There may be an even more direct way to get this.
                    for batched_graph, labels in training_data)
    accuracy_train = num_correct_train / num_tests_train

    num_correct_test = sum(test(batched_graph, labels)
                      for batched_graph, labels in testing_data)
    num_tests_test = sum(len(labels)  # There may be an even more direct way to get this.
                    for batched_graph, labels in testing_data)
    accuracy_test = num_correct_test / num_tests_test
    print(f'Epoch {name} has accuracy {accuracy_test} against the test data.')
    return accuracy_train, accuracy_test

if __name__ == "__main__":
    main(Settings.load(sys.argv[1]))
    
    

  
