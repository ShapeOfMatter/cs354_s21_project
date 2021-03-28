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

from dgldataset import SyntheticDataset
from state_classes import AdamTrainingProfile, ModelState, Settings, State, TrainingProfile

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
    
def make_dataloader(source_csvs: Sequence[str],  # filenames
                    use_indices: Sequence[int],  # typically used to split test/train data
                    sub_graph_choices: Sequence[int],  # typically used to reduce load for demo
                    max_batch_size: int  # will choose a batch size to evenly divide the data
                    ) -> GraphDataLoader:
    dataset = SyntheticDataset(source_csvs)
    dataset.partition(use_indices)
    dataset.select_samples(sub_graph_choices)
    dataset.process()
    batch_size = next(n for n in range(max_batch_size, 0, -1) if (len(dataset) % n) == 0)
    print(f'Using batch size {batch_size}.')
    sampler = SubsetRandomSampler(torch.arange(len(dataset)))  # I may have misunderstood?
    dataloader = GraphDataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=False)
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
    new = False
    if new:
        # Convert networks into DGL format. Original's are entire networks.
        convert_original('schoolnetJeffsNets')
        convert_original('med')
        # Pulls N samples. N = 10 
        graph_sample('schoolnetJeffsNets', 10, False)
        graph_sample('med', 10, False)
        
    # Assign test and train indicies. Note that there are 5000 files in each.
    test_train_splitter = np.random.default_rng(seed=settings.deterministic_random_seed)
    train_indices = test_train_splitter.choice(np.arange(5000),size = 4000)
    test_indices = np.array(list(set(range(5000)) - set(train_indices)))
    train_dataloader = make_dataloader(source_csvs=settings.source_csvs,
                                       use_indices=train_indices,
                                       sub_graph_choices=settings.sub_graph_choices,
                                       max_batch_size=settings.max_batch_size)
    test_dataloader = make_dataloader(source_csvs=settings.source_csvs,
                                      use_indices=test_indices,
                                      sub_graph_choices=range(10),
                                      max_batch_size=settings.max_batch_size)
    
    model = GCN(1, 4, 2) #dim of node data, conv filter size, number of classes.
    optimizer = make_optimizer(settings.training_profile, model)
    
    for epoch_number in range(50):
        state = State.load(settings.state_file)
        if state.models:
            # Depending if we're doing stuff in parallele, maybe we shouldn't reload every time?
            use_weights = state.models[0]  # for now there will be only one.
            model.load_state_dict(torch.load(use_weights.saved_in), strict=True)
            print(f'  Loaded model with accuracy {use_weights.accuracy} from {use_weights.saved_in}')
        new_accuracy = epoch(f'e_{epoch_number}',
                             training_data=train_dataloader,
                             testing_data=test_dataloader,
                             model=model,
                             optimizer=optimizer,
                             criterion=F.cross_entropy)
        if new_accuracy >= min((m.accuracy for m in state.models), default=0):
            old_weights = (
                state.models.pop(0)  # or some other index if we're using more than one.
                 if state.models
                 else ModelState(accuracy=0, saved_in=settings.model_name_format.format(len(state.models)))
            )
            print(f'  Overwriting weights (accuracy={old_weights.accuracy}).')
            torch.save(model.state_dict(), old_weights.saved_in)
            state.models.append(ModelState(accuracy=new_accuracy, saved_in=old_weights.saved_in))
            state.save(settings.state_file)
        else:
            print(f'  New weights (accuracy={new_accuracy}) will be discarded.')


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

    num_correct = sum(test(batched_graph, labels)
                      for batched_graph, labels in testing_data)
    num_tests = sum(len(labels)  # There may be an even more direct way to get this.
                    for batched_graph, labels in testing_data)
    accuracy = num_correct / num_tests
    print(f'Epoch {name} has accuracy {accuracy} against the test data.')
    return accuracy

if __name__ == "__main__":
    main(Settings.load(sys.argv[1]))
    
    

  
