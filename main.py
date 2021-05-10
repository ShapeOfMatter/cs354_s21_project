#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import os
import os.path as path
import numpy as np


from train_gcn.dgldataset import get_train_test_dataloaders, get_dataloaders
from train_gcn.model import GCN, make_tagcn, RelGraphConvN, RELU, RelationalTAGConv, TAGConvN
from dgl.nn.pytorch import TAGConv
from dgl.nn.pytorch.utils import Sequential
from dgl.nn.pytorch.glob import AvgPooling
from train_gcn.state_classes import Settings
from train_gcn.train import epoch, make_criterion, make_optimizer, validate
from plots import create_plots


def check_file_blocks(*files: str):
    '''Returns True iff none of the files exist yet.'''
    return not any(path.exists(f) for f in files)


def main(settings: Settings):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for crashing.
    print("settings.max_batch_size: ", settings.max_batch_size)
    train_dataloader, test_dataloader, val_dataloader = get_dataloaders(settings)

    output_width = train_dataloader.dataset.num_labels()  # accessing dataloader.dataset isn't documented as a safe thing to do...
    assert output_width == test_dataloader.dataset.num_labels() == val_dataloader.dataset.num_labels()
    print(f'There are {output_width} labels in the data.')
    node_attributes = train_dataloader.dataset.node_attrs
    assert node_attributes == test_dataloader.dataset.node_attrs == val_dataloader.dataset.node_attrs
    printf('Node attributes: ', node_attributes)
    edge_attributes = train_dataloader.dataset.edge_attrs
    assert node_attributes == test_dataloader.dataset.edge_attrs == val_dataloader.dataset.edge_attrs
    printf('Edge attributes: ', edge_attributes)
    
    stats = {}
    
    # Most basic GCN
    print('Starting GCN \n')
    model = GCN(len(node_attributes), 16, output_width)
    stats['GCN'] =  train_model(settings, model, 'GCN')

    # RelGraphConv
    print('Starting RelGraphConv')
    # We can't parameterize num_rels because it's not a simple function of edge_attrubutes.
    model = RelGraphConvN(len(node_attributes), 16, output_width, num_rels=3)
    stats['RelGraphConv'] = train_model(settings, model, 'RelGraphConv')

    # TAGConv
    print('Starting TAGConv \n')
    model = TAGConvN(len(node_attributes), 16, output_width)
    stats['TAGConv'] = train_model(settings, model, 'TAGConv')

    # R-TAG-Conv
    print('Starting Relational TAG-Conv \n')
    model = Sequential(RelationalTAGConv(radius=2, width_in=len(node_attributes), forward_edge=8, backward_edge=8),
                       TAGConv(16, output_width, k=0),
                       AvgPooling())
    stats['RTAG_Conv'] = train_model(settings, model, 'RTAG_GCN')
    
    create_plots(stats)
    

def train_model(settings, model, label):
    criterion = make_criterion()
    optimizer = make_optimizer(settings.training_profile, model)

    if not check_file_blocks():  # ATM we don't actually have any use for this.
        print("failed file checks")
        return

    best_accuracy = 0.0  # Only save to disk when we make an improvement.
    best_loss = np.inf
    
    losses = []
    accuracy = []
    for epoch_number in range(settings.epochs):
        new_loss, new_accuracy = epoch(f'e_{epoch_number}',
                             training_data=train_dataloader,
                             testing_data=test_dataloader,
                             model=model,
                             optimizer=optimizer,
                             criterion=criterion,
                             label=label)
        # TODO: log stats
        #if new_accuracy >= best_accuracy:
        if new_loss < best_loss:
            print('New lowest loss. Saving weights!')
            torch.save(model.state_dict(), settings.model_filename + label)
            best_loss = new_loss
        losses.append(new_loss)
        accuracy.append(new_accuracy)
    val_accuary = validate(val_dataloader, model, label)
    
    return {'loss':losses, 'test accuracy': accuracy, 'validation accuracy':val_accuary}



if __name__ == "__main__":
    main(Settings.load(sys.argv[1]))




