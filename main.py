#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import os
import os.path as path

from train_gcn.dgldataset import get_train_test_dataloaders
from train_gcn.model import GCN, make_tagcn, RelGraphConvN, RELU, RelationalTAGConv, TAGConvN
from dgl.nn.pytorch import TAGConv
from dgl.nn.pytorch.utils import Sequential
from dgl.nn.pytorch.glob import AvgPooling
from train_gcn.state_classes import Settings
from train_gcn.train import epoch, make_criterion, make_optimizer


def check_file_blocks(*files: str):
    '''Returns True iff none of the files exist yet.'''
    return not any(path.exists(f) for f in files)


def main(settings: Settings):
    node_attributes = ('true_degree', 'distance_to_seed')
    output_width = 2  # TODO: HOW MANY CLASSES ARE THERE?
    edge_attributes = ('f',  # forward
                       'b',  # backward
                       'r')  # recruitment (a sub-set of forward)

    # Most basic GCN
    print('Starting basic GCN \n')
    model = GCN(len(node_attributes), 16, output_width)
    train_model(settings, model, 'GCN')

    # RelGraphConv. Relations is boolean for bidirected.
    print('Starting RelGraphConv')
    model = RelGraphConvN(len(node_attributes), 16, output_width, len(edge_attributes))
    train_model(settings, model, 'RelGraphConv')

    # TAGConv
    print('Starting TAGConv \n')
    model = TAGConvN(len(node_attributes), 16, output_width)
    train_model(settings, model, 'TAGConv')

    # TAGGCM
    print('Starting TAG GCN \n')
    # model = make_tagcn(len(node_attributes), 5, 5, output_width, radius=5, nonlinearity=RELU)

    # This runs, but not sure I understand the arguments for RelationalTAGConv.
    model = Sequential(RelationalTAGConv(radius=2, width_in=2, attr=8), TAGConv(8, 4, k=0), AvgPooling())
    train_model(settings, model, 'TAG_GCN')


def train_model(settings, model, label):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for crashing.
    print("settings.max_batch_size: ", settings.max_batch_size)
    train_dataloader, test_dataloader = get_train_test_dataloaders(settings)

    criterion = make_criterion()
    optimizer = make_optimizer(settings.training_profile, model)

    if not check_file_blocks():  # ATM we don't actually have any use for this.
        print("failed file checks")
        return

    best_accuracy = 0.0  # Only save to disk when we make an improvement.
    for epoch_number in range(settings.epochs):
        new_accuracy = epoch(f'e_{epoch_number}',
                             training_data=train_dataloader,
                             testing_data=test_dataloader,
                             model=model,
                             optimizer=optimizer,
                             criterion=criterion,
                             label=label)
        # TODO: log stats
        if new_accuracy >= best_accuracy:
            print(f'  Saving weights!')
            torch.save(model.state_dict(), settings.model_filename + label)

if __name__ == "__main__":
    main(Settings.load(sys.argv[1]))




