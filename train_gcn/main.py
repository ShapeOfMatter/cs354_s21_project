#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import os
import os.path as path

from dgldataset import get_train_test_dataloaders
from model import GCN
from state_classes import Settings
from train import epoch, make_criterion, make_optimizer

def check_file_blocks(*files: str):
    '''Returns True iff none of the files exist yet.'''
    return not any(path.exists(f) for f in files)

def main(settings: Settings):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Fix for crashing.
    print("settings.max_batch_size: ", settings.max_batch_size)
    
    train_dataloader, test_dataloader = get_train_test_dataloaders(settings)
    model = GCN(2, 4, 2) #dim of node data, conv filter size, number of classes.
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
                             criterion=criterion)
        # TODO: log stats
        if new_accuracy >= best_accuracy:
            print(f'  Saving weights!')
            torch.save(model.state_dict(), settings.model_filename)

if __name__ == "__main__":
    main(Settings.load(sys.argv[1]))
    
    

  
