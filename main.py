#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import os
import os.path as path

from train_gcn.dgldataset import get_train_test_dataloaders
from train_gcn.model import GCN, make_tagcn, RELU
from train_gcn.state_classes import Settings
from train_gcn.train import epoch, make_criterion, make_optimizer

def check_file_blocks(*files: str):
    '''Returns True iff none of the files exist yet.'''
    return not any(path.exists(f) for f in files)

def main(settings: Settings):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Fix for crashing.
    print("settings.max_batch_size: ", settings.max_batch_size)
    
    train_dataloader, test_dataloader = get_train_test_dataloaders(settings)
    input_width = 1  # TODO: HOW MANY NODE ATTRIBUTES ARE THERE?
    output_width = 13  # TODO: HOW MANY CLASSES ARE THERE?
    model = make_tagcn(input_width, 5, 5, output_width, radius=5, nonlinearity=RELU)
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
    
    

  
